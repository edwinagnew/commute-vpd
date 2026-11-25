"""E126: is E119's emoticon Adam Pile-text copy (P(target) 0.744 on the L40 against E34's 0.902 on the H100) a card
difference or a batch-order outlier? Plan reports/plans/2026-11-06_plan_e126-copy-reproducibility.md; config, rules and
expected values in results/predictions.json (the run refuses unless it is committed unchanged).

E119's code path (experiments/teacher_route_split/run.py, which stays as it ran), Pile-text arm (B) only: per cell the
teacher is regenerated (E31's arm-A fine-tune, seed 0) and checked against E119's; then one student per batch order
(`teacher.train_student(seed=order)`, the only change from E119), read on its own state and labelled under E119's lines.
The control runs first: the emoticon Adam student at order 0 must reproduce E119's 0.744 within control_tol, or stop.
A student that might not finish inside budget_s exits 3 (`fleet/resubmit --code 3`); a rerun skips recorded students.

    run.py      # cluster/copy_reproducibility.sbatch; writes results/copy_reproducibility.json, results/large/<task>/<opt>_B_o<k>.pt
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from commute import arms, component_training, saved_states, teacher, vpd
from commute.data import pile_rows
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig, finetune_arm_a
from commute.jobs import hard_exit, require_filed
from commute.local_faith import route_cells, route_split
from commute.regating import load_windows
from commute.route_labels import Lines, label

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
PRED, LARGE, OUT = RESULTS / "predictions.json", RESULTS / "large", RESULTS / "copy_reproducibility.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E119 = EXP / "teacher_route_split" / "results"


@dataclass(frozen=True)
class Config:
    # (task, teacher optimizer, batch orders), run in this order: the control cell first (order 0 first)
    cells: tuple[tuple[str, str, tuple[int, ...]], ...] = (
        ("emoticon", "adam", (0, 1, 2)), ("r0_t352_v310", "adam", (1, 2)),
        ("r0_t352_v310", "sgd", (1, 2)), ("r1_t534_v369", "sgd", (1, 2)))
    # E119's settings (experiments/teacher_route_split/results/predictions.json `config`; run.py:155, 179-181, 205-208)
    sgd_lr: tuple[tuple[str, float], ...] = (("emoticon", 0.01), ("r0_t352_v310", 0.03), ("r1_t534_v369", 0.03))
    adam_lr: float = 3e-5
    steps: int = 1000
    student_lr: float = 1e-5
    batch_b: int = 8
    pile_start: int = 32768
    pile_train_rows: int = 8192
    # checks and rules
    control_tol: float = 0.005  # order-0 emoticon student P(target) against E119's 0.7442
    teacher_tol: float = 0.005  # teacher P(target), and do-nothing gain relative, against E119's on the same card
    near_h100: float = 0.852  # E34's order-0 0.902 minus E34's p_tol 0.05
    near_l40: float = 0.794  # E119's order-0 0.744 plus the same 0.05
    budget_s: float = 1560.0  # E121's: 26 min of the 30-min debug limit
    teacher_s: float = 45.0  # E119 log: fine-tune ~30 s, do-nothing read ~1 s
    student_s: float = 175.0  # E119 log: arm-B students 154-167 s, read included

    def __post_init__(self) -> None:
        assert self.cells[0][:2] == ("emoticon", "adam") and self.cells[0][2][0] == 0 and self.near_l40 < self.near_h100


CFG, LINES = Config(), Lines()


def p1(p: dict[str, float]) -> dict[str, Any]:
    """The emoticon cell from P(target) at orders 1 and 2 (order 0 is the control): both >= near_h100 'order-0 outlier';
    both <= near_l40 'card difference'; otherwise 'unstable'; a non-finite P 'unreadable'. Spread over all three orders."""
    a, b = p["1"], p["2"]
    if not (math.isfinite(a) and math.isfinite(b)):
        v = "unreadable"
    elif min(a, b) >= CFG.near_h100:
        v = "order-0 outlier"
    elif max(a, b) <= CFG.near_l40:
        v = "card difference"
    else:
        v = "unstable"
    ok = all(math.isfinite(x) for x in p.values())
    return {"verdict": v, "p": p, "spread": max(p.values()) - min(p.values()) if ok else math.nan}


def p2(labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Per re-switching corner cell: 'holds' if orders 1 and 2 both read re-switching, else 'breaks'; and the count."""
    per = {c: "holds" if all(x == "re-switching" for x in ls.values()) else "breaks" for c, ls in labels.items()}
    return {"per": per, "labels": labels, "n_hold": sum(v == "holds" for v in per.values()), "n": len(per)}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    tmp.replace(OUT)


def check(out: dict[str, Any], name: str, value: float, ok: bool) -> None:
    log(f"[check {name}] {value:.8g}: {ok}")
    out["checks"][name] = {"value": value, "pass": ok}
    if not ok:
        out["stopped"] = f"{name}: {value:.8g}"
    dump(out)
    if not ok:
        hard_exit(1)


def budget(out: dict[str, Any], t0: float, need: float, what: str) -> None:
    if time.time() - t0 + need > CFG.budget_s:
        log(f"[budget] {what} needs ~{need:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
        dump(out)
        hard_exit(3)


def read(c: Any, dW: dict[str, Any], comps: Any) -> dict[str, Any]:
    """E119's `Job.read` (run.py:158-163): the route split at target W0 + dW with components `comps` (None: released)."""
    with model_state(c.model, target={k: c.W0[k] + dW[k].to(c.W0[k].device) for k in c.W0}, components=comps):
        return route_split(c.model, c.held.tokens, c.held.half, c.token, c.base, c.comps0)


def run_cell(out: dict[str, Any], c: Any, task: str, opt: str, orders: tuple[int, ...], t0: float) -> None:
    rec, ref = out["cells"].setdefault(f"{task}/{opt}", {"orders": {}}), json.loads((E119 / f"teacher_route_split_{task}.json").read_text())
    todo = [o for o in orders if not (str(o) in rec["orders"] and (LARGE / task / f"{opt}_B_o{o}.pt").is_file())]
    if not todo:
        return
    budget(out, t0, CFG.teacher_s + CFG.student_s, f"{task}/{opt} teacher and a student")
    train, c.held, c.token = load_windows(task, TRANSPORT, LADDER)
    c.base = route_cells(c.model, c.held.tokens, c.held.half, c.token)  # E119 run.py:236
    lr = dict(CFG.sgd_lr)[task] if opt == "sgd" else CFG.adam_lr
    cfg = ArmAConfig(o_token=c.token, lr=lr, steps=CFG.steps, checkpoints=(CFG.steps,), seed=0, optimizer=opt)
    res = finetune_arm_a(c.model, train, c.held, cfg, c.device, log)  # E119 run.py:179-181
    dW_t, e119_t = res.deltas[CFG.steps], ref["teachers"][opt]
    nothing = read(c, dW_t, None)
    rec["teacher"] = {"p_target": res.evals[CFG.steps]["p_o"], "gain": nothing["gain"], "r": nothing["shares"]["r"]}
    check(out, f"{task}/{opt} teacher P(target) against E119's", rec["teacher"]["p_target"] - e119_t["p_target"],
          abs(rec["teacher"]["p_target"] - e119_t["p_target"]) <= CFG.teacher_tol)
    rel = nothing["gain"] / e119_t["nothing"]["gain"] - 1
    check(out, f"{task}/{opt} teacher do-nothing gain against E119's (relative)", rel, abs(rel) <= CFG.teacher_tol)
    net = teacher.teacher_net(c.model, dW_t)
    for o in todo:
        budget(out, t0, CFG.student_s, f"{task}/{opt} order {o}")
        t1 = time.time()
        st, _ = teacher.train_student(c.model, c.uv, c.W0, net, c.rows, c.held, c.token, CFG.batch_b, None, CFG.student_lr,
                                      CFG.steps, c.device, log, seed=o)  # E119 run.py:205-208, seed = batch order
        arms.save_offsets(LARGE / task / f"{opt}_B_o{o}.pt", c.uv_cpu, st, {"task": task, "teacher": opt, "arm": "B", "order": o})
        dW = {k: v[0] for k, v in st.items()}
        comps = component_training.components_with(c.model, c.alive, {k: (v[1], v[2]) for k, v in st.items()})
        a = read(c, dW, comps)
        sh, g0 = a["shares"], rec["teacher"]["gain"]
        rec["orders"][str(o)] = r = {"p_target": arms.heldout_of(c.model, c.W0, c.held, c.token, c.device, dW)["p_o"],
                                     "r": sh["r"], "e_gate": sh["e_gate"], "gain_ratio": a["gain"] / g0, "a": a,
                                     "label": label(sh["r"], sh["e_gate"], a["gain"], g0, LINES), "seconds": time.time() - t1}
        log(f"[{task}/{opt} order {o}] P {r['p_target']:.4f} r {r['r']:.4f} e_gate {r['e_gate']:.4f} gain ratio "
            f"{r['gain_ratio']:.3f} -> {r['label']}; E119 order 0: P {ref['students'][f'{opt}_B']['p_target']:.4f}; {r['seconds']:.0f} s")
        if (task, opt, o) == ("emoticon", "adam", 0):  # check() writes the record with its verdict, so no rerun skips it
            d = r["p_target"] - ref["students"]["adam_B"]["p_target"]
            check(out, "control: emoticon adam order-0 P(target) against E119's 0.7442", d, abs(d) <= CFG.control_tol)
        dump(out)
    del net


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    if not LARGE.is_dir():
        sys.exit(f"{LARGE} missing: run ~/commute-agents/fleet/link-data <role> first")
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"cells": {}, "checks": {}, "jobs": []}
    if "stopped" in out:
        sys.exit(f"{OUT} has a stop ({out['stopped']}): move it aside to start again")
    t0, card = time.time(), torch.cuda.get_device_name()
    if card != "NVIDIA L40":
        sys.exit(f"E126 reads the L40 (E119's card); this is {card}")
    out["jobs"].append({"card": card, "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    model = vpd.load_component_model("cuda")
    alive = saved_states.load_alive(TRANSPORT / "large")  # E119 run.py:145-150
    uv = component_training.alive_uv(model, alive)
    c = SimpleNamespace(model=model, device="cuda", alive=alive, uv=uv, W0=vpd.target_weights(model),
                        comps0=vpd.component_weights(model), uv_cpu={k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()},
                        rows=pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log))
    for task, opt, orders in CFG.cells:
        (LARGE / task).mkdir(exist_ok=True)
        run_cell(out, c, task, opt, orders, t0)
    cells = out["cells"]
    out["score"] = {"P1": p1({str(o): cells["emoticon/adam"]["orders"][str(o)]["p_target"] for o in (0, 1, 2)}),
                    "P2": p2({k: {o: v["label"] for o, v in cells[k]["orders"].items()} for k in cells if k != "emoticon/adam"}),
                    "emoticon_labels": {o: v["label"] for o, v in cells["emoticon/adam"]["orders"].items()}}
    out["done"] = True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
