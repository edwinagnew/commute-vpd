"""teacher_training E43 (2026-10-28): teacher training, then the rest of the fine-tune into Delta.
Plan: reports/plans/2026-10-28_plan_e43-teacher-remainder.md ("Scope as approved").

The update dW_k is E41's emoticon Adam 3e-5 arm-A fine-tune (all 24 matrices, batch 32, seed 0) stopped at step k in
(10, 1000). Teacher training is E41's one-shot task-text copy (path.student: Adam 1e-5, 1,000 steps, batch-order seed 0,
Gamma and Delta frozen), giving W_tt. "Teacher training plus remainder" keeps the copy's (U', V') and sets the target to
W0 + dW_k, so Delta' = Delta + R with the remainder R = W_ft - W_tt. Four states per step, on the same rows:
  do-nothing          dW_k in Delta (the positive control)
  least squares       split.split_state(dW_k)
  teacher training    the copy's own state (Delta untouched)
  plus remainder      the copy's (U', V'), target W0 + dW_k
Each is read through E41's Pile harness (`read`: harness.measure + arms.heldout_of + run.record; the route split's
e_delta is Delta' taken to zero, reported). Each but teacher training's own is also read with Delta' put back to Delta
(`removal`: the target W0 + (U'V'^T - UV^T)): KL(fine-tune || that model) on the held-out task windows (arm A's
weighting) and on the harness's 256 Pile val rows, each over the same KL of W0 (do-nothing's removal state, the same
path). Teacher training's own removal state is itself: 0 by construction, not read.
Stages, in order, each record dumped and logged as it finishes:
  C1   W0 under the Pile harness (job load): w0_check, floor 0.0748 and seed-0 PGD KL 0.69882 within tolerance; stop
  fine-tune; G1: T10 and T1000 P(target) and size as E41's; stop
  per step (10, then 1000): do-nothing (PC: its ratio as E41's, its task-text denominator as E41's gap at W0; stop),
       the copy (G3: as E41's one-shot copy; Delta share <= 1e-6; stop), plus remainder (G2: its removal state is the
       copy's own target; stop), least squares
  score  P2, P3 (scored) and the P1 check (`score`, pure) -> out["score"]
A step not marked done is redone from its start on a rerun (the copy lives in memory only).

Usage (repo root, project venv):  remainder.py --predictions   files results/predictions_e43.json (refuses to overwrite)
                                  remainder.py [cuda]          the run (GPU); refuses to start without the predictions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from path import student
from run import CFG, EMO, FT_ADAM_LR, RESULTS, Job, dump, halt, load, log, record

PRED, OUT = RESULTS / "predictions_e43.json", RESULTS / "teacher_training_e43.json"
ORDER = {"pass": 0, "inconclusive": 1, "fail": 2}  # the check's verdicts, best first


@dataclass(frozen=True)
class E43:
    steps: tuple[int, ...] = (10, 1000)  # fine-tune checkpoints read, in run order
    ft_lr: float = FT_ADAM_LR
    c1_floor: float = 0.0748  # E41's H100: W0's PGD floor 0.07482 ...
    c1_floor_tol: float = 0.10
    c1_kl: float = 0.69882  # ... and seed-0 PGD KL (E41, E41 Amendment 2)
    c1_kl_tol: float = 0.01
    # G1: E41's teachers (P(target), size), tolerance absolute
    g1: dict[str, tuple[float, float]] = field(default_factory=lambda: {"10": (0.99931, 0.01125), "1000": (0.99961, 0.03209)})
    g1_tol: tuple[float, float] = (0.005, 0.001)
    # PC: E41's do-nothing ratio and gap at W0 per step; relative tolerance (E41 A1, A2 reproduced both to 16 digits)
    pc: dict[str, tuple[float, float]] = field(default_factory=lambda: {"10": (2.51590, 5.07835), "1000": (1.49940, 5.08975)})
    pc_tol: float = 0.01
    # G3: E41's one-shot task-text copies (P(target) absolute, ratio relative)
    g3: dict[str, tuple[float, float]] = field(default_factory=lambda: {"10": (0.99941, 1.54653), "1000": (0.99675, 0.95909)})
    g3_tol: tuple[float, float] = (0.005, 0.05)
    g2_tol: float = 1e-3  # G2: |removal change - copy's delta-W| over the copy's largest entry (fp32 rounding ~1e-5)
    delta_max: float = 1e-6  # the copy's Delta share (by construction ~1e-9)
    denom_min: float = 0.005  # a removal share is unscored if W0's KL is below this (10 x E41's hook floor 0.0005)
    works: float = 0.05  # P1 check (unscored): task-text share <= this (E41's "works")
    inert: float = 0.2  # P2: plain-text share <= this "inert", >= carries "carries", else "between"
    carries: float = 0.5
    p3_ratio: float = 1.25  # P3: plus remainder's ratio <= this x teacher training's own (E41's tolerance)

    def __post_init__(self) -> None:
        keys = tuple(map(str, self.steps))
        assert self.steps == (10, 1000) and CFG.steps[-1] == 1000, "E41's T10 and T1000; the copy runs 1,000 steps"
        assert set(self.g1) == set(self.pc) == set(self.g3) == set(keys)
        assert 0 < self.inert < self.carries < 1 and self.p3_ratio > 1


AM = E43()


def read(job: Job, label: str, dW: dict[str, Any], st: dict[str, Any] | None) -> dict[str, Any]:
    """One state through the harness: run.record's numbers (no teacher gap), the route split's e_delta (Delta' to zero)
    and the check's verdict with no P bar ('faith')."""
    from commute import arms, faith_score, harness

    tctx = job.task(EMO)[0]
    m = harness.measure(tctx, label, dW, st, True, log)
    ho = arms.heldout_of(tctx.model, tctx.W0, tctx.held, tctx.token, tctx.device, dW)
    rec = record(m, ho, float("nan"), job.w0, CFG.p_min)
    rec.pop("teacher_gap")
    rec.update(e_delta=m["routes"]["shares"]["e_delta"], faith=faith_score.state_verdict(m["pile"], 1.0, job.w0)["verdict"])
    log(f"[{label}] " + " ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    return rec


def change_of(job: Job, st: dict[str, Any] | None) -> dict[str, Any]:
    """U'V'^T - UV^T per matrix (0 for do-nothing): the target change once Delta' is put back to Delta."""
    tctx = job.task(EMO)[0]
    out = {k: torch.zeros_like(w).cpu() for k, w in tctx.W0.items()}
    for k, (_, U2, V2) in (st or {}).items():
        U, V = tctx.uv_cpu[k]
        out[k] = (U2.float() @ V2.float().T - U.float() @ V.float().T).cpu()
    return out


def removal(job: Job, rows: Any, net: Any, change: dict[str, Any]) -> dict[str, float]:
    """KL(fine-tune || W0 + change) on the task windows (teacher.heldout_gap) and on the Pile rows (eval_arm_b)."""
    from commute import teacher
    from commute.faithfulness import model_state
    from commute.finetune import eval_arm_b

    tctx = job.task(EMO)[0]
    with model_state(tctx.model, target={k: tctx.W0[k] + change[k].to(tctx.W0[k].device) for k in tctx.W0}):
        s = tctx.model.target_model
        return {"kl_task": teacher.heldout_gap(s, net, tctx.held, tctx.device),
                "kl_pile": eval_arm_b(s, net, rows, tctx.device)["kl_to_ref"]}


def stop_unless(out: dict[str, Any], name: str, ok: bool, rec: dict[str, Any]) -> None:
    out.setdefault("gates", {})[name] = {**rec, "pass": ok}
    log(f"[{name}] {json.dumps(out['gates'][name], default=float)}")
    dump(OUT, out)
    if not ok:
        halt(OUT, out, f"{name}: {json.dumps(rec, default=float)}")


def one_step(job: Job, out: dict[str, Any], rows: Any, k: str, dW: dict[str, Any]) -> None:
    """The four states of one fine-tune step, controls first."""
    from commute import finetune, split, teacher

    tctx, s = job.task(EMO)[0], out["steps"][k]
    net = teacher.teacher_net(tctx.model, dW)
    s["nothing"] = {**read(job, f"T{k} do-nothing", dW, None), **removal(job, rows, net, change_of(job, None))}
    d_task, d_pile = s["nothing"]["kl_task"], s["nothing"]["kl_pile"]
    s["denominators"] = {"task": d_task, "pile": d_pile}
    r, g = AM.pc[k]
    stop_unless(out, f"PC T{k}", abs(s["nothing"]["pgd"] / r - 1) <= AM.pc_tol and abs(d_task / g - 1) <= AM.pc_tol,
                {"pgd": s["nothing"]["pgd"], "kl_task_w0": d_task, "kl_pile_w0": d_pile, "e41": [r, g]})
    t0 = time.time()
    curve, st = student(job, EMO, "A", net, None)
    dW_tt = {key: v[0] for key, v in st.items()}
    s["teacher_training"] = {**read(job, f"T{k} teacher training", dW_tt, st), "start_kl": curve[0]["loss"],
                             "seconds": time.time() - t0}
    p, ratio = AM.g3[k]
    tt = s["teacher_training"]
    stop_unless(out, f"G3 T{k}", abs(tt["p_target"] - p) <= AM.g3_tol[0] and abs(tt["pgd"] / ratio - 1) <= AM.g3_tol[1]
                and tt["delta_share"] <= AM.delta_max, {"p_target": tt["p_target"], "pgd": tt["pgd"],
                                                        "delta_share": tt["delta_share"], "e41": [p, ratio]})
    ch = change_of(job, st)
    big = max(float(d.abs().max()) for d in dW_tt.values())
    g2 = max(float((ch[key] - dW_tt[key]).abs().max()) for key in dW_tt) / big
    stop_unless(out, f"G2 T{k}", g2 <= AM.g2_tol, {"removal_vs_copy": g2})
    rem = {key: dW[key].cpu() - ch[key] for key in dW}
    s["plus_remainder"] = {**read(job, f"T{k} plus remainder", dW, st), **removal(job, rows, net, ch),
                           "cos_R_copy": finetune.cosines(rem, dW_tt)["all"]}
    st_ls = split.split_state(dW, tctx.uv_cpu, tctx.bases)
    s["least_squares"] = {**read(job, f"T{k} least squares", dW, st_ls), **removal(job, rows, net, change_of(job, st_ls))}
    for name in ("nothing", "plus_remainder", "least_squares"):
        x = s[name]
        x.update(b_task=x["kl_task"] / d_task, b_pile=x["kl_pile"] / d_pile)
        log(f"[T{k} {name}] removal share task {x['b_task']:.5g} pile {x['b_pile']:.5g}")
    s["done"] = True
    out["score"] = score(out)
    dump(OUT, out)
    log(f"[score after T{k}] {json.dumps(out['score'], default=float)}")


# ---- scoring (pure, on the results file) ----------------------------------------------------------------------------


def pile_category(s: dict[str, Any]) -> str | None:
    """P2's three-way reading of one step; None when a record is missing or W0's plain-text KL is below AM.denom_min."""
    rec, d = s.get("plus_remainder"), s.get("denominators", {}).get("pile")
    if rec is None or d is None or d < AM.denom_min:
        return None
    b = rec["kl_pile"] / d
    return "inert" if b <= AM.inert else "carries" if b >= AM.carries else "between"


def excess_removed(r: float, r_dn: float, pass_line: float) -> float | None:
    """(r_dn - r) / (r_dn - 1), only where do-nothing reads above the pass line."""
    return (r_dn - r) / (r_dn - 1) if r_dn > pass_line else None


def paired(s: dict[str, Any], pass_line: float) -> dict[str, Any] | None:
    """P3 for one step: plus remainder's ratio <= AM.p3_ratio x teacher training's own and its verdict no worse."""
    a, t, n = s.get("plus_remainder"), s.get("teacher_training"), s.get("nothing")
    if a is None or t is None or n is None:
        return None
    return {"holds": a["pgd"] <= AM.p3_ratio * t["pgd"] and ORDER[a["faith"]] <= ORDER[t["faith"]],
            "ratio_over_own": a["pgd"] / t["pgd"], "verdicts": [a["faith"], t["faith"]],
            "excess_removed": excess_removed(a["pgd"], n["pgd"], pass_line),
            "excess_removed_own": excess_removed(t["pgd"], n["pgd"], pass_line)}


def task_check(s: dict[str, Any]) -> bool | None:
    rec, d, t = s.get("plus_remainder"), s.get("denominators", {}).get("task"), s.get("teacher_training")
    if rec is None or t is None or not d or d < AM.denom_min:
        return None
    return rec["kl_task"] / d <= AM.works and t["p_target"] >= CFG.p_min


def overall(vals: list[Any], rank: dict[str, int]) -> str | None:
    return None if not vals or None in vals else max(vals, key=lambda v: rank[v])


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P2 and P3 (scored) and the P1 check (unscored), on whatever the file holds; a missing step makes a prediction
    unscored (None), never a pass."""
    from commute import faith_score

    pass_line = faith_score.LINE / (1 + out["W0"]["pgd_floor"])
    steps = {k: out.get("steps", {}).get(k, {}) for k in map(str, AM.steps)}
    p2 = {k: pile_category(s) for k, s in steps.items()}
    p3 = {k: paired(s, pass_line) for k, s in steps.items()}
    p3_vals = [None if v is None else v["holds"] for v in p3.values()]
    return {"P1_task_text_check": {k: task_check(s) for k, s in steps.items()},
            "P2_plain_text": {"outcome": overall(list(p2.values()), {"inert": 0, "between": 1, "carries": 2}), "per": p2},
            "P3_check_paired": {"holds": None if None in p3_vals else all(p3_vals), "per": p3},
            "pass_line": pass_line}


PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training E43 (reports/plans/2026-10-28_plan_e43-teacher-remainder.md, 'Scope as approved'). Emoticon, "
             "E41's Adam 3e-5 arm-A fine-tune (seed 0) at steps 10 and 1000, regenerated in-job; E41's one-shot task-text copy; "
             "'plus remainder' = the copy's (U', V') with target W0 + dW (Delta' = Delta + W_ft - W_tt). States on the same "
             "rows: do-nothing, least squares, teacher training's own, plus remainder. RunPod H100 80GB HBM3. Filed before any run",
    "controls": "C1 first, alone: w0_check, PGD floor 0.0748 within 10%, seed-0 PGD KL 0.69882 within 1%. G1: teachers P and "
                "size within 0.005 / 0.001 of E41's. PC per step: do-nothing ratio 2.5159 (T10) / 1.4994 (T1000) and its "
                "task-text KL at W0 5.0784 / 5.0897 within 1% (E41); its removal shares are 1 by construction (the "
                "denominators are its own). G3: the copy's P within 0.005 and ratio within 5% of E41's (0.99941, 1.5465; "
                "0.99675, 0.9591), Delta share <= 1e-6. G2: the copy's (U', V') minus W0's reproduces its delta-W to 1e-3. "
                "Any miss: stop",
    "measurements": {
        "removal share": "KL(fine-tune || state with Delta' put back to Delta) / KL(fine-tune || W0), on the held-out task "
                         "windows with arm A's weighting (b_task) and on the harness's 256 Pile val rows (b_pile). Do-nothing 1, "
                         "teacher training's own 0, both by construction. Blind to: text off those rows; the trigger alone",
        "check": "Pile harness PGD KL over W0's (one start, seed 0, 20 steps, 256 rows); verdict against this job's floor, "
                 "pass <= 1.5/(1+f) (1.396 at f 0.0748), fail >= 1.5(1+f) (1.612). A pass is a lower bound",
        "reported": "Delta' share ||Delta' - Delta||^2 / ||dW||^2 (harness delta_change_frac); e_delta (Delta' to zero, "
                    "log-odds at the trigger, saturates near P 1); cos(R, the copy's delta-W); start KL",
    },
    "expected": {
        "W0": "check 1; removal shares undefined (0/0)",
        "do-nothing": "check 2.516 fail / 1.499 inconclusive; b 1 and 1; Delta' share 1; e_delta 1",
        "least squares": "check 6.96 / 2.11 (E22); Delta' share 0.36 / 0.38; e_delta 0.09 / 0.28; b measured, between 0 and 1",
        "teacher training": "check 1.547 / 0.959; P 0.9994 / 0.9968; Delta share ~1e-9; b 0",
        "plus remainder": "b_task = the copy's gap share, 0.0013 / 0.0056 (E41); b_pile unknown (r1's task-text copies 0.30-0.33, "
                          "LoRA 1.06, E34 Amendment 6); Delta' share 0.9-1.1; cos(R, copy) about -0.3",
        "noise": "W0's adversary-seed floor 0.0748; batch-order spread of the ratio 21% at T10, 3.2% at T1000 (E41 A1); b_pile "
                 "batch-order spread 0.10 (E34 A6, Pile-text copies); gap share 0.002 (E34 A4)",
    },
    "predictions": [
        {"id": "P1_task_text_check", "scored": False,
         "rule": "per step: b_task(plus remainder) <= 0.05 and teacher training's own P(target) >= 0.9; None if a record is "
                 "missing or the denominator < 0.005 nats. Unscored: its value is known from E41 (0.0013, 0.0056)"},
        {"id": "P2_plain_text", "lean": "between, medium (r1's task-text copies read 0.30-0.33)",
         "rule": "per step: b_pile(plus remainder) <= 0.2 'inert', >= 0.5 'carries', else 'between' (both ends inclusive); "
                 "None if a record is missing or KL(fine-tune || W0) on the Pile rows < 0.005 nats. Outcome: the worst step "
                 "(inert < between < carries); unscored if any step is None. Least squares' b_pile reported beside it"},
        {"id": "P3_check_paired", "lean": "holds, high at T1000 (the null's gap ~15%); medium at T10",
         "rule": "per step: plus remainder's ratio <= 1.25 x teacher training's own ratio (relative, inclusive) and its "
                 "verdict no worse (pass < inconclusive < fail, the check alone; both failing counts as no worse). Holds if "
                 "both steps hold; unscored if a record is missing. Reported per step: the share of do-nothing's excess "
                 "removed, (r_dn - r)/(r_dn - 1), for plus remainder and for teacher training's own, None where do-nothing "
                 "reads at or below the pass line"},
    ],
    "not_run": [
        "the broad lowercase fine-tune (dropped: teacher training's own state fails there)",
        "T100; other tasks; SGD or LoRA teachers; Pile-text copies",
        "batch-order or seed repeats; adversary seeds 1-2; the budget adversary",
        "the least-squares split of R into the subcomponents; input-weighted shares; Qwen",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e43": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def e43(device: str) -> None:
    from commute.finetune import ArmAConfig, finetune_arm_a

    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (remainder.py --predictions)")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["e43"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another E43 config: {filed['e43']}"
    job = Job(device)
    out = load(OUT, job, {"steps": {}})
    floor, kl = job.w0["pgd_floor"], job.ctx.pile[0]["pgd_kl"]
    stop_unless(out, "C1", job.w0["pass"] and abs(floor / AM.c1_floor - 1) <= AM.c1_floor_tol
                and abs(kl / AM.c1_kl - 1) <= AM.c1_kl_tol, {"w0_check": job.w0["pass"], "pgd_floor": floor, "w0_pgd_kl": kl})
    tctx = job.task(EMO)[0]
    cfg = ArmAConfig(o_token=tctx.token, lr=AM.ft_lr, steps=AM.steps[-1], checkpoints=AM.steps, seed=0, optimizer="adam")
    res = finetune_arm_a(tctx.model, tctx.train, tctx.held, cfg, tctx.device, log)
    rows = torch.cat([b for block in job.ctx.blocks for b in block])  # the harness's 256 Pile val rows
    for k in map(str, AM.steps):
        ev = res.evals[int(k)]
        (p, size), (tp, ts) = AM.g1[k], AM.g1_tol
        stop_unless(out, f"G1 T{k}", abs(ev["p_o"] - p) <= tp and abs(ev["rel_norm_max"] - size) <= ts,
                    {"p_target": ev["p_o"], "size": ev["rel_norm_max"], "damage": ev["kl_other"], "e41": [p, size]})
    for k in map(str, AM.steps):
        if not out["steps"].get(k, {}).get("done"):
            out["steps"][k] = {}
            one_step(job, out, rows, k, res.deltas[int(k)])
    out["score"], out["done"] = score(out), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'], default=float)}")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else e43(args.device)
