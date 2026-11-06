"""E119: the route split of E34's teacher-trained decompositions on the three trigger tasks, beside each fine-tune's own.
Plan reports/plans/2026-11-06_plan_e119-teacher-route-split-triggers.md; config, labels, leans and expected values in
results/predictions.json (the run refuses unless it is committed unchanged; its `config` is read below).

Per task, for the SGD then the Adam teacher (E34's order): regenerate the teacher (E31's arm-A fine-tune, seed 0, step
1000), stop unless its held-out P(target) is within p_tol of E34's; read do-nothing and the least-squares split (the
fine-tune's own reading, E12's reference) with the W0 control; then E34's students, arm A (task windows) and arm B (Pile
rows), each stopped unless its P(target) is within p_tol of E34's, and read on its own state: target W0 + the student's
change, components the student's U', V' (Delta fixed, E12 Amendment 1's state (a)). Each record is written as it is read.

    run.py [TASK ...]       # default all three; cluster/teacher_route_split.sbatch passes its arguments through
Writes results/teacher_route_split_<task>.json, results/large/<task>/<opt>_<arm>.pt (bf16 offsets, ~60 MB each) and,
once all three tasks are done, results/score.json. A stop: `.stopped` in the task's file, exit 1.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from commute import anchors as AN
from commute import arms, component_training, saved_states, teacher, vpd
from commute.data import pile_rows
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig, finetune_arm_a
from commute.jobs import hard_exit, require_filed
from commute.keydep import sha256
from commute.local_faith import MIN_GAIN, route_cells, route_split

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
PRED, LARGE, SCORE = RESULTS / "predictions.json", RESULTS / "large", RESULTS / "score.json"
ROOT, CODE = EXP.parent, ("src/commute", "experiments/teacher_route_split/run.py", "experiments/teacher_route_split/results/predictions.json")
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E34, E31 = EXP / "teacher_training" / "results", EXP / "adjoint_sgd" / "results" / "adjoint_sgd.json"
E12A1 = EXP / "new_vs_existing" / "results" / "new_vs_existing_amend1.json"
CM = SimpleNamespace(**json.loads(PRED.read_text())["config"])  # every size and line; a missing file stops the import
OPTS, ARMS = ("sgd", "adam"), ("A", "B")


# ----------------------------------------------------------------------------- rules (pure), as filed


def label(r: float, e_gate: float, gain: float, teacher_gain: float) -> str:
    """E12's filed lines: re-switching r >= 0.6 with e_gate <= 0.25; masked-off e_gate >= 0.5 with r <= 0.35; else
    neither. Unreadable: a share not finite, gain under MIN_GAIN, or gain / teacher's gain under min_ratio (E12 A1's pool)."""
    ratio = gain / teacher_gain if teacher_gain > 0 else math.nan
    if not (math.isfinite(r) and math.isfinite(e_gate) and gain >= MIN_GAIN and ratio >= CM.min_ratio):
        return "unreadable"
    if r >= CM.r_hi and e_gate <= CM.masked_lo:
        return "re-switching"
    if e_gate >= CM.masked_hi and r <= CM.r_lo:
        return "masked-off"
    return "neither"


def majority(labels: list[str]) -> str:
    """The label of strictly more than half of the readable states; else 'mixed'; under min_class readable, 'unscorable'."""
    ok = [x for x in labels if x != "unreadable"]
    if len(ok) < CM.min_class:
        return "unscorable"
    top = max(sorted(set(ok)), key=ok.count)
    return top if ok.count(top) * 2 > len(ok) else "mixed"


def score(students: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P1 the majority label of the students; P2 how many readable pairs keep their fine-tune's label; the control is
    E12 Amendment 1's students under the same rule (copy tables should read re-switching, maps masked-off)."""
    labels = {n: s["label"] for n, s in students.items()}
    pairs = [n for n, s in students.items() if "unreadable" not in (s["label"], s["ft_label"])]
    same = [n for n in pairs if students[n]["label"] == students[n]["ft_label"]]
    p2 = "unscorable" if not pairs else "same" if len(same) == len(pairs) else "differs"
    return {"P1": {"verdict": majority(list(labels.values())), "labels": labels,
                   "by_arm": {a: majority([x for n, x in labels.items() if n.endswith(f"_{a}")]) for a in ARMS},
                   "counts": {x: list(labels.values()).count(x) for x in sorted(set(labels.values()))}},
            "P2": {"verdict": p2, "same": len(same), "pairs": len(pairs), "differ": [n for n in pairs if n not in same],
                   "ft_labels": {n: s["ft_label"] for n, s in students.items()}},
            "control_e12_amend1": control}


def e12_control() -> dict[str, dict[str, Any]]:
    """E12 Amendment 1's batch-order-0 students (state a), labelled by `label`, with their class."""
    a1 = json.loads(E12A1.read_text())
    out = {}
    for n, s in a1["students"].items():
        if "_o" not in n:
            sh = s["a"]["shares"]
            out[n] = {"class": "copy" if n.startswith("ident") else "map", "r": sh["r"], "e_gate": sh["e_gate"],
                      "label": label(sh["r"], sh["e_gate"], s["a"]["gain"], s["teacher_gain"])}
    return out


# ----------------------------------------------------------------------------- the job


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
    tmp.replace(path)


def check(path: Path, out: dict[str, Any], name: str, value: float, ok: bool) -> None:
    """A filed stop rule, kept in the file; a miss (a NaN too) stops the job, and a rerun checks it again."""
    log(f"[check {name}] {value:.8g}: {ok}")
    out.setdefault("checks", {})[name] = {"value": value, "pass": ok}
    if not ok:
        out["stopped"] = f"{name}: {value:.8g}"
    dump(path, out)
    if not ok:
        sys.exit(f"STOP: {out['stopped']}")


def near(path: Path, out: dict[str, Any], name: str, a: float, b: float, tol: float) -> None:
    check(path, out, f"{name} (difference)", a - b, abs(a - b) <= tol)


def code_unchanged(commit: str) -> bool:
    """The measurement code and the predictions are as they were at `commit` (later doc commits do not count)."""
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, "--", *CODE], check=False).returncode == 0


def windows(task: str) -> tuple[dict[str, Any], int]:
    """E34's windows and target token: emoticon's from transport_test (harness.load), the others from task_ladder."""
    if task == "emoticon":
        return (torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False, map_location="cpu"),
                json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"])
    w = torch.load(LADDER / f"windows_{task}.pt", weights_only=False, map_location="cpu")
    return w, w["task"]["target"]


class Job:
    def __init__(self, device: str) -> None:
        self.device, self.model = device, vpd.load_component_model(device)
        self.W0, self.comps0 = vpd.target_weights(self.model), vpd.component_weights(self.model)
        self.alive = saved_states.load_alive(TRANSPORT / "large")  # E34's alive.pt (harness.load)
        self.uv = component_training.alive_uv(self.model, self.alive)
        self.uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in self.uv.items()}
        self.lift = AN.lift_context(self.model, self.alive)
        self.pile: Any = None

    def rows(self) -> Any:
        if self.pile is None:
            self.pile = pile_rows("train", CM.pile_start, CM.pile_train_rows, log)
        return self.pile

    def read(self, t: Any, label: str, dW: dict[str, Any], comps: Any) -> dict[str, Any]:
        """The route split at target W0 + dW with components `comps` (None: released). Logged whole."""
        with model_state(self.model, target={k: self.W0[k] + dW[k].to(self.W0[k].device) for k in self.W0}, components=comps):
            rs = route_split(self.model, t.held.tokens, t.held.half, t.token, t.base, self.comps0)
        log(f"[read {label}] {json.dumps(rs, default=float)}")
        return rs


def open_task(task: str, card: str, head: str) -> tuple[Path, dict[str, Any]]:
    path = RESULTS / f"teacher_route_split_{task}.json"
    out = json.loads(path.read_text()) if path.is_file() else {"task": task, "card": card, "commit": head, "config": vars(CM),
                                                               "teachers": {}, "students": {}}
    if out["card"] != card or out["config"] != vars(CM) or not code_unchanged(out["commit"]):
        sys.exit(f"{path} was made on {out['card']} at {out['commit']} with another config, or the code or predictions changed since. "
                 "Move the file aside to start again")
    out.pop("stopped", None)
    return path, out


def fine_tune(job: Job, t: Any, opt: str) -> dict[str, Any]:
    """The teacher's delta-W, regenerated as E34 did (E31's arm A, seed 0), its P(target) checked against E34's."""
    lr = CM.sgd_lr[t.name] if opt == "sgd" else CM.adam_lr
    cfg = ArmAConfig(o_token=t.token, lr=lr, steps=CM.steps, checkpoints=(CM.steps,), seed=0, optimizer=opt)
    res = finetune_arm_a(job.model, t.w["train"], t.held, cfg, job.device, log)
    p = res.evals[CM.steps]["p_o"]
    t.out["teachers"].setdefault(opt, {})["p_target"] = p
    near(t.path, t.out, f"{opt} teacher P(target) against E34's", p, t.e34["teachers"][opt]["repro"]["p_target"], CM.p_tol)
    return res.deltas[CM.steps]


def references(job: Job, t: Any, opt: str, dW: dict[str, Any]) -> None:
    """Do-nothing and the least-squares split of the teacher, with the W0 control's checks."""
    rec, e31 = t.out["teachers"][opt], t.e31[opt]
    if "nothing" not in rec:
        rec["nothing"] = job.read(t, f"{opt} nothing", dW, None)
        rec["least_squares"] = job.read(t, f"{opt} ls", dW, AN.lifted(job.model, job.lift, dW))
        dump(t.path, t.out)
    no, ls = rec["nothing"]["shares"], rec["least_squares"]["shares"]
    near(t.path, t.out, f"{opt} do-nothing r against E31's", no["r"], e31["shares"]["r"], CM.r_tol)
    near(t.path, t.out, f"{opt} do-nothing e_gate against -r", no["e_gate"], -no["r"], CM.zero_tol)
    near(t.path, t.out, f"{opt} do-nothing r against least-squares r", no["r"], ls["r"], CM.zero_tol)


def student(job: Job, t: Any, opt: str, arm: str, net: Any) -> None:
    """E34's student of this teacher and arm, trained, saved, read on its own state and labelled beside its fine-tune."""
    name, path = f"{opt}_{arm}", LARGE / t.name / f"{opt}_{arm}.pt"
    if not (name in t.out["students"] and path.is_file()):
        rows, batch, half = (job.rows(), CM.batch_b, None) if arm == "B" else (t.w["train"].tokens, CM.batch_a, t.w["train"].half)
        t0 = time.time()
        st, _ = teacher.train_student(job.model, job.uv, job.W0, net, rows, t.held, t.token, batch, half, CM.student_lr, CM.steps,
                                      job.device, log)
        arms.save_offsets(path, job.uv_cpu, st, {"task": t.name, "teacher": opt, "arm": arm, "step": CM.steps})
        dW = {k: v[0] for k, v in st.items()}
        comps = component_training.components_with(job.model, job.alive, {k: (v[1], v[2]) for k, v in st.items()})
        p = arms.heldout_of(job.model, job.W0, t.held, t.token, job.device, dW)["p_o"]
        a = job.read(t, f"{name} student", dW, comps)
        ft = t.out["teachers"][opt]
        gain0, ls = ft["nothing"]["gain"], ft["least_squares"]
        t.out["students"][name] = rec = {
            "p_target": p, "seconds": time.time() - t0, "teacher_gain": gain0, "gain_ratio": a["gain"] / gain0, "a": a,
            "label": label(a["shares"]["r"], a["shares"]["e_gate"], a["gain"], gain0),
            "ft_label": label(ls["shares"]["r"], ls["shares"]["e_gate"], ls["gain"], ls["gain"])}
        dump(t.path, t.out)
        log(f"[{t.name} {name}] P {p:.4f} gain ratio {rec['gain_ratio']:.3f} r {a['shares']['r']:.4f} e_gate {a['shares']['e_gate']:.4f} "
            f"-> {rec['label']}; fine-tune (least-squares) r {ls['shares']['r']:.4f} e_gate {ls['shares']['e_gate']:.4f} -> {rec['ft_label']}; "
            f"{rec['seconds']:.0f} s")
    rec = t.out["students"][name]
    near(t.path, t.out, f"{name} student P(target) against E34's", rec["p_target"], t.e34["teachers"][opt][arm][str(CM.steps)]["p_target"], CM.p_tol)


def run_task(job: Job, task: str, card: str, head: str) -> None:
    path, out = open_task(task, card, head)
    w, token = windows(task)
    e31_all = json.loads(E31.read_text())["arms"]
    e31 = {"sgd": e31_all[f"{task}/s0/sgd_{CM.sgd_lr[task]:g}"], "adam": e31_all[f"{task}/s0/adam_{CM.adam_lr:g}"]}
    t = SimpleNamespace(name=task, w=w, held=w["heldout"], token=token, path=path, out=out,
                        e34=json.loads((E34 / f"teacher_training_{task}.json").read_text()),
                        e31={k: v["steps"][str(CM.steps)]["nothing"]["routes"] for k, v in e31.items()})
    t.base = route_cells(job.model, t.held.tokens, t.held.half, token)
    out["W0"] = {k: t.base[k] for k in ("T", "U", "L", "p_T", "p_U", "p_L")}
    for cell, key in (("T0", "T"), ("U0", "U"), ("L00", "L")):
        near(path, out, f"W0 {cell} against E31's", t.base[key], t.e31["sgd"]["cells"][cell], CM.w0_tol)
    for opt in OPTS:
        if all(f"{opt}_{a}" in out["students"] and (LARGE / task / f"{opt}_{a}.pt").is_file() for a in ARMS):
            for a in ARMS:
                student(job, t, opt, a, None)  # recorded: re-checks only
            continue
        dW = fine_tune(job, t, opt)
        references(job, t, opt, dW)
        net = teacher.teacher_net(job.model, dW)
        for a in ARMS:
            student(job, t, opt, a, net)
        del net
    out["done"] = True
    dump(path, out)


def main() -> None:
    tasks = sys.argv[1:] or list(CM.tasks)
    if not set(tasks) <= set(CM.tasks):
        sys.exit(f"tasks are {CM.tasks}")
    require_filed(PRED)
    inputs = [TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", *(LADDER / f"windows_{t}.pt" for t in CM.tasks if t != "emoticon")]
    for p in inputs:
        if not p.is_file():
            sys.exit(f"{p} is missing")
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    log(f"[start] {card} {head} tasks {tasks} inputs {json.dumps({p.name: sha256(p) for p in inputs})}")
    job = Job("cuda" if torch.cuda.is_available() else "cpu")
    for task in tasks:
        run_task(job, task, card, head)
    files = [RESULTS / f"teacher_route_split_{t}.json" for t in CM.tasks]
    outs = [json.loads(f.read_text()) for f in files if f.is_file()]
    if len(outs) == len(files) and all(o.get("done") for o in outs):
        sc = score({f"{o['task']}_{n}": s for o in outs for n, s in o["students"].items()}, e12_control())
        dump(SCORE, sc)
        log(f"[score] {json.dumps(sc)}")
    hard_exit(0)


if __name__ == "__main__":
    main()
