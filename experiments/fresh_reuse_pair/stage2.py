"""E125 stage 2: the decomposition side on stage 1's four saved states (`commute.decomp_reads`; plan
reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md, lines and bars in results/predictions.json). Cheapest first,
each record written as it is read, a rerun skips what is recorded:

  W0   route cells on the route windows (the first n_route fit windows, each task's own label as the target) and W0's
       mean trigger gates over the first n_shift training windows; E12 Amendment 1's students under the same label rule
       (positive control, from file).
  a    per state: the route split at do-nothing and of the least-squares split (E119's references), with E119's checks
       (e_gate = -r at do-nothing; least-squares r = do-nothing r; both to zero_tol): a miss stops the job, exit 1.
  b    per state: E38 Amendment 2's trigger-only removal of S against n_random random sets; the empty set's change
       must be within empty_tol (E38 A2's C3), else exit 1.
  c    per state: E34's arm-A student, saved as bf16 offsets, read by the route split on its own state; exit 3 when
       too little time is left for the next (resubmit; recorded students are kept).

    sbatch cluster/fresh_reuse_pair.sbatch 2
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import torch
from run import (
    HALF,
    HERE,
    LARGE,
    RESULTS,
    TASKS,
    Config,
    L,
    dump,
    load_config,
    load_dW,
    log,
    part,
    say,
    states,
    task,
)
from torch import Tensor

from commute import anchors as AN
from commute import arms, component_training, saved_states, teacher, vpd
from commute import decomp_reads as DR
from commute import definition_reads as DFR
from commute import label_reads as LR
from commute import route_labels as RLB
from commute import steering as ST
from commute.data import LAST
from commute.jobs import hard_exit
from commute.local_faith import MIN_GAIN, route_cells, trigger_mean_gates

ALIVE = HERE.parent / "transport_test" / "results" / "large"  # E34's and E119's alive.pt
E12A1 = HERE.parent / "new_vs_existing" / "results" / "new_vs_existing_amend1.json"


class Stop(Exception):
    pass


def check(res: dict[str, Any], name: str, value: float, tol: float) -> None:
    """A filed stop rule: |value| <= tol (a nan misses), kept in the file."""
    ok = abs(value) <= tol
    res.setdefault("checks", {})[name] = {"value": value, "pass": ok}
    log(f"[check {name}] {value:.3g} (tol {tol:g}): {ok}")
    if not ok:
        res["stopped"] = f"{name}: {value:.6g}"
        raise Stop


class Job:
    def __init__(self, net: ST.Net, C: Config, P: dict[str, Any]) -> None:
        self.net, self.C, self.D = net, C, P["decomposition"]
        self.lines = RLB.Lines(**self.D["route_lines"])
        self.model, self.W0 = net.model, net.W0
        self.alive = saved_states.load_alive(ALIVE)
        self.comps0 = vpd.component_weights(self.model)
        self.uv = component_training.alive_uv(self.model, self.alive)
        self.lift = AN.lift_context(self.model, self.alive)
        self.t = task()
        fit = self.t["toks"]["fit"][: C.n_route]
        self.route = {k: LR.labelled(fit, self.t["y"][k]["fit"][: C.n_route], self.t["ab"], HALF).tokens for k in TASKS}
        self.base = {k: route_cells(self.model, self.route[k], HALF, LAST, 64, ci_len=L) for k in TASKS}
        self.train = self.t["toks"]["train"][: C.n_shift]
        self.m0 = trigger_mean_gates(self.model, self.train, HALF)

    def split(self, k: str, dW: ST.DW, comps: Any) -> dict[str, Any]:
        return DR.read_split(self.model, self.W0, dW, comps, self.route[k], HALF, LAST, self.base[k], self.comps0, L)

    def label(self, rs: dict[str, Any], teacher_gain: float) -> str:
        return RLB.label(rs["shares"]["r"], rs["shares"]["e_gate"], rs["gain"], teacher_gain, self.lines)


def references(job: Job, k: str, dW: ST.DW, res: dict[str, Any], n: str) -> None:
    rec = res["states"].setdefault(n, {})
    if "least_squares" not in rec:
        rec["nothing"] = job.split(k, dW, None)
        rec["least_squares"] = job.split(k, dW, AN.lifted(job.model, job.lift, dW))
        ls = rec["least_squares"]
        rec["ft_label"] = job.label(ls, ls["gain"])
        say(f"split {n}", {"nothing": rec["nothing"]["shares"], "least_squares": ls["shares"], "gain": ls["gain"], "label": rec["ft_label"]})
        dump(2, res)
    no, ls = rec["nothing"]["shares"], rec["least_squares"]["shares"]
    check(res, f"{n} do-nothing e_gate + r", no["e_gate"] + no["r"], job.D["zero_tol"])
    check(res, f"{n} least-squares r - do-nothing r", ls["r"] - no["r"], job.D["zero_tol"])


def removal(job: Job, k: str, dW: ST.DW, res: dict[str, Any], n: str) -> None:
    rec = res["states"][n]
    if "removal" not in rec:
        S, _ = DR.switched_set(job.model, job.W0, dW, job.train, HALF, job.m0, job.alive, job.C.switch_thr)
        rec["removal"] = DR.removal_d(job.model, job.W0, dW, job.comps0, S, job.alive, job.route[k], HALF, LAST, job.C.n_random_sets, MIN_GAIN, L)
        r = rec["removal"]
        say(f"removal {n}", {"n_S": r["n_S"], "gain": r["gain"], "D_S": r["D"]["S"], "D_random_max": max(r["D"]["random"], default=float("nan")),
                             "rho_0_S": r["rho_0"]["S"], "empty": r["empty"], "class": r["class"]})  # fmt: skip
        dump(2, res)
    check(res, f"{n} empty-set removal lo change", rec["removal"]["empty"], job.D["empty_tol"])


def student(job: Job, k: str, dW: ST.DW, res: dict[str, Any], n: str) -> None:
    """E34's arm-A student of this state, read on its own state (Delta fixed) and labelled beside its fine-tune."""
    C, rec, t = job.C, res["states"][n], job.t
    t0, ev = time.time(), LR.labelled(t["toks"]["fit"][: C.n_student_eval], t["y"][k]["fit"][: C.n_student_eval], t["ab"], HALF)
    net_t = teacher.teacher_net(job.model, dW)
    st, curve = teacher.train_student(job.model, job.uv, job.W0, net_t, t["toks"]["train"], ev, LAST, C.student_batch, HALF, C.student_lr,
                                      C.student_steps, job.net.device, log)  # fmt: skip
    del net_t
    arms.save_offsets(LARGE / f"student_{n}.pt", {m: (u.cpu(), v.cpu()) for m, (u, v) in job.uv.items()}, st, {"state": n, "step": C.student_steps})
    dWs = {m: v[0] for m, v in st.items()}
    a = job.split(k, dWs, component_training.components_with(job.model, job.alive, {m: (v[1], v[2]) for m, v in st.items()}))
    held, yh = part(t, k, "held")
    fit, yf = part(t, k, "fit")
    d_s, d_t = DFR.label_diff(job.net, held, t["ab"], HALF, dWs), DFR.label_diff(job.net, held, t["ab"], HALF, dW)
    gain0 = rec["nothing"]["gain"]
    rec["student"] = {"a": a, "gain_ratio": a["gain"] / gain0 if gain0 else float("nan"), "label": job.label(a, gain0),
                      "agree_with_teacher_held": float(((d_s > 0) == (d_t > 0)).double().mean()), "acc_held": DFR.accuracy(d_s, yh),
                      "acc_fit": DFR.accuracy(DFR.label_diff(job.net, fit, t["ab"], HALF, dWs), yf), "last_curve": curve[-1], "seconds": time.time() - t0}  # fmt: skip
    s = rec["student"]
    say(f"student {n}", {"r": a["shares"]["r"], "e_gate": a["shares"]["e_gate"], "gain_ratio": s["gain_ratio"], "label": s["label"],
                         "ft_label": rec["ft_label"], "agree": s["agree_with_teacher_held"], "acc_fit": s["acc_fit"], "seconds": s["seconds"]})  # fmt: skip
    dump(2, res)


def summary(res: dict[str, Any], reads: dict[str, Any]) -> dict[str, Any]:
    """Per state, the definition's verdict beside the three decomposition labels (no rule joins them)."""
    return {n: {"definition": reads[n]["verdict"], "route_fine_tune": r["ft_label"], "route_copy": r.get("student", {}).get("label"),
                "removal_S": r["removal"]["class"]} for n, r in res["states"].items()}  # fmt: skip


def run(net: ST.Net, C: Config, P: dict[str, Any], t0: float) -> int:
    s1 = json.loads((RESULTS / "stage1.json").read_text())
    assert "stopped" not in s1 and len(s1.get("reads", {})) == len(states(C)), "stage 1 is not complete: stage 2 does not run"
    p = RESULTS / "stage2.json"
    res: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"states": {}}
    if "stopped" in res:
        sys.exit(f"stage 2 stopped before ({res['stopped']}): move stage2.json aside to start again")
    job = Job(net, C, P)
    res["W0"] = {k: {c: job.base[k][c] for c in ("T", "U", "L", "p_T")} for k in TASKS}
    res["control_e12_amend1"] = RLB.e12_control(E12A1, job.lines)
    say("W0 route cells; E12 A1 control", {"W0": res["W0"], "control": {n: v["label"] for n, v in res["control_e12_amend1"].items()}})
    dWs: dict[str, dict[str, Tensor]] = {n: load_dW(n, s1["states"][n]) for n in states(C)}
    try:
        for step in (references, removal):
            for n, dW in dWs.items():
                step(job, n.split("_s")[0], dW, res, n)
        for n, dW in dWs.items():
            if "student" in res["states"][n]:
                continue
            if time.monotonic() - t0 > (C.stage_minutes - C.student_minutes) * 60:
                log(f"[time] too little left for the student of {n}: exit 3, resubmit stage 2")
                dump(2, res)
                return 3
            student(job, n.split("_s")[0], dW, res, n)
    except Stop:
        log(f"[stop] {res['stopped']}")
        dump(2, res)
        return 1
    res["summary"] = summary(res, s1["reads"])
    dump(2, res)
    log("SUMMARY " + json.dumps(res["summary"]))
    return 0


def main() -> None:
    t0 = time.monotonic()
    C, P = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    code = run(ST.Net("cuda"), C, P, t0)
    log({0: "STAGE 2 DONE", 1: "STAGE 2 STOPPED (exit 1)"}.get(code, f"stage 2: exit {code}, more to do"))
    hard_exit(code)


if __name__ == "__main__":
    main()
