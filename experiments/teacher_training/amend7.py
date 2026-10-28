"""teacher_training Amendment 7 (2026-10-28): the LoRA's Pile-text copies on r1 and r0 to 10,000 steps. Plan:
reports/plans/2026-10-23_plan_teacher-training.md -> "Amendment 7" (binding); the approved branch of Amendment 6's tree
("plateau with a larger gap, or rising: that task's student B to 10,000 steps").

One job, every state read on E34's harness (run.read_state) plus Amendment 6's plain-text readings (amend6.pile_read).
The teachers are E9's rank-8 LoRA, regenerated in-job (amend3.lora_teacher); the student is Amendment 6's stage-d student
B (Pile train rows 32,768-40,959, batch 8, plain mean KL, Adam 1e-5, batch-order seed 0), run to 10,000 steps from
scratch: the schedule is constant and the batch order does not depend on the step count, so its first 5,000 steps are
Amendment 6's run.
  C1   W0 under the Pile harness (run.Job); then W0's own plain-text damage must read 0; stop otherwise
  G1   each LoRA teacher matches its same-card reference (Amendment 6's); stop otherwise
  G2   r1 first, alone: student B to 1,000 steps reproduces Amendment 6's state there (P, size); stop otherwise
  r1, r0   do-nothing (no adversary); student B to 10,000 steps, states at 5,000, 7,500, 10,000 (adversary at 5,000 and
       10,000); the step-10,000 state saved as bf16 offsets
  G3   each student's step-5,000 state reproduces Amendment 6's (P, adversarial ratio); stop otherwise (before r0)
  score P1-P4 (`score`, pure) -> out["score"]
The scored behaviour is P-bar, the mean of the free held-out P(target) over the last AM.window steps, not a single
checkpoint: Amendment 6's single readings of r1 wandered over 0.852-0.933 from step 1,500 on.
Each state is dumped to results/teacher_training_amend7.json as it is read and skipped on a rerun (same card and W0 floor
only, run.load); teachers are regenerated on a rerun (about 40 s each), and a student with any state missing is retrained.

Usage (repo root, project venv):  amend7.py --predictions   files results/predictions_amend7.json (refuses to overwrite)
                                  amend7.py [cuda]          the run (GPU); the predictions file must be filed first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from amend3 import lora_teacher
from amend4 import train_student_seeded
from amend6 import AM as AM6
from amend6 import pile_read, summary
from run import (
    CFG,
    LARGE,
    R0,
    R1,
    RESULTS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    read_state,
)

PRED7, OUT7 = RESULTS / "predictions_amend7.json", RESULTS / "teacher_training_amend7.json"


@dataclass(frozen=True)
class Amend7:
    tasks: tuple[str, ...] = (R1, R0)  # r1 first: its teacher also serves G2
    steps: tuple[int, ...] = (5000, 7500, 10000)  # states read (plain text at each)
    harness_at: tuple[int, ...] = (5000, 10000)  # the adversary read at these
    window: int = 1000  # P-bar: mean of the free held-out P(target) over the last `window` steps
    band: float = 0.03  # "at the bar": |P-bar - 0.9| < band (Amendment 4's batch-order spread of P, 0.032)
    rise: float = 0.03  # below the band: plateau if P-bar rose by less than this from step 5,000 to 10,000 (A6's rule)
    gap_far: float = 0.5  # P4: plain-text gap / the teacher's gap at W0 above this = not carried
    control_steps: int = 1000  # G2
    # Amendment 6's seed-0 states (teacher_training_amend6.json), same card and W0 floor; A6 reproduced A3 to 5e-5 in P
    a6_1000: dict[str, float] = field(default_factory=lambda: {"p_target": 0.87530, "size": 0.01634})  # r1 only
    a6_5000: dict[str, dict[str, float]] = field(default_factory=lambda: {
        R1: {"p_target": 0.88704, "pgd": 1.07108}, R0: {"p_target": 0.81925, "pgd": 0.95552}})
    repro_tol: dict[str, float] = field(default_factory=lambda: {"p_target": 0.002, "size": 0.0005, "pgd": 0.03})
    w0_pile_max: float = 1e-4  # W0's plain-text damage (Amendment 6's control)

    def __post_init__(self) -> None:
        assert self.steps[0] == AM6.seed0_steps[-1] == 5000 and self.steps[-1] == 10000 and set(self.harness_at) <= set(self.steps)
        assert all(s % self.window == 0 for s in (self.steps[0], self.steps[-1], self.control_steps))
        assert self.steps[-1] % AM6.curve_every == 0 and self.tasks[0] == R1


AM = Amend7()
REFS = {t: AM6.refs[f"{t}/lora"] for t in AM.tasks}  # the LoRA teachers' same-card readings (E27/A3 H100)


class Run7:
    """The job, its results file, the plain-text rows and W0's frozen copy."""

    def __init__(self, job: Job, out: dict[str, Any]) -> None:
        from commute import teacher

        self.job, self.out = job, out
        self.rows = torch.cat([b for block in job.ctx.blocks for b in block])  # the harness's 256 Pile val rows
        self.w0_net = teacher.teacher_net(job.ctx.model, {})

    def teacher(self, task: str) -> tuple[dict[str, Any], Any]:
        """G1: the regenerated LoRA teacher, its reading against REFS (stop on a mismatch) and its gaps at W0."""
        from commute import teacher
        from commute.finetune import eval_arm_b

        dW, reading = lora_teacher(self.job, task)
        tctx = self.job.task(task)[0]
        net = teacher.teacher_net(tctx.model, dW)
        ok = all(abs(reading[k] - v) <= AM6.ref_tol[k] for k, v in REFS[task].items())
        self.out["teachers"][task] = {"reading": reading, "reference": REFS[task], "matches": ok, "gap_w0": gap_w0(tctx, net),
                                      "pile_gap_w0": eval_arm_b(self.w0_net, net, self.rows, tctx.device)["kl_to_ref"]}
        log(f"[{task}/lora teacher] {json.dumps(self.out['teachers'][task], default=float)}")
        dump(OUT7, self.out)
        if not ok:
            halt(OUT7, self.out, f"G1: {task} LoRA teacher {json.dumps(reading, default=float)} against {REFS[task]}")
        return dW, net

    def state(self, task: str, label: str, upd: tuple[Any, Any], net: Any, harness: bool) -> dict[str, Any]:
        """E34's record of one state (the adversary if `harness`), its plain-text readings, and with the adversary the
        faithfulness verdict alone ('faith': state_verdict with no P bar); dumped and logged."""
        from commute import faith_score

        if label in self.out["states"]:
            return self.out["states"][label]
        tctx = self.job.task(task)[0]
        rec = {**read_state(self.job, tctx, label, upd, net, harness), **pile_read(tctx, self.w0_net, self.rows, upd[0], net)}
        if harness:
            rec["faith"] = faith_score.state_verdict({"pgd_kl": rec["pgd"], "stochastic_kl": rec["stochastic"]}, 1.0,
                                                     self.job.w0)["verdict"]
        log(f"[{label}] pile_damage {rec['pile_damage']:.5g} pile_gap {rec['pile_gap']:.5g}" + (f" faith {rec['faith']}" if harness else ""))
        self.out["states"][label] = rec
        dump(OUT7, self.out)
        return rec

    def student(self, task: str, net: Any, steps: tuple[int, ...], harness_at: tuple[int, ...], save: bool) -> None:
        """Student B of the task's LoRA teacher at batch-order seed 0, read at `steps` (the adversary at `harness_at`);
        the curve kept as Amendment 6's marks plus P-bar at every AM.window steps; the last state saved if `save`."""
        from commute import arms

        labels = {s: f"{task}/lora/B0@{s}" for s in steps}
        if all(lb in self.out["states"] for lb in labels.values()):
            return
        tctx = self.job.task(task)[0]
        t0 = time.time()
        curve, states = train_student_seeded(tctx, net, "adam", CFG.student_lr, steps, self.job.pile, CFG.batch_b, None, 0)
        p = [c["heldout_p_o"] for c in curve]
        self.out["curves"][f"{task}@{steps[-1]}"] = {
            "marks": summary(curve), "p_bar": {str(s): sum(p[s - AM.window:s]) / AM.window for s in range(AM.window, steps[-1] + 1, AM.window)}}
        log(f"[{task}/lora/B0] {steps[-1]} steps in {time.time() - t0:.0f} s; P-bar {json.dumps(self.out['curves'][f'{task}@{steps[-1]}']['p_bar'])}")
        for s, lb in labels.items():
            st = states.pop(s)
            self.state(task, lb, ({k: v[0] for k, v in st.items()}, st), net, s in harness_at)
            if save and s == steps[-1]:
                arms.save_offsets(LARGE / task / f"amend7_lora_B_seed0_step{s}.pt", tctx.uv_cpu, st,
                                  {"task": task, "teacher": "lora", "arm": "B", "seed": 0, "step": s})
                log(f"[saved] {task}/amend7_lora_B_seed0_step{s}.pt")


def control(r: Run7) -> None:
    """C1 (run.Job's W0 check) and W0's own plain-text damage, before anything else."""
    if not r.job.w0["pass"]:
        halt(OUT7, r.out, "C1: W0 is not at its known values; nothing read")
    ctx = r.job.ctx
    r.out["states"]["W0"] = pile_read(ctx, r.w0_net, r.rows, {k: torch.zeros_like(v) for k, v in ctx.W0.items()}, None)
    log(f"[W0] {json.dumps(r.out['states']['W0'], default=float)}")
    dump(OUT7, r.out)
    if r.out["states"]["W0"]["pile_damage"] > AM.w0_pile_max:
        halt(OUT7, r.out, f"W0 plain-text damage {r.out['states']['W0']['pile_damage']:.3g} > {AM.w0_pile_max:g}")


def reproduces(task: str, step: int, ref: dict[str, float], r: Run7) -> None:
    """G2 (step 1,000) and G3 (step 5,000): the state matches Amendment 6's within AM.repro_tol; stop otherwise."""
    rec, gate = r.out["states"][f"{task}/lora/B0@{step}"], "G2" if step == AM.control_steps else "G3"
    ok = all(abs(rec[k] - v) <= AM.repro_tol[k] for k, v in ref.items())
    r.out.setdefault(gate, {})[task] = {"read": {k: rec[k] for k in ref}, "a6": ref, "pass": ok}
    log(f"[{gate} {task}] {json.dumps(r.out[gate][task], default=float)}")
    dump(OUT7, r.out)
    if not ok:
        halt(OUT7, r.out, f"{gate}: {task} student B at step {step} reads {r.out[gate][task]['read']}, Amendment 6 {ref}")


def task_stage(r: Run7, task: str) -> None:
    from commute.data import pile_rows

    dW, net = r.teacher(task)
    r.state(task, f"{task}/lora/nothing", (dW, None), net, False)
    if r.job.pile is None:
        r.job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    if task == R1:  # G2: 1,000 steps alone (about 90 s) before 10,000
        r.student(task, net, (AM.control_steps,), (), save=False)
        reproduces(task, AM.control_steps, AM.a6_1000, r)
    r.student(task, net, AM.steps, AM.harness_at, save=True)
    reproduces(task, AM.steps[0], AM.a6_5000[task], r)
    log(f"[{task}] done; score so far {json.dumps(score(r.out), default=float)}")


# ---- scoring (pure: the results file only) -----------------------------------------------------------------------------

def category(p_end: float, p_mid: float, faith: str) -> str:
    """P-bar at 10,000 (`p_end`) and at 5,000 (`p_mid`), faith at 10,000: 'at the bar' if |p_end - 0.9| < AM.band;
    above it 'learnt' (faith pass) or 'learnt, not passed'; below it 'plateau' (rise < AM.rise) or 'rising'. A value on
    the band's edge counts as outside it."""
    d = p_end - CFG.p_min
    if abs(d) < AM.band:
        return "at the bar"
    if d > 0:
        return "learnt" if faith == "pass" else "learnt, not passed"
    return "plateau" if p_end - p_mid < AM.rise else "rising"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1-P4 over whatever the file holds; a prediction whose inputs are missing is absent, a partial pool never passes."""
    st, last, mid = out["states"], AM.steps[-1], AM.steps[0]
    sc: dict[str, Any] = {"C1": out["W0"]["pass"], "G2": out.get("G2"), "G3": out.get("G3"),
                          "G1": {t: v["matches"] for t, v in out["teachers"].items()}}
    per = {}
    for t in AM.tasks:
        rec, pb = st.get(f"{t}/lora/B0@{last}"), out["curves"].get(f"{t}@{last}", {}).get("p_bar", {})
        if rec and "faith" in rec and str(last) in pb:
            per[t] = {"p_bar": pb[str(last)], "p_bar_5000": pb[str(mid)], "faith": rec["faith"],
                      "category": category(pb[str(last)], pb[str(mid)], rec["faith"])}
    if R1 in per:
        sc["P1_r1_at_the_bar"] = {**per[R1], "pass": per[R1]["category"] == "at the bar"}
    if R0 in per:
        sc["P2_r0_reaches_the_bar"] = {**per[R0], "pass": per[R0]["category"] in ("at the bar", "learnt")}
    labels = [f"{t}/lora/B0@{s}" for t in AM.tasks for s in AM.harness_at]
    faith = {lb: st[lb]["faith"] for lb in labels if lb in st}
    if faith:
        sc["P3_copies_pass"] = {"per": faith, "pass": len(faith) == len(labels) and all(v == "pass" for v in faith.values())}
    gap = {}  # matching ratio; None when the teacher's own gap at W0 is not positive (then P4 cannot pass)
    for t in AM.tasks:
        lb, w0 = f"{t}/lora/B0@{last}", out["teachers"].get(t, {}).get("pile_gap_w0")
        if lb in st and w0 is not None:
            gap[t] = st[lb]["pile_gap"] / w0 if w0 > 0 else None
    if gap:
        sc["P4_gap_stays_far"] = {"per": gap, "pass": len(gap) == len(AM.tasks)
                                  and all(g is not None and g > AM.gap_far for g in gap.values())}
    return sc


# ---- predictions and the run --------------------------------------------------------------------------------------------

PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training Amendment 7 (reports/plans/2026-10-23_plan_teacher-training.md -> 'Amendment 7'): Amendment 6's "
             "approved branch, student B (Pile text) of E9's rank-8 LoRA teacher on r1 and r0 to 10,000 steps, batch-order "
             "seed 0, one RunPod H100 job. Teachers regenerated in-job (amend3.lora_teacher). The step-5,000 states must "
             "reproduce Amendment 6's (G3). Amendment 6's predictions file stands unedited. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2), "
              "faith_score.w0_check; expected pass, floor about 0.0748 on the H100. Fails: stop",
        "W0_plain_text": "W0's plain-text damage KL(W0 || W0) over the 256 rows <= 1e-4 (expected 0, Amendment 6 read 0). Fails: stop",
        "G1": "each LoRA teacher matches its same-card reading (Amendment 6's refs: P within 0.01, damage within 0.005, size "
              "within 0.003; A6 matched all). Fails: stop",
        "G2": "r1 first, alone: student B to 1,000 steps reads P(target) within 0.002 of 0.87530 and size within 0.0005 of "
              "0.01634 (Amendment 6's state; A6 reproduced A3's three step-1,000 readings to 5e-5). Fails: stop",
        "G3": "each 10,000-step student's step-5,000 state reads P(target) within 0.002 and adversarial ratio within 0.03 "
              "of Amendment 6's (r1 0.88704, 1.0711; r0 0.81925, 0.9555). Fails: stop (r1's before r0 starts)",
    },
    "measurements": {
        "p_bar": "the mean of the per-step free held-out P(target) at the trigger (component_training.train's curve, "
                 "'heldout_p_o') over the last 1,000 steps, at every 1,000 steps; read at 10,000 (steps 9,001-10,000) and "
                 "5,000 (4,001-5,000). Blind to batch order (one order) and to the up-to-0.002 difference between the "
                 "free reading and read_state's held-out P. Noise floor: Amendment 4's batch-order spread of P for "
                 "Pile-text copies, 0.032, and Amendment 6's second order at step 2,500 (0.002 r1, 0.032 r0)",
        "faith": "state_verdict on the adversarial and stochastic ratios to this job's W0, no P bar, at 5,000 and 10,000: "
                 "pass at or below 1.396 of the floor 0.0748, fail at or above 1.612. One adversary start of 20 steps, a "
                 "lower bound (Amendment 5)",
        "plain_text": "Amendment 6's pile_damage KL(W0 || state) and pile_gap KL(teacher || state) on the 256 Pile val "
                      "rows; the gap divided by the teacher's own KL(teacher || W0) on the same rows, 'matching' (1 at "
                      "W0, 0 at the teacher), scored; the gap divided by the do-nothing pile_damage, Amendment 6's "
                      "'filed' reading, is computed from the file in the analysis, not scored",
    },
    "expected": {
        "W0": "P(target) 0.04 r1, 0.15 r0; ratios 1; pile_damage 0; matching gap 1 by definition",
        "do_nothing": "P(target) 0.9993 r1, 0.9828 r0; adversarial ratio 1.003 r1, 1.009 r0 (A3, this card; not re-read); "
                      "pile_damage 0.01295 r1, 0.00650 r0; matching gap 0",
        "controls": "G2 0.87530 at 1,000; G3 at 5,000 r1 0.88704 and 1.071, r0 0.81925 and 0.956",
        "students": "r1 P-bar about 0.90 at 5,000 (A6's 50-step means 0.896-0.916 from step 3,000 on, training loss flat "
                    "from step 250); r0 about 0.78 at 5,000, rising about 0.07 per 1,000 steps; sizes 0.022-0.024 at "
                    "5,000 and growing; every adversarial read of a Pile-text LoRA copy so far passed (0.87-1.09)",
    },
    "predictions": [
        {"id": "P1_r1_at_the_bar", "confidence": "medium",
         "rule": "category of r1 at 10,000: d = P-bar(10,000) - 0.9; |d| < 0.03 -> 'at the bar'; d >= 0.03 -> 'learnt' "
                 "if faith 'pass' at 10,000, else 'learnt, not passed'; d <= -0.03 -> 'plateau' if P-bar(10,000) - "
                 "P-bar(5,000) < 0.03, else 'rising'. Pass = 'at the bar'",
         "predicted": "at the bar: r1's 50-step mean held-out P sat at 0.896-0.916 from step 3,000 to 5,000 with a flat "
                      "training loss, so 5,000 more steps of the same loss should leave it near 0.9"},
        {"id": "P2_r0_reaches_the_bar", "confidence": "medium-low",
         "rule": "category of r0 at 10,000 by P1's rule; pass = 'at the bar' or 'learnt'",
         "predicted": "reaches the bar (lean: at the bar, as r1 plateaued there): r0's 50-step mean rose 0.617 -> 0.793 "
                      "from step 2,500 to 5,000 and was still rising"},
        {"id": "P3_copies_pass", "confidence": "high",
         "rule": "faith at steps 5,000 and 10,000 on r1 and r0; pass = all four present and 'pass'",
         "predicted": "all pass (every Pile-text LoRA copy read so far passed, 0.87-1.09)"},
        {"id": "P4_gap_stays_far", "confidence": "medium",
         "rule": "at 10,000, per task: pile_gap / the teacher's KL(teacher || W0) on the same rows ('matching'); pass = "
                 "both present, denominators > 0, and both above 0.5",
         "predicted": "holds: at 5,000 the matching gaps read 1.12 r1 and 1.52 r0, having fallen by 0.13 and 0.30 from "
                      "step 1,000; plain text has not carried the LoRA's change to plain text and 5,000 more steps "
                      "should not close half of it"},
    ],
    "reported_not_scored": [
        "P-bar at every 1,000 steps; Amendment 6's curve marks (single P, 50-step mean, loss) every 250 steps",
        "single held-out P, size, damage, teacher gap and Delta's share at 5,000, 7,500, 10,000; ratios at 5,000 and 10,000",
        "plain-text damage and gap at each state (the analysis divides them by the do-nothing's plain-text damage)",
    ],
    "not_run": [
        "a second batch order (the only way to settle a category inside the band)",
        "emoticon past 5,000 (learnt in Amendment 6); task-text copies; any full-weight teacher",
        "more adversary starts or a longer adversary (decided 2026-10-27)",
        "the step-7,500 state under the adversary",
    ],
}


def file_predictions() -> None:
    if PRED7.is_file():
        sys.exit(f"{PRED7} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED7, {**PREDICTIONS, "config": asdict(CFG), "amend7": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED7}")


def amend7(device: str) -> None:
    if not PRED7.is_file():
        sys.exit(f"file the predictions first: {PRED7} (amend7.py --predictions)")
    filed = json.loads(PRED7.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend7"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend7']}"
    job = Job(device)
    r = Run7(job, load(OUT7, job, {"states": {}, "teachers": {}, "curves": {}}))
    dump(OUT7, r.out)
    control(r)
    for task in AM.tasks:
        task_stage(r, task)
    r.out["score"], r.out["done"] = score(r.out), True
    dump(OUT7, r.out)
    log(f"[score] {json.dumps(r.out['score'], default=float)}")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend7(args.device)
