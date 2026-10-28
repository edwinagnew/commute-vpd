"""teacher_training E42 (2026-10-28): teacher training on a behaviour the model lacks. Plan:
reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md (design check, numbers, predictions, decision tree).

E34 copied fine-tunes of behaviours the model half-has. Here the teacher is a fine-tune of E10's key -> value cipher
(experiments/kv_task; windows from its build stage, loaded by commute.kv_windows): after a marker token, a key word is
followed by table[key]. Per task (rand32 first; then ident32, the matched copy control, and rand1) the teacher is E10's
recipe run on to step 1000 (arm A's loss, all 24 matrices, Adam 3e-5, batch 32, seed 0; E34's teacher length), then
E34's arms A (task windows) and B (Pile train rows) with run.py's CFG, scored at steps 100 (no Pile harness) and 1000
(Pile harness). Every state also reads the route split at the trigger, whose e_gate share is the masked-off share.
Stages, each record dumped as it finishes to results/teacher_training_e42_<task>.json (one file per task, so tasks can
run on different cards), finished work skipped on a rerun on the same card:
  C1   W0 under the Pile harness, adversary seeds 0, 1, 2 (faith_score.w0_check); stop if it fails
  per task:
  G1   the teacher's held-out P(target) at step 10 within 0.02 of E10's (same recipe, same seed: a prefix of E10's
       run); stop otherwise (the windows, table or recipe differ from E10's)
  gate a teacher below P(target) 0.9 at step 1000 is reported and its students skipped
  refs the teacher's gap at W0; do-nothing (the reference) and the least-squares split (the harness's positive
       control), both through the Pile harness; what the teacher shows on plain Pile text at key words
  A, B the students; their step-1000 states saved as bf16 offsets
  score the filed rules on the task's file

Usage (repo root, project venv, GPU):  newskill.py --predictions          (files results/predictions_e42.json once)
                                       newskill.py rand32 [ident32 rand1]  [--device cuda]
The experiment's helpers come from run.py (the script's directory is on sys.path when it runs as a script).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run import (
    CFG,
    HERE,
    LARGE,
    RESULTS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    record,
    train_student,
)

PRED = RESULTS / "predictions_e42.json"
KV = HERE.parent / "kv_task" / "results" / "large"  # windows_<task>.pt from kv_task/run.py's build stage (E10)


def out_path(name: str) -> Path:
    return RESULTS / f"teacher_training_e42_{name}.json"


@dataclass(frozen=True)
class E42:
    tasks: tuple[str, ...] = ("rand32", "ident32", "rand1")  # rand32 first: 32 new pairs, E12's and E28's cipher
    teacher_lr: float = 3e-5  # E10's fine-tune (kv_task FT_LR): arm A's loss, Adam, batch 32, clip 1, seed 0
    teacher_checkpoints: tuple[int, ...] = (10, 300, 1000)  # 10 is G1's, 300 E10's last, 1000 the teacher
    # E10's held-out P(target) of the same run at steps 10 and 300 (kv_task.json, L40); E28 read 0.5858 for rand32 @10 on a 4090
    e10: tuple[tuple[str, float, float], ...] = (("rand32", 0.5847, 0.9965), ("ident32", 0.9595, 0.9995),
                                                  ("rand1", 0.9982, 1.0000))
    g1_tol: float = 0.02  # G1: |P(target) at step 10 - E10's| <= this
    teacher_p_min: float = 0.9  # the task gate
    pile_reading_rows: int = 1024  # the first of arm B's rows, for the teacher's reading at key words
    masked_hi: float = 0.5  # P2: e_gate share >= this -> 'masked-off'
    masked_lo: float = 0.25  # P2: e_gate share <= this -> 'not masked-off'; between -> 'mixed'

    def __post_init__(self) -> None:
        assert tuple(t for t, *_ in self.e10) == self.tasks
        assert self.teacher_checkpoints[0] == 10 and self.teacher_checkpoints[-1] == CFG.steps[-1]
        assert self.pile_reading_rows <= CFG.pile_train_rows and self.masked_lo < self.masked_hi


AM = E42()
LEAN_P2 = {"rand32": "masked-off", "rand1": "masked-off", "ident32": "not masked-off"}
PREDICTIONS: dict[str, Any] = {
    "about": "E42, teacher training on a behaviour the model lacks (reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md; "
             "gap 1 of reports/reviews/2026-10-28_review_technique-comparison.md). Tasks rand32, ident32, rand1: E10's key -> "
             "value windows (experiments/kv_task/results/large/windows_<task>.pt; 403 train, 97 held-out windows each). "
             "Teacher: E10's fine-tune recipe (arm A's loss with the table target, all 24 matrices, Adam 3e-5, batch 32, clip 1, "
             "seed 0) run to step 1000. Students: E34's arms A (task windows, arm A's weighting, Adam 1e-5, batch 32) and B (Pile "
             "train rows from 32,768, 8,192 rows, batch 8 of 512 tokens, plain mean KL, Adam 1e-5), 1,000 steps, batch-order "
             "seed 0, Gamma and Delta frozen. Step 100 read without the Pile harness, step 1000 with it. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2), "
              "faith_score.w0_check. Expected: pass; floor about 0.21 on an L40 (pass line 1.24, fail line 1.81), 0.075 on an "
              "H100 (1.40, 1.61). Fails: stop, nothing trained",
        "G1": "per task, before anything else is read: the teacher's held-out P(target) at step 10 within 0.02 of E10's same "
              "run (rand32 0.5847, ident32 0.9595, rand1 0.9982; E28 reproduced rand32 at 0.5858 on another card). Fails: "
              "stop with both numbers (windows, table or recipe differ from E10's)",
        "task_gate": "a teacher with held-out P(target) < 0.9 at step 1000 is reported with its references and its students "
                     "skipped (a skip: E10's run read 0.9965 / 0.9995 / 1.0000 at step 300)",
        "positive_control": "the least-squares split of the teacher's delta-W (split.split_state): target weights = the "
                            "teacher's, so P(target) is the teacher's; E10 read its Pile adversarial ratio at 38.9 / 9.4 / "
                            "1.73 at step 10. 'power' = its verdict is 'fail'. Without power on a task, a student's pass "
                            "there is read as 'no failing state shown on this task', not scored differently",
    },
    "measurements": {
        "per state": "run.record's numbers (adversarial and stochastic ratios to this job's W0 with faith_score.state_verdict, "
                     "held-out P(target), teacher gap, damage elsewhere, size, Delta's share) plus the route split at the "
                     "trigger on the held-out windows (local_faith.route_split): gain in log-odds of the target and the "
                     "shares d, r, i, e_gate, e_delta of it. e_gate is the masked-off share: the part of the gain present with "
                     "every subcomponent on but lost under Gamma's mask. Shares are not scored when the gain < 1.0 log-odds",
        "pile reading": "kv_windows.pile_reading on the first 1,024 of arm B's rows: mean P(table[x]) as the next token at "
                        "every position holding a key x, for the teacher and for W0; counts of the marker and of the marker "
                        "followed by a key. Reported; it says whether plain Pile text shows the teacher's map at all",
    },
    "expected": {
        "W0": "P(target) about 1.5e-5 (rand32), 1.7e-4 (ident32, rand1) (E10's route-split T0); every ratio 1; teacher gap = "
              "the teacher's whole effect, about -log P_W0(target): 11 nats rand32, 8.7 ident32 and rand1; shares undefined",
        "teacher (= do-nothing)": "P(target) >= 0.99 at step 1000 (E10: 0.9965 / 0.9995 / 1.0000 at 300); KL elsewhere 0.04-0.08, "
                                  "size 0.03-0.045 (E10 at 300: 0.052 / 0.051 / 0.045 and 0.031 / 0.027 / 0.025). Do-nothing: "
                                  "gap 0, Delta's share 1, e_delta 1, e_gate = -r; its adversarial ratio is not predicted (E10 "
                                  "at step 10, damage elsewhere 0.70 / 0.19 / 0.04: 4.73 / 8.12 / 1.06)",
        "least-squares split": "P = the teacher's, gap 0, adversarial ratio >= 1.81 on rand32 and ident32 (power), rand1 "
                               "uncertain; e_gate share 0.52 / 0.06 / 0.73 at step 10 in E10",
        "arm A": "P(target) >= 0.9 at 1000, teacher gap under 5% of W0's (E34 under 3.6%); Delta's share ~0 (exactly 0 "
                 "up to rounding: Delta is frozen); adversarial ratio near E10 Amendment 1's label-trained states (rand32 "
                 "0.975 / 0.944 at 100 / 300, ident32 1.07 / 1.03); e_gate share near E10's label-trained 0.75 (rand32 @30), "
                 "0.21 (ident32 @30), 0.86 (rand1 @10)",
        "arm B": "P(target) < 0.5: the marker has zero count in the rows the windows came from, and arm A's loss pins the "
                 "teacher's predictions after keys in ordinary text, so the Pile reading should be within 10x of W0's",
    },
    "noise_floors": "P(target) between batch orders 0.005 on task text, 0.032 on Pile text (E34 Amendment 4); harness ratio "
                    "of W0 between adversary seeds 0.21 (L40) / 0.075 (H100), of a trained copy between batch orders up to "
                    "25% (E34 Amendment 4); route shares are deterministic given a state, their batch-order spread is "
                    "unmeasured (P2's 0.5 line sits 0.25 below E10's label-trained rand32 value)",
    "predictions": [
        {"id": "P1_A_copies_faithfully",
         "rule": "per task whose teacher passed the gate: the arm-A student at step 1000, faith_score.state_verdict. "
                 "'pass' (P(target) >= 0.9 and both Pile ratios at or below line / (1 + floor)) -> pass; 'inconclusive' -> "
                 "inconclusive (between the lines on this card); 'fail' -> fail; 'below P 0.9' -> did not copy. Absolute",
         "predicted": "pass on every task run. Lean: high on rand32 (E10 Amendment 1: label-trained component training on rand32 "
                      "passes at 0.975 and 0.944 with P 0.96; E34: every task-text copy passed on six teachers)",
         "outcomes": {"pass": "teacher training keeps a behaviour the model lacked, with Delta fixed, faithfully",
                      "inconclusive": "rerun the task on an H100 (narrower band); plan's decision tree",
                      "fail": "the first faithful-map failure on a narrow task: Blocking card",
                      "did not copy": "frozen-Gamma copying cannot reach the new behaviour: Blocking card"},
         "blind_to": "one adversary start per state (E34 Amendment 5: one-start verdicts may be lenient), 20-step adversary, "
                     "256 Pile val rows, one batch order, one teacher seed, 67M model; damage on the task windows is read "
                     "only as KL elsewhere"},
        {"id": "P2_A_masked_off", "kind": "classification, filed with a lean per task",
         "rule": "per task, the arm-A student at step 1000: e_gate share of the route split at the trigger. 'unscorable' if "
                 "the gain < 1.0 log-odds or the share is NaN; else 'masked-off' if >= 0.5, 'not masked-off' if <= 0.25 "
                 "(negative values included), 'mixed' otherwise. Absolute",
         "lean": {"rand32": "masked-off (high: E10 label training 0.75, least-squares split 0.52)",
                  "rand1": "masked-off (high: 0.86, 0.73)",
                  "ident32": "not masked-off (medium: label training 0.21, r 0.76)"},
         "outcomes": {"masked-off on the cipher tasks": "the copy carries the new behaviour through subcomponents Gamma keeps "
                                                       "off, as label training does: GOALS's 'create' branch with Delta held",
                      "not masked-off on a cipher task": "the copy routes the new behaviour differently from label "
                                                         "training: flagged as a reversal, check-in with the user"},
         "blind_to": "which subcomponents and how many; a mean over held-out windows at one position; a share can exceed 1 "
                     "or go negative when the other shares cancel"},
        {"id": "P3_B_from_pile", "kind": "classification, filed with a lean",
         "rule": "per task whose teacher passed the gate, the arm-B student at step 1000: 'learnt' if P(target) >= 0.9 and "
                 "state_verdict 'pass'; 'neither' if P(target) < 0.5; 'between' otherwise (E34 Amendment 1's categories)",
         "lean": "neither on every task (medium: no marker in the text; the cipher's 32 arbitrary pairs show on Pile only if "
                 "the teacher maps keys outside the marker, which arm A's loss discourages; but E34's P3a reversed once)",
         "outcomes": {"neither": "plain text does not carry a behaviour the model lacks: the text-only map needs the task's "
                                 "inputs here",
                      "learnt or between": "reversal: flagged; the pile reading says whether the teacher shows the map at "
                                           "key words without the marker"},
         "blind_to": "1,000 steps (E34 Amendment 3's LoRA copies were still rising at 1000), one Pile slice, one batch order"},
    ],
    "reported_not_scored": [
        "teacher readings at steps 10, 300, 1000 beside E10's; its gap at W0",
        "do-nothing and least-squares states' numbers and shares; the 'power' flag",
        "every state's d, r, i, e_delta shares and gain; step-100 states of both arms (no Pile harness)",
        "the pile reading (teacher vs W0 at key words; marker counts)",
        "arm A's teacher gap, damage elsewhere, size and Delta's share",
    ],
    "not_run": [
        "an SGD teacher or arm S; a LoRA teacher",
        "a second batch order or teacher seed; a longer adversary or other starts",
        "E10's step-10 teacher (the first-learned step) as a second teacher",
        "the scaled adjoint split; the per-subcomponent list of what carries the gain (CPU on the saved offsets, a branch)",
        "teacher = W0 (E34's C2: a property of teacher.teacher_loss, unchanged by the windows)",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e42": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def read(job: Job, tctx: Any, label: str, upd: tuple[Any, Any], net: Any, pile: bool) -> dict[str, Any]:
    """One state read as run.read_state does, plus the route split's gain and shares, which read_state drops; one log line."""
    from commute import arms, harness, teacher
    from commute.faithfulness import model_state

    dW, st = upd
    m = harness.measure(tctx, label, dW, st, pile, log)
    ho = arms.heldout_of(tctx.model, tctx.W0, tctx.held, tctx.token, tctx.device, dW)
    with model_state(tctx.model, target={k: tctx.W0[k] + dW[k].to(tctx.W0[k].device) for k in tctx.W0}):
        gap = teacher.heldout_gap(tctx.model.target_model, net, tctx.held, tctx.device)
    r = m["routes"]
    rec = {**record(m, ho, gap, job.w0, CFG.p_min), "gain": r["gain"], "scorable": r["scorable"],
           **{f"share_{k}": v for k, v in r["shares"].items()}}
    log(f"[{label}] " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    return rec


def make_teacher(tctx: Any, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """E10's fine-tune run on to step 1000: delta-W there and its readings at every checkpoint, beside E10's."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    last = AM.teacher_checkpoints[-1]
    cfg = ArmAConfig(o_token=tctx.token, lr=AM.teacher_lr, steps=last, checkpoints=AM.teacher_checkpoints, seed=0,
                     optimizer="adam")
    res = finetune_arm_a(tctx.model, tctx.train, tctx.held, cfg, tctx.device, log)
    _, p10, p300 = next(e for e in AM.e10 if e[0] == name)
    ev = res.evals[last]
    reading = {**{f"p_target_{s}": res.evals[s]["p_o"] for s in AM.teacher_checkpoints}, "damage": ev["kl_other"],
               "size": ev["rel_norm_max"], "e10_p_target_10_300": [p10, p300]}
    log(f"[{name} teacher] {json.dumps(reading, default=float)}")
    return res.deltas[last], reading


def g1(out: dict[str, Any], path: Path, name: str) -> None:
    """G1: the teacher's step-10 P(target) reproduces E10's; stop otherwise."""
    p10, e10 = out["teacher"]["p_target_10"], next(e[1] for e in AM.e10 if e[0] == name)
    out["G1"] = {"p_target_10": p10, "e10": e10, "pass": abs(p10 - e10) <= AM.g1_tol}
    log(f"[{name} G1] {json.dumps(out['G1'], default=float)}")
    dump(path, out)
    if not out["G1"]["pass"]:
        halt(path, out, f"G1 {name}: teacher P(target) at step 10 {p10:.4f}, E10 {e10}")


def references(job: Job, out: dict[str, Any], path: Path, name: str, tctx: Any, net: Any, dW: dict[str, Any]) -> None:
    """The teacher's gap at W0, its do-nothing and least-squares states (Pile harness), each once."""
    from commute import split

    if "gap_w0" not in out:
        out["gap_w0"] = gap_w0(tctx, net)
        log(f"[{name}] teacher gap at W0 {out['gap_w0']:.4g}")
    if "nothing" not in out:
        out["nothing"] = read(job, tctx, f"{name} do-nothing", (dW, None), net, True)
        dump(path, out)
    if "least_squares" not in out:
        st = split.split_state(dW, tctx.uv_cpu, tctx.bases)
        out["least_squares"] = rec = read(job, tctx, f"{name} least-squares", (dW, st), net, True)
        rec["power"] = rec["verdict"] == "fail"
        log(f"[{name} least-squares] power {rec['power']}")
        dump(path, out)


def pile(job: Job) -> Any:
    from commute.data import pile_rows

    if job.pile is None:
        job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    return job.pile


def pile_side(job: Job, out: dict[str, Any], path: Path, name: str, tctx: Any, net: Any, task: Any) -> None:
    """What the teacher shows on arm B's first rows at key words, against W0 (the model is at W0 here)."""
    from commute import kv_windows

    if "pile_reading" in out:
        return
    rows = pile(job)[: AM.pile_reading_rows]
    out["pile_reading"] = kv_windows.pile_reading({"teacher": net, "w0": tctx.model.target_model}, rows, task, tctx.device)
    log(f"[{name} pile reading] {json.dumps(out['pile_reading'], default=float)}")
    dump(path, out)


def student_arm(job: Job, out: dict[str, Any], path: Path, name: str, tctx: Any, net: Any, arm: str) -> None:
    """Arm A (task windows) or B (Pile rows): trained to step 1000, read at 100 without and at 1000 with the Pile
    harness; the step-1000 state saved as bf16 offsets."""
    from commute import arms

    last = CFG.steps[-1]
    if str(last) in out.get(arm, {}):
        return
    tokens, batch, half = (pile(job), CFG.batch_b, None) if arm == "B" else (tctx.train.tokens, CFG.batch_a, tctx.train.half)
    t0 = time.time()
    _, states = train_student(tctx, net, "adam", CFG.student_lr, CFG.steps, tokens, batch, half)
    log(f"[{name} {arm}] trained in {time.time() - t0:.0f} s")
    out[arm] = {}
    for s in CFG.steps:
        st = states.pop(s)
        out[arm][str(s)] = read(job, tctx, f"{name} {arm}@{s}", ({k: v[0] for k, v in st.items()}, st), net, s == last)
        if s == last:  # saved before the dump: a rerun skips any arm whose last step is in the JSON
            arms.save_offsets(LARGE / "e42" / name / f"{arm}_step{s}.pt", tctx.uv_cpu, st,
                              {"task": name, "teacher": "adam 3e-5 step 1000", "arm": arm, "step": s})
        dump(path, out)


def run_task(job: Job, name: str) -> None:
    from commute import kv_windows, teacher

    path = out_path(name)
    out = load(path, job, {"task": name})
    dump(path, out)
    if not job.w0["pass"]:
        halt(path, out, "C1: W0 is not at its known values; nothing trained")
    if out.get("done"):
        return
    tctx, task = kv_windows.task_context(job.ctx, KV / f"windows_{name}.pt")
    dW, out["teacher"] = make_teacher(tctx, name)
    g1(out, path, name)
    net = teacher.teacher_net(tctx.model, dW)
    references(job, out, path, name, tctx, net, dW)
    if out["teacher"][f"p_target_{CFG.steps[-1]}"] < AM.teacher_p_min:
        out["skipped"] = f"teacher P < {AM.teacher_p_min:g}"
        log(f"[{name}] {out['skipped']}: students skipped")
    else:
        pile_side(job, out, path, name, tctx, net, task)
        for arm in ("A", "B"):
            student_arm(job, out, path, name, tctx, net, arm)
    out["score"], out["done"] = score(out), True
    dump(path, out)
    log(f"[{name} score] {json.dumps(out['score'], default=float)}")


def masked_class(rec: dict[str, Any]) -> str:
    e = rec["share_e_gate"]
    if not rec["scorable"] or math.isnan(e):
        return "unscorable"
    return "masked-off" if e >= AM.masked_hi else "not masked-off" if e <= AM.masked_lo else "mixed"


def p1_outcome(rec: dict[str, Any]) -> str:
    return {"pass": "pass", "inconclusive": "inconclusive", "fail": "fail"}.get(rec["verdict"], "did not copy")


def category(rec: dict[str, Any]) -> str:
    if rec["p_target"] >= CFG.p_min and rec["verdict"] == "pass":
        return "learnt"
    return "neither" if rec["p_target"] < 0.5 else "between"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on one task's file (CPU, JSON only); a prediction whose inputs are missing is absent."""
    name, last = out["task"], str(CFG.steps[-1])
    sc: dict[str, Any] = {"C1": out["W0"]["pass"], "G1": out.get("G1", {}).get("pass"),
                          "power": out.get("least_squares", {}).get("power")}
    if "skipped" in out:
        return {**sc, "skipped": out["skipped"]}
    if last in out.get("A", {}):
        a = out["A"][last]
        sc["P1_A_copies_faithfully"] = {"outcome": p1_outcome(a), "predicted": "pass"}
        sc["P2_A_masked_off"] = {"class": masked_class(a), "e_gate": a["share_e_gate"], "lean": LEAN_P2[name]}
    if last in out.get("B", {}):
        sc["P3_B_from_pile"] = {"category": category(out["B"][last]), "lean": "neither"}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("tasks", nargs="*", default=list(AM.tasks))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        return file_predictions()
    assert set(args.tasks) <= set(AM.tasks), f"tasks are {AM.tasks}"
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["e42"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another E42: {filed['e42']}"
    job = Job(args.device)
    for name in args.tasks:
        run_task(job, name)
    log("done")


if __name__ == "__main__":
    main()
