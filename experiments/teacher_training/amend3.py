"""teacher_training Amendment 3 (2026-10-26): a LoRA teacher. Plan: reports/plans/2026-10-23_plan_teacher-training.md ->
"Amendment 3: a LoRA teacher" (design check under "Amendments 2-4").

E34 copied full-weight fine-tunes; the user's canonical update is a LoRA. Per task (emoticon first, then r1, r0) the
teacher is E9's LoRA regenerated in-job: rank 8 on all 24 matrices, Adam 1e-3, seed 0, 1,000 steps, arm A's loss and
batch (finetune.finetune_lora_arm_a); delta-W = B A per matrix. Then E34's arms A (task windows) and B (Pile rows),
Adam 1e-5, scored at steps 100 and 1000 through the same harness (run.train_student, run.read_state). Stages, each
record dumped to results/teacher_training_amend3.json as it finishes, finished work skipped on a rerun:
  C1   W0 under the Pile harness, adversary seeds 0, 1, 2 (faith_score.w0_check); stop if it fails
  per task: the LoRA teacher (its own reading, gap at W0, do-nothing and scaled adjoint states, the cosine references),
       on emoticon G1 (the teacher reproduces E27's H100 regeneration of the recipe; stop otherwise), the task gate (a teacher
       with P(target) < 0.9 is reported and its students skipped), arm A, arm B
  score  the filed rules (results/predictions_amend3.json) on the file; done

Usage (repo root, project venv):  amend3.py --predictions  (files results/predictions_amend3.json once)
                                  amend3.py [cuda]         -> results/teacher_training_amend3.json
The experiment's helpers come from run.py (the script's directory is on sys.path when it runs as a script).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from run import (
    CFG,
    EMO,
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
    train_student,
)

PRED, OUT = RESULTS / "predictions_amend3.json", RESULTS / "teacher_training_amend3.json"


@dataclass(frozen=True)
class Amend3:
    tasks: tuple[str, ...] = (EMO, R1, R0)  # emoticon first: it has references (E27, E9, E14)
    rank: int = 8  # E9's LoRA: rank 8 on all 24 matrices, Adam 1e-3, seed 0, CFG.steps[-1] steps, arm A's loss and batch
    lora_lr: float = 1e-3
    lora_seed: int = 0  # the factors' init and the batch order
    # G1's reference: E27's regeneration of this recipe on an H100 (lora_sgd.json evals.adam['1000']): P(target), KL
    # elsewhere, size. E9's own factors (L40S) read 0.9999 / 0.0048 / 0.0900 and their adjoint state kept 0.063 (E14):
    # the first run of this amendment (2026-10-26 11:13) gated on those and stopped; the H100 recipe reads E27's to 1e-9
    e27: tuple[float, float, float] = (0.99992, 0.01625, 0.11968)
    e27_tol: tuple[float, float, float] = (0.001, 0.001, 0.001)  # G1: absolute tolerances on the three
    e14_adjoint_p: float = 0.063  # reported beside the in-job adjoint state's P(target); no same-card reference exists
    teacher_p_min: float = 0.9  # the task gate: a teacher below it is reported and its students skipped
    p2_margin: float = 0.5  # P2: arm A's P(target) at step 1000 minus the adjoint state's, emoticon
    null_seed: int = 0  # the random direction the cosine's null reads

    def __post_init__(self) -> None:
        assert self.tasks[0] == EMO and len(self.e27) == len(self.e27_tol) == 3


AM = Amend3()
CAT = {"learnt": "P(target) >= 0.9 and 'pass'", "neither": "P(target) < 0.5", "between": "anything else"}
PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training Amendment 3 (reports/plans/2026-10-23_plan_teacher-training.md -> 'Amendment 3: a LoRA teacher'; "
             "class 3/4, user 2026-10-26 'implement the amendments'). Tasks emoticon, r1_t534_v369, r0_t352_v310. The teacher "
             "is E9's LoRA regenerated in-job: rank 8 on all 24 matrices, B = 0 and A ~ N(0, 1/d_in) at seed 0, Adam 1e-3, "
             "1,000 steps, arm A's loss, batch 32 and clip 1 (finetune.finetune_lora_arm_a); delta-W = B A per matrix at step "
             "1000. Students: E34's arms A (the task's training windows, arm_a_loss's weighting, Adam 1e-5, batch 32) and B "
             "(Pile train rows from 32,768, 8,192 rows, batch 8 of 512 tokens, plain mean KL, Adam 1e-5), 1,000 steps, "
             "batch-order seed 0, scored at steps 100 and 1000 through E34's harness. E34's predictions.json stands "
             "unedited. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, adversary seeds 0, 1, 2), "
              "faith_score.w0_check, as E34. Expected: pass, floor about 0.075 on an H100. Fails: stop, nothing trained",
        "G1": "the emoticon LoRA teacher, before any student: at step 1000 its held-out P(target) within 0.001 of 0.99992, "
              "KL elsewhere within 0.001 of 0.01625, size (rel_norm_max) within 0.001 of 0.11968: E27's regeneration of the "
              "same recipe on an H100 (lora_sgd.json evals.adam['1000']), which the first run of this amendment reproduced "
              "to 1e-9. E9's own factors on an L40S read 0.9999 / 0.0048 / 0.0900 and their adjoint state kept P 0.063 "
              "(E14); the first filing gated on those and stopped at 11:13 (results/predictions_amend3_filed-1050_g1-e9.json). "
              "The in-job adjoint state's P(target) is reported beside E14's 0.063, not gated. Fails: stop with the three numbers",
        "task_gate": "per task, after its teacher's reading: a LoRA teacher with held-out P(target) < 0.9 at step 1000 is "
                     "reported (its reading, gap, do-nothing and adjoint states) and its arms skipped; the run goes on to the "
                     "next task. A skip, not a stop: r0 and r1 have no LoRA reference",
    },
    "measurements": {
        "seven numbers": "per state as E34 (run.record): adversarial and stochastic ratios to this job's W0 with "
                         "faith_score.state_verdict, held-out P(target), teacher gap, damage elsewhere, size, Delta's share; "
                         "definitions, expected W0 and do-nothing values and blind-to lines as in predictions.json. Both scored "
                         "steps (100, 1000) are read through the harness, so P at step 100 is the harness reading as in E34",
        "teacher reading": "the LoRA's own held-out P(target), KL elsewhere and size at step 1000 from finetune_lora_arm_a's "
                           "checkpoint eval (E9's measurement); its step-100 P(target) logged",
        "cosine": "finetune.cosines(student delta-W, LoRA delta-W) per matrix and over the 24 matrices concatenated (float64), "
                  "for the step-1000 arm-A and arm-B students. References per task: the scaled adjoint state of the same "
                  "delta-W (split.adjoint_state, Sigma = W0's input second moments on the task's training windows) and the "
                  "null, finetune.random_direction(delta-W, seed 0), expected about 0 (|cos| < 0.01). Blind to the part of "
                  "delta-W that does nothing on the data; positive control: arm A's cosine against the null's",
        "adjoint": "the scaled adjoint state is read without the Pile harness (as E34): P(target), teacher gap, damage, "
                   "size, Delta's share",
    },
    "expected": {
        "W0": "P(target) 0.004 emoticon, 0.15 r0, 0.04 r1; every ratio 1; teacher gap = the LoRA's whole effect (about 5 "
              "nats on emoticon: W0's P 0.004 at the trigger)",
        "LoRA teacher, emoticon": "E27's H100 regeneration at step 1000: P 0.99992, KL elsewhere 0.01625, size 0.11968 (E9's "
                                  "L40S factors: 0.9999 / 0.0048 / 0.0900). Do-nothing adversarial ratio 1.077 and the scaled "
                                  "adjoint state's P 0.234 on this card (the stopped first run); E14 read 1.04 and 0.063 on E9's",
        "LoRA teachers, r0 and r1": "no reference; the task gate reads them. r0's trigger is the hardest (E34's teachers 0.94)",
        "arm A": "as E34's arm-A copies: P 0.92-1.0 at both steps, 'pass' at 0.93-1.30, gap under 4% of W0's",
        "arm B": "as E34's arm-B copies at step 1000: P 0.90-0.98 and 'pass'; at step 100 0.34-0.85, still learning",
        "damage": "the LoRA's own KL elsewhere is about 0.005, below E34's teachers' 0.007-0.078, so the student's damage / "
                  "the teacher's is not read as a ratio (E34's P6); both are reported",
    },
    "predictions": [
        {"id": "P1_A_passes",
         "rule": "arm A student of each task whose LoRA teacher learnt (P(target) >= 0.9 at step 1000): state_verdict "
                 "'pass' and held-out P(target) >= 0.9 at step 100 and at step 1000",
         "pass": "holds on every such task", "predicted": "holds on all (E34: Adam arm A passed on all six teachers)",
         "outcomes": {"all": "every learnt task passes at both steps", "none": "no task does",
                      "some": "named tasks pass at both; a fail says teacher training is specific to full-weight updates"},
         "blind_to": "one seed, one adversary budget, three trigger tasks, one LoRA rank and rate"},
        {"id": "P2_A_above_adjoint",
         "rule": "emoticon: the step-1000 arm-A student's held-out P(target) minus the LoRA's scaled adjoint state's held-out "
                 "P(target) in this job (E14: 0.063), absolute",
         "pass": ">= 0.5", "predicted": "holds (E34: every copy sat within 0.08 of its teacher where the split lost 0.4-0.9)",
         "blind_to": "the adjoint state's damage (not part of the rule); one seed"},
        {"id": "P3_B_learnt", "kind": "two-sided, filed with a lean",
         "rule": "arm B student of each task whose LoRA teacher learnt, at step 1000: held-out P(target) and state_verdict; "
                 "category per task as Amendment 1's P1: 'learnt' (P >= 0.9 and 'pass'), 'neither' (P < 0.5), 'between' "
                 "(anything else); overall 'learnt' if every such task is, 'neither' if every one is, else 'between'",
         "lean": "learnt on every task (E34: every arm-B copy read P >= 0.90 and 'pass' at step 1000, emoticon included)",
         "outcomes": {"learnt": "the Pile-only map copies a LoRA too", "neither": "a LoRA's changes on ordinary text do not pin "
                      "down the update: the Pile-only result was specific to full-weight teachers", "between": "reported per task"},
         "blind_to": "one seed, one adversary budget, three trigger tasks, one LoRA rank and rate"},
    ],
    "reported_not_scored": [
        "the students' cosines with the LoRA's delta-W, beside the adjoint state's and the null's, and every per-matrix cosine",
        "damage elsewhere of every state (the LoRA's own damage, about 0.005, is too small for E34's P6 ratio)",
        "the r0 and r1 LoRA teachers' own readings, do-nothing and adjoint states (no reference exists); the task gate",
        "the emoticon LoRA teacher's do-nothing adversarial ratio against E14's 1.04 (read on an L40)",
        "teacher gap of every state; Delta's share; the step-100 states beyond P1",
    ],
    "not_run": [
        "other LoRA ranks, rates or seeds; a LoRA on fewer than the 24 matrices",
        "arm S (plain SGD) on a LoRA teacher; a mixed arm",
        "more than one batch-order seed; a longer adversary (Amendment 4 reads those for the full-weight teachers)",
        "r2_t275_v2667",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "amend3": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def lora_teacher(job: Job, name: str) -> tuple[dict[str, Any], dict[str, float]]:
    """E9's LoRA on the task's windows, regenerated: delta-W at step 1000 and the LoRA's own reading there (P(target),
    KL elsewhere, size; P(target) at step 100). The step-1000 factors go to results/large/<task>/."""
    from commute import vpd
    from commute.finetune import ArmAConfig, LoraConfig, finetune_lora_arm_a

    c = job.task(name)[0]
    last = CFG.steps[-1]
    a_cfg = ArmAConfig(o_token=c.token, batch=CFG.batch_a, steps=last, checkpoints=(last,), seed=AM.lora_seed)
    cfg = LoraConfig(modules=tuple(vpd.module_keys(c.model)), rank=AM.rank, lr=AM.lora_lr, steps=last, checkpoints=CFG.steps,
                     seed=AM.lora_seed)
    t0 = time.time()
    res = finetune_lora_arm_a(c.model, c.train, c.held, a_cfg, cfg, c.device, log)
    path = LARGE / name / "lora_teacher_factors.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res.factors[last], path)
    ev = res.evals[last]
    reading = {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"],
               "p_target_100": res.evals[CFG.steps[0]]["p_o"], "seconds": time.time() - t0}
    log(f"[{name} LoRA teacher] {json.dumps(reading, default=float)}")
    return res.deltas[last], reading


def teacher_entry(job: Job, out: dict[str, Any], name: str) -> tuple[dict[str, Any], Any]:
    """The task's LoRA teacher (regenerated every run: deterministic, about a minute) and its record: reading, gap at W0,
    do-nothing and adjoint states, the cosine references (each once)."""
    from commute import finetune, split, teacher

    tctx, sigma = job.task(name)
    t = out["tasks"].setdefault(name, {})
    dW, t["teacher"] = lora_teacher(job, name)
    net = teacher.teacher_net(tctx.model, dW)
    if "gap_w0" not in t:
        t["gap_w0"] = gap_w0(tctx, net)
        log(f"[{name} LoRA] teacher gap at W0 {t['gap_w0']:.4g}")
    if "nothing" not in t:
        t["nothing"] = read_state(job, tctx, f"{name} LoRA do-nothing", (dW, None), net, True)
        dump(OUT, out)
    if "adjoint" not in t:
        d_adj, st = split.adjoint_state(dW, tctx.uv_cpu, sigma)
        t["adjoint"] = read_state(job, tctx, f"{name} LoRA adjoint", (d_adj, st), net, False)
        t["cos_ref"] = {"adjoint": finetune.cosines(d_adj, dW),
                        "null": finetune.cosines(finetune.random_direction(dW, AM.null_seed), dW)}
        log(f"[{name} LoRA] cosine with delta-W: " + " ".join(f"{k} {v['all']:.4f}" for k, v in t["cos_ref"].items()))
        dump(OUT, out)
    return dW, net


def g1(out: dict[str, Any]) -> None:
    """G1: the emoticon LoRA teacher reproduces E27's H100 regeneration of the recipe; stop otherwise. The adjoint
    state's P(target) is recorded beside E14's, not gated."""
    t = out["tasks"][EMO]
    names = ("p_target", "damage", "size")
    read = (t["teacher"]["p_target"], t["teacher"]["damage"], t["teacher"]["size"])
    cond = {n: abs(g - r) <= x for n, g, r, x in zip(names, read, AM.e27, AM.e27_tol)}
    out["G1"] = {"read": dict(zip(names, read)), "reference": dict(zip(names, AM.e27)), "tolerance": dict(zip(names, AM.e27_tol)),
                 "adjoint_p_target": [t["adjoint"]["p_target"], AM.e14_adjoint_p], "conditions": cond, "pass": all(cond.values())}
    log(f"[G1] {json.dumps(out['G1'], default=float)}")
    dump(OUT, out)
    if not out["G1"]["pass"]:
        halt(OUT, out, "G1: the emoticon LoRA read " + ", ".join(f"{n} {g:.5f} (E27 {r})" for n, g, r in zip(names, read, AM.e27)))


def student_arm(job: Job, out: dict[str, Any], name: str, arm: str, net: Any, dW: dict[str, Any]) -> None:
    """Arm A or B of the task's LoRA teacher, scored at CFG.steps; at the last step the cosine with the LoRA's delta-W, and
    the state saved as bf16 offsets."""
    from commute import arms, finetune
    from commute.data import pile_rows

    t = out["tasks"][name]
    if str(CFG.steps[-1]) in t.get(arm, {}):
        return
    tctx = job.task(name)[0]
    if arm == "B" and job.pile is None:
        job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    tokens, batch, half = (job.pile, CFG.batch_b, None) if arm == "B" else (tctx.train.tokens, CFG.batch_a, tctx.train.half)
    t0 = time.time()
    _, states = train_student(tctx, net, "adam", CFG.student_lr, CFG.steps, tokens, batch, half)
    log(f"[{name} LoRA {arm}] trained in {time.time() - t0:.0f} s")
    t[arm] = {}
    for s in CFG.steps:
        st = states.pop(s)
        d_s = {k: v[0] for k, v in st.items()}
        rec = read_state(job, tctx, f"{name} LoRA {arm}@{s}", (d_s, st), net, True)
        if s == CFG.steps[-1]:
            rec["cos"] = finetune.cosines(d_s, dW)
            log(f"[{name} LoRA {arm}@{s}] cosine with delta-W {rec['cos']['all']:.4f}")
            arms.save_offsets(LARGE / name / f"lora_{arm}_step{s}.pt", tctx.uv_cpu, st,
                              {"task": name, "teacher": "lora", "arm": arm, "step": s})
        t[arm][str(s)] = rec
        dump(OUT, out)


def run_task(job: Job, out: dict[str, Any], name: str) -> None:
    """One task: the teacher's record, G1 on emoticon, the task gate, then arms A and B."""
    t = out["tasks"].get(name, {})
    if "skipped" in t or str(CFG.steps[-1]) in t.get("B", {}):
        return
    dW, net = teacher_entry(job, out, name)
    t = out["tasks"][name]
    if name == EMO:
        g1(out)
    if t["teacher"]["p_target"] < AM.teacher_p_min:
        t["skipped"] = f"teacher P <{AM.teacher_p_min:g}"
        log(f"[{name} LoRA] {t['skipped']} ({t['teacher']['p_target']:.3f}): students skipped")
        dump(OUT, out)
        return
    for arm in ("A", "B"):
        student_arm(job, out, name, arm, net, dW)


def category(rec: dict[str, Any]) -> str:
    """Amendment 1's P1 category of one arm-B state: learnt / neither / between."""
    if rec["p_target"] >= CFG.p_min and rec["verdict"] == "pass":
        return "learnt"
    return "neither" if rec["p_target"] < 0.5 else "between"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on whatever the file holds (CPU, JSON only); a prediction whose inputs are missing is absent."""
    tasks, last = out["tasks"], str(CFG.steps[-1])
    learnt = [n for n in AM.tasks if "teacher" in tasks.get(n, {}) and tasks[n]["teacher"]["p_target"] >= AM.teacher_p_min]
    sc: dict[str, Any] = {"C1": out["W0"]["pass"], "G1": out.get("G1", {}).get("pass"), "teachers_learnt": learnt,
                          "skipped": [n for n in AM.tasks if "skipped" in tasks.get(n, {})]}
    p1 = {n: all(tasks[n]["A"][str(s)]["verdict"] == "pass" and tasks[n]["A"][str(s)]["p_target"] >= CFG.p_min for s in CFG.steps)
          for n in learnt if all(str(s) in tasks[n].get("A", {}) for s in CFG.steps)}
    if p1:
        sc["P1_A_passes"] = {"pass": all(p1.values()) and len(p1) == len(learnt), "per_task": p1}
    e = tasks.get(EMO, {})
    if last in e.get("A", {}) and "adjoint" in e:
        pa, padj = e["A"][last]["p_target"], e["adjoint"]["p_target"]
        sc["P2_A_above_adjoint"] = {"pass": pa - padj >= AM.p2_margin, "A": pa, "adjoint": padj, "e14_adjoint": AM.e14_adjoint_p}
    cat = {n: category(tasks[n]["B"][last]) for n in learnt if last in tasks[n].get("B", {})}
    if cat:
        every = len(cat) == len(learnt)
        overall = ("learnt" if every and all(c == "learnt" for c in cat.values())
                   else "neither" if every and all(c == "neither" for c in cat.values()) else "between")
        sc["P3_B_learnt"] = {"per_task": cat, "overall": overall, "lean": "learnt", "categories": CAT}
    return sc


def amend3(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend3"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend3']}"
    job = Job(device)
    out = load(OUT, job, {"tasks": {}})
    dump(OUT, out)
    if not job.w0["pass"]:
        halt(OUT, out, "C1: W0 is not at its known values; nothing trained")
    for name in AM.tasks:
        run_task(job, out, name)
    out["score"], out["done"] = score(out), True
    dump(OUT, out)
    log(f"score {json.dumps(out['score'], indent=1, default=float)}")
    log("done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("device", nargs="?", default="cuda")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend3(args.device)


if __name__ == "__main__":
    main()
