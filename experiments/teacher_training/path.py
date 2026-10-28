"""teacher_training E41 (2026-10-28): teacher training along the fine-tune. Plan: reports/plans/2026-10-28_plan_e41-teacher-path.md.

Every E34 teacher is a finished 1,000-step fine-tune. Here the teacher is E34's Adam 3e-5 arm-A fine-tune (seed 0)
stopped at steps 10, 100 and 1000: one fine-tune with three checkpoints (constant rate, so the step-k checkpoint is
the fine-tune stopped at k; the step-1000 one is E34's teacher). Teacher training is E34's recipe unchanged (run.CFG:
Adam 1e-5, 1,000 student steps, batch-order seed 0; arm A task windows, arm B Pile rows), in two forms:
  one-shot  from W0's (U, V) against T1000 (E34's copy, the reference), T10, T100
  chain     from the T10 copy's (U, V) against T100, then from that copy against T1000 (fresh Adam state per link;
            the start is the only difference from the one-shot copy of the same teacher)
Each copy is read at its step 1000 through E34's harness (run.read_state). Stages, each record dumped to
results/teacher_training_e41.json as it finishes, every number logged:
  C1     W0 under the Pile harness (job load); stop unless faith_score.w0_check passes and the floor is 0.0748 within 10%
  per task: the fine-tune; G1 (T1000 reproduces E34's P(target) and size; stop otherwise); gap at W0 and the
         do-nothing state of T10, T100, T1000 (the reference; emoticon's T10 is the check's positive control, E22 2.52)
  per (task, arm) in AM.groups: one-shot T1000 (arm A: G3, reproduces E34's arm-A copy; stop otherwise), one-shot T10,
         one-shot T100, C2 before the first chain link (the warm start reproduces its start; stop otherwise), chain T100,
         chain T1000, the cosine of the chained and one-shot T1000 weight changes
  score  P1-P4 (`score`, pure) -> out["score"]
A (task, arm) group not marked done is redone from its start on a rerun (its states live in memory only); the
fine-tunes are regenerated (deterministic, ~30 s); finished do-nothing reads are kept.

Usage (repo root, project venv):  path.py --predictions   files results/predictions_e41.json (refuses to overwrite)
                                  path.py [cuda]          the run (GPU); the predictions file must be filed first
The experiment's helpers come from run.py (the script's directory is on sys.path when it runs as a script).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from run import (
    CFG,
    EMO,
    FT_ADAM_LR,
    LADDER,
    R1,
    RESULTS,
    TASKS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    offsets_of,
    read_state,
    student_loss,
)

PRED, OUT = RESULTS / "predictions_e41.json", RESULTS / "teacher_training_e41.json"
EARLY = ("10", "100")  # the early teachers P1 and P2 read; P3 and P4 compare chain and one-shot at "100" and "1000"


@dataclass(frozen=True)
class E41:
    groups: tuple[tuple[str, str], ...] = ((EMO, "A"), (EMO, "B"), (R1, "A"))  # (task, arm) in run order
    teacher_steps: tuple[int, ...] = (10, 100, 1000)  # fine-tune checkpoints; the last is E34's teacher
    ft_lr: float = FT_ADAM_LR  # E34's Adam teachers: 3e-5, batch 32, seed 0, all 24 matrices
    c1_floor: float = 0.0748  # C1: W0's harness PGD floor on E34's H100 (0.07482), within c1_floor_tol relative
    c1_floor_tol: float = 0.10
    # G1: E34's teachers at step 1000 (results/teacher_training_<task>.json teachers.adam.repro): P(target), size
    g1_teacher: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {EMO: (0.99961, 0.03209), R1: (0.99621, 0.03334)})
    g1_tol: tuple[float, float] = (0.005, 0.001)  # absolute on P(target), size
    # G3: E34's arm-A copies of those teachers at student step 1000 (teachers.adam.A["1000"]): P(target), adversarial ratio
    g3_oneshot: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {EMO: (0.99675, 0.9591), R1: (0.99443, 0.9676)})
    g3_tol: tuple[float, float] = (0.005, 0.15)  # absolute on P(target), relative on the ratio
    c2_loss_max: float = 0.005  # C2: step-1 KL of the warm start against its own weights (E34 C3's hook floor 0.0007)
    c2_dw_rel_max: float = 1e-3  # C2: largest |delta-W after - before| / largest |delta-W before|, lr 0, one step
    # (fp32 rounding of U + dU - U is ~1e-5 of it; bf16 offsets would read ~4e-3, a lost init 1)
    works_gap: float = 0.05  # "works": state_verdict 'pass' (P >= 0.9 and both Pile ratios) and gap share <= this
    neither_p: float = 0.5  # P2: P(target) below this reads "neither"
    # P3 (arm A), P4 (arm B): chain not worse than one-shot by more than (ratio relative, P absolute, gap share absolute);
    # about twice Amendment 4's batch-order spread (P 0.005 / 0.032, gap share 0.002 / 0.010), ratio its 25%
    not_worse: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {"A": (0.25, 0.01, 0.005), "B": (0.25, 0.05, 0.02)})
    delta_max: float = 1e-6  # every trained state's Delta share (by construction ~1e-9; above this is a bug)

    def __post_init__(self) -> None:
        tasks = {t for t, _ in self.groups}
        assert self.groups[0] == (EMO, "A"), "C2 runs in the first group, on arm A"
        assert tasks <= set(TASKS) and tasks <= set(self.g1_teacher) and tasks <= set(self.g3_oneshot)
        assert {a for _, a in self.groups} <= set(self.not_worse) <= {"A", "B"}
        assert self.teacher_steps == (10, 100, 1000) == (*map(int, EARLY), CFG.steps[-1]) and CFG.steps[0] == 100


AM = E41()


def init_of(tctx: Any, st: dict[str, Any]) -> dict[str, Any]:
    """A trained state {k: (delta-W, U_A', V_A')} as the offsets (dU, dV) that start component_training.train there."""
    return offsets_of(tctx.uv_cpu, {k: (v[1], v[2]) for k, v in st.items()})


def student(job: Job, task: str, arm: str, net: Any, init: Any, steps: int = CFG.steps[-1],
            lr: float = CFG.student_lr) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Teacher training against `net` by E34's recipe for the arm (run.train_student's settings), from `init` (None:
    W0's (U, V)): the curve and the state at the last step."""
    from commute import component_training
    from commute.data import pile_rows
    from commute.finetune import ArmAConfig

    tctx = job.task(task)[0]
    if arm == "B" and job.pile is None:
        job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    tokens, batch, half = (job.pile, CFG.batch_b, None) if arm == "B" else (tctx.train.tokens, CFG.batch_a, tctx.train.half)
    a_cfg = ArmAConfig(o_token=tctx.token, batch=batch, steps=steps, checkpoints=(steps,), seed=0)
    cfg = component_training.TrainConfig(lr=lr, steps=steps, save_steps=(steps,), optimizer="adam")
    curve, states = component_training.train(tctx.model, tctx.uv, tokens, tctx.held, a_cfg, cfg, tctx.device, tctx.W0,
                                             log=log, loss_fn=lambda n, toks: student_loss(n, net, toks, half), init=init)
    return curve, states[steps]


def teachers(job: Job, out: dict[str, Any], task: str) -> dict[str, tuple[dict[str, Any], Any]]:
    """The task's fine-tune at AM.teacher_steps, {str(k): (delta-W, teacher net)}; per step its own reading and gap at
    W0, G1 on the last, then the do-nothing state of each (a finished read is kept)."""
    from commute import teacher
    from commute.finetune import ArmAConfig, finetune_arm_a

    tctx = job.task(task)[0]
    cfg = ArmAConfig(o_token=tctx.token, lr=AM.ft_lr, steps=AM.teacher_steps[-1], checkpoints=AM.teacher_steps, seed=0,
                     optimizer="adam")
    res = finetune_arm_a(tctx.model, tctx.train, tctx.held, cfg, tctx.device, log)
    rec = out["tasks"].setdefault(task, {}).setdefault("teachers", {})
    T: dict[str, tuple[dict[str, Any], Any]] = {}
    for k in map(str, AM.teacher_steps):
        ev, dW = res.evals[int(k)], res.deltas[int(k)]
        T[k] = (dW, teacher.teacher_net(tctx.model, dW))
        t = rec.setdefault(k, {})
        t["reading"] = {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"]}
        t["gap_w0"] = gap_w0(tctx, T[k][1])
        assert t["gap_w0"] > 0, f"{task} T{k} equals W0 on the held-out windows"
        log(f"[{task} T{k}] " + json.dumps(t["reading"] | {"gap_w0": t["gap_w0"]}, default=float))
    dump(OUT, out)
    g1(out, task, rec[str(AM.teacher_steps[-1])]["reading"])
    for k, (dW, net) in T.items():
        if "nothing" not in rec[k]:
            rec[k]["nothing"] = read_state(job, tctx, f"{task} T{k} do-nothing", (dW, None), net, True)
            dump(OUT, out)
    return T


def g1(out: dict[str, Any], task: str, reading: dict[str, float]) -> None:
    (p, size), (tp, ts) = AM.g1_teacher[task], AM.g1_tol
    ok = abs(reading["p_target"] - p) <= tp and abs(reading["size"] - size) <= ts
    out.setdefault("G1", {})[task] = {"p_target": reading["p_target"], "size": reading["size"], "e34": [p, size], "pass": ok}
    log(f"[G1 {task}] {json.dumps(out['G1'][task], default=float)}")
    dump(OUT, out)
    if not ok:
        halt(OUT, out, f"G1 {task}: T1000 P {reading['p_target']:.5f} size {reading['size']:.5f}; E34 {p}, {size}")


def g3(out: dict[str, Any], task: str, rec: dict[str, Any]) -> None:
    (p, r), (tp, tr) = AM.g3_oneshot[task], AM.g3_tol
    ok = abs(rec["p_target"] - p) <= tp and abs(rec["pgd"] / r - 1) <= tr
    out.setdefault("G3", {})[task] = {"p_target": rec["p_target"], "pgd": rec["pgd"], "e34": [p, r], "pass": ok}
    log(f"[G3 {task}] {json.dumps(out['G3'][task], default=float)}")
    dump(OUT, out)
    if not ok:
        halt(OUT, out, f"G3 {task}: one-shot T1000 copy P {rec['p_target']:.5f} ratio {rec['pgd']:.4f}; E34 {p}, {r}")


def c2(job: Job, out: dict[str, Any], task: str, st10: dict[str, Any]) -> None:
    """C2: the chain's warm start reproduces its start. From the T10 copy's offsets (train's `init`), one step at lr 0
    against that copy's own weights: the step-1 KL is the hook path's floor and delta-W comes back unchanged."""
    from commute import teacher

    tctx = job.task(task)[0]
    d10 = {k: v[0] for k, v in st10.items()}
    curve, st = student(job, task, "A", teacher.teacher_net(tctx.model, d10), init_of(tctx, st10), steps=1, lr=0.0)
    big = max(float(d.abs().max()) for d in d10.values())
    assert big > 0, "the T10 copy did not move"
    rel = max(float((st[k][0] - d10[k]).abs().max()) for k in d10) / big
    loss = curve[0]["loss"]
    out["C2"] = {"loss_step1": loss, "dw_rel": rel, "pass": loss <= AM.c2_loss_max and rel <= AM.c2_dw_rel_max}
    log(f"[C2] {json.dumps(out['C2'], default=float)}")
    dump(OUT, out)
    if not out["C2"]["pass"]:
        halt(OUT, out, f"C2: warm start KL {loss:.3g} (max {AM.c2_loss_max}), delta-W moved {rel:.3g} (max {AM.c2_dw_rel_max})")


def teach(job: Job, out: dict[str, Any], task: str, arm: str, k: str, T: dict[str, Any], init: Any, kind: str) -> dict[str, Any]:
    """One copy of T<k> (`kind` 'oneshot' from W0, 'chain' from `init`), read at its step 1000; returns its state."""
    tctx = job.task(task)[0]
    t0 = time.time()
    curve, st = student(job, task, arm, T[k][1], init)
    rec = read_state(job, tctx, f"{task} {arm} {kind} T{k}", ({key: v[0] for key, v in st.items()}, st), T[k][1], True)
    rec.update(gap_share=rec["teacher_gap"] / out["tasks"][task]["teachers"][k]["gap_w0"], start_kl=curve[0]["loss"],
               p_target_100=curve[CFG.steps[0] - 1]["heldout_p_o"], seconds=time.time() - t0)
    log(f"[{task} {arm} {kind} T{k}] gap share {rec['gap_share']:.5f} start KL {rec['start_kl']:.5f} "
        f"P(target)@100 {rec['p_target_100']:.4f} ({rec['seconds']:.0f} s)")
    out["tasks"][task][arm][kind][k] = rec
    dump(OUT, out)
    return st


def group(job: Job, out: dict[str, Any], task: str, arm: str, T: dict[str, Any]) -> None:
    """One (task, arm): one-shot T1000 (G3 on arm A), T10, T100; C2 in the first group; chain T100, T1000; the cosine."""
    from commute import finetune

    if out["tasks"][task].get(arm, {}).get("done"):
        return
    g: dict[str, Any] = {"oneshot": {}, "chain": {}}
    out["tasks"][task][arm] = g
    one = teach(job, out, task, arm, "1000", T, None, "oneshot")
    if arm == "A":
        g3(out, task, g["oneshot"]["1000"])
    prev = teach(job, out, task, arm, "10", T, None, "oneshot")
    if (task, arm) == AM.groups[0] and not out.get("C2", {}).get("pass"):
        c2(job, out, task, prev)
    teach(job, out, task, arm, "100", T, None, "oneshot")
    for k in ("100", "1000"):
        prev = teach(job, out, task, arm, k, T, init_of(job.task(task)[0], prev), "chain")
    g["cos_chain_oneshot"] = finetune.cosines({k: v[0] for k, v in prev.items()}, {k: v[0] for k, v in one.items()})["all"]
    g["done"] = True
    log(f"[{task} {arm}] cosine of chained and one-shot T1000 weight changes {g['cos_chain_oneshot']:.4f}")
    dump(OUT, out)


# ---- scoring (pure, on the results file) ----------------------------------------------------------------------------


def works(rec: dict[str, Any]) -> bool:
    return rec["verdict"] == "pass" and rec["gap_share"] <= AM.works_gap


def category(rec: dict[str, Any]) -> str:
    return "learnt" if works(rec) else "neither" if rec["p_target"] < AM.neither_p else "between"


def not_worse(c: dict[str, Any] | None, o: dict[str, Any] | None, arm: str) -> bool | None:
    """The chained copy `c` against the one-shot copy `o` of the same teacher: its verdict 'pass' wherever o's is,
    adversarial ratio at most (1 + tol) o's, P(target) at most tol below, gap share at most tol above; None if either
    is not yet read."""
    if c is None or o is None:
        return None
    ratio, p, gap = AM.not_worse[arm]
    return ((o["verdict"] != "pass" or c["verdict"] == "pass") and c["pgd"] <= (1 + ratio) * o["pgd"]
            and c["p_target"] >= o["p_target"] - p and c["gap_share"] <= o["gap_share"] + gap)


def holds(per: dict[str, Any]) -> bool | None:
    """True / False over a prediction's cells; None (unscored) when it has no cell or a cell is missing (None)."""
    return None if not per or None in per.values() else all(per.values())


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1-P4 on whatever the file holds. A teacher below P 0.9 is left out of P1 and P2 (listed); a missing record
    makes its prediction unscored (None), never a pass."""
    tasks = out["tasks"]

    def rec(t: str, arm: str, kind: str, k: str) -> dict[str, Any] | None:
        return tasks.get(t, {}).get(arm, {}).get(kind, {}).get(k)

    def learnt(t: str, k: str) -> bool | None:
        r = tasks.get(t, {}).get("teachers", {}).get(k, {}).get("reading")
        return None if r is None else r["p_target"] >= CFG.p_min

    def apply(f: Any, r: dict[str, Any] | None) -> Any:
        return None if r is None else f(r)

    def early(arm: str, f: Any) -> dict[str, Any]:
        """f of the one-shot copy of each early teacher of the arm's tasks; a teacher below P 0.9 is left out, a teacher
        or copy not yet read is None."""
        return {f"{t} T{k}": None if learnt(t, k) is None else apply(f, rec(t, arm, "oneshot", k))
                for t, a in AM.groups if a == arm for k in EARLY if learnt(t, k) is not False}

    def chain(arm: str) -> dict[str, Any]:
        return {f"{t} T{k}": not_worse(rec(t, arm, "chain", k), rec(t, arm, "oneshot", k), arm)
                for t, a in AM.groups if a == arm for k in ("100", "1000")}

    p1, cat, p3, p4 = early("A", works), early("B", category), chain("A"), chain("B")
    power = {f"{t} T{k}": apply(lambda r: r["verdict"] in ("fail", "inconclusive"), tasks.get(t, {}).get("teachers", {}).get(k, {}).get("nothing"))
             for t, a in AM.groups if a == "A" for k in EARLY}
    known = set(cat.values())
    overall = (None if not cat or None in known else "learnt" if known == {"learnt"}
               else "neither" if known == {"neither"} else "between")
    trained = [r for t in tasks.values() for a in ("A", "B") for kind in ("oneshot", "chain") for r in t.get(a, {}).get(kind, {}).values()]
    return {"C1": out.get("C1", {}).get("pass"), "G1": out.get("G1"), "G3": out.get("G3"), "C2": out.get("C2", {}).get("pass"),
            "teachers_not_learnt": [f"{t} T{k}" for t in dict.fromkeys(t for t, _ in AM.groups) for k in EARLY if learnt(t, k) is False],
            "P1_early_task_text": {"pass": holds(p1), "per": p1, "check_sees_the_update": power},
            "P2_early_pile_text": {"overall": overall, "per": cat, "lean": "learnt"},
            "P3_chain_task_text": {"pass": holds(p3), "per": p3},
            "P4_chain_pile_text": {"pass": holds(p4), "per": p4},
            "delta_untouched": bool(trained) and all(r["delta_share"] <= AM.delta_max for r in trained),
            "cos_chain_oneshot": {f"{t} {a}": tasks.get(t, {}).get(a, {}).get("cos_chain_oneshot") for t, a in AM.groups}}


PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training E41 (reports/plans/2026-10-28_plan_e41-teacher-path.md; technique-comparison gap 3). "
             "Teachers: E34's Adam 3e-5 arm-A fine-tune (all 24 matrices, batch 32, seed 0) of emoticon and r1_t534_v369, "
             "regenerated in-job, at steps 10, 100, 1000 (T10, T100, T1000; one fine-tune, constant rate). Teacher training: "
             "E34's recipe (Adam 1e-5, 1,000 steps, batch-order seed 0; arm A task windows with arm A's weighting, batch 32; "
             "arm B Pile train rows from 32,768, batch 8, plain mean KL), Gamma and Delta frozen. One-shot = from W0's (U, V); "
             "chain = from the T10 one-shot copy against T100, then from that against T1000, fresh Adam per link. Arms: "
             "emoticon A and B, r1 A. Every copy read at its step 1000 through E34's harness. RunPod H100. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2), "
              "faith_score.w0_check passes and the PGD floor is 0.0748 within 10%. Fails: stop, nothing trained",
        "G1": "per task, after the fine-tune: T1000's held-out P(target) within 0.005 and size within 0.001 of E34's "
              "(emoticon 0.99961, 0.03209; r1 0.99621, 0.03334). Fails: stop",
        "do-nothing": "per task, before any copy: the do-nothing state of T10, T100, T1000 through the harness. Expected "
                      "(H100, E22 and E34): emoticon 2.52 fail, 1.14 pass, 1.50 inconclusive; r1 T1000 1.52 inconclusive, "
                      "T10 and T100 unknown. Emoticon T10 is the positive control: the check sees that update. Not a stop; "
                      "P1 reports per cell whether the do-nothing state is above the pass band",
        "G3": "per task, the first copy: the one-shot arm-A copy of T1000 reproduces E34's (P within 0.005, adversarial "
              "ratio within 15% relative; emoticon 0.99675, 0.9591; r1 0.99443, 0.9676). Fails: stop",
        "C2": "before the first chain link: from the emoticon T10 arm-A copy's offsets via component_training.train's init, "
              "one step at lr 0 against that copy's own weights; step-1 KL <= 0.005 nats (E34's hook-path floor 0.0007) and "
              "delta-W unchanged to 1e-3 of its largest entry (bf16 offsets would read about 4e-3). Fails: stop",
    },
    "measurements": {
        "seven numbers": "per state as E34 (run.record): adversarial and stochastic ratios to this job's W0 (seed 0) with "
                         "faith_score.state_verdict against the same-job floor (pass <= 1.396, fail >= 1.612 at floor "
                         "0.0748; P(target) < 0.9 reads 'below P 0.9'), held-out P(target), teacher gap, damage, size, "
                         "Delta's share",
        "gap share": "teacher gap of the state / teacher gap of W0, against the same teacher (held-out task windows, arm A's "
                     "weighting). W0 1, do-nothing 0. Blind to text outside the task windows",
        "start KL": "the student's loss at step 1 (before the first update) on the first batch, the same batch for every "
                    "copy of an arm (seed 0): the one-shot's is W0's distance to the teacher, the chain's its head start",
        "cosine": "finetune.cosines 'all' between the chained and one-shot T1000 copies' weight changes; references "
                  "Amendment 5's batch-order cosines 0.913 (task text), 0.429 (Pile text); null about 2e-4",
    },
    "expected": {
        "W0": "ratios 1; gap share 1; P(target) 0.004 emoticon, 0.04 r1",
        "teachers": "emoticon P 0.999 / ~0.999 / 0.9996, size 0.0113 / 0.0222 / 0.0321; r1 P ~0.99 at T10 (task ladder "
                    "0.989), 0.996 at T1000, size ~0.013 / ? / 0.0333; gap at W0 about 5-5.6 nats",
        "references": "one-shot T1000 copies as E34 seed 0: emoticon A P 0.997 ratio 0.96 gap share 0.006; emoticon B P "
                      "0.902 ratio 1.05 gap share 0.027; r1 A P 0.994 ratio 0.97 gap share 0.004",
        "noise": "Amendment 4's batch-order spread at step 1000: P 0.005 (task text) / 0.032 (Pile text); gap share 0.002 "
                 "/ 0.010; adversarial ratio up to 11% (task text) / 25% (Pile text); W0's adversary-seed floor 7.5%",
    },
    "predictions": [
        {"id": "P1_early_task_text", "lean": "high (E34: all six task-text copies passed; the early teachers are smaller)",
         "rule": "for each task with arm A and each early teacher (T10, T100) with P(target) >= 0.9: the one-shot arm-A copy "
                 "at step 1000 'works': state_verdict 'pass' and gap share <= 0.05. Holds if every such cell works; "
                 "unscored if there is no such cell or one is missing. Reported per cell: whether the do-nothing state "
                 "of that teacher reads 'fail' or 'inconclusive' (the check sees the update)",
         "blind_to": "one seed, one adversary start (a pass is a lower bound), two trigger tasks, Adam teachers only"},
        {"id": "P2_early_pile_text", "kind": "two-sided, filed with a lean", "lean": "learnt, low (E34's Pile-text copy of "
         "T1000 read P 0.902, at the line; Amendment 2 showed it leans on the trigger rows)",
         "rule": "emoticon, each early teacher with P(target) >= 0.9: the one-shot arm-B copy at step 1000 is 'learnt' if it "
                 "works (as P1), 'neither' if P(target) < 0.5, 'between' otherwise; overall 'learnt' if both are, "
                 "'neither' if both are, else 'between'; unscored if a cell is missing",
         "blind_to": "as P1; one task"},
        {"id": "P3_chain_task_text", "lean": "high (each link is a full E34 copy; the start is the only difference)",
         "rule": "for each task with arm A, at T100 and at T1000: the chained copy against the one-shot copy of the same "
                 "teacher is not worse: verdict 'pass' wherever the one-shot's is, adversarial ratio <= 1.25 x the "
                 "one-shot's, P(target) >= the one-shot's - 0.01, gap share <= the one-shot's + 0.005. Holds if all four "
                 "comparisons do; unscored if one is missing",
         "blind_to": "the chain has 2,000-3,000 student steps against the one-shot's 1,000 (named, not removed); links of "
                     "1,000 steps, not the short links a per-step map would use; one seed"},
        {"id": "P4_chain_pile_text", "lean": "holds, medium (Pile-text copies land at different weight changes from batch "
                                               "order alone, Amendment 5 cosine 0.43, with the same behaviour)",
         "rule": "emoticon arm B, at T100 and at T1000: as P3 with tolerances 1.25 x on the ratio, 0.05 on P(target), "
                 "0.02 on the gap share",
         "blind_to": "as P3; one task"},
    ],
    "reported_not_scored": [
        "every state's seven numbers, gap share, start KL and P(target) at student step 100 from the training curve",
        "the do-nothing states' readings and the teachers' own P(target), damage and size at each step",
        "the cosine of the chained and one-shot T1000 copies' weight changes",
        ("the one-shot arm-B T1000 copy against E34's (P 0.902, ratio 1.05); Delta's share of every copy (expected ~1e-9; "
        "above 1e-6 is a bug)"),
    ],
    "not_run": [
        "SGD teachers; LoRA teachers; r0 and the cipher tasks; arm B on r1",
        ("short chain links (fewer student steps per link) and finer chains (steps 1, 3, 30, 300); a one-shot copy trained "
        "3,000 steps to match the chain's total"),
        "a second batch order or fine-tune seed; more adversary starts or a longer adversary; the budget adversary",
        "the scaled adjoint split of the early teachers (E22 read it); the route split",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e41": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def e41(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (path.py --predictions)")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["e41"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another E41 config: {filed['e41']}"
    assert (LADDER / f"windows_{R1}.pt").is_file(), f"the r1 windows file is missing: {LADDER / f'windows_{R1}.pt'}"
    job = Job(device)
    out = load(OUT, job, {"tasks": {}})
    floor = job.w0["pgd_floor"]
    out["C1"] = {"w0_check": job.w0["pass"], "pgd_floor": floor,
                 "pass": job.w0["pass"] and abs(floor / AM.c1_floor - 1) <= AM.c1_floor_tol}
    log(f"[C1] {json.dumps(out['C1'], default=float)}")
    dump(OUT, out)
    if not out["C1"]["pass"]:
        halt(OUT, out, f"C1: W0 check {job.w0['pass']}, floor {floor:.4f} (expected {AM.c1_floor} within {AM.c1_floor_tol:.0%})")
    for task in dict.fromkeys(t for t, _ in AM.groups):
        T = teachers(job, out, task)
        for arm in (a for t, a in AM.groups if t == task):
            group(job, out, task, arm, T)
        del T
    out["score"], out["done"] = score(out), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'], default=float)}")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else e41(args.device)
