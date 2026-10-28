"""teacher_training E41 Amendment 1 (2026-10-28): the emoticon step-10 task-text copy at batch orders 1 and 2. Plan:
reports/plans/2026-10-28_plan_e41-teacher-path.md -> "Amendment 1" (the approved branch for P1's miss, read as
"verdict not pass, behaviour kept").

E41's one-shot task-text copy of the emoticon step-10 teacher (T10) read an adversarial ratio of 1.547 at its step
1000: inconclusive (pass <= 1.396, fail >= 1.612), behaviour kept. This retrains the same copy at batch-order seeds 1
and 2 by Amendment 4's recipe (amend4.train_student_seeded: E34's arm A, Adam 1e-5, 1,000 steps, offsets from zero;
the batch order is the only training randomness) and reads each through E34's harness at its step 1000, as E41 did.
Every harness read also keeps each 128-row Pile block's adversarial ratio to W0's same block (E41's excess sat in one
of the two blocks). Stages, each record dumped to results/teacher_training_e41_amend1.json as it finishes, every
number logged:
  C1  W0 under the Pile harness (job load); stop unless faith_score.w0_check passes and the floor is 0.0748 within 10%
  the emoticon fine-tune (E41's: Adam 3e-5, batch 32, seed 0) at checkpoints 10 and 1000; G1: both reproduce E41's
      P(target) and size; stop otherwise
  T10's gap at W0 and its do-nothing state: the positive control (E41: 2.516, fail), reported
  G2  the task-text copy of T1000 at seed 1 reproduces Amendment 4's seed-1 copy (P 0.99790, ratio 0.9901): the
      seeded path is Amendment 4's; stop otherwise
  the task-text copy of T10 at seed 1, then at seed 2
  score P1, P2 (`score`, pure) -> out["score"]
Finished records are kept on a rerun; the fine-tune is regenerated (deterministic, ~30 s).

Usage (repo root, project venv):  path_amend1.py --predictions   files results/predictions_e41_amend1.json (refuses to overwrite)
                                  path_amend1.py [cuda]          the run (GPU); the predictions file must be filed first
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from amend4 import train_student_seeded
from run import (
    CFG,
    EMO,
    FT_ADAM_LR,
    RESULTS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    read_state,
)

PRED, OUT = RESULTS / "predictions_e41_amend1.json", RESULTS / "teacher_training_e41_amend1.json"
VERDICTS = ("pass", "inconclusive", "fail")  # state_verdict's, in order; the outcome is their median over seeds 0-2


@dataclass(frozen=True)
class Amend1:
    seeds: tuple[int, ...] = (1, 2)  # batch orders of the T10 copy (E41's was 0)
    teacher_steps: tuple[int, ...] = (10, 1000)  # fine-tune checkpoints used: T10 (scored), T1000 (G2)
    ft_lr: float = FT_ADAM_LR  # E34's Adam teacher: 3e-5, batch 32, seed 0, all 24 matrices
    c1_floor: float = 0.0748  # C1: W0's harness PGD floor on the H100 (E34, E41: 0.07482), within c1_floor_tol relative
    c1_floor_tol: float = 0.10
    # G1: E41's emoticon teacher readings (results/teacher_training_e41.json tasks.emoticon.teachers): P(target), size
    g1_teacher: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"10": (0.99931, 0.01125), "1000": (0.99961, 0.03209)})
    g1_tol: tuple[float, float] = (0.005, 0.001)  # absolute on P(target), size (E41's)
    # G2: Amendment 4's emoticon Adam arm-A copy at batch-order seed 1, step 1000 (teacher_training_amend4.json): P, ratio
    g2_seed: int = 1
    g2_a4: tuple[float, float] = (0.99790, 0.99007)
    g2_tol: tuple[float, float] = (0.005, 0.15)  # absolute on P(target), relative on the ratio (E41's G3)
    # E41's seed-0 copy of T10 (tasks.emoticon.A.oneshot["10"]; per-block ratios 1.06 / 2.00 from its [faith] lines)
    seed0: dict[str, Any] = field(default_factory=lambda: {
        "pgd": 1.54653, "stochastic": 0.98143, "p_target": 0.99941, "gap_share": 0.00128, "verdict": "inconclusive"})
    works_gap: float = 0.05  # behaviour kept: P(target) >= CFG.p_min and gap share <= this (E41's "works" with a pass)
    spread_tol: float = 0.111  # P2: E34 Amendment 4's largest task-text batch-order ratio gap (r0 Adam A), relative
    near: float = 0.075  # W0's adversary-seed floor: a ratio within this (relative) of a verdict line is flagged near it
    delta_max: float = 1e-6  # every trained state's Delta share (by construction ~1e-9; above this is a bug)

    def __post_init__(self) -> None:
        assert 0 not in self.seeds and len(set(self.seeds)) == len(self.seeds) and len(self.seeds) % 2 == 0, \
            "an odd number of seeds with seed 0, so the median verdict is one seed's"
        assert self.g2_seed == 1 and self.teacher_steps == (10, CFG.steps[-1])
        assert set(self.g1_teacher) == set(map(str, self.teacher_steps))
        assert self.seed0["pgd"] > 0 and self.seed0["verdict"] in VERDICTS[1:], "the branch reads a seed-0 copy that missed"


AM = Amend1()
T10, T1000 = map(str, AM.teacher_steps)
LINES = (1.5 / (1 + AM.c1_floor), 1.5 * (1 + AM.c1_floor))  # faith_score.verdict's pass and fail lines at the floor


class Tee(io.StringIO):
    """Keeps what is printed while passing it on to the real stdout."""

    def __init__(self, out: Any) -> None:
        super().__init__()
        self.out = out

    def write(self, s: str) -> int:
        self.out.write(s)
        return super().write(s)

    def flush(self) -> None:
        self.out.flush()


def with_blocks[R](f: Callable[[], R]) -> tuple[R, list[float]]:
    """f(), and the adversarial KL of every Pile block read inside it, in order: faithfulness.faithfulness logs one
    '[faith] ... pgd_kl <v> ...' line per 128-row block, and the harness keeps only their mean."""
    tee = Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        r = f()
    return r, [float(line.split(" pgd_kl ")[1].split()[0]) for line in tee.getvalue().splitlines() if " [faith] " in line]


def read(job: Job, out: dict[str, Any], label: str, upd: tuple[Any, Any], net: Any) -> dict[str, Any]:
    """run.read_state of an emoticon state with the Pile harness, plus each block's adversarial ratio to W0's block."""
    tctx = job.task(EMO)[0]
    rec, kl = with_blocks(lambda: read_state(job, tctx, label, upd, net, True))
    w0 = out["W0_pgd_blocks"]
    assert len(kl) == len(w0), f"{label}: {len(kl)} [faith] lines, expected {len(w0)}"
    assert abs(sum(kl) / sum(w0) / rec["pgd"] - 1) <= 1e-3, f"{label}: block KLs {kl} do not give the ratio {rec['pgd']}"
    rec["pgd_blocks"] = [k / w for k, w in zip(kl, w0, strict=True)]
    log(f"[{label}] adversarial ratio per 128-row block " + " ".join(f"{v:.4f}" for v in rec["pgd_blocks"]))
    return rec


def teachers(job: Job, out: dict[str, Any]) -> dict[str, tuple[dict[str, Any], Any]]:
    """The emoticon fine-tune at AM.teacher_steps, {str(k): (delta-W, teacher net)}; G1 on both readings; T10's gap at
    W0 and its do-nothing state (kept on a rerun)."""
    from commute import teacher
    from commute.finetune import ArmAConfig, finetune_arm_a

    tctx = job.task(EMO)[0]
    cfg = ArmAConfig(o_token=tctx.token, lr=AM.ft_lr, steps=AM.teacher_steps[-1], checkpoints=AM.teacher_steps, seed=0,
                     optimizer="adam")
    res = finetune_arm_a(tctx.model, tctx.train, tctx.held, cfg, tctx.device, log)
    T = {str(k): (res.deltas[k], teacher.teacher_net(tctx.model, res.deltas[k])) for k in AM.teacher_steps}
    rec = out.setdefault("teachers", {})
    for k in T:
        ev = res.evals[int(k)]
        rec.setdefault(k, {})["reading"] = {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"]}
        log(f"[emoticon T{k}] " + json.dumps(rec[k]["reading"], default=float))
    g1(out)
    t = rec[T10]
    if "gap_w0" not in t:
        t["gap_w0"] = gap_w0(tctx, T[T10][1])
        log(f"[emoticon T{T10}] gap at W0 {t['gap_w0']:.5g}")
    if "nothing" not in t:
        t["nothing"] = read(job, out, f"emoticon T{T10} do-nothing", (T[T10][0], None), T[T10][1])
    dump(OUT, out)
    return T


def g1(out: dict[str, Any]) -> None:
    tp, ts = AM.g1_tol
    out["G1"] = {k: {"p_target": (r := out["teachers"][k]["reading"])["p_target"], "size": r["size"], "e41": [p, s],
                     "pass": abs(r["p_target"] - p) <= tp and abs(r["size"] - s) <= ts}
                 for k, (p, s) in AM.g1_teacher.items()}
    log(f"[G1] {json.dumps(out['G1'], default=float)}")
    dump(OUT, out)
    if not all(v["pass"] for v in out["G1"].values()):
        halt(OUT, out, f"G1: the regenerated teachers differ from E41's: {json.dumps(out['G1'], default=float)}")


def teach(job: Job, out: dict[str, Any], T: dict[str, Any], k: str, seed: int) -> dict[str, Any]:
    """The task-text copy of T<k> at batch-order `seed`, read at its step 1000; T10's gets its gap share."""
    tctx = job.task(EMO)[0]
    t0 = time.time()
    curve, states = train_student_seeded(tctx, T[k][1], "adam", CFG.student_lr, CFG.steps[-1:], tctx.train.tokens,
                                         CFG.batch_a, tctx.train.half, seed)
    st = states[CFG.steps[-1]]
    label = f"emoticon A T{k} seed {seed}"
    rec = read(job, out, label, ({key: v[0] for key, v in st.items()}, st), T[k][1])
    rec.update(start_kl=curve[0]["loss"], p_target_100=curve[CFG.steps[0] - 1]["heldout_p_o"], seconds=time.time() - t0)
    if k == T10:
        rec["gap_share"] = rec["teacher_gap"] / out["teachers"][T10]["gap_w0"]
    log(f"[{label}] gap share {rec.get('gap_share', float('nan')):.5f} start KL {rec['start_kl']:.5f} "
        f"P(target)@100 {rec['p_target_100']:.4f} ({rec['seconds']:.0f} s)")
    return rec


def g2(job: Job, out: dict[str, Any], T: dict[str, Any]) -> None:
    if "record" not in out.get("G2", {}):
        out["G2"] = {"record": teach(job, out, T, T1000, AM.g2_seed)}
    (p, r), (tp, tr), rec = AM.g2_a4, AM.g2_tol, out["G2"]["record"]
    out["G2"] = {"record": rec, "a4": [p, r], "pass": abs(rec["p_target"] - p) <= tp and abs(rec["pgd"] / r - 1) <= tr}
    log(f"[G2] P {rec['p_target']:.5f} ratio {rec['pgd']:.4f}; Amendment 4 {p}, {r}; pass {out['G2']['pass']}")
    dump(OUT, out)
    if not out["G2"]["pass"]:
        halt(OUT, out, f"G2: T1000 copy at seed {AM.g2_seed} P {rec['p_target']:.5f} ratio {rec['pgd']:.4f}; "
                       f"Amendment 4 {p}, {r}")


# ---- scoring (pure, on the results file) ----------------------------------------------------------------------------


def category(rec: dict[str, Any]) -> str:
    """'behaviour lost' (P below 0.9 or gap share above 0.05), else the state_verdict ('pass', 'inconclusive', 'fail')."""
    if rec["p_target"] < CFG.p_min or rec["gap_share"] > AM.works_gap:
        return "behaviour lost"
    assert rec["verdict"] in VERDICTS, rec["verdict"]
    return rec["verdict"]


def near_line(ratio: float) -> str | None:
    """The nearer verdict line ('pass' or 'fail') if the adversarial ratio is within AM.near of it, relative; else None."""
    d = {name: abs(ratio / line - 1) for name, line in zip(("pass", "fail"), LINES, strict=True)}
    name = min(d, key=lambda n: d[n])
    return name if d[name] <= AM.near else None


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1 and P2 on whatever the file holds; a missing seed makes both unscored (None), never a pass."""
    r0 = AM.seed0["pgd"]
    recs = {"0": AM.seed0} | {str(s): out.get("copies", {}).get(str(s)) for s in AM.seeds}
    read_ = {s: r for s, r in recs.items() if r is not None}
    cats = {s: category(r) for s, r in read_.items()}
    if len(read_) < len(recs):
        outcome = None
    elif "behaviour lost" in cats.values():
        outcome = "behaviour lost"
    else:
        outcome = sorted(cats.values(), key=VERDICTS.index)[len(cats) // 2]
    within = [None if (r := recs[str(s)]) is None else (1 - AM.spread_tol) * r0 <= r["pgd"] <= (1 + AM.spread_tol) * r0
              for s in AM.seeds]
    ratios = [r["pgd"] for r in read_.values()]
    nothing = out.get("teachers", {}).get(T10, {}).get("nothing")
    top = None if nothing is None or nothing["pgd"] <= 1 else float(nothing["pgd"])  # do-nothing, above W0
    mine = [*(recs[str(s)] for s in AM.seeds), out.get("G2", {}).get("record")]  # this job's states (seed 0 is E41's)
    trained = [r for r in mine if r is not None]
    return {"C1": out.get("C1", {}).get("pass"), "G1": out.get("G1"), "G2": out.get("G2", {}).get("pass"),
            "check_sees_the_update": None if nothing is None else nothing["verdict"] in ("fail", "inconclusive"),
            "P1_batch_order": {"outcome": outcome, "per_seed": cats, "counts": dict(Counter(cats.values())),
                               "split": "pass" in cats.values() and "fail" in cats.values(),
                               "near_line": {s: near_line(r["pgd"]) for s, r in read_.items()}, "lean": "inconclusive"},
            "P2_spread": {"pass": None if None in within else all(within),
                          "ratio_over_seed0": {s: r["pgd"] / r0 for s, r in read_.items()},
                          "spread_max_over_min": max(ratios) / min(ratios) - 1},
            "pgd_blocks": {s: r.get("pgd_blocks") for s, r in read_.items()},
            "excess_removed": {s: None if top is None else (top - r["pgd"]) / (top - 1) for s, r in read_.items()},
            "delta_untouched": bool(trained) and all(r["delta_share"] <= AM.delta_max for r in trained)}


PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training E41 Amendment 1 (reports/plans/2026-10-28_plan_e41-teacher-path.md -> 'Amendment 1'): the "
             "approved branch for E41 P1's miss, read as 'verdict not pass, behaviour kept' (the emoticon T10 task-text "
             "copy read 1.547, inconclusive, P 0.9994, gap share 0.0013). The teacher is E41's emoticon T10 (E34's Adam "
             "3e-5 arm-A fine-tune, all 24 matrices, batch 32, seed 0, stopped at step 10), regenerated in-job. The copy "
             "is E34's arm A (task windows with arm A's weighting, batch 32, Adam 1e-5, 1,000 steps, from W0's (U, V), "
             "Gamma and Delta frozen) at batch-order seeds 1 and 2 by Amendment 4's recipe (amend4.train_student_seeded), "
             "read at its step 1000 through E34's harness. RunPod 'NVIDIA H100 80GB HBM3', E41's card. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2), "
              "faith_score.w0_check passes and the PGD floor is 0.0748 within 10% (E41 0.07482). Fails: stop",
        "G1": "after the fine-tune: T10 and T1000 held-out P(target) within 0.005 and size within 0.001 of E41's (T10 "
              "0.99931, 0.01125; T1000 0.99961, 0.03209). Fails: stop",
        "positive control": "T10's do-nothing state through the harness: expected 2.516, fail, per block 1.17 / 3.79 (E41; "
                            "E22 2.52). Not a stop; the score reports whether it reads above the pass band",
        "G2": "before any T10 copy: the task-text copy of T1000 at batch-order seed 1 reproduces Amendment 4's (P within "
              "0.005 of 0.99790, adversarial ratio within 15% of 0.99007): the seeded path is Amendment 4's. Fails: stop",
    },
    "measurements": {
        "seven numbers": "per state as E34 (run.record): adversarial and stochastic ratios to this job's W0 (seed 0) with "
                         "faith_score.state_verdict against the same-job floor (pass <= 1.396, fail >= 1.612 at floor "
                         "0.0748; P(target) < 0.9 reads 'below P 0.9'), held-out P(target), teacher gap, damage, size, "
                         "Delta's share",
        "per-block ratio": "the adversarial KL of each of the harness's two 128-row Pile blocks (faithfulness's [faith] "
                           "lines) over W0's same block under adversary seed 0 (E41: 0.6794 / 0.7182); their mean ratio is "
                           "the harness's. W0's second block alone spreads 20% over adversary seeds 0-2. Reported",
        "gap share": "teacher gap of the copy / T10's gap at W0 (held-out task windows, arm A's weighting). W0 1, do-nothing "
                     "0; E41 seed 0 0.00128",
        "seed 0": "E41's copy, not rerun: ratio 1.54653 (blocks 1.06 / 2.00), stochastic 0.98143, P 0.99941, gap share "
                  "0.00128, inconclusive",
    },
    "expected": {
        "W0": "ratios 1; P(target) 0.004; gap share 1",
        "teachers": "T10 P 0.99931, size 0.01125, damage 0.0554, gap at W0 5.078; T1000 P 0.99961, size 0.03209",
        "do-nothing T10": "ratio 2.516, stochastic 1.003, fail",
        "G2": "P 0.99790, ratio 0.990, pass (to the digit: same card, same code path)",
        "T10 copies, seeds 1 and 2": "P about 0.999, gap share about 0.001-0.002, stochastic about 0.98, size about 0.014, "
                                     "damage about 0.058, Delta share about 1e-9; ratio about 1.45-1.65",
        "noise": "E34 Amendment 4's batch-order ratio gaps of task-text copies: largest 11.1% (r0 Adam), emoticon 3.2% and "
                 "0.1%; P 0.005; gap share 0.002. W0's adversary-seed floor 7.5%. The pass line is 9.7% below 1.547, the "
                 "fail line 4.2% above",
    },
    "predictions": [
        {"id": "P1_batch_order", "kind": "three-way plus 'behaviour lost', filed with a lean",
         "lean": "inconclusive, medium (emoticon's task-text copies moved 0.1% and 3.2% with batch order in Amendment 4; "
                 "1.547 is 9.7% above the pass line and 4.2% below the fail line)",
         "rule": "each seed's T10 task-text copy at step 1000 (seed 0 is E41's, seeds 1 and 2 new) is 'behaviour lost' if "
                 "P(target) < 0.9 or gap share > 0.05, else its state_verdict: 'pass', 'inconclusive' or 'fail'. Outcome: "
                 "unscored if a seed is missing; 'behaviour lost' if any seed is; else the median of the three verdicts "
                 "ordered pass < inconclusive < fail. Every combination: two or three passes -> pass; two or three fails "
                 "-> fail; any other (three inconclusive, one pass and two inconclusive, one fail and two inconclusive, "
                 "one each) -> inconclusive. Reported beside: the counts, 'split' (a pass and a fail both present), and "
                 "per seed the line its ratio is within 7.5% of (W0's adversary-seed floor), if any. Near-line readings "
                 "keep their verdict: the inconclusive band is already the floor around the 1.5 line, so the rule adds no "
                 "second margin; the flag says which verdicts one adversary start could flip",
         "reading": "pass: E41's miss was a batch-order draw; the map covers the emoticon T10 update at the typical order. "
                    "fail: the step-10 emoticon update is a real limit of teacher training from task text at 1,000 steps. "
                    "inconclusive: the cell sits in the band across batch orders; one 20-step start cannot decide it. "
                    "split: the batch order moves the copy across the whole band",
         "blind_to": "one fine-tune seed, one adversary start per state (a pass is a lower bound), one task, three batch "
                     "orders"},
        {"id": "P2_spread", "kind": "two-sided",
         "lean": "holds, medium (Amendment 4: every task-text gap within 11.1%, emoticon's within 3.2%; the T10 copy's "
                 "excess sat in one of the harness's two blocks, which could make it more order-sensitive)",
         "rule": "per new seed: 0.889 x 1.54653 <= adversarial ratio <= 1.111 x 1.54653 (1.3748 to 1.7182; E34 Amendment "
                 "4's largest task-text batch-order gap, relative to seed 0, inclusive). Holds if both seeds do; unscored "
                 "if one is missing. Reported: max / min - 1 of the ratio over seeds 0-2",
         "reading": "fails: the step-10 copy is more sensitive to batch order than any finished teacher's task-text copy",
         "blind_to": "as P1"},
    ],
    "reported_not_scored": [
        "every state's seven numbers, gap share, start KL, P(target) at student step 100, per-block adversarial ratios",
        "the share of do-nothing's excess each copy removes, (do-nothing - copy) / (do-nothing - 1); E41 seed 0 0.64",
        "Delta's share of every trained state (the score's delta_untouched)",
    ],
    "not_run": [
        "T100 copies at other batch orders (T100's do-nothing passes, so the check has no power there)",
        "a T1000 copy at seed 2 (T1000's spread is Amendment 4's seed 1 against E34's seed 0, 3.2%)",
        ("adversary seeds 1 and 2 on the copies (would separate adversary-start noise from batch-order noise; a departure "
         "from decisions.md's one start, 2026-10-27)"),
        "the r1 T10 copy at other batch orders (it passed, 1.376); Pile-text copies (T10 passed, 1.134)",
        "a copy trained 3,000 steps; the budget adversary; a second fine-tune seed",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e41_amend1": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def amend1(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (path_amend1.py --predictions)")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["e41_amend1"] == json.loads(json.dumps(asdict(AM))), f"filed for another amendment: {filed['e41_amend1']}"
    job, kl = with_blocks(lambda: Job(device))
    out = load(OUT, job, {"copies": {}})
    out["W0_pgd_blocks"] = kl[:len(job.ctx.blocks)]  # harness.load reads adversary seed 0's blocks first
    assert abs(sum(out["W0_pgd_blocks"]) / len(job.ctx.blocks) / job.ctx.pile[0]["pgd_kl"] - 1) <= 1e-3, kl
    floor = job.w0["pgd_floor"]
    out["C1"] = {"w0_check": job.w0["pass"], "pgd_floor": floor,
                 "pass": job.w0["pass"] and abs(floor / AM.c1_floor - 1) <= AM.c1_floor_tol}
    log(f"[C1] {json.dumps(out['C1'], default=float)}; W0 seed-0 block KLs {out['W0_pgd_blocks']}")
    dump(OUT, out)
    if not out["C1"]["pass"]:
        halt(OUT, out, f"C1: W0 check {job.w0['pass']}, floor {floor:.4f} (expected {AM.c1_floor} within {AM.c1_floor_tol:.0%})")
    T = teachers(job, out)
    g2(job, out, T)
    for s in AM.seeds:
        if str(s) not in out["copies"]:
            out["copies"][str(s)] = teach(job, out, T, T10, s)
            dump(OUT, out)
    out["score"], out["done"] = score(out), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'], default=float)}")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend1(args.device)
