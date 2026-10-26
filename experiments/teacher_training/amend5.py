"""teacher_training Amendment 5 (2026-10-26, lead-2): why the two adversaries disagree, and is a copy unique? Plan:
reports/plans/2026-10-23_plan_teacher-training.md -> "Amendment 5" (binding).

The two paths, harness H (faithfulness.pile_faithfulness -> faithfulness.harness_pgd -> the authors'
calc_multibatch_pgd_masked_recon_loss) and budget sweep S (amend4.budget_sweep -> gamma.pgd_sources + cancel.masked_kl).
Every difference that could change the KL read on a fine-tune left in Delta, file:line H then S:
  rows     H: 256 Pile val rows, two blocks of 128 (16 batches of 8), one adversary per block, mean of the blocks
           (faithfulness.py:142-155). S: the first 128 rows (= H's block 0), one adversary (amend4.py:327).      READ
  start    same seed, different numbers: H draws each matrix's uniform start in model.target_module_paths order
           (pgd_utils.py:168-177; config order c_fc, down_proj, q, k, v, o), S in sorted order (gamma.py:226-232,
           vpd.py:94-95).                                                    READ (gamma.harness_start as S's init)
  readout  H returns its own last objective: bf16-autocast masked forward against bf16-autocast target logits
           (pgd_utils.py:200-201, faithfulness.py:130-134). S reads cancel.masked_kl: fp32 masked forward against an
           fp32 target forward (cancel.py:308-335, attribution.py:12-15). pgd_sources' own KL (gamma.py:200-208, 247) is
           H's readout of S's sources.                                                                         READ
  precision of the steps: none. Every forward of both loops runs under bf16 autocast (faithfulness.py:130; S: g and
           target logits vpd.py:176, masked forward local_faith.py:54-56) and the KL is taken in fp32 (recon_loss_kl's
           log_softmax/softmax/kl_div autocast to fp32; vpd.py:207-211). "Precision" is therefore the readout row.
  step size at 160: H's faithfulness fixes 0.1 (faithfulness.py:106, 112); S uses 2/n (amend4.py:202).   READ (H at 2/160)
  masks    same: g + (1 - g) a on the subcomponents, a's last entry alone on Delta, routing "all" (pgd_utils.py:69-92,
           315-335; gamma.py:189-191, cancel.py:315-317); Delta = weight_deltas inside the state in both
           (faithfulness.py:126; gamma.py:229, cancel.py:327). No reading.
  CI       lower-leaky, continuous, in both (pgd_utils.py:262-265 with faithfulness.py:133; vpd.py:180-182); H recomputes
           g every step, S caches it once: the same numbers. No reading.
  loss reduction  mean over every position of every batch in both (pgd_utils.py:200-201; gamma.py:206-208,
           cancel.py:330-335). Gradients: H sums each batch's summed-KL gradient, S each batch's mean-KL gradient; equal
           batches, so the factor is 4096 = 2^12, exact in floating point: the same signs. No reading.
  other    H re-seeds before each block (faithfulness.py:128), so both blocks share one start; nothing before it draws.

Part 1 (seed 0; each reading at W0, then at the regenerated r1 SGD teacher's do-nothing state; every ratio is to W0
under the same reading). Defaults: H per block at 20 steps of 0.1 (harness_b0, harness_b1; "harness" = their mean) and
S at 20 steps of 2/20 (budget: fp32 = S's readout, bf16 = H's readout of the same sources). Variants, one change each:
  budget.bf16      S with H's readout                         start (fp32)   S from H's start
  start.bf16       H's loop rebuilt from S's code on block 0 (should equal harness_b0: the identity check)
  rows256          S on all 256 rows in one adversary         harness_b0   H on S's rows only
  budget160        S at 160 steps of 2/160 (both readouts)    harness160_b0  H at 160 steps of 2/160 on block 0
Part 2: arms A and B of the same teacher at batch-order seeds 0 and 1 (E34's recipe, 1,000 steps), bf16 offsets saved;
cosines between the seeds per arm, with delta-W, A against B at seed 0, and the random-direction null.

Stages, each dumped to results/teacher_training_amend5.json as it finishes, finished ones skipped on a rerun:
  C1   W0 under the Pile harness (job load); stop unless faith_score.w0_check passes and the floor is 0.075 within 5%
  D    W0 defaults; the teacher (fine-tune seed 0) and its defaults; G1: P, size, harness and budget ratios against E34
       and A4; stop otherwise
  V    the 20-step variants at W0, then at the teacher
  P2   arms A, B at seeds 0, 1; then the cosines
  L    the 160-step readings at W0, then at the teacher
  score P1-P4 (`score`, pure) -> out["score"]

Usage (repo root, project venv):  amend5.py --predictions   files results/predictions_amend5.json (refuses to overwrite)
                                  amend5.py [cuda]          the run (GPU); the predictions file must be filed first
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from amend4 import at, train_student_seeded
from run import CFG, LARGE, R1, RESULTS, Job, dump, halt, load, log

PRED5, OUT5 = RESULTS / "predictions_amend5.json", RESULTS / "teacher_training_amend5.json"
ARMS = ("A", "B")


@dataclass(frozen=True)
class Amend5:
    task: str = R1  # the teacher E34 read inconclusive (1.59) and A4's budget sweep read at 9.24
    teacher: str = "sgd"
    seed: int = 0  # adversary seed of every Part 1 reading
    steps: int = 20  # the default budget; the harness's step size 0.1 = 2 / 20
    long_steps: int = 160
    c1_floor: float = 0.075  # C1: W0's harness PGD floor (H100), within c1_floor_tol relative
    c1_floor_tol: float = 0.05
    g1_p_size: tuple[float, float] = (0.983, 0.0287)  # G1: E34's teacher at step 1000, P(target) and size
    g1_p_tol: float = 0.01
    g1_size_tol: float = 0.002
    g1_harness: float = 1.588  # G1: E34's harness ratio of the do-nothing state, within g1_harness_tol absolute
    g1_harness_tol: float = 0.01
    g1_budget: float = 9.24  # G1: A4's budget ratio at 20 steps, within g1_budget_tol relative
    g1_budget_tol: float = 0.05
    rows_tol: float = 0.25  # P1; the copies' batch-order spread (Amendment 4)
    move_tol: float = 0.25  # P2
    w0_tol: float = 0.10  # P3
    same_way: float = 0.8  # P4: seed-to-seed cosine at or above this on both arms
    many: float = 0.5  # P4: below this on either arm, a copy is one of many placements
    student_seeds: tuple[int, ...] = (0, 1)
    null_seed: int = 0

    def __post_init__(self) -> None:
        assert 2 / self.steps == 0.1 and self.long_steps > self.steps and self.student_seeds == (0, 1)


AM = Amend5()
DEFAULTS, VARIANTS, LONG = ("harness_b0", "harness_b1", "budget"), ("start", "rows256"), ("budget160", "harness160_b0")
# P2: each reading is one change from S's default and one from H's block 0; the factor it implicates if it lands near
# the harness ratio, or near the budget ratio. P3: each variant's default at W0.
P2_READINGS = {"budget.bf16": ("readout", "start"), "start": ("start", "readout")}
P3_PAIRS = {"budget.bf16": "budget", "start": "budget", "start.bf16": "budget", "rows256": "budget",
            "rows256.bf16": "budget", "harness_b0": "harness", "budget160.bf16": "budget160", "harness160_b0": "budget160"}
# Reported only, added after filing (review of a16d366); kept out of Amend5 so the filed `amend5` config still matches.
IDENTITY_TOL = 0.07  # start.bf16 / harness_b0 within this at W0 and at the teacher: the two loops are one loop


def _harness(model: Any, bs: list[Any], steps: int) -> dict[str, float]:
    from commute.faithfulness import harness_pgd

    return {"bf16": harness_pgd(model, bs, steps, AM.seed, 2 / steps)}


def _budget(model: Any, bs: list[Any], steps: int, init: Any = None) -> dict[str, float]:
    """S's adversary (`steps` sign steps of 2 / steps from the seed's start, or `init`), read under the sources found
    by S's readout (fp32) and by H's (bf16, pgd_sources' own KL)."""
    from commute import cancel, gamma

    src, own = gamma.pgd_sources(model, bs, steps, step_size=2 / steps, seed=AM.seed, init=init)
    return {"fp32": cancel.masked_kl(model, bs, src), "bf16": own}


def _start(model: Any, bs: list[Any], steps: int) -> dict[str, float]:
    from commute import gamma

    return _budget(model, bs, steps, gamma.harness_start(model, AM.seed))


READ = {"harness_b0": lambda m, bl: _harness(m, bl[0], AM.steps), "harness_b1": lambda m, bl: _harness(m, bl[1], AM.steps),
        "budget": lambda m, bl: _budget(m, bl[0], AM.steps), "start": lambda m, bl: _start(m, bl[0], AM.steps),
        "rows256": lambda m, bl: _budget(m, bl[0] + bl[1], AM.steps),
        "budget160": lambda m, bl: _budget(m, bl[0], AM.long_steps),
        "harness160_b0": lambda m, bl: _harness(m, bl[0], AM.long_steps)}


def kls(rec: dict[str, Any]) -> dict[str, float]:
    """One state's readings as flat names: 'harness' (mean of the blocks), the bf16-only harness readings by name, and
    S's readings by name (fp32) and name.bf16."""
    flat: dict[str, float] = {}
    for k, r in rec.items():
        main = "bf16" if k.startswith("harness") else "fp32"
        flat.update({k if kind == main else f"{k}.{kind}": v for kind, v in r.items()})
    if "harness_b0" in flat and "harness_b1" in flat:
        flat["harness"] = (flat["harness_b0"] + flat["harness_b1"]) / 2
    return flat


def part1(job: Any, out: dict[str, Any], state: str, names: tuple[str, ...], ctx_of: Any) -> None:
    """The readings `names` of one state ('W0' or 'teacher'), each dumped and logged with its ratio to W0's."""
    rec = out["part1"].setdefault(state, {})
    for name in names:
        if name in rec:
            continue
        t0 = time.time()
        with ctx_of():
            rec[name] = READ[name](job.ctx.model, job.ctx.blocks)
        w0 = out["part1"]["W0"][name]
        log(f"[part1 {state} {name}] " + " ".join(f"KL_{k} {v:.5f} ratio_{k} {v / w0[k]:.4f}" for k, v in rec[name].items())
            + f" ({time.time() - t0:.0f} s)")
        dump(OUT5, out)


def ratios(out: dict[str, Any]) -> dict[str, float]:
    w, t = kls(out["part1"]["W0"]), kls(out["part1"].get("teacher", {}))
    return {k: t[k] / w[k] for k in t}


def g1(job: Any, out: dict[str, Any], repro: dict[str, Any]) -> None:
    """G1: the regenerated teacher reproduces E34 (P, size, harness ratio) and A4 (budget ratio at 20); stop otherwise."""
    r, (p, size) = ratios(out), AM.g1_p_size
    checks = {"p_target": abs(repro["p_target"] - p) <= AM.g1_p_tol, "size": abs(repro["size"] - size) <= AM.g1_size_tol,
              "harness": abs(r["harness"] - AM.g1_harness) <= AM.g1_harness_tol,
              "budget": abs(r["budget"] / AM.g1_budget - 1) <= AM.g1_budget_tol}
    w0_h = kls(out["part1"]["W0"])["harness"]
    out["G1"] = {"p_target": repro["p_target"], "size": repro["size"], "damage": repro["damage"], "harness": r["harness"],
                 "budget": r["budget"], "checks": checks, "pass": all(checks.values()),
                 "w0_harness_vs_job": w0_h / job.ctx.pile[0]["pgd_kl"] - 1}  # harness_pgd against the job's own C1 read
    log(f"[G1] {json.dumps(out['G1'], default=float)}")
    dump(OUT5, out)
    if not out["G1"]["pass"]:
        halt(OUT5, out, f"G1: {checks}; P {repro['p_target']:.4f} size {repro['size']:.4f} harness {r['harness']:.4f} "
                        f"budget {r['budget']:.4f}")


def zero_step(job: Any, out: dict[str, Any]) -> None:
    """Reported only (review of a16d366): W0 on block 0 at 0 steps, the harness's code against pgd_sources from the
    harness's start. Both are then the KL at the same random start, so they should agree to ~1e-6."""
    from commute import gamma
    from commute.faithfulness import harness_pgd

    if "zero_step" in out:
        return
    m, b0 = job.ctx.model, job.ctx.blocks[0]
    h = harness_pgd(m, b0, 0, AM.seed)
    s = gamma.pgd_sources(m, b0, 0, seed=AM.seed, init=gamma.harness_start(m, AM.seed))[1]
    out["zero_step"] = {"harness": h, "pgd_sources_from_harness_start": s, "rel_diff": s / h - 1}
    log(f"[zero_step W0] {json.dumps(out['zero_step'], default=float)}")
    dump(OUT5, out)


def copy_path(arm: str, seed: int) -> Any:
    return LARGE / AM.task / f"amend5_{arm}_{AM.teacher}_seed{seed}_step{CFG.steps[-1]}.pt"


def student(job: Any, out: dict[str, Any], net: Any, arm: str, seed: int) -> None:
    """One copy (E34's recipe at batch-order `seed`), saved as bf16 offsets; P(target) at steps 100 and 1000 from the
    training curve's free held-out reading."""
    from commute import arms
    from commute.data import pile_rows

    cp = out["part2"]["copies"]
    if f"{arm}{seed}" in cp:
        return
    tctx = job.task(AM.task)[0]
    if arm == "B" and job.pile is None:
        job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    tokens, batch, half = (job.pile, CFG.batch_b, None) if arm == "B" else (tctx.train.tokens, CFG.batch_a, tctx.train.half)
    t0 = time.time()
    curve, states = train_student_seeded(tctx, net, "adam", CFG.student_lr, CFG.steps[-1:], tokens, batch, half, seed)
    arms.save_offsets(copy_path(arm, seed), tctx.uv_cpu, states[CFG.steps[-1]],
                      {"task": AM.task, "teacher": AM.teacher, "arm": arm, "seed": seed, "step": CFG.steps[-1]})
    cp[f"{arm}{seed}"] = {f"p_target_{s}": curve[s - 1]["heldout_p_o"] for s in CFG.steps} | {"seconds": time.time() - t0}
    log(f"[part2 {arm} seed {seed}] {json.dumps(cp[f'{arm}{seed}'], default=float)}")
    dump(OUT5, out)


def cosines(job: Any, out: dict[str, Any], dW: dict[str, Any]) -> None:
    """The four copies' weight changes rebuilt from their saved offsets: cosine between the seeds per arm, each with
    delta-W, A against B at seed 0, and a random direction's against delta-W and each arm's seed-0 copy (the null)."""
    from commute import finetune, saved_states

    if "cos" in out["part2"]:
        return
    uv = job.task(AM.task)[0].uv_cpu
    d = {f"{a}{s}": saved_states.offsets_state(copy_path(a, s), uv)[0] for a in ARMS for s in AM.student_seeds}

    def c(x: dict[str, Any], y: dict[str, Any]) -> float:
        return finetune.cosines(x, y)["all"]

    out["part2"]["cos"] = {"seeds": {a: c(d[f"{a}1"], d[f"{a}0"]) for a in ARMS}, "with_dW": {k: c(v, dW) for k, v in d.items()},
                           "A_vs_B_seed0": c(d["A0"], d["B0"]),
                           "null": {k: c(finetune.random_direction(v, AM.null_seed), v)
                                    for k, v in {"dW": dW, "A0": d["A0"], "B0": d["B0"]}.items()}}
    log(f"[part2 cos] {json.dumps(out['part2']['cos'], default=float)}")
    dump(OUT5, out)


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1-P4 over the results file (pure; tests/test_teacher_amend5.py). A reading not yet in the file passes nothing."""
    w, r = kls(out["part1"]["W0"]), ratios(out)
    by_state = {st: kls(rec) for st, rec in out["part1"].items()}

    def near(a: str, b: str, tol: float, rr: dict[str, float]) -> bool:
        return a in rr and b in rr and abs(rr[a] / rr[b] - 1) <= tol

    p1 = {"rows256_vs_budget": near("rows256", "budget", AM.rows_tol, r), "harness_b0_vs_harness": near("harness_b0", "harness", AM.rows_tol, r)}
    p2: dict[str, Any] = {}
    for k, (if_h, if_b) in P2_READINGS.items():
        h, b = near(k, "harness", AM.move_tol, r), near(k, "budget", AM.move_tol, r)
        p2[k] = {"to_harness": h, "to_budget": b, "implicates": if_h if h else if_b if b else None}
    p3 = {k: near(k, v, AM.w0_tol, w) for k, v in P3_PAIRS.items()}
    cos = out.get("part2", {}).get("cos")
    p4: dict[str, Any] = {"pass": False, "reading": None}
    if cos:
        s = cos["seeds"]
        p4 = {"seeds": s, "pass": all(v >= AM.same_way for v in s.values()),
              "reading": "same way" if all(v >= AM.same_way for v in s.values())
              else "one of many" if any(v < AM.many for v in s.values()) else "between"}
    identity = {st: f["start.bf16"] / f["harness_b0"] for st, f in by_state.items() if "start.bf16" in f}
    named = [v["implicates"] for v in p2.values()]
    return {"ratios": r, "w0": w, "identity_start_bf16_over_harness_b0": identity,
            "identity_ok": set(identity) == {"W0", "teacher"} and all(abs(v - 1) <= IDENTITY_TOL for v in identity.values()),
            "implicates_agree": None not in named and len(set(named)) == 1,
            "P1": {"pass": all(p1.values()), "per": p1},
            "P2": {"pass": any(v["to_harness"] or v["to_budget"] for v in p2.values()), "per": p2},
            "P3": {"pass": all(p3.values()), "per": p3}, "P4": p4}


PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training Amendment 5 (reports/plans/2026-10-23_plan_teacher-training.md -> 'Amendment 5'). Part 1: "
             "the harness's adversary (H: faithfulness.harness_pgd, the authors' calc_multibatch_pgd_masked_recon_loss) and "
             "the budget sweep's (S: gamma.pgd_sources + cancel.masked_kl) read at W0 and at the regenerated r1_t534_v369 "
             "SGD teacher's do-nothing state (fine-tune seed 0, step 1000), adversary seed 0, each variant changing one "
             "difference (amend5.py's docstring lists them with file:line). Part 2: arms A and B of that teacher at "
             "batch-order seeds 0 and 1, E34's recipe, 1000 steps; cosines of the weight changes. Filed before any run",
    "controls": {
        "C1_w0": "first, alone: W0 under the Pile harness (256 rows, 20 steps, adversary seeds 0, 1, 2); "
                 "faith_score.w0_check passes and its PGD floor (max/min - 1 over the seeds) is 0.075 within 5% relative "
                 "(A4 on an H100: 0.0748). Fails: stop, nothing read",
        "G1_teacher": "after W0's and the teacher's default readings, before any variant: the teacher reproduces E34 "
                      "(held-out P(target) within 0.01 of 0.983, size within 0.002 of 0.0287, harness ratio within 0.01 "
                      "absolute of 1.588) and A4 (budget ratio at 20 steps within 5% relative of 9.24). Fails: stop. "
                      "Reported beside it: W0's harness reading rebuilt per block against the job's C1 reading of the "
                      "same seed (expected 0 to GPU nondeterminism: harness_pgd is faithfulness's PGD, split out)",
    },
    "measurements": {
        "KL": "every reading is a mean KL in nats over every position of its rows, from the state's own logits to its "
              "logits under the adversary's masks (g + (1 - g) a on the subcomponents, a alone on Delta); held fixed: "
              "Gamma, Delta, the Pile val rows, adversary seed 0; varied: one item of the docstring's list per reading. "
              "Ratio = KL(teacher) / KL(W0) under the same reading, relative",
        "harness": "H on blocks 0 and 1 (rows 0-127, 128-255, batches of 8) at 20 steps of 0.1, the mean of the blocks "
                   "(the harness's pgd_kl); harness_b0 = block 0 alone (S's rows)",
        "budget": "S on block 0 at 20 steps of 2/20 from pgd_sources' sorted-order start; fp32 = cancel.masked_kl (S's "
                  "readout, A4's number), budget.bf16 = pgd_sources' own KL under the same sources (H's readout)",
        "start": "S from H's start (gamma.harness_start as pgd_sources' init); start (fp32) = S's readout; start.bf16 = "
                 "H's readout, which is H's loop on block 0 rebuilt from S's code (the identity: should equal harness_b0)",
        "rows256": "S on all 256 rows in one adversary (32 batches of 8), both readouts",
        "budget160": "S at 160 steps of 2/160 on block 0, both readouts (A4's long reading)",
        "harness160_b0": "H at 160 steps of 2/160 on block 0 (S's rows; one block to keep the job near 30 min)",
        "cosines": "finetune.cosines 'all' (the matrices concatenated, float64) of weight changes rebuilt from the saved "
                   "bf16 offsets: seed 1 against seed 0 per arm, each copy against delta-W, arm A against arm B at seed "
                   "0, and a random direction (finetune.random_direction, seed 0) against delta-W and each arm's seed-0 "
                   "copy. Blind to placements equal up to a reparametrisation of the subcomponents (the cosine is on "
                   "the weight change, which is gauge-free). No positive control; the null is the floor (Amendment 1: "
                   "0.0002). Expected with delta-W: 0.08-0.22 (the plan's E34 range); no noise floor for the seed "
                   "cosine exists yet (this is the first)",
        "p_target": "each copy's held-out P(target) at steps 100 and 1000 from the training curve's free reading",
    },
    "expected": {
        "W0": "ratio 1 on every reading by definition. KL on an H100: harness 0.699 (A4 C1, seed 0; blocks unknown "
              "singly); budget 0.663 (A4 C1b); budget.bf16, start (fp32), start.bf16, rows256 (both readouts) within 10% of "
              "budget; harness_b0 within 10% of harness; budget160 0.708 (A4 C1b); budget160.bf16 and harness160_b0 "
              "within 10% of budget160 (but W0's seed-to-seed spread at 160 steps was 23%, A4 C1b)",
        "teacher": "harness 1.588 (E34), budget 9.24 (A4), budget160 10.45 (A4); P(target) 0.983, size 0.0287, damage "
                   "0.056; the variants: P1-P2",
        "copies": "P(target) at 1000 about 0.996 (arm A) and 0.974 (arm B) at seed 0 (E34), within 0.05 at seed 1 (A4: "
                  "within 0.03)",
        "noise": "the budget measure's own 7% at 20 steps and 23% at 160 (A4 C1b, W0 seed 1 / seed 0); the harness's "
                 "seed floor 0.075 (C1); one adversary seed per reading here",
        "time": "H100: load and C1 ~3 min; defaults and G1 ~3 min; variants ~2 min; Part 2 ~10 min (B streams Pile "
                "train rows first, ~3 min, unless cached); the 160-step readings ~9 min",
    },
    "predictions": [
        {"id": "P1_rows_do_not_carry_it", "confidence": "medium (the plan's one claim; its null is that the rows carry it)",
         "rule": "on the teacher: |ratio(rows256) / ratio(budget) - 1| <= 0.25 (fp32 readout) and "
                 "|ratio(harness_b0) / ratio(harness) - 1| <= 0.25; both must hold",
         "predicted": "holds", "why": "25% = the copies' batch-order spread (Amendment 4)",
         "blind_to": "rows past 256; other adversary seeds"},
        {"id": "P2_in_the_code_path", "confidence": "medium that one moves it; low on which (no prediction names it)",
         "rule": "on the teacher, at least one single-difference reading, budget.bf16 (S with H's readout; also H's "
                 "block 0 with S's start) or start (fp32) (S from H's start; also H's block 0 with S's readout), has a "
                 "ratio within 25% relative of this job's harness ratio (gated to 1.588 +- 0.01) or of its budget ratio "
                 "(gated to 9.24 +- 5%)",
         "predicted": "holds",
         "report": "each reading's ratio, which reference it lands near, and the difference that implicates: budget.bf16 "
                   "near the harness or start (fp32) near the budget implicates the readout; budget.bf16 near the budget "
                   "or start (fp32) near the harness implicates the start (amend5.P2_READINGS)",
         "blind_to": "the rows item (P1) and the 160-step readings, reported only; other adversary families"},
        {"id": "P3_w0_agrees", "confidence": "high (A4: the two measures agreed on W0 within 0.04 nats)",
         "rule": "at W0, every variant's KL within 10% relative of its own default's: budget.bf16, start (fp32), "
                 "start.bf16, rows256, rows256.bf16 against budget; harness_b0 against harness; budget160.bf16 and "
                 "harness160_b0 against budget160",
         "predicted": "holds for all eight", "noise": "7% at 20 steps, 23% at 160 (A4 C1b)"},
        {"id": "P4_copies_point_the_same_way", "confidence": "medium (the end states moved by 0.005 in P(target) in "
                                                              "Amendment 4, but nothing has read the weights)",
         "rule": "cosine between the seed-0 and seed-1 copies' weight changes, 'all' entry, per arm: >= 0.8 on both arms "
                 "reads 'same way' (pass); < 0.5 on either arm reads 'one of many' (a copy is one of many behaviourally "
                 "equal placements); otherwise 'between'",
         "predicted": "same way", "blind_to": "reparametrisations of the subcomponents; one teacher, one task"},
    ],
    "reported_not_scored": [
        ("the identity: start.bf16 / harness_b0 at W0 and at the teacher (H's loop rebuilt from S's code; expected 1 to "
         "GPU nondeterminism; a gap means the docstring's list missed a difference)"),
        "the teacher's 160-step ratios (budget160 both readouts, harness160_b0) against A4's 10.45",
        "each copy's cosine with delta-W, arm A against arm B at seed 0, the nulls; P(target) of each copy",
        "harness_b1's ratio",
    ],
    "not_run": [
        "H at 160 steps on block 1 (one block, to keep the job near 30 min)",
        "S from H's start at 160 steps; H with S's start (calc_multibatch takes no initial sources)",
        "adversary seeds past 0; other adversary families; other teachers and tasks",
        "harness reads (verdicts) of the copies; budgets past 160",
    ],
}


def amend5(device: str) -> None:
    from commute import teacher

    if not PRED5.is_file():
        sys.exit(f"file the predictions first: {PRED5} (amend5.py --predictions)")
    filed = json.loads(PRED5.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend5"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend5']}"
    job = Job(device)
    out = load(OUT5, job, {"part1": {}, "part2": {"copies": {}}})
    floor = job.w0["pgd_floor"]
    out["C1"] = {"w0_check": job.w0["pass"], "pgd_floor": floor,
                 "pass": job.w0["pass"] and abs(floor / AM.c1_floor - 1) <= AM.c1_floor_tol}
    log(f"[C1] {json.dumps(out['C1'], default=float)}")
    dump(OUT5, out)
    if not out["C1"]["pass"]:
        halt(OUT5, out, f"C1: W0 check {job.w0['pass']}, floor {floor:.4f} (expected {AM.c1_floor} within {AM.c1_floor_tol:.0%})")
    w0 = contextlib.nullcontext
    part1(job, out, "W0", DEFAULTS, w0)
    tctx = job.task(AM.task)[0]
    dW, repro = job.teacher(AM.task, AM.teacher)
    out["repro"] = repro

    def nothing() -> Any:  # a fresh do-nothing context per reading
        return at(tctx, dW, None)

    part1(job, out, "teacher", DEFAULTS, nothing)
    g1(job, out, repro)
    zero_step(job, out)
    part1(job, out, "W0", VARIANTS, w0)
    part1(job, out, "teacher", VARIANTS, nothing)
    net = teacher.teacher_net(tctx.model, dW)
    for arm in ARMS:
        for seed in AM.student_seeds:
            student(job, out, net, arm, seed)
    cosines(job, out, dW)
    part1(job, out, "W0", LONG, w0)
    part1(job, out, "teacher", LONG, nothing)
    out["score"] = score(out)
    out["done"] = True
    dump(OUT5, out)
    sc = out["score"]
    log("[score] " + json.dumps(sc, default=float))
    log(f"[score] P2 {sc['P2']['pass']} (as filed); reported beside it: implicates "
        f"{ {k: v['implicates'] for k, v in sc['P2']['per'].items()} } agree {sc['implicates_agree']}; identity_ok "
        f"{sc['identity_ok']} {json.dumps(sc['identity_start_bf16_over_harness_b0'], default=float)}")
    log("done")


def file_predictions() -> None:
    if PRED5.is_file():
        sys.exit(f"{PRED5} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED5, {**PREDICTIONS, "config": asdict(CFG), "amend5": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend5(args.device)
