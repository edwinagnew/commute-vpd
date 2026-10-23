"""adjoint_sgd (E31): does the adjoint split of a plain-SGD fine-tune keep behaviour, the floor and Delta together?
Plan: reports/2026-10-22_plan_adjoint-sgd.md (design check, mechanism hypothesis, P1-P4) and its Agent notes.

Arm-A fine-tunes of all 24 matrices to 1000 steps. At steps 10, 100 and 1000 each update is read four ways: the
fine-tune's own held-out P(target) and size; the reproduced share R (split.reproduced_share: how much of delta-W the
scaled adjoint split's weight change reproduces on the data); the do-nothing state and the scaled adjoint state
(split.adjoint_state) through harness.measure (Pile ratios to the same-job W0, route split, Delta's share). Order, each
state written to results/adjoint_sgd.json as it finishes:
  1. W0 under the Pile harness, adversary seeds 0 and 1 (faith_score.w0_check); stop if it fails.
  2. Reproductions, emoticon seed 0: Adam 3e-5, SGD 0.01, Adam 3e-6 (E22's adjoint P(target) within 0.05; the large
     one's R within 0.03 of E14's); stop if one fails.
  3. Emoticon seed 1: SGD 0.01, Adam 3e-6, Adam 3e-5.
  4. r1_t534_v369, then r0_t352_v310: the five-rate SGD grid at step 10 with E4's rule picks the rate, then that SGD
     fine-tune and Adam 3e-5. Sigma (for c and R) is the task's own training windows' input second moments.
  5. Budget sweep (E20 amendment 3's recipe) on W0 seeds 0 and 1 and the step-1000 adjoint states of SGD 0.01 and
     Adam 3e-6, emoticon seed 0.

Usage (repo root):  run.py --predictions  (files results/predictions.json once)  |  run.py all  (one job, H100)
  run.py shard X   one of the cluster shards below -> results/adjoint_sgd_X.json, results/large/full_X.log; each
                   runs the same-job W0 floor first. A (the reproductions, gated) runs first; B-E refuse to start
                   unless A's file shows W0 and the Adam 3e-5 and SGD 0.01 reproductions passed
  run.py merge     the shard files -> results/adjoint_sgd.json (JSON only, no model; each row keeps its shard's floor)
  run.py --predictions amend1   files results/predictions_amend1.json once (amendment 1: task vs SGD rate)
  run.py amend1 r1|r0|emo       one cell of the 2x2 -> results/adjoint_sgd_amend1_<cell>.json, results/large/
                                full_amend1_<cell>.log: same-job W0 (G0), the 10-step fine-tune against its earlier
                                reading (A1_G1), then one SGD fine-tune read as an E31 arm. r1 runs first, alone
  run.py amend1_merge           the cell files -> results/adjoint_sgd_amend1.json (JSON only), score_amend1
  run.py --predictions amend2   files results/predictions_amend2.json once (amendment 2: a second training seed)
  run.py amend2 <cell>          r1_03 | r0_03 | r1_01 | r0_01 | emo_03 -> results/adjoint_sgd_amend2_<cell>.json, results/
                                large/full_amend2_<cell>.log: same-job W0 (G0), the cell's 10-step seed-0 fine-tune and
                                its adjoint state against seed 0's known step-10 reading (A2_G1), then the seed-1 arm
  run.py amend2_merge           the cell files -> results/adjoint_sgd_amend2.json (JSON only), score_amend2
Amendment 1 (plan section "Amendment 1" of the plan above): r1, r0 at emoticon's SGD 0.01; emoticon seed 0 at 0.03.
Amendment 2 (plan section "Amendment 2"): batch-order seed 1 for the five single-seed cells of task x SGD rate.
Shards: A emoticon seed 0 (Adam 3e-5, SGD 0.01, Adam 3e-6; G1, G2) | B emoticon seed 1 | C r1_t534_v369 | D r0_t352_v310
| E budget sweep: SGD 0.01 and Adam 3e-6 emoticon seed 0 retrained to step 1000 in-job (A's recipe; their step-1000
fine-tune P and KL are recorded beside A's), then the sweep.
Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "adjoint_sgd.json"
SHARDS = ("A", "B", "C", "D", "E")
LOG = LARGE / "full.log"  # main() points it at full_<shard>.log for a shard
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"

STEPS, TRAIN_STEPS, TOL = (10, 100, 1000), 1000, 1e-6
PILE_ROWS, PILE_PGD_STEPS, PILE_SEEDS = 256, 20, (0, 1)
EMO, R1, R0 = "emoticon", "r1_t534_v369", "r0_t352_v310"
SGD, SMALL, BIG = ("sgd", 0.01), ("adam", 3e-6), ("adam", 3e-5)
SGD_LRS, P_MIN = (0.001, 0.003, 0.01, 0.03, 0.1), 0.9  # sgd_finetune's grid; E4's rule
KL_LIMIT = {R1: 0.24301687240600586, R0: 0.20851298211847694}  # 2 x Adam 3e-5's step-10 KL elsewhere (sgd_ladder.json)
# E22 (small_finetune_splits.json) adjoint P(target) at STEPS, emoticon seed 0; E14 (lora_split.json) R of Adam 3e-5.
E22_P = {BIG: (0.9333, 0.5997, 0.1038), SGD: (0.7144, 0.7535, 0.6132), SMALL: (0.6936, 0.4667, 0.2378)}
E14_R = (0.5763, 0.4222, 0.3064)
P_TOL, R_TOL = 0.05, 0.03
DELTA_MAX, KEEP, GAP, R_MIN, RHO_MIN = 1e-3, 0.5, 0.2, 0.8, 0.7
BUDGETS, BUDGET_ROWS, BUDGET_BATCH = (20, 40, 80, 160), 128, 8  # E20 amendment 3: n sign steps of 2 / n, seed 0
W0_BUDGET20, W0_BUDGET_TOL, BUDGET_PASS = 0.6604, 0.05, 1.241  # E20 A3's W0 at 20 steps (L40); P10's line
assert STEPS[-1] == TRAIN_STEPS and len(E14_R) == len(STEPS)

# Amendment 1: each cell (task, SGD rate, seed 0) with its 10-step fine-tune reading (P(target), KL elsewhere) from
# E31's grid (r1, r0) and sgd_finetune's rate.json (emoticon 0.03), and E31's step-1000 SGD keep ratio it is read against.
PRED_A1, OUT_A1 = RESULTS / "predictions_amend1.json", RESULTS / "adjoint_sgd_amend1.json"
CELLS = {"r1": (R1, 0.01), "r0": (R0, 0.01), "emo": (EMO, 0.03)}
STEP10_REF = {"r1": (0.8364350128173829, 0.22717439651489257), "r0": (0.8427424384552298, 0.15844085841502958),
              "emo": (0.9953822044477071, 0.03702437918480128)}
E31_KEEP = {R1: 0.06678242969068876, R0: 0.27771562753766704, EMO: 0.6043229053484978}  # SGD 0.03, 0.03, 0.01
STEP10_TOL, NEAR = 0.01, 0.1

# Amendment 2: seed 1 of each single-seed cell. A2_G1 reruns the cell's 10-step seed-0 fine-tune and reads it against
# the seed-0 run's step-10 row: (fine-tune P(target), KL elsewhere, R, adjoint P(target)); r1_03, r0_03 from E31
# (adjoint_sgd.json, L40), the others from amendment 1 (adjoint_sgd_amend1.json, L40S).
PRED_A2, OUT_A2 = RESULTS / "predictions_amend2.json", RESULTS / "adjoint_sgd_amend2.json"
CELLS_A2 = {"r1_03": (R1, 0.03), "r0_03": (R0, 0.03), "r1_01": (R1, 0.01), "r0_01": (R0, 0.01), "emo_03": (EMO, 0.03)}
SEED0_STEP10 = {
    "r1_03": (0.9707483673095703, 0.1441326904296875, 0.49896096830844383, 0.5477322933450341),
    "r0_03": (0.9135012580353079, 0.08073844261539792, 0.526734481158538, 0.6065211821163932),
    "r1_01": (0.8373663139343261, 0.22759564399719237, 0.47254548679184305, 0.31344195043842776),
    "r0_01": (0.8426726628275751, 0.15810585021972656, 0.4959894917067962, 0.4941858236268987),
    "emo_03": (0.9954503464372191, 0.03694799501601964, 0.5802935283464365, 0.8214077924008238)}
A2_TOL = (0.01, 0.01, 0.01, 0.02)  # absolute, in SEED0_STEP10's order
# Step-1000 (keep, size) of the states measured before this amendment, (task, SGD rate, seed); all have fine-tune P >= 0.9,
# Delta share <= 1e-4 and verdict 'pass' (E31, amendment 1).
EARLIER = {(EMO, 0.01, 0): (0.6043229053484978, 0.012241833843290806), (EMO, 0.01, 1): (0.5669233692677633, 0.012304019182920456),
           (R1, 0.03, 0): (0.06678242969068876, 0.02860904298722744), (R0, 0.03, 0): (0.27771562753766704, 0.029994947835803032),
           (R1, 0.01, 0): (0.13025632970036657, 0.014092597179114819), (R0, 0.01, 0): (0.39218637943327195, 0.014790496788918972),
           (EMO, 0.03, 0): (0.16631553673750296, 0.027045780792832375)}
SAME, APART, SIZE_MATCH = 0.10, GAP, 1.5  # B1's seed bands on |keep difference|; B3's max/min size at "matched size"


def label(task: str, seed: int, opt: str, lr: float) -> str:
    return f"{task}/s{seed}/{opt}_{lr:g}"


VERDICT = ("faith_score.verdict on the adversarial (PGD) ratio to the same-job W0 under adversary seed 0: pass <= 1.5 / "
           "(1 + f), fail >= 1.5 (1 + f); f = max/min - 1 of W0's PGD KL over seeds 0, 1 (H100 about 0.075 -> 1.40 / 1.61)")
PREDICTIONS: dict[str, Any] = {
    "about": "E31 adjoint_sgd. Plan reports/2026-10-22_plan_adjoint-sgd.md (design check there; the user asked for this "
             "experiment 2026-10-22). Filed before the run; never edited after.",
    "setup": {
        "finetunes": f"arm A (CE at the trigger + KL elsewhere), batch 32, clip 1, all 24 matrices, {TRAIN_STEPS} steps, "
                     f"delta-W at {STEPS}. Emoticon seed 0: Adam 3e-5, SGD 0.01, Adam 3e-6; emoticon seed 1: the same three; "
                     f"{R1} and {R0} seed 0: SGD at the rate E4's rule picks from {SGD_LRS} at step 10 (P >= {P_MIN}, KL "
                     f"elsewhere <= {KL_LIMIT}), and Adam 3e-5",
        "states": "per checkpoint: do-nothing and the scaled adjoint split (split.adjoint_state) through harness.measure: "
                  f"Pile ratios ({PILE_ROWS} rows, {PILE_PGD_STEPS} PGD steps), held-out P(target) at the trigger (route "
                  "split T1), Delta's share of ||delta-W||^2",
        "reproduced_share": "R = 1 - sum_k ||(c_k K(dW_k) - dW_k) Sigma_k^1/2||^2 / sum_k ||dW_k Sigma_k^1/2||^2 over the "
                            "non-zero matrices (split.reproduced_share), K(M) = U U^T M + M V V^T, c_k adjoint_split's scale; "
                            "Sigma = W0's input second moments on the task's training windows (also the scale c's)",
        "keep_ratio": "adjoint state's held-out P(target) / the fine-tune's own held-out P(target), same checkpoint",
        "budget": f"gamma.pgd_sources, n in {BUDGETS} sign steps of 2/n from the seed's random start, first {BUDGET_ROWS} "
                  f"Pile val rows in batches of {BUDGET_BATCH}, fp32 KL (cancel.masked_kl); ratio = state KL (seed 0) / "
                  "W0 KL (seed 0) at the same n",
        "verdict": VERDICT,
    },
    "gates": {
        "G0_w0": "first, alone: faith_score.w0_check. Fails: stop, nothing scored",
        "G1_reproduce_E22": f"emoticon seed 0 Adam 3e-5, SGD 0.01, Adam 3e-6: adjoint P(target) within {P_TOL} of E22's "
                            f"{ {f'{o} {lr:g}': v for (o, lr), v in E22_P.items()} } at every step. Fails: stop",
        "G2_R_reproduces_E14": f"emoticon seed 0 Adam 3e-5: R within {R_TOL} of E14's {E14_R}. Fails: P3 unscored "
                               "(positive control of the new measurement), the run goes on",
        "C_delta": f"every adjoint state's Delta share <= {DELTA_MAX} (positive control that the measurement reads 0)",
        "C_budget_w0": f"W0 seed 0 at 20 budget steps within {W0_BUDGET_TOL:.0%} of E20 A3's {W0_BUDGET20} (L40). Fails: "
                       "P4 reported, unscored",
    },
    "predictions": [
        {"id": "P1_seed", "measure": "emoticon seed 1, step 1000: SGD 0.01's adjoint state; Adam 3e-6's adjoint state",
         "pass": f"SGD: P(target) >= {KEEP}, Delta share <= {DELTA_MAX}, verdict 'pass'; and Adam 3e-6's P(target) <= "
                 f"SGD's - {GAP}", "relative_vs_absolute": "absolute P(target) and Delta share; ratio relative to W0",
         "predicted": "pass: E22 seed 0 read SGD 0.61 and Adam 3e-6 0.24 at step 1000, both passing with Delta 0",
         "expected": "W0: P(target) 0.004, ratio 1. Fine-tunes about 1.0. SGD adjoint 0.6-0.75, Adam 3e-6 adjoint about 0.25",
         "blind_to": "damage the 20-step adversary does not find; off-target damage of the adjoint state; P only at the "
                     "trigger", "positive_control": "C_delta; G1", "noise_floor": "P over the held-out windows "
                     "(deterministic); one more training seed"},
        {"id": "P2_tasks", "measure": f"{R1} and {R0}, step 1000: the SGD fine-tune's adjoint state vs Adam 3e-5's",
         "pass": f"per task: SGD's adjoint P(target) >= {KEEP} and >= Adam 3e-5's + {GAP}, Delta share <= {DELTA_MAX}, "
                 "verdict 'pass'. Reading: 'both' (pass), 'one task' (two-sided, reported which), 'neither' (fail). A task "
                 "whose grid passes no rate is 'unscored' for that task",
         "relative_vs_absolute": "absolute P(target) and difference",
         "predicted": "pass on both, medium-low confidence: the mechanism is optimiser-level, not task-level",
         "expected": "Adam 3e-5 adjoint about 0.1 at step 1000 (E14/E16 on emoticon); SGD rate 0.03 on r1 (E24)",
         "blind_to": "as P1; tasks from one family (single-token trigger and target)",
         "positive_control": "the Adam 3e-5 adjoint state's low P(target) on each task", "noise_floor": "as P1"},
        {"id": "P3_mechanism", "measure": "R of every fine-tune at every step; Spearman of keep_ratio on R over every "
                                          "adjoint state (n reported)",
         "pass": f"(a) every SGD fine-tune's R >= {R_MIN} at every step and above every Adam fine-tune's R on the same "
                 f"task and seed at the same step; (b) Spearman >= {RHO_MIN}. Both parts reported separately",
         "relative_vs_absolute": "absolute R; rank correlation",
         "predicted": "pass: one plain SGD step's adjoint split is one component-training step, and E22's outside-both "
                      "share for SGD is 0.03-0.06; Adam's per-coordinate rescaling is not a sum of gradients (E14 R 0.58 -> "
                      "0.42 -> 0.31)",
         "expected": "Adam 3e-5 emoticon R 0.58 / 0.42 / 0.31 (G2); SGD high; a random update's R is low (E14 did not "
                     "read it; not run here)",
         "blind_to": "cause: a correlation across optimisers and tasks, not an intervention on R",
         "positive_control": "G2", "noise_floor": "R deterministic given delta-W; Spearman over about 30 states that are "
                                                  "not independent (three steps per fine-tune)"},
        {"id": "P4_budget", "measure": "budget ratio of the SGD 0.01 emoticon seed-0 adjoint state at step 1000, n = 40, "
                                       "80, 160",
         "pass": f"<= {BUDGET_PASS} at all three (E20 A3's P10 line); two-sided: a rise above it is reported as the 20-step "
                 "verdict not holding at larger budgets",
         "relative_vs_absolute": "ratio to W0 seed 0 at the same budget",
         "predicted": "pass: E20 A3 read Adam 3e-5's adjoint state at 1.04-1.05 at every budget",
         "expected": "W0 seed 0: 0.66 / 0.68 / 0.70 / 0.71 (L40), seed 1 / seed 0 up to 1.21 at 160. Adam 3e-6 reported",
         "blind_to": "other adversary families; budgets above 160", "positive_control": "W0's seed 1 at each budget",
         "noise_floor": "W0's seed-1 / seed-0 ratio at each budget"},
    ],
    "reported_not_scored": ["do-nothing verdicts", "stochastic ratios", "route split d, r", "per-matrix c and R",
                            "the rate grids", "Adam 3e-6's budget ratios", "keep_ratio per state"],
}


CELL_RULE = (f"keep = adjoint state's held-out P(target) / the same fine-tune's held-out P(target), step {TRAIN_STEPS} "
             f"(relative to the fine-tune's own). In order: fine-tune P(target) < {P_MIN} -> 'uninformative: the fine-tune "
             f"did not learn' (no reading of the split, not a pass or fail; reported, not rerun); Delta share > {DELTA_MAX} "
             "or verdict not 'pass' -> 'not faithful'; else the named outcome below")
PREDICTIONS_A1: dict[str, Any] = {
    "about": "E31 amendment 1 (adjoint_sgd): task vs SGD rate. Plan reports/2026-10-22_plan_adjoint-sgd.md -> "
             "'Amendment 1' (design check there; the user asked for it 2026-10-23). results/predictions.json stands as "
             "filed. Filed before the run; never edited after.",
    "setup": {
        "cells": f"plain SGD, arm A, seed 0, {TRAIN_STEPS} steps, E31's recipe (run.py arm(): batch 32, clip 1, all 24 "
                 f"matrices, readings at {STEPS}): {R1} at 0.01, {R0} at 0.01, emoticon at 0.03. With E31's emoticon 0.01 "
                 f"and {R1}/{R0} at 0.03 this completes task x rate",
        "keep_ratio": PREDICTIONS["setup"]["keep_ratio"], "states": PREDICTIONS["setup"]["states"],
        "verdict": VERDICT,
        "references": f"E31 step-{TRAIN_STEPS} SGD keep: {R1} 0.03 {E31_KEEP[R1]:.3f}, {R0} 0.03 {E31_KEEP[R0]:.3f}, emoticon "
                      f"0.01 seed 0 {E31_KEEP[EMO]:.3f} (seed 1 0.567); Adam 3e-5: {R1} 0.053, {R0} 0.248, emoticon 0.102",
    },
    "gates": {
        "G0_w0": "first in every job, alone: faith_score.w0_check. Fails: stop, nothing scored",
        "A1_G1_step10": f"before the arm: the 10-step SGD fine-tune (same call as E31's grid) reads P(target) and KL elsewhere "
                        f"within {STEP10_TOL} (absolute) of {STEP10_REF} (r1, r0: E31's grid; emoticon 0.03: "
                        "sgd_finetune rate.json). Fails: stop that job (the recipe drifted)",
        "C_delta": f"every adjoint state's Delta share <= {DELTA_MAX}",
        "run_order": "cell r1 runs first and alone; r0 and emo start only after its G0 and A1_G1 have passed",
    },
    "predictions": [
        {"id": "A1_word_tasks", "measure": f"keep at step {TRAIN_STEPS} of {R1} and {R0} at SGD 0.01, each against the "
                                           "same task's SGD 0.03 keep (E31)",
         "rule": CELL_RULE, "pass": f"per task: 'rate' if keep >= {KEEP} and >= the 0.03 keep + {GAP}; 'task' if keep <= the "
                                    f"0.03 keep + {NEAR}; else 'partial'. Both tasks the same -> that reading; differing -> "
                                    "'mixed' (reported which); a task 'uninformative' -> 'unscored'",
         "relative_vs_absolute": "keep is relative to the fine-tune's own P(target); thresholds on it are absolute",
         "predicted": "'task', medium-low confidence: plain SGD at a third of the rate for 1000 steps roughly follows the "
                      "0.03 path to step ~330, where E31's 0.03 keep (interpolated in log steps) is about 0.16 on r1 and "
                      "0.38 on r0, both well below 0.5; and at matched update size (0.03, step 100: size 0.012) E31 already "
                      "read 0.26 and 0.49 against emoticon's 0.60",
         "expected": "W0: P(target) small (below the 10-step rate-0.001 grid's 0.05 r1, 0.20 r0), adversarial ratio 1, floor "
                     "0.209 (L40). Do-nothing: keep about 1 (Delta carries delta-W), ratio up to about 2.3. Fine-tune "
                     f"P(target) >= {P_MIN} by step {TRAIN_STEPS} (0.84 at step 10). Update size at step 1000 about 0.01-0.015",
         "blind_to": "one seed per cell; the lr x steps path argument is not tested (no matched-path checkpoint); P only at "
                     "the trigger; damage the 20-step adversary does not find",
         "positive_control": "A1_G1 (the recipe reproduces); E31's Adam 3e-5 keep on each task (low)",
         "noise_floor": "emoticon's two seeds at SGD 0.01 differ by 0.04 in keep; none measured on r1/r0"},
        {"id": "A2_emoticon_fast", "measure": f"keep at step {TRAIN_STEPS} of emoticon seed 0 at SGD 0.03, against "
                                              "emoticon SGD 0.01 seed 0's keep (E31)",
         "rule": CELL_RULE, "pass": f"'rate' if keep <= the 0.01 keep - {GAP} (<= {E31_KEEP[EMO] - GAP:.3f}); 'task' if "
                                    f"keep >= {KEEP}; else 'partial'",
         "relative_vs_absolute": "as A1",
         "predicted": "'task', low confidence: by the same path argument 0.03 to step 1000 is about 0.01 to step 3000, and "
                      "emoticon 0.01's keep falls slowly (0.72 / 0.71 / 0.60 at 10 / 100 / 1000), so about 0.5-0.55",
         "expected": "W0 and do-nothing as A1. Fine-tune P(target) about 1 (0.995 at step 10). Size at step 1000 about "
                     "0.03, near Adam 3e-5's 0.031 (whose keep is 0.10)",
         "blind_to": "as A1", "positive_control": "A1_G1; E31's emoticon Adam 3e-5 keep 0.10",
         "noise_floor": "as A1 (seed difference 0.04)"},
    ],
    "joint_reading": "A1 'task' and A2 'task': the effect belongs to emoticon. A1 'rate' and A2 'rate': it belongs to the "
                     "rate (or the update size it sets). A1 'rate' with A2 'task' or the reverse: both matter; reported as "
                     "such. Not a prediction; written into the analysis",
    "reported_not_scored": ["keep at steps 10 and 100", "update size", "R", "do-nothing verdicts", "route split",
                            "keep against update size across E31 and this amendment"],
}


STATE_RULE = (f"keep = adjoint state's held-out P(target) / the same fine-tune's held-out P(target), step {TRAIN_STEPS} "
              f"(relative to the fine-tune's own). A state is 'uninformative' if its fine-tune P(target) < {P_MIN}, else 'not "
              f"faithful' if its Delta share > {DELTA_MAX} or its verdict is not 'pass'; a comparison that needs such a state "
              "is 'unscored' (reported which state), never rerun inside the amendment")
PREDICTIONS_A2: dict[str, Any] = {
    "about": "E31 amendment 2 (adjoint_sgd): batch-order seed 1 for the five single-seed cells of task x SGD rate. Plan "
             "reports/2026-10-22_plan_adjoint-sgd.md -> 'Amendment 2' (design check there; the user asked for it "
             "2026-10-23). predictions.json and predictions_amend1.json stand as filed. Filed before the run; never edited "
             "after.",
    "setup": {
        "cells": f"plain SGD, arm A, seed 1 (the batch order; the model, data, Gamma and Delta are the same), {TRAIN_STEPS} "
                 f"steps, E31's recipe (run.py arm(), unchanged): {R1} and {R0} at 0.03 (E31) and at 0.01, emoticon at 0.03 "
                 "(amendment 1). Emoticon at 0.01 already has seeds 0 and 1",
        "keep_ratio": PREDICTIONS["setup"]["keep_ratio"], "states": PREDICTIONS["setup"]["states"],
        "size": "the fine-tune's largest relative matrix change max_k ||dW_k|| / ||W0_k|| at the step",
        "verdict": VERDICT + "; each cell against its own job's W0, so cells may run on different cards (card recorded)",
        "references": {f"{t} {lr:g} seed {sd}": {"keep": k, "size": z} for (t, lr, sd), (k, z) in EARLIER.items()},
    },
    "gates": {
        "G0_w0": "first in every job, alone: faith_score.w0_check. Fails: stop, nothing scored",
        "A2_G1_seed0_step10": f"before the seed-1 arm: the cell's 10-step seed-0 SGD fine-tune, its R and its adjoint state's "
                              f"held-out P(target) (harness.measure, no Pile) read (fine-tune P, KL elsewhere, R, adjoint P) "
                              f"within {A2_TOL} (absolute) of the seed-0 run's step-10 row {SEED0_STEP10}. Positive control "
                              "that the job's recipe (data, Sigma, split, readout) is the one seed 0 ran. Fails: stop that "
                              "job, nothing scored",
        "C_delta": f"every adjoint state's Delta share <= {DELTA_MAX}",
        "run_order": "r1_01 runs first and alone; the other four start after its G0 and A2_G1 lines pass",
    },
    "predictions": [
        {"id": "B1_seed", "measure": f"per cell, |keep(seed 1) - keep(seed 0)| at step {TRAIN_STEPS}",
         "rule": STATE_RULE, "pass": f"per cell: 'replicates' if the difference <= {SAME}; 'differs' if >= {APART}; else "
                                     "'loose'. Reported per cell and as the count of each",
         "relative_vs_absolute": "keep is relative to the fine-tune's own P(target); the bands on its difference are absolute",
         "predicted": "'replicates' in all five, medium confidence: emoticon 0.01's two seeds differ by 0.04, and amendment "
                      "1's 10-step gate showed the step-10 fine-tunes repeat to 0.001; but keep falls steeply between steps "
                      "100 and 1000 on the 0.03 cells, where a small change in the path could move it more",
         "expected": "W0: P(target) small, adversarial ratio 1. Do-nothing: keep about 1. Seed-1 fine-tune P(target) >= 0.9 "
                     "(seed 0: 0.93-1.00). Seed-1 keep near seed 0's: r1 0.03 0.07, r0 0.03 0.28, r1 0.01 0.13, r0 0.01 "
                     "0.39, emoticon 0.03 0.17",
         "blind_to": "other sources of variation (the decomposition, the task windows, the held-out set); two seeds give a "
                     "spread, not a distribution",
         "positive_control": "A2_G1 (the recipe reproduces seed 0 at step 10); emoticon 0.01's measured 0.04",
         "noise_floor": "emoticon 0.01's two seeds differ by 0.04 in keep, the only seed spread measured; this amendment "
                        "measures five more"},
        {"id": "B2_size", "measure": f"per task, step {TRAIN_STEPS}: keep of the SGD 0.01 states against the SGD 0.03 "
                                     "states, both seeds each (four states)",
         "rule": STATE_RULE + ". First: every 0.03 state's size must exceed every 0.01 state's, else 'unscored: the sizes "
                              "did not separate'",
         "pass": "per task: 'falls' if the lower of the two 0.01 keeps is above the higher of the two 0.03 keeps; 'rises' if "
                 "the higher 0.01 keep is below the lower 0.03 keep; else 'overlap'",
         "relative_vs_absolute": "as B1; the rule is an ordering of keeps, with no threshold",
         "predicted": "emoticon 'falls' (high confidence: 0.60 and 0.57 against 0.17); r0 'falls' (medium-low: seed 0's gap "
                      "is 0.11, about three emoticon seed spreads); r1 'falls' (low: seed 0's gap is 0.06, 'overlap' nearly "
                      "as likely)",
         "expected": "sizes about 0.012-0.015 at 0.01 and 0.027-0.030 at 0.03 (seed 0); the do-nothing keep about 1 at both",
         "blind_to": "the shape of the curve (two sizes per task); size against path (no size-matched pair of rates); size "
                     "as the largest single matrix's change",
         "positive_control": "emoticon, where seed 0's fall (0.44) is ten times the seed spread",
         "noise_floor": "the two seeds of each cell (B1)"},
        {"id": "B3_order", "measure": f"at matched size: the six SGD 0.01 states at step {TRAIN_STEPS} (three tasks x two "
                                      "seeds)",
         "rule": STATE_RULE + f". First: the six sizes must lie within a factor {SIZE_MATCH} (max / min), else 'unscored: "
                              "sizes not matched'",
         "pass": "'holds' if both emoticon keeps are above both r0 keeps and both r0 keeps are above both r1 keeps; "
                 "'reversed' if, for either pair (emoticon vs r0, r0 vs r1), both keeps of the first are below both of the "
                 "second; else 'overlap' (reported which pair)",
         "relative_vs_absolute": "as B2",
         "predicted": "'holds', medium-high confidence: seed 0 reads 0.60, 0.39, 0.13, gaps of 0.21 and 0.26 against a "
                      "measured seed spread of 0.04",
         "expected": "sizes 0.012-0.015 (seed 0), max / min about 1.2; emoticon 0.60 and 0.57, r0 about 0.39, r1 about 0.13",
         "blind_to": "why the tasks differ; tasks outside one family; P only at the trigger",
         "positive_control": "the emoticon pair, already measured (0.60, 0.57)",
         "noise_floor": "as B2"},
    ],
    "reported_not_scored": ["keep at steps 10 and 100 for seed 1", "update size", "R", "do-nothing verdicts", "route split",
                            "matched size across rates: seed 1's 0.03 step-100 keep against its 0.01 step-1000 keep",
                            "the card of each cell"],
}


# ----------------------------------------------------------------------------- io


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"adjoint_sgd_{out['shard']}.json" if out.get("shard") else OUT
    path.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def save(out: dict[str, Any]) -> None:  # one job's file (a shard or `all`); merge() builds the union
    out["rows"] = rows(out)
    shard = str(out.get("shard"))
    out["score"] = (score_amend1(out) if shard.startswith("amend1") else score_amend2(out) if shard.startswith("amend2")
                    else score(out))
    dump(out)


# ----------------------------------------------------------------------------- the fine-tunes and their states


class Task:
    """A task's context for harness.measure (its windows and target) and Sigma on its training windows."""

    def __init__(self, ctx: Any, name: str) -> None:
        from commute import harness, split
        from commute.data import batches

        if name == EMO:
            self.ctx, self.sigma = ctx, ctx.sigma  # harness.load read the emoticon windows
            return
        w = torch.load(LADDER / f"windows_{name}.pt", weights_only=False)
        self.ctx = harness.with_windows(ctx, w, w["task"]["target"])
        moments = split.input_second_moments(ctx.model, batches(w["train"].tokens, 32), list(ctx.W0))
        self.sigma = {k: v.cpu() for k, v in moments.items()}


def finetune(task: Task, seed: int, opt: str, lr: float, steps: int) -> Any:
    from commute.finetune import ArmAConfig, finetune_arm_a

    c = task.ctx
    ckpts = tuple(s for s in STEPS if s <= steps) or (steps,)
    cfg = ArmAConfig(o_token=c.token, lr=lr, steps=steps, checkpoints=ckpts, seed=seed, optimizer=opt)
    return finetune_arm_a(c.model, c.train, c.held, cfg, c.device, log)


def arm(out: dict[str, Any], task: Task, name: str, seed: int, opt: str, lr: float) -> dict[int, Any]:
    """One fine-tune and its three checkpoints' readings, saved after each; returns the adjoint states by step."""
    from commute import harness, split

    key, t0 = label(name, seed, opt, lr), time.time()
    res = finetune(task, seed, opt, lr, TRAIN_STEPS)
    out["arms"][key] = rec = {"task": name, "seed": seed, "opt": opt, "lr": lr, "steps": {}}
    adj: dict[int, Any] = {}
    for s in STEPS:
        dW = res.deltas.pop(s)
        R = split.reproduced_share(dW, task.ctx.uv_cpu, task.sigma)
        dW_adj, st = split.adjoint_state(dW, task.ctx.uv_cpu, task.sigma)
        rec["steps"][str(s)] = {
            "ft": res.evals[s], "R": R["pooled"],
            "per_matrix": {k: {"c": p["c"], "share": p["share"]} for k, p in R["per_matrix"].items()},
            "nothing": harness.measure(task.ctx, f"{key}@{s} nothing", dW, None, True, log),
            "adjoint": harness.measure(task.ctx, f"{key}@{s} adjoint", dW_adj, st, True, log)}
        adj[s] = (dW_adj, st)
        save(out)
        r = next(x for x in out["rows"] if x["arm"] == key and x["step"] == s)
        log(f"[{key} step {s}] ft P {r['ft_p']:.4f} KL {r['ft_kl']:.4f} size {r['size']:.5f} | R {r['R']:.4f} | adjoint P "
            f"{r['adj_p']:.4f} keep {r['keep']:.3f} pgd {r['adj_pgd']:.3f} ({r['adj_verdict']}) Delta {r['adj_delta']:.2e} | "
            f"nothing pgd {r['nothing_pgd']:.3f} ({r['nothing_verdict']}) | {time.time() - t0:.0f} s")
    return adj


def gate_repro(out: dict[str, Any], opt: str, lr: float) -> bool:
    key = label(EMO, 0, opt, lr)
    st = out["arms"][key]["steps"]
    p = [st[str(s)]["adjoint"]["routes"]["p_cells"]["T1"] for s in STEPS]
    g: dict[str, Any] = {"p": p, "ref": E22_P[(opt, lr)], "ok": all(abs(a - b) <= P_TOL for a, b in zip(p, E22_P[(opt, lr)], strict=True))}
    if (opt, lr) == BIG:
        g["R"] = [st[str(s)]["R"] for s in STEPS]
        g["R_ok"] = all(abs(a - b) <= R_TOL for a, b in zip(g["R"], E14_R, strict=True))
    out["gates"][key] = g
    log(f"[gate {key}] {json.dumps(g)}")
    return g["ok"]


def pick_rate(out: dict[str, Any], task: Task, name: str) -> float:
    """E4's rule over the SGD grid at step 10 (arms.select_rate) with the task's KL limit; the rate is used either way."""
    from commute import arms

    grid = {f"{lr:g}": finetune(task, 0, "sgd", lr, 10).evals[10] for lr in SGD_LRS}
    lr, passed = arms.select_rate(grid, P_MIN, KL_LIMIT[name])
    out["grids"][name] = {"grid": grid, "limit": KL_LIMIT[name], "selected": float(lr), "passed": passed}
    log(f"[{name} grid] {json.dumps({k: (round(v['p_o'], 4), round(v['kl_other'], 4)) for k, v in grid.items()})} -> {lr} "
        f"passed {passed}")
    return float(lr)


def at(ctx: Any, adj: Any) -> Any:
    """The model at an adjoint state (dW_adj, state); None: W0."""
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    if adj is None:
        return contextlib.nullcontext()
    dW, st = adj
    return model_state(ctx.model, target={k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0},
                       components=components_with(ctx.model, ctx.alive, {k: (v[1], v[2]) for k, v in st.items()}))


def budget(out: dict[str, Any], ctx: Any, keep: dict[str, Any]) -> None:
    from commute import cancel, gamma
    from commute.data import batches, pile_rows

    bs = batches(pile_rows("val", 0, BUDGET_ROWS, log), BUDGET_BATCH)
    out["budget"] = kl = {}
    for name, seed, adj in (("W0", 0, None), ("W0", 1, None), *((k, 0, v) for k, v in keep.items())):
        t0 = time.time()
        with at(ctx, adj):
            for n in BUDGETS:
                src, _ = gamma.pgd_sources(ctx.model, bs, n, step_size=2 / n, seed=seed)
                kl.setdefault(name, {}).setdefault(str(seed), {})[str(n)] = v = cancel.masked_kl(ctx.model, bs, src)
                save(out)
                log(f"[budget {name} seed {seed}] {n} steps of {2 / n:.4f}: KL {v:.5f} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- the run


def setup(device: str, shard: str | None, pred: Path = PRED) -> tuple[Any, dict[str, Any]]:
    """Load, the same-job W0 floor first (G0), saved; stops on a failed W0."""
    from commute import faith_score, harness

    if not pred.is_file():
        sys.exit(f"file the predictions first: {pred} (run.py --predictions)")
    t0 = time.time()
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, TOL, PILE_ROWS,
                       PILE_PGD_STEPS, PILE_SEEDS, log)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "shard": shard,
                           "W0": faith_score.w0_check(ctx.pile), "arms": {}, "gates": {}, "grids": {}, "t0": t0}
    save(out)
    log(f"[W0] {json.dumps(_finite(out['W0']))} ({time.time() - t0:.0f} s)")
    if not out["W0"]["pass"]:
        stop(out, "G0: W0 is not at its known values")
    return ctx, out


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    save(out)
    sys.exit(why)


def finish(out: dict[str, Any]) -> None:
    out["seconds"] = time.time() - out.pop("t0")
    save(out)
    log(f"done in {out['seconds']:.0f} s; score {json.dumps(_finite(out['score']), indent=1)}")


def reproductions(out: dict[str, Any], emo: Task) -> dict[str, Any]:
    """Emoticon seed 0, gated in order (G1, G2); returns the step-1000 adjoint states of SGD 0.01 and Adam 3e-6."""
    keep = {}
    for opt, lr in (BIG, SGD, SMALL):
        adj = arm(out, emo, EMO, 0, opt, lr)
        if (opt, lr) in (SGD, SMALL):
            keep[label(EMO, 0, opt, lr)] = adj[TRAIN_STEPS]
        if not gate_repro(out, opt, lr):
            stop(out, f"G1: {label(EMO, 0, opt, lr)} does not reproduce E22")
    return keep


def other_task(out: dict[str, Any], ctx: Any, name: str) -> None:
    task = Task(ctx, name)
    arm(out, task, name, 0, "sgd", pick_rate(out, task, name))
    arm(out, task, name, 0, *BIG)


def stage_all(device: str) -> None:
    ctx, out = setup(device, None)
    emo = Task(ctx, EMO)
    keep = reproductions(out, emo)
    for opt, lr in (SGD, SMALL, BIG):
        arm(out, emo, EMO, 1, opt, lr)
    for name in (R1, R0):
        other_task(out, ctx, name)
    budget(out, ctx, keep)
    finish(out)


def control_passed() -> None:
    """Shards B-E start only after shard A's W0 and its first two reproductions (Adam 3e-5, SGD 0.01) passed."""
    path = RESULTS / "adjoint_sgd_A.json"
    a = json.loads(path.read_text()) if path.is_file() else {}
    g = a.get("gates", {})
    ok = a.get("W0", {}).get("pass") and all(g.get(label(EMO, 0, *x), {}).get("ok") for x in (BIG, SGD))
    if not ok or "stopped" in a:
        sys.exit(f"shard A has not passed W0 and the Adam 3e-5 and SGD 0.01 reproductions yet ({path})")


def retrained(out: dict[str, Any], emo: Task) -> dict[str, Any]:
    """Shard E: SGD 0.01 and Adam 3e-6, emoticon seed 0, retrained with A's recipe; their step-1000 adjoint states."""
    from commute import split

    keep, out["retrained"] = {}, {}
    for opt, lr in (SGD, SMALL):
        res = finetune(emo, 0, opt, lr, TRAIN_STEPS)
        out["retrained"][label(EMO, 0, opt, lr)] = res.evals[TRAIN_STEPS]
        keep[label(EMO, 0, opt, lr)] = split.adjoint_state(res.deltas[TRAIN_STEPS], emo.ctx.uv_cpu, emo.sigma)
        log(f"[retrained {label(EMO, 0, opt, lr)}] step {TRAIN_STEPS}: {json.dumps(res.evals[TRAIN_STEPS])}")
    save(out)
    return keep


def stage_shard(device: str, shard: str) -> None:
    if shard != "A":
        control_passed()
    ctx, out = setup(device, shard)
    emo = Task(ctx, EMO)
    if shard == "A":
        reproductions(out, emo)
    elif shard == "B":
        for opt, lr in (SGD, SMALL, BIG):
            arm(out, emo, EMO, 1, opt, lr)
    elif shard in ("C", "D"):
        other_task(out, ctx, R1 if shard == "C" else R0)
    else:
        budget(out, ctx, retrained(out, emo))
    finish(out)


def merge() -> None:
    """The shard files -> adjoint_sgd.json: arms, gates, grids and budget unioned, rows each read against their own
    shard's W0 floor, the score on the union."""
    parts = {s: json.loads(p.read_text()) for s in SHARDS if (p := RESULTS / f"adjoint_sgd_{s}.json").is_file()}
    out: dict[str, Any] = {"W0": {"pass": all(p["W0"]["pass"] for p in parts.values())}, "arms": {}, "gates": {},
                           "grids": {}, "rows": [], "shards": {}}
    for s, p in parts.items():
        out["shards"][s] = {k: p.get(k) for k in ("device", "W0", "seconds", "stopped", "retrained")}
        for k in ("arms", "gates", "grids"):
            out[k] |= p[k]
        out["rows"] += p["rows"]
        if "budget" in p:
            out["budget"] = p["budget"]
    out["score"] = score(out)
    dump(out)
    print(json.dumps(_finite(out["score"]), indent=1))


def stage_amend1(device: str, cell: str) -> None:
    """One cell: W0 (G0), the 10-step fine-tune against its earlier reading (A1_G1, stop on a miss), then the arm."""
    name, lr = CELLS[cell]
    ctx, out = setup(device, f"amend1_{cell}", PRED_A1)
    task = Task(ctx, name)
    ev = finetune(task, 0, "sgd", lr, 10).evals[10]
    ref = STEP10_REF[cell]
    out["gates"]["A1_G1"] = g = {"p_o": ev["p_o"], "kl_other": ev["kl_other"], "ref": ref,
                                 "ok": abs(ev["p_o"] - ref[0]) <= STEP10_TOL and abs(ev["kl_other"] - ref[1]) <= STEP10_TOL}
    save(out)
    log(f"[gate A1_G1 {cell}] {json.dumps(g)}")
    if not g["ok"]:
        stop(out, f"A1_G1: the 10-step SGD {lr:g} fine-tune on {name} does not reproduce its earlier reading")
    arm(out, task, name, 0, "sgd", lr)
    finish(out)


def merge_amend1() -> None:
    parts = {c: json.loads(p.read_text()) for c in CELLS if (p := RESULTS / f"adjoint_sgd_amend1_{c}.json").is_file()}
    out: dict[str, Any] = {"shard": "amend1", "arms": {}, "rows": [], "cells": {}}
    for c, p in parts.items():
        out["cells"][c] = {k: p.get(k) for k in ("device", "W0", "gates", "seconds", "stopped")}
        out["arms"] |= p["arms"]
        out["rows"] += p["rows"]
    out["score"] = score_amend1(out)
    OUT_A1.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    print(json.dumps(_finite(out["score"]), indent=1))


def stage_amend2(device: str, cell: str) -> None:
    """One cell: W0 (G0), the seed-0 10-step fine-tune and its adjoint state against seed 0's step-10 row (A2_G1, stop on
    a miss), then the seed-1 arm."""
    from commute import harness, split

    name, lr = CELLS_A2[cell]
    ctx, out = setup(device, f"amend2_{cell}", PRED_A2)
    task = Task(ctx, name)
    res = finetune(task, 0, "sgd", lr, 10)
    ev, dW = res.evals[10], res.deltas[10]
    dW_adj, st = split.adjoint_state(dW, task.ctx.uv_cpu, task.sigma)
    adj = harness.measure(task.ctx, f"A2_G1 {cell} seed 0 @10 adjoint", dW_adj, st, False, log)
    read = (ev["p_o"], ev["kl_other"], split.reproduced_share(dW, task.ctx.uv_cpu, task.sigma)["pooled"],
            adj["routes"]["p_cells"]["T1"])
    ok = all(abs(a - b) <= t for a, b, t in zip(read, SEED0_STEP10[cell], A2_TOL, strict=True))
    out["gates"]["A2_G1"] = g = {"read": read, "ref": SEED0_STEP10[cell], "tol": A2_TOL, "ok": ok}
    save(out)
    log(f"[gate A2_G1 {cell}] {json.dumps(g)}")
    if not ok:
        stop(out, f"A2_G1: the 10-step seed-0 SGD {lr:g} fine-tune on {name} does not reproduce seed 0's step-10 row")
    arm(out, task, name, 1, "sgd", lr)
    finish(out)


def merge_amend2() -> None:
    parts = {c: json.loads(p.read_text()) for c in CELLS_A2 if (p := RESULTS / f"adjoint_sgd_amend2_{c}.json").is_file()}
    out: dict[str, Any] = {"shard": "amend2", "arms": {}, "rows": [], "cells": {}}
    for c, p in parts.items():
        out["cells"][c] = {k: p.get(k) for k in ("device", "W0", "gates", "seconds", "stopped")}
        out["arms"] |= p["arms"]
        out["rows"] += p["rows"]
    out["score"] = score_amend2(out)
    OUT_A2.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    print(json.dumps(_finite(out["score"]), indent=1))


# ----------------------------------------------------------------------------- rows and score


def rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat record per (fine-tune, step) measured so far: what plot.py and the report read."""
    from commute.faith_score import verdict

    f = out["W0"]["pgd_floor"]
    res = []
    for key, a in out["arms"].items():
        for s, r in a["steps"].items():
            ad, no = r["adjoint"], r["nothing"]
            p = ad["routes"]["p_cells"]["T1"]
            res.append({"arm": key, "task": a["task"], "seed": a["seed"], "opt": a["opt"], "lr": a["lr"], "step": int(s),
                        "ft_p": r["ft"]["p_o"], "ft_kl": r["ft"]["kl_other"], "size": r["ft"]["rel_norm_max"], "R": r["R"],
                        "adj_p": p, "keep": p / r["ft"]["p_o"], "adj_pgd": ad["pile"]["pgd_kl"],
                        "adj_verdict": verdict(ad["pile"]["pgd_kl"], f), "adj_stochastic": ad["pile"]["stochastic_kl"],
                        "adj_delta": ad["delta_change_frac"], "nothing_pgd": no["pile"]["pgd_kl"],
                        "nothing_verdict": verdict(no["pile"]["pgd_kl"], f), "floor": f})
    return res


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> Tensor:
        t = torch.tensor(v, dtype=torch.float64)
        return (t[:, None] > t[None, :]).sum(1) + 0.5 * ((t[:, None] == t[None, :]).sum(1) - 1)  # average ranks

    a, b = ranks(x), ranks(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def keeps(r: dict[str, Any]) -> bool:
    return r["adj_p"] >= KEEP and r["adj_delta"] <= DELTA_MAX and r["adj_verdict"] == "pass"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on whatever is measured so far; a prediction whose inputs are missing is absent."""
    R = {(r["arm"], r["step"]): r for r in out["rows"]}
    end = {k: r for (k, s), r in R.items() if s == TRAIN_STEPS}
    sc: dict[str, Any] = {"G0_w0": out["W0"]["pass"], "G1": {k: g["ok"] for k, g in out["gates"].items()},
                          "C_delta": max((r["adj_delta"] for r in out["rows"]), default=None)}
    g2 = out["gates"].get(label(EMO, 0, *BIG), {}).get("R_ok")
    sc["G2_R_reproduces_E14"] = g2
    sgd1, adam1 = label(EMO, 1, *SGD), label(EMO, 1, *SMALL)
    if sgd1 in end and adam1 in end:
        a, b = end[sgd1], end[adam1]
        sc["P1_seed"] = {"sgd_p": a["adj_p"], "adam_p": b["adj_p"], "sgd_keeps": keeps(a),
                         "pass": keeps(a) and b["adj_p"] <= a["adj_p"] - GAP}
    per: dict[str, Any] = {}
    for name in (R1, R0):
        grid = out["grids"].get(name)
        if grid is None:
            continue
        sgd, big = end.get(label(name, 0, "sgd", grid["selected"])), end.get(label(name, 0, *BIG))
        if sgd and big:
            ok = keeps(sgd) and sgd["adj_p"] >= big["adj_p"] + GAP
            per[name] = {"sgd_p": sgd["adj_p"], "adam_p": big["adj_p"], "rate": grid["selected"],
                         "pass": ok if grid["passed"] else "unscored: no rate met E4's rule"}
    if len(per) == 2:
        v = [p["pass"] for p in per.values() if isinstance(p["pass"], bool)]
        reading = ("unscored" if not v else "partial: one task unscored" if len(v) == 1 else  # a fallback rate is no reading
                   "both" if all(v) else "neither" if not any(v) else "one task")
        sc["P2_tasks"] = {"per_task": per, "reading": reading, "pass": reading == "both" if len(v) == 2 else reading}
    groups: dict[tuple[str, int], dict[str, str]] = {}
    for key, a in out["arms"].items():
        if len(a["steps"]) == len(STEPS):
            groups.setdefault((a["task"], a["seed"]), {})[key] = a["opt"]
    cond = {}
    for g in groups.values():
        for k, o in g.items():
            if o != "sgd":
                continue
            for s in STEPS:
                cond[f"{k}@{s} R>={R_MIN}"] = R[(k, s)]["R"] >= R_MIN
                cond |= {f"{k}@{s} > {j}": R[(k, s)]["R"] > R[(j, s)]["R"] for j, oj in g.items() if oj == "adam"}
    st = out["rows"]
    rho = spearman([r["R"] for r in st], [r["keep"] for r in st]) if len(st) >= 3 else None
    sc["P3_mechanism"] = {"a_read": all(cond.values()) if cond else None, "a_conditions": cond, "spearman": rho,
                          "n": len(st), "b_read": None if rho is None else rho >= RHO_MIN}
    for part in ("a", "b"):  # the filed rule: a failed G2 leaves P3 unscored; the readings stay as diagnostics
        sc["P3_mechanism"][f"{part}_pass"] = "unscored: G2 failed" if g2 is False else sc["P3_mechanism"][f"{part}_read"]
    kl = out.get("budget", {})
    sk = label(EMO, 0, *SGD)
    if "0" in kl.get("W0", {}) and len(kl.get(sk, {}).get("0", {})) == len(BUDGETS):
        w0 = kl["W0"]["0"]
        ratio = {m: {n: v["0"][n] / w0[n] for n in v["0"]} for m, v in kl.items() if m != "W0" and "0" in v}
        c = abs(w0["20"] / W0_BUDGET20 - 1) <= W0_BUDGET_TOL
        p4 = all(ratio[sk][str(n)] <= BUDGET_PASS for n in BUDGETS[1:])
        sc["P4_budget"] = {"ratio": ratio, "C_budget_w0": c, "pass": p4 if c else f"unscored (C_budget_w0 failed); read {p4}",
                           "w0_seed1_over_seed0": {n: kl["W0"]["1"][n] / w0[n] for n in kl["W0"].get("1", {})}}
    return sc


def cell_status(r: dict[str, Any]) -> str | None:
    """Amendment 1's rule, shared by amendment 2: None if the state can be read, else why not."""
    if r["ft_p"] < P_MIN:
        return "uninformative: the fine-tune did not learn"
    if r["adj_delta"] > DELTA_MAX or r["adj_verdict"] != "pass":
        return "not faithful"
    return None


def cell_reading(r: dict[str, Any]) -> str:
    """A1's per-cell rule at step 1000 on keep = adjoint P(target) / the fine-tune's own (see predictions_amend1.json)."""
    if (why := cell_status(r)) is not None:
        return why
    k, ref = r["keep"], E31_KEEP[r["task"]]
    if r["task"] == EMO:  # read against emoticon SGD 0.01 seed 0
        return "rate" if k <= ref - GAP else "task" if k >= KEEP else "partial"
    return "rate" if k >= KEEP and k >= ref + GAP else "task" if k <= ref + NEAR else "partial"


def score_amend1(out: dict[str, Any]) -> dict[str, Any]:
    end = {r["arm"]: r for r in out["rows"] if r["step"] == TRAIN_STEPS}
    sc: dict[str, Any] = {"C_delta": max((r["adj_delta"] for r in out["rows"]), default=None), "cells": {}}
    for c, (name, lr) in CELLS.items():
        if (r := end.get(label(name, 0, "sgd", lr))) is not None:
            sc["cells"][c] = {"keep": r["keep"], "ft_p": r["ft_p"], "adj_p": r["adj_p"], "size": r["size"],
                              "verdict": r["adj_verdict"], "reading": cell_reading(r)}
    words = [sc["cells"][c]["reading"] for c in ("r1", "r0") if c in sc["cells"]]
    if len(words) == 2:
        both = words[0] if words[0] == words[1] else "mixed"
        a1 = "unscored" if any(w.startswith("uninformative") for w in words) else both
        sc["A1_word_tasks"] = {"reading": a1, "as_predicted": a1 == "task"}
    if "emo" in sc["cells"]:
        a2 = sc["cells"]["emo"]["reading"]
        sc["A2_emoticon_fast"] = {"reading": a2, "as_predicted": a2 == "task"}
    return sc


def order(hi: list[float], lo: list[float]) -> str:
    """'above' if every keep of `hi` is above every keep of `lo`, 'below' if every one is below, else 'overlap'."""
    return "above" if min(hi) > max(lo) else "below" if max(hi) < min(lo) else "overlap"


def score_amend2(out: dict[str, Any]) -> dict[str, Any]:
    """B1-B3 of predictions_amend2.json on the seed-1 rows measured so far and the earlier states (EARLIER)."""
    new = {(r["task"], r["lr"], 1): r for r in out["rows"] if r["step"] == TRAIN_STEPS and r["seed"] == 1}
    states = {k: {"keep": v[0], "size": v[1], "status": None} for k, v in EARLIER.items()}
    states |= {k: {"keep": r["keep"], "size": r["size"], "status": cell_status(r)} for k, r in new.items()}
    sc: dict[str, Any] = {"C_delta": max((r["adj_delta"] for r in out["rows"]), default=None), "B1_seed": {}}
    for c, (name, lr) in CELLS_A2.items():
        if (name, lr, 1) in new:
            s0, s1 = states[(name, lr, 0)], states[(name, lr, 1)]
            d = abs(s1["keep"] - s0["keep"])
            sc["B1_seed"][c] = {"keep": (s0["keep"], s1["keep"]), "diff": d, "reading": s1["status"] or (
                "replicates" if d <= SAME else "differs" if d >= APART else "loose")}

    def group(keys: list[tuple[str, float, int]]) -> list[dict[str, Any]] | str:
        missing = [k for k in keys if k not in states]
        bad = [f"{label(t, sd, 'sgd', lr)}: {states[(t, lr, sd)]['status']}" for t, lr, sd in keys
               if (t, lr, sd) in states and states[(t, lr, sd)]["status"]]
        return f"unscored: {bad}" if bad else "missing" if missing else [states[k] for k in keys]

    sc["B2_size"] = {}
    for name in (EMO, R0, R1):
        slow, fast = group([(name, 0.01, 0), (name, 0.01, 1)]), group([(name, 0.03, 0), (name, 0.03, 1)])
        if isinstance(slow, str) or isinstance(fast, str):
            sc["B2_size"][name] = {"reading": slow if isinstance(slow, str) else fast}
            continue
        sizes_apart = min(x["size"] for x in fast) > max(x["size"] for x in slow)
        o = order([x["keep"] for x in slow], [x["keep"] for x in fast])
        sc["B2_size"][name] = {"keep_01": [x["keep"] for x in slow], "keep_03": [x["keep"] for x in fast],
                               "reading": {"above": "falls", "below": "rises", "overlap": "overlap"}[o] if sizes_apart
                               else "unscored: the sizes did not separate"}
    per = {name: group([(name, 0.01, 0), (name, 0.01, 1)]) for name in (EMO, R0, R1)}
    if "missing" in per.values():
        return sc
    if bad := [v for v in per.values() if isinstance(v, str)]:
        sc["B3_order"] = {"reading": bad[0]}
        return sc
    lists = {k: v for k, v in per.items() if isinstance(v, list)}
    sizes = [x["size"] for v in lists.values() for x in v]
    pairs = {f"{EMO} vs {R0}": order([x["keep"] for x in lists[EMO]], [x["keep"] for x in lists[R0]]),
             f"{R0} vs {R1}": order([x["keep"] for x in lists[R0]], [x["keep"] for x in lists[R1]])}
    reading = ("unscored: sizes not matched" if max(sizes) / min(sizes) > SIZE_MATCH else
               "holds" if all(v == "above" for v in pairs.values()) else
               "reversed" if any(v == "below" for v in pairs.values()) else "overlap")
    sc["B3_order"] = {"pairs": pairs, "size_ratio": max(sizes) / min(sizes), "reading": reading,
                      "keep": {k: [x["keep"] for x in v] for k, v in lists.items()}}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="?", const="main", choices=["main", "amend1", "amend2"])
    ap.add_argument("stage", nargs="?", choices=["all", "shard", "merge", "amend1", "amend1_merge", "amend2", "amend2_merge"])
    ap.add_argument("shard", nargs="?", choices=(*SHARDS, *CELLS, *CELLS_A2))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    global LOG
    if args.predictions:
        path, body = {"main": (PRED, PREDICTIONS), "amend1": (PRED_A1, PREDICTIONS_A1),
                      "amend2": (PRED_A2, PREDICTIONS_A2)}[args.predictions]
        if path.is_file():
            sys.exit(f"{path} exists; filed predictions are never edited")
        RESULTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_finite({**body, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}), indent=1))
        print(f"filed {path}")
    elif args.stage == "all":
        stage_all(args.device)
    elif args.stage == "shard" and args.shard in SHARDS:
        LOG = LARGE / f"full_{args.shard}.log"
        stage_shard(args.device, args.shard)
    elif args.stage == "merge":
        merge()
    elif args.stage == "amend1" and args.shard in CELLS:
        LOG = LARGE / f"full_amend1_{args.shard}.log"
        stage_amend1(args.device, args.shard)
    elif args.stage == "amend1_merge":
        merge_amend1()
    elif args.stage == "amend2" and args.shard in CELLS_A2:
        LOG = LARGE / f"full_amend2_{args.shard}.log"
        stage_amend2(args.device, args.shard)
    elif args.stage == "amend2_merge":
        merge_amend2()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
