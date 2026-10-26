"""regating_leak (E18): is re-gating a fixed switch pattern the original subcomponents turn into the behaviour, or do
the per-window gate values carry the updated model's computation past the decomposition (a leak)?
Plan: reports/plans/2026-10-21_plan_regating-leak.md (approved class 4; build notes there, 2026-10-21).

Do-nothing states (target W0 + delta-W, released subcomponents C0, Gamma frozen, Delta out of the masked model) of the
step-10 fine-tunes on three tasks, seeds 0 (E11's saved delta-W) and 1 (trained here). On each task's held-out windows,
with the gates G0 of W0 everywhere except the trigger position, the masked model's trigger log-odds under:
  trig    each window's own state gates at the trigger;
  shared  each window's W0 gates plus one shared change per matrix (the mean over the TRAINING windows of the state's
          trigger gates minus W0's), clipped to [0, 1];
  mean    the state's mean trigger gates over the training windows, the same for every window;
  cross   the shared change of a different-target fine-tune (to ' http', 10 steps, same windows);
  random  the shared change's values on S_on (members up by > 0.3) moved onto a random same-size set, 20 draws.
Recovery = (lo - L00) / (L01 - L00), L01 - L00 the full re-gating effect (r x gain). Then J_nat (S_on against the top
subcomponents by W0's gate contrast at Pile positions whose next token is the target) and, unscored, the adversary's
turned-down share of S_on on Pile.

  run.py --predictions     file results/predictions.json (once)
  run.py                   -> results/regating_leak.json (written after every stage), results/large/full.log
  run.py --amend1 --predictions   file results/predictions_amend1.json (once)
  run.py --amend1          amendment 1 (plan, "Amendment 1"): why rho_shared < rho_mean on the word tasks; G1, then the
                           E18 repeat gate, then the arms -> results/regating_leak_amend1.json; Pile stages not rerun
  run.py --amend2 --predictions   file results/predictions_amend2.json (once)
  run.py --amend2          amendment 2 (plan, "Amendment 2"): the shift shuffled over each matrix's alive subcomponents
                           (20 draws), clipped and not, and the no-clip shift by bins of its size; G1, then the exact
                           repeat of amendment 1 -> results/regating_leak_amend2.json
  run.py --amend3 --predictions   file results/predictions_amend3.json (once)
  run.py --amend3          amendment 3 (plan, "Amendment 3"): arms the leading hypothesis H can fail (the state's own
                           values minus the fixed change; another window's values; the change's turned-up and
                           turned-down halves against sign-kept shuffles; the change on Pile rows); G1, then the exact
                           repeat of amendment 2 -> results/regating_leak_amend3.json
  run.py --amend4 --predictions   file results/predictions_amend4.json (once)
  run.py --amend4          amendment 4 (plan, "Amendment 4"): does the remainder (own values minus the fixed change)
                           carry anything specific to its window? The remainder taken from another window against the
                           window's own; G1, then amendment 3's gate and remainder repeated -> results/regating_leak_amend4.json
Inputs (cluster NFS): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
experiments/task_ladder/results/large/windows_<task>.pt, experiments/reading_a_coverage/results/large/<task>/dW_ft_step10_s0.pt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
OUT = "regating_leak.json"
TRANSPORT = HERE.parent / "transport_test" / "results"
LADDER = HERE.parent / "task_ladder" / "results" / "large"
COVERAGE = HERE.parent / "reading_a_coverage" / "results" / "large"

TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
R_REF = {"emoticon": (0.5965, 0.05), "r1_t534_v369": (0.7461, 0.1), "r2_t359_v816": (0.2844, 0.1)}  # E6 ft_s0; E5 do-nothing
FT_LR, FT_STEPS, SEEDS = 3e-5, 10, (0, 1)
NULL_TOKEN = 3944  # ' http': E6's null_b target
THR, N_RANDOM, ZERO_TOL, MIN_GAIN, MIN_GAP = 0.3, 20, 1e-6, 1.0, 0.5
NAT_ROWS, MIN_NAT = ("val", 0, 600), 200
ADV_ROWS, ADV_STEPS = ("val", 0, 64), 20
BATCH = 64
PATTERNS = ("trig", "shared", "mean", "cross")

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-21, before any run; plan reports/plans/2026-10-21_plan_regating-leak.md (Predictions, build notes)",
    "states": "do-nothing placement of the step-10 arm-A fine-tune (Adam 3e-5, batch 32, 10 steps): target W0 + delta-W, "
              "released subcomponents C0, Gamma frozen, Delta out of the masked model. Seed 0 = E11's saved delta-W, seed 1 "
              "trained in the job. Null: the same recipe to ' http' (token 3944), seed 0. Tasks emoticon (target 'o'), "
              "r1_t534_v369 (' which' -> ' was'), r2_t359_v816 (' we' -> ' just')",
    "measurements": {
        "r": "(L01 - L00) / (T1 - T0) on the held-out windows' trigger log-odds of the target; L = CI-masked model, C0",
        "recovery": "(lo_pattern - L00) / (L01 - L00); every pattern is W0's gates except at the trigger position",
        "rho_trig": "the state's own per-window gates at the trigger",
        "rho_shared": "W0's per-window trigger gates + mean over the training windows of (state - W0) trigger gates (alive "
                      "subcomponents; dead 0), clipped to [0, 1]",
        "rho_mean": "the state's mean trigger gates over the training windows, for every window (reported; rho_mean_W0, the "
                    "same with W0's mean, is its reference)",
        "rho_cross": "the null fine-tune's shared change, read at this task's target and divided by this task's gap",
        "random": f"S_on = alive subcomponents whose shared change > {THR}; the shared change's values on S_on moved onto "
                  f"{N_RANDOM} random same-size-per-matrix sets outside S_on (recruit.random_like seeds 0..19), recovery mean",
        "J_nat": f"Jaccard of S_on (seed 0) with the top-|S_on| alive subcomponents by W0's gate contrast (mean g at "
                 f"positions whose next token is the target minus mean g over all positions) on Pile {NAT_ROWS}; random: "
                 f"mean Jaccard of {N_RANDOM} random_like sets with the same top set",
    },
    "blind_to": {
        "recovery": "trigger windows only; interactions between gate changes at different positions (rho_trig measures how "
                    "much sits at the trigger). Positive control for the leak reading: rho_trig itself (per-window values "
                    "do recover). Null: the random moved shift",
        "J_nat": "how 'natural' positions are defined (next-token identity; contexts where the target is predicted but not "
                 "emitted); E6 flagged it as weak on emoticon",
    },
    "expected": {"r": "emoticon 0.60, r1_t534 0.75, r2_t359 0.28 (seed 1 within ~0.05 of seed 0)",
                 "W0_gates": "recovery 0 exactly (trig with W0's gates; shared with a zero shift)",
                 "random": "about 0", "J_nat_random": "about |S_on| / 9881 / 2, about 0.01"},
    "scorable": f"P1 to P3 read only the task-seed states with gain >= {MIN_GAIN} nat and re-gating gap L01 - L00 >= "
                 f"{MIN_GAP} nats (the others are reported); P4 reads only tasks with >= {MIN_NAT} natural positions, and is "
                 "unscored if none has",
    "gate": {"id": "G1_reproduce", "rule": "seed 0: |r - ref| <= 0.05 on emoticon (ref 0.5965, E6 ft_s0) and <= 0.1 on "
                                          "r1_t534 (ref 0.7461) and r2_t359 (ref 0.2844; E5's do-nothing at step 10, the "
                                          "delta-W E11 regenerated); gain >= 1; recovery with W0's gates |.| <= 1e-6",
             "if_fail": "stop; nothing below is run"},
    "predictions": [
        {"id": "P1_trigger_carries", "rule": "rho_trig >= 0.7 on every task and seed", "predicted": "pass"},
        {"id": "P2_shared_pattern", "rule": "rho_shared >= 0.7 rho_trig on every task and seed, and the random mean <= 0.1",
         "leak": "rho_shared <= 0.3 rho_trig", "predicted": "pass on emoticon with confidence; pass on the ladder tasks with "
                                                            "low confidence; a fail on the ladder tasks only is the mixed branch"},
        {"id": "P3_task_specific", "rule": "rho_cross <= 0.1 on every task and seed", "predicted": "pass"},
        {"id": "P4_natural_machinery", "rule": "J_nat (seed 0) >= 3 x the random mean on every task",
         "conditional_on": f">= {MIN_NAT} natural positions in {NAT_ROWS[2]} rows for the task",
         "predicted": "pass on the ladder tasks, uncertain on emoticon"},
    ],
    "reported_not_scored": "rho_mean and rho_mean_W0; |S_on| per task and seed, by layer; Jaccard of S_on across seeds and "
                           "with the null's S_on; the null's held-out P(' http'); J_nat at seed 1; the adversary's "
                           f"turned-down share (source < 0.5) of S_on vs of all alive, at the state and at W0 ({ADV_STEPS} "
                           f"steps, seed 0, Pile {ADV_ROWS})",
}

# ----------------------------------------------------------------------------- amendment 1 (2026-10-21)
PRED_A1 = RESULTS / "predictions_amend1.json"
OUT_A1 = "regating_leak_amend1.json"
E18_TOL, MIN_D, F_READ = 0.02, 0.3, 0.5
PREDICTIONS_A1: dict[str, Any] = {
    "filed": "2026-10-21, before the amendment ran; E18's own predictions.json stands unedited. Plan: "
             "reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 1; analysis reports/analyses/2026-10-21_analysis_regating-leak.md",
    "question": "E18's filed shared pattern clip(g0(x) + s) recovered 0.40 and 0.62 of rho_trig on the word tasks where the "
                "unscored mean pattern m1 (= m0 + s on alive subcomponents) recovered 0.94 and 1.11. Which difference lowers "
                "rho_shared: the clip, W0's per-window variation g0(x) - m0, or the dead subcomponents (shared keeps g0(x) "
                "there, mean uses m1)?",
    "states": "as E18 (same tasks, seeds 0 and 1, do-nothing placement); seed 1 retrained with the same seed",
    "arms": {
        "trig, shared, mean": "E18's, repeated (the gate reads shared and mean)",
        "noclip": "g0(x) + s, not clipped (recruit.added): differs from shared only by the clip",
        "mean_s": "m0 + s for every window (alive: m1; dead: m0): differs from noclip only by g0(x) - m0, and from mean only "
                  "on dead subcomponents (m0 instead of m1)",
        "S_on": f"clip(g0(x) + s on S_on, 0 elsewhere), S_on = alive with s > {THR} (reported: is the off-S_on shift what costs)",
        "mean_W0": "m0 for every window (E18's reference, repeated; = mean_s with s = 0)",
    },
    "measurements": {
        "R_x": "rho_x / rho_trig, rho = recovery (lo - L00) / (L01 - L00) on the held-out windows, as E18's P2",
        "D": "R_mean - R_shared, the gap to explain (E18: emoticon 0.19, r1_t534 0.53, r2_t359 0.49)",
        "f": "f_clip = (R_noclip - R_shared) / D, f_variation = (R_mean_s - R_noclip) / D, f_dead = (R_mean - R_mean_s) / D; "
             "they sum to 1 exactly",
        "f_variation_W0": "R_mean_W0 / D: the part of f_variation that W0's own per-window variation costs with no shift "
                          "(mean_s - noclip at s = 0); f_variation - f_variation_W0 = how much the variation interacts with s",
    },
    "expected": {"W0_gates": "noclip with s = 0 recovers 0 to 1e-6; S_on and shared with s = 0 likewise (E18's G1); mean_s "
                             "with s = 0 is mean_W0, not 0 (E18: 0.10 to 0.28)",
                 "repeat": "trig, shared, mean, mean_W0 within 0.02 of E18",
                 "f_dead": "about 0 (dead subcomponents' gates are near 0 at W0 and the state)"},
    "gate": {"id": "A1_repeat", "rule": f"G1 as E18; and |rho_shared - E18| <= {E18_TOL} and |rho_mean - E18| <= {E18_TOL} "
                                        "on every task and seed; noclip with s = 0 recovers |.| <= 1e-6",
             "if_fail": "stop; no new arm is run"},
    "blind_to": "the decomposition is a path (shared -> noclip -> mean_s -> mean); with interactions the shares depend on the "
                "order. Trigger position and held-out windows only, as E18",
    "noise": "E18's two seeds agree within 0.02 on R_shared and R_mean; the readings' thresholds sit 0.25 of R from each "
             "other on the word tasks",
    "predictions": [
        {"id": "A1_P1_what_lowers_shared",
         "rule": f"on every task-seed state with D >= {MIN_D} (the word tasks): the reading is each term with f >= {F_READ} "
                 "(clip, variation, dead); 'split' if none",
         "clip": f"f_clip >= {F_READ}: R_noclip >= R_shared + {F_READ} D (r1_t534 about >= 0.67, r2_t359 >= 0.87); the "
                 "filed pattern's construction lowers it, the mechanism reading stands",
         "variation": f"f_variation >= {F_READ}: R_noclip stays near R_shared, R_mean_s near R_mean; the shift added to "
                      "each window's own W0 gates does worse than the same mean for every window",
         "dead": f"f_dead >= {F_READ}: R_mean_s near R_noclip, R_mean well above",
         "predicted": "variation on both word tasks, medium confidence: a switched-on subcomponent with W0 gates near 1 in "
                      "some windows and near 0 in others gets s added half-way in the off windows; the clip only acts on "
                      "windows already near 1. f_clip within +/- 0.2; f_dead within +/- 0.1"},
    ],
    "reported_not_scored": "emoticon (D about 0.19); R_S_on (expected within 0.1 of R_shared if the off-S_on shift is "
                           "inert); f_variation - f_variation_W0 (predicted > 0.2, low confidence: the shift interacts "
                           "with the per-window variation); |S_on|",
}

# ----------------------------------------------------------------------------- amendment 2 (2026-10-21)
PRED_A2 = RESULTS / "predictions_amend2.json"
OUT_A2 = "regating_leak_amend2.json"
A2_TOL, IDENTITY, BULK = 1e-6, 0.2, 0.5
BIN_EDGES = (-math.inf, -0.1, -0.01, 0.01, 0.03, 0.1, 0.3, math.inf)
PREDICTIONS_A2: dict[str, Any] = {
    "filed": "2026-10-21, before the amendment ran; E18's and amendment 1's predictions stand unedited. Plan: "
             "reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 2; analysis reports/analyses/2026-10-21_analysis_regating-leak.md "
             "-> Amendment 1 -> What next 1 and 2",
    "question": "Amendment 1 found most of the shared pattern's recovery outside S_on, in the many alive subcomponents with "
                "s <= 0.3, which E18's random null never moved. Does it matter which subcomponent gets which value of the "
                "shared change s, or does any shift with the same values per matrix do as well?",
    "states": "as E18 and amendment 1 (same tasks, seeds 0 and 1, do-nothing placement); seed 1 retrained with the same seed",
    "arms": {
        "trig, shared, noclip": "amendment 1's, repeated: the gate and the unshuffled references",
        "shared_perm": f"clip(g0(x) + s_perm) at the trigger, s_perm = s's values permuted among each matrix's alive "
                       f"subcomponents (recruit.shuffled, seeds 0..{N_RANDOM - 1}; dead stay 0)",
        "noclip_perm": "g0(x) + s_perm, not clipped, the same draws",
        "bins": f"g0(x) + s restricted to the alive subcomponents with s in one bin, not clipped; edges {BIN_EDGES[1:-1]} "
                "(reported)",
    },
    "measurements": {
        "R_x": "rho_x / rho_trig, rho = recovery (lo - L00) / (L01 - L00) on the held-out windows, as amendment 1",
        "ratio": "mean over the 20 draws of rho_perm, divided by the unshuffled rho of the same pattern (relative; = the "
                 "ratio of the R's)",
        "spread": "sd, min and max of R_perm over the 20 draws: the null's noise floor",
    },
    "expected": {"W0_gates": "with s = 0 every shuffled and binned pattern is g0 and recovers 0 exactly (amendment 1 checked "
                             "noclip with s = 0); not rerun",
                 "repeat": "trig, shared, noclip equal amendment 1's to 1e-6 (amendment 1 repeated E18 exactly)",
                 "unshuffled_R": "shared 0.73/0.72, 0.41/0.40, 0.62/0.62; noclip 0.81/0.79, 0.51/0.50, 0.78/0.78"},
    "gate": {"id": "A2_repeat", "rule": f"G1 as E18; and |rho - amendment 1| <= {A2_TOL} for trig, shared and noclip on every "
                                        "task and seed", "if_fail": "stop; no shuffled arm is run"},
    "blind_to": "the shuffle keeps each matrix's values and their sum, so it cannot tell whether which matrix (layer, "
                "attention or MLP) gets how much matters; it also moves negative values onto subcomponents that are off "
                "at the trigger (clipped to no effect in shared_perm, a sign flip in noclip_perm), so noclip_perm is the "
                "harsher null; trigger position and held-out windows only",
    "noise": f"the spread over the {N_RANDOM} draws (reported per state); E18's random null on S_on had max 0.007; seeds "
             "agree within 0.02 on the unshuffled R",
    "predictions": [
        {"id": "A2_P1_identity_shared",
         "rule": f"on every task-seed state with gap >= {MIN_GAP} nats: ratio(shared) <= {IDENTITY} reads 'identity' (which "
                 f"subcomponent gets which value matters); >= {BULK} reads 'bulk' (a shift of many values carries it); "
                 "between otherwise. Pass: identity on every state. Relative to the unshuffled shared recovery",
         "predicted": "identity on every state, medium-high confidence: E18's random set and the ' http' change, both "
                      "fixed shifts of comparable size, gave <= 0.01 and negative recoveries"},
        {"id": "A2_P2_identity_noclip",
         "rule": f"as A2_P1 for the no-clip pattern: ratio(noclip) <= {IDENTITY} identity, >= {BULK} bulk; pass: identity on "
                 "every state",
         "predicted": "identity on every state, medium confidence; the mean may be negative (shuffled negative values "
                      "flip the sign of subcomponents that are off)"},
    ],
    "reported_not_scored": "per bin: its size, the sum of s over it, and R_bin; the sum of R_bin over bins against "
                           "R_noclip (they need not add). Low-confidence guess: the negative bins (s <= -0.01, 391 to 646 "
                           "entries, sums -29 to -105) and the mid bins (0.03 to 0.3) carry more than S_on's bin",
}


# ----------------------------------------------------------------------------- amendment 3 (2026-10-21)
PRED_A3 = RESULTS / "predictions_amend3.json"
OUT_A3 = "regating_leak_amend3.json"
N_DERANGE, POOL_MIN, STRONG = 5, 0.1, 0.1
H_BAR, LEAK_BAR = 0.15, 0.3  # residual R: H if <= H_BAR, leak if >= LEAK_BAR
MIS_SAME, MIS_DROP, MIS_W0_MAX = 0.8, 0.5, 0.2  # mismatched-window R: H if >= MIS_SAME, drop if <= MIS_DROP
CORR_MARGIN, MIN_HALF = 0.2, 0.1
PREDICTIONS_A3: dict[str, Any] = {
    "filed": "2026-10-21, before the amendment ran; E18's, amendment 1's and amendment 2's predictions stand unedited. Plan: "
             "reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 3; decision reports/decisions/2026-10-21_decision_mechanism-reading.md",
    "hypothesis": "H: after a narrow fine-tune, the switching effect at the trigger is carried by one change to Gamma's "
                  "importance values that is (a) the same for every input, (b) specific to which subcomponents receive "
                  "it, and (c) enough on its own: the part of the updated model's values that varies with the input adds "
                  "nothing. A leak is the failure of (c). Every scored arm below is one H can fail",
    "notation": "g0(x), g1(x): W0's and the state's gates at the trigger of held-out window x; m0, m1: their means over the "
                "training windows; s = m1 - m0 on alive subcomponents (0 on dead), the fixed change; s_held: the same "
                "mean change over the held-out windows; s+ = max(s, 0) (turned up), s- = min(s, 0) (turned down); "
                "clip = clip to [0, 1]. Every pattern is W0's gates everywhere except the trigger position",
    "states": "as E18 and amendments 1 and 2 (three tasks, seeds 0 and 1, do-nothing placement); seed 1 retrained with the "
              "same seed",
    "arms": {
        "trig, shared, noclip": "g1(x); clip(g0(x) + s); g0(x) + s: amendment 2's, repeated (the gate)",
        "mean, mean_W0": "m1 and m0 for every window: amendment 1's, repeated (the additivity grid)",
        "resid": "clip(g1(x) - s): the state's own values with the fixed change removed, the input-varying part kept",
        "resid_noclip": "g1(x) - s, not clipped (the grid's missing cell; values below 0 flip a subcomponent's sign)",
        "resid_held": "clip(g1(x) - s_held): the same with the held-out mean change removed (a purely input-varying "
                      "remainder; reported)",
        "mismatch": f"window x gets g1(pi(x)) at the trigger, pi a derangement of the held-out windows "
                    f"(recruit.derangement seeds 0..{N_DERANGE - 1}); mismatch_W0: g0(pi(x)), its reference",
        "up, down": "clip(g0(x) + s+), clip(g0(x) + s-); also not clipped (up_noclip, down_noclip)",
        "down_pool": f"clip(g0(x) + s-) with s- kept only on the pool P = alive subcomponents with m0 >= {POOL_MIN} (the "
                     f"subcomponents on enough at the trigger to be turned down; every s <= -m0, so all s <= -{POOL_MIN} "
                     "are in P)",
        "down_perm": f"s- on P permuted among P (recruit.shuffled with P as the pool, seeds 0..{N_RANDOM - 1}), clipped: "
                     "the same turn-down values on other subcomponents that are on",
        "up_perm": f"s+ permuted among the alive subcomponents (seeds 0..{N_RANDOM - 1}), clipped",
        "pile": f"clip(g0(x) + s) at position `half` of Pile rows {ADV_ROWS} cut to the task's window length, W0's gates "
                "elsewhere (reported)",
    },
    "measurements": {
        "R_x": "rho_x / rho_trig, rho = recovery (lo - L00) / (L01 - L00) on the held-out windows, as amendments 1 and 2",
        "R_mismatch": f"mean over the {N_DERANGE} derangements of R",
        "c_own, c_donor": "per window, the masked gain b(x) = lo(x) - lo00(x) under the mismatched values; a(x) the same "
                          "under its own values. c_own = corr(b, a), c_donor = corr(b, a o pi) (the donor's own gain); mean "
                          "over derangements. c_null = corr(a, a o pi), the chance level",
        "ratio_half": "mean over the 20 draws of R_perm divided by the unshuffled R of the same half (down: down_pool)",
        "survival": "R_down / R_down_noclip: the share of the unclipped turned-down recovery (amendment 2's bins) left "
                    "once no value may go below 0, i.e. once sign flips are removed",
        "grid": "the effect of s at three backgrounds, if effects add: none (R_mean - R_mean_W0), W0's own variation "
                "(R_noclip), the state's own variation (1 - R_resid_noclip)",
        "pile_ratio": "(lo_pile with s - lo_pile with W0's gates) / (rho_shared x gap): the change's effect in nats on "
                      "Pile rows as a share of its effect at the trigger",
    },
    "expected": {"W0": "with no update (g1 = g0, s = 0) resid, resid_noclip, resid_held, up, down, down_pool, the perms and "
                       "pile all read 0; mismatch reads mismatch_W0, not 0; the per-window gains a(x) average to rho_trig x "
                       "gap exactly (checked)",
                 "repeat": "trig, shared, noclip equal amendment 2's to 1e-6; mean and mean_W0 equal amendment 1's (0.92/0.91,"
                           " 0.94/0.94, 1.11/1.15; 0.11/0.10, 0.24/0.24, 0.27/0.29)",
                 "c_null": "about 0 +/- 0.1 (73 to 106 windows)",
                 "survival": "low-confidence guess 0.5 to 0.9",
                 "pile_ratio": "low-confidence guess <= 0.3 (the change raises the target in context)"},
    "gate": {"id": "A3_repeat", "rule": f"G1 as E18; and |rho - amendment 2| <= {A2_TOL} for trig, shared and noclip on every "
                                        "task and seed", "if_fail": "stop; no new arm is run"},
    "blind_to": {
        "resid": "a leak that acts only together with the fixed change (the grid's v1 cell, 1 - R_resid_noclip, reads that); "
                 "the clip removes negative remainders (windows that switched less than average)",
        "mismatch": "the level is blind to a leak that carries a per-window strength (it averages out over the "
                    "derangement): c_own vs c_donor reads that; the level mixes in what mismatching costs with no update "
                    "(mismatch_W0)",
        "halves": "the shuffles keep each matrix's values (and the pool), so they cannot see whether the split across "
                  "matrices matters; down_perm cannot separate identity when P is not much larger than the strongly "
                  "turned-down set (n_down_strong / n_pool reported)",
        "all": "trigger position (Pile arm aside) and held-out windows only; one model, three tasks",
    },
    "noise": f"seeds agree within 0.02 on the unshuffled R; amendment 2's shuffle sd 0.02 to 0.04; correlations over 73 "
             f"to 106 windows have a chance spread of about 0.1 (c_null reported); {N_DERANGE} derangements",
    "scorable": f"states with gap >= {MIN_GAP} nats (all six in amendments 1 and 2); A3_P2 also needs |R_mismatch_W0| <= "
                f"{MIS_W0_MAX}; A3_P4 needs R_down_pool >= {MIN_HALF} and A3_P5 R_up >= {MIN_HALF} (otherwise reported)",
    "predictions": [
        {"id": "A3_P1_fixed_part_needed", "fails": "(c), necessity of the fixed part",
         "rule": f"R_resid <= {H_BAR} reads 'H' (the input-varying part alone does nothing); >= {LEAK_BAR} reads 'leak'; "
                 "between otherwise. Pass: 'H' on every scorable state. Absolute, in shares of rho_trig",
         "predicted": "pass on emoticon, medium confidence; on the word tasks medium-low (their fixed change leaves 0.5 "
                      "and 0.2 of rho_trig unexplained, some of which could sit in the remainder alone)"},
        {"id": "A3_P2_other_windows_values", "fails": "(a), the same for every input",
         "rule": f"R_mismatch >= {MIS_SAME} reads 'H' (another window's values do about as well as the own ones); <= "
                 f"{MIS_DROP} reads 'own window needed'; between otherwise. Pass: 'H' on every scorable state "
                 f"(|R_mismatch_W0| <= {MIS_W0_MAX})",
         "predicted": "pass, medium confidence (the mean pattern, with no per-window term, reads 0.91 to 1.15)"},
        {"id": "A3_P3_gain_follows_recipient", "fails": "(a)/(c) for a leak that carries a per-window strength",
         "rule": f"c_donor - c_own >= {CORR_MARGIN} reads 'donor' (the masked gain travels with the values: a leak); "
                 f"c_own - c_donor >= {CORR_MARGIN} reads 'recipient'; 'neither' otherwise. Pass: no state reads 'donor'",
         "predicted": "pass, medium-high confidence; 'recipient' on the word tasks (varied contexts), possibly 'neither' "
                      "on emoticon (windows alike)"},
        {"id": "A3_P4_down_identity", "fails": "(b) for the turned-down half",
         "rule": "ratio_half(down_pool) <= 0.2 reads 'identity'; >= 0.5 'bulk'; between otherwise. Pass: 'identity' on "
                 f"every state with R_down_pool >= {MIN_HALF}. Relative to the unshuffled down_pool",
         "predicted": "pass, medium-low confidence: turning down any active subcomponents could lower competing tokens "
                      "generically"},
        {"id": "A3_P5_up_identity", "fails": "(b) for the turned-up half",
         "rule": f"as A3_P4 for s+ (up_perm against up), on every state with R_up >= {MIN_HALF}",
         "predicted": "pass, high confidence (E18's random set on S_on and amendment 2's shuffle: <= 0.01)"},
    ],
    "reported_not_scored": "R_resid_noclip, R_resid_held; the grid (does s's effect depend on the background: the "
                           "reference subtraction's additivity); R_mismatch_W0 and c_null; R_up, R_down clipped and not, "
                           "survival, R_up + R_down against R_shared and the unclipped halves against R_noclip; pool sizes; "
                           "pile_ratio",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _json(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    raise TypeError(type(o))


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-9 else float("nan")


def dump(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(_finite(obj), indent=1, sort_keys=True, default=_json))


# ----------------------------------------------------------------------------- loading


def load_task(name: str) -> tuple[Any, Any, int]:
    if name == "emoticon":
        w = torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False)
        return w["train"], w["heldout"], json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    w = torch.load(LADDER / f"windows_{name}.pt", weights_only=False)
    return w["train"], w["heldout"], w["task"]["target"]


def finetune(ctx: dict[str, Any], task: dict[str, Any], token: int, seed: int) -> tuple[dict[str, Tensor], float]:
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=token, lr=FT_LR, steps=FT_STEPS, checkpoints=(FT_STEPS,), seed=seed)
    res = finetune_arm_a(ctx["model"], task["train"], task["held"], cfg, ctx["device"], log)
    return res.deltas[FT_STEPS], res.evals[FT_STEPS]["p_o"]


def state_of(ctx: dict[str, Any], dW: dict[str, Tensor]) -> Any:
    from commute.faithfulness import model_state

    W0 = ctx["W0"]
    return model_state(ctx["model"], target={k: W0[k] + dW[k].to(W0[k].device, W0[k].dtype) for k in W0})


def lo_of(ctx: dict[str, Any], task: dict[str, Any], gates: Any) -> float:
    from commute.local_faith import masked_lo

    held = task["held"]
    return masked_lo(ctx["model"], held.tokens, held.half, task["target"], gates, BATCH)[0]


# ----------------------------------------------------------------------------- stages


def stage_g1(ctx: dict[str, Any], name: str) -> dict[str, Any]:
    """W0's cells, seed 0's state and r; recovery with W0's own gates (must be 0 exactly)."""
    from commute import recruit as R
    from commute.local_faith import route_cells

    train, held, target = load_task(name)
    task: dict[str, Any] = {"train": train, "held": held, "target": target, "half": held.half}
    model, h = ctx["model"], held.half
    task["base"] = route_cells(model, held.tokens, h, target, BATCH)
    task["dW0"] = torch.load(COVERAGE / name / f"dW_ft_step{FT_STEPS}_s0.pt", weights_only=True)
    with state_of(ctx, task["dW0"]):
        task["cur0"] = route_cells(model, held.tokens, h, target, BATCH)
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    gap, gain = task["cur0"]["L"] - L00, task["cur0"]["T"] - task["base"]["T"]
    zero = {"trig": lo_of(ctx, task, R.with_trigger(g0, h, R.at_trigger(g0, h))) - L00,
            "shared": lo_of(ctx, task, R.with_trigger(g0, h, R.shifted(g0, h, {k: torch.zeros(v.shape[-1]) for k, v in g0[0].items()}))) - L00}
    ref, tol = R_REF[name]
    r = div(gap, gain)
    ok = abs(r - ref) <= tol and gain >= MIN_GAIN and all(abs(div(v, gap)) <= ZERO_TOL for v in zero.values())
    rec = {"target": target, "n_train": len(train), "n_heldout": len(held), "r": r, "r_ref": ref, "r_tol": tol, "gain": gain,
           "gap": gap, "cells": {"T0": task["base"]["T"], "T1": task["cur0"]["T"], "L00": L00, "L01": task["cur0"]["L"],
                                 "U0": task["base"]["U"], "U1": task["cur0"]["U"]},
           "zero_recovery": {k: div(v, gap) for k, v in zero.items()}, "ok": ok}
    log(f"[G1 {name}] r {r:.4f} (ref {ref} +/- {tol}) gain {gain:.3f} gap {gap:.3f} | W0-gate recovery trig "
        f"{rec['zero_recovery']['trig']:.2e} shared {rec['zero_recovery']['shared']:.2e} | {'ok' if ok else 'FAIL'}")
    ctx["tasks"][name] = task
    return rec


def patterns(ctx: dict[str, Any], task: dict[str, Any], cur: dict[str, Any], shift: dict[str, Tensor],
             mean1: dict[str, Tensor]) -> dict[str, Any]:
    """Trigger log-odds minus L00 under each pattern, and under the random moved shifts (list)."""
    from commute import recruit as R

    g0, h, L00 = task["base"]["gates"], task["half"], task["base"]["L"]
    trig = {"trig": R.at_trigger(cur["gates"], h), "shared": R.shifted(g0, h, shift), "mean": R.constant(g0, h, mean1),
            "cross": R.shifted(g0, h, task["shift_null"])}
    out: dict[str, Any] = {p: lo_of(ctx, task, R.with_trigger(g0, h, t)) - L00 for p, t in trig.items()}
    on = R.switched(shift, THR)
    out["random"] = [lo_of(ctx, task, R.with_trigger(g0, h, R.shifted(g0, h, R.moved(shift, on, R.random_like(on, ctx["alive"], i)))))
                     - L00 for i in range(N_RANDOM)]
    return out


def stage_task(ctx: dict[str, Any], name: str) -> dict[str, Any]:
    from commute import recruit as R
    from commute.local_faith import route_cells, trigger_mean_gates

    task, model, alive = ctx["tasks"][name], ctx["model"], ctx["alive"]
    train, held, h = task["train"], task["held"], task["half"]
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    m0 = trigger_mean_gates(model, train.tokens, h, BATCH)
    dW_null, p_null = finetune(ctx, task, NULL_TOKEN, 0)
    with state_of(ctx, dW_null):
        task["shift_null"] = R.mean_change(trigger_mean_gates(model, train.tokens, h, BATCH), m0, alive)
    on_null = R.switched(task["shift_null"], THR)
    lo_mean_w0 = lo_of(ctx, task, R.with_trigger(g0, h, R.constant(g0, h, m0))) - L00
    rec: dict[str, Any] = {"null": {"p_heldout": p_null, "n_on": R.size(on_null)}, "seeds": {}}
    for seed in SEEDS:
        if seed == 0:
            dW, cur = task["dW0"], task["cur0"]
        else:
            dW, _ = finetune(ctx, task, task["target"], seed)
            with state_of(ctx, dW):
                cur = route_cells(model, held.tokens, h, task["target"], BATCH)
        with state_of(ctx, dW):
            mean1 = trigger_mean_gates(model, train.tokens, h, BATCH)
        shift = R.mean_change(mean1, m0, alive)
        on = R.switched(shift, THR)
        torch.save({"shift": shift, "shift_null": task["shift_null"]}, LARGE / f"{name}_s{seed}_shift.pt")
        gap, gain = cur["L"] - L00, cur["T"] - task["base"]["T"]
        d = patterns(ctx, task, cur, shift, mean1)
        rho = {p_: div(d[p_], gap) for p_ in PATTERNS} | {"mean_W0": div(lo_mean_w0, gap)}
        rnd = [div(x, gap) for x in d["random"]]
        s = {"p_heldout": cur["p_T"], "gain": gain, "gap": gap, "r": div(gap, gain), "rho": rho, "random": rnd,
             "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd), "shared_over_trig": div(rho["shared"], rho["trig"]),
             "n_on": R.size(on), "by_layer_on": R.by_layer(on), "scorable": gain >= MIN_GAIN and gap >= MIN_GAP}
        rec["seeds"][str(seed)] = s
        task[f"on{seed}"] = on
        log(f"[{name} s{seed}] r {s['r']:.4f} gain {gain:.3f} | rho trig {rho['trig']:.4f} shared {rho['shared']:.4f} "
            f"mean {rho['mean']:.4f} (W0 {rho['mean_W0']:.4f}) cross {rho['cross']:.4f} | random mean {s['random_mean']:.4f} "
            f"max {s['random_max']:.4f} | shared/trig {s['shared_over_trig']:.4f} | |S_on| {s['n_on']}")
    rec["J_seeds"] = R.jaccard(task["on0"], task["on1"])
    rec["J_null"] = R.jaccard(task["on0"], on_null)
    log(f"[{name}] J(s0, s1) {rec['J_seeds']:.4f} J(s0, null) {rec['J_null']:.4f} | null P(http) {p_null:.4f} |S_on null| {R.size(on_null)}")
    for k in ("base", "cur0", "dW0"):
        task.pop(k)
    return rec


def stage_natural(ctx: dict[str, Any]) -> dict[str, Any]:
    """W0's gate contrast at Pile positions whose next token is each task's target; J_nat against S_on."""
    from commute import recruit as R
    from commute.data import pile_rows
    from commute.signatures import top_by
    from commute.vpd import compute_ci

    rows = pile_rows(*NAT_ROWS, log=log)
    masks = {n: R.precede_mask(rows, t["target"]) for n, t in ctx["tasks"].items()}
    accs = {n: R.ContrastAcc() for n in masks}
    for s in range(0, rows.shape[0], 8):
        ci = compute_ci(ctx["model"], rows[s : s + 8], chunk=8)
        for n, acc in accs.items():
            acc.add(ci.g, masks[n][s : s + 8])
        del ci
    out: dict[str, Any] = {}
    for n, acc in accs.items():
        t, rec = ctx["tasks"][n], {"n_positions": acc.n_on, "scorable": acc.n_on >= MIN_NAT}
        contrast = acc.result(ctx["alive"])
        for seed in SEEDS:
            on = t[f"on{seed}"]
            nat = top_by(contrast, ctx["alive"], R.size(on))
            rnd = [R.jaccard(R.random_like(on, ctx["alive"], i), nat) for i in range(N_RANDOM)]
            rec[f"s{seed}"] = {"J": R.jaccard(on, nat), "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd)}
        out[n] = rec
        log(f"[natural {n}] {acc.n_on} positions | J_nat s0 {rec['s0']['J']:.4f} (random {rec['s0']['random_mean']:.4f}) "
            f"s1 {rec['s1']['J']:.4f} (random {rec['s1']['random_mean']:.4f})")
    return out


def stage_adversary(ctx: dict[str, Any]) -> dict[str, Any]:
    """Unscored: the share of S_on (seed 0) the Pile adversary turns down, at the state and at W0, vs all alive."""
    from commute import gamma
    from commute.data import batches, pile_rows

    rows = batches(pile_rows(*ADV_ROWS, log=log), 8)
    alive = ctx["alive"]

    def down_share(src: dict[str, Tensor], sel: dict[str, Tensor]) -> float:
        n = sum(int((src[k][:-1][v] < gamma.DOWN).sum()) for k, v in sel.items())
        return n / max(sum(int(v.numel()) for v in sel.values()), 1)

    all_alive = {k: v.cpu().bool().nonzero().flatten() for k, v in alive.items()}
    w0_src, w0_kl = gamma.pgd_sources(ctx["model"], rows, ADV_STEPS, seed=0)
    out: dict[str, Any] = {"W0": {"kl": w0_kl, "down_alive": down_share(w0_src, all_alive)}}
    for n, t in ctx["tasks"].items():
        dW = torch.load(COVERAGE / n / f"dW_ft_step{FT_STEPS}_s0.pt", weights_only=True)
        with state_of(ctx, dW):
            src, kl = gamma.pgd_sources(ctx["model"], rows, ADV_STEPS, seed=0)
        out[n] = {"kl": kl, "down_alive": down_share(src, all_alive), "down_S_on": down_share(src, t["on0"]),
                  "down_S_on_at_W0": down_share(w0_src, t["on0"])}
        log(f"[adversary {n}] KL {kl:.4f} (W0 {w0_kl:.4f}) | turned down: S_on {out[n]['down_S_on']:.3f} (W0's adversary "
            f"{out[n]['down_S_on_at_W0']:.3f}), all alive {out[n]['down_alive']:.3f} (W0 {out['W0']['down_alive']:.3f})")
    return out


# ----------------------------------------------------------------------------- scoring


def score(out: dict[str, Any]) -> dict[str, Any]:
    T = out["tasks"]
    per = [(n, s, v) for n, t in T.items() for s, v in t["seeds"].items() if v["scorable"]]
    if not per:
        return {"status": "unscored: no task and seed with gain >= 1 and re-gating gap >= 0.5 nats"}
    ratio = {f"{n}/s{s}": v["shared_over_trig"] for n, s, v in per}
    p2 = {f"{n}/s{s}": ("mechanism" if ratio[f"{n}/s{s}"] >= 0.7 else "leak" if ratio[f"{n}/s{s}"] <= 0.3 else "between")
          for n, s, _ in per}
    res: dict[str, Any] = {
        "scored_states": [f"{n}/s{s}" for n, s, _ in per],
        "P1_trigger_carries": {"rho_trig": {f"{n}/s{s}": v["rho"]["trig"] for n, s, v in per},
                               "pass": all(v["rho"]["trig"] >= 0.7 for _, _, v in per)},
        "P2_shared_pattern": {"shared_over_trig": ratio, "reading": p2, "random_mean": {f"{n}/s{s}": v["random_mean"] for n, s, v in per},
                              "pass": all(r >= 0.7 for r in ratio.values()) and all(v["random_mean"] <= 0.1 for _, _, v in per)},
        "P3_task_specific": {"rho_cross": {f"{n}/s{s}": v["rho"]["cross"] for n, s, v in per},
                             "pass": all(v["rho"]["cross"] <= 0.1 for _, _, v in per)},
    }
    nat = out.get("natural", {})
    if nat:
        p4: dict[str, Any] = {n: {"J": v["s0"]["J"], "random_mean": v["s0"]["random_mean"], "scorable": v["scorable"],
                                  "pass": v["s0"]["J"] >= 3 * v["s0"]["random_mean"]} for n, v in nat.items()}
        scorable = [v["pass"] for v in p4.values() if v["scorable"]]
        p4["pass"] = all(scorable) if scorable else "unscored: no task with enough natural positions"
        res["P4_natural_machinery"] = p4
    return res


# ----------------------------------------------------------------------------- amendment 1


def a1_gate(ctx: dict[str, Any], name: str, e18: dict[str, Any]) -> dict[str, Any]:
    """Per seed: the state's gap and E18's trig, shared and mean recoveries, against E18's values; the shift kept in the
    task for `a1_arms`. Plus noclip with a zero shift (must be 0)."""
    from commute import recruit as R
    from commute.local_faith import route_cells, trigger_mean_gates

    task, model, alive = ctx["tasks"][name], ctx["model"], ctx["alive"]
    train, held, h = task["train"], task["held"], task["half"]
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    task["m0"] = m0 = trigger_mean_gates(model, train.tokens, h, BATCH)
    zero = lo_of(ctx, task, R.with_trigger(g0, h, R.added(g0, h, {k: torch.zeros(v.shape) for k, v in m0.items()}))) - L00
    rec: dict[str, Any] = {"zero_noclip": div(zero, task["cur0"]["L"] - L00), "seeds": {}}
    for seed in SEEDS:
        dW = task["dW0"] if seed == 0 else finetune(ctx, task, task["target"], seed)[0]
        with state_of(ctx, dW):
            cur = task["cur0"] if seed == 0 else route_cells(model, held.tokens, h, task["target"], BATCH)
            mean1 = trigger_mean_gates(model, train.tokens, h, BATCH)
        shift, gap = R.mean_change(mean1, m0, alive), cur["L"] - L00
        pats = {"trig": R.at_trigger(cur["gates"], h), "shared": R.shifted(g0, h, shift), "mean": R.constant(g0, h, mean1)}
        rho = {p: div(lo_of(ctx, task, R.with_trigger(g0, h, t)) - L00, gap) for p, t in pats.items()}
        ref = {p: e18[str(seed)]["rho"][p] for p in rho}
        ok = all(abs(rho[p] - ref[p]) <= E18_TOL for p in ("shared", "mean"))
        task[f"shift{seed}"] = shift
        rec["seeds"][str(seed)] = {"gap": gap, "rho": rho, "rho_E18": ref, "ok": ok}
        log(f"[A1 gate {name} s{seed}] gap {gap:.3f} | rho trig {rho['trig']:.4f} (E18 {ref['trig']:.4f}) shared "
            f"{rho['shared']:.4f} (E18 {ref['shared']:.4f}) mean {rho['mean']:.4f} (E18 {ref['mean']:.4f}) | "
            f"{'ok' if ok else 'FAIL'}")
    rec["ok"] = all(v["ok"] for v in rec["seeds"].values()) and abs(rec["zero_noclip"]) <= ZERO_TOL
    log(f"[A1 gate {name}] noclip with s = 0: {rec['zero_noclip']:.2e} | {'ok' if rec['ok'] else 'FAIL'}")
    for k in ("cur0", "dW0"):
        task.pop(k)
    return rec


def a1_arms(ctx: dict[str, Any], name: str, rec: dict[str, Any]) -> None:
    """noclip, mean_s, S_on and mean_W0 per seed (into `rec`), and the path shared -> noclip -> mean_s -> mean as shares of
    D = R_mean - R_shared."""
    from commute import recruit as R

    task = ctx["tasks"][name]
    g0, h, L00, m0 = task["base"]["gates"], task["half"], task["base"]["L"], task["m0"]
    lo_w0 = lo_of(ctx, task, R.with_trigger(g0, h, R.constant(g0, h, m0))) - L00
    for seed, s in rec["seeds"].items():
        shift = task.pop(f"shift{seed}")
        on = R.switched(shift, THR)
        pats = {"noclip": R.added(g0, h, shift), "mean_s": R.constant(g0, h, {k: m0[k] + shift[k] for k in m0}),
                "S_on": R.shifted(g0, h, R.moved(shift, on, on))}  # moved(shift, on, on): the shift on S_on only
        rho = s["rho"] | {p: div(lo_of(ctx, task, R.with_trigger(g0, h, t)) - L00, s["gap"]) for p, t in pats.items()}
        rho["mean_W0"] = div(lo_w0, s["gap"])
        Rx = {p: div(v, rho["trig"]) for p, v in rho.items()}
        D = Rx["mean"] - Rx["shared"]
        f = {"clip": div(Rx["noclip"] - Rx["shared"], D), "variation": div(Rx["mean_s"] - Rx["noclip"], D),
             "dead": div(Rx["mean"] - Rx["mean_s"], D), "variation_W0": div(Rx["mean_W0"], D)}
        s.update(rho=rho, R=Rx, D=D, f=f, n_on=R.size(on), scorable=D >= MIN_D)
        log(f"[A1 {name} s{seed}] R shared {Rx['shared']:.4f} noclip {Rx['noclip']:.4f} mean_s {Rx['mean_s']:.4f} mean "
            f"{Rx['mean']:.4f} S_on {Rx['S_on']:.4f} mean_W0 {Rx['mean_W0']:.4f} | D {D:.4f} | f clip {f['clip']:.3f} "
            f"variation {f['variation']:.3f} (W0's {f['variation_W0']:.3f}) dead {f['dead']:.3f} | |S_on| {R.size(on)}")
    task.pop("base")


def a1_score(out: dict[str, Any]) -> dict[str, Any]:
    per = {f"{n}/s{s}": v for n, t in out["tasks"].items() for s, v in t["seeds"].items() if v["scorable"]}
    if not per:
        return {"status": f"unscored: no task-seed state with D >= {MIN_D}"}
    reading = {k: [t for t in ("clip", "variation", "dead") if v["f"][t] >= F_READ] or ["split"] for k, v in per.items()}
    return {"A1_P1_what_lowers_shared": {"f": {k: v["f"] for k, v in per.items()}, "reading": reading,
                                         "pass": all(r == ["variation"] for r in reading.values())}}


def run_amend1(device: str) -> None:
    if not PRED_A1.is_file():
        sys.exit(f"file the amendment's predictions first: {PRED_A1} (run.py --amend1 --predictions)")
    t0 = time.time()
    e18 = json.loads((RESULTS / OUT).read_text())["tasks"]
    ctx = context(device)
    out: dict[str, Any] = {"G1": {}, "tasks": {}}
    for name in TASKS:
        out["G1"][name] = stage_g1(ctx, name)
        out["tasks"][name] = a1_gate(ctx, name, e18[name]["seeds"])
        dump(OUT_A1, out)
    out["gate_pass"] = all(v["ok"] for v in out["G1"].values()) and all(v["ok"] for v in out["tasks"].values())
    if not out["gate_pass"]:
        out["status"] = "stopped at the gate"
        dump(OUT_A1, out)
        log(f"A1 gate FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for name in TASKS:
        a1_arms(ctx, name, out["tasks"][name])
        dump(OUT_A1, out)
    out["score"] = a1_score(out)
    log("A1 score " + json.dumps(_finite(out["score"]), default=_json))
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(OUT_A1, out)
    log(f"wrote {RESULTS / OUT_A1} ({out['seconds']:.0f} s)")


# ----------------------------------------------------------------------------- amendment 2


def recovery(ctx: dict[str, Any], task: dict[str, Any], trig: Any, gap: float) -> float:
    """rho of a trigger-only pattern: (lo - L00) / gap, W0's gates elsewhere."""
    from commute import recruit as R

    g0, h, L00 = task["base"]["gates"], task["half"], task["base"]["L"]
    return div(lo_of(ctx, task, R.with_trigger(g0, h, trig)) - L00, gap)


def a2_gate(ctx: dict[str, Any], name: str, a1: dict[str, Any]) -> dict[str, Any]:
    """Per seed: trig, shared and noclip against amendment 1's values to A2_TOL (the run is deterministic); the shift
    kept in the task for `a2_arms`."""
    from commute import recruit as R
    from commute.local_faith import route_cells, trigger_mean_gates

    task, model, alive = ctx["tasks"][name], ctx["model"], ctx["alive"]
    train, held, h = task["train"], task["held"], task["half"]
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    m0 = trigger_mean_gates(model, train.tokens, h, BATCH)
    rec: dict[str, Any] = {"seeds": {}}
    for seed in SEEDS:
        dW = task["dW0"] if seed == 0 else finetune(ctx, task, task["target"], seed)[0]
        with state_of(ctx, dW):
            cur = task["cur0"] if seed == 0 else route_cells(model, held.tokens, h, task["target"], BATCH)
            mean1 = trigger_mean_gates(model, train.tokens, h, BATCH)
        shift, gap = R.mean_change(mean1, m0, alive), cur["L"] - L00
        pats = {"trig": R.at_trigger(cur["gates"], h), "shared": R.shifted(g0, h, shift), "noclip": R.added(g0, h, shift)}
        rho = {p: recovery(ctx, task, t, gap) for p, t in pats.items()}
        ref = {p: a1[str(seed)]["rho"][p] for p in rho}
        ok = all(abs(rho[p] - ref[p]) <= A2_TOL for p in rho)
        task[f"shift{seed}"] = shift
        rec["seeds"][str(seed)] = {"gap": gap, "rho": rho, "rho_A1": ref, "ok": ok, "scorable": gap >= MIN_GAP}
        log(f"[A2 gate {name} s{seed}] gap {gap:.3f} | " + " ".join(f"{p} {rho[p]:.6f} (A1 {ref[p]:.6f})" for p in rho)
            + f" | {'ok' if ok else 'FAIL'}")
    rec["ok"] = all(v["ok"] for v in rec["seeds"].values())
    for k in ("cur0", "dW0"):
        task.pop(k)
    return rec


def spread(draws: list[float], unshuffled: float, trig: float) -> dict[str, Any]:
    """The null's draws as R (share of rho_trig): mean, sd, min, max, and the ratio of means to the unshuffled."""
    Rd = [div(x, trig) for x in draws]
    mean = sum(Rd) / len(Rd)
    sd = math.sqrt(sum((x - mean) ** 2 for x in Rd) / (len(Rd) - 1))
    R0 = div(unshuffled, trig)
    return {"R_draws": Rd, "R_mean": mean, "R_sd": sd, "R_min": min(Rd), "R_max": max(Rd), "R_unshuffled": R0,
            "ratio": div(mean, R0)}


def a2_bins(ctx: dict[str, Any], task: dict[str, Any], shift: dict[str, Tensor], s: dict[str, Any]) -> dict[str, Any]:
    """The no-clip shift restricted to each bin of its value (alive lo < s <= hi), as R."""
    from commute import recruit as R

    g0, h, out = task["base"]["gates"], task["half"], {}
    for lo, hi in itertools.pairwise(BIN_EDGES):
        sel = R.in_range(shift, ctx["alive"], lo, hi)
        rho = recovery(ctx, task, R.added(g0, h, R.moved(shift, sel, sel)), s["gap"])
        out[f"({lo}, {hi}]"] = {"n": R.size(sel), "sum_s": float(sum(float(shift[k][v].sum()) for k, v in sel.items())),
                                "R": div(rho, s["rho"]["trig"])}
    return out


def a2_arms(ctx: dict[str, Any], name: str, rec: dict[str, Any]) -> None:
    """Per seed (into `rec`): the shift shuffled over alive subcomponents, N_RANDOM draws, clipped and not; the bins."""
    from commute import recruit as R

    task = ctx["tasks"][name]
    g0, h = task["base"]["gates"], task["half"]
    for seed, s in rec["seeds"].items():
        shift = task.pop(f"shift{seed}")
        draws: dict[str, list[float]] = {"shared": [], "noclip": []}
        for i in range(N_RANDOM):
            sp = R.shuffled(shift, ctx["alive"], i)
            draws["shared"].append(recovery(ctx, task, R.shifted(g0, h, sp), s["gap"]))
            draws["noclip"].append(recovery(ctx, task, R.added(g0, h, sp), s["gap"]))
        s["perm"] = {p: spread(v, s["rho"][p], s["rho"]["trig"]) for p, v in draws.items()}
        s["bins"] = a2_bins(ctx, task, shift, s)
        for p, v in s["perm"].items():
            log(f"[A2 {name} s{seed} {p}_perm] R unshuffled {v['R_unshuffled']:.4f} | shuffled mean {v['R_mean']:.4f} sd "
                f"{v['R_sd']:.4f} min {v['R_min']:.4f} max {v['R_max']:.4f} | ratio {v['ratio']:.4f} | draws "
                + " ".join(f"{x:.4f}" for x in v["R_draws"]))
        bins = s["bins"]
        log(f"[A2 {name} s{seed} bins, noclip] " + " | ".join(f"{b} n {v['n']} sum {v['sum_s']:.2f} R {v['R']:.4f}"
                                                         for b, v in bins.items())
            + f" | sum of R {sum(v['R'] for v in bins.values()):.4f} (noclip {s['perm']['noclip']['R_unshuffled']:.4f})")
    task.pop("base")


def a2_reading(ratio: float) -> str:
    return "identity" if ratio <= IDENTITY else "bulk" if ratio >= BULK else "between"


def a2_score(out: dict[str, Any]) -> dict[str, Any]:
    per = {f"{n}/s{s}": v for n, t in out["tasks"].items() for s, v in t["seeds"].items() if v["scorable"]}
    if not per:
        return {"status": f"unscored: no task-seed state with gap >= {MIN_GAP}"}
    res = {}
    for pid, p in (("A2_P1_identity_shared", "shared"), ("A2_P2_identity_noclip", "noclip")):
        ratio = {k: v["perm"][p]["ratio"] for k, v in per.items()}
        reading = {k: a2_reading(r) for k, r in ratio.items()}
        res[pid] = {"ratio": ratio, "reading": reading, "pass": all(r == "identity" for r in reading.values())}
    return res


def run_amend2(device: str) -> None:
    if not PRED_A2.is_file():
        sys.exit(f"file the amendment's predictions first: {PRED_A2} (run.py --amend2 --predictions)")
    t0 = time.time()
    a1 = json.loads((RESULTS / OUT_A1).read_text())["tasks"]
    ctx = context(device)
    out: dict[str, Any] = {"G1": {}, "tasks": {}}
    for name in TASKS:
        out["G1"][name] = stage_g1(ctx, name)
        out["tasks"][name] = a2_gate(ctx, name, a1[name]["seeds"])
        dump(OUT_A2, out)
    out["gate_pass"] = all(v["ok"] for v in out["G1"].values()) and all(v["ok"] for v in out["tasks"].values())
    if not out["gate_pass"]:
        out["status"] = "stopped at the gate"
        dump(OUT_A2, out)
        log(f"A2 gate FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for name in TASKS:
        a2_arms(ctx, name, out["tasks"][name])
        dump(OUT_A2, out)
    out["score"] = a2_score(out)
    log("A2 score " + json.dumps(_finite(out["score"]), default=_json))
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(OUT_A2, out)
    log(f"wrote {RESULTS / OUT_A2} ({out['seconds']:.0f} s)")


# ----------------------------------------------------------------------------- amendment 3


def a3_gate(ctx: dict[str, Any], name: str, a2: dict[str, Any]) -> dict[str, Any]:
    """Per seed: trig, shared and noclip against amendment 2's values to A2_TOL; kept in the task for the arms: the
    state's mean trigger gates, the fixed change s, the held-out mean change s_held and the own trigger values g1."""
    from commute import recruit as R
    from commute.local_faith import route_cells, trigger_mean_gates

    task, model, alive = ctx["tasks"][name], ctx["model"], ctx["alive"]
    train, held, h = task["train"], task["held"], task["half"]
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    task["m0"] = m0 = trigger_mean_gates(model, train.tokens, h, BATCH)
    rec: dict[str, Any] = {"seeds": {}}
    for seed in SEEDS:
        dW = task["dW0"] if seed == 0 else finetune(ctx, task, task["target"], seed)[0]
        with state_of(ctx, dW):
            cur = task["cur0"] if seed == 0 else route_cells(model, held.tokens, h, task["target"], BATCH)
            mean1 = trigger_mean_gates(model, train.tokens, h, BATCH)
        shift, gap = R.mean_change(mean1, m0, alive), cur["L"] - L00
        g1 = [{k: v.clone() for k, v in t.items()} for t in R.at_trigger(cur["gates"], h)]
        pats = {"trig": g1, "shared": R.shifted(g0, h, shift), "noclip": R.added(g0, h, shift)}
        rho = {p: recovery(ctx, task, t, gap) for p, t in pats.items()}
        ref = {p: a2[str(seed)]["rho"][p] for p in rho}
        ok = all(abs(rho[p] - ref[p]) <= A2_TOL for p in rho)
        task[f"st{seed}"] = {"shift": shift, "mean1": mean1, "g1": g1,
                             "s_held": R.trigger_dg(cur["gates"], g0, h, alive)}
        rec["seeds"][str(seed)] = {"gap": gap, "rho": rho, "rho_A2": ref, "ok": ok, "scorable": gap >= MIN_GAP}
        log(f"[A3 gate {name} s{seed}] gap {gap:.3f} | " + " ".join(f"{p} {rho[p]:.6f} (A2 {ref[p]:.6f})" for p in rho)
            + f" | {'ok' if ok else 'FAIL'}")
    rec["ok"] = all(v["ok"] for v in rec["seeds"].values())
    for k in ("cur0", "dW0"):
        task.pop(k)
    return rec


def a3_core(ctx: dict[str, Any], task: dict[str, Any], st: dict[str, Any], gap: float) -> dict[str, float]:
    """rho of the mean patterns (amendment 1's, for the grid) and of the residuals: own values minus the fixed change."""
    from commute import recruit as R

    g0, h = task["base"]["gates"], task["half"]
    minus_s = {k: -v for k, v in st["shift"].items()}
    pats = {"mean": R.constant(g0, h, st["mean1"]), "mean_W0": R.constant(g0, h, task["m0"]),
            "resid": R.plus(st["g1"], minus_s, clip=True), "resid_noclip": R.plus(st["g1"], minus_s, clip=False),
            "resid_held": R.plus(st["g1"], {k: -v for k, v in st["s_held"].items()}, clip=True)}
    return {p: recovery(ctx, task, t, gap) for p, t in pats.items()}


def corr(x: Tensor, y: Tensor) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    return div(float((xc * yc).sum()), float(xc.norm() * yc.norm()))


def a3_mismatch(ctx: dict[str, Any], task: dict[str, Any], st: dict[str, Any], gap: float) -> dict[str, Any]:
    """Each held-out window given another window's own values (N_DERANGE derangements): rho, its W0 reference, and
    whether the per-window masked gain follows the window (c_own) or the donor of the values (c_donor)."""
    from commute import recruit as R
    from commute.local_faith import masked_lo_windows

    g0, h, held = task["base"]["gates"], task["half"], task["held"]

    def gain(trig: Any) -> Tensor:
        return masked_lo_windows(ctx["model"], held.tokens, h, task["target"], R.with_trigger(g0, h, trig), BATCH) - lo00

    lo00 = masked_lo_windows(ctx["model"], held.tokens, h, task["target"], g0, BATCH)
    a = gain(st["g1"])
    draws = []
    for i in range(N_DERANGE):
        p = R.derangement(len(held), i)
        b = gain(R.permuted(st["g1"], p))
        draws.append({"rho": div(float(b.mean()), gap), "rho_W0": recovery(ctx, task, R.permuted(R.at_trigger(g0, h), p), gap),
                      "c_own": corr(b, a), "c_donor": corr(b, a[p]), "c_null": corr(a, a[p])})
    mean = {k: sum(d[k] for d in draws) / len(draws) for k in draws[0]}
    return {"draws": draws, **mean, "rho_trig_per_window": div(float(a.mean()), gap), "sd_gain_nats": float(a.std())}


def a3_halves(ctx: dict[str, Any], task: dict[str, Any], st: dict[str, Any], gap: float) -> dict[str, Any]:
    """The turned-up and turned-down halves of s, clipped and not; each against a shuffle that keeps its sign: s+ over
    the alive subcomponents, s- (on the pool P of subcomponents with m0 >= POOL_MIN) over P."""
    from commute import recruit as R

    g0, h, alive = task["base"]["gates"], task["half"], ctx["alive"]
    shift = st["shift"]
    pool = {k: alive[k] & (task["m0"][k] >= POOL_MIN) for k in alive}
    up, down = R.signed(shift, True), R.signed(shift, False)
    down_p = R.on_pool(down, pool)
    rho = {"up": recovery(ctx, task, R.shifted(g0, h, up), gap), "down": recovery(ctx, task, R.shifted(g0, h, down), gap),
           "up_noclip": recovery(ctx, task, R.added(g0, h, up), gap),
           "down_noclip": recovery(ctx, task, R.added(g0, h, down), gap),
           "down_pool": recovery(ctx, task, R.shifted(g0, h, down_p), gap)}
    draws = {"up": [recovery(ctx, task, R.shifted(g0, h, R.shuffled(up, alive, i)), gap) for i in range(N_RANDOM)],
             "down_pool": [recovery(ctx, task, R.shifted(g0, h, R.shuffled(down_p, pool, i)), gap) for i in range(N_RANDOM)]}
    strong = R.in_range(shift, alive, -math.inf, -STRONG)
    return {"rho": rho, "draws": draws, "n_pool": R.size({k: v.nonzero().flatten() for k, v in pool.items()}),
            "n_down_strong": R.size(strong), "n_up": R.size(R.in_range(shift, alive, 0.0, math.inf)),
            "n_down": R.size(R.in_range(shift, alive, -math.inf, -1e-12))}


def a3_pile(ctx: dict[str, Any], task: dict[str, Any], shifts: dict[str, dict[str, Tensor]]) -> dict[str, Any]:
    """The fixed change, clipped, at position `half` of Pile rows (W0's gates elsewhere): the target's masked log-odds
    change there, per seed (nats)."""
    from commute import recruit as R
    from commute.data import pile_rows
    from commute.local_faith import masked_lo, route_cells

    if "pile" not in ctx:
        ctx["pile"] = pile_rows(*ADV_ROWS, log=log)
    rows, h, model = ctx["pile"][:, : task["held"].tokens.shape[1]], task["half"], ctx["model"]
    base = route_cells(model, rows, h, task["target"], BATCH)
    g = base["gates"]
    d = {s: masked_lo(model, rows, h, task["target"], R.with_trigger(g, h, R.shifted(g, h, sh)), BATCH)[0] - base["L"]
         for s, sh in shifts.items()}
    return {"lo00": base["L"], "p00": base["p_L"], "d_nats": d}


def a3_state(s: dict[str, Any], core: dict[str, float], mis: dict[str, Any], halves: dict[str, Any]) -> None:
    """Fold one state's arms into its record `s` as shares R of rho_trig, with the derived numbers the predictions read."""
    trig = s["rho"]["trig"]
    s["rho"] |= core | {f"half_{k}": v for k, v in halves["rho"].items()}
    Rx = {p: div(v, trig) for p, v in s["rho"].items()}
    s["R"] = Rx
    s["grid"] = {"none": Rx["mean"] - Rx["mean_W0"], "W0_variation": Rx["noclip"], "state_variation": 1 - Rx["resid_noclip"]}
    s["mismatch"] = mis | {"R": div(mis["rho"], trig), "R_W0": div(mis["rho_W0"], trig)}
    s["halves"] = {k: v for k, v in halves.items() if k not in ("rho", "draws")}
    s["halves"]["perm"] = {"up": spread(halves["draws"]["up"], halves["rho"]["up"], trig),
                           "down_pool": spread(halves["draws"]["down_pool"], halves["rho"]["down_pool"], trig)}
    s["halves"]["survival"] = div(Rx["half_down"], Rx["half_down_noclip"])
    s["halves"]["sum_clip"] = Rx["half_up"] + Rx["half_down"]
    s["halves"]["sum_noclip"] = Rx["half_up_noclip"] + Rx["half_down_noclip"]


def a3_log(name: str, seed: str, s: dict[str, Any]) -> None:
    Rx, m, hv = s["R"], s["mismatch"], s["halves"]
    log(f"[A3 {name} s{seed}] R resid {Rx['resid']:.4f} resid_noclip {Rx['resid_noclip']:.4f} resid_held "
        f"{Rx['resid_held']:.4f} | mean {Rx['mean']:.4f} mean_W0 {Rx['mean_W0']:.4f} | grid none {s['grid']['none']:.4f} "
        f"W0_var {s['grid']['W0_variation']:.4f} state_var {s['grid']['state_variation']:.4f}")
    log(f"[A3 {name} s{seed} mismatch] R {m['R']:.4f} (W0 {m['R_W0']:.4f}) | c_own {m['c_own']:.4f} c_donor "
        f"{m['c_donor']:.4f} c_null {m['c_null']:.4f} | sd gain {m['sd_gain_nats']:.3f} nats | per-window rho_trig "
        f"{m['rho_trig_per_window']:.6f} (masked_lo {s['rho']['trig']:.6f}) | draws " + " ; ".join(
            f"R {div(d['rho'], s['rho']['trig']):.4f} W0 {div(d['rho_W0'], s['rho']['trig']):.4f} own {d['c_own']:.3f} "
            f"donor {d['c_donor']:.3f} null {d['c_null']:.3f}" for d in m["draws"]))
    log(f"[A3 {name} s{seed} halves] up {Rx['half_up']:.4f} (noclip {Rx['half_up_noclip']:.4f}) down {Rx['half_down']:.4f} "
        f"(noclip {Rx['half_down_noclip']:.4f}, survival {hv['survival']:.4f}) down_pool {Rx['half_down_pool']:.4f} | "
        f"sum clip {hv['sum_clip']:.4f} (shared {Rx['shared']:.4f}) noclip {hv['sum_noclip']:.4f} (noclip "
        f"{Rx['noclip']:.4f}) | n up {hv['n_up']} down {hv['n_down']} strong-down {hv['n_down_strong']} pool {hv['n_pool']}")
    for p, v in hv["perm"].items():
        log(f"[A3 {name} s{seed} {p}_perm] R unshuffled {v['R_unshuffled']:.4f} | shuffled mean {v['R_mean']:.4f} sd "
            f"{v['R_sd']:.4f} min {v['R_min']:.4f} max {v['R_max']:.4f} | ratio {v['ratio']:.4f} | draws "
            + " ".join(f"{x:.4f}" for x in v["R_draws"]))


def a3_arms(ctx: dict[str, Any], name: str, rec: dict[str, Any]) -> None:
    task = ctx["tasks"][name]
    shifts = {}
    for seed, s in rec["seeds"].items():
        st = task.pop(f"st{seed}")
        core = a3_core(ctx, task, st, s["gap"])
        mis = a3_mismatch(ctx, task, st, s["gap"])
        halves = a3_halves(ctx, task, st, s["gap"])
        a3_state(s, core, mis, halves)
        a3_log(name, seed, s)
        shifts[seed] = st["shift"]
    pile = a3_pile(ctx, task, shifts)
    for seed, s in rec["seeds"].items():
        s["pile"] = {"d_nats": pile["d_nats"][seed], "lo00": pile["lo00"], "p00": pile["p00"],
                     "ratio": div(pile["d_nats"][seed], s["rho"]["shared"] * s["gap"])}
        log(f"[A3 {name} s{seed} pile] W0 masked lo {pile['lo00']:.3f} (P {pile['p00']:.4f}) | with s +{s['pile']['d_nats']:.3f}"
            f" nats vs +{s['rho']['shared'] * s['gap']:.3f} at the trigger | ratio {s['pile']['ratio']:.4f}")
    task.pop("base")


def a3_read(x: float, lo: float, hi: float, low: str, high: str) -> str:
    return low if x <= lo else high if x >= hi else "between"


def a3_score(out: dict[str, Any]) -> dict[str, Any]:
    per = {f"{n}/s{s}": v for n, t in out["tasks"].items() for s, v in t["seeds"].items() if v["scorable"]}
    if not per:
        return {"status": f"unscored: no task-seed state with gap >= {MIN_GAP}"}
    p1 = {k: a3_read(v["R"]["resid"], H_BAR, LEAK_BAR, "H", "leak") for k, v in per.items()}
    p2 = {k: a3_read(v["mismatch"]["R"], MIS_DROP, MIS_SAME, "own window needed", "H") for k, v in per.items()
          if abs(v["mismatch"]["R_W0"]) <= MIS_W0_MAX}
    d = {k: v["mismatch"]["c_donor"] - v["mismatch"]["c_own"] for k, v in per.items()}
    p3 = {k: "donor" if x >= CORR_MARGIN else "recipient" if x <= -CORR_MARGIN else "neither" for k, x in d.items()}
    res: dict[str, Any] = {
        "A3_P1_fixed_part_needed": {"R_resid": {k: v["R"]["resid"] for k, v in per.items()}, "reading": p1,
                                    "pass": all(r == "H" for r in p1.values())},
        "A3_P2_other_windows_values": {"R_mismatch": {k: v["mismatch"]["R"] for k, v in per.items()},
                                       "R_mismatch_W0": {k: v["mismatch"]["R_W0"] for k, v in per.items()}, "reading": p2,
                                       "pass": all(r == "H" for r in p2.values()) if p2 else "unscored: |R_W0| too large"},
        "A3_P3_gain_follows_recipient": {"c_donor_minus_c_own": d, "reading": p3,
                                         "pass": all(r != "donor" for r in p3.values())},
    }
    for pid, half, unshuf in (("A3_P4_down_identity", "down_pool", "half_down_pool"), ("A3_P5_up_identity", "up", "half_up")):
        ratio = {k: v["halves"]["perm"][half]["ratio"] for k, v in per.items() if v["R"][unshuf] >= MIN_HALF}
        reading = {k: a2_reading(r) for k, r in ratio.items()}
        res[pid] = {"ratio": ratio, "reading": reading,
                    "pass": all(r == "identity" for r in reading.values()) if ratio else f"unscored: R < {MIN_HALF}"}
    return res


def run_amend3(device: str) -> None:
    if not PRED_A3.is_file():
        sys.exit(f"file the amendment's predictions first: {PRED_A3} (run.py --amend3 --predictions)")
    t0 = time.time()
    a2 = json.loads((RESULTS / OUT_A2).read_text())["tasks"]
    ctx = context(device)
    out: dict[str, Any] = {"G1": {}, "tasks": {}}
    for name in TASKS:
        out["G1"][name] = stage_g1(ctx, name)
        out["tasks"][name] = a3_gate(ctx, name, a2[name]["seeds"])
        dump(OUT_A3, out)
    out["gate_pass"] = all(v["ok"] for v in out["G1"].values()) and all(v["ok"] for v in out["tasks"].values())
    if not out["gate_pass"]:
        out["status"] = "stopped at the gate"
        dump(OUT_A3, out)
        log(f"A3 gate FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for name in TASKS:
        a3_arms(ctx, name, out["tasks"][name])
        dump(OUT_A3, out)
    out["score"] = a3_score(out)
    log("A3 score " + json.dumps(_finite(out["score"]), default=_json))
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(OUT_A3, out)
    log(f"wrote {RESULTS / OUT_A3} ({out['seconds']:.0f} s)")


# ----------------------------------------------------------------------------- amendment 4 (2026-10-22)
PRED_A4 = RESULTS / "predictions_amend4.json"
OUT_A4 = "regating_leak_amend4.json"
SAME_INFO, OWN_NEEDED = 0.8, 0.5  # D = R(donor remainder) / R(own remainder): no window information if >= SAME_INFO
PREDICTIONS_A4: dict[str, Any] = {
    "filed": "2026-10-22, before the amendment ran; earlier predictions stand unedited (A3_P1 failed as filed). Plan: "
             "reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 4; user's choice B, 2026-10-22",
    "question": "amendment 3's remainder clip(g1(x) - s) recovers 0.16 to 0.31 of the switching effect (A3_P1 failed). "
                "Does it carry anything specific to window x? If not, the window it comes from does not matter",
    "notation": "as amendment 3: g1(x) the state's gates at the trigger of held-out window x, s the fixed change, pi a "
                "derangement of the held-out windows (recruit.derangement seeds 0..4), R = rho / rho_trig",
    "arms": {"resid, resid_noclip, mean_W0": "amendment 3's own remainder and W0's mean gates, repeated (the gate)",
             "resid_donor": "clip(g1(pi(x)) - s): window x gets another window's remainder; mean over 5 derangements",
             "resid_donor_noclip": "g1(pi(x)) - s, unclipped"},
    "measure": "D = R_resid_donor / R_resid, per task and seed (relative); also reported: R_resid_donor - R_mean_W0",
    "expected": "D near 1 if the remainder carries no window information. Reference values: amendment 3's mismatched W0 "
                "gates g0(pi(x)) recover 0.01 to 0.06 (what mismatching costs with no update), its mismatched own values "
                "0.89 to 1.00",
    "blind_to": "information that a derangement keeps (windows alike, emoticon's gain spread 1.2 nats); whether the "
                "remainder works with the fixed change present (only alone); the clip (unclipped reported)",
    "noise": "5 derangements; their spread is reported; amendment 3's donor spread was a few hundredths",
    "gates": [{"id": "A4_gate", "rule": f"G1, amendment 2's trig/shared/noclip (a3_gate) and amendment 3's resid, "
                                        f"resid_noclip and mean_W0 reproduced within {A2_TOL}", "if_fail": "stop"}],
    "predictions": [
        {"id": "A4_P1_remainder_not_window_specific", "scored": True,
         "pass": f"D >= {SAME_INFO} on every scorable task-seed state (gap >= {MIN_GAP})",
         "null": f"D <= {OWN_NEEDED} on some state: the window's own remainder carries window-specific information, a "
                 "leak in part (c) of H",
         "between": f"{OWN_NEEDED} < D < {SAME_INFO}: some window information",
         "predicted": "pass, moderate confidence: another window's whole values recover 0.89 to 1.00, and the remainder "
                      "beats W0's mean gates by 0.00 to 0.07 only"},
    ],
}


def a4_gate(ctx: dict[str, Any], name: str, a2: dict[str, Any], a3: dict[str, Any]) -> dict[str, Any]:
    """a3_gate, then amendment 3's remainder arms repeated per seed; the seed states stay in the task for the arms."""
    rec = a3_gate(ctx, name, a2)
    task = ctx["tasks"][name]
    for seed, s in rec["seeds"].items():
        core = a3_core(ctx, task, task[f"st{seed}"], s["gap"])
        rho = {p: core[p] for p in ("resid", "resid_noclip", "mean_W0")}
        ref = {p: a3[seed]["rho"][p] for p in rho}
        s["rho"] |= rho
        s["ok"] = s["ok"] and all(abs(rho[p] - ref[p]) <= A2_TOL for p in rho)
        log(f"[A4 gate {name} s{seed}] " + " ".join(f"{p} {rho[p]:.6f} (A3 {ref[p]:.6f})" for p in rho)
            + f" | {'ok' if s['ok'] else 'FAIL'}")
    rec["ok"] = all(v["ok"] for v in rec["seeds"].values())
    return rec


def a4_arms(ctx: dict[str, Any], name: str, rec: dict[str, Any]) -> None:
    """Each window given another window's remainder, clipped and not, over N_DERANGE derangements."""
    from commute import recruit as R

    task = ctx["tasks"][name]
    for seed, s in rec["seeds"].items():
        st = task.pop(f"st{seed}")
        minus_s = {k: -v for k, v in st["shift"].items()}
        draws = {"resid_donor": [], "resid_donor_noclip": []}
        for i in range(N_DERANGE):
            donor = R.permuted(st["g1"], R.derangement(len(task["held"]), i))
            draws["resid_donor"].append(recovery(ctx, task, R.plus(donor, minus_s, clip=True), s["gap"]))
            draws["resid_donor_noclip"].append(recovery(ctx, task, R.plus(donor, minus_s, clip=False), s["gap"]))
        trig = s["rho"]["trig"]
        s["R_draws"] = {p: [div(x, trig) for x in v] for p, v in draws.items()}
        s["rho"] |= {p: sum(v) / len(v) for p, v in draws.items()}
        s["R"] = Rx = {p: div(v, trig) for p, v in s["rho"].items()}
        s["D"] = div(Rx["resid_donor"], Rx["resid"])
        s["D_noclip"] = div(Rx["resid_donor_noclip"], Rx["resid_noclip"])
        log(f"[A4 {name} s{seed}] R resid {Rx['resid']:.4f} donor {Rx['resid_donor']:.4f} (draws "
            + " ".join(f"{x:.4f}" for x in s["R_draws"]["resid_donor"]) + f") D {s['D']:.4f} | noclip "
            f"{Rx['resid_noclip']:.4f} donor {Rx['resid_donor_noclip']:.4f} D {s['D_noclip']:.4f} | mean_W0 "
            f"{Rx['mean_W0']:.4f}")
    task.pop("base")


def a4_score(out: dict[str, Any]) -> dict[str, Any]:
    per = {f"{n}/s{s}": v for n, t in out["tasks"].items() for s, v in t["seeds"].items() if v["scorable"]}
    reading = {k: a3_read(v["D"], OWN_NEEDED, SAME_INFO, "own remainder needed", "no window information")
               for k, v in per.items()}
    return {"A4_P1_remainder_not_window_specific": {
        "D": {k: v["D"] for k, v in per.items()}, "D_noclip": {k: v["D_noclip"] for k, v in per.items()},
        "reading": reading, "pass": all(r == "no window information" for r in reading.values()) if per else "unscored"}}


def run_amend4(device: str) -> None:
    if not PRED_A4.is_file():
        sys.exit(f"file the amendment's predictions first: {PRED_A4} (run.py --amend4 --predictions)")
    t0 = time.time()
    a2 = json.loads((RESULTS / OUT_A2).read_text())["tasks"]
    a3 = json.loads((RESULTS / OUT_A3).read_text())["tasks"]
    ctx = context(device)
    out: dict[str, Any] = {"G1": {}, "tasks": {}}
    for name in TASKS:
        out["G1"][name] = stage_g1(ctx, name)
        out["tasks"][name] = a4_gate(ctx, name, a2[name]["seeds"], a3[name]["seeds"])
        dump(OUT_A4, out)
        if not (out["G1"][name]["ok"] and out["tasks"][name]["ok"]):
            out["status"] = "stopped at the gate"
            dump(OUT_A4, out)
            log(f"A4 gate FAILED on {name}: stopping ({time.time() - t0:.0f} s)")
            return
        a4_arms(ctx, name, out["tasks"][name])
        dump(OUT_A4, out)
    out["score"] = a4_score(out)
    log("A4 score " + json.dumps(_finite(out["score"]), default=_json))
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(OUT_A4, out)
    log(f"wrote {RESULTS / OUT_A4} ({out['seconds']:.0f} s)")


# ----------------------------------------------------------------------------- main


def context(device: str) -> dict[str, Any]:
    from commute import vpd

    model = vpd.load_component_model(device)
    alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
    return {"device": device, "model": model, "W0": vpd.target_weights(model), "tasks": {},
            "alive": {k: v.cpu().bool() for k, v in alive.items()}}


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    t0 = time.time()
    ctx = context(device)
    out: dict[str, Any] = {"G1": {}, "tasks": {}}
    for name in TASKS:
        out["G1"][name] = stage_g1(ctx, name)
        dump(OUT, out)
    out["G1_pass"] = all(v["ok"] for v in out["G1"].values())
    if not out["G1_pass"]:
        out["status"] = "stopped at G1"
        dump(OUT, out)
        log(f"G1 FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for name in TASKS:
        out["tasks"][name] = stage_task(ctx, name)
        out["score"] = score(out)
        dump(OUT, out)
    out["natural"] = stage_natural(ctx)
    out["score"] = score(out)
    dump(OUT, out)
    log("score " + json.dumps(_finite(out["score"]), default=_json))
    out["adversary"] = stage_adversary(ctx)
    out["seconds"] = time.time() - t0
    out["status"] = "done"
    dump(OUT, out)
    log(f"wrote {RESULTS / OUT} ({out['seconds']:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file the predictions (of the chosen amendment) and exit")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--amend1", action="store_true", help="amendment 1 instead of the scored E18 run")
    which.add_argument("--amend2", action="store_true", help="amendment 2 instead of the scored E18 run")
    which.add_argument("--amend3", action="store_true", help="amendment 3 instead of the scored E18 run")
    which.add_argument("--amend4", action="store_true", help="amendment 4 instead of the scored E18 run")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    path, preds, runner = ((PRED_A1, PREDICTIONS_A1, run_amend1) if args.amend1 else
                           (PRED_A2, PREDICTIONS_A2, run_amend2) if args.amend2 else
                           (PRED_A3, PREDICTIONS_A3, run_amend3) if args.amend3 else
                           (PRED_A4, PREDICTIONS_A4, run_amend4) if args.amend4 else (PRED, PREDICTIONS, run))
    if args.predictions:
        if path.is_file():
            sys.exit(f"{path} exists; filed predictions are never edited")
        dump(path.name, preds)
        print(f"filed {path}")
        return
    runner(args.device)


if __name__ == "__main__":
    main()
