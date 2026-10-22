"""cancellation_complement (E20): is the adversary's excess in the original subcomponents? E17's revert with the roles
swapped. Plan: reports/2026-10-21_plan_cancellation-complement.md (approved classes 1 and 4).

Per state (E17's eight, rebuilt by `commute.saved_states`) and at W0, on the first 128 Pile val rows: the state's own
adversary (gamma.pgd_sources, as E17), its fp32 masked KL F_all, and `cancel.Remover` reverts of the adversary's fraction
w = (1 - g)(1 - a) of the alive originals with the changes kept (F_orig), of the changes with the originals kept (F_adv,
E17's), of both (F_both), F_orig under permuted a (F_orig_perm), F_orig per importance quartile and per layer, and the
mean removed fraction per bin. X = F_all(s) - F_all(W0) = O + F_adv + I + R exactly (`decompose`).
Gate G1 (per state, W0 first): the hooks at the adversary's fractions of every block reproduce its fp32 masked forward.
Control C0 (component training emoticon, second): F_adv and the adversary's KL reproduce E17's.

  run.py --predictions   file results/predictions.json (once)
  run.py all             -> results/complement.json (written after every state), results/large/full.log
Amendment 1 (plan -> Amendment 1): the same run with R split into the dead block and Delta' alone, together, and at W0's
adversary levels, plus the levels themselves; gate G2 is E20's F_all, F_both, F_orig reproduced within 1e-3 relative.
  run.py --predictions amend1   file results/predictions_amend1.json (once)
  run.py amend1          -> results/complement_amend1.json (after every model), results/large/amend1.log, the
                            adversary's sources per model in results/large/sources/
Amendment 2 (plan -> Amendment 2): W0's saved levels for every block at each model, and a stronger adversary (more
steps, restarts, the alive-whole start) at W0 and r1 t534.
  run.py --predictions amend2_fixed   file results/predictions_amend2.json (once, both stages)
  run.py amend2_fixed    -> results/complement_amend2_fixed.json, results/large/amend2_fixed.log
  run.py amend2_strong   -> results/complement_amend2_strong.json, results/large/amend2_strong.log, the levels at every
                            checkpoint in results/large/sources_amend2/
  Both read amendment 1's complement_amend1.json and results/large/sources/.
Amendment 3 (plan -> Amendment 3): the adversary at budgets of 20, 40, 80, 160 sign steps with the step size 2 / steps
(the VPD paper's n_adv * lr_adv = 2), from a random start, at W0 and five states; the ratio to W0 per budget.
  run.py --predictions amend3_budget   file results/predictions_amend3.json (once)
  run.py amend3_budget --shard N        -> results/complement_amend3_shard{N}.json, results/large/amend3_shard{N}.log,
                                           the levels in results/large/sources_amend3/; shard 0 (W0) runs first, alone
  run.py amend3_score                   merges the shard files -> results/complement_amend3.json (JSON only, no model)
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}; E11's results/large/<task>/ (cluster NFS);
experiments/cancellation/results/cancellation.json (E17, in git).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
OUT = RESULTS / "complement.json"
PRED_A1 = RESULTS / "predictions_amend1.json"
OUT_A1 = RESULTS / "complement_amend1.json"
LOG = LARGE / "full.log"  # amend1 sets LARGE / "amend1.log"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
E11 = HERE.parent / "reading_a_coverage" / "results" / "large"
E17 = HERE.parent / "cancellation" / "results" / "cancellation.json"
EMOTICON = "emoticon"
CONTROL = f"ct {EMOTICON}"
FAILING = tuple(f"ct {t}" for t in ("r0_t352_v310", "r1_t534_v369", "r2_t275_v2667"))
SCORED = (*FAILING, f"ls {EMOTICON}")
OTHERS = ("ct r1_t588_v452", "ct r2_t359_v816", f"adjoint {EMOTICON}")
ORDER = (CONTROL, *SCORED, *OTHERS)  # after W0
ROWS, BATCH, PGD_STEPS, N_PERMS, N_BINS = 128, 8, 20, 3, 4
TOL, STEP = 1e-6, 10  # the split's SVD tolerance; E11's scored step
G1_TOL, ZERO_TOL, C0_TOL = 1e-4, 1e-6, 0.25
G2_TOL, JOINT, ALONE_FIX, LEVEL_TOL = 1e-3, 0.5, 0.5, 0.1  # amendment 1's gate and thresholds
PRED_A2 = RESULTS / "predictions_amend2.json"
OUT_A2 = {"fixed": RESULTS / "complement_amend2_fixed.json", "strong": RESULTS / "complement_amend2_strong.json"}
SOURCES = LARGE / "sources"  # amendment 1's adversary levels per model (20 steps, seed 0)
SAME_MASK, OTHER_MASK, FLOOR, SEARCH_SHARE = 0.5, 0.2, 1.0, 0.6  # amendment 2's thresholds
STRONG_MODELS = ("W0", FAILING[1])  # W0 and r1 t534
RUNS = {"seed0": (0, False, (20, 50, 100)), "alive_whole": (0, True, (0, 20, 50)),  # name: seed, alive-whole start,
        "seed1": (1, False, (20,)), "seed2": (2, False, (20,))}                    # checkpoints (cumulative steps)
BUDGETS = (20, 40, 80, 160)  # amendment 3: sign steps, each run from the seed's random start with step size 2 / steps
AMEND3_SHARDS = ((("W0", 0), ("W0", 1)), ((FAILING[1], 0), (FAILING[0], 0)), ((FAILING[2], 0), (f"ls {EMOTICON}", 0)),
                 ((f"adjoint {EMOTICON}", 0),))  # (model, seed); shard 0 is the control and runs first
STAY_FAILING, STAY_PASSING, STABLE = 1.5, 1.241, 1.5  # amendment 3's thresholds (ratios to W0 at the same budget)
PRED_A3 = RESULTS / "predictions_amend3.json"
OUT_A3 = RESULTS / "complement_amend3.json"

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-21",
    "design": "reports/2026-10-21_plan_cancellation-complement.md (approved classes 1 and 4; the named next step of "
              "reports/2026-10-21_analysis_cancellation.md)",
    "setup": {
        "states": f"W0 and E17's eight: component training at step {STEP} (seed 0) on emoticon, r0_t352_v310, r1_t534_v369, "
                  "r1_t588_v452, r2_t275_v2667, r2_t359_v816; the least-squares and scaled adjoint splits of the emoticon "
                  "fine-tune's delta-W (commute.saved_states, as E17)",
        "rows": f"Pile val rows 0..{ROWS} in batches of {BATCH}; every KL fp32; the adversary is gamma.pgd_sources "
                f"({PGD_STEPS} steps of 0.1, seed 0, one source per matrix over all rows) at W0 and at each state",
        "numbers": "w = (1 - g)(1 - a) per alive subcomponent and position. F_all = KL(state || fp32 masked forward); "
                   "X = F_all(s) - F_all(W0). Reverts of w (cancel.Remover): F_orig (alive originals, changes kept), F_adv "
                   "(changes, originals kept), F_both (new pieces); O = F_orig(s) - F_orig(W0), I = F_both - F_orig - F_adv, "
                   "R = [F_all - F_both](s) - [F_all - F_orig](W0), so X = O + F_adv + I + R exactly. F_orig_perm: a "
                   f"permuted among each matrix's alive subcomponents, {N_PERMS} draws. e_b = F_orig_b(s) - F_orig_b(W0) for "
                   f"{N_BINS} equal-count bins of W0's mean importance (round-1 alive.pt, imp0 lowest) and per layer; "
                   "removed_b = mean w per bin",
        "blind_to": "Pile rows only; KL is not additive (I, R and cross-bin effects absorb it); importance bins are W0's; "
                    "R lumps the dead subcomponents, Delta' and their interaction; each model under its own adversary; one "
                    "adversary seed",
        "noise": f"one adversary seed (E11 f = 0.075, about 0.05 nats at W0); spread over {N_PERMS} permutations; C0's gap "
                 "to E17; per-bin differences under 0.05 nats are noise",
    },
    "gates": [
        {"id": "G1_closure", "rule": f"first batch, W0 first then every state: KL(fp32 masked forward || hooks at the "
                                     f"adversary's fractions of alive new pieces, dead subcomponents and Delta') <= {G1_TOL}; "
                                     f"hooks at zero weight give KL <= {ZERO_TOL}; at W0 also F_adv <= {ZERO_TOL} and "
                                     f"|F_both - F_orig| <= {ZERO_TOL}", "if_fail": "stop"},
        {"id": "C0_reproduces_E17", "arm": CONTROL, "rule": f"F_adv and the bf16 adversary KL within {C0_TOL:.0%} of E17's "
                                                            "(0.00149, 0.7286)", "if_fail": "stop: rows or code differ from E17"},
    ],
    "predictions": [
        {"id": "P1_excess_in_originals", "arm": list(SCORED), "scored": True, "pass": "O >= 0.5 X on every scored state",
         "null": "O < 0.5 X: the excess sits in the interaction I or in R (dead subcomponents, Delta')",
         "predicted": "pass, moderate confidence: F_adv <= 0.07 X there and the changes are small beside the originals",
         "blind_to": "which originals; R is a remainder"},
        {"id": "P2_pattern_matters", "arm": list(SCORED), "scored": True, "pass": "F_orig >= 2 x mean F_orig_perm on every scored state",
         "null": "F_orig about F_orig_perm: any removal of that size does the damage",
         "predicted": "pass: turning every maskable subcomponent down together costs a fiftieth of the adversary's mask "
                      "(reports/2026-10-20_analysis_adversarial-vs-stochastic.md)", "blind_to": "within-matrix structure beyond a"},
        {"id": "P3_low_importance_carries_it", "arm": list(SCORED), "scored": True,
         "pass": "sum of max(e_b, 0) over imp0 and imp1 >= 0.5 x the sum over all four importance bins, on >= 3 of 4 scored states",
         "null": "the high-importance half carries it",
         "predicted": "pass, low confidence: g near 0 makes a subcomponent removable everywhere, and minimality made it inert "
                      "only on W0's activations", "blind_to": "cross-bin interaction; the state's own importance"},
    ],
    "reported_not_scored": "the decomposition on ct emoticon, r1_t588_v452, r2_t359_v816 and the adjoint (X small; parts in "
                           "nats); per-layer e_b (expected: largest share in layer 0 or 3); removed_b at each state and W0 "
                           "(expected imp0 highest, state within 0.1 of W0 per bin); F_all against the bf16 adversary KL",
}

PREDICTIONS_AMEND1: dict[str, Any] = {
    "filed": "2026-10-21",
    "amends": "results/predictions.json (stays as filed); plan reports/2026-10-21_plan_cancellation-complement.md -> "
              "Amendment 1, from reports/2026-10-21_analysis_cancellation-complement.md -> What next (approved by the user)",
    "setup": {
        "same_as_E20": "W0 first, then the same eight states in the same order, the same 128 Pile val rows, the same "
                       "adversary (seed 0, 20 steps), every E20 number recomputed; the adversary's sources saved per model",
        "numbers": "Reverts over the hooks of state_blocks with the alive blocks kept: F_dead (each dead subcomponent's "
                   "(1 - g)(1 - a) removed), F_dense (1 - a_Delta of Delta' removed), F_rest (both), F_dense_fix and "
                   "F_dead_fix (the same at W0's levels a, with the model's own g). Per state against W0: "
                   "D = dF_dense, K = dF_dead, P = d(F_rest - F_dense - F_dead), J = d(F_all - F_both - F_rest) with "
                   "F_orig for F_both at W0, so E20's R = D + K + P + J exactly; D_fix = F_dense_fix(s) - F_dense(W0), "
                   "K_fix = F_dead_fix(s) - F_dead(W0). Levels: mean over matrices of 1 - a_Delta, mean of 1 - a over "
                   "the dead subcomponents pooled, and the dead block's mean removed fraction (1 - g)(1 - a)",
        "reading": f"per state with R > 0: J >= {JOINT} R -> 4 (only jointly with the alive removal); else the largest of "
                   f"D, K, P: P -> dead with Delta'; D -> 1 if D_fix <= {ALONE_FIX} D (W0's levels take most of it "
                   f"away: the adversary removes more) else 2 (the same removal costs more); K -> 1 if K_fix <= "
                   f"{ALONE_FIX} K else 3 (the dead subcomponents carry more)",
        "expected_by_reading": {
            "1_removes_more": f"D or K >= 0.25 R and the larger; its _fix <= {ALONE_FIX} of it; that block's level up by "
                              f">= {LEVEL_TOL}; J < {JOINT} R",
            "2_same_removal_costs_more": f"D the largest, D_fix > {ALONE_FIX} D; Delta' level within {LEVEL_TOL}; J < {JOINT} R",
            "3_dead_carry_more": f"K the largest, K_fix > {ALONE_FIX} K; dead level within {LEVEL_TOL} (the removed "
                                 f"fraction may rise through g); J < {JOINT} R",
            "4_only_jointly": f"J >= {JOINT} R; D, K, P each small; levels within {LEVEL_TOL}",
            "W0": "D = K = P = J = 0 by construction; F_dense_fix = F_dense and F_dead_fix = F_dead (same sources)",
        },
        "blind_to": "Pile rows only; KL is not additive (P and J absorb it); for the least-squares state Delta' is not W0's "
                    "Delta, so D there also reads the update moved into Delta'; the fixed-level arms hold a, not g; one "
                    "adversary seed; no fixed-level arm for the alive removal",
        "noise": "E20 reproduced E17 to ratio 1.00 on the same card, so the arms are deterministic up to fp32 order; one "
                 "adversary seed (E11 f = 0.075); thresholds are coarse (halves of R, 0.1 in level)",
    },
    "gates": [
        {"id": "G1_closure", "rule": "as E20, per model", "if_fail": "stop"},
        {"id": "G2_reproduces_E20", "rule": f"F_all, F_both, F_orig within {G2_TOL} relative of results/complement.json, "
                                            f"per model; |F_hooks_all - F_all| <= {G1_TOL} (every block removed at the "
                                            "adversary's fractions through the alive and dead caches); at W0 also |F_dense_fix - F_dense| and |F_dead_fix - F_dead| <= "
                                            f"{ZERO_TOL}", "if_fail": "stop"},
    ],
    "predictions": [
        {"id": "P4_only_jointly", "arm": list(SCORED), "scored": True, "pass": "reading 4 on >= 3 of 4 scored states",
         "null": "a block alone (reading 1, 2 or 3) or the dead-with-Delta' pair carries R",
         "predicted": "pass, low confidence: Delta and the dead subcomponents are near-inert alone at W0 and unchanged by "
                      "component training, and E15 stage 0 saw the alive subcomponents matter only jointly"},
        {"id": "P5_levels_unchanged", "arm": list(SCORED), "scored": True,
         "pass": f"|d mean(1 - a_Delta)| <= {LEVEL_TOL} and |d mean dead (1 - a)| <= {LEVEL_TOL} on >= 3 of 4 scored states",
         "null": "the adversary sets different levels at the state (reading 1 possible)",
         "predicted": "pass: sign steps drive every level to a bound wherever the gradient keeps its sign, so W0's "
                      "adversary already removes most of Delta (W0 mean 1 - a_Delta >= 0.8, reported)"},
    ],
    "reported_not_scored": "every part (D, K, P, J, D_fix, K_fix) and level on all eight states and W0; per-matrix "
                           "1 - a_Delta; the dead block's mean removed fraction; W0's mean 1 - a_Delta (expected >= 0.8)",
}


PREDICTIONS_AMEND2: dict[str, Any] = {
    "filed": "2026-10-21",
    "amends": "results/predictions.json and predictions_amend1.json (stay as filed); plan "
              "reports/2026-10-21_plan_cancellation-complement.md -> Amendment 2, from "
              "reports/2026-10-21_analysis_cancellation-complement.md -> Amendment 1 -> What next, items 1 and 2",
    "setup": {
        "rows": f"the first {ROWS} Pile val rows in batches of {BATCH}, every KL fp32 (cancel.masked_kl: KL(model || model "
                "masked by g + (1 - g) a over all C subcomponents, Delta' under a_Delta), g always the model's own Gamma)",
        "fixed": "stage amend2_fixed: W0's saved levels a (results/large/sources/W0.pt, amendment 1's 20-step seed-0 "
                 "adversary) for every block at W0 and each of the eight states, F_w0lv; and the state's own saved "
                 "levels, F_own. X = F_own(s) - F_own(W0) (amendment 1's excess), Y = F_w0lv(s) - F_w0lv(W0), share = Y / X",
        "fixed_reading": f"per state with X > 0: share >= {SAME_MASK} -> 2 (the same mask costs more: the model changed); "
                         f"share <= {OTHER_MASK} -> 1 (the state's adversary found a different mask); between -> both in part",
        "strong": f"stage amend2_strong at W0 and {FAILING[1]}: gamma.pgd_sources (sign steps of 0.1, one source per matrix) "
                  f"in four runs, fp32 KL read at cumulative checkpoints: {json.dumps({r: c for r, (_, _, c) in RUNS.items()})}; "
                  "seed0, seed1, seed2 from the seed's random draw; alive_whole from the model's saved 20-step levels "
                  "with every alive level set to 1 (cancel.alive_whole). M = the largest KL over every checkpoint of >= 20 "
                  "steps of every run; M_random the same over the random-start runs; F20 = seed0 at 20 steps (amendment "
                  "1's adversary). X20 = F20(r1 t534) - F20(W0); X_strong = M(r1 t534) - M(W0)",
        "expected_at_W0": "fixed: F_w0lv = F_own = 0.660 (the same sources), Y = X = 0. strong: F20 = 0.660, alive_whole "
                          "at 0 steps = 1.075 (amendment 1's F_rest); M >= 1.075 if sign steps from there do not fall",
        "blind_to": "Pile rows only; the fixed arm holds the levels a, not g (the state's Gamma sets g), so share reads "
                    "a fixed choice of a, not a fixed removed fraction; for the least-squares state Delta' differs; the "
                    "strong adversary is still sign steps of 0.1 with one source per matrix (no step decay, no per-row "
                    "sources), so M is a lower bound on the true maximum; two models only in the strong stage",
        "noise": "the fixed stage is deterministic (no search): fp32 summation order, about 1e-7 nats (amendment 1's "
                 "G2); the strong stage's search noise is read from the spread of seed0, seed1, seed2 at 20 steps "
                 "(E11 seed floor f = 0.075, about 0.05 nats at W0)",
    },
    "gates": [
        {"id": "C1_W0_levels_at_W0", "stage": "fixed", "rule": f"F_w0lv(W0) within {ZERO_TOL} nats of amendment 1's "
                                                                "F_all(W0) (0.660363)", "if_fail": "stop"},
        {"id": "G3_own_levels", "stage": "fixed", "rule": f"F_own within {G1_TOL} nats of amendment 1's F_all, per state "
                                                          "(the state rebuild and the saved sources agree)", "if_fail": "stop"},
        {"id": "C2_strong_reproduces", "stage": "strong", "rule": f"seed0 at 20 steps within {G2_TOL} relative of "
                                                                  "amendment 1's F_all (W0 0.660, r1 t534 1.529); alive_whole "
                                                                  f"at 0 steps within {G1_TOL} nats of amendment 1's F_rest "
                                                                  "(W0 1.0754, r1 t534 0.9433); W0 first", "if_fail": "stop"},
    ],
    "predictions": [
        {"id": "P6_same_mask_costs_more", "stage": "fixed", "arm": list(SCORED), "scored": True,
         "pass": f"share >= {SAME_MASK} (reading 2) on >= 3 of 4 scored states (relative to each state's X)",
         "null": f"share < {SAME_MASK}: W0's own mask costs little more at the state; the state's adversary found a "
                 "different mask (reading 1, or the search)",
         "predicted": "fail, low confidence: E15 stage 0 found masks do not transfer between a state and W0, and at "
                      "W0's levels the dead block alone costs less at the failing states (amendment 1's K_fix < 0)"},
        {"id": "P7_floor_understated", "stage": "strong", "arm": "W0", "scored": True,
         "pass": f"M(W0) >= {FLOOR} nats (absolute), over every checkpoint of >= 20 steps of every run",
         "null": f"M(W0) < {FLOOR}: the stronger search, even from the alive-whole choice, stays near 0.66",
         "predicted": "pass, moderate confidence: the alive-whole start is at 1.075 and ascent steps should keep most of it"},
        {"id": "P8_excess_partly_the_search", "stage": "strong", "arm": FAILING[1], "scored": True,
         "pass": f"X_strong <= {SEARCH_SHARE} x X20 (relative)",
         "null": f"X_strong > {SEARCH_SHARE} x X20: the stronger adversary raises both models alike; the excess is not "
                 "an artefact of where a 20-step search lands",
         "predicted": "pass, low confidence: W0's maximum rises to about 1.08 through the alive-whole start, while r1 "
                      "t534's alive-whole start (0.94) is below its own 1.53, so its maximum should rise less"},
    ],
    "reported_not_scored": "fixed: F_w0lv, Y, share and reading on all eight states (the passing ones have X under 0.16, "
                           "share unstable); strong: every checkpoint's KL, M_random(W0) and M_random(r1 t534) "
                           "(expected: M_random(W0) < 1.0, the random starts stay near 0.66), the seeds' spread at 20 steps, "
                           "seed0's gain from 20 to 100 steps, X_strong",
}

PREDICTIONS_AMEND3: dict[str, Any] = {
    "filed": "2026-10-22",
    "amends": "results/predictions.json, predictions_amend1.json, predictions_amend2.json (stay as filed); plan "
              "reports/2026-10-21_plan_cancellation-complement.md -> Amendment 3, from "
              "reports/2026-10-22_review_vpd-adversary-convergence.md",
    "setup": {
        "rows": f"the first {ROWS} Pile val rows in batches of {BATCH}, every KL fp32 (cancel.masked_kl, as amendment 2)",
        "adversary": f"gamma.pgd_sources: one source per subcomponent shared over all rows plus one per matrix for Delta, "
                     f"random start from the seed, n sign steps of 2 / n for n in {list(BUDGETS)} (each budget its own run; "
                     "n = 20 is the project's adversary, 20 steps of 0.1)",
        "models": "W0 (seeds 0 and 1) and, seed 0: " + ", ".join(m for sh in AMEND3_SHARDS[1:] for m, _ in sh),
        "ratio": "r(s, n) = KL(s, n) / KL(W0 seed 0, n): the project's faithfulness ratio at budget n",
        "expected_at_W0": "n = 20 reproduces amendment 1 (0.660); KL rises with n but more slowly than amendment 2's "
                          "fixed-0.1 runs (2.04 at 50, 13.98 at 100); the paper's own table reads 0.83 / 1.35 / 3.84 / 25.3 "
                          "at 20 / 40 / 80 / 160 (step size unstated)",
        "expected_at_20": "ratios as amendment 1: r0 t352 1.82, r1 t534 2.32, r2 t275 6.72, ls emoticon 1.93, adjoint emoticon 1.05",
        "blind_to": "one adversary family (sign steps, one source per subcomponent shared over rows); one seed per updated "
                    "model, two at W0, so a ratio's noise at n > 20 is read only from W0's two seeds; whether the KL ever "
                    "levels off (no budget past 160); Pile rows only",
        "noise": "W0 seed 0 vs seed 1 at each budget; at 20 steps amendment 2's three seeds spread 0.645 to 0.706",
        "positive_control": "the adjoint emoticon state, which passes at 20 steps (1.05), must not fail at every budget "
                            "for the ratio to separate anything (P10)",
    },
    "gates": [
        {"id": "C3_20_reproduces", "rule": f"each model's seed-0 KL at 20 steps within {G2_TOL} relative of amendment 1's "
                                           "F_all; W0 first, alone, before any other shard", "if_fail": "stop"},
    ],
    "predictions": [
        {"id": "P9_failing_stay_failing", "arm": list(SCORED), "scored": True,
         "pass": f"r(s, n) >= {STAY_FAILING} at every n in {list(BUDGETS[1:])} on >= 3 of the 4 scored states (absolute ratio)",
         "null": "the failing states' excess shrinks relative to W0 as the search gets stronger: the 20-step verdicts "
                 "were an artefact of a short search",
         "predicted": "pass, moderate confidence: amendment 2's fixed-0.1 seed-0 run kept r1 t534 at 2.3 to 2.7 times W0 "
                      "at 20, 50 and 100 steps"},
        {"id": "P10_passing_stays_passing", "arm": f"adjoint {EMOTICON}", "scored": True,
         "pass": f"r(adjoint emoticon, n) <= {STAY_PASSING} at every n in {list(BUDGETS[1:])} (absolute ratio)",
         "null": "a passing state starts failing under a stronger search: the 20-step passes were an artefact too",
         "predicted": "pass, moderate confidence: its 20-step excess is 0.03 nats"},
        {"id": "P11_ratio_stable", "arm": list(SCORED), "scored": True,
         "pass": f"max over n of r(s, n) / min over n of r(s, n) <= {STABLE} on >= 3 of the 4 scored states (relative)",
         "null": "the ratio moves by more than half across budgets: a verdict depends on the budget it was read at",
         "predicted": "pass, low confidence: one pair (r1 t534, amendment 2) only, at a different step size"},
    ],
    "reported_not_scored": "KL per model and budget (curve against the paper's table); W0's seed spread per budget; "
                           "r(s, n) for every model and budget; r1 t534's 20-step ratio against amendment 2's",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))


# ----------------------------------------------------------------------------- one model


def _new_target(ctx: dict[str, Any], dW: dict[str, Tensor] | None, uv2: dict[str, tuple[Tensor, Tensor]] | None
                ) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, Tensor]]:
    """A state's alive subcomponents (U2, V2) per matrix and its target weights, on the device (both None: W0)."""
    uv, W0, dev = ctx["uv"], ctx["W0"], ctx["device"]
    new = {k: uv[k] if uv2 is None or k not in uv2 else (uv2[k][0].to(dev), uv2[k][1].to(dev)) for k in uv}
    target = {k: W0[k] + dW[k].to(dev) if dW is not None and k in dW else W0[k] for k in W0}
    return new, target


def at_state(ctx: dict[str, Any], dW: dict[str, Tensor] | None, uv2: dict[str, tuple[Tensor, Tensor]] | None) -> Any:
    """The model set to a state (dW, uv2 None: W0) for the duration of a `with` block."""
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    new, target = _new_target(ctx, dW, uv2)
    return model_state(ctx["model"], target=target, components=components_with(ctx["model"], ctx["alive"], new))


def measure(ctx: dict[str, Any], name: str, dW: dict[str, Tensor] | None, uv2: dict[str, tuple[Tensor, Tensor]] | None,
            extra: Any = None) -> dict[str, Any]:
    """Every number of one state (dW, uv2 None: W0) under its own adversary. Returns early after a failed G1.
    `extra(name, rec, rem, cached, dead, sources)` adds amendment 1's arms under the same hooks (dead: the dead cache)."""
    from commute import cancel, gamma, vpd
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    model, alive, uv, dev = ctx["model"], ctx["alive"], ctx["uv"], ctx["device"]
    new, target = _new_target(ctx, dW, uv2)
    pieces: cancel.Pieces = {k: (uv[k][0], uv[k][1], new[k][0], new[k][1]) for k in uv}
    rec: dict[str, Any] = {}
    t0 = time.time()
    with model_state(model, target=target, components=components_with(model, alive, {k: (p[2], p[3]) for k, p in pieces.items()})):
        sources, rec["pgd_kl"] = gamma.pgd_sources(model, ctx["batches"], PGD_STEPS, seed=0)
        cached, dead, rec["F_all"], first = cancel.cache_adversary(model, ctx["batches"], alive, sources)
        rem = cancel.Remover(model, cancel.state_blocks(model, alive, pieces))
        try:
            b0, ref0, _ = cached[0]
            rec["G1"] = {"kl_zero": float(vpd.kl_per_position(ref0, rem.logits(b0, {})).mean()),
                         "kl_closure": float(vpd.kl_per_position(first["masked_logits"], rem.logits(b0, first["weights"])).mean())}
            rec["G1"]["ok"] = rec["G1"]["kl_zero"] <= ZERO_TOL and rec["G1"]["kl_closure"] <= G1_TOL
            log(f"[{name}] G1 {rec['G1']} | pgd (bf16) {rec['pgd_kl']:.4f} F_all (fp32) {rec['F_all']:.4f}")
            if not rec["G1"]["ok"]:
                return rec
            del first
            wf = cancel.adversary_weights(sources, alive, dev)
            for key, (o, n) in {"F_orig": (1, 0), "F_adv": (-1, 1), "F_both": (0, 1)}.items():
                rec[key] = cancel.revert_kl(rem, cached, cancel.arm(wf, o, n))
            rec["F_orig_perm"] = [cancel.revert_kl(rem, cached, cancel.arm(cancel.adversary_weights(sources, alive, dev, perm_seed=i), 1, 0))
                                  for i in range(N_PERMS)]
            rec["F_orig_bin"] = {b: cancel.revert_kl(rem, cached, cancel.arm(wf, 1, 0, sel)) for b, sel in ctx["bins"].items()}
            rec["removed"] = cancel.bin_means(cached, wf, ctx["bins"])
            if extra is not None:
                extra(name, rec, rem, cached, dead, sources)
        finally:
            rem.remove()
    rec["seconds"] = time.time() - t0
    log(f"[{name}] F_orig {rec['F_orig']:.4f} F_adv {rec['F_adv']:.5f} F_both {rec['F_both']:.4f} "
        f"F_orig_perm {[round(x, 4) for x in rec['F_orig_perm']]} | {rec['seconds']:.0f} s")
    log(f"[{name}] F_orig per bin {json.dumps({b: round(v, 4) for b, v in rec['F_orig_bin'].items()})}")
    log(f"[{name}] removed fraction per bin {json.dumps({b: round(v, 3) for b, v in rec['removed'].items()})}")
    return rec


# ----------------------------------------------------------------------------- scoring


def decompose(s: dict[str, Any], w0: dict[str, Any]) -> dict[str, Any]:
    """X = O + F_adv + I + R (exact by definition) and the per-bin excess e_b, for one state against W0."""
    X = s["F_all"] - w0["F_all"]
    O = s["F_orig"] - w0["F_orig"]
    I = s["F_both"] - s["F_orig"] - s["F_adv"]
    R = (s["F_all"] - s["F_both"]) - (w0["F_all"] - w0["F_orig"])
    e = {b: s["F_orig_bin"][b] - w0["F_orig_bin"][b] for b in s["F_orig_bin"]}
    pos = {b: max(v, 0.0) for b, v in e.items() if b.startswith("imp")}
    low = sum(pos.get(f"imp{i}", 0.0) for i in range(N_BINS // 2))
    return {"X": X, "O": O, "F_adv": s["F_adv"], "I": I, "R": R, "e_bin": e,
            "low_share": low / sum(pos.values()) if sum(pos.values()) > 0 else float("nan"),
            "perm_ratio": s["F_orig"] / (sum(s["F_orig_perm"]) / len(s["F_orig_perm"]))}


def score(out: dict[str, Any]) -> dict[str, Any]:
    st, w0 = out["states"], out["W0"]
    got = [n for n in SCORED if n in st and "F_orig" in st[n]]
    d = {n: decompose(st[n], w0) for n in st if "F_orig" in st[n]}

    def verdict(ok: Any, need: int | None = None) -> dict[str, Any]:
        if len(got) < len(SCORED):
            return {"verdict": "not run", "missing": [n for n in SCORED if n not in got]}
        per = {n: bool(ok(d[n])) for n in got}
        return {"verdict": "pass" if sum(per.values()) >= (need or len(per)) else "fail", "per_state": per}

    return {"G1_closure": {n: r["G1"]["ok"] for n, r in [("W0", w0), *st.items()] if "G1" in r},
            "C0_reproduces_E17": out.get("C0", {"verdict": "not run"}),
            "P1_excess_in_originals": verdict(lambda r: r["O"] >= 0.5 * r["X"]),
            "P2_pattern_matters": verdict(lambda r: r["perm_ratio"] >= 2),
            "P3_low_importance_carries_it": verdict(lambda r: r["low_share"] >= 0.5, need=3),
            "decomposition": d}


# ----------------------------------------------------------------------------- the run


def load(device: str) -> dict[str, Any]:
    from commute import cancel, component_training, saved_states, vpd
    from commute.data import batches, pile_rows

    model = vpd.load_component_model(device)
    alive = saved_states.load_alive(ROUND1)
    bins = {**cancel.importance_bins(saved_states.load_mean_g(ROUND1), alive, N_BINS), **cancel.layer_bins(alive)}
    return {"device": device, "model": model, "alive": alive, "W0": vpd.target_weights(model), "bins": bins,
            "uv": component_training.alive_uv(model, alive), "batches": batches(pile_rows("val", 0, ROWS, log), BATCH)}


def c0(rec: dict[str, Any]) -> dict[str, Any]:
    ref = json.loads(E17.read_text())["states"][CONTROL]
    r = {"F_adv": rec["F_adv"] / ref["F_adv"], "pgd_kl": rec["pgd_kl"] / ref["pgd_kl"]}
    return {"ratio_to_E17": r, "verdict": "pass" if all(abs(v - 1) <= C0_TOL for v in r.values()) else "fail"}


def _states(ctx: dict[str, Any]) -> dict[str, Any]:
    """E17's eight states, rebuilt by `commute.saved_states` (dW, uv2) per name."""
    from commute import saved_states

    states = {f"ct {t}": saved_states.ct_state(E11, t, STEP) for t in sorted({n.split()[1] for n in ORDER if n.startswith("ct ")})}
    for kind, upd in saved_states.split_states(ctx["model"], ctx["uv"], E11, ROUND1, EMOTICON, STEP, TOL).items():
        states[f"{kind} {EMOTICON}"] = upd
    return states


def stage_all(device: str) -> None:
    assert PRED.is_file(), "file predictions first: run.py --predictions"
    t0 = time.time()
    ctx = load(device)
    log(f"loaded; bins {[(b, sum(int(m.sum()) for m in sel.values())) for b, sel in ctx['bins'].items()]} ({time.time() - t0:.0f} s)")
    out: dict[str, Any] = {"states": {}, "W0": measure(ctx, "W0", None, None)}
    w0 = out["W0"]
    dump(OUT, out)
    if not w0["G1"]["ok"] or w0["F_adv"] > ZERO_TOL or abs(w0["F_both"] - w0["F_orig"]) > ZERO_TOL:
        sys.exit(f"G1 failed at W0: {w0['G1']}, F_adv {w0.get('F_adv')}, F_both - F_orig {w0.get('F_both', 0) - w0.get('F_orig', 0)}")
    states = _states(ctx)
    for name in ORDER:
        rec = measure(ctx, name, *states[name])
        out["states"][name] = rec
        dump(OUT, out)
        if not rec["G1"]["ok"]:
            sys.exit(f"G1 failed on {name}: {rec['G1']}")
        dd = decompose(rec, w0)
        log(f"[{name}] X {dd['X']:.4f} = O {dd['O']:.4f} + F_adv {dd['F_adv']:.4f} + I {dd['I']:.4f} + R {dd['R']:.4f} | "
            f"F_orig/perm {dd['perm_ratio']:.2f} | low-importance share {dd['low_share']:.2f} | "
            f"e_b {json.dumps({b: round(v, 4) for b, v in dd['e_bin'].items()})}")
        if name == CONTROL:
            out["C0"] = c0(rec)
            log(f"[C0] {out['C0']}")
            dump(OUT, out)
            if out["C0"]["verdict"] != "pass":
                sys.exit(f"C0 failed: {out['C0']}")
    out["score"] = score(out)
    out["seconds"] = time.time() - t0
    dump(OUT, out)
    log(f"score {json.dumps({k: v for k, v in out['score'].items() if k != 'decomposition'}, default=_json_default)}")
    log(f"done in {out['seconds']:.0f} s -> {OUT}")


# ----------------------------------------------------------------------------- amendment 1: splitting R


def amend1_arms(ctx: dict[str, Any]) -> Any:
    """`measure`'s extra for amendment 1: the dead-block and Delta' reverts at the model's own levels and at W0's, the
    levels, and the sources saved. W0 runs first, so its sources become ctx["src0"] for every later model."""
    import torch

    from commute import cancel

    alive, dev = ctx["alive"], ctx["device"]
    every_dead = {"dead": {k: torch.ones(int((~m).sum()), dtype=torch.bool) for k, m in alive.items()}}

    def extra(name: str, rec: dict[str, Any], rem: Any, cached: Any, dead: Any, sources: dict[str, Tensor]) -> None:
        (LARGE / "sources").mkdir(parents=True, exist_ok=True)
        torch.save(sources, LARGE / "sources" / f"{name.replace(' ', '_')}.pt")
        src0 = ctx.setdefault("src0", sources)
        both = [(b, ref, (g, gd)) for (b, ref, g), (_, _, gd) in zip(cached, dead)]
        rec["F_hooks_all"] = cancel.revert_kl(rem, both, cancel.all_blocks_arm(sources, alive, dev))
        for key, (src, d, n) in {"F_dead": (sources, True, False), "F_dense": (sources, False, True),
                                 "F_rest": (sources, True, True), "F_dead_fix": (src0, True, False),
                                 "F_dense_fix": (src0, False, True)}.items():
            rec[key] = cancel.revert_kl(rem, dead, cancel.rest_arm(src, alive, dev, d, n))
        rec["levels"] = cancel.levels(sources, alive)
        rec["levels"]["dead_removed"] = cancel.bin_means(dead, cancel.dead_weights(sources, alive, dev), every_dead)["dead"]
        lv = rec["levels"]
        log(f"[{name}] F_hooks_all {rec['F_hooks_all']:.6f} (F_all {rec['F_all']:.6f}) | F_dead {rec['F_dead']:.4f} F_dense {rec['F_dense']:.4f} F_rest {rec['F_rest']:.4f} "
            f"F_dead_fix {rec['F_dead_fix']:.4f} F_dense_fix {rec['F_dense_fix']:.4f} | 1-a_Delta mean {lv['dense_mean']:.3f} "
            f"dead 1-a mean {lv['dead_mean']:.3f} dead removed {lv['dead_removed']:.3f}")
        log(f"[{name}] 1-a_Delta per matrix {json.dumps({k: round(v, 3) for k, v in lv['dense'].items()})}")

    return extra


def split_rest(s: dict[str, Any], w0: dict[str, Any]) -> dict[str, Any]:
    """E20's R = D + K + P + J (exact by definition), the fixed-level parts, the level changes and the reading."""
    R = (s["F_all"] - s["F_both"]) - (w0["F_all"] - w0["F_orig"])
    D, K = s["F_dense"] - w0["F_dense"], s["F_dead"] - w0["F_dead"]
    P = (s["F_rest"] - s["F_dense"] - s["F_dead"]) - (w0["F_rest"] - w0["F_dense"] - w0["F_dead"])
    J = (s["F_all"] - s["F_both"] - s["F_rest"]) - (w0["F_all"] - w0["F_orig"] - w0["F_rest"])
    d = {"R": R, "D": D, "K": K, "P": P, "J": J, "D_fix": s["F_dense_fix"] - w0["F_dense"],
         "K_fix": s["F_dead_fix"] - w0["F_dead"],
         **{f"d_{x}": s["levels"][x] - w0["levels"][x] for x in ("dense_mean", "dead_mean", "dead_removed")}}
    return {**d, "reading": reading(d)}


def reading(d: dict[str, float]) -> str:
    """The filed rule (predictions_amend1.json -> setup.reading)."""
    if d["R"] <= 0:
        return "none: R <= 0"
    if d["J"] >= JOINT * d["R"]:
        return "4"
    top = max(("D", "K", "P"), key=lambda x: d[x])
    if top == "P":
        return "dead with Delta'"
    return "1" if d[f"{top}_fix"] <= ALONE_FIX * d[top] else ("2" if top == "D" else "3")


def score_amend1(out: dict[str, Any]) -> dict[str, Any]:
    st, w0 = out["states"], out["W0"]
    d = {n: split_rest(st[n], w0) for n in st if "F_rest" in st[n]}
    got = [n for n in SCORED if n in d]

    def verdict(ok: Any) -> dict[str, Any]:
        if len(got) < len(SCORED):
            return {"verdict": "not run", "missing": [n for n in SCORED if n not in got]}
        per = {n: bool(ok(d[n])) for n in got}
        return {"verdict": "pass" if sum(per.values()) >= 3 else "fail", "per_state": per}

    return {"G2_reproduces_E20": out["G2"],
            "P4_only_jointly": verdict(lambda r: r["reading"] == "4"),
            "P5_levels_unchanged": verdict(lambda r: abs(r["d_dense_mean"]) <= LEVEL_TOL and abs(r["d_dead_mean"]) <= LEVEL_TOL),
            "readings": {n: r["reading"] for n, r in d.items()}, "split": d}


def g2(name: str, rec: dict[str, Any], e20: dict[str, Any]) -> dict[str, Any]:
    """Gate G2 for one model: E20's F_all, F_both, F_orig reproduced, the hooks with every block at the adversary's
    fractions over the alive and dead caches give F_all (the dead cache is right), and at W0 the fixed-level identity."""
    ref = e20["W0"] if name == "W0" else e20["states"][name]
    rel = {k: abs(rec[k] - ref[k]) / abs(ref[k]) for k in ("F_all", "F_both", "F_orig")}
    ok = all(v <= G2_TOL for v in rel.values()) and abs(rec["F_hooks_all"] - rec["F_all"]) <= G1_TOL
    if name == "W0":
        ok = ok and abs(rec["F_dense_fix"] - rec["F_dense"]) <= ZERO_TOL and abs(rec["F_dead_fix"] - rec["F_dead"]) <= ZERO_TOL
    return {"rel": rel, "hooks_all_minus_F_all": rec["F_hooks_all"] - rec["F_all"], "ok": ok}


def stage_amend1(device: str) -> None:
    global LOG
    LOG = LARGE / "amend1.log"
    assert PRED_A1.is_file(), "file predictions first: run.py --predictions amend1"
    e20 = json.loads(OUT.read_text())
    t0 = time.time()
    ctx = load(device)
    extra = amend1_arms(ctx)
    log(f"amendment 1: loaded ({time.time() - t0:.0f} s)")
    out: dict[str, Any] = {"states": {}, "G2": {}}

    def one(name: str, dW: Any, uv2: Any) -> None:
        rec = measure(ctx, name, dW, uv2, extra)
        if name == "W0":
            out["W0"] = rec
        else:
            out["states"][name] = rec
        dump(OUT_A1, out)
        if not rec["G1"]["ok"]:
            sys.exit(f"G1 failed on {name}: {rec['G1']}")
        out["G2"][name] = g2(name, rec, e20)
        dump(OUT_A1, out)
        log(f"[{name}] G2 {out['G2'][name]}")
        if not out["G2"][name]["ok"]:
            sys.exit(f"G2 failed on {name}: does not reproduce E20")
        if name != "W0":
            r = split_rest(rec, out["W0"])
            log(f"[{name}] R {r['R']:.4f} = D {r['D']:.4f} + K {r['K']:.4f} + P {r['P']:.4f} + J {r['J']:.4f} | "
                f"D_fix {r['D_fix']:.4f} K_fix {r['K_fix']:.4f} | d 1-a_Delta {r['d_dense_mean']:+.3f} "
                f"d dead 1-a {r['d_dead_mean']:+.3f} d dead removed {r['d_dead_removed']:+.3f} | reading {r['reading']}")

    one("W0", None, None)
    states = _states(ctx)  # after W0, as E20
    for name in ORDER:
        one(name, *states[name])
    out["score"] = score_amend1(out)
    out["seconds"] = time.time() - t0
    dump(OUT_A1, out)
    log(f"score {json.dumps({k: v for k, v in out['score'].items() if k != 'split'}, default=_json_default)}")
    log(f"done in {out['seconds']:.0f} s -> {OUT_A1}")


# ----------------------------------------------------------------------------- amendment 2: fixed levels, a stronger search


def saved_sources(name: str) -> dict[str, Tensor]:
    """Amendment 1's adversary levels for one model (results/large/sources/)."""
    import torch

    return torch.load(SOURCES / f"{name.replace(' ', '_')}.pt", weights_only=False)


def fixed_reading(share: float) -> str:
    """The filed rule (predictions_amend2.json -> setup.fixed_reading)."""
    if math.isnan(share):
        return "none: X <= 0"
    return "2" if share >= SAME_MASK else "1" if share <= OTHER_MASK else "both"


def fixed_one(ctx: dict[str, Any], name: str, update: Any, src0: dict[str, Tensor], f0: float, a1_all: float) -> dict[str, Any]:
    """One state's F_own (its saved levels) and F_w0lv (W0's), its excess X, W0's-mask excess Y and the share Y / X."""
    from commute import cancel

    with at_state(ctx, *update):
        own = cancel.masked_kl(ctx["model"], ctx["batches"], saved_sources(name))
        w0lv = cancel.masked_kl(ctx["model"], ctx["batches"], src0)
    X, Y = own - f0, w0lv - f0
    share = Y / X if X > 0 else float("nan")
    return {"F_own": own, "F_w0lv": w0lv, "X": X, "Y": Y, "share": share, "reading": fixed_reading(share),
            "G3": {"own_minus_amend1": own - a1_all, "ok": abs(own - a1_all) <= G1_TOL}}


def stage_amend2_fixed(device: str) -> None:
    global LOG
    LOG = LARGE / "amend2_fixed.log"
    assert PRED_A2.is_file(), "file predictions first: run.py --predictions amend2_fixed"
    from commute import cancel

    a1, out_path = json.loads(OUT_A1.read_text()), OUT_A2["fixed"]
    t0 = time.time()
    ctx = load(device)
    src0 = saved_sources("W0")
    with at_state(ctx, None, None):
        f0 = cancel.masked_kl(ctx["model"], ctx["batches"], src0)
    out: dict[str, Any] = {"W0": {"F_own": f0, "F_w0lv": f0}, "states": {},
                           "C1": {"minus_amend1": f0 - a1["W0"]["F_all"], "ok": abs(f0 - a1["W0"]["F_all"]) <= ZERO_TOL}}
    dump(out_path, out)
    log(f"[W0] W0's levels at W0: {f0:.7f} (amendment 1 {a1['W0']['F_all']:.7f}) | C1 {out['C1']} ({time.time() - t0:.0f} s)")
    if not out["C1"]["ok"]:
        sys.exit(f"C1 failed: W0's saved levels do not reproduce amendment 1 at W0: {out['C1']}")
    states = _states(ctx)
    for name in ORDER:
        rec = fixed_one(ctx, name, states[name], src0, f0, a1["states"][name]["F_all"])
        out["states"][name] = rec
        dump(out_path, out)
        log(f"[{name}] F_own {rec['F_own']:.5f} F_w0lv {rec['F_w0lv']:.5f} | X {rec['X']:.4f} Y {rec['Y']:.4f} "
            f"share {rec['share']:.3f} reading {rec['reading']} | G3 {rec['G3']}")
        if not rec["G3"]["ok"]:
            sys.exit(f"G3 failed on {name}: own saved levels do not reproduce amendment 1's F_all")
    per = {n: out["states"][n]["reading"] == "2" for n in SCORED}
    out["score"] = {"P6_same_mask_costs_more": {"verdict": "pass" if sum(per.values()) >= 3 else "fail", "per_state": per},
                    "share": {n: r["share"] for n, r in out["states"].items()}}
    out["seconds"] = time.time() - t0
    dump(out_path, out)
    log(f"score {json.dumps(out['score'], default=_json_default)}")
    log(f"done in {out['seconds']:.0f} s -> {out_path}")


def c2_gate(name: str, run: str, step: int, kl: float, a1: dict[str, Any]) -> dict[str, Any] | None:
    """Control C2 at the checkpoints it reads (seed0 at 20 steps, alive_whole at 0), else None."""
    ref = a1["W0"] if name == "W0" else a1["states"][name]
    if (run, step) == ("seed0", 20):
        return {"ref": ref["F_all"], "rel": abs(kl - ref["F_all"]) / ref["F_all"], "ok": abs(kl - ref["F_all"]) <= G2_TOL * ref["F_all"]}
    if (run, step) == ("alive_whole", 0):
        return {"ref": ref["F_rest"], "diff": kl - ref["F_rest"], "ok": abs(kl - ref["F_rest"]) <= G1_TOL}
    return None


def search(ctx: dict[str, Any], name: str, run: str, report: Any) -> None:
    """One run of the stronger adversary on the model as set: sign steps chained between RUNS[run]'s checkpoints, the
    fp32 KL and the levels saved at each. `report(run, step, kl)` records, logs and dumps each checkpoint."""
    import torch

    from commute import cancel, gamma

    seed, whole, checkpoints = RUNS[run]
    src = cancel.alive_whole(saved_sources(name), ctx["alive"]) if whole else None
    done = 0
    for step in checkpoints:
        if step > done:
            src, _ = gamma.pgd_sources(ctx["model"], ctx["batches"], step - done, seed=seed, init=src)
            done = step
        assert src is not None, f"{run}: a random start needs a first checkpoint > 0"
        (LARGE / "sources_amend2").mkdir(parents=True, exist_ok=True)
        torch.save(src, LARGE / "sources_amend2" / f"{name.replace(' ', '_')}_{run}_{step}.pt")
        report(run, step, cancel.masked_kl(ctx["model"], ctx["batches"], src))


def score_strong(kl: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """M, M_random, F20 per model, X20 and X_strong, and the verdicts of P7 and P8 (filed rules)."""
    def top(runs: dict[str, dict[str, float]], random_only: bool) -> float:
        return max(v for r, c in runs.items() if not (random_only and RUNS[r][1]) for s, v in c.items() if int(s) >= 20)

    w0, r1 = kl["W0"], kl[FAILING[1]]
    M = {n: top(kl[n], False) for n in kl}
    X20, X_strong = r1["seed0"]["20"] - w0["seed0"]["20"], M[FAILING[1]] - M["W0"]
    return {"M": M, "M_random": {n: top(kl[n], True) for n in kl}, "F20": {n: kl[n]["seed0"]["20"] for n in kl},
            "X20": X20, "X_strong": X_strong, "X_strong_over_X20": X_strong / X20,
            "P7_floor_understated": "pass" if M["W0"] >= FLOOR else "fail",
            "P8_excess_partly_the_search": "pass" if X_strong <= SEARCH_SHARE * X20 else "fail"}


def stage_amend2_strong(device: str) -> None:
    global LOG
    LOG = LARGE / "amend2_strong.log"
    assert PRED_A2.is_file(), "file predictions first: run.py --predictions amend2_strong"
    from commute import saved_states

    a1, out_path = json.loads(OUT_A1.read_text()), OUT_A2["strong"]
    t0 = time.time()
    ctx = load(device)
    updates = {"W0": (None, None), FAILING[1]: saved_states.ct_state(E11, FAILING[1].split()[1], STEP)}
    out: dict[str, Any] = {"kl": {}, "C2": {}}
    for name in STRONG_MODELS:
        out["kl"][name] = {r: {} for r in RUNS}

        def report(run: str, step: int, kl: float, name: str = name) -> None:
            out["kl"][name][run][str(step)] = kl
            gate = c2_gate(name, run, step, kl, a1)
            if gate is not None:
                out["C2"][f"{name} {run} {step}"] = gate
            dump(out_path, out)
            log(f"[{name}] {run} step {step}: KL (fp32) {kl:.5f}" + (f" | C2 {gate}" if gate else "") + f" ({time.time() - t0:.0f} s)")
            if gate is not None and not gate["ok"]:
                sys.exit(f"C2 failed on {name} {run} {step}: {gate}")

        with at_state(ctx, *updates[name]):
            for run in RUNS:
                search(ctx, name, run, report)
    out["score"] = score_strong(out["kl"])
    out["seconds"] = time.time() - t0
    dump(out_path, out)
    log(f"score {json.dumps(out['score'], default=_json_default)}")
    log(f"done in {out['seconds']:.0f} s -> {out_path}")

def stage_amend3_budget(device: str, shard: int) -> None:
    """Amendment 3: each (model, seed) of the shard at every budget, fp32 KL and levels saved after every run."""
    global LOG
    LOG = LARGE / f"amend3_shard{shard}.log"
    assert PRED_A3.is_file(), "file predictions first: run.py --predictions amend3_budget"
    import torch

    from commute import cancel, gamma

    a1, out_path = json.loads(OUT_A1.read_text()), RESULTS / f"complement_amend3_shard{shard}.json"
    if shard > 0:
        w0 = json.loads((RESULTS / "complement_amend3_shard0.json").read_text())
        assert w0["C3"]["W0"]["ok"] and len(w0["kl"]["W0"]["0"]) == len(BUDGETS), "shard 0 (W0) must finish and pass first"
    t0 = time.time()
    ctx = load(device)
    states = _states(ctx) if shard > 0 else {}
    out: dict[str, Any] = {"kl": {}, "C3": {}}
    (LARGE / "sources_amend3").mkdir(parents=True, exist_ok=True)
    for name, seed in AMEND3_SHARDS[shard]:
        out["kl"].setdefault(name, {})[str(seed)] = {}
        with at_state(ctx, *(states[name] if name != "W0" else (None, None))):
            for n in BUDGETS:
                src, _ = gamma.pgd_sources(ctx["model"], ctx["batches"], n, step_size=2 / n, seed=seed)
                kl = cancel.masked_kl(ctx["model"], ctx["batches"], src)
                torch.save(src, LARGE / "sources_amend3" / f"{name.replace(' ', '_')}_s{seed}_{n}.pt")
                out["kl"][name][str(seed)][str(n)] = kl
                gate = None
                if (n, seed) == (20, 0):
                    ref = a1["W0"]["F_all"] if name == "W0" else a1["states"][name]["F_all"]
                    gate = out["C3"][name] = {"ref": ref, "rel": abs(kl - ref) / ref, "ok": abs(kl - ref) <= G2_TOL * ref}
                dump(out_path, out)
                log(f"[{name}] seed {seed} {n} steps of {2 / n:.4f}: KL (fp32) {kl:.5f}" + (f" | C3 {gate}" if gate else "")
                    + f" ({time.time() - t0:.0f} s)")
                if gate is not None and not gate["ok"]:
                    sys.exit(f"C3 failed on {name}: {gate}")
    out["seconds"] = time.time() - t0
    dump(out_path, out)
    log(f"done in {out['seconds']:.0f} s -> {out_path}")


def stage_amend3_score() -> None:
    """Merge the shard files, the ratios to W0 seed 0 per budget, and the verdicts of P9-P11 (filed rules)."""
    kl: dict[str, Any] = {}
    for i in range(len(AMEND3_SHARDS)):
        kl.update(json.loads((RESULTS / f"complement_amend3_shard{i}.json").read_text())["kl"])
    w0 = kl["W0"]["0"]
    r = {m: {n: kl[m]["0"][n] / w0[n] for n in w0} for m in kl if m != "W0"}
    later = [str(n) for n in BUDGETS[1:]]
    stays = {m: all(r[m][n] >= STAY_FAILING for n in later) for m in SCORED}
    spread = {m: max(r[m].values()) / min(r[m].values()) for m in SCORED}
    adj = f"adjoint {EMOTICON}"
    score = {"P9_failing_stay_failing": {"verdict": "pass" if sum(stays.values()) >= 3 else "fail", "per_state": stays},
             "P10_passing_stays_passing": "pass" if all(r[adj][n] <= STAY_PASSING for n in later) else "fail",
             "P11_ratio_stable": {"verdict": "pass" if sum(v <= STABLE for v in spread.values()) >= 3 else "fail",
                                  "max_over_min": spread},
             "W0_seed_spread": {n: abs(kl["W0"]["0"][n] - kl["W0"]["1"][n]) for n in w0}}
    dump(OUT_A3, {"kl": kl, "ratio": r, "score": score})
    print(json.dumps(score, indent=1, default=_json_default))



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all", "amend1", "amend2_fixed", "amend2_strong", "amend3_budget",
                                                 "amend3_score"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        path, pred = {"amend1": (PRED_A1, PREDICTIONS_AMEND1), "amend2_fixed": (PRED_A2, PREDICTIONS_AMEND2),
                      "amend2_strong": (PRED_A2, PREDICTIONS_AMEND2),
                      "amend3_budget": (PRED_A3, PREDICTIONS_AMEND3)}.get(args.stage or "", (PRED, PREDICTIONS))
        if path.is_file():
            sys.exit(f"{path} exists; predictions are not edited after filing (add a dated report instead)")
        dump(path, pred)
        print(f"filed {path}")
    elif args.stage == "all":
        stage_all(args.device)
    elif args.stage == "amend1":
        stage_amend1(args.device)
    elif args.stage == "amend2_fixed":
        stage_amend2_fixed(args.device)
    elif args.stage == "amend2_strong":
        stage_amend2_strong(args.device)
    elif args.stage == "amend3_budget":
        stage_amend3_budget(args.device, args.shard)
    elif args.stage == "amend3_score":
        stage_amend3_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
