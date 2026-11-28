# P2.90 (due 11-28): redteam of the unified re-gating share (E327, E328)

I checked regating-lead's new re-gating share for Qwen. The share is now read in whole-answer log-odds, the same unit as the 67M, instead of a KL ratio. I checked the code that computes that unit, the 67M reproduction and the Qwen readings at main 420d9a4a. The code is right on paper and the 67M paper's numbers reproduce. But the Qwen check at the unchanged model cannot fail, so it does not control anything. And the random updates already read have re-gating shares on the decomposition as large as the fine-tunes'. Until the size-matched random updates land, no claim that the Qwen fine-tunes "re-gate" is supported.

Verdict: **code OK; E327 holds; E328's fine-tune numbers are correct as read but have no reference yet (MUST 1).**

## What was checked

| Item | Finding | Level |
|---|---|---|
| `share.answer_logodds` (src/commute/share.py:110-130) | Derived by hand: `identity.log_odds` is log p - log(1-p), so d = -log_odds gives log p = -softplus(d) and log(1-p) = -softplus(-d). The recursion 1 - P_{<=t} = (1 - P_{<t}) + P_{<t}(1 - p_t) is exact, and the update order (log_q before log_p) uses P_{<t}. Position start + j predicts ids[j], matching `spans_from_marks` (positions p..q predict t[p+1..q+1]). Padded positions are clamped and masked by `on`. One-token answers take `log_odds` itself. | OK |
| Numerical test of the multi-token path | Closed (update 11-27, after regating-lead's reply): `tests/test_share_unit.py::test_multi_token_against_brute_force` checks answers of 1, 3 and 6 tokens in float64 against log P - log(-expm1(log P)) within 1e-5, and `test_finite_near_one` covers P near 1. Both passed on the E328 rerun pod and on the four E330 pods. I read the tests: the reference is computed without the recursion. | OK |
| What "the answer" includes | `spans_from_marks` keeps ids = t[p+1 : q+2], and t[q+1] is the end-of-response token, so P(ans) includes P(end token). That is a defensible choice, but the paper's definition ("the product of the answer tokens' probabilities") should say so. | SHOULD 2 |
| Qwen W0 controls (10 rows) | All numerators are exactly 0, as filed. But T1 and T0 (and L01, L00) are the same weights here, so the check passes whenever the forward pass is deterministic. It also reads 16 rows, not the 128 that are scored. It is a determinism check, not a control of the share, and it must not be described as one. | MUST 2 |
| Qwen fine-tune shares (36 rows) | Recomputed from `means`: numerator = L01 - L00 and gain = T1 - T0 match `numerator` and `gain` on every row I spot-checked. VPD 0.56-0.74 and neuron control 0.30-0.46, with a seed spread of 0.03 or less per side. | OK |
| References for those shares | The run-1 random updates (matched to the fine-tune's KL, not to its gain) lower the answer log-odds: gains are -20 to -9 on four sides, and gold is +1.2. On VPD, re-gating reproduces 0.48-0.81 of that drop (copy 0.48, cipher 0.50, known 0.81, invented 0.58), as much as it reproduces the fine-tunes' rise (0.56-0.74). The neuron control's randoms read -0.40 to 0.22 against 0.30-0.46 for its fine-tunes. I agree these are not gain-matched references, but they do not show that VPD's fine-tune share sits above a random change's. Until E330's gain-matched randoms land, no sentence may say the Qwen fine-tunes are re-gated more than a random change, or more on VPD than on neurons for that reason. | MUST 1 |
| Old KL share | The old KL ratio read 0.99-1.01 for VPD's random updates and 0.99-1.0 for its fine-tunes, so it could not tell them apart. Every rendered sentence that cites App H's old values (0.92-1.0, "near 1") must change together with the CSV, not only App H. | SHOULD 3 |
| E327, 67M reproduction | W0 numerator is 0 and one-token answers equal `logodds_at` exactly, both as filed. 47 of 48 states are within the filed 0.002. e53/lowercase_s0 is off by 0.00245 (s1 0.00196, s2 0.00124), so the per-state prediction **missed on 1 of 48**. The filed expectation of "bit for bit on the card and batch E136 used" held on 0 of 48, so this 4090 is not the old reads' card, or the path differs. The 0.002 floor was measured across batch sizes on one card, not across cards, so it does not cover this read. A card change is the likely cause, but it is unverified. Report it as a miss of 0.00045 on one state. The shares range 0.4-1.1, so it moves no reading. | SHOULD 4 |
| Masked-model gap at W0 | At W0, L00 differs from T0 by up to 31 log-odds on VPD (gold: T0 -230.9, L00 -199.9; cipher +3.3; copy -3.1; on 128 rows). The share is a difference, so the gap cancels to first order. Still, the appendix should give it once as context for how far the masked model is from the target. | NIT |
| Weighting | A ratio of means over answers weights each answer by its size in log-odds, so long answers (gold, gains of about 155) dominate their side's mean. This is not a problem within a side, but the text should not compare sides as if each answer counted equally. | NIT |

## Asks to regating-lead

1. MUST 1: hold any comparative sentence until E330. When E330 lands, send me its rows and I will check them against these fine-tunes.
2. MUST 2: call the W0 rows a "determinism check" (or drop them from the paper). A real do-nothing reference is the gain-matched random.
3. SHOULD 1: on E330's pod, add a stage-0 check of `answer_logodds` against log P - log(-expm1(log P)) in float64, on one batch of multi-token answers where P is not near 1. Tolerance from fp32 rounding.
4. SHOULD 2-4: say the end token is part of the answer; list every paper sentence citing the old KL values for writeup-lead; report E327 as 47 of 48 within the bar, with the miss and its unverified cause.

## Not run

- Any check of E330 (gain-matched randoms): still running.
- `qwen_reread.py` and `share_reading.answer_cells` read line by line: I checked their outputs' arithmetic, not the row selection (128 rows, whole answers only) against the App H inventory.
- The draft CSV and analysis (`qwen_share_new.csv`, `2026-11-27_analysis_p290-share-unified.md`): not yet in the repo.

## Re-check of the final page (2026-11-27)

Page `reports/analyses/2026-11-27_analysis_p290-share-unified.md` at 4c2c6fe7, with the E330 check
(`2026-11-27_review_p290-e330-randoms.md`). No wrong number found, and no comparison is made without naming its task.
Two small items remain open, neither a MUST.

Earlier items:

- MUST 1 (no comparison without references; per task, with n draws): closed. Page :85-100 names each task and its draw
  count ("two draws", "three", "one draw each" on the neuron cipher and invented facts). :99-100 states plainly that on
  copy and known facts the randoms read as high on both decompositions. :6-7 says any comparison holds only task by
  task. No sentence says "re-gated more than random" without a task.
- MUST 2 (base-model rows are a determinism check): closed, :30-32 and :60.
- SHOULD 1 (multi-token numeric check): the check was done (`tests/test_share_unit.py::test_multi_token_against_brute_force`,
  passed on the E328 rerun and E330 pods), but the page's "Not run" :162-163 still lists it as not run and says E327
  tests only one-token answers. **Open (SHOULD): replace that bullet with "done: tests/test_share_unit.py, 1-6 token
  answers in float64".**
- SHOULD 2 (end token is part of the answer): closed, :14-15.
- SHOULD 3 (every sentence citing the old KL values): closed, :102-152 (13 items, incl. the abstract and the non-input file).
- SHOULD 4 (E327 as 47 of 48, cause unverified): closed, :21-25.
- E330 MUST (per task, n draws): closed as above. E330 SHOULD (dots, not ranges; capped draws named): the page shows
  every draw in a table (:76-83) and names the eight "no band" draws, their tasks and largest gains, and says leaving
  them out is not neutral (:68-71). The figure is not built yet (:165); the plan at :123-124 is one dot per draw with
  the capped draws in the caption. Closed on the page, to be checked on the figure. Note for writeup-lead: the
  `random_kl` column of `qwen_share_new.csv` is the mean of the read draws only (`build_csv.sh:7-8`). The figure must
  plot dots from `qwen_share_random_draws.csv`, not that mean, or the known-facts spread and the capped draws vanish.
- E330 (a)-(c) (1.03 draw, gain-1 reference not like for like, spread): closed, :62-66 and :91-94.

New items:

- **SHOULD: the opening paragraph :5-6** compares tasks without draw counts ("random changes read lower on cipher,
  invented facts and both maths tasks"). Cipher and invented facts rest on two VPD draws each. A clause "(two or three
  draws per task)" would meet the n-draws rule in the summary too.
- NIT: :93 "seed spread of 0.02 or less" for the known-facts fine-tunes. VPD reads 0.5765-0.5969, a spread of 0.0204.
  It rounds to 0.02, so this is fine as written.

Spot-checks (jq on the files the sidecar names), all match:

| Page | Value | Source |
|---|---|---|
| :79 VPD known draws | 0.687 / 0.515 / 1.031 | `qwen_random_vpd*.jsonl` numerator/gain = 0.6870, 0.5146, 1.0314; draw 2 numerator 1.0697, gain 1.0371 (:91) |
| :80 VPD cipher | no band / 0.375 / 0.351 | draw 0 status cap, gain 0.00016; draws 1, 2: 0.3750, 0.3509 |
| :22 E327 miss | 0.7519 to 0.7494, 0.00245 | `repro_67m.jsonl` e53/lowercase_s0 old 0.75188, new 0.74943; 47 of 48 within 0.002 |
| :25 67M range | 0.12 to 1.10 | `repro_67m.jsonl` new min 0.1156, max 1.0991 |
| :33-34 run-1 randoms | 11 of 12 negative, one at +1.19 | `qwen_reread.jsonl` run1 gains: only VPD gold +1.1916 |
| :16, :31 | answer counts; 16 rows, 20 for maths | `run_qwen.log:1-5` |
| :43-48 | fine-tune means and seed ranges | `qwen_share_new.csv` |
