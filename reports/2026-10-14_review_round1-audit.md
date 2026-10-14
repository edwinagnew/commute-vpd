# Review: code audit of the transport test, round 1

*2026-10-14, coding agent. The user asked for a careful code audit before anything further, after several bugs
were found mid-run. Two subagents audited the code read-only:

- the transport, fine-tune and data path, behind predictions 1a, 1b and 2;
- the faithfulness and PGD path, behind predictions 3 and 4.

Each wrote CPU tests on small real `param-decomp` models: `tests/test_transport_invariants.py` (23 tests) and
`tests/test_faithfulness_invariants.py` (12 tests). `make test` passes 37 tests. The lead agent re-derived the two
findings below that change a reading. The filed predictions are not edited.*

## Verdict

**No code bug changes a scored verdict.** The scored numbers in the analysis report match the committed JSON.
Several possible failures were checked and ruled out:

- a silent no-op in the transport, fine-tune or window code;
- an off-by-one error in a positions mask;
- leakage of held-out windows into training;
- dependent seeds for the null;
- a wrong norm match for arm N;
- a bf16 bias on the 0.014-nat arm-B effect (the hooked and the native KL agree to 0.1%);
- a stale calibration used in the faithfulness ratios.

Applied to the code at `d0ed6c5`, the faithfulness tests fail 3 of 12, so they catch the historical PGD bug. The
PGD history is settled in the erratum to `2026-10-14_decision_pgd-scope.md`. Only the adversary's scope moved the
number; the "grad-disabled decorator" explanation given in chat was wrong.

Two readings in the analysis report are wrong, though, and one of them is a qualitative surprise.

## 1. The arm-B functional fits failed to optimise; they did not overfit (prediction 2, part ii)

- **What the record says.** The analysis report says training-set capture was not measured, so overfitting could not
  be told apart from under-parametrisation. `TRIED.md` says "functional fits overfit". The notebook says
  training-set capture was not recorded. All three are wrong.
- **What the fit losses show.** `transport_B.json` records `fit_loss_first_last` for every fit. The first value is
  the KL with a zero edit, i.e. the starting point. At the scored step 100, all 10 top-k T1 fits end with a *higher*
  training KL than they started with.
- **What the logs show.** The losses spike during the fit:
  - T1 write k=10: 0.018 → 0.32 → 0.044 (`results/large/logs/all.log`).
  - T0 functional: 0.012 → 0.046 → 0.014.
- **Cause.** Adam at lr 1e-2 is far too large a step for a 0.014-nat target. The same code converges on arm A (41 of
  44 fits), and the CPU tests show descent.
- **Consequence.** Capture −0.09 is a failure of the optimiser, not evidence about whether a rescaling of
  subcomponents can represent the update. By the filed rule prediction 2(ii) still reads "fail", but that half
  carries no information. Prediction 2 stays reversed on its Frobenius half, which does not depend on any fit.
- **Arm A at k ≥ 30.** "Overfitting 365 windows" is also contradicted: at k = 100 the training loss rose, 0.188 →
  0.242. That is the same optimiser failure.

## 2. Something unexpected: arm A's update moves into the span as training continues

The analysis report says δW is orthogonal to the dictionary "at every checkpoint". In absolute terms that holds:
under 0.4% pooled, at most 1.1% in any matrix. But relative to a random direction of the same per-matrix norms, the
two arms behave differently:

| step | 10 | 30 | 100 | 300 |
|---|---|---|---|---|
| arm A, fraction / random expectation | 1.7 | 2.3 | 5.4 | **10.2** |
| arm B, fraction / random expectation | 1.8 | 1.8 | 1.5 | 1.4 |

- **What it means.** The random expectation is $\sum_m w_m\, n_{\text{alive},m}/(d_\text{out} d_\text{in})$, where
  $w_m$ is matrix m's share of $\|\delta W\|_F^2$. It matches the measured arm-N values (0.000346 against 0.000342).
  So by step 300 arm A meets prediction 2's 10× threshold. That prediction was filed for arm B, where the ratio
  falls.
- **No explanation yet.** The emoticon fine-tune keeps training long after P(o) has saturated at 0.999 (step 10), so
  most of its later δW is driven by the KL penalty on the surrounding 40 positions.

## 3. Numbers in the analysis report that do not match the JSON

| Where | Says | JSON |
|---|---|---|
| Reading, measure 1 | per-matrix fraction "never exceeds 0.13%" | 0.15% at A step 10 (`h.2.attn.o_proj`); 1.1% at A step 300 |
| Reading, emoticon arm | k = 3 random 0.59 to 0.83, "random spread covering the top value" | 0.586 to 0.825; the random maximum is below top-k (0.832) |
| Reading, pretraining arm | k-sparse edits "−0.16 to −32" | −0.049 to −32.3 |
| Reading, faithfulness | unmasked KL without Δ under the null transport "0.034 to 0.063" | 0.021 to 0.046 (0.063 is T0 functional at A step 10) |
| What was run, arm A | KL to W0 on the other 40 positions "0.04 to 0.05" | 0.037 to 0.053 |

## 4. Cosmetic

- Arm B's arm-N null used arm B's own δW norms rather than arm A's, as filed. Ratio 1.51 against 1.54 with arm A's
  norms; the verdict is unchanged.
- Data overlaps that are not leakage:
  - the faithfulness rows (val 0..255) include arm B's held-out rows (val 64..127);
  - the window scan (train 0..32767) overlaps arm B's training rows (8192..12991).
- `plot.py`:
  - line 67 clips arm B's read k = 10 point (−3.6) without a note;
  - the arm-N line is drawn once per arm and labelled only "matched to arm A".
- `check_pgd.py`:
  - line 73's help text describes no-grad modes that it does not run;
  - line 128 calls `float()` on a grad tensor.

## What this changes

- **Prediction 1a:** stands as a reversal, and as a test with no power.
- **Prediction 2:** stands as reversed on its Frobenius half only. Its functional half is uninformative until it is
  re-fitted with a convergence check: a smaller learning rate, and a final training loss below the starting loss.
- **The arm-A trend** (section 2) is an open question for round 2, which goes through `/design`.
- **Not re-run.** Round 1 is not re-run as filed. No bug invalidates its scored numbers, and the user's clarified
  question (2026-10-14) changes what round 2 should measure. In particular, the faithfulness measure lets Δ absorb
  δW, so the trivial "do nothing" map passes it.
