# P1.67 (due 11-28): redteam of E329, retained share at VPD under the adjoint split

E329 read the retained share on the built edits with each fine-tune's weight change split by the adjoint split instead of least squares. The code applies the split correctly: the pieces sum to the weight change, the controls pass, and least squares reproduces the earlier reading. But the run used the adjoint split without its scale. `GOALS.md` defines the adjoint split with a per-matrix scale, and the cause regating-lead proposes for the result is exactly what that scale exists to remove. So the paper may say the unscaled adjoint does not separate the built edits. It may not say "the adjoint split" does.

Verdict: **code OK, readings as reported; MUST 1 on the paper line; recommend run 2 be the scaled adjoint (needs the user's yes).**

## What was checked (main 31be0ad9)

| Item | Finding | Level |
|---|---|---|
| `kstar_vpd._pieces` / `_site` (src/commute/kstar_vpd.py:104-131) | The layout is Ua = U.T (d_out, C) and Va = V (d_in, C). `split.adjoint_split(dW, Ua, Va)` returns dU = dW Va and dV = dWᵀ Ua, which is the Terms' dU = δW V, dV = δWᵀ U. The write and read sides are dUa Vaᵀ and Ua dVaᵀ, the same expressions the least-squares arm uses. The remainder is dW - (write + read), so the pieces sum to dW by construction, and `split_fp64` checks that. The same `_site` serves both arms, so the split is the only difference. No bug found. | OK |
| Fresh search of the needed set per arm | Stage 2 searches the needed set on the adjoint featurization (run.py:157-164) and also reports E_b over the least-squares set. Set sizes are within a few hundred of the least-squares ones (x2 38935 against 38935; new_few 31263 against 31161). | OK |
| Controls | Do-nothing reads 1 under both splits. The adjoint identity swap flips 0 answers (largest deviation 5.1e-5). Least squares reproduces the earlier reading (E268) within 2e-9 to 1e-7 relative on all seven items. | OK |
| Readings | Least squares → adjoint: x2 and x10 0.761 → 0.207, rebuilt 0.623 → 0.012, new_few 0.013 → 0.013, new_thin 0.019 → 0.013, randoms 0.011 → 0.0016 and 0.0018. These are rival b's "anything else" (the gain edits lower), as filed, and they do not meet run 2's condition. | OK |
| Which adjoint split | `GOALS.md` Terms defines the adjoint split as "Applied as a state with a per-matrix scale c = argmin ‖(c K(δW) − δW) Σ^½‖ (user-confirmed 2026-10-20)". E329 used c = 1, as the cell's wording (`split.adjoint_split`) allowed. On x2 the unscaled pieces have about 42× (write side) and 8.5× (read side) the energy of δW, cancelled by the remainder (60×) and a cross term (-131×). That overshoot is the mismatch c is defined to correct. The proposed paper line, "tried on the built edits, it does not separate them (gain edits 0.21 vs least squares 0.76)", would describe the Terms' split, which was not run. | MUST 1 |
| Cause | "Pieces many times \|ΔW\|² that cancel" is stated as unverified, which is right. A scaled run is the direct test: with c, the pieces shrink to roughly δW's size. | NIT |

## Asks

1. MUST 1 (regating-lead, then writeup-lead): the paper line names the *unscaled* adjoint split (c = 1) and says that the Terms' scaled split was not read on these edits. For example: "Splitting by the adjoint split without its scale does not separate them (gain edits 0.21 against 0.76 under least squares); the scaled split was not tried."
2. Recommendation (whip-lead to put to the user, since it changes run 2's design): P1.67 has one run left. Spend it on the scaled adjoint (`split.adjoint_scale`, Σ the input second moment per matrix) on the same seven built edits, not on the trained fine-tunes as the cell's run 2 is written. It costs the same as run 1's stage 2, and it says whether the adjoint split as defined separates the built edits.

## Not run

- Any scaled-adjoint reading.
- `experiments/adjoint_rho/run.py` read in full: I read the stage-2 path and the featurization, not stages 0-1 line by line.
- A check that `shares_of_dW2` sums to 1 on every matrix: I checked it on x2's totals only (42.1 + 8.5 + 59.7 + 22.0 - 131.3 = 1.0).
