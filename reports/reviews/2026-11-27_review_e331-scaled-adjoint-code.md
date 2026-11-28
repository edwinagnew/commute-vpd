# Pre-run check of E331, the scaled adjoint split on the built edits (review, 2026-11-27)

I read the code for run 2 of cell P1.67 before it runs. This run splits each built edit's weight change with the
adjoint split and its per-matrix scale, as the project's terms define it. Run 1 had left the scale out. The code applies
the scale the way the definition states it, the remainder still takes whatever the scaled pieces miss, and the first
stage would catch the likely set-up errors within minutes. Nothing blocks the run. Two small changes are recommended
below.

Scope: `src/commute/kstar_vpd.py` at 24dd506a (`_pieces`, `_site`, `featurize`), `experiments/adjoint_rho/run2.py`,
`experiments/adjoint_rho/results/predictions_run2.json` and `card_run2.txt` at dd096f16, and the helpers they call in
`src/commute/split.py` (`adjoint_split`, `adjoint_scale`, `dsigma`, `input_second_moments`, `weighted_energy`).

## What was checked, and what it reads

| Check | Finding |
|---|---|
| The scale is the GOALS Terms' c, per matrix | Yes. `adjoint_scale` gives c = <K(dW), dW>_Σ / <K(dW), K(dW)>_Σ, the minimiser of ‖(c K(dW) − dW) Σ^½‖. Both pieces are multiplied by the same c (`kstar_vpd.py` `_pieces`). |
| The remainder takes dW − c K(dW) | Yes. `_site` builds the write side c dU Vᵀ and the read side U c dVᵀ from the scaled pieces, and the remainder is dW minus both, so the pieces sum to dW. `split_fp64` records the gap. |
| Σ is on the matrix's input side | Yes. The weights are `nn.Linear`, laid out (d_out, d_in); `input_second_moments` hooks each layer's input and returns a d_in × d_in average. `weighted_energy(A, Σ)` is tr(A Σ Aᵀ), which matches that layout. A wrong orientation would also fail on the non-square MLP matrices at stage 0. |
| Σ is read on W0, on the item's own rows | It is read on `w.kl.net.model`, over the item's windows D. `with_delta` copies the network, so building an F does not change it. |
| A matrix with no first-order change | K(dW) at rounding level returns c = 0. The pieces are then zero and the remainder takes all of dW, so no variable is marked changed. This is the right reading. |
| Do-nothing | dW = 0 returns zero pieces before Σ is used, so rule R0a is unaffected by the scale. |
| Errors reach stage 0 | Yes. Stage 0 builds Σ and featurizes with it, both for the do-nothing and for the identity interchange on x2. A key mismatch between `w.keys` and the decomposition's keys, or a shape error, stops the job in its first minutes. |

## Recommended (SHOULD), none blocking

1. **Check W0 on the model Σ is read from, for every item.** `w.w0_check` is computed once, when the world loads. It
   compares `kl.M`, not `kl.net.model.target_model`, against the released W0. Nothing I read changes the decomposition
   model's target weights between items. Still, a single `torch.equal` of the target weights against W0 inside
   `featurized` would show that Σ was always read on W0. It costs a few milliseconds per item.
2. **The card and the filed expectation disagree on rival a.** `card_run2.txt` says rival a predicts x2 and x10 "near
   or above" their least-squares 0.76. `predictions_run2.json` says "above 0.761 by more than the scan noise 0.003".
   The run is descriptive, so neither wording is scored. The result page should quote the filed wording.

## Worth recording in the result page

- c can come out negative, when K(dW) points away from dW on the data. That flips both pieces. The expected range
  (0, 1) assumes otherwise. Report the per-matrix minimum of c, which `scales()` already saves.
- R, the share of dW's weighted energy that c K(dW) reproduces, is the number that says how far the adjoint family can
  reach for each edit. It belongs beside ρ in the figure, because a low ρ with a low R points to the split, not to the
  edit.

## Not run

- I did not run the code. The cluster's GPU node is timing out, and the card has the pod's stage 0 serve as the
  rehearsal.
- I did not re-derive E329's featurization path. It was checked in `2026-11-27_review_e329-adjoint-rho.md`, and the
  only new path here is the scale.
