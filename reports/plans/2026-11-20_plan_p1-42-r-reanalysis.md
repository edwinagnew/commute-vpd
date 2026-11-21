# P1.42: does a different way of computing R order the built edits? ($0 reanalysis)

*2026-11-20, definition-lead. Cell P1.42, card `reports/plans/2026-11-20_plan_p1-42-card.md`. Code
`experiments/r_reanalysis/`. Saved files only, no GPU, no model load. A description: no verdict on the definition.*

## The request, in the Terms' words

The user (2026-11-20 ~18:00): R "fails at the first experiment"; consider changing how R is measured. Restated: from
E226's saved per-variable grades and behaviour weights, recompute the reuse score two other ways and say whether either
puts every built reuse edit (x2, x10, moved, rebuilt) above both built new edits (new_few, new_thin), and whether its
gaps on E187's nine pairs still follow update size. Beside them, list which already-saved simple readings do.

## Measures

All read E226's saved `items_*.npz` (`experiments/kstar_vpd/results/large/`): `s` (row 1 read positions, main;
row 0 every position, side), `main_in_K_polished`, `main_attribution` (row 0: signed d KL(F‖F_c)/d c_u at c = 1/2,
the same point for every item, main; row 2, at the search's best prefix, side).

1. **R as filed.** Mean of s over K*. Must reproduce E226's `per_item.R_read` to 1e-6 (code check).
2. **Weighted R over K*.** Sum of |w_u| s_u over K* divided by the sum of |w_u| over K*.
3. **Weighted R over all variables.** The same sum over every variable, no K*.
4. **Size-removed R.** Fit log(1 − R) = a + b log(update size) over every changed state (the six built edits and
   E187's 68 states), by least squares; an item's size-removed score is minus its residual (higher = closer to M than
   its size predicts). Done for measures 1-3. Update size is ‖δW‖/‖W0‖ over the 24 matrices: E187 states' saved
   `size.whole`; for the built edits sqrt(Σ dW2 / ‖W0‖²), with ‖W0‖² from the E187 states (constant across them).
   At blocks (E232): each pair's matched gap minus the size-only c² rule's prediction (`gap_c2_law`).
5. **Simple readings, saved only.** Update size (smaller = reuse), KL(F‖M) at the read positions (smaller = reuse),
   held-model survival S reading 3 (E197/E205), gain-only reach (E231, compare-lead, read only).

Band for 2-3: the largest half-1 minus half-2 difference over the six built edits, from the saved half-D circuits and
attributions (grades are from all of D). Pairs: gap = reuse-side mean − new-side mean; "follows size" = Spearman of
the nine gaps against minus the update-size gap (E226 read 0.87 for measure 1).

## What each number should read (written before computing)

- **Do-nothing.** Measures 1-3 read exactly 1 (every s is 1). Update size 0, KL 0. Measure 4 is undefined
  (log of 0) and not read.
- **If a measure works.** The lowest of x2, x10, moved, rebuilt above the higher of new_few, new_thin, by more than
  the band; and on the pairs a Spearman against size well below 0.87.
- **Rival b (the grade is the fault).** Measures 2-3 keep x10 lowest: E226 found x10's changed variables average a
  grade of 0.74 against 0.97-0.98 for the new edits, and the ten-fold edit sits on copy-path variables the copy behaviour
  uses, so weighting by use should push x10 lower, not higher. Measure 4 could lift x10, since its update is the
  largest of the six (about 0.15 against 0.01-0.05), but moved (s = 0 on its eight appended neurons at the size of x2)
  and rebuilt may then fall.
- **Simple readings.** Update size fails (x10 is the largest update). KL(F‖M) orders the six (reuse 0 to 0.71 nats,
  new about 10), but only because the new edits teach a different behaviour: it is a positive control that the items
  differ, not a reuse reading. S and gain-only reach were not read on most built edits; the table says "not read".

## Not run
- Weighted R at the coarse levels (no per-variable behaviour weights saved there).
- Any exact single-variable ablation weight (none saved; the saved weight is first order).
