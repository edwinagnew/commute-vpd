# compare-lead digest 2026-11-12 (evening): E133 figure filled

P2.23 (due 11-12), P2.24 (due 11-13) and P2.25 (due 11-13) are all filled. The E133 figure is live in the research log (v335). Every cell has a value except two label twins in the "neurons for half the gain" column, which are read in a different unit. Spend was $0.

## Cells
- **P2.23** (0 runs): the figure redrawn under one colour rule in every column. A cell is "apart" only when the seed ranges do not overlap, which moved 23 cells from no call to apart. The card's method table now says how each column is measured.
- **P2.24** (1/1 runs, E172, Slurm 59606, 2.5 min): neurons for half the gain on all nine built pairs. The reuse side needs fewer neurons on six pairs, with gaps of 0.52 to 0.90 in log10, and there is no difference on three. Five of the six also have a clearly larger new-side update.
- **P2.25** (1/1 runs, E171, CPU): effective rank on the four 96-key rows. All four are apart, reuse side higher.

All other compare-lead cells are filled or closed.

Pages:
- `reports/analyses/2026-11-12_analysis_p2-23-e133-redraw.md`
- `reports/analyses/2026-11-12_analysis_p2-24-neurons-for-half-gain.md`
- `reports/analyses/2026-11-12_analysis_p2-25-refit-rank.md`

## For the 11-18 gate
- **Premise doubts on P2.23 and P2.24:** a column coloured "apart" does not show what a method adds beyond update size. Five of the six k_half separations come with a larger new-side update.
- **Unexplained number:** the k_half value for copy maps against random maps was relayed as +0.48. We read 0.70 as the ratio of means and 0.60 as the mean of logs.

## Asks waiting on the user
None.

## Friction noted today
- The auto-mode classifier blocked `fleet/cell add` again.
- `final.json` does not record each column's sign convention.
- The size cells of the 96-key rows use E58's norm, whose unit does not match the refit's.
