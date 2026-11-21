# Plan: R under wording A on the built edits and the nine pairs (E???, cell pending)

*2026-11-20, definition-lead (writer agent). Draft: the cell awaits `fleet/cell approve 42`, so no E-number is
claimed and nothing is filed or submitted. Card `reports/plans/2026-11-20_plan_r-under-a-card.md`; draft rules
`experiments/kstar_vpd_wording_a/results/predictions.json`; code `experiments/kstar_vpd_wording_a/run.py`,
`src/commute/stand_in.py`; Slurm `cluster/kstar_vpd_wording_a.sbatch`.*

**The request** (user, 2026-11-20 ~19:50, "A, reduced redo on built edits and nine pairs"; decisions.md row
"Definition wording A adopted"). Wording A: "A fine-tune F reuses the base model M, for a behavior, to the extent that
M's mechanisms, each up to an output gain, can stand in for those of the variables the behavior runs through and still
preserve the behavior." Restated as operations: at E226's VPD featurization with site remainders and E226's circuit
K*, R_A is the largest share of K* whose variables can all take M's mechanism at once, each times one gain, inside F's
forward, with the behaviour preserved. M is the base 67M; F is a built edit or a fine-tune.

## How each part of the sentence is read (the three grading choices)

1. **"Preserve the behavior."** F's forward with the set S stood in (every other variable keeps F's mechanism; nothing
   is ablated) must keep KL(F || F_S) at E226's read positions, mean over the windows D, at or under the circuit's
   own allowance, (1 - alpha) KL(F || F_all_ablated) with alpha 0.9. This is the bar E226 filed for K*, so no new
   number enters; it is a tenth of what ablating everything costs (0.32 to 0.77 nats on these items).
2. **"Up to an output gain."** One scalar per variable, any sign, fitted in closed form by least squares of M's write
   to F's write over every position of D in F's own run: g = sum <write_F, write_M> / sum |write_M|^2. It is 1 exactly
   for an unchanged variable.
3. **"The largest share."** E226's search code (`kstar_vpd.search`: attribution order, bisection, two re-rankings, a
   chunk polish, 2,000 evaluations) run over K*'s changed members for the smallest set that must keep F's mechanism.
   R_A = 1 - (that set + K*'s grafts) / |K*|. Unchanged members stand in at no cost; grafts (neurons M lacks) never
   can. When the search hits its cap, R_A is a lower bound. If every changed member stood in at once already passes,
   R_A is read without a search.

Choice 1 departs from a reading inside the circuit (the other variables mean-ablated, as K* was found): E226's
polished K* sits just under its bar, so inside the circuit almost any change would fail and R_A would read the polish's
slack. This is put to the lead as a definition question.

## What each number should read (from E226's `results/items/*.json`)

| item | built as | KL(F‖M) | bar | 1 - f (floor of R_A) | expected R_A | old R |
|---|---|---|---|---|---|---|
| do-nothing | control | 0 | 0.377 | 1 | 1 exactly, no search | 1 |
| x2 | reuse | 0.101 | 0.437 | 0.911 | 1 (everything stands in at once) | 0.9992 |
| moved | reuse | ~0 | 0.446 | 0.912 | 1 - 8/31,888 = 0.99975 (grafts) | 0.9989 |
| rebuilt | reuse | 0.120 | 0.432 | 0.626 | about 1 | 0.9954 |
| x10 | reuse | 0.714 | 0.438 | 0.907 | 0.99 or above (gains take up the ten-fold) | 0.9755 |
| new_few | new | 10.2 | 0.772 | 0.728 | 0.85 to 0.97 | 0.9942 |
| new_thin | new | 10.3 | 0.765 | 0.889 | 0.89 to 0.95 | 0.9963 |

Trained states: f is about 0.9999 and KL(F‖M) 0.27 to 6.6 against bars of 0.32 to 0.65, so R_A can range over most
of 0 to 1; no direction is guessed.

**Rules.** R0 stops the job on any failed control: the do-nothing first (every gain exactly 1, every variable stood in
at once gives KL exactly 0, R_A 1); per item, E226's split and gate guards, KL(F || F_all_ablated) equal to E226's
within 1e-4 relative, every unchanged variable stood in leaves F's logits exactly, and, for items without grafts,
every changed variable stood in with gain 1 gives M's logits within 1e-3 (the positive control: an independent
forward). P1, filed by the cell: min R_A over the four reuse items minus max over the two new edits above the
split-half band (E226's two halves of D, each with its own K*). D1, descriptive: per pair, reuse side minus new side
on R_A and on the old R, at matched size.

**Risk for x10 (rival b).** The least-squares split spreads even a pure output rescale over many subcomponents (a toy
check: doubling one subcomponent's write changed all four and the remainder), so per-variable gains need not undo it.
x10 can then read near its floor 0.907, at or below new_thin.

## Runtime (measured item time x count; cluster debug QoS, L40)

One evaluation of the stood-in forward is taken as one K* evaluation (stage 0 measures it and stops over 6 h):
0.46 s per evaluation on 512 cue windows and 0.107 s on 75 copy windows (E240's L40 logs, 928 s and 214 s per 2,000);
E187 states 0.24 s (E226's H100 ratio, 297 s against 552 s, applied to 0.46 s). Per search 2,000 evaluations.

- Built items: new_thin and new_few 3 searches x 2,000 x 0.46 s = 46 min each; x2, x10, moved, rebuilt 3 x 2,000 x
  0.107 s = 11 min each at most (x2, moved, rebuilt likely one evaluation): 44 min. Three jobs.
- Pairs: 68 states (40 larger-side states at matched size, 28 distinct smaller-side states). Smaller side: one
  search, 2,000 x 0.24 s + 1 min = 9 min, x 28 = 4.2 h. Larger side: E226's K* search first (8 min) and then the
  stand-in search: 17.5 min, x 40 = 11.7 h. Twelve jobs of 5 or 6 states, about 1.5 h each.
- Each job adds about 4 min (world load and the do-nothing). Total about 18 L40-hours in 15 jobs of under 2 h; with
  four L40s free, about 6 h of wall time; $0. On four H100 pods about half (9 GPU-hours, ~$25, ~2.5 h), but the 68
  saved states would have to be uploaded first.

```
for t in "known0 --items new_thin" "known1 --items new_few" "known2 --items x2 x10 moved rebuilt"; do sbatch --comment="E### ${t%% *}" cluster/kstar_vpd_wording_a.sbatch $t; done
for i in $(seq 0 11); do sbatch --comment="E### pairs$i" cluster/kstar_vpd_wording_a.sbatch pairs$i --pairs --shard $i 12; done
```

## Not run

Reading inside the circuit (choice 1's alternative); gains fitted on the behaviour rather than the write; fractional
stand-in; an uncapped search; variables outside K*; the trained pairs at their own sizes; any other split.
