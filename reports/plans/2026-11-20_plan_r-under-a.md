# Plan: R under wording A on the built edits, a matched random update, the nine pairs, E62 and E237 (E242, P1.46)

*2026-11-20, definition-lead (writer agent). Cell P1.46, one run, due 11-23. Card
`reports/plans/2026-11-20_plan_r-under-a-card.md`; filed rules `experiments/kstar_vpd_wording_a/results/predictions.json`;
code `experiments/kstar_vpd_wording_a/run.py`, `score.py`, `src/commute/stand_in.py`; Slurm
`cluster/kstar_vpd_wording_a.sbatch`.*

**The request** (user, 2026-11-20 ~19:50, "A, reduced redo on built edits and nine pairs"; decisions.md row
"Definition wording A adopted"; extended by the user to E62's copy/lookup pair and E237's key tasks). Wording A: "A
fine-tune F reuses the base model M, for a behavior, to the extent that M's mechanisms, each up to an output gain, can
stand in for those of the variables the behavior runs through and still preserve the behavior." As operations: at
E226's VPD featurization with site remainders and E226's circuit K*, R_A is the largest share of K* whose variables can
all take M's mechanism at once, each times one positive gain, inside F's forward, with the behaviour preserved. M is the
base 67M; F is a built edit, a fine-tune or a random update.

## How each part of the sentence is read

1. **"Preserve the behavior."** F's full forward with the set S stood in (every other variable keeps F's mechanism;
   nothing is ablated; whip-lead, 2026-11-20) must keep KL(F || F_S) at E226's read positions, mean over the windows D,
   at or under the circuit's own allowance, (1 - alpha) KL(F || F_all_ablated) with alpha 0.9. That is E226's bar for
   K*, so no new number enters. Standing S in inside the circuit instead (the rest mean-ablated) is the setting
   `stand_in_forward: "circuit"`, one line.
2. **"Up to an output gain."** One scalar per variable, fitted in closed form by least squares of M's write to F's
   write over every position of D in F's own run. Positive gains only (user, 2026-11-20 ~21:00): a variable whose
   fitted gain is zero or negative cannot stand in and keeps F's mechanism. An unchanged variable's gain is exactly 1.
3. **"The largest share."** E226's search code (`kstar_vpd.search`, 2,000 evaluations) over K*'s standable changed
   members, for the smallest set that must keep F's mechanism. R_A = 1 - (that set + the members that cannot stand
   in: grafts and gains <= 0) / |K*|, a lower bound when the search is capped. If every standable member stood in at
   once already passes, R_A is read without a search.

`src/commute/stand_in.py` is featurization-agnostic: a site is any object with `module`, `idx`, `gain_sums(y)` and
`correction(y, t, g)` (module docstring); `vpd_sites` is the VPD adapter, and sae-lead can write an SAE one.

## Stages, in order, each a separate submission

1. **Do-nothing**, first in every job: F = M, every gain exactly 1, everything stood in gives KL exactly 0, R_A 1.
2. **Matched random update** (control): E241's Gaussian per matrix (`gating_vpd/run.py random_dw`, seed 241001) at
   new_thin's update size (E240's common size), on x2's copy windows; its K* searched with E226's code. Beside it (phase C,
   rule P2k) E241's "kl" arm, seed 241002, scaled to new_few's KL(F‖M) at E240's matched size (0.159, on new_few's
   windows; likely under the bar there, about 0.37, so likely R_A about 1 and P2k failing, as P2); the control job
   takes about 26 min (up to 41 if both stand-in searches run), the built stage about 76 min in all.
3. **Six built edits** at their own size, E226's saved K* on all of D and its two halves (the band). This job refuses
   to start before the random update is read.
4. After stage 3 passes, separately: **E187's nine pairs** at matched size (76 items), **E62's copy/lookup pair**
   (6 states), **E237's 15 key items** (its saved VPD K*), all descriptive. Each refuses to start before stages 2-3.

## What each number should read (E226's and E237's saved item files)

| item | built as | KL(F‖M) | bar | 1 - f | expected R_A | old R |
|---|---|---|---|---|---|---|
| do-nothing | control | 0 | 0.377 | 1 | 1 exactly, no search | 1 |
| random | control | small | ~0.44 | ~0 | **1** if KL(F‖M) is under the bar | |
| x2 | reuse | 0.101 | 0.437 | 0.911 | 1 (everything stands in at once) | 0.9992 |
| moved | reuse | ~0 | 0.446 | 0.912 | about 0.99975 (its grafts) | 0.9989 |
| rebuilt | reuse | 0.120 | 0.432 | 0.626 | about 1 | 0.9954 |
| x10 | reuse | 0.714 | 0.438 | 0.907 | 0.99 or above | 0.9755 |
| new_few | new | 10.2 | 0.772 | 0.728 | 0.85 to 0.97 | 0.9942 |
| new_thin | new | 10.3 | 0.765 | 0.889 | 0.89 to 0.95 | 0.9963 |

Pairs, E62 and E237's items change nearly every variable (f about 0.9999) and sit 2 to 7 nats from M against bars of
0.3 to 0.65, so R_A can range over most of 0 to 1; no direction is guessed.

**Rules.** R0 stops the job on any failed control: per item E226's split and gate guards, KL(F || F_all_ablated)
equal to the saved value within 1e-4 relative, unchanged variables stood in leave F's logits exactly, and, for items
without grafts, every changed variable stood in with gain 1 gives M's logits within 1e-3 (the positive control). P1:
min R_A over the four reuse items minus max over the two new edits above the split-half band. P2: min R_A over the
reuse items minus the random update's above the band. D1 to D3: pairs, E62, E237, descriptive, beside the old R.

**P2 is expected to fail by construction.** A Gaussian update at new_thin's size moves KL far less than a targeted edit
(E241's plan); if KL(F‖M) is under the bar, every member stands in and the random update reads 1, level with x2. The
lead filed the rule as asked; a KL-matched random (E241's "kl" arm) would test it.

**Risk for x10 (P1's rival b).** The least-squares split spreads even a pure output rescale over many subcomponents.
In a toy check (one 5 x 6 matrix, four subcomponents), tripling one subcomponent's write changed all four and the
remainder, and the fitted gains (2.57, 1.07, 1.03, 1.08; remainder 1.0) leave a max output error of 1.87. Every gain
there is positive, so the positive-only rule changes nothing in the toy. Stood in with gain 1 the matrix matches M to
4e-6. x10 can read near its floor of 0.907, at or below new_thin.

## Runtime (one measured item time x count; cluster debug QoS, L40)

One stood-in evaluation is taken as one K* evaluation (stage 0 measures it and stops over 6 h): 0.46 s on 512 cue
windows and 0.107 s on 75 copy windows (E240's L40 logs, 928 s and 214 s per 2,000); E187 states 0.24 s (E226's H100
ratio, 297 s against 552 s). Each job adds about 4 min (world load and the do-nothing).

- Control: K* search 2,000 x 0.107 s = 4 min, stand-in search 1 to 2,000 evaluations, up to 4 min: about 12 min.
- Built, three parallel jobs: new_thin and new_few 3 x 2,000 x 0.46 s = 46 min each; four reuse items at most 3 x
  2,000 x 0.107 s = 11 min each. **Built stage wall time: about 65 min** (12 min control, then about 50 min).
- Pairs: 76 items (`group_items`, checked on CPU): 36 smaller-side states with E226's saved K* x 9 min = 5.4 h; 40
  larger-side states x 17.5 min (K* search, then stand-in) = 11.7 h; sixteen jobs of at most 5 items, at most 92 min.
  E62: 6 x 9 min, one job (copy states already read in the pairs are skipped). E237: 15 x 9 min, two jobs of about 70 min.
- In all about 23 L40-hours in 23 jobs of under 2 h, $0.

## Not run

Standing S in inside the circuit; gains fitted on the behaviour rather than the write; fractional stand-in; an
uncapped search; variables outside K*; a KL-matched random update; E237's random controls; any other split.
