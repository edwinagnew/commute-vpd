# B's rate grid and T0's line after the size-scaling rehearsal — advice (2026-10-28, read-only)

**Plain words.** The two rehearsals show that the largest usable teacher-training rate falls about in proportion to
the number of alive parameters, so at full size every rate in the filed grid should blow up by step 25 and B would
stop the paid run with nothing learned. The cheap fix is to extend the grid downwards by the measured law before the
run, as a dated amendment, and to set T0's pass line from a measurement instead of the unsourced 0.5. Both can be
checked for $0 on a cluster L40 in under half an hour before any pod starts.

## What the rehearsals say (code 5c49414, `results/amend1_rehearse_e40a1_r3_c16/`, `_c384/`)
- Divergence boundary: 16 columns per site, 3e-5 fine (0.94 at step 25), 1e-4 x3.3 (check_teacher) → ~5e-5. 384
  columns, 1e-6 fine (0.87), 3e-6 x1.55, 1e-5 x4.4 → ~1.7e-6. A 24x size cut the boundary ~30x: **lr_max ∝ n^-1.1**,
  steeper than the 1/√n I guessed earlier (α itself went 2.79e-3 → 7.4e-4, ∝ n^-0.42, as expected). The likely
  reason: Adam's per-coordinate steps are sign-coherent across the alive set on 8 rows, so the step's effect on the
  output grows with n, not √n. Expected full-size boundary (5.7x more than 384): ~2.5e-7 (1/n) to ~7e-7 (1/√n).
- The good rate sits 0.2-0.6x below the boundary (16 col: 1e-5 best, 0.718 at 100; 384 col: 1e-6, 0.788 at 100).
  Full size: ~5e-8 to 4e-7. The filed floor 1e-6 is above the boundary under either law.
- T0 at 16 columns: 0.57 at 300, 0.53 at 600; floor, not still falling (0.535 → 0.5285 over the last 100). 0.5 is
  not reached at the small size and will not be at full size. Per-row: at 384 columns the largest maths row
  (0.018 of the 0.08 total) barely moved in 100 steps (0.0181 → 0.0170); the fall is carried by the other 7 rows.

## Options
1. **Extend the grid by the measured law (recommended).** `B_rate.rates` = {1e-6, 3e-7, 1e-7, 3e-8}, everything
   else as filed (abort at 25 above 1.2x, pick the lowest fall at 100). 1e-6 stays as the continuity point and
   should abort (25 steps); 3e-7 and 1e-7 bracket both scaling laws; 3e-8 catches a steeper one. Source line:
   "boundary 5e-5 at 16 columns, 1.7e-6 at 384 (jobs r3_c16, r3_c384), ∝ n^-1.1; full size 5.7x 384". Cost on the
   H200 (step time still unmeasured, plan's 1-3 s): ≤ 325 steps + four α evaluations ≈ 15-25 min, ~$1.5. Dated
   amendment (amended_4) and the user's OK before the pod.
2. **Re-set T0 from measurement, not 0.5 (recommended with 1).** First rehearse T0 at 384 columns at 1e-6 to 600
   steps ($0, ~10 min: B's 384-col step is 0.7 s). Then file `fall_max` = that curve's 300-step reading + 0.1
   (expect ~0.7-0.75), keep the 300 → 600 "still falling" extension, keep the FineWeb band, and report the 600-step
   fall unscored. T0 is an apparatus check (gradient flows toward a reachable target, student not harmed); the depth
   of the fall is an Adam floor T1's P3/P5 do not read, so it should not burn the pod. Cost: ~$2 for 600 steps.
3. **Change the parameterisation so the rate does not scale with n** (e.g. lr ∝ 1/n_alive built into the
   optimiser, or a trust-ratio step). Principled, but new unrehearsed training code the day before a gated run;
   option 1 is the same thing done by hand for one n. Defer to the backlog; revisit if the Qwen line runs many sizes.
4. **As filed.** B stops after ~$3 of set-up with "none qualifies". Nothing learned about the recipe. Not advised.

## What a further $0 L40 rehearsal settles first (all < 30 min, debug QoS)
- (a) B at ~96 columns, grid {3e-5, 1e-5, 3e-6, 1e-6}: a third size point, 6x from each existing one, pins the
  exponent (1/√n predicts a boundary ~2e-5 there, 1/n ~8e-6) and so which end of the new grid the full run uses.
- (b) T0 at 384 columns, 1e-6, 600 steps: sets option 2's line and shows whether the fall at the biggest size the
  card fits is slower than at 16 columns (0.788 vs 0.718 at step 100 hints it is).
- (c) From (b)'s per-row log: if one row holds over half the KL and does not move, score the median over rows.
  A 480-column cut is a weak lever (1.25x over 384; 37 GB peak at 384, ~46 GB at 480) and not worth the slot.

## What would change the recommendation
- (a) shows a boundary ≥ 2e-5 at 96 columns (1/√n): drop 3e-8, grid {1e-6, 3e-7, 1e-7}; the paid B gets cheaper.
- Full-size step time > 4 s in B's first arm: cut the grid to three rates before T0 (the projection rule does it).
- (b) reaches ≤ 0.55 by 300 at 384 columns: keep 0.5 at 600 as the line (then it is sourced); otherwise option 2.
- The paid B's chosen rate reads > 0.85 at step 100: the recipe, not the rate, is the problem (Adam floor, 8-row
  batches); stop and ask before T0, as the third review's guard already says.

Friction: the full-size step time and peak memory are still unmeasured, so every dollar figure here is the plan's
1-3 s guess; `kl_rows` made the per-row reading possible this time, but the 384-col run has no T0, so the line in
option 2 cannot be set from files today.
