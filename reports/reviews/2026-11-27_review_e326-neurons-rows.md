# Review: the top-neuron rows of Fig 2 on all 16 known-answer tasks (E326, cell P1.66)

Redteam by a fresh reviewer who wrote none of the code. I checked the neuron-hook fix, the evidence for its cause, the
rerun's 72 items against the rows and the figure, and the old-against-fixed difference. Everything was read with git,
jq and awk; nothing was rerun. The rows are sound and the figure shows them faithfully. Two SHOULD items remain, both
about stating the convention: 24 of the panel's 96 points are still read under the old hook, and the README does not
mention the hook change.

Scope: `src/commute/needed_neurons.py` at d2d81436, `experiments/fig2_all_tasks/` (run.py, score.py, plot.py,
ident_straddle.py), results at 167ca7fe and 6433be77.

## MUST

None.

## SHOULD

**S1. The top-neuron panel mixes two hooks, and nothing says so.** The panel has 96 points: 72 come from E325 (the
fixed hook), and 24 come from the e58 overlap states (`source` = E290 in `definition_rows.csv`, which E298 carried),
read under the old hook. `plot.py`'s docstring says "The top-neuron panel is read under the fixed neuron hook (E326
rerun 2, d2d81436)", which is true of only 72 of the 96. On the 72 items, the measured old-against-fixed difference
is at most 3.8e-6 absolute and 4.2e-4 relative (S-check 4 below). The e58 repro under the fixed hook moves by 3.4e-4
relative. So the mixing cannot be seen at the figure's scale (0 to 0.07). It is still a difference in convention
between points in one panel, and the project rule is the same runs behind compared panels. There are two fixes:
- (a) Say it, in the README's "Other departures from E298" and in `plot.py`'s docstring: the 24 e58 rows are read
  under the old hook, and the measured bound on the difference is given.
- (b) Read the 12 e58 states and their randoms again under the fixed hook. The repro took ~40 s per item, so 24 items
  take about 16 min on one H100. The `debug` QoS should do.

I recommend (a) at least. Do (b) if the paper's text claims one convention for the panel.

**S2. The hook change is not among the README's departures.** `experiments/fig2_all_tasks/README.md` → "Other
departures from E298" does not mention that the neuron swap now reads x's activation from F's cached run, not from the
rerun. That is a departure from E290 and E298's neuron protocol. It is also why the filed neuron reproduction (tol
5e-5) now misses by design, recorded under ask 60. The README should give one line each for the change, its reason
(the identity straddle) and the repro note. A smaller point in the same vein: the "Interchange" paragraph of the
module docstring in `needed_neurons.py` still does not say which copy of x's activation is split. The SAE `Swap`
docstring states this for its own case.

## OK (with evidence)

**1. The fix is correct and narrow.** In `Swap.mlp`, the only change is `cur`. Before, it was x's activation taken
from the rerun's `inp[0]`. Now it is `self.a[ex[...]]`, F's cached run on x, for the swapped neuron entry and for the
whole row of a remainder swap. `new` is still `self.a[pt[...]]`, so x′'s value comes from x′. `ex` is per row:
`scan_sites` calls `s.set(v, ex[k], pt[k])` and reruns from `run.blocks[s.layer][ex[k]]`, so the cache row and the
batch row are the same window. A neuron or remainder row changes nothing upstream of `down_proj` in its layer, because
the head hook touches only rows with `var < H`. So `inp[0]` equals the cache up to batch rounding. ident_straddle
measured that noise at ≤ 1.1e-4. The fix therefore differs from the old hook in two ways only:
- at straddling entries, by a whole bound;
- elsewhere, by rounding noise on the swapped entry or remainder row.

Unswapped neurons in the row still come from the rerun. This is the same fix already made to the SAE swap on
2026-11-24 (`needed_sae.Swap` docstring: "x's side read from the recomputed out let batch rounding flip a JumpReLU
gate at threshold").

**1b. The identity does not pass trivially, though it is now blind to one class of bug.** With x′ = x,
`(cur - s(cur)) + s(cur)` returns `cur` exactly. So the identity no longer tests the split itself. It also could not
catch a hook that read x where x′ belongs. Three facts show the hook does use x′:
- K is non-empty and large on every item (|K| 3,466 to 8,139);
- K under the fixed hook matches K under the old hook exactly on 50 of 72 items, and is off by ≤ 2 on the rest;
- the e58 repro matches E298's K on 4,170 of 4,171 entries.

A hook that ignored x′ would give K = ∅. The neuron line has no must-flip control like the SAE's, but this agreement
does that job.

**2. The stage-0 records and ident_straddle.json support the cause.**
- Old-hook identity: max_dev was 0.02715 on e62/copy_s0 in rehearse, neurons_a and neurons_b.
- Fixed-hook identity: max_dev was 7.81e-5 in fix and fix2, under the filed ~8e-5, with 0 flips throughout.
- `ident_straddle.json` reproduces the old values exactly with an `OldSwap` whose `mlp` matches the pre-d2d81436 code
  line for line: 0.027154 on e62/copy_s0 and 0.069206 on e46/ident32_s0. It finds 1 and 2 straddles there, and 0
  straddles on e58/copy_s0 and c3/copy_s0, whose old max_dev was already 4.6e-5 to 5.1e-5. Under the fix all four sit
  at 5.1e-5 to 7.8e-5.
- The expectations were filed in d2d81436, before the run.

The straddle count is not tied to the deviating case, since the script does not log the max_dev of the straddling
cases. Even so, rerun noise of ≤ 1e-4 cannot produce a jump of 0.03 to 0.07 except through a jump of a whole bound,
and the jump appears exactly where straddles occur. The cause is well supported.

**3. The 72 items, the rows and the figure agree.**
- `items/neurons/` holds 72 JSONs: 12 new tasks × seeds 0-2 × trained and random, 6 per task.
- `definition_rows.csv` has 96 `top` rows: 72 with `source` E325 and 24 with E290 (the e58 overlap). There are no
  duplicate (item, side) keys.
- The item `E_b_K` values joined to the E325 CSV values on (item, side): 72 of 72, keys identical, maximum absolute
  difference 0. The CSV is computed from the npz, so the npz and the fixed JSONs agree.
- `summary.json`: n_rows 288 of 288, `missing` empty, and the E298 check holds on 52 of 52.
- `run_neurons_fix2.log` shows 72 "E_b over K" lines, no traceback and exit 0. Its stage-0 identity (7.8e-5) shows the
  fixed hook was loaded in that process.

**4. The old-against-fixed difference is as small as claimed.** I compared each item in 167ca7fe^ with 167ca7fe: 72 of
72 old items are present.
- max |ΔE_b over K| = 3.776e-6 (e46/ident32_s1), matching the lead's 3.8e-6;
- max relative difference = 4.2e-4 (c3/rank1_s2_rand);
- |K| differs on 22 of 72 items, by at most 2 (the lead's "≤ 2 on 22 of 72" confirmed), and n_changed_K differs on
  the same 22;
- leftovers dropped: max |ΔE_b| = 2.8e-5.

**5. The figure does not misrepresent the rows.** `figures/fig2_all_tasks.csv` has 96 top points, and they equal the
CSV rows value for value (after stripping CRLF). Every value lies inside the shared x range [0, 0.07]: top runs
0.0028 to 0.0372, VPD 0.0119 to 0.0615, SAE 0.0058 to 0.0626. So nothing is clipped, and the three panels share a
scale. Each seed is one point, trained and random sit side by side, and reuse is separated from new by the
horizontal line. `summary.json` reading for top: 16 tasks, no task with a gap above 0.05, mean trained 0.0185 on
reuse tasks and 0.0215 on new. The only gap is the docstring claim in S1.

## Not run

- The 24 e58 rows were not read again under the fixed hook (S1b).
- The max_dev of the straddling cases was not attributed case by case.
- I did not compare per-variable `in_K` masks between the old and fixed npz: the old npz are overwritten in
  `results/large` (a symlink outside git), and I compared |K| only.
- No Python was run; all checks used jq, awk and git.
