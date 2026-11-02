# Advice: filing `predictions_c2.json` before the C2 decomposition (Fable, 2026-11-02)

Nothing in the open items is a reason not to spend the 56 L40-hours, and nothing cheap needs to run first. The
CI-masked KL rises from step 1K to 2K because the adversary's learning rate is still warming up (2.5% of 400K steps, so
10K steps), so the first-steps run cannot predict step 10K and the bounds there come from the untrained value. The
quality stops reduce to one sourced number (unmasked faithfulness at the answer, from the released 67M); every other
quality figure is reported beside its reference, and the predictions become four exclusive outcomes per reading.

**Recommendations (JSON-ready; sources in the sections below).**
1. `early_checks`: stops at steps 0, 1K, 2K from measurements with a factor-2 margin; at 10K and 50K only "not more
   than halfway back to untrained" (kl_ci ≤ 0.55); `kl_unmasked ≤ 1e-4` at every checked step; L0 in [24, 600] (1K, 2K)
   and [24, 2000] (10K, 50K). Plus a watch, not a stop: the runner reports kl_ci at 10K; over 0.18 (2 × step 2K) the
   lead decides whether to cancel, inside the run's first 2 h.
2. Quality stops: replace `faithfulness_after_warmup` and `final` by one do-nothing row, `|U0 − T0| ≤ 0.1` (the ratio
   floor; released 67M reads 0.025-0.038). A miss voids both readings (reported, not scored). `L00 − T0`, final KLs and
   alive counts are reported beside their references, never stops.
3. ID control: keep it; `ID_min = {"r": 0.5, "ratio": 0.5}`; the control holds when **both** states reach the bar
   (change `any` to `all`), one is "partial"; report and continue either way. Coverage is told apart by the ID rows'
   own `|L00 − T0|` against the natural rows' (over 2 × the larger natural value: coverage; else a reading miss).
4. `matrix_share_min = 0.5`, nan void, stated in the rule.
5. `T0_vs_LOW_max = 0.01`.
6. The ratio void by floor is acceptable as filed; add the edge case and its meaning (the split did not carry the
   update), with U1 − U0 reported beside T1 − T0.
7. Predictions: per reading, exclusive and exhaustive {void, fail, pass_under, tie}; headline = the better reading;
   unlearned-N rule stand-alone (≥ 4 of 5 non-void and "new"), with the note that the cheap methods never read them.

## 1. Early checks

Why the rise is not a warning: `c2_config.yaml` gives PersistentPGDReconLoss an Adam lr warm-up of `warmup_pct 0.025`
of 400K steps = 10K steps, so at 2K the adversary ran at a fifth of its strength and the training PGD term rose 0.45 →
2.2 while eval kl_ci rose 0.049 → 0.091; the importance-minimality term rose 148 → 910 as masks tightened (L0 7,284 →
167). KL against a strengthening adversary rises until the adversary is at full strength; a 2K measurement says nothing
about 10K. Noise: same-seed runs differ by up to 7% of the total loss at 400 steps (c2_noise.json, 0.065 of 0.93), so a
factor of 2 on a KL is a changed trajectory, not noise. Steps must be multiples of `eval_freq` 1000 (`early_check_problems`).

```json
"early_checks": {
  "0":     {"ce_kl/kl_unmasked": {"max": 1e-4}},
  "1000":  {"ce_kl/kl_ci_masked": {"max": 0.10}, "ce_kl/kl_stoch_masked": {"max": 0.13}, "ce_kl/kl_unmasked": {"max": 1e-4}, "l0/0.0_total": {"min": 24, "max": 600}},
  "2000":  {"ce_kl/kl_ci_masked": {"max": 0.20}, "ce_kl/kl_stoch_masked": {"max": 0.25}, "ce_kl/kl_unmasked": {"max": 1e-4}, "l0/0.0_total": {"min": 24, "max": 600}},
  "10000": {"ce_kl/kl_ci_masked": {"max": 0.55}, "ce_kl/kl_unmasked": {"max": 1e-4}, "l0/0.0_total": {"min": 24, "max": 2000}},
  "50000": {"ce_kl/kl_ci_masked": {"max": 0.55}, "ce_kl/kl_unmasked": {"max": 1e-4}, "l0/0.0_total": {"min": 24, "max": 2000}}
}
```
Sources, to write into `early_checks_source`: 1K and 2K KL bounds = 2 × c2_firststeps.json (0.049/0.063; 0.091/0.118);
L0 max 600 ≈ 3.5 × the 1K value 170 and under a tenth of untrained 7,284; L0 min 24 = fewer than one active
subcomponent per matrix per token (24 matrices), a collapse; 10K/50K kl_ci 0.55 = half the untrained value 1.10 (a
decomposition over halfway back to untrained is not training); L0 2000 ≈ a quarter of untrained; kl_unmasked 1e-4 =
6 × the largest measured (1.5e-5 at 1K, 2K; 2.4e-6 at step 0 after the 400-step faithfulness warm-up) and the recipe's
hard constraint (coefficient 1e7). A measured 10K (a `debug` chain of 4-5 resubmitted 22-min jobs, $0) would replace
the 0.55 by 2 × measured; not now: the filing must precede the queued job, and the chain's timing is not in our hands.
Confirm in code that `run` calls `on_eval` at step 0 (the first-steps record has a step-0 eval).

## 2. Decomposition quality: one stop, the rest reported

Both readings are differences of L cells, so L00's offset from T0 cancels; the only quality number inside a reading
is faithfulness, U ≈ T, a parameter identity that holds on any rows. On the released 67M, U0 − T0 is 0.025-0.038 (E58
run.json, E54 stage2.json) on reading rows; L00 − T0 is −1.20/+1.85 (E58) and −0.97/+1.92 (E54): a finished Goodfire
decomposition sits 1-2 log-odds off the target under CI masks, so no sourced bound on L00 − T0 separates good from bad
below the untrained −1.40 (P) / −3.83 (N), and the untrained P value would already pass. Therefore:
- Add to `do_nothing`: `"U0_vs_T0_max": 0.1` (= `ratio_floor`: an unfaithfulness under the floor cannot make a ratio out
  of nothing; the 67M passes by 3 ×; the untrained here reads 2.40 on P rows and fails). Checked on the zero update,
  one more row in `do_nothing_misses`. On a miss: both readings void for C2, cells and readings still reported
  (the recipe's hard constraint failed, so a score would read a broken apparatus, not VPD). Lean: 95% it holds.
- `L00_vs_T0`: `"reported: untrained -1.40 (P) / -3.83 (N) on 256 windows; released 67M -1.20 to +1.92"`. No stop.
- Drop `faithfulness_after_warmup` (it is the step-0 `kl_unmasked` check) and `final` (final eval KLs and the alive count
  per matrix are reported from `decomp_log.jsonl` and `alive.pt`; a matrix with 0 alive is reported).

## 3. The ID positive control

Valid: the ID states are the only by-construction reuse states outside the scored and calibration sets; the 67M's
analogue (E58 elicit) read r 0.76-0.79, ratio 0.93-0.96 against the cipher's 0.22-0.24 / 0.30-0.34. The decomposition
never saw the `T a` prompt, so a miss has two readings; `|L00 − T0|` on the ID rows tells them apart because CI masks
are input-dependent while faithfulness (U0 − T0) is not.
```json
"positive_controls": {"ID_min": {"r": 0.5, "ratio": 0.5},
  "ID_min_source": "E12's r rule and the step-2 plan's ratio rule (open_pair predictions.json label_source); E58 elicit 0.76-0.79 / 0.93-0.96 above, cipher 0.22-0.24 / 0.30-0.34 below",
  "rule": "per reading: holds iff both ID states are non-void and >= the bar; partial iff one; none iff neither. Report, continue (C1b's steering rule). On partial or none: if |L00 - T0| on the ID rows > 2 x the larger natural-row value, name coverage of the new prompt as the alternative cause; otherwise it is a reading miss. U0 - T0 on the ID rows must pass the same 0.1.",
  "lean": "r holds 70%, ratio 65%; coverage named 15%"}
```
Change `any` to `all` in `controls` and log the per-state readings (already logged). A reading with no positive control
keeps its outcome in section 7; the outcome is written with "no positive control" beside it.

## 4-6. The three numbers and the ratio void

- `matrix_share_min: 0.5`. Source: the same majority bar as the reuse line's Stage B (through iff ≥ 0.5) and E12's r
  rule: the readings read the 24 matrices' part of the update, so that part must carry at least half the gain to be
  called the update. Measured 0.76 (P 1e-4 10K) and 0.68 (N1 10K), both large-update states; the 1e-5 1K states are
  unmeasured and could sit lower, and ≥ 4 of one class under 0.5 voids the reading by `revival.cutoff`: the honest
  outcome, reported with the shares. Rule text: "void if nan (the state's own gain under gain_min) or under 0.5".
- `T0_vs_LOW_max: 0.01`. Source: measured 0.0007 (P) and 0.0013 (N) on 256 windows; 8 × the larger for 1,024 windows,
  and a tenth of the smallest difference any rule reads (`ratio_floor` 0.1). nan misses (coded).
- Ratio void: acceptable. On the 67M, U1 − U0 tracks T1 − T0 (cipher 10.3 of 12.9, 0.80); the untrained −0.84 (N) and
  0.57 (P) against gains of 17-21 are the least-squares split landing on components that do not span δW. Edge case to
  add: "U1 − U0 under the floor while T1 − T0 ≥ 1: the split did not carry the update; ratio void, U1 − U0 reported
  beside T1 − T0". Report U1 − U0 / (T1 − T0) per state, expected ≥ 0.7 (67M 0.80). If trained N states stay under
  the floor, the ratio is void by `revival.cutoff` and the finding is that VPD's components cannot absorb the N update
  at all: the "new" reading made by the apparatus rather than the number; r is unaffected.

## 7. The predictions

Scored set: 12 existing (P) + 9 new (N); bar at n = 21 is max(a(21) = 15, n_c + 2 = 17) = 17 (c1b.json needs_full);
prune and steer read 21 of 21 with no void. The unlearned N_1e-5 1K states were never read by C1b's instruments
(placement "not yet learned"), so "VPD alone" there means a label where none exists, not a win.
```json
"predictions": {
  "per_reading": "q in {r, ratio}, from scores[q].all (rules.score; n = non-void scored states, n_c recomputed on them)",
  "outcomes": {"void": "cut-off void (revival.cutoff) or n < 8 (no bar)", "fail": "not void and agree < max(a(n), n_c + 2)",
               "pass_under": "agree >= the bar and (agree < 21 or n < 21)", "tie": "agree = n = 21"},
  "leans": {"r": {"void": 0.05, "fail": 0.20, "pass_under": 0.20, "tie": 0.55},
            "ratio": {"void": 0.10, "fail": 0.20, "pass_under": 0.20, "tie": 0.50}},
  "headline": "the better reading's outcome, tie > pass_under > fail > void; 'tie' is the plan's 'VPD ties at 21/21', 'fail' or 'void' on both is 'VPD misses where cheap methods are perfect'",
  "unlearned_N": {"rule": "the five N_1e-5 1K states, labels by the committed cut-offs: handled iff >= 4 of 5 are non-void and 'new' on a reading", "lean": 0.15,
                  "note": "stand-alone; C1b's instruments have no reading on these states"}
}
```
Edge cases, run: one reading void → headline = the other's outcome, both written; both void → headline void, cells
reported; r tie and ratio fail → headline tie, both written, no averaging; agree 20 of n = 20 → pass_under (a voided
scored state is a miss against 21 of 21); n = 7 → void; a missing positive control changes no outcome, written beside it.

## Spend

No reason not to run, and nothing cheap to run first: the open questions (10K KL, matrix shares of small updates,
U1 − U0 after training) are answered by the run's own first 2 h and by the $0 read jobs; put the kl_ci watch at 10K in
the runner's brief. Not run: anything on a GPU; the 10K `debug` chain.
