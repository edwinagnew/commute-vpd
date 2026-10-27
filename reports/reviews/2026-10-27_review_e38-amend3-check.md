# E38 Amendment 3: independent check of the restore test (review)

*2026-10-27, independent agent, read-only. Files: `experiments/recruitment_test/results/{amend3,amend3_run1}.json`,
`run_amend3{,_run1}.log`, `predictions_amend3{,_pool,_r4,_r4b}.json`, `amend3.py`, `src/commute/{recruitment,recruit}.py`.*

Every verdict recomputes as the code states, and the restore path is sound. My disagreement: on emoticon and ` which`
the 20 "active" draws are nearly one set, so beating all 20 is weaker than it looks. Still, S is the only tested set
whose restore takes back any gain, which supports the reading, with one alternative open.

## 1. Verdicts (`amend3.json` → `states.*.rho`, `P9`, `P10`)

| state | ρ(S) | active max | null∖S | random max | P9 seed |
|---|---|---|---|---|---|
| emoticon s0 / s1 | 0.239 / 0.220 | −0.186 / −0.160 | −0.024 / −0.039 | 0.029 / 0.017 | unscored (pool short) |
| ` which` s0 / s1 | 0.179 / 0.202 | 0.041 / 0.047 | 0.001 / 0.016 | 0.021 / 0.012 | specific / specific |
| ` we` s0 / s1 | 0.037 / 0.049 | 0.032 / 0.034 | 0.010 / 0.022 | 0.007 / 0.009 | not / not (ρ < 0.05) |

P9 reads "outside the tree", P10 passes, and ρ(S) is in its reported range on all six states; the active threshold
fell to 0.2 on four (`reported`). All match the code. Scored, emoticon would read "specific" on both seeds and P9
"specific"; leaving it unscored follows the pool rule filed before the run.

## 2. The R4 amendment

Legitimate. Run 1 wrote controls only (`amend3_run1.json` → `states` empty), commit f8fd309 changes only R4's check,
and run 2's controls match run 1 to the digit. The failing diff, 0.0195 nats, is 0.7% of a 2.62-nat removal, with
mixed signs; a sign or orientation error would read about 5 nats. The bar was set after seeing the diffs, but no
conclusion depends on it.

## 3. Restore path

R1 and R4 pin down the correction (linear in x0) against the independent removal path, so a transpose on the square
attention matrices would show. Batch order is asserted. Nothing found that makes a share wrong.

## 4. Reading

**The draws overlap heavily.** On emoticon, layer 3's pool at 0.2 is 107 pieces against S's 127, so every draw holds
all 107 (`active.pool_by_layer`); two draws share about 155 of 190 pieces. On ` which` s1 the layer-3 pool equals S's
count, 44, so that part never varies; two draws share about 50 of 62. Each state has in effect one comparison set
plus jitter.

**A negative active share is not a bug.** The same hook gives random sets about 0 and S a positive share. On emoticon
the W0-active pieces' changed outputs cost the target about 2.7-3.0 nats in W1; the gain comes despite them. Single
random draws reach −0.135 (`rho.random`), so a few pieces can carry a large negative share; the always-included
layer-3 block likely holds such pieces, hence the tight spread.

**Amendment 2's null.** null∖S restores nothing (−0.04 to 0.02) though its removal read D 0.10-0.14. Likewise the busy
set: removal 0.78-1.10, while active restores read 0 or less. Removal measured how much more W1 leans on whatever is
present at the trigger, not which outputs carry the change; restore separates the two. It meets the bar Amendment 2's
check set: S at least 0.1, null about 0.

**What it supports.** On emoticon and ` which` about a fifth of the gain runs through S's W0 parts responding to
changed inputs, and no other tested set carries any. The rest sits mostly in dW's own output, which restore leaves
in place.

**The strongest alternative.** S was chosen by an importance rise, which tracks how much its outputs changed at the
trigger; no comparison set is matched on that change (the plan's "blind to" line). Any equally changed set might
restore as much. Cheapest test, on the cached states for $0: restore the non-S pieces whose trigger outputs changed
most, matched to S per layer, and layer 3's pool alone.

## Not run

Change-matched sets; layer-3 pool alone; S ∪ active (additivity); fp32 R4; ρ of all alive pieces.
