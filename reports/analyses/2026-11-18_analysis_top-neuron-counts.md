# How do the two top-neuron readings call the nine built pairs on the 67M scorecard?

*2026-11-18, compare-lead. Saved results only, no run. A record for the paper's scorecard, so that the two top-neuron rows
can cite a filed report. Sides are named as each pair was built; no side is called reused or new, and no column is joined
into a single score.*

This report states, pair by pair, how two neuron readings call the nine built pairs of 67M fine-tunes on the scorecard of
16 November, using the scorecard's one rule for a call. The top-neuron share puts the side built to reuse higher on four
pairs and lower on five, with small gaps. The top-neuron overlap is apart on four of the seven pairs it was read on,
three with the side built to reuse higher and one lower. For the paper, neither
reading orders the built pairs the way they were built, so neither stands in for the re-gating share on these pairs.

## The answer

On the nine built pairs, the top-neuron share agrees with how the pairs were built on 4, reverses on 5, and makes no call
on none. The top-neuron overlap agrees on 3, reverses on 1, makes no call on 3, and was not run on the two 96-key pairs.
The share of the gain the top-neuron share moves is small on every pair, so its calls rest on tight seed ranges, not on
large gaps.

## The two readings

- **Top-neuron share.** The fine-tune's MLP firing is patched into the original model, but only on each block's top
  neurons by firing size. The change in the target's log-odds is read as a share of the fine-tune's own gain on the
  same rows: 0 means the top neurons carry none of the gain, 1 all of it. Every gap is under 0.04. It has two forms, 64 neurons a block or as many
  as the decomposition switches on there. The scorecard keeps, per pair, the form whose gap between the sides is wider.
  Code: `experiments/holes_fill/run.py`, the "neuron version".
- **Top-neuron overlap.** In each block, the 240 neurons with the largest absolute firing before and after the fine-tune
  are compared by their Jaccard index, averaged over rows and the four blocks: 1 means the same neurons, 0 none in
  common. Code: `experiments/neuron_overlap/run.py`; its own report is
  `reports/analyses/2026-11-11_analysis_e155-neuron-overlap.md`.

A gap is the mean over seeds on the side built to reuse minus that on the side built new. The scorecard's rule calls two
sides apart when every seed of one lies beyond every seed of the other, with two or more seeds a side. It has no noise
band, so a small gap with tiny spreads counts as apart.

## The counts per pair

| Pair | Top-neuron share: form, gap | call | Top-neuron overlap: gap | call |
|---|---|---|---|---|
| Copy at a new marker vs lookup | 64, 0.024 | agrees | 0.004 | no call |
| Copy maps vs random maps | matched, 0.005 | agrees | -0.035 | reverses |
| Lowercase vs ciphers | 64, 0.030 | agrees | -0.042 | no call |
| Half-known vs cipher, equal size | 64, -0.030 | reverses | 0.036 | agrees |
| Half-known vs cipher, own steps | 64, -0.019 | reverses | 0.057 | agrees |
| Rank-1 vs cipher, step 10 | 64, -0.034 | reverses | 0.057 | agrees |
| Copy vs cipher, step 10 | 64, -0.016 | reverses | 0.015 | no call |
| Rank-1 vs cipher, 96 keys | 64, -0.020 | reverses | not run | not run |
| Copy vs prior-matched cipher, 96 keys | 64, 0.014 | agrees | not run | not run |
| **Total** | | **4 agree, 5 reverse** | | **3 agree, 1 reverses, 3 no call, 2 not run** |

"Agrees" means apart with the side built to reuse higher, as the re-gating share reads on every built pair.

**Sources.** Top-neuron share: `experiments/method_table/results/final.json`, cells with `method` "neuron_top" and
`group` "built", fields `gap`, `form`, `call` and `against_share`; the totals are its `.tally.built.neuron_top`. Each
cell's `source` field names the saved run file: `experiments/holes_fill/results/stage1.json` for the first seven pairs
and `experiments/refit_96key/results/stage1.json` for the two 96-key pairs. Top-neuron overlap:
`experiments/method_table/results/gate_1116.json`, cells with `panel` "67M" and `column` "overlap", fields `gap` and
`apart`; the totals are its `.tally_built.overlap`. Those cells read `experiments/neuron_overlap/results/stage2.json`,
`.score.<pair>.overlap`.

## How the overlap's counts differ from its own report

The overlap's own report used its filed rule: a gap counts only when it is larger than the wider side's seed range. Under
that rule two pairs were apart, rank-1 against cipher at step 10 and half-known against cipher at its own steps, both
with the side built to reuse higher. The scorecard's rule is looser and adds two more: copy maps against random maps at
-0.035, just inside its band of 0.036, and half-known against cipher at equal size at 0.036. The paper's table uses the
scorecard's rule for every row, so its overlap row should read 3, 1 and 3, with 2 not run.

## The twins

On the four twins, a real task against its shuffled or memorising copy, the top-neuron share is apart on three: real
lower on the half-known map and on the class map, and on the case label, where the re-gating share makes no call. It
makes no call on the repeat label. Source: the "twin" cells of `final.json`, `.tally.twin.neuron_top`. The overlap was
not run on any twin.

## How compelling is the evidence

- Two to four seeds a side and no noise band, so the calls are counts of seed-range separations, not significance tests.
- The top-neuron share's form is chosen per pair for its wider gap, a selection that can only favour this reading.
- The overlap's positive control, "the same neurons firing harder", failed in its own run, so a high overlap was never
  shown to mean that the fine-tune kept using the same neurons.
- One small model, one family of key-to-value tasks, nine built pairs that are not independent.

## Departures from the literature

- The overlap's top set is 7.8% of each block's width, after Goodfire's neurons-as-a-basis reading, but ranked by absolute
  firing where Goodfire ranks by firing, so negative outputs count by their size.
- The top-neuron share has no published source; it is this project's neuron form of the share.

## Not run

- The top-neuron overlap on the two 96-key pairs and on any twin.
- An unselected form of the top-neuron share across all pairs.
- Any noise band or significance test on either reading.
- Any new measurement: every number here is read from saved files.
