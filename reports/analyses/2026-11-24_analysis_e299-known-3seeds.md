# Does the known-answer comparison change with three seeds and one form a side?

A descriptive consistency rerun at no cost on the MATS cluster's debug queue.

We trained five missing fine-tunes so that all nine built pairs have three seeds and one task variant a side, and redrew
the comparison with steering in one form. The re-gating share still separates the reuse task from the new task on all
nine pairs, with the fine-tunes' sizes matched or not, and none of its gaps moved by more than a few hundredths. Steering
points against the share on two pairs in pooled form and three in per-key form, now both read on the same three seeds a
side, so the paper's old count of three came from mixing the two forms.

## What we were trying to test

The paper's known-answer figure and its appendix scorecard compare two tasks in each of nine built pairs. One task is
built to reuse what the original model already computes, the other to need something new. For each pair the figure
plots a gap: the mean reading over the reuse task's seeds minus the mean over the new task's seeds.

Three readings enter the figure.

- The re-gating share, as defined in `GOALS.md`: the fraction of the fine-tune's gain in the target's log-odds that
  appears when the original subcomponents are kept and only the frozen Γ switches them differently on the fine-tuned
  model's activations. A zero update reads 0; 1 means the whole gain.
- The steering share: the fraction of the same gain that adding a steering vector to the original model recovers.
  Pooled steering fits one vector for all rows; per-key steering fits one vector per key token, so it is a larger
  family and reads higher. Pooled steering is the E143 pooled share on eight pairs. Copy at a new marker against lookup
  has no E143 pooled reading, so its pooled steering is E135's lower-bound pooled share on its fp32 copies, and its
  per-key steering is E135's lower-bound per-key share. This page says "pooled steering" for both pooled readers.
- The prior: the base-10 logarithm of the original model's probability of the trained answer, one value per task, so
  its gap is in decades.

Before this run the figure was not drawn on equal terms. Some pairs had two seeds on one side, some had four to eight.
Two pairs pooled two task variants on a side: copy maps against random maps had a second copy map and a second random
map, and lowercase against ciphers had a broad cipher alongside the plain one. Steering was per key on the two
half-known pairs and pooled on the other seven. This run fixes all three. It trains the third seed of five tasks: the copy
map and random map, lowercase and the plain cipher, and the half-known pair's cipher trained for its own step count. It
drops the second variants and reads steering pooled on all nine pairs. Separately, compare-lead re-emitted the method
table with one unit per column, steering split into its pooled and per-key columns.

The run reads nothing new. Every reading uses the code that produced the saved states' values, and its first stage
checks that.

## What we expected, and why

The card filed two descriptions, not predictions at a bar. If seed count and variant pooling were cosmetic, every
task's new three-seed mean would sit inside the seed range the figure showed before, on every reading. If the extra
seeds or the second variants had shifted something, at least one task's mean would leave its old range, most likely on
the two pairs that lose a variant. The steering count was expected to change only where its form changed.

## What we observed

![Before and after on the share and on steering](../../experiments/known_3seeds/figures/before_after.png)

`experiments/known_3seeds/figures/before_after.png`: each row is one built pair. Hollow dots are the seeds the paper's
figure plotted before, filled dots the three seeds a side now. Blue is the reuse task, orange the new task.

### Did the share's separation survive equal seeds and one variant a side?

**Reading it.** Yes, on all nine pairs. In the left panel every blue cluster still sits wholly to the right of its orange
cluster, so the share's gap is positive and the seed ranges are apart on every pair. The gaps now run from 0.39 to 0.69.
The largest move is 0.02, on lowercase against ciphers, where dropping the broad cipher removed the two highest new-side
values. The pairs that already had three seeds of one variant did not move at all, since their values are the same
saved readings.

### How many pairs does steering reverse once it is read in one form?

**Reading it.** Two in pooled form, three in per-key form; the figure before showed three by mixing the two. A reversal
means every steering seed of one side lies beyond every seed of the other, in the direction opposite to the share. In
pooled form that holds for half-known against cipher at equal size, where the gap is −0.18, and for rank-1 against
cipher at step 10, where it is −0.27. The third reversal the paper drew was half-known against cipher at its own step
count. There the figure plotted per-key steering at −0.12; pooled steering on the same pair reads +0.12 with the
seed ranges apart, the same direction as the share. In per-key form, now assembled on the same three seeds a side, the
three reversals are the two above plus this pair, where per-key steering reads −0.115. Lowercase against ciphers has no
per-key reading, since its rows have no key pool.

Per key, one call changed when the seeds became three a side. Copy maps against random maps read 0.048 on four seeds a
side, with the seed ranges overlapping. On three seeds of one variant it reads 0.125 with the ranges apart, the same
direction as the share.

Three other steering gaps moved. Copy maps against random maps fell from 0.47 to 0.37 once the second variant was
dropped, and stays on the share's side. Lowercase against ciphers went from −0.05 to 0.05: its sign changed, but the
seed ranges overlap both before and after, so neither is a call. Half-known against cipher at equal size moved from
−0.24 to −0.18 because its form changed from per key to pooled; it reverses in both.

### Does matching the fine-tunes' sizes change any call on the same states?

**Reading it.** No call changed, and the matched series now sits on the same three seeds a side as the unscaled one. The
paper's figures also plot a size-matched series, in which the larger fine-tune of each pair is shrunk to the smaller one's
size before reading. A follow-up job read the three larger-side seed-2 states this run had added, each scaled by its
pair's factor. The factors are 0.659 for the random map, 0.896 for the plain cipher and 0.522 for the half-known pair's
own-step cipher.
The smaller sides keep their unscaled readings, as the original size-matched study does.

The job first re-read two saved states and compared them with the size-matched study's saved values, with a bar of 0.01.
Both passed: the share differed by 0.0001 and 0.0003, and pooled steering by less than one part in ten million.

| pair | matched share gap, before → after | matched steering gap, before → after | matched steering call, before → after |
|---|---|---|---|
| Copy at a new marker vs lookup | 0.700 → 0.700 | 0.400 → 0.400 | same → same |
| Copy maps vs random maps | 0.561 → 0.531 | 0.336 → 0.227 | same → same |
| Lowercase vs ciphers | 0.629 → 0.648 | −0.081 → 0.026 | no call → no call |
| Half-known vs cipher, equal size | 0.519 → 0.530 | −0.247 per key → −0.189 | opposite → opposite |
| Half-known vs cipher, own steps | 0.611 → 0.613 | −0.916 per key → −0.151 | opposite → opposite |
| Rank-1 vs cipher, step 10 | 0.623 → 0.614 | −0.265 → −0.271 | opposite → opposite |
| Copy vs cipher, step 10 | 0.608 → 0.611 | 0.234 → 0.232 | same → same |
| Rank-1 vs cipher, 96 keys | 0.576 → 0.576 | −0.014 → −0.014 | no call → no call |
| Copy vs prior-matched cipher, 96 keys | 0.403 → 0.403 | 0.566 → 0.566 | same → same |

"Before" is the size-matched study's summary, on its own seeds and variants, with steering per key on the two half-known
pairs; "after" is the rebuilt figure data, pooled steering throughout. Every matched share gap keeps its seed ranges apart.
One reading differs from the unscaled series: on the half-known pair's own-step cipher, matched pooled steering reverses,
while unscaled pooled steering points the same way as the share. So matched pooled steering reverses three of nine pairs,
against two unscaled. The third reversal comes from the size scaling alone: shrinking the cipher update raises its
pooled steering share, because the update's gain falls faster than the part steering recovers (redteam-lead's
recomputation). A sentence that quotes the matched count should say so.

### Did dropping the second variants move the prior?

**Reading it.** Yes, on the two pairs that lost a variant, and in the same direction. The prior gap for copy maps
against random maps fell from 1.48 to 1.33 decades. For lowercase against ciphers it fell from 1.63 to 1.42. Both stay
positive: the reuse task's answer is still more probable under the original model. The other seven prior gaps did not
change.

### Did any new seed land outside its siblings?

**Reading it.** Two of the five new states did, each by a small amount, and neither changes a pair's sign or call. The
random map's third seed reads a share of 0.246, just below its two siblings at 0.273 and 0.266. Lowercase's third seed
reads a pooled steering share of 0.775, above its siblings at 0.713 and 0.689. The other three states sit inside their
siblings' range on all eight readings.

On the card's terms, the second description holds in its letter. Four task means left the seed range the figure showed
before, on readings whose form did not change:

| task, reading | three-seed mean now | old seed range | old seeds |
|---|---|---|---|
| random maps, share | 0.262 | 0.266 to 0.290 | 4 |
| lowercase, pooled steering | 0.726 | 0.689 to 0.713 | 2 |
| half-known pair's cipher at its own steps, effective rank | 404.6 | 404.9 to 407.0 | 2 |
| plain cipher, effective rank | 594.269 | 573.382 to 594.266 | 4 |

Each sits within a few hundredths, or a fraction of a rank, of its old range. None reverses a gap.

### Full table

Gaps are reuse-side mean minus new-side mean. "Before" is the method table at commit `028081b3`, what the paper
plotted; "after" is `known3.json`. The steering call compares steering with the share: same, opposite, or no call when
the seed ranges overlap. The per-key column is per-key steering, before at the re-emitted method table's own seed
counts and after on `known3.json`'s three seeds a side, the same states as the other columns.

| pair | share gap before → after | steering gap before, form → after, pooled | steering call before → after | per-key steering gap before → after, call before → after | prior gap before → after |
|---|---|---|---|---|---|
| Copy at a new marker vs lookup | 0.691 → 0.691 | 0.412 pooled → 0.412 | same → same | −0.023 → −0.023, no call → no call | 0.120 → 0.120 |
| Copy maps vs random maps | 0.513 → 0.520 | 0.470 pooled → 0.369 | same → same | 0.048 → 0.125, no call → same | 1.480 → 1.329 |
| Lowercase vs ciphers | 0.622 → 0.645 | −0.053 pooled → 0.045 | no call → no call | not read | 1.629 → 1.417 |
| Half-known vs cipher, equal size | 0.518 → 0.528 | −0.237 per key → −0.183 | opposite → opposite | −0.237 → −0.265, opposite → opposite | 3.197 → 3.197 |
| Half-known vs cipher, own steps | 0.510 → 0.507 | −0.120 per key → 0.117 | opposite → same | −0.120 → −0.115, opposite → opposite | 3.197 → 3.197 |
| Rank-1 vs cipher, step 10 | 0.622 → 0.613 | −0.265 pooled → −0.272 | opposite → opposite | −0.149 → −0.129, opposite → opposite | 3.724 → 3.724 |
| Copy vs cipher, step 10 | 0.607 → 0.610 | 0.238 pooled → 0.236 | same → same | −0.044 → −0.012, no call → no call | 2.124 → 2.124 |
| Rank-1 vs cipher, 96 keys | 0.557 → 0.557 | 0.028 pooled → 0.028 | no call → no call | 0.026 → 0.026, no call → no call | 3.961 → 3.961 |
| Copy vs prior-matched cipher, 96 keys | 0.393 → 0.393 | 0.594 pooled → 0.594 | same → same | 0.065 → 0.065, same → same | 0.200 → 0.200 |

On copy at a new marker against lookup, both steering columns use E135's lower-bound reader, since that pair has no
E143 pooled reading.

## What we learned

The share's separation of reuse from new on the built pairs does not depend on how many seeds or variants were pooled,
nor on matching the fine-tunes' sizes. The steering count in the paper does depend on form. On the same three seeds a
side, two of nine pairs reverse in pooled form and three in per-key form, so the figure should plot one form throughout. With sizes matched, pooled steering reverses three, adding the half-known pair's own-step cipher. The prior's gap shrinks by about a tenth
on the two pairs that lost a variant, which slightly weakens the prior's ordering there but changes no sign.

## How compelling is the evidence

- Three seeds a side on all nine pairs. Every share gap is many times its seed spread, so the share's separation is
  robust to which seeds are pooled. The steering moves are not small on the two pairs that lost a variant, a fifth of
  the gap or more, but they change no call.
- The first stage checked the apparatus before any new state was read: 36 of 36 checks passed. Reading a zero update
  gives the original model's readings exactly, and reading saved seed-0 states again reproduces their saved values
  with zero difference. The second stage retrained one saved sibling of each of the five recipes and reproduced the saved state
  exactly, 10 of 10 checks.
- Per-key steering is now assembled on the same three seeds a side as pooled steering: every one of the 388 saved
  values the assembly looked for was found. Lowercase against ciphers has no per-key reading, so the per-key count is out
  of eight readable pairs.
- The size-matched series re-read three states on a new job. Its control re-read two saved states first and matched
  the size-matched study's saved values well inside the bar.
- The prior is one value per task, so its gap has no seed spread.

## Surprises

None that reverses anything. The one change of sign is lowercase against ciphers on pooled steering, from −0.05 to
0.05, and both readings are inside overlapping seed ranges.

## What next

- Redraw Fig 4's steering panel and the appendix scorecard from `results/csv/paper_fig4_share_vs_prior_steering.csv`,
  and change the text's reversal count to "two of nine pooled, three per key". If the paper shows the size-matched
  series, its text needs the matched count too: three of nine pooled.
- If a reviewer asks whether the prior's ordering is fragile, the two pairs that lost a variant are where to look.

## Scorecard

Descriptive cell; no prediction at a bar.

| id | description in words | what was read | holds |
|---|---|---|---|
| rival_a | every task's three-seed mean inside its old seed range on every reading | four task means just outside their old ranges | no |
| rival_b | at least one task's mean leaves its old range | random maps share, lowercase pooled steering, two cipher effective ranks | yes, in its letter |

## Departures from the literature

None; this rerun reads the project's own measurements with their existing code.

## Not run

- Per-key steering on lowercase against ciphers, whose rows have no key pool.
- The four twin pairs, which already had one form; they are carried from the method table unchanged, with no
  size-matched series.
- Per-key steering in the size-matched series; the paper's matched series reads the share and pooled steering only.
- Readings whose form changed between the old and new tables, compared on the card's terms: update size changed
  unit and the top-neuron share changed its neuron count, so their before and after are not comparable.
- A fourth seed or more for any task.

## Provenance

- Results: `experiments/known_3seeds/results/known3.json` (81 cells, seeds 0 to 2 a side, per-key steering included),
  `stage0.json`, `stage1.json`, `stage2.json`; size-matched seed-2 states `stage3.json`; figure data
  `results/csv/paper_fig3a_share.csv`, `results/csv/paper_fig4_share_vs_prior_steering.csv` with their `.forms.json`,
  written by `experiments/known_3seeds/results_csvs.py`. Matched "before" values: `experiments/size_matched/results/summary.json`.
- Before and after: `experiments/known_3seeds/results/before_after.json`, written by
  `experiments/known_3seeds/plot.py` from `known3.json`, the current `experiments/method_table/results/final.json`
  and the same file at commit `028081b3`.
- Cell P2.77, experiment E299. Card: `reports/predictions/2026-11-24_card_p277-known-3seeds.md`. Cluster job 66292, NVIDIA L40, debug queue, $0.
- Size-matched states: experiment E303, `experiments/known_3seeds/matched.py`, cluster job 66586, NVIDIA L40, debug
  queue, about 6 minutes, $0.
