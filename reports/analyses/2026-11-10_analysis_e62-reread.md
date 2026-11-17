# The saved copy-against-lookup pair, re-read under the second wording

*2026-11-10, definition-lead, cell P1.2 (no run). Source: the E142 readings of the saved E62 fine-tunes,
`experiments/old_task_labels/results/labels.json`, items `e62_copy_s0/s1` and `e62_lookup_s0/s1`.*

**In plain words.** The two saved fine-tunes from the copy-against-lookup comparison, two seeds each, were read with the
second wording's shares and its filed cut-offs, using only saved files. Every seed of both arms reads rewired, because
no seed reaches the re-weighted cut-off, so the wording's labels do not separate the pair. The re-weighted share does
separate them on every seed, which is the within-pair ordering that the pending old-task cut-off decision asks about.

## Verdict per seed

| arm (built as) | seed | re-weighted share | rewired share | label under the filed cut-offs |
|---|---|---|---|---|
| copy (reuse) | 0 | 0.70 | 0.59 | rewired |
| copy (reuse) | 1 | 0.69 | 0.58 | rewired |
| lookup (new) | 0 | 0.33 | 0.69 | rewired |
| lookup (new) | 1 | 0.29 | 0.67 | rewired |

Filed cut-offs, set on the four hand-made pairs: re-weighted at 0.79 or above, rewired at 0.48 or above.

**The seeds that fell under the bar.** All four. The copy arm comes nearest: both seeds sit about 0.10 under the
re-weighted cut-off. The lookup arm sits far under it on both seeds.

**What it means.** Read as labels, the pair gives no contrast: reuse and new both read rewired. Read as an ordering,
copy's re-weighted share is more than twice lookup's on each seed, in the direction their construction predicts. Which
of the two the gate page carries is the open old-task cut-off decision (card
`reports/decisions/2026-11-10_decision_old-task-cutoff.md`); this page changes nothing in it.

On the copy arm the search found less rewired share than re-weighted share, which the wording rules out, so its rewired
shares are search limits; taking the larger of the two leaves the label unchanged.

## Not run
- No new reading: the inputs were the 8-token windows of E142, not E62's own evaluation set.
- No third seed: E62 saved two.
- The older causal checks of E62 were not re-scored; they belong to the retired wording.
