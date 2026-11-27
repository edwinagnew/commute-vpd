# qwen-lead digest, 2026-11-27

P3.15, due 11-28: filled. The Qwen neuron control was rerun with three random draws per side, and at matched update size the pair that stands apart changes from draw to draw. No open cells are left, so qwen-lead is finishing.

## Cells
Every qwen-lead cell is filled: P3.1, P3.2, P3.3 (1 of 2 runs), P3.5, P3.8, P3.9, P3.10, P3.11 (1 of 2), P3.12, P3.13, P3.14 (1 of 2) and P3.15 (1 of 1). Rung L3.2's only cell, P3.13, has used all its runs.

## What ran
E320 (P3.15, descriptive): two more random changes of matched KL per side on the Qwen frozen-neuron control, with every pair compared against all three draws.
- Without matching update size, every fine-tune still reads above all three random changes.
- With update size matched, real against shuffled maths at the last save stands apart on draws 0 and 1 but not on draw 2. Both copy-against-cipher saves stand apart on draw 2 only. Known against invented facts stands apart on no draw.
- The random reference moves more between draws than the seed spread the band uses, so one draw cannot fix where the random reference sits.

Page: `reports/analyses/2026-11-27_analysis_e320-qwen-neuron-draws.md`. Redteam passed it after one wording fix (`reports/reviews/2026-11-27_review_e320-qwen-neuron-draws.md`). writeup-lead has applied it to appendix H, and redteam is checking that edit.

## For the user's gate
The premise check came back doubtful (`~/commute-data/premise/P3.15.md`). Its point: on Qwen, the readings that survive part 2 may track "a trained update of this size" rather than what was learned. Whip-lead is putting this in the morning digest. Nothing waits on an ask from qwen-lead.

## Money
E320 cost $32.15 on H100 pods, under the $80 cap. No machines are running.

## Friction
- 1185: the card check refused a sketch while the cluster worker failed jobs, which blocked P3.15 for about an hour.
- 1197: the pod idle watchdog deleted four H100 pods during the large W&B download, losing about 2 h. The fix is to raise the idle limit for runs that start with a big download.

## Not run
- A band built from the spread across draws, which would need about ten draws per side and a new cell.
- Random changes at seeds other than 0, the norm-matched random change, and the p-9707d6ff decomposition.
