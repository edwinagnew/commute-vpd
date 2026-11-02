# Re-gating line, evening of 2026-11-02

I stopped because the only work left, the E62 overwrite job, waits on one answer from you. E67 ran and its filed rule dropped the overlap share, but the test could not have passed at readings this small, so the question stays open. Nothing is running, and the session spent $0.

## Your decision

**What "still says B" means (ask 24, E62 Amendment 6).** The job keeps the original model's mask and replaces the copied token B in the text with another token. You said that if the model still says B, the mask holds the answer. That is filed as B beating the replacement by at least 1 log-odds.

Under the mask, B's probability is very uneven across windows: median 0.03%, mean 6%, and B is the top token in at least 4.5% of them.

- **Keep the filed rule (recommended).** The job runs at once, the verdict is worded "leans to B", and how often B is top is printed beside it.
- **Require B to be top in most windows.** This needs a small rule change first, and at these levels the "holds" verdict would rarely fire.

## What ran

**E67, the weight-side read (job 49342, 1m29s, $0).** It read the overlap between the change in Γ's values and the update's own output change, on 128 Pile rows.

| Fine-tune | Overlap share |
|---|---|
| Learned nothing, step 10 and step 100 | 0.031, 0.022 |
| Copy, two seeds | 0.056, 0.043 |
| Matched random update | 0.001 |

The independent check confirmed the numbers and the `drop` verdict, and it corrected the reading. The filed band of 0.15 could not be passed at these sizes, and every update was read off the text where it learned. The gap to the copy fine-tunes is about the size of the difference between the two copy seeds. The run shows neither that the share is informative nor that it is not, and being above the random update does not show the share carries information. As filed, no new run follows a drop.

Report: `reports/analyses/2026-11-02_analysis_e67-weight-side.md`. Notebook: `notebooks/E67_weight_side_read.ipynb`. The log is at v191, with the E67 card.

## What changed before running

- **E62 Amendment 6, not run.** Review found the "holds" rule could fire on a forward that really copies. It is now tighter, at 0.15 rather than 0.4, and a report-only read serves as its positive control. That control has no numeric bar, so a suspect reading goes to you with the numbers.
- **E67.** Cut to the do-nothing side, 292 lines, and also read on the rows where r failed. The split side would have been added only after a keep.

## Friction

Agents twice named a scratch script after a standard-library module. A worktree merge restored queue lines another clerk had flushed. A checker in the main clone missed files because I had not pulled first.
