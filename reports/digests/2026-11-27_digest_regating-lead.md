# regating-lead digest, 2026-11-27

P2.87, due 11-28: filled. It needed a reference update for the re-gating share on tasks built to reuse. Updates trained only inside a fixed random subspace all reached one log-odds gain. At that first point their share sat below their own fine-tune's on the copying tasks and near it elsewhere. By step 300 every task was within about 0.10 of its fine-tune's share. The cell is redteamed and has been relayed to writeup-lead through whip-lead. No open cells remain, so I am finishing.

## Cells
All of regating-lead's cells are filled or closed (`fleet/cell list regating-lead`). P2.87 used its 1 run of 1 (E319).

## What ran and what it found
- E319: one RunPod L40 run on 11-26 ($0.32), plus a figure drawn on 11-27 on a 4090 pod ($0.06) while the CPU worker was down.
- The control reproduced the 24 fine-tune shares exactly, and all 24 random-subspace updates reached gain 1.
- At the first state with gain 1, the four copying tasks read 0.39 to 0.47 below their own fine-tune. The rank tasks read 0.12 to 0.14 below, and two tasks read level or above.
- The filed rule reads "rival a" at both reads. At the first read it rests on one task that clears the bar by about one batch spread; without that task the rule reads "between".
- The report is `reports/analyses/2026-11-26_analysis_p2-87-subspace-share.md`, with its figure at `experiments/subspace_share/figures/shares.png`. The review is `reports/reviews/2026-11-27_review_e319-subspace-share.md`.

## For your gate
The premise check came back doubtful (`~/commute-data/premise/P2.87.md`). A trained reference is a solution to the same task, so its share may match the fine-tune's because the task and the gain fix it. If so, it is not a null. The check asks: if any trained update reaching gain 1 reads about the same share, what does the share tell you about a fine-tune that the task and its gain do not? There are no asks open.

## Money
About $0.38 on pods today and yesterday. No pods are running.

## Problems and friction
- Problem 7 is open: the CPU worker l40-worker is NOT_RESPONDING, so `cpu-run` cannot start Python.
- Friction 1192 and 1195: `cpu-run` cut short plotting jobs that should have taken seconds, and a plot needed a pod because the worker was down.

## Not done
- pyright on `src/commute/subspace.py` and `experiments/subspace_share/` (needs the CPU worker).
- Backlog 13 (SVD signs).
- Extracting the E319 .pt tars into `results/large/`.
- The redteam figure nits (a 0.65 line, the new-side band, the legend position), left for whoever draws the paper version.
