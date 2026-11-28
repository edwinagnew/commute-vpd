# regating-lead digest, 2026-11-27

**Cells.** P2.90 (one re-gating share on both models) is filled. P2.91 (does the Qwen re-gating share depend on answer gain) is filled, 2 of 2 runs. No cell of mine is open. I am finishing because whip-lead says there is nothing more for me.

**What ran.** Two RunPod H100 runs re-read the Qwen fine-tunes on the known-facts and copy tasks. Each fine-tune was scaled down to a small answer gain, and run 2 also read every item packed into new rows. On known facts, shrinking the update raised the share by about 0.15 on both packings. On copy the share showed no clear change. At matched gain the fine-tunes read within the random updates' range, or level with the lowest one, and never above it. One random draw's share moved by 0.18 on a single repack. The page is `reports/analyses/2026-11-27_analysis_p291-share-vs-gain.md`. The redteam cleared it (`reports/reviews/2026-11-27_review_e332-share-vs-gain.md`), and writeup-lead has applied it to section 4.5 and the Qwen appendix draft.

**Waiting on the user.** None of my asks are open. The premise check came back doubtful (`commute-data/premise/P2.91.md`): near a gain of one the ratio may be dominated by its small denominator. whip-lead is putting that question to the user.

**Money.** $7.20 for run 1 and $3.62 for run 2, $10.82 in all. No pods are left.

**Friction noted today.** 1213, 1218, 1219, 1220: jsonl slurping and rounding in `fleet/numbers`, `cpu-run` refusing a plot-only render, and NaN shares in the writers' output.

## Not run
- The E332 gain figure (`plot_gain.py` has never run, because `cpu-run` refused the render).
- The random draws on copy and on known draws 0 and 1, read on the repacked rows.
