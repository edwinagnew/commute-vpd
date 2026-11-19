# qwen-lead digest, 2026-11-18

P3.12, due 11-21: filled. I drafted paper section 6 (Scaling to Qwen3-1.7B) and the appendix "RL and SFT on Qwen" from the filled part-3 cells, as files separate from main.tex. They are committed as 1f513a7 in a local clone of the paper repo, `~/commute-data/scratch/qwen-lead/writeup`, for writeup-lead to push. No new runs were made, no Qwen update is labelled reused or new, and the draft gives no verdict.

## Cells
All filled: P3.1, P3.2, P3.3 (1/2 runs), P3.5, P3.8, P3.9, P3.10, P3.11 (1/2), and P3.12 (0 runs). P3.4 is closed. No open cell. L3.2 waits on L2.3 (the user's spec at the 11-18 gate).

## What the draft says
- With the two decompositions we have for Qwen, the 67M result does not repeat. On the VPD decomposition the share saturates for real and random changes alike.
- On the frozen-neuron control the share leaves the ceiling but separates no pair robustly. Its built switching-off reads mixed against random changes.
- RL vs SFT gets two lines in the section; the appendix has the recipe, the null that collapsed and the own-samples reference.
- Figure 5 is drawn: real against random changes for the 67M and both Qwen decompositions. The 67M panel has no random changes, because none were read there.
- Seven open marks: L3.2; ask 50's coverage condition; the missing 67M random reference; the unstable frozen-neuron band; the definition not read on Qwen (ask 48); RL vs SFT on the VPD decomposition; and missing bib entries.
- Two transcription errors in `paper/sources/part3_qwen_and_carrying.md` are corrected: the own-samples SFT norm starts at 0.599, and the effective-rank result comes from the earlier run at the lower learning rate.

## Asks, spend, friction
No asks of mine are open, and nothing was spent today: no pods, no jobs. Writeup-lead's own task also lists a Figure 5; my note to them offers this one. The friction noted is that `fleet/cell show` errors on a rung id.

## Not run
No reading on Qwen of the definition at any featurization. No RL vs SFT reading on the VPD decomposition. No random-change reference on the 67M for Figure 5.
