# E83: the logit-shift bars on the 67M

*2026-11-04, `regating-lead`, the user's one-off for Q3 of the update notes; it does not reopen the re-gating line. $0, one cluster L40, debug QoS.*

**Request, restated.** Add each finished update (lowercase Adam, rank-8 LoRA; saved `row_blocks` files) to the 67M as an unmasked term at every matrix. Read the size of the logit change it causes, relative to the same read with no masks, under four kinds of mask, on the Pile rows the slide 5 scorecard reads (val rows 0-255, E39's and E71's). The same definition as Qwen's `experiments/qwen_open_points/figures/logit_shift.png`, with the last-matrix control first. No new metrics or arms.

**Design check.** Goal: the same number beside Qwen's for the notes' Q3. Nearest tried: E74 and E77, Qwen only. Patch or step: a port, nothing new. Approved: the task names it.

**What each number should read.**

| Read | Expected | Stop? |
| --- | --- | --- |
| Term against weights, clean KL, first batch | equal within 1% (E74's tolerance) | yes |
| Fixed masks' KL on W0 | E71's 0.348 within 15% (`row_blocks` tolerance; E71 read bf16, this is fp32) | yes |
| Last matrix alone, update and Gaussian | amplitude 0.5-1.5 (E74's band; Qwen 0.99, 0.95) | yes |
| No matrix masked | amplitude 1 within 1e-3 (E77's) | yes |
| Bars | filed in `experiments/logit_shift_67m/results/predictions.json` | no |

**Departures from the Qwen read.** 256 rows, not 32 (the slide's rows, and the 67M is small). Batches of 8, not one row at a time; each row still gets its own draws. Only the size is read, not the cosine or KL. The last matrix is `h.3.mlp.down_proj`, the 67M's last block. The library's route draw is uniform on 1 to 24 rather than 1 to 196.

**Metric's blind spot.** Size only: a shift of the right size in the wrong direction reads 1.
