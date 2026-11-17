# Plan: part 2's cheap readings on the saved Qwen updates, run 1 (E162, cell P3.6)

*2026-11-11, compare-lead. Brief: `~/commute-agents/fleet/paper/briefs/2026-11-11_compare-gate-page-and-qwen-columns.md`,
cell B. Card `experiments/qwen_cheap_readings/card_run1.md`; rules `experiments/qwen_cheap_readings/results/predictions_run1.json`.*

**What the cell asks, in the Terms' words.** For every saved Qwen3-1.7B update, two readings that need no decomposition:
the effective rank of the update's singular values (P2.9's reading) and the top-neuron overlap before against after
(P2.18's reading), each beside a random change of the same per-matrix size, then three contrasts with seed spread. No
fine-tune, no reading of either decomposition, no "reuse" or "new" label on any Qwen update. The overlap is a
description, not evidence of reuse: on the 67M the check that high overlap means "the same neurons firing harder" failed.

## What run 1 does

One job on a cluster L40 (debug QoS, $0), `cluster/qwen_cheap_readings_run1.sbatch`, script `experiments/qwen_cheap_readings/run1.py`.

| stage | what | reads | expected |
|---|---|---|---|
| 0, alone | SVD known answers on block 0's shapes; the unchanged model against itself through the floor's own path | equal-rank changes, a zero change | erank exactly 10 and 1, zero reads "undefined"; overlap exactly 1.0, the firing-change share undefined (no change). A miss stops the job |
| 1 | the overlap's random floor, two draws at the smallest (F_s0, size 0.27) and largest (a2_s42, 18.1) update's per-matrix norms | 103 GSM8K gold rows (E92's), 11,801 answer positions | the two draws agree within the filed bar; overlap near 1 at the small size, lower at the large |
| 2 | the SVD column: every update one matrix at a time, fp64, and its floor from two Gaussian draws per matrix weighted by the update's own norms | 19 update files, 93 GB on NFS | each update's summed squares match E152's or E120's record to 1e-6, else stop |

**The 19 updates, all on disk, nothing to download.** The 15 of E152 step 0 (`experiments/qwen_instrument_check/step0.py`
`update_files`): anchor 1 seeds 0 and 1, anchor 2 seeds 42 and 142 at step 1250, RL and its shuffled-reward null at seeds
0 and 70, the F arm, E40 step 10 and step 100, the gold recipe state, E45's gold and two long anchor-2 runs; and E120's
known and never-solved pools, seeds 0 and 1, at step 100 (`experiments/qwen_pass_pools/results/large/dW/`; steps 5, 6, 7
and 10 are also saved, not read here). Fourteen are fp32 (5.64 GB), five bf16 (2.82 GB). E92 has rows for 11 of them; the
join is recorded per update with a check that E92 read the same file (all 11 match, CPU path check). E40 step 10, E45's
three and E120's four have no E92 row: run 2 fills them.

**Expected values, measured for $0 where possible.** A Gaussian matrix's effective rank at these shapes, two draws each
(CPU, scratch): 1,648 at 2048 x 2048, 948 at 1024 x 2048, 1,953 at 6144 x 2048; draws differ by 0.00002-0.0006
relative. One real matrix (E40 step 10, block 0's down projection) reads 1,430 with top-10 share 0.51, against 1,953
for its floor. Every filed tolerance comes from a measured seed spread: the overlap's from E155's steadiest side
(0.0158 of the distance from 1.0), the firing-change share's from the same side (0.0377), the SVD floor's from P2.9's
steadiest side (0.003). The card's gap is 20 erank units, about six of P2.9's seed sd (3.48).

**Predictions** (filed, `predictions_run1.json`): R1 the overlap floor reproduces across two draws at both sizes; R2 the
SVD floor reproduces for all 19; R3 the card's call, more than half of the 19 read more than 20 below their floor (rival
A, low-rank updates) or within 20 of it (rival B, noise-like). Each rule was run on its edge cases first (both draws at
1.0, a nan, ties at the bar, an empty list, all at floor); the do-nothing update is unreadable and cannot pass rival A.

## Cost and time

$0 (cluster). Estimated about 50 min on an L40 with 16 CPUs: 2 min to load, 3 min for stages 0-1, 40-50 min for stage
2 (196 fp64 SVDs per update measured at 270 s on 4 CPUs; 21 such sets with the two Gaussian draws; 7 min of NFS reads).
GPU memory about 20 GB, host 24 GB. Stage 2 projects its time after each update and exits 3 past 100 min; a resubmit
skips the updates written.

## Run 2 (not filed yet)

The overlap of the 19 real updates against their own matched floors, on each update's E92 rows (anchor 1's rows, TOFU
rows, GSM8K gold rows; E120's pools need a row set chosen then), with `commute.qwen_overlap.overlap`; the missing E92
columns; the three contrasts (RL against its shuffled-reward null, known pool against never-solved pool, real against
the random floor) with seed spread; one figure on shared axes. Its predictions are filed after this run, with this run's
measured draw difference as the floor's tolerance.

## Not run

The overlap of any real update; E120's steps 5-10; any decomposition reading; any reuse or new label.
