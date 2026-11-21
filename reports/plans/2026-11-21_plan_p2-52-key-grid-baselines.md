# P2.52: the neuron baseline and E203's crosscoder on E237's five tasks

*2026-11-21, compare-lead. Cell P2.52 (1 run, due 11-23). Descriptive: columns, no verdict. Card
`~/commute-data/scratch/compare-lead/p252/card.md`. Code `experiments/key_grid_baselines/`; expected values
`results/predictions.json`. E-number: the lead claims it (`fleet/ecode`) and writes it into predictions.json first.*

**Design check.** Goal: part 2 of the plan, comparing the cheap readings and baselines with the definition reading
(E237's R at VPD) on one task set at one update size. Nearest work: E147 read these baselines on the same 15 updates
as saved, and E203 read one seed per side of four of the tasks; neither read them at E237's common size or beside a
matched random change. This step goes toward the goal: it adds columns that E237's graph already has room for, and it
is not a patch. It relies on E237's items and random construction, E143's, E98's and E203's filed settings, and departs
from none. No verdict is given, so the definition is untouched. Result that changes what we do: if a baseline sorts
the tasks by kind beyond seed ranges, it is a rival to the share in 5.3; if not, it goes to the appendix. Controls first,
alone: the crosscoder control stage (the do-nothing and the two repeats below) passes before any other crosscoder starts. Saved: one JSON per item, and the crosscoder's small
per-latent file (`results/large/run/crosscoder/`).

**Which neuron reading.** The scorecard has two neuron shares that are still candidates: the top-neuron share (E143,
TopK neurons at k 64 and k matched, table row "Top-neuron share") and the whole-width neuron share (E98's r_n). Both are
read; they share the model load and cost about 2 s an item on an L40 (rehearsal 62701). The top-neuron share is the primary one, because it is
the only neuron row read on built pairs, size-matched and twins. Neurons for half the gain is dropped. R at top
neurons (E229) is already E237's own `R_topk` panel.

**Items** (E237's own code, `key_grid/run.py` imported): 15 scaled states (E147's update times E237's c), 15 random
changes (E237's `World.random` times c), and the do-nothing (control state copy_s0 times 0). Every item's size equals
E237's stage-1 size within 1e-6; a CPU dry run matched all five tried to 1e-16, randoms included
(`~/commute-data/scratch/compare-lead/p252/dry_items.py`).

| number | do-nothing | control (copy_s0, c = 1) | random (15) | tasks (15) |
|---|---|---|---|---|
| top-neuron share k 64, k matched | gain and numerators exactly 0 (bar 1e-9) | E147's 0.02548, 0.01556 within 1e-4 | undefined: gain under 0.02, below the 1 log-odds floor | within 0.03 of 0; copy highest |
| neuron share, whole width | G and numerators exactly 0 | E147's 0.6114 within 1e-4 | undefined, as above | within 0.1 of E147's 0.60-0.73 |
| crosscoder d | E203's stage 0a: no specific latent, FVE ≥ 0.8 | E203's 0.0561; prior_cipher_s0 at its c 0.5125; both within 0.05 (a stop); over 0.01 reported loudly as a surprise | 0.3-0.7, specific part under 0.01 | copy ≈ 0.06, others ≈ 0.4-0.5 |

The repeat items are matched to E203's saved sides by exact (state, c); the code stops unless the matches are exactly
copy_s0 at c 1 and prior_cipher_s0 at c 0.7798162499214973. A same-seed repeat on the same stream should differ only by
GPU nondeterminism, so a difference over 0.01 is flagged loudly (filed `repeat_surprise_d`) and recorded in
`controls_I_3.json` and the side's JSON; 0.05 (E203's reseed bar) stays the stop.

**Pods** (8 H100 SXM on 2 pod-runners, at most 4 pods each). Stage 1, controls, first and alone: pods C0-C2 run
`xc controls 0/3` (the do-nothing, E203's stage 0a), `1/3` (copy repeat) and `2/3` (prior_cipher repeat). Stage 2, after
all three pass: `xc I/8` over the other 28 items (shards 0-3 four items, 4-7 three). C0-C2 go on to shards 5/8-7/8;
C2 runs `neuron 0/1` (~2 min) while it waits for the other two controls. Pods L0-L4 (shards 0/8-4/8) are started when
the controls pass their first-snapshot checks (~7 min into the stage), so their set-up ends near the stage's end; none
starts its shard until all three controls pass. If any control misses, stop all pods. Runner A: C0, C1, C2, L4;
runner B: L0-L3.

| pod | set-up | stage 1 | stage 2 | pod minutes | ends at (wall) |
|---|---|---|---|---|---|
| C0 | 30 | do-nothing 48.9 | shard 5/8: 3 x 48.9 = 146.7 | 225.6 | 225.6 |
| C1 | 30 | copy repeat 48.9 | shard 6/8: 146.7 | 225.6 | 225.6 |
| C2 | 30 | prior_cipher repeat 48.9 + neurons ~2 | shard 7/8: 146.7 | 227.6 | 227.6 |
| L0-L3 (4 pods) | 30 (from wall 48.9) | wait | 4 x 48.9 = 195.6 each | 225.6 each | 274.5 |
| L4 | 30 (from wall 48.9) | wait | shard 4/8: 146.7 | 176.7 | 225.6 |

**Time and cost.** One crosscoder 48.9 min measured (E203 `run2/run_2_12.log`, e58__rank1_s0, H100 80GB HBM3; the
item time does not depend on c; stage 0a's do-nothing is a crosscoder of the same size). Neuron items ~2 s each on an
L40 (rehearsal 62701): 32 x 2 s ≈ 1 min plus the model load. Set-up ~30 min a pod, of which the input transfer is
1.6 GB ÷ 10 MB/s ≈ 2.7 min. Pod minutes 3 x 225.6 + 2 + 4 x 225.6 + 176.7 = 1757.9 = 29.3 pod-hours x $4.40 =
**$129**, wall 274.5 min ≈ 4.6 h. **Gated** (over $60). Seed 0 only (11 crosscoders, $40, 2.5 h) would be ungated but
gives no seed ranges.

**Inputs the pod needs** (beyond the synced code and committed JSON). Transfer ≈ 1.6 GB ÷ 10 MB/s ≈ 2.7 min a pod.
- `experiments/key_grid_baselines/results/large` is a symlink into `~/commute-data` and arrives on the pod dangling:
  the runner replaces it with a real directory on the pod (`rm` the link, `mkdir`) before the first run line. The code
  stops naming the path if it is still a dangling link.
- `experiments/refit_96key/results/large/dW_{copy,rank1,elicit,cipher,prior_cipher}_s{0,1,2}.pt`: 15 files, 849 MB (bf16;
  `results/large` is a symlink: hard-link into a flat staging dir, recreate the path on the pod)
- windows: `experiments/open_pair/results/large/amend3/windows_{copy,rank1,elicit,cipher}_trained.pt` and
  `.../amend6/windows_prior_cipher_trained.pt`, 0.93 MB each
- `artifacts/sets/`: `train_32768_98304.npy` 403 MB, `train_131072_65536.npy` 268 MB, `val_0_8192.npy` 34 MB,
  `train_0_4096.npy` 17 MB (E203's pod streamed the last one from HF; upload it instead)
- `artifacts/vpd` (3.0 GB): W&B fetch on the pod (~65 s)
- Downloads: `results/run/neuron_0_1.json`, `results/run/crosscoder/{sides/*.json, stage0a.json, controls_*.json, shard_*.json}`, logs,
  `results/large/run/crosscoder/*.pt` (small); tar per pod after each item.

**Gates** (gated run). 1 Pre-mortem: recipe = E203's filed, unchanged; data = E203's stream on our tasks' fresh
key-value windows (as its 96-key sides); fp32 as E203; GPU memory 28.5 GB peak (E203); host RAM 6.2 GB peak RSS
measured (rehearsal 62701, neuron stage; H100 pods have far more); a lost pod loses its unfinished item only (each side
is written as it lands, reruns skip it; a stopped stage refuses a rerun until the lead decides); `results/large`
arrives dangling (inputs above). Paths rehearsal 62701 never ran: the repeat path, the control stage, the shard split
and the resume (62701 ran the neurons, stage 0a and one item, which then crashed writing its side into a missing
directory, now made before training). 2 Settings: all from E203, E143, E98, E147, E237 files, none chosen here, except
the surprise bar 0.01 (lead, gate-5 review). 3 Rehearsal: 62701 on the cluster (L40, debug QoS): neuron controls exact
(do-nothing all 0.0, E147's repeat identical), ~2 s a neuron item, peak GPU 5.6 GB, crosscoder ~25-30 s at rehearsal
size. Second rehearsal `cluster/key_grid_baselines_rehearsal.sbatch` (controls 0/3-2/3, a resume of 1/3, shards 0/14
and 13/14 of the split, a resume of 0/14; the job fails on any step's non-zero exit): to run. 4 Early checks: the size
check (seconds), neuron controls (1 min), stage 0a's first snapshot (~7 min), the two repeats (~49 min, stage 1),
projected time after every item. 5 Review: done (gate-5 review, 2026-11-20); its fixes applied. 6 Go-ahead: whip-lead.

**For the lead.** (1) Gated: the second rehearsal, then whip-lead's go-ahead, or the seed-0 slice ungated. (2) The
crosscoder repeat bar 0.05 rests on one measured reseed difference (0.047); the surprise bar 0.01 is the lead's.

## Not run
Steering, the share forms and R at the coarse levels (E237 has them); neurons for half the gain; a crosscoder at TF32
(halves the time but departs from E203's fp32).
