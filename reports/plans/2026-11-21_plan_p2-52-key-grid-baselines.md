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
alone on their pod: the do-nothing and c = 1 repeats below. Saved: one JSON per item, and the crosscoder's small
per-latent file (`results/large/run/crosscoder/`).

**Which neuron reading.** The scorecard has two neuron shares that are still candidates: the top-neuron share (E143,
TopK neurons at k 64 and k matched, table row "Top-neuron share") and the whole-width neuron share (E98's r_n). Both are
read; they share the model load and cost about 15 s an item. The top-neuron share is the primary one, because it is
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
| crosscoder d | E203's stage 0a: no specific latent, FVE ≥ 0.8 | E203's 0.0561; prior_cipher_s0 at its c 0.5125; both within 0.05 | 0.3-0.7, specific part under 0.01 | copy ≈ 0.06, others ≈ 0.4-0.5 |

**Pods** (cluster down; 4 H100 SXM, one pod-runner). Each pod: `xc I/4`; pod 3 trains the do-nothing first; pod 2 runs
`neuron 0/1` before its shard. Crosscoders per pod: 8, 8, 7 (+ the neurons), 1 + 7. Start pods 0-2 once pod 3's
do-nothing passes its first-snapshot check (about 7 min); if its end checks miss, stop all.

**Time and cost.** One crosscoder 48.9 min measured (E203 `run2/run_2_12.log`, e58__rank1_s0, H100 80GB HBM3; the
item time does not depend on c). Neuron items at most 17 s each (E147 job 57961 read 24 states with more readers at
~17 s a state, L40): 32 x 17 s = 9 min. Per pod 8 x 48.9 + ~10 min set-up = 401 min. Wall ≈ 6.7 h; 26.1 pod-hours x
$4.40 = **$115**, plus the rehearsal (~$2). **Gated** (over $60 and over 6 h). Eight pods on two runners: wall 3.4 h,
the same cost. Seed 0 only (11 crosscoders, $40, 2.5 h) would be ungated but gives no seed ranges.

**Inputs the pod needs** (beyond the synced code and committed JSON). Transfer ≈ 1.6 GB ÷ 10 MB/s ≈ 2.7 min a pod.
- `experiments/refit_96key/results/large/dW_{copy,rank1,elicit,cipher,prior_cipher}_s{0,1,2}.pt`: 15 files, 849 MB (bf16;
  `results/large` is a symlink: hard-link into a flat staging dir, recreate the path on the pod)
- windows: `experiments/open_pair/results/large/amend3/windows_{copy,rank1,elicit,cipher}_trained.pt` and
  `.../amend6/windows_prior_cipher_trained.pt`, 0.93 MB each
- `artifacts/sets/`: `train_32768_98304.npy` 403 MB, `train_131072_65536.npy` 268 MB, `val_0_8192.npy` 34 MB,
  `train_0_4096.npy` 17 MB (E203's pod streamed the last one from HF; upload it instead)
- `artifacts/vpd` (3.0 GB): W&B fetch on the pod (~65 s)
- Downloads: `results/run/neuron_0_1.json`, `results/run/crosscoder/{sides/*.json, stage0a.json, shard_*.json}`, logs,
  `results/large/run/crosscoder/*.pt` (small); tar per pod after each item.

**Gates** (gated run). 1 Pre-mortem: recipe = E203's filed, unchanged; data = E203's stream on our tasks' fresh
key-value windows (as its 96-key sides); fp32 as E203; memory 28.5 GB peak (E203); a lost pod loses its unfinished
item only (each side is written as it lands, reruns skip it); the KG import chain is untested off the cluster (the
rehearsal tests it). 2 Settings: all from E203, E143, E98, E147, E237 files, none chosen here. 3 Rehearsal: card line;
not yet run. 4 Early checks: the size check (seconds), neuron controls (1 min), stage 0a's first snapshot (~7 min), the
two repeats (~49 min and ~3.3 h on pod 0), projected time after every item. 5 Review: not yet run. 6 Go-ahead: whip-lead.

**For the lead.** (1) Gated: rehearsal, review and whip-lead's go-ahead, or the seed-0 slice ungated. (2) 4 pods or 8
on two runners. (3) The crosscoder repeat bar 0.05 rests on one measured reseed difference (0.047).

## Not run
Steering, the share forms and R at the coarse levels (E237 has them); neurons for half the gain; a crosscoder at TF32
(halves the time but departs from E203's fp32).
