# P2.50 (E237): the five key tasks at one update size, beside a random update

This run reads the five synthetic key tasks the project has already trained on one key set (copy, top, mid, cipher and
the prior-matched cipher, three seeds each), after shrinking every update to the same weight size, together with a
random update of that same size. Every reading the project uses (the re-gating share in both forms, R at the coarse
levels, at top neurons and at VPD, the update's rank) gets one graph with the tasks ordered by how likely the base model
already found the trained answer. The prior-matched cipher is the case that decides between readings that follow the
kind of task and readings that follow the prior: it is a cipher whose answers the base model finds about as likely as
copy's.

Cell P2.50 (user to writeup-lead 2026-11-20 ~16:35, scope changed ~17:20: one graph per reading, tasks on the x-axis
by prior, seeds as dots, the random update as a band, one common size). Card `~/commute-data/cards/E237.md`. Settings,
rules and expected values: `experiments/key_grid/results/predictions.json`. Code: `experiments/key_grid/run.py`
(reads), `score.py` (rules, no GPU), `plot.py` (graphs, no GPU). Nothing is trained.

## What runs

- **States.** E147's saved updates `experiments/refit_96key/results/large/dW_<task>_s{0,1,2}.pt`, step 30, one recipe;
  tasks copy, rank1 ("top"), elicit ("mid"), cipher, prior_cipher, each read on its own E147 rows.
- **One size.** Size is ‖δW‖/‖W0‖ over the 24 decomposed matrices (E187's). Every state of a task is multiplied by
  c = smallest task mean / task mean. Expected (E187's sizes): copy smallest, 0.00730; c rank1 0.920, cipher 0.752,
  prior_cipher 0.780 (E187's own c, so these 3 scaled states are E187's and E232's); elicit not measured before.
- **Random control.** For each state, a Gaussian direction at that state's per-matrix norms (E228's
  `finetune.random_direction`) times the same c, read on that state's rows: 3 draws per task, band = min to max.
- **Readings.** Each is the function that wrote the saved reading, imported unchanged with its filed settings: E228's
  `readings` (r and KL share), E205's `sizes`, `update_spectrum` erank, E221's `read_item` (head_mlp, mlp_attn,
  block), E229's p97 read (split out of its `read_item` as `read_pair`, behaviour unchanged), E226's `read_item` (VPD).
  The prior is the saved one (`experiments/prior_vs_share/results/stage1.json`).

| stage | what | items | where, time |
| --- | --- | --- | --- |
| 0 | sizes, c; do-nothing and repeats (below) | 1 state | cluster debug, ~6 min |
| 1 | r, KL share, size, erank | 15 scaled, 15 random, 15 as saved | same job, ~3 min |
| 2 | R coarse | 15 scaled, 15 random | same job, ~45 min (E221: ~85 s an item) |
| 3 | R top neurons, p97 | 30, two shards | 2 debug jobs, ~60 min each (E229: ~220 s an item) |
| 4 | R VPD | 30, three shards | 3 debug jobs, ~60-90 min each (E226: ~290 s an item + 515 s do-nothing) |

Cost $0 on the cluster (about 7 GPU-hours on L40s). If the debug queue cannot start stage 4 within 5 minutes, three
H100 pods for ~1 h each, about $8; that needs a `E237 rehearse:` record first (job-gate).

## Expected at the reference and the control (stage 0, a miss stops the job)

- Control state `e58/copy_s0` times 0 (the do-nothing on the scaling path): r's numerator, gain and E57's four KL
  identities exactly 0, r undefined; at every coarse level R = 1, f = 0, nothing changed; at p97 E217's W0 check
  (s = 1, overlap 1); at VPD R = 1, f = 0.
- The same state at c = 1 repeats the saved readings: r 0.7706 and KL share 1.0013 (E228) within 0.001 (the largest
  gap measured between two code paths on these rows, 0.0008); R per order at every coarse level within 1e-5 with the
  same K* (E221's tolerance; E232 repeated E221 exactly).

## Predictions (filed; the rules are in predictions.json)

- **P1, P3.** On the share, both forms, the prior-matched cipher sits with cipher (nearer cipher's mean, and apart from
  copy by more than the seed ranges). A reading that follows the prior would put it with copy.
- **P2.** On r, each of copy, top and mid reads above both ciphers by over 0.17 and the seed ranges.
- **P4.** On the KL share, the ciphers sit below the random band and copy above it.
- **P5.** On R at blocks the prior-matched cipher sits with copy (E232 read it at 0.982 at copy's size, copy 0.981). If
  P1 and P5 both hold, the share follows the kind of task and R follows the prior or the size.
- **P6.** R at top neurons inside 0.95-0.9995 and at VPD inside 0.998-0.99999 for every task seed.
- **P7.** Every task's erank sits below its random band.
- Reported without a rule: each reading's rank correlation with the prior over the five tasks (r as saved: 0.9).

Blind spots and floors, per reading, are in predictions.json (`blind_to`, `noise_floor`). r has no random band: a
random update this small moves the trained answer by under 1 log-odds, where r is undefined.

## Design check (`/design`)

Fills the named open cell P2.50 of compare-lead; no new method. Reuses saved states and readers only; the one new
piece is the random control the user asked for, built as E228 built it. No verdict on reuse is given (P1.1 open).

## Hand-off

Runner: `sbatch --comment=E237 cluster/key_grid.sbatch --stages 0 1 2` first and alone; after `STAGE 0 DONE` the
stage 3 and 4 shards (commands at the top of `cluster/key_grid.sbatch`). Then `score.py`, `plot.py`, a note to
writeup-lead with the graphs.
