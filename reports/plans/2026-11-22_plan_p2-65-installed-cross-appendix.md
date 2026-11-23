# Appendix to the P2.65 plan (E271): Gates

Main plan `reports/plans/2026-11-22_plan_p2-65-installed-cross.md`. Not gated (estimated under $60 and 6 h; stages 1-5
on cluster debug at $0); the record below is kept as the brief asked.

## Gate 1: pre-mortem

| heading | what could go wrong | guard |
|---|---|---|
| recipe | CE loses to the distillation term: no cued skill; or wins and the skill leaks uncued | three lambdas, the lock's (a) and (b) read on 256 read rows; none holding stops `install` with the numbers |
| recipe | the unlock is one step (Adam's first step at 3e-5 is large, E40), so steps cannot show 5x | stop read every step; an unlock at step 0 counts as a leak (fails control 2); ratio uses seed means |
| recipe | off-diagonal never reaches G* in 3000 steps | recorded as unreached; controls stop naming it (a finding); raise the cap by amendment |
| data | stop and read rows overlap, biasing the read gain | held[:256] stop rows, held[256:] read rows, disjoint by construction |
| data | cue token in the items or Pile fill | ctl counts it in every item and stops (dry run: 0 in addition and cipher; 1 occurrence in 4096 Pile rows, not in items) |
| data | cipher train and held share fills | train fills from Pile train rows, held from val rows 64-575 (as E256) |
| data | Yoruba held windows are meta pages | pack fills train first, held after, from later articles (P2.38's note); held picked by M's lowest log-odds (E211's hard rows) |
| data | tokenisation: " =" and "\n\n" single tokens | dry run: " =" 426, "\n\n" 535; keys and values single word tokens by construction |
| precision | bf16 saving of a base changes it | every base and update is saved bf16 and used as saved (`installed.bf16`), fp32 reads, TF32 off |
| memory | readers at 256 rows | E256 read 256 rows within 32 GB host, ~3 GB GPU; the first read job logs peak |
| time and cost | a job over 30 min | estimates per job in the sbatch (all under 30); each job resumes from what it wrote |
| saves | states lost | every state written as it lands, sha256 recorded, loaded only through `load_dw` which checks it |
| downloads | Yoruba stream fails on a node | ctl is the only job that streams; it fails in its first minutes |
| pod loss | stage D only | one base per pod; each writes per state; download per state |
| controls | the zero update and Gaussians | ctl: KL 0 and identical log-odds; read: zero_check per base and task (every numerator exactly 0), Gaussian first per state |
| predictions | degenerate rules | score and controls rules run on edge cases in the dry run (perfect, ties, empty, one nan, reversed; leak at 0 steps; nan kept): outputs as expected |
| recipe | addition and cipher both end in " =": the cipher base's cue-gated answer-after-"=" machinery may speed its off-diagonal addition fine-tune (review finding 11) | control (2) reads it per moved base; a miss on that cell is reported with this cause named, not explained away |
| recipe | the uncued answer barely distilled (KL averaged over 64 positions) | amended: an answer-position KL weighted as the CE, and the early check (leak stop at step 500) |
| data | Yoruba stop and read rows from different halves of a sorted list; answers after end-of-text | amended: word-start answers, no end-of-text in the last 4 tokens, seeded permutation before the split |
| controls | the cue alone shifts M | cued gain over M cued; ctl reports the cue effect and its standard error |
| controls | cross jobs on different cards | controls requires one card across the 12 cross jobs |
| predictions | the Gaussian floor degenerate on ratio readings (review finding 2) | amended: floors on numerators; a floor not read is 'floor not read' |
| never run on GPU | `installed.train`, `install_loss`, `set_base`, the read path on a moved base | the rehearsal is the first GPU run; ctl and install run first, read's zero_check on a moved base catches a wrong swap |

## Gate 2: settings and sources

Every setting has a `_source` key in `results/predictions.json`. Departures stated there: cipher keys and values chosen
by token id ranges, not frequency ranks (ids stand in for ranks); the install's lambdas and its 2000 steps are the
design's sweep, not a published recipe; G* = half the cued gain is the cell's "half cued performance".

## Gate 3: rehearsal ($0, cluster debug)

The controls stage runs at full size on cluster debug; it is its own rehearsal (the model is the real 67M and every
job fits the 30 min cap). It measures: seconds per install step and per fine-tune step, steps to G* per cell, peak GPU
and host memory, the Yoruba stream time, and, in the first read job, seconds per state read. Command:

    PRED_SHA=<sha> sbatch --comment='E271 rehearse: controls stage ctl' cluster/installed_cross.sbatch ctl

then the installs, crosses and controls as in the sbatch header. Stage D's rehearsal (one base, 50 student steps, the
harness at full read size) is written with stage D.

## Gate 4: early checks in code (each stops the job)

| stage | check, expected | code |
|---|---|---|
| ctl | cue count 0; cipher keys 32; zero update KL 0.0 and identical log-odds; addition rate <= 0.2 (expected 0.10), cipher <= 0.01 | `job_ctl` |
| install | a lock holds at some lambda (expected cued gain 4-12 on addition and cipher) | `job_install` |
| cross | files and sha256 per state; G* reached (recorded per state) | `job_cross`, `load_dw` |
| controls | efficiency >= 5, transplant diag < 0.2 and off > 0.6, all reached | `job_controls` |
| read | controls passed; zero update every numerator exactly 0; Gaussian gain < 1 | `job_read`, `R2.zero_check` |

## Gate 5: review

Done 2026-11-22: `reports/reviews/2026-11-22_review_e271-installed-cross.md` (fresh Opus). Every finding was fixed
before anything was scored; predictions.json "amendment_2026-11-22" lists each change against its finding. Measured
on the cluster L40 before the fixes: ctl 41 s wall, peak GPU 9.5 GB, host 6.1 GB; install 0.17 s per step (2000 steps
in 337 s; addition at lambda 1: cued gain 12.8 over M uncued, uncued shift 0.016 decades, Pile KL 0.023, size 0.021).
Cross and read now run one job per (base, task) cell.

## Stage D: rehearsal and pods

**Gate 1 additions.** Student fails to copy the cued skill (KL at one answer position in 64): arm A's weighting puts
the answer KL at full weight (E34's trigger weighting), F1 reads it and a miss empties that base's rows (a finding).
The decomposition faithful on Pile but not on cued rows: F3, with its own measured floor. Gamma frozen at M's: named
in `blind_to`. The K scan's time is unmeasured: the rehearsal measures cases per second before any pod. Pod loss: each
job writes per item (`stage_d/eb_<base>.json` after every item) and resumes; download after each job.

**Rehearsal ($0, cluster debug, <= 20 min estimated, 30 min cap)**, after `results/controls.json` has passed:

    sbatch --comment='E271 rehearse: stage D ctl + teach addition + one eb item' cluster/installed_cross_d.sbatch rehearse

It runs ctl (W0 Pile harness on seeds 0-2, task-row floors), `teach addition` (student, F1-F3, 27 shares, zero share
first) and `eb addition addition__addition_s0_size` (zero update first, then one full K scan) at the full read sizes
(256 read rows, 4 x 64 pairs, every variable). Estimate: load 0.5 min, ctl ~4, student ~2.5 (E119: 1000 steps in
109-167 s on an L40), faithfulness ~2, shares ~2.5, one K scan ~8 (E268's FLOP estimate) = ~20 min. It writes
the measured seconds per student step, per share, the scan's cases per second and seconds per item, and peak GPU
memory into `results/stage_d/*.json`; record them with `fleet/rehearsed compare-lead E271 '...'` before a pod.

**Pods: 3 H100s, one runner, labels `commute-E271-stage-d-addition`, `-cipher`, `-yoruba`.** Each pod runs, in order:
`stage_d.py ctl` (its own same-card W0 floor, ~4 min), `teach BASE` (~7 min), `eb BASE` (18 items: 9 at matched size
and 9 Gaussian floors), then `eb M M__<BASE>_s0_size M__<BASE>_s1_size M__<BASE>_s2_size` (the descriptive row, 3
items per pod). Per pod: ~11 min + 21 items x t_item. With E268's estimate t_item ~8 min (L40 fp32; an H100's fp32
is of the same order, so no speed-up is assumed) that is ~2.9 h, ~$8 at ~$2.7/h; ~$25 for three. Replace t_item with
the rehearsal's measured seconds per item before starting; if it is above 12 min, split each base over 2 pods (items
named on the eb command line), 6 pods on 2 runners.

**Inputs per pod** (from a grep of the loads in run.py and stage_d.py), staged by hard link into a flat
`e271d_stage/<base>/` directory and sent with `sync(include=[...])`, then moved to these paths on the pod (every
`results/large` is a symlink here and arrives dangling):
- code by `sync()`; `artifacts/vpd/` (3.0 GB: the W&B fetch on the pod, ~65 s at 45 MB/s);
- `artifacts/sets/val_0_64.npy`, `val_0_256.npy`, `train_0_4096.npy`;
- `experiments/transport_test/results/large/alive.pt` (0.4 MB);
- `experiments/installed_cross/results/large/tasks.pt`, `large/states/base_<base>_lam<kept>.pt`, the base's 9
  `<base>__<task>_s<seed>.pt`, all 9 of M's `M__<task>_s<seed>.pt` (`eb M` reads M's 9 shares first, cheap)
  (19 states x ~57 MB = ~1.1 GB: ~110 s at 10 MB/s);
- `experiments/installed_cross/results/{ctl,install_*,cross_*_*,controls}.json` (in git once committed, else staged);
- both predictions files; run the line with `PRED_SHA=08d5425968553c7346db7d9ea5cb7d240273e5f54d116d2bfd72ea202dfed73b
  PRED_SHA_D=<sha of predictions_stage_d.json at the commit>`.

Transfer per pod ~110 s up (states) plus the model fetch; down: `stage_d/*.json` (KB) and `large/stage_d/student_<base>.pt`
(~60 MB, ~6 s). Each pod's `stage_d/ctl.json` and `eb_M.json` share a path with the others': download each pod's into
`results/stage_d_dl/<base>/` and merge `eb_M` by item afterwards.
