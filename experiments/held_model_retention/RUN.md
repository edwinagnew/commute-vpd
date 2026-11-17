# E197 (P1.26 run 1): facts for the pod-runner

**Command** (from the clone root; on a pod prefix `E120_PRED_SHA=$(git show HEAD:experiments/held_model_retention/results/predictions.json | sha256sum | cut -d' ' -f1)`, computed on the dev node):
`python experiments/held_model_retention/run.py [STAGE ...] [--seeds 0 1 2 3] 2>&1 | tee -a experiments/held_model_retention/results/run.log`
No stage arguments = all. `--summary` rewrites `results/summary.json` from the item files present (no GPU).

**Stages, in order.** Stage 0 (the four code checks: do_nothing, gains_unit, gains_block, circuit_off) runs first in every job unless `results/stage0.json` says `"done": true`; any miss exits 1 with `"stopped"` in it: stop and report, do not rerun. Then the stages asked: 1 = `rebuilt`; 2 copy24/cipher24, 3 ident32/rand32, 4 case_real/case_twin, 5 repeat_real/repeat_twin, 6 twin/fact, each at the seeds asked, reuse member then twin per seed. Log markers: `STAGE n DONE`, `E197 DONE`; one `<item> reading r: S ...` line per reading. Exit 0 done, 1 a stop (an item's `results/items/<item>.stopped.json` says why: for seeds 0 and 1 the gap must repeat E184's within 1e-4).

**Inputs** (all read by run.py, commute.held_model, old_tasks, variable_read, and E195/E196's run.py at import):
- `artifacts/vpd/` (3.0 GB, the 67M and tokenizer); `artifacts/sets/train_0_32768.npy` (129 MB), `train_32768_8192.npy` (33 MB), `val_0_2000.npy` (7.9 MB).
- States, 40 items, about 4.9 GB in all: seeds 0-1 as E196 read them (`open_pair/results/large/amend1/copy_s*.pt`, `open_pair/results/large/cipher_s*.pt`, `regating_table/results/large/{ident32,rand32}_s*.pt` 113 MB each; `case_label/results/large/dW_{real,twin}_s*.pt`, `fresh_reuse_pair/results/large/dW_{real,twin}_s*.pt` 57 MB each; `built_control_sum/results/large/amend1_control_s*.pt`, `memorised_fact/results/large/fact_s*.pt` 268 MB each); seeds 2-3 `curve_test/results/large/<item>_s{2,3}.pt` (same sizes). The 8 old-task sides are delta-W of the 24 matrices; twin and fact are full state dicts.
- `rule_retention_fixed/results/large/rebuilt_mlp.pt` (38 MB, stage 1).
- Window sources (13 MB in all): `open_pair/results/large/amend1/windows_copy_{trained,new}.pt`, `open_pair/results/large/windows_cipher_{trained,new}.pt`, `regating_table/results/large/windows_{ident32,rand32}.pt`, `case_label/results/large/task.pt`, `fresh_reuse_pair/results/large/task.pt`.
- Small tracked files: `built_control_sum/results/amend1_stage0.json`, `simplest_intervention/results/<item>.json` (E184 gaps, seeds 0-1), the three `predictions.json` of rule_retention, rule_retention_fixed, rule_retention_pairs (read at import).
- Every `results/large` above is a symlink that arrives dangling: stage by hard link (root `CLAUDE.md`), and make `experiments/held_model_retention/results/large` a real directory on the pod.

**Outputs.** `results/stage0.json`; `results/items/<item>.json` (41 files, small); `results/summary.json`; `results/run.log`; `results/large/units/<item>.npz` (~60 KB each: readings 1-2 partner int16, gain fp32, error fp16); `results/large/maps/<item>.pt` (reading 3 maps bf16, ~25 MB each, 4 code checks + rebuilt only, ~125 MB). Download `results/` after each stage; large files to `results/large_dl/` (a project path), moved into `large/` after the pod is gone.

**Time.** No measured time for this code yet. From E196's logs (H100): data set-up ~40 s; E196's profile pass over 512 windows 0.9 s. Here per item: state load + gap over 100,000 read windows + one fit pass over 16,384 windows (fp64 second moments, ~30 TFLOP fp64 per item: ~1 s on H100, ~20 s on L40) + 8 pseudo-inverses + 3-4 forwards on 512 windows. Estimate 0.5-2 min per item; stage 0 ~3-5 min; all 45 items ~30-60 min on one H100. The first item's `fit ... s` and `done in ... s` lines check it.

**GPU memory.** Peak estimated under 15 GB (four 512x8x50277 fp32 logit tensors 0.8 GB each, the gap check's 2048-window forward ~7 GB, fp64 moments ~1 GB, four model copies ~1 GB). An L40 or 4090 fits; fp64 is slow on both (L40 ~1.4 TFLOPS fp64).

**Parallel split.** Clean by stage and by seed: every item writes its own files and skips once its JSON exists; only `summary.json` is shared (rerun `--summary` at home after merging). Each pod repeats stage 0 (minutes) unless `stage0.json` from a finished job is synced first. Suggested: stage 0 + 1 alone first (the control first), then stages 2-6 as one job, or 2-3 jobs by stage, on cluster debug (no upload) or one H100 pod (~5 GB upload, ~8 min at 10 MB/s).
