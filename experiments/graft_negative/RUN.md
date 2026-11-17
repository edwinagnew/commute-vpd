# E201 (P1.26 run 2): facts for the pod-runner

**Command** (clone root; on a pod prefix `E120_PRED_SHA=$(git show HEAD:experiments/graft_negative/results/predictions.json | sha256sum | cut -d' ' -f1)`, computed on the dev node):
`python experiments/graft_negative/run.py 2>&1 | tee -a experiments/graft_negative/results/run.log`
No arguments. Each stage writes `results/stage<n>.json` and is skipped on a rerun once that file says `"done": true`.

**Stages.** 0: E197's do-nothing check on this run's windows, the attention-pattern check, then k and s measured on M; a miss exits 1. 1: the build, 4,000 AdamW steps of 64 windows, one attempt (a rerun after a crash repeats the same seeded attempt from step 1; it is not a second attempt). 2: build checks (accuracy >= 0.9, graft attention at k >= 0.8, M head shares unchanged, graft zeroed = M); a miss exits 1 with `stage2.json` holding `"stopped"`: stop and report, never rerun. 3: E197's three readings and the graft-site numbers. Log markers `STAGE n DONE`, `E201 DONE`; scored lines start `build checks:`, `graft reading 1/2/3:`, `graft site sizes`.

**Inputs.** `artifacts/vpd/` (3.0 GB, the 67M); `artifacts/sets/train_0_32768.npy` (129 MB) and `val_0_2000.npy` (7.9 MB), both cached. Read at import: the `predictions.json` of `held_model_retention`, `rule_retention`, `rule_retention_fixed`, `rule_retention_pairs` (small, tracked). No `results/large` input of any experiment.

**Outputs.** `results/stage0.json` .. `stage3.json`, `results/run.log` (commit all, small); `results/large/graft.pt` (1.5 MB, the graft's four weights, sha256 in `stage1.json`); `results/large/map3_attn3.pt` (1.2 MB, reading 3's map at the graft's site, bf16). On a pod make `experiments/graft_negative/results/large` a real directory and download both files to `results/large_dl/` before terminating.

**Time** (not measured; estimates): data and model ~1 min; stage 0 ~2 min (one fit over 131,072 positions, three 512-window reads, the pattern pass); stage 1 ~3-6 min (4,000 steps through the frozen 4-layer model, full logits of 64 x 32 tokens); stages 2-3 ~3 min. About 10-15 min in all. The first `step 250` line gives the step time: over 0.15 s per step means stage 1 alone passes 10 min.

**GPU memory.** Peak ~16 GB, in E197's `read` (three 512 x 32 x 50,277 fp32 logit tensors, 3.3 GB each, plus the logit-error temporaries); training under 3 GB. An L40 fits.

**Where.** Cluster `debug` QoS on one L40 (`--comment=E201`, `--gres=gpu:1`, `--mem=48G`, `--time=00:45:00`): $0, inside the 2 h cap, no upload. A pod only if debug cannot start it within 5 min (inputs ~3.1 GB to sync, ~5 min at 10 MB/s).

**Order.** The code imports E197's `run.py` (`read`, `misses`, `CHECKS`) and `commute.held_model` unchanged; it does not read E197's results. Run it after E197 reports, as the cell plans.
