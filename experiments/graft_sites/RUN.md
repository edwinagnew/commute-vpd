# E206 (P1.28): facts for the pod-runner

**Command** (clone root): `sbatch --comment=E206 cluster/graft_sites.sbatch` (cluster debug, one L40, `--mem=48G`,
`--time=00:50:00`). It runs `python experiments/graft_sites/run.py 2>&1 | tee -a experiments/graft_sites/results/run.log`, no
arguments. On a pod, prefix `E120_PRED_SHA=<git show HEAD:experiments/graft_sites/results/predictions.json | sha256sum>`.

**Stages.** 0: controls (do-nothing, zero-write grafts, attention pattern, head shares against E201's, lookup set and
M's null); a miss exits 1 with `results/stage0.json` holding `"stopped"`: stop and report, never rerun. Then per graft
(`last_lowest`, `neurons`, `early_lowest`, `last_rank5`, `last_rank9`, `last_rank13`): `results/grafts/<role>_build.json`
(4,000 steps), `_checks.json` (a miss is reported, the job goes on), `_read.json`. Every file is skipped on a rerun once
written, so a crash or timeout resumes by resubmitting the same command. Last: `results/summary.json`. Log markers
`STAGE0 DONE`, `<ROLE>_BUILD DONE`, `<ROLE>_CHECKS DONE`, `<ROLE>_READ DONE`, `E206 DONE`. Scored lines start
`<role> build checks:`, `<role> reading`, `<role> graft site`, `S per graft`, `rules:`.

**Inputs.** `artifacts/vpd/` (the 67M); `artifacts/sets/train_0_32768.npy` and `val_0_2000.npy` (cached Pile rows);
`experiments/graft_negative/results/stage0.json` (E201's shares, tracked). Read at import: the `predictions.json` of
`graft_negative`, `held_model_retention`, `rule_retention`, `rule_retention_fixed` and `rule_retention_pairs` (small,
tracked). No `results/large` input of any experiment.

**Outputs.** `results/stage0.json`, `results/grafts/*.json`, `results/summary.json`, `results/run.log` (commit all,
small); `results/large/<role>.pt` (graft weights, 1.5-1.6 MB each) and `results/large/map3_<role>.pt` (bf16, 1.2 MB per
head, 4.7 MB for the neurons).

**Time** (estimated from E201's 3.4 min job): stage 0 about 3 min, each graft 2-3 min, about 20 min in all. The first
`step 500` line gives the step time: over 0.1 s per step means the six builds alone pass 40 min.

**GPU memory.** Peak about 16 GB in the reads (E201's), an L40 fits.
