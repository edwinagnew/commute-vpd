# Run brief: E44 (qwen_maps), Q0 + Q1 on E40's saved δW

Plan `reports/plans/2026-10-28_plan_qwen-q0q1.md` (the spec; §4 stages, §5 rules). Rules: `results/predictions.json` (run refuses to start without it). Code: `run.py`, `job.py`, `src/commute/qwen_maps.py`, `src/commute/maps_score.py`.

**1. Rehearsal, $0, cluster `debug` (before any pod; `cluster/qwen_maps.sbatch` has the commands).** (a) `--stages R` at full size: R.json `holds` true (Gram ranks = SVD ranks, residual and split exactness < 1e-6); false → the pod run adds `--svd-bases` and H is dropped. Write down R's per-site Gram/SVD times. (b) `--rehearse --stages C,S,D,K,M,G,A,H`: every stage end to end at tiny size; expected misses are logged `REHEARSAL: would stop here` (control_W0, probably teeth); anything else (a traceback, a NaN) is fixed before the pod. Note host peak (`host N GB` per stage) and the S first-layer time. (c) the same command again: every stage `done earlier: skipped`.

**2. Pod.** One RunPod H100 80 GB SXM, US-MO-1, network volume `ydavm7ct4x` at `/workspace`, container disk 100 GB, ~$3.0/h. `remote-kernels.toml` must name `ydavm7ct4x` and US-MO-1 **before `start()`** (it names `2bun37yawb`, US-NC-1 now); the lead edits it, never while a pod of the session runs. Label `commute-qwen_maps-<id>`.
- Check on the pod: `ls -l /workspace/e40/dW/` shows `step_1.safetensors`, `step_10.safetensors`, `step_100.safetensors` (2.82 GB each) and `step_100_fp32.safetensors` (5.6 GB); missing → stop, tell the user. Decomposition at `/workspace/qwen3-1.7b-p-e9fcffd5-torch` (`index.json` present); else fetch with `commute.qwen_files` into `/root/qwen` (W&B, ~45 MB/s).
- Small inputs not in git (~20 MB), staged by hard link into a path ending `results/large/` and synced with `sync(include=[that dir])`; sha256 both sides. From `~/commute-data/results-large/qwen_sft/nowarmup/`: `sources/{G_fineweb_seed0,W0_fineweb_seed1,W0_fineweb_seed2,G_maths_seed0}.safetensors`, `gstats/W0_{maths,fineweb}.safetensors`, `step0_test.jsonl`. On the pod, replace the dangling symlink `experiments/qwen_sft/results/large` with a real directory holding `nowarmup/` with those files (`run.py` reads `--e40-large`, default that path).
- `df -h /root` before S: bases ~34 GB + S products ~22 GB (+11 GB at H) must fit with 2 GB spare (the code checks and raises).

**Command** (repo root, background subprocess, tee to the console): `.venv/bin/python experiments/qwen_maps/run.py --stages C,S,D,K,M,G,A,H --out /root/e44 --e40 /workspace/e40 --decomp /workspace/qwen3-1.7b-p-e9fcffd5-torch --usd-per-h 3.0 --prior-usd <setup spend so far> 2>&1 | tee -a experiments/qwen_maps/results/console.log /proc/1/fd/1`. Add `--svd-bases` if R missed. A resume is the same command: finished stages are skipped, D/K/M/G resume per state and step; S is redone if `/root/e44` was lost.

| stage | time, $ | first values to see (log) | stop rule (in code) | download when it ends |
|---|---|---|---|---|
| C | 12 min, 0.60 | `[C] {"ci": {"maths": ~0.05886, "web": ~0.14167} ... "replay": ~1.7032, 1.4365, 1.7472, 4.2351 ... "kl_to_start": ~0.1062, 0.5204, 0.664 (both 100 files)}` | any part of control_W0 off by > 1% (steps 1, 10: 1% + 3·gap) → `STOP`, `stopped.json` | `results/{C,score}.json`, `run.log` |
| S | 35 min, 1.75 | `[sigma] 196 sites`, `[S] layer 0: N s` (N × 28 ≤ 3000 s) | first layer × 28 > 50 min → STOP | `S.json` (~1 MB), `/root/e44/arrays/placement_step{1,10}.safetensors` (~8 MB each) |
| D | 8 min, 0.40 | `[D] memory rule: first batch peak N GB` (expect 45-75; > 70 halves the batch); D r1m step 1 ~2-3.5, step 10 ~8-11 | out-of-memory after the halving → STOP | `D.json` |
| K | 24 min, 1.20 | `[score] ... "outcome": {"teeth": ...}` | no teeth at step 10 → STOP before M; tell the user (decision tree) | `K.json`, `score.json` |
| M | 24 min, 1.20 | `pass_ls_10`, `pass_inspan_10`, `pass_adjoint_10` in score | — | `M.json`, `score.json` |
| G | 12 min, 0.60 | `[do_nothing 1] batch 1/2 ...` | — | `G.json` (responses in `/root/e44/acc/*.jsonl`, ~0.5 MB each, optional) |
| A | 22 min, 1.10 | `seed 0 step 0: KL ...` each step | spend + 22 min > $9.5 → skipped | `A.json`, `/root/e44/A_sources.safetensors` (2.8 MB) |
| H | ≤ 20 min, ≤ 1.00 | only if a split passed at 10 | spend + 20 min > $9.5 → skipped | `H.json`, `placement_step100.safetensors` |

Spend reaching $10 after any stage stops the job. Download into `experiments/qwen_maps/results/` (JSONs, log) and `experiments/qwen_maps/results/e44_dl/` (arrays; never into a `results/large` symlink); terminate once the last JSON's sha256 matches, before any local move. Every number is also in `results/run.log`. Report to the lead at each stage's end; any STOP or surprise goes to the user before the next stage.
