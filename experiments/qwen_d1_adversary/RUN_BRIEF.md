# Run brief: E80 (qwen_d1_adversary), can VPD's adversary fail do-nothing on Qwen3-1.7B?

Plan `reports/plans/2026-11-04_plan_e80-adversary-do-nothing.md`, its last section over the rest. Rules and early checks: `results/predictions.json` (run refuses to start without it, and while `results/stopped.json` exists). Code: `run.py`, `rules.py`, `upload.py`. Under the gate; no rehearsal, the control runs first.

**1. Upload first, $0, cluster.** From the clone root, `WANDB_API_KEY` in the shell (if missing, name it and stop): `sbatch cluster/qwen_d1_adversary_upload.sbatch`, ~5 min. The log ends `[upload] edwinagnew-mats/commute/qwen3-1.7b-e80-inputs:v0 ...`. If it prints a version other than v0, set `VERSION` in `upload.py` and commit before the pod.

**2. Pod.** One RunPod H100 80 GB (SXM or PCIe), any region, no network volume, container disk 120 GB (decomposition 53.7 GB, δW 14.1 GB, in-span X 5.6 GB, inputs 0.44 GB, venv), ~$3.99/h. `cleanup = "terminate"`. Never edit `remote-kernels.toml` while a machine runs; label through `start()`: `commute-qwen_d1_adversary-<id>`. Needs `WANDB_API_KEY` on the pod. Sync the tracked repo as other pods do; every `results/large` arrives as a dangling symlink, and this run reads and writes none (no staging needed).

Set-up, ~30 min, ~$2. `uv sync` in one background process, and in another, each teed to the console (no torch; exit 0 means sizes and, for the last two, sha256 match the committed sums):
```
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/qwen        # ~20 min at ~45 MB/s
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files fetch /root/e40_nowarmup v0   # qwen3-1.7b-e40-nowarmup-dW:v0, ~5 min
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_d1_adversary/upload.py fetch /root/e80_inputs v0   # qwen3-1.7b-e80-inputs:v0, ~10 s
```
The decomposition artifact is `qwen3-1.7b-p-e9fcffd5-torch:latest`. The run calls the same fetches and keeps what is complete, so the inputs must sit beside `--decomp`, as above.

**Command** (repo root, background subprocess; resume: the same command): `.venv/bin/python experiments/qwen_d1_adversary/run.py --decomp /root/qwen 2>&1 | tee -a experiments/qwen_d1_adversary/results/console.log /proc/1/fd/1`. No `execute()` or `wait()` over 5 min; re-arm. `console.log` is the runner's: download it only if `run.log` is short of something.

**GPU memory (plan).** Fixed 43.5 GB (target 6.9, fp32 components 19.5, Γ 17.1), + δW 2.8, + A's cut offsets 1.7 = 48 GB. Row costs are estimates (a 512-token row ~10-15 GB, 256 about a quarter); no one has logged them. Peaks: W0 W ~60, W0 M ~52, N M ~58, N W ~66, X ~52 (~2-4 GB of fp64 for the largest site, no adversary running), I M ~58, A M ~58, stage 6 ~56. All on 80 GB. Each start logs its peak; report them. Host ~35 GB (δW is on the card; 13 GB of alive columns pinned; FT's clean logits for 16 rows 2.5 GB).

| stage | time, $ (at $3.99/h) | first values to see (`results/run.log`) | stop (in code) | download at its end |
|---|---|---|---|---|
| load | ~6 min, 0.40 | `[dW] ... sha256 8b8509d2...`, `[load] ... resumed []` | an input's sha256, alive count, card under 70 GiB: exit with a traceback, tell the lead | `run.log` |
| 1 W0 W s0 | 21 min, 1.40 | `[W0_W_s0] seed 0 step 0: KL ~0.09`, ~60 s per step; `final KL ~1.70` | first step not finite; final outside 1.36-2.04 | `results.json`, `run.log` |
| 2 W0 M s0, s1 | 11 min, 0.70 | `[stage 2] W0 on M ...: band ...`, ~15 s per step | first step not finite | same |
| 3 N M s0, W s0 | 26 min, 1.75 | `[check N_clean] <1e-4` first | N's clean KL not under 1e-4; first step | same, after each start |
| 4 I M s0 | 15 min, 1.00 | `[I] X ready`, ~5-10 min; `[check I_clean] ~0.009` | I's clean KL above 0.05; first step | same |
| 5 A M s0 | 6 min, 0.40 | `[A_M_s0] final KL` | first step | same |
| 6 own, weights | 5 min, 0.35 | `[own W0 M] ... "self": ~0.043`; `[weights] ... "A": ~1.36, "N": 1.0, dW_squared ~3.49` | none | `results.json`, `run.log`: then terminate |

In all ~1.5 h of run, ~2 h with set-up, ~$8. Download `experiments/qwen_d1_adversary/results/results.json` and `run.log` into the matching local folder after every start (they are small, and nothing else is kept); check the layout after the first. Terminate once the final `results.json`'s sha256 matches on both sides, before anything local. A STOP writes `results/stopped.json` and exits 1 after scoring: download `results.json`, `run.log`, `stopped.json`, terminate, tell the lead; no code clears it. A crash (out of memory, no stopped.json): tell the lead with the last peak before any rerun.

**Reporting.** At each stage's end: timestamp, peak GPU, the KLs. Any stop or surprise goes to the lead before the next stage. Commit `results/*.json` and `run.log`; no `results/large` exists for this job.
