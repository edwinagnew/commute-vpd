# Run brief: E80 Amendment 1 (qwen_d1_adversary), do placements separate from do-nothing on the FineWeb rows?

Plan `reports/plans/2026-11-04_plan_e80-adversary-do-nothing.md`, "Amendment 1" over the rest. Rules and stops: `results/predictions_amend1.json` (the run refuses to start without it, and while `results/stopped_amend1.json` exists). Code: `amend1.py` (imports E80's `run.py` unchanged in behaviour), `rules_amend1.py`. Under the gate; no rehearsal, the control runs first. E80's `results/results.json` and `results/run.log` are E80's: this job reads the first and never writes either.

**1. Inputs.** The upload of E80 is done (`qwen3-1.7b-e80-inputs:v0`); no cluster step. `results/results.json` (with `W0_W_s0`, `N_W_s0`) must be in the synced repo: the run asserts it.

**2. Pod.** One RunPod H100 PCIe 80 GB, any region, no network volume, container disk 120 GB (decomposition 53.7 GB, δW 14.1 GB, in-span X 5.6 GB, inputs 0.44 GB, venv), expected $2.89/h, up to $3.99/h. `cleanup = "terminate"`. Never edit `remote-kernels.toml` while a machine runs; label through `start()`: `commute-qwen_d1_adversary-a1-<id>`. Needs `WANDB_API_KEY` on the pod. Sync the tracked repo as other pods do; every `results/large` arrives as a dangling symlink, and this run reads and writes none.

Set-up, the same fetches as E80 (`uv sync` in one background process, and in another, each teed to the console; no torch; exit 0 means sizes and, for the last two, sha256 match the committed sums):
```
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/qwen        # ~20 min at ~45 MB/s
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files fetch /root/e40_nowarmup v0   # qwen3-1.7b-e40-nowarmup-dW:v0, ~5 min
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_d1_adversary/upload.py fetch /root/e80_inputs v0   # qwen3-1.7b-e80-inputs:v0, ~10 s
```
The decomposition artifact is `qwen3-1.7b-p-e9fcffd5-torch:latest`. The run calls the same fetches and keeps what is complete, so the inputs sit beside `--decomp`, as above. Set-up on a fresh pod took ~10 min on E80's pod (measured), so the plan's ~10 min holds; the per-fetch times above are older estimates.

**Command** (repo root, background subprocess; resume: the same command): `.venv/bin/python experiments/qwen_d1_adversary/amend1.py --decomp /root/qwen 2>&1 | tee -a experiments/qwen_d1_adversary/results/console_amend1.log /proc/1/fd/1`. No `execute()` or `wait()` over 5 min; re-arm. `console_amend1.log` is the runner's: download it only if `run_amend1.log` is short of something.

**GPU memory.** As E80 (fp32 components, Γ bf16): peak ~66 GB on W (E80: N W 66.1, W0 W 60.5), host ~35 GB. Each start logs its peak; report them. The card must be 80 GB (the run asserts over 70 GiB).

**Speed.** ~91 s per step on this card for a W start (E80, measured), 20 steps, ~31 min per start (E80: 1870 s).

| stage | time, $ (at $2.89/h) | first values to see (`results/run_amend1.log`) | stop (in code) | download at its end |
|---|---|---|---|---|
| load | ~6 min after the fetches, 0.30 | `[dW] ... sha256 8b8509d2...`, `[load] ... resumed []` | an input's sha256, alive count, card under 70 GiB: traceback, tell the lead | `run_amend1.log` |
| 1 W0 W s1 (control) | 31 min, 1.50 | `[W0_W_s1] seed 1 step 0: KL` finite, ~91 s per step; `final KL` ~1.65-1.85 (no stop on the value; tell the lead if outside) | first step not finite | `amend1.json`, `run_amend1.log` |
| 2 N W s1 | 32 min, 1.55 | `[check N_clean]` under 1e-4 first; `final KL` about 1.28 x W0 s1; then `[reproduces] reproduces` | N's clean KL not under 1e-4; first step; **`[reproduces]` missed: `stopped_amend1.json`, score, exit 1, nothing further is paid for** | same, after each start; `stopped_amend1.json` if present |
| 3 I W s0 | 32 min (X 58 s on E80), 1.55 | `[I] X ready`; `[check I_clean]` ~0.009 | I's clean KL above 0.05; first step | same |
| 4 A W s0 | 31 min, 1.50 | `[A_W_s0] final KL` | first step | same |
| 5 I W s1 | 32 min, 1.55 | `[check I_clean]`, `[I_W_s1] final KL` | as stage 3 | same |
| 6 A W s1 | 31 min, 1.50 | `[A_W_s1] final KL`, then `[score] {...}` | first step | `amend1.json`, `run_amend1.log`: then terminate |

Each start also logs `mean of the last 5 step reads` (not ruled on). In all six starts ~3.1 h of run, ~3.3 h with set-up, ~$10 at $2.89/h (up to ~$13 at $3.99/h). Download `experiments/qwen_d1_adversary/results/amend1.json` and `run_amend1.log` into the matching local folder after every start (they are small, and nothing else is kept); check the layout after the first. Terminate once the final `amend1.json`'s sha256 matches on both sides, before anything local. A STOP writes `results/stopped_amend1.json` and exits 1 after scoring: download `amend1.json`, `run_amend1.log`, `stopped_amend1.json`, terminate, tell the lead; no code clears it. A crash (out of memory, no stopped file): tell the lead with the last peak before any rerun.

**Reporting.** At each start's end: timestamp, peak GPU, the final KL, the mean of the last five reads. Any stop or surprise goes to the lead before the next start. Commit `results/amend1.json` and `run_amend1.log` (and `stopped_amend1.json` if present); never touch `results.json` or `run.log`.
