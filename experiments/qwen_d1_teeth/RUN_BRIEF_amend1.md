# Run brief: E85 Amendment 1 (qwen_d1_teeth), positive controls

Plan `reports/plans/2026-11-04_plan_e85-adversary-teeth.md`, "Amendment 1". Rules and stops: `results/predictions_amend1.json` (the run refuses to start without it, and while `results/stopped_amend1.json` exists). Code: `amend1.py`, `rules_amend.py`, over `run.py`, `rules.py` and `experiments/qwen_d1_adversary/`, all unchanged. Under the gate, no rehearsal: the controls run first. **Read Amendment 2's `[clean Rand x30]` first; if it is under 0.2 tell the lead, and do not start a pod.**

**Card.** One H200 NVL on Vast at about $4.06/h (RunPod refused E80), over 66 GB of GPU memory in use, `set_up` asserts over 70 GiB. Another card model works, but W0 is then read fresh, 22-31 min more: tell the lead before starting one. Container disk 100 GB.

**Inputs. Nothing outside git and W&B; nothing from any `results/large`, so no hard-link staging.** Sync the tracked repo, which includes `experiments/qwen_d1_teeth/results/run.log` (E85's W0 trace), `experiments/qwen_d1_adversary/results/run_amend1.log` (the host check's reference), `experiments/qwen_adversary/results/tokens.json` (the rows). `WANDB_API_KEY` and Hugging Face access (GSM8K) on the pod. Set-up, E85's three fetches, with `uv sync` alongside:
```
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/qwen                # decomposition, 53.7 GB
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files fetch /root/e40_nowarmup v0   # δW, 14.1 GB; step_10's sha256 is checked
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_d1_adversary/upload.py fetch /root/e80_inputs v0   # statistics, cut, 0.44 GB
```
**Command**, repo root, background subprocess, resume by the same command. `--perm` is the lead's choice, in the brief; the plan recommends `rare`, since `all` is trivially failed:
```
.venv/bin/python experiments/qwen_d1_teeth/amend1.py --decomp /root/qwen --perm rare 2>&1 | tee -a experiments/qwen_d1_teeth/results/console_amend1.log /proc/1/fd/1
```
No `execute()` or `wait()` over 5 min; re-arm.

| stage | time, $ | first values in `results/run_amend1.log` | stop (in code) |
|---|---|---|---|
| load | ~2 min | `[load] NVIDIA H200 NVL, GPU peak 48.2 GB` | |
| W0 | ~6 min, 0.40 | `[clean W0]`: gold 2.49521, ci ~0.1417; `[check gold_FT]` ~1.81; `[host check]` 0.0918, 0.1131, 0.1376; `[w0] 1.70315...` | gold over 0.1%, ci over 25%, gain over 5%, host over 2e-4 |
| Rand30 clean | ~2 min, 0.15 | `[clean Rand30]`, `[kept Rand30]`; **report kl_W0 at once** | kl_W0 under 0.2 |
| Rand30 start | 22 min, 1.50 | `seed 0 step 0: KL` 0.085 to 0.0918 + 1.5 kl_W0, step 2 above it; a `ratio ..., excess ...` line per step; `[score]` | first steps. **Report the final ratio and excess** |
| Stage 2 clean, only if `stage2_goes_on` | ~2 min, 0.15 | `[check share_PermRare]` ~0.8 (`share_Perm` ~2); `[clean PermRare]`: gold W0's, kl_W0 0, ci near 0.14 (Perm: several nats) | share, gold, kl_W0, ci |
| Stage 2 start | 22 min, 1.50 | PermRare step 0 at or over 0.085; Perm step 0 over 0.5, expected over 2.74 | first steps |

In all ~74 min, ~$5; with set-up included. If stage 1 misses the bar, `[stage 2] not run` ends it at ~$3.35.

**Results.** After every read download `results/amend1.json`, `run_amend1.log`, `console_amend1.log` into `experiments/qwen_d1_teeth/results/` (`stopped_amend1.json` too on a STOP). Nothing large. A lost pod costs the current start; a same-model card resumes from them. Terminate as soon as the final `amend1.json`'s sha256 matches on both sides, before anything local. Never edit `remote-kernels.toml`. A STOP: tell the lead before anything more is paid for. Commit the results files.
