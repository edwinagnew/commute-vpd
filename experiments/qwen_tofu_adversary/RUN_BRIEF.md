# Run brief: E89 (qwen_tofu_adversary), the adversary on E81's TOFU update

Plan `reports/plans/2026-11-05_plan_e89-tofu-adversary.md`. Rules and stops: `results/predictions.json` (the run refuses to start without it, and while `results/stopped.json` exists). Code: `run.py`, `rules.py`, `upload.py`, over `experiments/qwen_d1_teeth/` and `experiments/qwen_d1_adversary/`, all unchanged. Under the gate: the control runs first.

**Before the pod.** The artifact `qwen3-1.7b-e81-tofu-dW:v0` must exist: `sbatch cluster/qwen_tofu_adversary_upload.sbatch` from the clone root (CPU, ~5 min), and its log's last lines name the artifact and the sha256. If the version is not v0, tell the lead.

**Card.** 80 GB card, H200 NVL. Over 69 GB of GPU memory in use, `set_up` asserts over 70 GiB. Another card model moves W0's numbers: tell the lead before starting one. Container disk 100 GB; about 75 GB of it is used, with the half's X in `/root/e89_rare` (5.6 GB).

**Inputs. Nothing outside git and W&B; nothing from any `results/large`, so no hard-link staging.** Sync the tracked repo, which includes `experiments/qwen_adversary/results/tokens.json` (the rows). `WANDB_API_KEY` and Hugging Face access (Qwen's weights; GSM8K is not read) on the pod. Set-up, three fetches, with `uv sync` alongside:
```
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/qwen                                    # decomposition, 53.7 GB
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_tofu_adversary/upload.py fetch /root/e81_tofu v0   # δW, 5.64 GB; its sha256 is checked
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_d1_adversary/upload.py fetch /root/e80_inputs v0   # statistics, cut, 0.44 GB
```
E40's fetch is not needed. If `/root/e85_rare` exists on a reused disk, leave it: this run reads and writes `/root/e89_rare`, and that directory must be new to this update.

**Command**, repo root, background subprocess, resume by the same command:
```
.venv/bin/python experiments/qwen_tofu_adversary/run.py --decomp /root/qwen 2>&1 | tee -a experiments/qwen_tofu_adversary/results/console.log /proc/1/fd/1
```
No `execute()` or `wait()` over 5 min; re-arm.

| stage | time | first values in `results/run.log` | stop (in code) |
|---|---|---|---|
| load | ~2 min | `[dW] ... 5.64 GB`; `[load] NVIDIA H200 NVL, GPU peak` about 51 GB, `resumed []` | sha256, alive count, card under 70 GiB |
| clean N | ~2 min | `[check N_clean]` under 1e-4, about 1e-9 on u1; `[clean N] KL against W0`; **report it at once**; `[check N_moves]` | path check at or over 1e-4; KL under 1e-3 |
| X, left in Δ, clean Rare | ~3 min | `[Rare] X ready`, ~41 s; `[left in delta]`, ‖δW‖ 17.998449, share above 0.0984, may pass 1; `[clean Rare]`; `[check Rare_same_weights]` near 0 | share not finite or at or under 0.09844809; clean KL over 1e-6 from N's |
| W0 s0 | 22 min | `seed 0 step 0: KL 0.0918`, then 0.1131, 0.1376; final 1.703; `[check W0_s0]`. **If the three steps miss on this card, tell the lead at once** | a first step not finite; over 20% from 1.7031564672943205 |
| W0 s1 | 22 min | final 1.438; `[check W0_s1]`; `[score]` with `W0_spread` about 0.18 | over 20% from 1.4379397351294756 |
| N s0 | 22 min | step 0 finite and unmeasured; `[N_W_s0] final KL`, peak about 69 GB; `[score]` with `excess`, `last5`. **Report the ratio and excess** | a first step not finite; path check |
| Rare s0 | 22 min | `[Rare_W_s0] final KL`, peak about 69 GB; `[score]`. **Report the ratio and excess** | a first step not finite |
| N s1 | 22 min | as N s0 | as N s0 |
| Rare s1 | 22 min | as Rare s0; `[done]`, and `[score]` with `do_nothing` and `pair`. **Report both labels** | as Rare s0 |

In all ~2.7-3 h with set-up.

**Results.** After every read download `results/results.json`, `run.log`, `console.log` into `experiments/qwen_tofu_adversary/results/` (`stopped.json` too on a STOP). Nothing large. A lost pod costs the current start; a same-model card resumes from them, and rebuilds X if `e89_rare` is gone. **Terminate as soon as the final `results.json`'s sha256 matches on both sides, before anything local.** Never edit `remote-kernels.toml`. A STOP: tell the lead before anything more is paid for. Commit the results files.
