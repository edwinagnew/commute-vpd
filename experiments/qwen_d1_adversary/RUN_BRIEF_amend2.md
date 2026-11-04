# Run brief: E80 Amendment 2 (qwen_d1_adversary), the Qwen3-1.7B column of the slide's scorecard

Plan `reports/plans/2026-11-04_plan_e80-adversary-do-nothing.md`, "Amendment 2". Rules and stops: `results/predictions_amend2.json` (the run refuses to start without it, and while `results/stopped_amend2.json` exists). Code: `amend2.py` (imports `amend1.py` and `run.py`, unchanged), `rules_amend2.py`. Under the gate; no rehearsal, the controls run first. It only reads `results/amend1.json` and `run_amend1.log`, and never writes them.

**1. Do not terminate after Amendment 1.** That brief's last line says terminate; the lead's brief for this run overrides it. Keep the pod: the inputs under `/root` (decomposition, δW, `e80_inputs/`, `e80_X/`) are re-used. Nothing lives in Amendment 1's process, so it may exit.

**2. Order at Amendment 1's end.** Download `amend1.json` and `run_amend1.log`, check sha256 both sides (a later sync must not put an older copy on the pod). Then `sync` these tracked files: `amend2.py`, `rules_amend2.py`, `results/predictions_amend2.json`. Never edit `remote-kernels.toml`.

**Command** (repo root, background subprocess; resume: the same command): `.venv/bin/python experiments/qwen_d1_adversary/amend2.py --decomp /root/qwen 2>&1 | tee -a experiments/qwen_d1_adversary/results/console_amend2.log /proc/1/fd/1`. It needs HF access for GSM8K (`qwen_sets.load`, seconds). No `execute()` or `wait()` over 5 min; re-arm.

**Times, at about $4.06/h.** Starts are measured on this card; the rest is scaled from named logs, not timed here.

| stage | time, $ | first values to see (`results/run_amend2.log`) | stop (in code) | at its end |
|---|---|---|---|---|
| load | ~3 min, 0.20 (amend1's 96 s, GSM8K) | `[card]` the same two names; `[load] ... resumed []` | card differs from Amendment 1's: `same_card` | download log |
| scales | ~5 min, 0.35 (Σ 177 s on E68's L40) | `[sigma]`; `[left in delta]`; `[check energy]` ~1440.09, `[check LS_sigma]` ~0.36374, `[check ADJ_sigma]` ~0.00046571 | any over 1% from E68's | `amend2.json`, log |
| gain reads | ~4 min, 0.25 (gold read under 30 s, E40's log; a state under ~10 s to build, E80's log) | `[check gold_W0]` 2.49521; `[check gold_FT]` gain ~1.81; `[check kept_N]`, `[check kept_LS]` 1.00; `[kept ADJ]`, `[kept A]`; `[check kept_W0_end]` 0.00 | W0 over 0.1%, gain over 5%, N's clean KL, kept_N, kept_LS, kept_W0_end over 1e-3 | same; **report the two kept numbers at once** |
| LS W s0 | 23 min, 1.55 | first step finite; `final KL`; `[score]` | first step not finite | same, then report the cell |
| ADJ W s0 | 23 min, 1.55 | as above | as above | same |
| LS W s1 | 23 min, 1.55 | as above | as above | same |
| ADJ W s1 | 23 min, 1.55 | as above, then the last `[score]` | as above | `amend2.json`, `run_amend2.log`: then terminate |

In all ~1.75 h, ~$7. **GPU memory:** ~66 GB per start, as N (66.1): LS and ADJ hold one copy of the weights, 5.6 GB, and move the columns in place. Scales add Σ (~11 GB) to the 48 GB load, brief. Host as Amendment 1.

**Terminate** when the final `amend2.json`'s sha256 matches on both sides, before anything local. A STOP writes `results/stopped_amend2.json` and exits 1 after scoring: download `amend2.json`, `run_amend2.log`, `stopped_amend2.json`, tell the lead before anything more is paid for. A crash: tell the lead with the last peak before any rerun.

**Reporting.** After every read, the number and the cell: the user asked for each as it comes. Commit `results/amend2.json` and `run_amend2.log` (and `stopped_amend2.json` if present). A third start is the user's decision, and is not in this code.
