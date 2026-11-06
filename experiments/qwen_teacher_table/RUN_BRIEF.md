# E111 run brief

Plan `reports/plans/2026-11-06_plan_e111-teacher-table.md` (Overrides). First: `sbatch cluster/qwen_teacher_table_upload.sbatch inputs R_s70` (5.6 GB); the other ten (53.6 GB) before stage 2.

**Pod**: H100 or A100 80GB, 128 GB RAM, 150 GB disk, `WANDB_API_KEY`, HF. Sync the tracked repo; beside `uv sync`:
```
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/e111/qwen
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_teacher_table/upload.py fetch /root/e111 R_s70
.venv/bin/python experiments/qwen_teacher_table/run.py --stage 1 --root /root/e111 2>&1 | tee -a experiments/qwen_teacher_table/results/console.log
```
Then `--stage 2` to `5`; each fetches its δW. A rerun resumes.

**Minutes** (L40 reads: `qwen_d1_table/results/run.log:219-230`; 0.77 s per step: `qwen_masked_teacher/results/run.log:35`; H100 untimed): fetch 25; stage 1 35, 2 70, 3 80, 4 70, 5 50.

**Memory**: card about 56 GiB, stop over 76; host about 100 GB (E93 89, `run.log:232`).

**Download** after each state: `results/<state>/state.json` (~0.4 MB), `run.log`, `table.json`, `console.log`; `stopped.json` on a stop. Offsets stay off the dev node: `.venv/bin/python experiments/qwen_teacher_table/upload.py offsets /root/e111/large` (0.44 GB per arm).

**Stops**: exit 1 with `stopped.json` is a missed early check: tell the lead, never rerun. Stage 1 goes to the lead before stage 2. Terminate once `table.json`'s sha256 matches.
