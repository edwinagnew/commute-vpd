# Working in this repository

Instructions for coding agents (and humans). Claude Code loads this file automatically. Keep it short; details live
where the layout points. Conventions are inherited from `../mono/feature_wars` (2026-10-14).

## What this is
Research code for `commute`: can a small gradient update from post-training be commuted past a parameter
decomposition (VPD) of the model, so that the decomposition stays interpretable continuously along training? If
not, is that because post-training teaches new capabilities rather than strengthening existing ones, or because
parameter decompositions are too brittle? Read `README.md` first, then `GOALS.md`.

**Before proposing any new experiment, task, architecture, theory direction or measurement, run `/design`. It checks
the proposal against `GOALS.md`, `experiments/TRIED.md` and `decisions.md`, and subagents must do the same. If the
skill won't load, read `.claude/skills/design/SKILL.md` and follow it. A high-level design choice needs the user's
confirmation after they have seen the check; an earlier "go ahead" does not count.**

## Layout (one line each)
- `experiments/<name>/` = `run.py` + `plot.py` + `results/` (incl. pre-registered `predictions.json`) + `figures/`.
- `experiments/TRIED.md` = everything tried or dropped, and why we moved on; update it with every experiment.
- `theory/` = notes and derivations. `notebooks/` = look at results here. `reports/` = dated write-ups
  (`YYYY-MM-DD_<prediction|analysis|plan|review|decision>_<topic>.md`). `backlog.md` = deferred ideas.
  One plan file per round. `README.md`'s "Current plan" line always names the latest plan, or says where the next
  round's inputs are collected (one `backlog.md` entry per upcoming round); keep both current.
- `GOALS.md` = the aim (user-owned). `decisions.md` = working choices, revisable.
- `artifacts/`, `data/` = downloaded checkpoints and datasets, gitignored, fetched by script. Untrusted: keep our
  scripts out of them and pass paths as arguments.
- `cluster/` = MATS Slurm scripts: venv setup job, `sync.sh` up/down, and `jupyter.sh` (a cluster-GPU kernel for VS
  Code). Read `cluster/README.md` first.

## Rules
- No `sys.path` manipulation and no scripts at the repo root. Everything imports from the editable install (`uv sync`).
- Never name a module after a stdlib module. Prefer `check_numbers.py`, `plot.py`, `run.py`.
- File predictions before running: `run.py --predictions` writes `results/predictions.json`; the run refuses to start
  without it. Do not edit a prediction after its experiment has run; add a dated report instead. A prediction is
  scorable only if the file also fixes the measurement rule, every threshold, and relative vs absolute.
- Results are committed (small summaries); anything over ~1 MB per run goes in the gitignored `results/large/`.
- Notebooks: short description before each figure, analysis after. Names in words, one question per figure, in its
  title. Lead with "could it do it?" (absolute) and "does it match the prediction?" (side by side).
- Concerns about existing claims go in a review report, not inline boxes.
- An edit to measurement code (anything in `src/commute/` or a `run.py` stage) is not done until you have re-read
  the edited function body and a CPU test in `tests/` (or a logged assertion in the run) shows the new path runs.
  Never launch a paid run on an unverified string replacement. Name a bug's cause only after the fix moves the number.
- Pin dependency versions in `pyproject.toml`. Compute choices follow `../guides/GPU_Platforms_Guide.md`.
- Anything that trains or fine-tunes a model, or runs more than one benchmark forward pass, goes to a GPU, never local
  CPU. MATS cluster first: `ssh e.agnew@185.141.218.207` (the address that works for this user; the parent
  `../CLAUDE.md` lists `94.156.8.208`), Slurm account `c11`, partition `compute`, follow
  `../guides/mats_cluster_handbook.md`. rsync the tree to `/mnt/nw/home/e.agnew/commute`, `sbatch`, record the job
  ID, rsync `results/` back and commit locally. Caches in `/ephemeral/$USER`, never `/tmp`. If the queue blocks the
  work, use RunPod via the remote-kernels plugin (one 4090-class GPU, label pods `commute-<experiment>-<id>`).
- Compute budgets stay low: the per-session cap is $10. Anything expected to cost more needs the user's explicit
  confirmation before launch; benchmark a small slice and extrapolate first.
- Credentials (`WANDB_API_KEY`, `RUNPOD_API_KEY`, `HF_TOKEN`) come from the shell. Never write them into a repo file;
  if one is missing, name the variable and stop.
- External code: Goodfire's `param-decomp` (`vpd-paper` branch; `main` has moved on) is used through a pinned
  dependency (git pin to commit `74146b5` in `pyproject.toml`), never copied in. A read-only clone for browsing lives
  at `../param-decomp` (outside this repo, user's choice 2026-10-14); keep it at the pinned commit.

## Commands
```
make sync        # uv sync (editable install, dev tools)
make test        # pytest, fast
```
