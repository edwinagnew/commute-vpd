# Working in this repository

Instructions for coding agents (and humans). Claude Code loads this file automatically. Keep it short: a rule that
does not change what the agent does goes. Conventions are inherited from `../mono/feature_wars` (2026-10-14).

## What this is
Research code for `commute`: can a small gradient update from post-training be commuted past a parameter
decomposition (VPD) of the model, so that the decomposition stays interpretable continuously along training? If
not, is that because post-training teaches new capabilities rather than strengthening existing ones, or because
parameter decompositions are too brittle? Read `README.md` first, then `GOALS.md`.

**Before proposing any new experiment, task, architecture, theory direction or measurement, run `/design`** (or read
`.claude/skills/design/SKILL.md` if it won't load). A high-level design choice needs the user's confirmation after
they have seen the check; an earlier "go ahead" does not count. Subagents return the check instead of building.

## Layout (one line each)
- `experiments/<name>/` = `run.py` + `plot.py` + `results/` (incl. `predictions.json`) + `figures/`.
- `experiments/TRIED.md` = everything tried or dropped, and why we moved on; update it with every experiment.
- `theory/` notes. `notebooks/` look at results. `reports/` dated write-ups
  (`YYYY-MM-DD_<prediction|analysis|plan|review|decision>_<topic>.md`). `backlog.md` deferred ideas. `README.md`'s
  "Current plan" line names the latest plan; keep it current.
- `GOALS.md` = the aim (user-owned). `decisions.md` = working choices, revisable.
- `src/commute/` = everything shared between experiments. `artifacts/`, `data/` = downloads, gitignored, untrusted
  (keep our scripts out of them, pass paths as arguments). `cluster/` = Slurm scripts; read `cluster/README.md`.

## Writing code
The goal is code simple enough that a strange number can be told apart from a bug by reading it. More tests and more
process do not get there; less code does.
- **Fewer measurements.** An experiment reports the few numbers its predictions use, plus one positive control.
  Before writing code, write down what each number should read at W0, for the do-nothing reference and for the
  control. A number without an expected value is cut or given one. If a result can't be read as "as expected" or
  "not", the measurement is too complicated: simplify it before running more.
- **Delete rather than add.** Fix a bug by removing the structure that allowed it, not by adding a check beside it.
  Delete dead arms, flags and filed-prediction text that the run no longer reads.
- **One config.** Every size that differs between the full run and `--smoke` lives in one frozen dataclass whose
  `__post_init__` asserts the sizes agree (see `experiments/task_ladder/run.py`). No `x if args.smoke else y`.
- **No copying between experiments.** Shared steps (fine-tune, pull-back, evaluation) live in `src/commute/` and are
  called. Finished experiments' `run.py` stay as they ran; do not use them as templates. Aim for run scripts of
  ~300 lines and functions of under ~50 lines; split a function rather than add a branch to it.
- **Read the existing code critically.** It was written fast. Follow its interfaces, not its habits.
- `make lint` (ruff + pyright) is clean on any file you touch.
- An edit to measurement code (`src/commute/`, a `run.py` stage) is done when you have re-read the edited function
  and it has run on the smoke path (CPU test or GPU smoke). Never launch a paid run on code that has not run. Name a
  bug's cause only after the fix moves the number.
- **One review before a paid run** of new or changed measurement code: run the Codex adversarial review on the diff
  (codex-companion `adversarial-review`, started via Monitor), fix what is real, say what you skipped and why.

## Sessions
Quality drops in long sessions. **Tell the user, in one line at the end of a reply, when to start a new session**
and what to hand it (the plan or report file, the files to read). Do it when:
- a deliverable is finished (a run analysed, a refactor reviewed) and the next request is a different task;
- the context has been compacted, or the session has covered more than one experiment;
- you are about to write substantial new code after a long run/debug/analysis stretch;
- the next step is reviewing code written in this session (a fresh session reviews better than the author).
Write the hand-off into the relevant report or plan, not only the chat, so the new session can start from files.

## Rules
- No `sys.path` manipulation and no scripts at the repo root. Everything imports from the editable install (`uv sync`).
  Never name a module after a stdlib module.
- Predictions are filed in `results/predictions.json` before running; the run refuses to start without it. Never
  edit them after the run; add a dated report instead. A prediction fixes its measurement rule, thresholds, and
  relative vs absolute; the plan states what each metric is blind to, its positive control, and its noise floor.
  Every analysis report opens with a plain-words paragraph (three sentences, no symbols: what was done, what was found,
  what it means for `GOALS.md`) and ends with a "Not run" list.
- Results are committed (small summaries); anything over ~1 MB per run goes in the gitignored `results/large/`.
- **Every write-up (notebook, analysis report, result message, notebook-writing brief) follows `/writeup`**
  (`.claude/skills/writeup/SKILL.md`): the answer first in 80 plain words, the experiment as a story, one figure
  per claim with its reference on the same axes, about two numbers and no codes per sentence, and
  `check.py` run on the file before it is called done.
- Concerns about existing claims go in a review report, not inline boxes.
- Compute follows `../guides/GPU_Platforms_Guide.md`. Training, fine-tuning, or more than one benchmark forward pass
  goes to a GPU, never local CPU. MATS cluster first (`ssh e.agnew@185.141.218.207`, account `c11`, partition
  `compute`, `../guides/mats_cluster_handbook.md`; record job IDs; caches in `/ephemeral/$USER`). If the queue blocks,
  meaning the job has not started within 30 min (less for short runs; `sbatch --test-only` is meaningless on the
  `debug` QoS), RunPod via remote-kernels (one GPU, the cheapest adequate card in `remote-kernels.toml`'s list; pods
  `commute-<experiment>-<id>`).
- **Every GPU run goes through `/pod-runner`** (`.claude/skills/pod-runner/SKILL.md`): the lead writes the brief and
  spawns one `pod-runner` agent (Opus) in the background per run. The lead never runs `sbatch`, `srun` or `start()`
  inline, and doesn't use a fork for this: a fork re-reads the lead's whole context on every polling turn.
- Budget: $10 per session; anything more needs the user's explicit confirmation. Benchmark a slice first.
- Credentials (`WANDB_API_KEY`, `RUNPOD_API_KEY`, `HF_TOKEN`) come from the shell, never a repo file; if one is
  missing, name the variable and stop.
- Pin dependency versions in `pyproject.toml`. Goodfire's `param-decomp` is a pinned dependency (commit `74146b5`,
  `vpd-paper` branch), never copied in; the read-only clone at `../param-decomp` stays at that commit.

## Commands
```
make sync        # uv sync (editable install, dev tools)
make test        # pytest, fast
make lint        # ruff + pyright on src/ tests/ and the current experiment
```
