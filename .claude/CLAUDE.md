# Working in this repository

Instructions for coding agents (and humans). Claude Code loads this file automatically. Keep it short: a rule that
does not change what the agent does goes. Conventions are inherited from `../mono/feature_wars` (2026-10-14).

## What this is
Research code for `commute`: can a small gradient update from post-training be commuted past a parameter
decomposition (VPD) of the model, so that the decomposition stays interpretable continuously along training? If
not, is that because post-training teaches new capabilities rather than strengthening existing ones, or because
parameter decompositions are too brittle? Read `README.md` first, then `GOALS.md`.

**Before proposing any new experiment, task, architecture, theory direction or measurement, run `/design`** (or read
`.claude/skills/design/SKILL.md` if it won't load). Since 2026-10-21 (user) the check is written into the plan and
work goes ahead without sign-off when it falls in an approved class of work or an approved plan's decision tree
(`/design` → "Approved classes"); everything else still needs the user's confirmation after they have seen the check.

**Terms.** `GOALS.md` → Terms is the vocabulary (component training, least-squares split, adjoint split, rescaling
split, outside share, do-nothing, LoRA split, fixed-shift hypothesis); its "old names" column decodes older reports. "Pull-back", "lift",
"projection" and "H′" (now the fixed-shift hypothesis) are retired: not in new prose or code (`.claude/skills/writeup/check.py` flags them). When the user's words
and a file's definition of a term disagree, the user's words win: say so at once, give both readings in one line each,
and get the user's explicit acknowledgement of which holds before designing or running anything. Never pick a side
silently. A plan restates the user's request in the Terms' words, and it is that restatement the user confirms
(2026-10-20, `reports/reviews/2026-10-20_review_terminology.md`).

## Layout (one line each)
- `experiments/<name>/` = `run.py` + `plot.py` + `results/` (incl. `predictions.json`) + `figures/`.
- `experiments/TRIED.md` = everything tried or dropped, and why we moved on; update it with every experiment.
- `docs/research_log/parts/` = the research log's source, one file per section and per experiment card, built by
  `docs/research_log/build.py` into four pages (main, experiments, glossary, changelog) and published as the artifact
  "Update Decomposition Log" (URL, layout and publish recipe in `docs/research_log/README.md`). Edit only the parts a
  change touches; grep, don't read whole sections. **Updating it is part of the research flow**, batched since 2026-10-21 (user): every analysis
  report, decision report, new `TRIED.md` row, dropped proposal or changed claim appends a line to
  `docs/research_log/PENDING.md` when it is written; the queue is applied and republished before the session ends
  (recipe and completeness check in `docs/research_log/README.md`). Run its `--check` against the live pages before
  editing; another session may have published.
- `theory/` notes. `notebooks/E<NN>_<name>.ipynb` explain results, one per experiment. `reports/<type>/` dated
  write-ups, one subdirectory per type (`plans/ analyses/ reviews/ decisions/ predictions/ digests/`), each file
  `YYYY-MM-DD_<type>_<topic>.md`; nothing at the top of `reports/` (user, 2026-10-26). `backlog.md` deferred ideas.
  `README.md`'s "Current plan" line names the latest plan; keep it current.
- `GOALS.md` = the aim (user-owned). `decisions.md` = working choices, revisable.
- `src/commute/` = everything shared between experiments. `artifacts/`, `data/` = downloads, gitignored, untrusted
  (keep our scripts out of them, pass paths as arguments). `cluster/` = Slurm scripts; read `cluster/README.md`.
- `~/commute-agents/fleet/` = the harness that runs the agents (`regating-lead`, `commute-lead`) on the dev node: one
  copy outside this repo, its own git repository since 2026-10-27; `README.md` there. On the dev node, Python, tests
  and lint run only through `~/commute-agents/fleet/cpu-run` (a hook refuses them otherwise).

## Writing code
The goal is code simple enough that a strange number can be told apart from a bug by reading it. More tests and more
process do not get there; less code does.
- **Fewer measurements.** An experiment reports the few numbers its predictions use, plus one positive control.
  Before writing code, write down what each number should read at W0, for the do-nothing reference and for the
  control. A number without an expected value is cut or given one. If a result can't be read as "as expected" or
  "not", the measurement is too complicated: simplify it before running more.
- **Delete rather than add.** Fix a bug by removing the structure that allowed it, not by adding a check beside it.
  Delete dead arms, flags and filed-prediction text that the run no longer reads.
- **One config.** If a run has a `--smoke` path, every size that differs lives in one frozen dataclass whose
  `__post_init__` asserts the sizes agree (see `experiments/task_ladder/run.py`). No `x if args.smoke else y`.
- **No copying between experiments.** Shared steps (fine-tune, component training, the split, evaluation) live in `src/commute/` and are
  called. Finished experiments' `run.py` stay as they ran; do not use them as templates. Aim for run scripts of
  ~300 lines and functions of under ~50 lines; split a function rather than add a branch to it.
- **Read the existing code critically.** It was written fast. Follow its interfaces, not its habits.
- `make lint` (ruff + pyright) is clean on any file you touch, when it can run. If `~/commute-agents/fleet/cpu-run` says the worker is
  drained or down, commit unlinted and lint later; never start a pod just to lint (user, 2026-10-26).
- **The scored run is its own smoke** (user, 2026-10-21). Stages run cheapest first, the W₀ control first, and each
  writes its results as it finishes, so a code error stops the job in its first minutes. A separate smoke only when a
  failure is expensive: a paid run over ~30 min, or an expensive stage that starts only after a long one (smoke that
  stage, tiny sizes). New experiments need no `--smoke` path. Never load the model locally to test (the laptop
  OOM-killed the session, 2026-10-21). An edit to measurement code is done when you have re-read the edited function
  and it has run once on a GPU. Name a bug's cause only after the fix moves the number.
- **The control runs first.** The arm that validates the apparatus (W₀, identity, harmlessness) runs first and
  alone, with an early-stop rule in the runner's brief, before any arm it is meant to read; a plan that orders it
  later is sent back at `/design`. Any recipe that reproduces a reference procedure (a warm start, a refit, a replay)
  lists what differs from the reference's end state (optimizer state and schedule position, adversary state, data,
  coefficients) and either removes each difference or names it as the thing the control tests (E15, 2026-10-21:
  $10 went on an arm before its control showed the recipe drifted on its own).
- **One review before a paid run** of new or changed measurement code: run the Codex adversarial review on the diff
  (codex-companion `adversarial-review`, started via Monitor), fix what is real, say what you skipped and why. Codex
  is out of quota until ~2026-11-18 (user, 2026-10-26); until then spawn a fresh Opus agent that did not write the
  code or run the analysis, giving it only the scope and the repo path.

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
  goes to a GPU, never local CPU. Route for the earliest results, not the
  lowest $/h (user, 2026-10-20; this overrides `../CLAUDE.md`'s cluster-first and no-over-provisioning defaults for
  this project): runs of ≤ 30 min go to the MATS cluster's `debug` QoS (`ssh e.agnew@185.141.218.207`, account
  `c11`, partition `compute`, `../guides/mats_cluster_handbook.md`; record job IDs; caches in `/ephemeral/$USER`),
  and to RunPod if not started within 10 min. Longer runs go straight to RunPod via remote-kernels (one GPU, a 4090 or
  faster, an H100 for runs over ~1 h; pods `commute-<experiment>-<id>`). Qwen line only, for now (user, 2026-10-27):
  a longer run that fits an L40 may use the cluster's `normal` QoS when the queue is empty at submission.
- **Every GPU run goes through `/pod-runner`** (`.claude/skills/pod-runner/SKILL.md`): the lead writes the brief and
  spawns one `pod-runner` agent (Opus) in the background per run. The lead never runs `sbatch`, `srun` or `start()`
  inline, and doesn't use a fork for this: a fork re-reads the lead's whole context on every polling turn.
- Budget: the session's `REMOTE_KERNELS_BUDGET` (`.claude/settings.json`; $40, user 2026-10-21) is the limit, with no
  per-run sign-off below it. Plan to stay ~15% under it: the plugin terminates machines when it is hit. Anything more
  needs the user's explicit confirmation. Benchmark a slice first.
- Credentials (`WANDB_API_KEY`, `RUNPOD_API_KEY`) come from the shell, never a repo file; if one is
  missing, name the variable and stop.
- Pin dependency versions in `pyproject.toml`. Goodfire's `param-decomp` is a pinned dependency (commit `74146b5`,
  `vpd-paper` branch), never copied in; the read-only clone at `../param-decomp` stays at that commit.

## Commands
```
make sync        # uv sync (editable install, dev tools)
make test        # pytest, fast
make lint        # ruff + pyright on src/ tests/ and the current experiment
```
