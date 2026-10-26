---
name: pod-runner
description: Hand a GPU run (cluster or RunPod) to the pod-runner agent. Use when the user types /pod-runner, or before launching any smoke or scored GPU run, to write the brief and spawn the agent in the background.
argument-hint: "[experiment / stage / notes]"
---

# Hand a run to pod-runner

The lead session never runs `sbatch`, `srun` or remote-kernels `start()` itself. It writes a brief and spawns one
`pod-runner` agent (Agent tool, `subagent_type: "pod-runner"`, in the background) per run. The agent starts with no
memory of this conversation, so the brief has to stand on its own. Write it from what you know; `$ARGUMENTS` (if any)
names the run.

Before spawning, check that:
- for a scored run, `experiments/<name>/results/predictions.json` exists; a separate smoke only when the rule in
  `.claude/CLAUDE.md` asks for one (a paid run over ~30 min, or an expensive late stage);
- `make lint` is clean on the changed files, and the adversarial review was run if measurement code changed.

If one of these fails, tell the user instead of spawning.

## Brief (fill in every line; write "none" rather than leave one out)

```
Experiment: <name>   Run: <smoke | scored | smoke, then scored if clean | setup/other>
Command: <exact command(s) or sbatch script, with stage arguments>
Expected: <runtime from the smoke/benchmark>, <GPU memory>, <CPU RAM>
Outputs: <files the run writes; mark which are the results to bring home; which are large/>
         (the job's final "FINISHED" line names these paths, so list them all)
Inputs not in git: <gitignored files the run needs, or none>
Platform: by expected run time (default): ≤ 30 min → cluster debug, RunPod if no start within 10 min;
          longer → RunPod directly, card 4090 or faster (H100 for runs over ~1 h) | cluster only because <reason>
Pod: fresh (default) | reuse an idle commute pod of this session if status() shows one | keep running after,
     the next run follows within ~1 h
Budget: <$ cap for RunPod: what is left of the session's REMOTE_KERNELS_BUDGET ($40), less ~15% headroom>
Watch for: <known failure modes, numbers to copy from the log>
```

By default write "scored" (the scored run is its own smoke, `.claude/CLAUDE.md`). Where a smoke is required, write
"smoke, then scored if clean" and spawn one agent: it submits the scored job itself once the smoke log is clean. Two agents in sequence cost a second queue wait and a
second sync (2026-10-20: a 15-min emoticon run took over two hours this way).

After spawning, tell the user in one line which agent is running what and the expected wall time from the brief's
"Expected" line (queue cap + setup + run), and how many pod-runners are alive. When a runner hands back, say so in
the next reply: the user sees its tab and cannot otherwise tell a finished runner from a hung one. The agent prints its own ETA line in its tab before it launches anything
and again whenever the plan changes. Don't monitor the run yourself; wait for the agent's report, then check that the
files it lists are on disk before analysing them.
