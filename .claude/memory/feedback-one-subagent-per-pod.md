---
name: feedback-one-subagent-per-pod
description: "Experiment runs (RunPod pods and cluster job batches alike) are driven by one pod-runner agent (Opus) per run, via /pod-runner; not a fork. The lead stays free for predictions, review, commits"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44afedb9-936d-43c9-8c5a-1d7e8ebad2be
  modified: 2026-10-14T11:43:52.508Z
---

The user asked (2026-10-14, commute first experiment) to "split into subagents for monitoring runpods": every RunPod
machine is driven by its own fork subagent (start, sync, install, run, download, terminate), labelled
`commute-<experiment>-<id>`. The lead never babysits a pod inline.

**Why:** pod runs block for tens of minutes; a fork keeps the lead free and keeps pod output out of the lead's
context. Same rule as feature_wars ([[feedback-one-subagent-per-pod]] there, [[feedback-no-reassigning-pod-work]]).
Pod names at the provider (user, 2026-10-14): the first pod came up as `remote-kernels-01M4X3ZPYNWDRS23CRZSGNMAVN`,
which the user did not want. The `start()` label is local display only; the RunPod name is `<name>-<id>` with `name`
from `remote-kernels.toml`. Set `name = "commute-<experiment>"` there before `start()` so pods read
`commute-<experiment>-<id>` (set to `commute-transport_test` on 2026-10-14; change it per experiment).

**Cluster too (user, 2026-10-14, round 2):** "in a seperate subagent as usual" — implementing, submitting and
monitoring a batch of cluster jobs also goes to a fork, not inline in the lead.

**Re-emphasised 2026-10-19 (lora_commute):** the lead had started the pod, created the kernel and begun setup inline
while waiting on a Codex review; the user interrupted: "remember to manage pods in seperate subagent!!". Even the
start/kernel/sync steps belong in the fork. Spawn the fork before `start()`, not after.

**Changed 2026-10-19 (user):** use the `pod-runner` agent (`.claude/agents/pod-runner.md`, model opus) via the
`/pod-runner` skill instead of a fork, to save weekly usage: a fork carries the lead's whole context (on the lead's
expensive model) into every polling turn, while a fresh agent with a written brief needs only ~10-20k tokens. The
agent defaults to the cluster for plain single-GPU scripts. Forks and general-purpose agents stay on the main model
for exploratory work.

**Amended 2026-10-20 (user):** "this has taken an awfully long time for what was promised to be a quick cluster
job ... two pod-runners going - possibly clashing ... the pod-runner documentation needs improving". A 15-min
emoticon rerun took over two hours: smoke agent (13-min sync of results/large, 8-min queue, then a hung laptop-side
ssh polling loop that noticed the 1.5-min job 40 min late), then a second agent and a second queue wait for the scored
run. Fixes in `.claude/agents/pod-runner.md` and the skill: a smoke and its scored run are ONE agent when the brief
says "smoke, then scored if clean"; waits are one ssh command running the loop on the controller under `timeout`;
`sync.sh up` excludes results/large (`up-large` sends it on demand); the lead tells the user how many runners are
alive and when one has handed back.

**How to apply:** one pod-runner per pod or job batch (a short smoke plus its scored run count as one batch), spawned before `sbatch`/`start()` with the brief from the skill; set the `name` prefix in `remote-kernels.toml` before starting; the agent downloads results before it ends; predictions are filed by the lead
before the scored run; the agent never terminates a machine another agent is meant to reuse unless told to. See
[[project-mats-cluster]] for when the cluster is used instead.

**Amended 2026-10-20 (user, E12):** "i stopped the cluster job because it was taking too long and the subagent wasnt
responding to me telling it that it was taking too long." The runner was inside one long queue wait. Rule added to
`.claude/agents/pod-runner.md`: while queued, waits are at most 5 min each and re-armed, so a user message in the
runner's tab is acted on within minutes; the queue cap for a short debug smoke is 10 min, then RunPod. The lead also
relays the user's instruction to the runner with SendMessage at once.
