# Agents on the cluster: hand-off (plan, 2026-10-26)

**Claude, read this first.** The user has installed Claude Code on the cluster, copied this file over, and
started you. Walk them through the rest one step at a time: give one short instruction or one command, wait for
them to confirm, then give the next. Keep each message to a few lines. Order:
1. The human items in section 0b (2FA, GitHub push protection, the two PATs, the RunPod key and Secrets, 1Password,
   and checking which Claude account is in use). Mention the laptop's disk encryption once, as something to raise
   with the MATS team.
2. The rest of section 1: the secrets file, the clone, the git credential helper, and the plugins. Skip the steps
   they've already done.
3. Then do sections 2-6 yourself, starting with showing them section 3's plan. Also do the repo fixes from 0b in
   the clone: `exclude-newer` in `[tool.uv]` (followed by `uv lock` through `cpu-run` or a Slurm job, and stop if
   packages change), and dropping `HF_TOKEN` from `inherit-env` in `remote-kernels.toml` (only while no pod runs).

This file is for the first Claude session started on the MATS cluster's dev node, plus the person setting it up.
Its job is to build a small harness that keeps Claude agents working on `commute` overnight. The harness copies the
working parts of Will Mayner's `mats-fleet`, starts with one agent, and adds agents by adding a line. The human steps
(install, login, keys) are in section 1 and have to be done by you, since the agent can't do them. Sections 2-6 are
the agent's work. Show the user the plan in section 3 before building, because it changes how the repo is laid out
on the cluster.

## 0. Decisions already made (user, 2026-10-26)

1. **Compute split unchanged** (`.claude/CLAUDE.md` → Rules): runs of ≤ 30 min go to the cluster's `debug` queue,
   longer runs go to RunPod via remote-kernels, within the `REMOTE_KERNELS_BUDGET` ($40 per session) and RunPod credit.
2. **Keys may sit on the cluster, as mats-fleet's do,** in one file with mode `600`, outside every repo (section 1).
3. **Adopt mats-fleet's operating choices:** restart loop, handoff notes, context-size restarts, blocked-prompt
   detection, background job waits, a memory cap and no-compute hook, and model routing for subagents.
4. **Scalable design:** one agent (`lead`) to start. Adding a worker means adding a roster line and a role brief.
5. **Memory moves into the repo** (section 4). The reports and log don't hold it: about 35 of the 53 memory files
   are the user's working preferences, such as "short chat replies", "plots over tables", "user reads log and chat
   only" and "never leave a job unwatched". None of those are in `.claude/CLAUDE.md` or the skills (checked
   2026-10-26).

## 0b. Security compliance (MATS "Security & Best Practices", saved at `../guides/Security & Best Practices.html`, checked 2026-10-26)

| Rule | Status |
|---|---|
| Keys on a remote machine live there, private, outside shared team folders | ✓ `~/.config/commute/secrets.env`, mode 600, in the 700 home, not `/mnt/nw/teams` |
| Coding agents: expose only the credentials the task needs | **changed:** `HF_TOKEN` dropped (the tokenizer is public). Keys: `RUNPOD_API_KEY` (remote-kernels), `WANDB_API_KEY` (VPD fetch on pods), `GIT_PAT` (read-only, pods). Pushes from the cluster use a **separate** read/write PAT in git's credential store, so pods never hold a write token. Agents are denied `Read` of the secrets file |
| Keys created in the MATS project or workspace and kept in a password manager | human: the RunPod key from the MATS team account, the Claude login on the MATS-provided account (if it's a personal one, ask the Compute Team), every key in 1Password |
| 2FA (authenticator or passkey, not SMS) on every account the compute budget touches | human: GitHub, RunPod, Claude, W&B, Google |
| `.gitignore` credential patterns | ✓ added 2026-10-26 (`.env.*`, `*.key`, `*.pem`, `secrets/`, `*credentials*.json`, …). The history has no matches for common key prefixes |
| GitHub push protection | human: GitHub → Settings → Code security → Push protection |
| Never put literal keys in batch scripts or shared templates; RunPod keys go in RunPod **Secrets** | **deviation:** `remote-kernels.toml` `inherit-env` passes the values in as plain pod environment variables. Test `[env] WANDB_API_KEY = "{{ RUNPOD_SECRET_commute_wandb }}"` (and the same for `GIT_PAT`) on one cheap pod, then drop them from `inherit-env` |
| Slurm: export in the submission shell, or load a private file inside the job | ✓ sbatch's default `--export=ALL` carries the launcher's environment |
| Supply chain: no package versions under a week old | **to do:** add `exclude-newer` to `[tool.uv]` (section 1). Pin the Claude install to a known version |
| Full-disk encryption on the laptop | **✗ the laptop's `/` is plain ext4 (`nvme0n1p2`, no LUKS).** It needs a reinstall with encryption, or ask the MATS team. Until then the laptop holds MATS data unencrypted |
| Research security: an agent attempting unauthorised access ("sandbox escape", "unsolicited cyber actions") must be reported to the RM | The lead role brief must say: read and write only your clone, `commute-data/` and Slurm. Any attempt by an agent to read other users' folders or reach other systems is reported to the user, who tells the RM |
| A key is exposed | Revoke it, tell Compute (Ask-Compute: provider, project, time), replace it, restart the agents (`fleet/up stop && fleet/up`) |

## 1. Human steps (once)

```bash
ssh e.agnew@185.141.218.207
# PATH: ~/.local/bin (uv is already there) is not on the cluster's PATH yet
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
# Claude Code, native installer -> ~/.local/bin/claude (no Node needed). Laptop runs 2.1.310.
curl -fsSL https://claude.ai/install.sh | bash -s 2.1.310   # pinned to the laptop's version
claude --version
# Keys: one file, readable by you only, never sourced by .bashrc, never inside a repo
mkdir -p ~/.config/commute && chmod 700 ~/.config/commute
install -m 600 /dev/null ~/.config/commute/secrets.env
nano ~/.config/commute/secrets.env      # lines: export RUNPOD_API_KEY=... WANDB_API_KEY=... GIT_PAT=... (read-only PAT)
# GitHub push token: a SECOND fine-grained PAT for edwinagnew/commute-vpd only (contents read/write), short expiry
git clone https://github.com/edwinagnew/commute-vpd.git /mnt/nw/home/e.agnew/commute-agents/lead
cd /mnt/nw/home/e.agnew/commute-agents/lead
git config credential.helper store       # first push asks for user + PAT, stored in ~/.git-credentials (600)
# Login and plugins, in a first interactive session
tmux new -s setup
claude                                   # /login: open the URL on the laptop, paste the code back
```

Inside that first `claude`, add the marketplaces and plugins you use on the laptop:
```
/plugin marketplace add Crazytieguy/alignment-hive
/plugin marketplace add Crazytieguy/precis
/plugin marketplace add AlexanderMattTurner/agent-sanitizer
/plugin marketplace add Crazytieguy/codex-plugin-cc      # only if Codex reviews are wanted (needs Node + OpenAI login)
/plugin        # install: hive, debrief, mats, tldr, remote-kernels (x86_64), precis (x86_64), agent-sanitizer
```
Leave out `runpod` and `rust-analyzer-lsp`; remote-kernels covers RunPod. Set `AGENT_SANITIZER_SECRETS_ENABLED=1`
in `~/.claude/settings.json` → `env`, as on the laptop. Then tell this session: "read
`reports/plans/2026-10-26_plan_overnight-claude.md` and do sections 2-6".

**Why `/login` rather than `claude setup-token`:** the research log is published through the Artifact tool, and the
claude.ai connectors aren't reported to load with a setup-token token. Revoke the cluster login (claude.ai →
settings) when you stop using the cluster.

**Optional, for the reference material:** `rsync -a ~/work/mats/mats-fleet/{CLAUDE.md,ci,roles,.claude,.githooks,setup.sh,fleet.conf}
e.agnew@185.141.218.207:/mnt/nw/home/e.agnew/ref/mats-fleet/` from the laptop. Section 3 cites those files; it's
read-only reference, never committed into `commute`.

## 2. Facts about the cluster (checked 2026-10-26)

- The dev node is the host you SSH into. Its hostname is `vm1e3c1290f17b6`, with 32 cores and 62 GB, and it's
  shared. mats-fleet runs about 13 agents on it around the clock (its scripts reach it as 94.156.8.208). The only
  Slurm node is `l40-worker` (252 CPUs, 458 GB, 8×L40). `sacct` doesn't work from the dev node, so use `squeue` and
  `scontrol`.
- `systemd-run --user --scope` works, and so do `crontab` and `/usr/bin/tmux`. There's no `node` and no `gh`.
- `/mnt/nw/home/e.agnew` (mode 700) is on NFS. Admins can read it, other fellows can't. `/ephemeral` is local to
  the dev node, 83% full.
- **`/mnt/nw/home/e.agnew/commute` is not a git clone.** It's an rsync copy kept by the laptop's `cluster/sync.sh`.
  It holds the CUDA `.venv`, `artifacts/` (3.2 GB) and `experiments/*/results/large/`. `cluster/env.sh` hard-codes
  `COMMUTE_REPO` to it, and 47 sbatch files name it. Don't delete it or point an agent at it before section 3 is
  agreed.

## 3. The harness to build (show the user before building)

Model it on `mats-fleet`; the file names below are theirs. Code goes in `fleet/` inside `commute` (a new top-level
folder; add it to the Layout list in `.claude/CLAUDE.md`).

| Piece | From mats-fleet | Ours |
|---|---|---|
| Names and folders | `fleet.conf`, `ci/fleet.sh` | `fleet/fleet.conf`: `FLEET_PREFIX=commute`, `FLEET_AGENTS=/mnt/nw/home/e.agnew/commute-agents`, `FLEET_DATA=/mnt/nw/home/e.agnew/commute-data`, `FLEET_SECRETS=~/.config/commute/secrets.env`; `CLAUDE_CODE_TMPDIR=/ephemeral/$USER/claude-tmp` |
| Launcher | `ci/fleet-up` | `fleet/up [ROLE...] \| status \| stop`. One tmux window per role, one clone per role. It sources the secrets file into the window's command only. Loop: `while :; do systemd-run --user --scope -p MemoryMax=8G --slice=commute.slice -- claude --permission-mode auto -n commute-<role> "/loop <prompt>"; sleep 60; done`, with the slice capped at 16 G. The roster is `$FLEET_DATA/roster`, default `lead` |
| Iteration start | `ci/iteration-start` | `fleet/iteration-start <role>`: pull (merge, never rebase), restart request?, the role's open items, its live jobs and pods, the `docs/research_log/PENDING.md` length |
| Handoff | `ci/handoff`, `<data>/agents/<role>/handoff.md` | same path under `FLEET_DATA`. About one screen: doing now, tried, next, mistakes to avoid. Updated at the end of every iteration |
| Context restarts | `ci/check_context.sh` (cron, 5 min, 300k tokens) | same. **Plus our rule:** an agent may touch `restart-ready` only with no live pod-runner subagent and no unwatched pod. A restart kills background subagents, and pods with `cleanup = "terminate"` die with their session (2026-10-26 lesson). Slurm jobs survive |
| Stuck detection | `ci/check_blocked.sh` (cron, 5 min) | same: list windows at a permission prompt or question for > 10 min in `$FLEET_DATA/blocked.txt`. Raise a phone notification (PushNotification, or the morning debrief) |
| Stall detection | `ci/check_stalls.sh` | later, when there's more than one agent |
| Waiting | `ci/wait-job` | `fleet/wait-job JOBID...`, run in the background: it returns when the Slurm jobs end, so no `squeue` polling. Wake-ups at least 20 min apart while waiting, 60 min when idle |
| No compute on the dev node | `.claude/hooks/no-compute-on-node.py`, `ci/cpu-run` | Port both. Refuse `python`, `pytest`, `uv run` and `make test/lint` on the dev node. `fleet/cpu-run` sends them to `l40-worker` as a 1-CPU, 2 G CPU job (`-A c11 -p compute --qos=debug`). **Unlike theirs, allow GPU Slurm requests** (we use the L40s). The no-model-on-CPU check stays |
| Subagent models | `.claude/hooks/agent-model.py`, `.claude/agents/{runner,clerk}.md` | Copy them. Keep `pod-runner` on Opus as now. The fork ban matches our rule already |
| Git guards | `.githooks/pre-push`, `private-patterns.txt`, `gitleaks` | Copy the private-strings check (`~/.claude`, `secrets.env`, key prefixes), plus deny rules for force-push, `reset --hard` and `git clean` |
| Permissions | `.claude/settings.json` allow list | Allow `fleet/*`, `sbatch`, `squeue`, `scontrol`, `scancel`, git commit/pull/push, and the remote-kernels tools, including `terminate`, so the classifier can't strand a pod (2026-10-23). Deny `Read(~/.config/commute/**)`, `Bash(env*)`, `Bash(printenv*)` |
| Roles | `roles/*.md` | `fleet/roles/lead.md` (picks the next item from the approved decision tree, runs it through `/pod-runner`, writes the report, queues the log line). `fleet/roles/worker.md` comes later |

**Repo layout on the cluster:**
- Each role gets its own git clone under `commute-agents/<role>`.
- The big shared files go in one place, `commute-data/`: `artifacts/`, `results-large/` and the CUDA venv (shared
  per `uv.lock` hash, as `ci/fleet.sh` → `fleet_venv` does). Each clone symlinks to them.
- `cluster/env.sh` should find `COMMUTE_REPO` from the clone it is sourced in (`git rev-parse --show-toplevel`) instead of the fixed path. The old rsync copy then
  stays as the laptop's sync target until the user retires it.
- Everyone commits to `main` with merge pulls, as mats-fleet does. Laptop sessions already share the tree this way.

## 4. Memory in the repo

1. `git mv` the laptop's `~/.claude/projects/-home-edwin-work-mats-commute/memory/` to `.claude/memory/` in the
   repo. There are 53 files (228 KB) and no secrets (grepped 2026-10-26). Add `!.claude/memory/` to `.gitignore`,
   which ignores `.claude/*` by default.
2. On every machine and clone, replace `~/.claude/projects/<path-key>/memory` with a symlink to that clone's
   `.claude/memory`. The path key is the clone's absolute path with `/` and `.` turned into `-`, e.g.
   `-mnt-nw-home-e-agnew-commute-agents-lead`. `fleet/up` creates it for each role. On the laptop, do it once by hand.
3. A memory an agent writes then reaches the other agents and the laptop through git. Memory files are one fact per
   file, so merges rarely conflict; for `MEMORY.md`, keep both sides' lines.

## 5. How the agent works overnight

- **Scope:** only work inside an approved class or a decision tree the user has confirmed (`/design` → "Approved
  classes"). Anything else becomes a pending-decision card in the log, and the agent moves on to the next item.
- **Rigor:** follow mats-fleet's two tiers on top of our rules:
  - Exploration is labelled provisional.
  - A claim that goes into the log's sections above the cards needs filed predictions, the run, and an
    independent check: Codex `adversarial-review`, or a fresh Opus agent that didn't run the analysis.
- **Morning:** before the user's day, flush the research log (`docs/research_log/README.md`) and write a dated
  digest in `reports/`: what ran, what it found, the pending decisions, the money spent. Then send one notification.

## 6. Order of work, and the first night

1. Build section 3's pieces in the order: guards (hooks, settings), then `up` and `iteration-start`, then `handoff`,
   then the crons. `make lint` must pass on the new files. Run them only through `cpu-run`, never on the dev node.
2. Section 4 (memory), then `cluster/env.sh`.
3. The first night: `lead` only, on a decision tree the user confirms that evening. The `check_blocked` and
   `check_context` crons must be running before the user goes to bed.
4. After two clean nights, add `worker-1` with `fleet/roles/worker.md`.

## Not run

- Nothing has been installed on the cluster. Whether `/login` and the remote-kernels plugin work on the dev node is
  untested.
- The admins' view of extra agents on the dev node is unasked. A one-line note to `#support-mats-cluster` is
  polite, and it can also ask for the handbook's missing "Security & Best Practices" section.
- Whether `PushNotification` reaches the user's phone from a cluster session is untested.
