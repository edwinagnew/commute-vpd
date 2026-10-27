---
name: project-mats-cluster
description: MATS cluster access for feature_wars (available 2026-10-12): login, Slurm account c11, storage paths, verified limits and quirks
metadata:
  type: project
---

The MATS cluster became available to the user on 2026-10-12. Facts checked over SSH that day:

- Login: `ssh e.agnew@185.141.218.207` (key auth works non-interactively). This is a different IP from the
  handbook's and `../../CLAUDE.md`'s `94.156.8.208`; the user's address is the one that works.
- Slurm account `c11` (the default association, so `--account=c11` is optional but explicit is safer). Partition
  `compute` = one node `l40-worker`, 8× L40, 252 CPUs, ~458 GB RAM, shared by every fellow.
- QoS read from `scontrol show assoc_mgr` (sacctmgr fails: slurmdbd connection refused): `normal` max wall 24h,
  6 GPUs/user; `debug` max wall 2h, 1 GPU/job. Also exist: `conf_deadline`, `elastic` (not looked at).
- Storage: home `/mnt/nw/home/e.agnew` (private NFS), team `/mnt/nw/teams/team_sharkey_c11` (group-writable, has
  `theory_job/`, `weights/` from teammates). Scratch/caches `/ephemeral/e.agnew` on the worker; never `/tmp`.
- Controller: Python 3.10.12 system, no `uv` installed (project needs >=3.11, so uv must be installed and fetch
  Python), apptainer and git present. Controller is for submitting only.
- Load on 2026-10-12: all 8 GPUs allocated, 47 jobs pending, 400/458 GB RAM allocated. Queue waits can be long,
  so a short or interactive job may still be cheaper on RunPod ([[project-runpod-remote-kernels]]).

**Why:** the GPU guide sends single-GPU 1–24h batch work to the MATS cluster first; it is free, whereas RunPod
spends the $10/session budget.
**How to apply:** decisions (user, 2026-10-12):
- Code goes over by rsync to `/mnt/nw/home/e.agnew/feature_wars`; submit with sbatch; rsync `results/` back and
  commit locally. If rsync gets tedious, switch to a git remote (user's suggestion).
- (Superseded for commute on 2026-10-20 by [[feedback-results-fast-runpod-default]]: cluster only for runs ≤ 30 min.)
  Try the cluster first for GPU batch training; the user is sceptical given the queue, so report wait times honestly
  and fall back to RunPod when the queue blocks the work. Interactive GPU work stays on RunPod.
- Big CPU sweeps can also go on the cluster (CPU-only jobs, no `--gres`); small slices and analysis stay local.
- Record every job ID; follow `../../../guides/mats_cluster_handbook.md` etiquette. See [[feedback-big-runs-on-gpu]].

**commute set-up (2026-10-14).**
- **Location (since 2026-10-26):** one git clone per agent under `/mnt/nw/home/e.agnew/commute-agents/<role>`;
  `artifacts/`, `.venv`, `cluster/logs` and `results/large/` are symlinks (`~/commute-agents/fleet/link-data`) into
  `/mnt/nw/home/e.agnew/commute-data`. The old rsync copy `/mnt/nw/home/e.agnew/commute` is retired (it still holds
  the venv, which `commute-data/venv` links to until a rebuild). Submit sbatch from the clone root.
- **Jupyter from VS Code:** `cluster/jupyter.sh start|connect|stop`. Paste the printed localhost URL into VS Code's
  "Existing Jupyter Server".
- **Why a relay:** the controller cannot reach the node's ports (only slurmd's 6818). `connect` therefore relays
  through a CPU step of the job to the node's sshd.
- **GPU isolation:** jobs see all 8 GPUs in nvidia-smi; only `CUDA_VISIBLE_DEVICES` isolates them, so never
  override it. Don't do GPU work over Remote-SSH to the node.
- **Credentials (since 2026-10-26):** `~/.config/commute/secrets.env` (mode 600) holds RUNPOD_API_KEY and
  WANDB_API_KEY; `~/commute-agents/fleet/up` sources it into each agent's command only. Agents may not read the file.
- **RunPod pods fetch the VPD files from W&B themselves** (`vpd.fetch()`, 3.2 GB in 30–50 s on 2026-10-15, about a cent).
  Keep it that way: the laptop has no copy (`artifacts/vpd` is empty locally) and its uplink would be far slower.
  Only the small gitignored inputs (`transport_test/results/large/{alive,windows}.pt`) need copying from local;
  `sync-include` did not carry them, so copy via a temporary non-ignored folder.
