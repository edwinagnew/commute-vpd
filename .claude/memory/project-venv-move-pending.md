---
name: project-venv-move-pending
description: "REMIND THE USER when the cluster takes jobs again - rebuild the venv out of ~/commute, then they delete ~/commute (2026-10-26)"
metadata:
  node_type: memory
  type: project
  originSessionId: e3f26284-49c4-493e-a656-8e1561337d4c
  modified: 2026-10-26T19:23:52.798Z
---

The only commute venv is `~/commute/.venv` (7.7 GB), inside the retired laptop sync copy; `~/commute-data/venv` and
each clone's `.venv` are links to it. The user wants `~/commute` deleted, and asked (2026-10-26) to be reminded once
the cluster node `l40-worker` takes jobs again (it was draining from 2026-10-26 01:35 UTC, "Kill task failed";
`fleet/check` shows its state).

Then: build a fresh venv at `~/commute-data/venv-py313` with `UV_PROJECT_ENVIRONMENT=... sbatch cluster/setup.sbatch`
through a pod-runner; `ln -sfn venv-py313 ~/commute-data/venv`; once a job has run on it, tell the user to delete
`~/commute` (old venv plus PDFs already copied to `~/commute-data/lit_review_pdfs`). Also delete `cluster/sync.sh`
and the `~/commute` mention in `fleet/roles/lead.md`. Delete this memory when done.

**Status 2026-10-26 19:25 UTC:** build queued as Slurm 32273 (GPUs busy, est. start 20:48); 32287 swaps the link
atomically afterok. Next session: check the tail of `cluster/logs/commute-venv-swap-32287.out` (FINISHED, readlink →
venv-py313). If good: tell the user `rm -rf ~/commute` is safe (nothing unique in it, checked 2026-10-26), drop the
`~/commute` line in `fleet/roles/lead.md` and the `sync.sh` row in `cluster/README.md`, delete this memory.
`cluster/sync.sh` is already deleted (77fa552).

**Why:** the user finds the many venv folders confusing and wants one place; the reminder belongs in memory, not in
`fleet/check` (user, 2026-10-26).
**How to apply:** at the start of a session, if `sinfo -p compute` shows the node idle/mixed, raise this first.
