# Cluster worker won't launch jobs: P1.65 and Fig 7 are stuck overnight

**Question for you:** l40-worker hands out job slots but won't start anything in them. Should an admin restart it, or may we treat this the way we treat "cluster down" and go to a pod without the pre-run figure sketch?

**What happened (2026-11-26, from ~20:30).** l40-worker is the cluster's only compute node. Slurm still gives our jobs a slot on it, but each one then fails to start with "Socket timed out on send/recv". This happened three times: jobs 69486 (~21:00), 69495 (21:15) and 69501 (21:35). The node reads "mixed + completing", and jobs that started earlier keep running. Slurm's job-history database is also refusing connections (evidence file).

**Why a pod doesn't get around it.** A pod needs an experiment number. Getting one requires the pre-run figure sketch, which is drawn with a small Python job on the cluster. The harness lets the sketch wait until the run only while the cluster is *down*, and this node doesn't read as down. Nothing can run Python on the dev node. So no sketch means no number, and no number means no pod. I didn't change the rule because another lead asked me to.

**What it blocks.**
- definition-lead, P1.65 runs 1–2 (due 11-27). This is the one GPU-dependent K number, main.tex:432. It's about 2 h on an H100, ~$6, in the plan and already approved. The other two K re-reads are done and with writeup-lead.
- writeup-lead, the Fig 7 redraw. This is a few seconds of plotting. writeup-lead's text push is waiting on it, so the caption won't describe 7 panels while the image shows 5.

**Options.**
1. **Ask the cluster admin to restart slurmd on l40-worker (recommended).** This fixes both items, and nothing else changes. Both finish within ~2 h of the restart.
2. **Change the rule so "sketch: pending" passes when a step launch times out, as well as when the node is down.** P1.65 goes to a pod at once and draws its sketch as the run's first step. Fig 7 still needs the worker, or a pod stage-0 rider under P1.65's number.
3. **Wait.** If the node clears on its own, nothing needs doing, but P1.65 likely misses 11-27.

Evidence: ~/commute-data/agents/whip-lead/evidence-slurmd-2026-11-26.txt (sinfo, node state, sacct refusal); the job IDs above, as definition-lead reported them.
