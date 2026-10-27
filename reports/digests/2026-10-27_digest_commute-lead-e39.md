# Digest: commute-lead, 2026-10-27 afternoon (E39)

**Finished because the task is done.** You asked why teacher training's decomposition of the broad lowercase
fine-tune fails the adversarial check, using quick cheap tests, and whether SGD or LoRA updates do the same. E39 and
its Amendment 1 answer both as far as cheap tests go. What comes next is your call (log card, below).

**What we found.** A random move of the subcomponents of the same weight size passes the check, so the failure is not
simply any move of that size, though the random move changed predictions far less. On the Adam fine-tune the extra
fragility sits in the MLP matrices' changes taken together: putting the original MLP subcomponents back removes it,
putting back the attention ones does not. Putting back the half Γ rates rarely important removes the least, so Γ's
importance ranking does not explain it. On your question: not in general. The rank-8 LoRA's trained decomposition
passes (1.27 against a pass line of 1.40), but SGD's, nearly as concentrated (stable rank 3.5 against the LoRA's 1.9
and Adam's 28.5), fails like Adam's (2.07 against 2.20). So how spread out the update is does not decide it; why the
LoRA passes is untested.

Detail: log card E39; `reports/analyses/2026-10-27_analysis_broad-fragility.md` (independently checked,
`reports/reviews/2026-10-27_review_broad-fragility-analysis.md`); notebook `notebooks/E39_broad_fragility.ipynb`.

**Pending decision for you.** The log's new "Soon" card `decision-lora-passes` (log v65; `decision-broad-fragile`, which
you decided): whether to look into why the LoRA passes, or accept that broad updates are messier and move on (e.g. to
the larger model). No run is proposed.

**Money.** Two RunPod H100 pods, $3.62 this session of $40 (E39 ~$1.9 plus a failed first start, Amendment 1 $0.91).
No pods or Slurm jobs of this line are running.

**Incidents.** The plugin lost SSH to the E39 pod right after the job finished; you copied the final JSON and log by
hand. A git pull put another agent's uncommitted log edits into conflict; reset to the published v62 with your OK,
the edits kept in `stash@{0}`. Both are recorded as friction notes. Nothing in `~/commute-data/blocked.txt` from
this session.
