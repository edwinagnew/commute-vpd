# Commute-lead digest, 2026-11-02

**Finished because everything left waits on you: is the commute line's weak version on the 67M settled?**
E12 is done, and your (b) went to baseline-lead, whose card it was. We recommend one free read weighted by real inputs,
then closing the line on this model with the wording that read gives.

**Pending decisions.**
- `decision-commute-settled` (ask 12): the question above. Recommendation (a), the free read first. Fable's advice: `reports/plans/2026-11-02_plan_commute-next-draft.md`.
- `decision-chain-size` (ask 9, no rush): stop reading how far subcomponents moved as what repeated teacher training taught. Recommendation (a), wording only.
- `decision-e47-split-keep` (no rush): how to word the scaled adjoint split's loss of behaviour. Recommendation (a), reword.

**What ran, $0, no pods.**
- **E12 Amendment 3, job 49192.** We read every saved student and fine-tune in fp32 as well as bf16. No fall moved by more than 0.0064 nats, so the fall on the second map table does not come from read precision. Its cause stays open, and bf16 is the authors' own setting, so fp32 is a test of dependence and not a truer value. Report: `reports/analyses/2026-11-02_analysis_e12-amend3-fp32-read.md`.
- **E34 Amendment 9, CPU.** We compared each student's weight change with the part of its fine-tune's update the subcomponents could write. On the real task about an eighth of the squared change lies along that part. A fine-tune that teaches no task reads about a fifth, inside the range of tasks that were learned, so this cosine does not measure teaching. All weight directions count alike here, so the read cannot change the headline wording. Report: `reports/analyses/2026-11-02_analysis_e34-amend9-student-move.md`.

**Checks.** The independent check of E12 Amendment 2 kept its numbers and corrected the reading; all findings are applied. The check of E34 Amendment 9 recomputed every read with its own code and found no bug, but three readings overreached; those are corrected too. E12 Amendment 3 had a code review and no separate check of its write-up, because a flat control adds no claim to the log's main sections. E34 Amendment 9 had no review before its run, only the check after it.

**Open oddity.** In E34 Amendment 9, one matrix, `h.3.attn.k_proj`, reads a slightly negative cosine on the real students and +0.56 on the no-task one. It holds 1.2% of the student's squared size and was not run down.

**Log.** v190 is live with the new card. Queued for the next flush: three wording fixes on E34's lines. A trim made for the page's length rule cut detail from three claims rows; the numbers remain on their experiment cards.

**Friction noted.** `experiments/SAVED.md` is stale; two experiments share the name `amend2.py`; the dev node has no `node`, so the dataviz palette validator cannot run; the log's length rule cost about an hour of trimming for one card.
