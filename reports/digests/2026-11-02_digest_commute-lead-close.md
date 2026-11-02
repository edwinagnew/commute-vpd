# Commute-lead digest, 2026-11-02, closing the 67M

**Finished because the task is done: the commute line closes on the 67M, on your option (a).** The decomposition after teacher
training passes VPD's checks, and on real inputs its weight change sits further from the fine-tune's than a rerun of that fine-tune does. So "copies
the fine-tune's change on real inputs" is not available. An independent check confirmed this by re-running the filed rules on the raw reads.

**The wording, per family.** Scope: pooled over the 24 matrices, at each matrix's output on W₀'s inputs, every position of held-out
windows and of Pile, not about the trigger. The behaviour half is cited from earlier runs, not read here.
- **Real task:** reproduces its behaviour by a change further from the fine-tune's than a rerun is: cosine 0.38 to 0.40, where reruns read 0.83 to 0.84.
- **`rand32`:** reproduces its behaviour by a change further than a 10-step rerun is.
- **`rand32b`:** further than a 30-step rerun, keeps its gain, loses about a fifth of the re-switching term; "reproduces its behaviour" is not available.

Report: `reports/analyses/2026-11-02_analysis_e34-amend10-input-weighted.md`. Log: E34 card, and the new row in the claims table.

**Limits worth knowing.**
- Input weighting moved the reading only part of the way: at each family's midpoint it narrowed the gap to the reruns from 0.52 to 0.43 on the real task, and no student reached the line.
- A filed control stopped the first job (49463) before any input-weighted read of a student. Its bound on the unweighted cosine assumed 3/8 of the update on matrices with a near-empty input direction; about 26% was measured. No verdict reads it, so it was amended with a date to be written, not a stop.
- Unfiled and provisional: per matrix, 1 to 4 matrices on the map tables read above their rerun line (review `reports/reviews/2026-11-02_review_e34-amend10-analysis.md`).
- No cause of the gap was measured, and whether the difference matters downstream was not read. The strong version, a gated re-decomposition of one fine-tuned 67M, is not run.

**Pending decisions.**
- `decision-chain-size` (ask 9, no rush): stop reading how far subcomponents moved as what repeated teacher training taught. Recommendation (a), wording only.
- `decision-e47-split-keep` (no rush): how to word the scaled adjoint split's loss of behaviour. Recommendation (a), reword.

**Money and runs.** $0, no pods: jobs 49463 and 49467 on the cluster's `debug` queue. Log published as nav v193 (changelog row 193), queue empty; the flush also carried regating-lead's E62 Amendment 6 lines. Nothing in `blocked.txt` is mine.

**Friction.** `tests/test_teacher_amend10.py` fails at collection without a `PYTHONPATH` that nothing states (383), hit twice. `amend10.py` grew to 640 lines on review fixes, against an aim of ~300 (381). The log's `--check` needs pages that the publish recipe doesn't read (flush clerk, not yet noted before today).
