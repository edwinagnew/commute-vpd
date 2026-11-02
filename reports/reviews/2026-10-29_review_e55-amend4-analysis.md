# Review: E55 Amendment 4 analysis, independent check (2026-10-29)

A fresh Opus agent that neither ran nor wrote the analysis checked the Amendment 4 section of
`reports/analyses/2026-10-29_analysis_e55-stage-a.md` (commit 1d8f32a) against `amend4.json`, the per-pair `.npy`
files and its own model-free reruns (`~/commute-data/scratch/reuse-lead/e55-amend4-check/`). Verdict: **stands with
corrections**. All ten corrections were applied at 58c58d5.

Verified: every number in the tables, bars and scorecard; the answer breakdown for all six states; that the pre-run
amendment (d62c43f, 23:45:45) precedes the job's first log line (23:46:18) and the results commit (3b4f962), with no
later change to the code or predictions; the three verdicts (P1 met, P1s met, P2 missed) under the rules as amended.

What holds: a separate count of b is ruled out cleanly. Both seeds of W1 1K answer `count(b)` on 0.001 of off-label
rows, against 0.906 for the count-b control N1 1K.

Corrections, in short:
1. The opening said "the two early states that reuse it"; M3 1K reads neither. Reworded.
2. The count-a patch share is not specific: LOW reads 0.575 and W1 10K 0.499 on the same pairs, against the seeds'
   0.505 and 0.496. A strengthened count rests on the behaviour share alone (0.67 against LOW's 0.35).
3. The count-b patch share has a floor of 0.404 for LOW, so the seeds' 0.19 says nothing about a pull from b.
4. The "pull from b" was overstated. In most rows the step towards `count(b)` is also the step back towards the
   pre-swap text, and only 138 rows separate the two (13% towards b, none back). With a + b + c = 40 it is equally a
   dependence on count c, and the breakdown was not filed.
5. Two rows (0.2%) sit two steps towards b; "nothing else appears" was too strong.
6. The pre-run rule change also saved the count-b control: N1 1K's off-label count-a patch share is 0.123, over the
   lower bar of 0.097, so under the drafted rule Part 2 would have been void.
7. The figure title said "none runs through count b", which is false for the control.
8. The notebook table lacked LOW's row.
9. LOW's B_a of 0.266 comes from removals of two letters (77% of 252 rows) and of one letter (31% of 234).
10. The one-third rule coincides with `count(a) + 1` on 16.8% of rows, so W1 10K's 0.187 share right is almost all
    one-third answers.
