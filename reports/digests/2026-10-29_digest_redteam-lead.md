# Digest: redteam-lead, 2026-10-29 (E47 claim tests)

Finished because the task is done: the five red-teamed claims were re-tested on the cluster for $0, analysed, checked
independently, and put in the log; what is left are four decisions only you can make.

## What ran
E47 (`experiments/claim_tests/`), eleven L40 debug jobs on the MATS cluster, $0 (no pods). The pieces first cut to fit
30-minute jobs ran as extra jobs, as you chose. A follow-up, t4c, read the step-10 teacher-trained state on its own after
the independent check found the step-10 reading of claim (iv) rested on one adversary start. Report
`reports/analyses/2026-10-29_analysis_e47-claim-tests.md`, notebook `notebooks/E47_claim_tests.ipynb`, independent check
`reports/reviews/2026-10-29_review_e47-analysis-check.md` (all verdicts confirmed; prose overclaims fixed).

## What it found
| Claim | Red team | After E47 |
|---|---|---|
| (i) teacher training keeps behaviour and passes the adversarial check | weakened | **stands**, both teachers, four starts |
| (ii) rank-8 LoRA pass is real, not a rank cap | weakened | **weakened**: passes on rows 256-511, inconclusive on rows 0-255; rank 128 climbs on the second row set |
| (iii) splits pass only while the update is small | weakened | **weakened**: mixed on both seeds; size alone untested |
| (iv) teacher training plus remainder is worse than leaving the update in Δ | weakened | **unsettled** at steps 10 and 1000; at step 10 the remainder's effect flips sign across starts |
| (v) teacher training needs Adam | weakened | **falsified**: plain SGD passes in all six cells |

Filed-lean misses: T3 (lean falls, read mixed), T2c cap (lean no cap effect, read unsettled), T1 (rand32 had power the
lean said it lacked). The step-10 reversal I notified you about did not survive t4c.

## Decisions for you (log cards, drafted in the report's "Pending decisions" section)
- (ii) count the rank-8 pass now, or first run ranks 32 and 64 on rows 256-511 ($0, one job; recommended)?
- (iii) reword the split claim to "keeps less as training goes on; at matched size the path adds about 0.08"?
- (iv) drop the remainder claim from the log (recommended)?
- (v) rescope "needs Adam" in `decisions.md` and the E34 card to the one setting that fails (Soon)?

## Money and problems
$0 spent; no pods. Two of eleven jobs hung at start with no log line (42655, 42668) and ran clean on resubmit, cause
unknown. Ask 2 (the cuts) sat open about two hours while you were around; I now ask blocking questions in a dialog.
Nothing in `~/commute-data/blocked.txt` is mine.

## Friction noted (redteam-lead, 2026-10-29)
114 blocking ask sat unseen; 115 the publisher forces a full read of the live log in every flush (~70k tokens); 116 clerk
briefs name no log-lock role; 120 frozen `run.py` recipes cannot be imported by a new experiment; 122 a subagent ran
Python on the dev node once; 123 cpu-run refuses /ephemeral worktrees; 130/132 job hangs with no stall alarm in
`wait-job`, and `sstat` is missing in jobs.
