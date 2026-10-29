# Review: does the E47 analysis score the five claims correctly from the raw results?

*2026-10-29. Independent check by an agent that wrote neither the code nor the analysis. Read-only except this file.
Checked: `reports/analyses/2026-10-29_analysis_e47-claim-tests.md`, `notebooks/E47_claim_tests.ipynb` (markdown
cells), against `reports/plans/2026-10-28_plan_claim-tests.md` (incl. Amendment 1),
`experiments/claim_tests/results/predictions.json`, `results/{t1,t1b,t2,t2b,t2c,t3,t4,t4b,t5,t5b,combined}.json`,
`results/large/*.log`, `cluster/logs/commute-e47-*.out`, and the code `experiments/claim_tests/{rules,stages,job,recipes,run}.py`,
`src/commute/{faith_score,faithfulness,finetune}.py`. Every number below was recomputed by hand from the JSONs (jq).*

Every verdict in the report follows from the raw results under the rules filed before the run, and the scoring code
matches the filed text. The main problems are in the prose: the step-10 result for claim (iv) is described more
strongly than its three readings support, and the analysis says "every control matched its expected value" when the
claim (i) control read about a quarter of its expected value. Neither changes a verdict.

## Filing and code

- `predictions.json` was last changed in 368feb5 (10:32:48). The first job (t3) wrote results at 10:35, so the rules
  were filed before any result existed. No result commit touches the file.
- `rules.py` matches the filed text for T1, T2, T2 combined, T3, T4 and T5, including how ties, missing starts and
  early stops are handled. One small difference, which matters nowhere here: in T1, a passing copy beside an
  *unscored* do-nothing is scored "unsettled" in the code. The filed text could also be read as "no power".
  `faith_score.verdict`: pass ≤ 1.5/(1+f), fail ≥ 1.5(1+f), as filed. `run.py combine` pools f over t1's h0 and h1 and
  t1b's h2, giving 0.2088, as filed.
- t4 and t4b run the same code; only k (1000 or 10) and the reference values change (`stages.t4`, `T4_REF`). The
  fine-tune has no learning-rate schedule. Batches are drawn per step from a generator seeded with 0
  (`finetune_arm_a`), so a 10-step run gives the same step-10 update as a 1000-step run, up to the card's numerics.
  Teacher training uses the same recipe at both steps (Adam 1e-5, batch 32, order 0, 1000 steps).

## Per claim

| claim | report's verdict | recomputed | status |
|---|---|---|---|
| (i) T1 combined | stands, both teachers | stands, both teachers | **confirmed** |
| (ii) T2 combined | absolute unsettled, cap unsettled | the same | **confirmed** |
| (iii) T3 | mixed on both seeds | mixed on both seeds | **confirmed** |
| (iv) T4 (step 1000; the claim as filed) | unsettled | unsettled | **confirmed** |
| (iv) T4b (step 10) | falls | falls | **confirmed by the filed rule; prose overclaims** |
| (v) T5 combined | falls | falls | **confirmed; small overclaim** |

**(i).** The lines are 1.2409 and 1.8132. The r1 copy's worst reading is 1.2249 (h1), a pass, 1.3% under the line.
The rand32 copy's worst is 1.0117 (h2), a pass. The r1 do-nothing reads 2.1615 at the budget start (an early-stop fail)
and 3.5299 at h2. The rand32 do-nothing reads 3.3963 at h1, a fail. Separation: 1.2249 × 1.2088 = 1.481 < 3.530, and
1.0117 × 1.2088 = 1.223 < 3.396. The verdict is stands for both teachers, and t1 alone already gives it. The gates pass:
the r1 teacher's P is 0.9909 and its size 0.02861 (filed 0.9834 and 0.02875), and rand32's P at step 10 is 0.5847.
Every table value matches to 3 decimals.

**(ii).** On rows 0-255, rank 8's mean is 1.2762, which is inconclusive. Rank 32 reads 1.2640 and rank 128 1.3673,
both under the band of 1.2762 × 1.2088 = 1.5427, so no cap effect. The control reads 2.1711, a fail.

On rows 256-511 (t2c), f = 0.65199/0.63011 − 1 = 0.03473, so the lines are 1.4497 and 1.5521. Rank 8's mean is 1.1536,
a pass, so the absolute reading there is "stands". Rank 128's mean is 1.5324. That is under the fail line and over the
band of 1.1937, so the cap reading is unsettled. The control reads 1.8974, a fail. Combined: the absolute reading is
(unsettled, stands) → unsettled, and the cap reading is (no cap, no cap, unsettled) → unsettled.

*Does the two-start floor decide anything?* No. A third start can only raise f, since f is max/min − 1. Rank 8 stays a
pass for any f up to 1.5/1.1536 − 1 = 0.300, which is above the 0.25 point where C0 stops the job. Rank 128 stays
"unsettled" for any f from 0.022 to 0.328. It would turn into a cap artefact only if f were *smaller* than 0.022, which
more starts cannot do. So both of t2c's verdicts hold for every allowed f, including the review's expected 0.15. The
report's caveat is fair, but it could say this outright.

Smaller checks: rank 8 reads identically in t2 and t2b. t2 (job 42719) loaded the rank-128 offsets that t2c trained
(`high_rank.source`). Rank 128 has gain 0.1666 and size 0.0766, rank 32 has 0.1648 and 0.0445, as reported.

**(iii).** The differences at matched size are +0.0791, +0.0880 (seed 0) and +0.0886, +0.0804 (seed 1). All four lie
in (0.05, 0.10], so part (a) neither stands nor falls. Part (b) is unscored, since its denominator P is 0.060/0.259
(seed 0) and 0.060/0.239 (seed 1), under 0.5. So each seed is mixed, and the stage is mixed. The control reproduces
E31's keep within 0.0036 and its size within 0.19%. *Overclaim:* "size moves keep by ~0.3, path by ~0.08" (table, (iii)
prose). The drop of about 0.3 is measured *along* a run, where size and step count rise together. The arm that holds
the path fixed and changes size alone, (b), could not be scored. The notebook's "Size still does most of the work" has
the same problem. Better wording: "at matched size the path adds 0.08-0.09. Size alone was not isolated."

**(iv), step 1000 (T4).** q = 1.2674, 1.0048, 1.1064. None is ≤ 1.0, so it does not fall. It does not stand either:
the lowest plus-remainder ratio (1.762) is under the highest do-nothing ratio (4.706), and q at h1 is under 1.1. So the
outcome is unsettled. The gates pass: G1 P 0.99910 and size 0.03149 (−1.9%), G3 P 0.99685 with Δ share 8.7e-10, G2
2.9e-5.

**(iv), step 10 (T4b), the reversal.** q = 0.9877/1.5102 = 0.6540, 1.3454/1.4000 = 0.9610, and
1.0089/0.9387 = 1.0748. q ≤ 1.0 at h0 and h1, so it **falls by the filed rule**. The verdict is real, and the gates
passed as filed: G1 P 0.99928 against 0.99931 and size 0.011191 against 0.01125, which is −0.53% (the report says
"within 0.5%", a rounding slip). G3 P is 0.99939 against 0.99941, with Δ share 1.5e-9. G2 is 4.1e-5.

Three caveats the report does not state:

1. **The gates cannot detect the thing that changed.** The P values are all ≥ 0.999, so the 0.02 P tolerance tells
   nothing apart. E43's own G3 also checked the teacher-trained state's *adversarial ratio* (E41's 1.547, within 5%).
   E47 never reads the step-10 teacher-trained state at all. So "every reproduction check passed" pins down the
   update's size and behaviour, but not how the state reads under the attacker.
2. **One state moved far more between cards than the others.** At h0, the L40 against the H100: do-nothing at step
   1000 reads 1.39 against 1.50 (−7%), plus remainder at step 1000 1.76 against 1.87 (−6%), and do-nothing at step 10
   2.18 against 2.52 (−13%). Plus remainder at step 10 reads 1.43 against 3.92 (−64%). It is also below E41's H100
   reading of the step-10 teacher-trained state *alone* (1.55). That points at this one state, or at how the attacker
   responds to it on this card. It is also a cheap first check for "What next" 1: read the step-10 teacher-trained state
   (no remainder) at h0-h2 on the L40.
3. **The attacker is not stable across cards or starts.** r1's do-nothing reads 9.24 at the budget start on the H100 and
   2.16 on the L40. The step-1000 do-nothing reads 1.39 at h0 and 4.65 at h1. The "falls" rule fires on a single start
   at q ≤ 1.0, and here it is carried by one start (q 0.65). The other two starts have q of 0.96 and 1.07.

*Overclaims:* "putting the remainder in the residual read better than leaving the whole update there, the reverse of
the earlier finding" (the answer section, and the notebook's), and "falsified at step 10" (table). Accurate wording:
"at step 10, plus remainder was not worse than do-nothing. It read better at one of three starts (q 0.65) and within
4-7% at the other two. That breaks the filed 'worse at every start'; it does not show the opposite ordering."
Two more points:

- The filed `amendment_1.combined.T4` says "claim (iv) as filed is T4's", which is the step-1000 test. Claim (iv)'s
  verdict is therefore unsettled, with the step-10 result reported beside it. The scorecard handles this correctly;
  the headline table's "falsified" does not.
- Surprise 5 says t4b's first submission (42668) "hung at start with no log line". Its cluster log and `t4b.log` show
  it logged "bases and input moments for 24 matrices in 1723 s" at 11:37:32 and was cancelled one second later. It was
  slow, not silent. t2's first submission (42655) did log nothing.

**(v).** All three filed cells pass on both teachers. The Adam teacher reads 1.0265, 0.9783 and 1.0332, and the LoRA
teacher 1.1718, 1.0428 and 1.0457. Each is the worse of the budget start and h0, and all are under 1.2409. Every P is
≥ 0.9864. So "falls" holds with n_read = 3 per teacher.

**The positive control is a genuine fail at the same rows.** It is the Adam teacher at rate 0.03, step 1000, read at
h0, which is one of the cells' two starts, on rows 0-255. It reads 3.300, 1.8× the fail line, and t5b reproduces it
to every digit. Its budget start was not read, as filed. It is also the continuation of the same SGD run whose step-100
state passes (1.027), which makes it a clean within-run contrast. Two limits:

- The control's P was not recorded (`p: null`), so whether the failing state still keeps the behaviour is unknown.
- It exists for the Adam teacher only. *Small overclaim:* "plain SGD at rate 0.03 for 1000 steps fails on these
  teachers" (report (v)). In E47 that is shown on the Adam teacher only. The LoRA teacher's 2.86 is the earlier H100
  reading.

## Other prose

- "Every W₀ and positive control matched its expected value" ("How compelling"). Every control passed its *rule*, but
  T1's control was filed as "budget ≈ 9 (H100 9.24)" and read 2.16, a quarter of that. T1b's r1 do-nothing at h2 was
  expected at "about 1.5-1.8, inconclusive alone" and read 3.53. Neither is mentioned, and both are evidence for caveat 3
  under (iv).
- Every other number in the report's tables, scorecard, C0 line and provenance (job IDs, seconds, GPU peaks
  7.5-28.1 GB) matches the JSONs. The notebook's markdown agrees with the report and carries the same overclaims for
  (iii) and (iv).

## Not run

- No code was executed. `tests/test_claim_tests.py` was not run, and t3's cell scoring (`stages.t3`) was checked only
  through its output.
- The notebook's code cells and figures were not checked against the JSONs, only its markdown.
- E43's own JSON (`teacher_training_e43.json`) was not opened. Its H100 values were taken from the filed expectations
  in `predictions.json` and from `remainder.py`'s constants.
- Whether E43's `harness.measure` h0 attacker is the same code path as `multi_start`'s "h0" was not traced. W₀ agrees
  (0.6917 on the L40 against 0.6988 on the H100), which suggests it is.
