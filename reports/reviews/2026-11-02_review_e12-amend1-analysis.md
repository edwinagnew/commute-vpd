# Review: independent check of the E12 Amendment 1 analysis

Fresh Opus agent, 2026-11-02; it wrote neither the code nor the report. Target:
`reports/analyses/2026-11-02_analysis_e12-teacher-route-split.md` (line numbers as of 0d6b06a). Working files:
`~/commute-data/scratch/commute-lead/e12a1-check/`.

**Verdict.** The scored numbers and verdicts are right: P1 separates, P2 inherited, P3 not masked-off. The code measures
what the report says. Three interpretive claims overreach or are wrong, all at lines readers will quote.

## High

**H1. "The overshoot did not appear" is contradicted by the report's own table.** Lines 99 and 188. Lines 107-109 show
the six step-10 states ending at 1.23-1.35 times their teacher's size, which matches E43's overshoot (0.0141 against
0.0113, 1.25 times, `2026-10-28_analysis_e43-teacher-remainder.md`). The plan's 5.1× is the remainder's squared size,
which this run did not read (line 237). What was absent is a large move in re-switching.
- Line 99: "The students of step-10 teachers did overshoot in size, ending 1.23 to 1.35 times their teachers, as E43 saw. That did not move the re-switching term, which is what we had leaned on. E43's remainder size was not read here."
- Line 188: "An overshoot in size appeared on the step-10 teachers but did not move re-switching."

**H2. "A student's activations give nearly the same answer" about which subcomponents is not measured.** Line 174, line
170 ("one new fact"), notebook cell 33. Line 180 says the run read shares, not sets; `.students.*.a.max_dg` is 1.0 on all
eight states, so it says nothing; on the second map table the number differs by a fifth.
- Wording: "On six of eight, Γ's values on the student's activations give the released subcomponents about the same masked log-odds as on the fine-tune's. On the second map table they give 1.0 to 1.2 nats less, about a fifth of the term. Whether the same subcomponents switch was not read."

**H3. The summary's "therefore ... not how the update was placed" overreaches.** Line 7, notebook cell 33 ("so the map
that placed the update does not change it"), line 174 ("the map does not change the difference"). The re-switching term
reads released components and Γ on activations (`local_faith.py:266-267`), so placement reaches it only through the
weights; line 170 concedes this is close to arithmetic. The scored number did move with placement: masked-off rises
0.13-0.29 on every state, the gap goes from 0.331 to 0.476. Two placements were compared and no alternative cause of the
class difference was tested.
- Wording: "For `GOALS.md`, the re-switching share that divides the classes barely depends on whether the update sits in the components or in Δ, except for a fall of a fifth on one map table. The masked-off share takes up what Δ no longer holds. This run does not test faithfulness or why the classes differ."

## Medium

**M1. "Both seeds agree" is not evidence about batch order.** Lines 122, 183. All eight students train at batch-order 0
(`amend1.py:259`, generator seeded at `component_training.py:144`), so s0 and s1 students on one table draw identical
batches and differ only in their teacher.
- Line 122: "Batch order: unmeasured on this table. Both seeds' students used the same batch order, so their agreement bears only on the teacher's seed. Noise on the other two tables is 0.0006."
- Line 183 should say the s0/s1 comparison cannot separate shared batches from teacher seed.

**M2. "The numbers rule out four explanations" is inconsistent.** Line 117: the fourth is "unlikely but unmeasured"; say
"rule out three and leave one unmeasured". Line 121's "The student's components. Not it" is correct only for the
read-out; the student's components produce the weights that produce those activations. Header: "Which components are
read."

**M3. The reach figure's axis is mostly teacher size, and its title answers more than the text allows.** Lines 103-124;
`plot_amend1.py:174`. After 1,000 steps student size stays in 0.0170-0.0262 while teacher size spans 0.0137-0.0379. Ratio
under 1 coincides exactly with teacher step ≥ 30. C1's size also grows by a third with nothing to learn (0.0182 to
0.0242), so the numerator includes drift. "The only ones that end smaller" is close to "the only teachers larger than a
1,000-step student reaches". Add one sentence saying this. Retitle: "Share of re-switching lost against student/teacher
size: a correlation on four map settings".

**M4. The proposed read uses lossy files without saying so.** Lines 197, 236. Saved students are bf16 offsets
(`arms.py:52`); `amend1.py:3` and the filing's `measurements.state` say students are compared only from fp32 weights. The
follow-up should first reproduce the 1.02 and 1.22 nat falls from reloaded offsets, or retrain in-job at about 105 s per
student. The rounding's size was not measured.

**M5. "Optional ... if the fall matters to the user" sits badly with two standing user rules** (run down anomalies first;
don't ask for cheap recommended checks). Line 197. The lead's call: see "What the lead did".

## Low

- Line 11, "Only one number told the classes apart". E12's report parts the classes on two route numbers and on switched-on counts (107-139 against 21-51), layers and seed overlap (2026-10-20 analysis, lines 158-162, 221). Wording: "E12's filed route numbers parted the classes; no number about the weight update did."
- Line 99, "The step-1000 teachers from E42 sit near that upper group". True of the maps (0.053, 0.063); E42's copy student reads 0.027. Say "E42's map students".
- Line 145 and notebook cell 33, "within 0.008". The largest difference is 0.0062 (rand32 s1). Say 0.007.
- Line 145, "as E12 and E46 would lead you to expect at step 1000". Neither run has a student or a step 1000. Cite E42 or cut it.
- Line 158, C1. "Loss 0.0078 at step 1 to 0.0266 at step 1000" hides the curve: it peaks at 0.179 at step 10, then falls about sevenfold (`.c1.curve`). Give all three values.
- Line 181, "Its six-state agreement is partly agreement with a half-learned teacher". Only two of the six are half-learned. Say "the first map table's agreement is with a half-learned teacher".
- Line 223, "within 0.071". The value is 0.0711.

## Checked and found sound

Provenance (results sha256, commit 48cf644, log's `[score]` identical to `.score`, 44 checks, 12 offset files). Sections
1-5 recomputed from raw cells (`recheck.py`), all shares, gaps, ranges, verdict tables and the nats claims. `amend1.py` and
`route_split` implement the report's cells, shares, states (a) and (b) and the pool; P1-P3 match the filing; identities
read exactly 0. Student and teacher sizes share one definition; hard-coded numbers in `plot_amend1.py:30-31` match
`regating_table/results/large/full.log`. Section 4's inference holds: the fall lies in Γ's values, as L11 differs only
through the gates. Figures match the data apart from M3's title. Notebook cells 22-33 are consistent with the report and
carry only the H2, H3 and 0.008 fixes.

**Not checked.** Whether E42's windows match these windows; E50 Amendment 1's claim beyond its `TRIED.md` row.

## What the lead did

All findings applied. M5: the fall is run down as E12 Amendment 2, a free cluster job (one bullet in the report points to
it) rather than left as an option for the user; the filed clause's "nothing further runs" is departed from for this
anomaly only, and the departure goes to the user as an FYI.
