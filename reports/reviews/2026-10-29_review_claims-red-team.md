# Red team of the commute line's five load-bearing claims (2026-10-28 evening, dated for the morning read)

**The answer.** Five reviewers, one per claim, each tried to break it with numbers already in the repo. None of the
five claims survived intact, and none was overturned outright. Every one rests on a single adversary start, one model
and usually one task or seed, and its pass margin sits within the adversary's own seed-to-seed spread. Two claims say more
than their data: "teacher training needs Adam" ignores passing plain-SGD copies in E34 itself, and "splits fail when
large" was already shown not to be about size for the adversarial check (E22). Each claim has a $0 cluster test that
would settle it.

What was done: five independent Opus agents (GPT-6.1 Sol was unavailable: model-router is unwired until the Codex quota
resets ~2026-11-18), each given the claim's analysis, notebook, `TRIED.md` rows and check reports, and told to attack
support (seeds × tasks), the adversary's noise floor, missing controls and wording. Their full reviews are summarised
below; the two attacks that decide a verdict (claims iii and v) were re-checked by the lead against the JSON. The regating
line (Q5) was out of scope. "Pass line" below is the harness's ratio to the unchanged model's adversarial KL: pass
under 1.40, fail over 1.61 on the H100 (card-dependent: 1.24 on the L40).

## The table

| Claim | Support (seeds × tasks) | What breaks it | Verdict | Cheapest test |
|---|---|---|---|---|
| (i) Teacher training keeps the behaviour and passes the adversarial check with Δ fixed (E34, E41, E42) | E34: 3 tasks × 2 teacher optimisers × 2 text arms × 2 batch orders; LoRA teacher 3 tasks × 1 order; E41 2 tasks × 3 checkpoints; E42 3 ciphers × 1 order. Fine-tune seed 0 only, 67M only, one 20-step adversary start per reading | The harness never failed a do-nothing state on E34's six teachers (1.12-1.59) or E42's three ciphers (1.46-1.56), so the check had no shown power there; the filed positive control (r0 Adam do-nothing fails) passed at 1.35. Where do-nothing does fail (E41 step 10, r2), the copies read 1.28-1.55: 3 of 6 pass, all within the 7.5% adversary-seed floor of the line. The clear separation (copies 0.95-1.15 vs do-nothing 1.68-10.45) is Amendment 4's one budget-start reading, where W₀ alone at seed 1 reads 1.231 against a 1.241 line. Behaviour is kept (P 0.92-1.00) for an Adam student on task text; text-only copies lose 2 of 3 ciphers | **weakened** | Cluster debug job, ≤30 min, $0: read the saved E42 task-text copies and the Amendment 5 r1 copies beside W₀ and regenerated do-nothing states under 4 adversary starts (seeds 0-2 + the budget start), max over starts. Stands if every copy ≤ 1.40 × W₀'s max and every do-nothing ≥ 1.61; falls if any copy ≥ 1.61 |
| (ii) The rank-8 LoRA's pass is real and not a rank cap (E37, E39) | 2 LoRA seeds (batch order changed with seed) × 1 task × 2 row sets; adversary seed 0 only | Pass repeats (1.27, 1.27, 1.21, 1.24) but margin to the line is only ×1.10 under one start. "Not a rank cap" rests on one plain-SGD arm held to rank 8 (2.12): it shows the cap is not sufficient under SGD, not that the Adam LoRA's pass does not need it; no LoRA at another rank, no Adam update truncated to rank 8. The SGD-vs-LoRA contrast lives in one 128-row block (SGD 2.93, LoRA 1.42-1.44 there). LoRA's do-nothing already starts lower (2.65 vs Adam's 4.87); teacher-trained / do-nothing is 0.48 vs 0.45, so the Adam contrast also fits "teacher training halves it, the LoRA starts lower" | **weakened** | Cluster debug job, ~25 min, $0: Adam LoRAs at rank 8, 32, 128 (1e-4, 300 steps), teacher-train each, read at adversary seeds 0-2 on both row sets; re-read the five saved states in `experiments/broad_fragility/results/large/` at seeds 1-2 and the second rows. Cap artefact if the check climbs past 1.61 with rank; not real if a LoRA's mean over starts exceeds 1.40 |
| (iii) Splits pass only while the update is small (E16, E22, E31) | E31: 3 tasks × SGD at 2 rates × 2 batch orders; E16, E22, Amendment 3: emoticon, one seed. No δW saved | Two claims under one name. *Adversarial check:* contradicted in the repo; E22's SGD update at size 0.0122 passes (0.98, least-squares 1.01) while Adam at 0.0113 fails (2.52, 6.96), and E22 already concluded "size alone does not decide" (`reports/analyses/2026-10-22_analysis_small-finetune-splits.md:21`); failures cluster at step 10, the smallest size. *Keep (scaled adjoint split's P(target)):* falls along SGD training within each task, but size is confounded with steps and rate; at near-matched size the run with fewer steps keeps more (r1 0.25 vs 0.13); r1 never reaches E31's 0.5 bar at any size; the unscaled split (c = 1) keeps 1.00-1.03 in 8 of 9 states, so the fall comes from the fitted scale c, which changes with δW's direction, not its size. No size threshold was ever filed | **weakened** (adversarial half falsified) | Cluster debug job, ~25 min, $0, r1 seed 0: (a) SGD 0.03 checkpointed at SGD 0.01's sizes 0.008 and 0.014; (b) SGD 0.01's step-1000 δW scaled down to sizes 0.0034 and 0.008. Size stands if (a) matches keep within 0.05 and (b) restores keep to ~0.37-0.42; falls if (a) differs by > 0.1 or (b) stays near 0.13 |
| (iv) E43: teacher training plus remainder is worse than leaving the update in Δ | 1 task (emoticon) × 1 fine-tune seed × 1 batch order × 1 adversary start; 2 steps (10, 1000) | "Worse than do-nothing" was never filed (P3 compares with the teacher-trained state). Worse on the gradient adversary only (3.92 vs 2.52 at step 10; 1.87 vs 1.50 at step 1000); the stochastic ratios are equal (1.003 vs 1.003; 0.917 vs 0.916). At step 1000 the 24% margin is one 128-row block (1.57 vs 1.02), where W₀ alone ranges ~20% over adversary seeds; the other block goes the other way (1.04 vs 1.08). At step 10, Δ′ is 5.1 × the fine-tune's squared size and mostly cancels the teacher's change (cosine −0.90), so the result is near-expected from size; do-nothing's step 10 (2.52) is itself an outlier along steps (1.05-1.50 elsewhere). Do-nothing was run fresh, same rows and seed | **weakened** | Cluster debug job, 12-30 min, $0 (no states saved; regenerates deterministically): do-nothing and plus remainder at steps 10 and 1000, adversary seeds 0-2, compared within the job only. Stands if plus remainder's lowest > do-nothing's highest at step 1000 and the ratio ≥ 1.1 at every seed; falls if the ranges overlap |
| (v) Teacher training needs Adam; plain SGD as its optimiser fails (E34 Amendment 6) | 1 task (r1) × 3 teachers (SGD, Adam, LoRA) × 1 SGD rate (0.03) × 1 step read (1000) × 1 batch order × 1 adversary start | Scored result stands (3.14, 3.14, 2.86 against 1.61; behaviour kept, P ≥ 0.985; far outside noise). But E34's own arm S (plain SGD as the student, on the SGD teachers) passes on emoticon at rate 0.01 (1.112 at step 100, 1.111 at 1000) and on r1 at step 100 (1.000): `experiments/teacher_training/results/teacher_training_{emoticon,r1_t534_v369}.json .teachers.sgd.S`, lead-checked. The Amendment 6 analysis, notebook, log card (`e34.html:41`, "so the map needs Adam") and `decisions.md:50` omit these. One untuned rate borrowed from E21, step-1 size never matched to Adam's, step 100 not read on the Adam/LoRA teachers (on the Not-run list). Same scope error as the earlier SGD grid null | **weakened** (wording exceeds the grid) | $0 now: rescope the card, notebook and `decisions.md` row to "plain SGD at 0.03, read at step 1000, fails on r1's Adam and LoRA teachers; passes on emoticon and at step 100 on r1's SGD teacher". Then a cluster debug job, ≤30 min: plain-SGD copies of r1's Adam and LoRA teachers at rates 0.003, 0.01, 0.03, read at steps 100 and 1000. Stands if no cell keeps P ≥ 0.9 and passes on both teachers |

## What cuts across all five

- **One adversary start everywhere.** Every reading in (i)-(v) is one 20-step start (`src/commute/faithfulness.py:128`
  seeds it the same for every state). E34 Amendment 5 read one state at 1.6 from the harness start and 9.2 from the
  budget start; W₀ alone moves up to ~7.5% across seeds, ~20% on a 128-row block. Margins in (i), (ii) and (iv) are
  ×1.03-1.24. A shared multi-start reader (`pile_faithfulness` already takes `seed`) would serve four of the five tests.
- **Few saved states.** The E34 main grid, Amendment 4, E41, E43, E16, E22, E31 and the plain-SGD copies were not saved
  (`experiments/SAVED.md`), so all but the (i) and (ii) re-reads need regeneration. All regenerate in ≤30 min on the
  cluster.
- **Card dependence.** The pass line is 1.40 on the H100 but 1.24 on the L40 (W₀ floor 0.0748 vs 0.209,
  `src/commute/faith_score.py:16`); cluster re-runs must compare within their own job, not against H100 numbers.

## Not run

- No reviewer ran code; every number is from saved JSON, logs and reports. None of the five cheapest tests has run.
- No second reviewer per claim, and no GPT-family reviewer (unavailable).
- Notebooks were read only partly (claim i's reviewer used grep on E34's analysis; E41 step-10 numbers came from
  `TRIED.md` and the check reports, not the JSON).
- The regating line (Q5) was out of scope.

The five reviewers' full reports: `reports/reviews/2026-10-29_review_claims-red-team_full.md`.

## Addendum 2026-10-29: each claim after E47

E47 ran the cheapest test of every row above, on the MATS cluster's L40, with two to four adversary starts per state and
each job's own lines (pass 1.241, fail 1.813 on rows 0-255). Analysis: `reports/analyses/2026-10-29_analysis_e47-claim-tests.md`;
notebook `notebooks/E47_claim_tests.ipynb`. Verdicts relative to this review's:

| Claim | This review | After E47 | What decided it |
|---|---|---|---|
| (i) | weakened | **stands** (two teachers, four starts, rows 0-255) | Every teacher-trained decomposition passes at every start, worst 1.225 (r1, h1) and 1.012 (rand32, h2); every do-nothing fails, r1 at budget 2.16 and h2 3.53, rand32 at h1 3.40 and h2 2.97. The rand32 power came from h1 and h2, not the budget start. Narrowest margin: r1's 1.225 against 1.241 |
| (ii) | weakened | **weakened** (unchanged) | Rank 8 inconclusive on rows 0-255 (mean 1.276) but passes on rows 256-511 (1.154 vs 1.450). No cap effect on rows 0-255 at rank 32 (1.264) or rank 128 (1.367); on rows 256-511 rank 128 reads 1.532, 1.33× rank 8, just under the fail line 1.552 (h0 alone 1.657). That row set's f is 0.035 from two starts. Combined: absolute unsettled, cap unsettled |
| (iii) | weakened (adversarial half falsified) | **weakened** (unchanged) | At matched size the rate-0.03 run keeps 0.08-0.09 more than the rate-0.01 run at all four points: between the 0.05 agree and 0.10 differ lines, so mixed on both seeds. Size alone was not isolated: keep drops ~0.29 between the two sizes along each run, where size and step count rise together, and the arm that changes size alone (the scaled-down update) was unscored, the scaled updates keeping P 0.06-0.26 of the target, under the 0.5 rule |
| (iv) | weakened | **unsettled** (the filed claim, step 1000); the step-10 reading falls by its filed rule, carried by one start | Step 1000 (T4, the claim as filed): q = 1.27, 1.005, 1.11, worse at every start but within the start spread. Step 10 (T4b, reported beside): q = 0.65 at h0, 0.96 and 1.07 at h1-h2 (within 4-7%), so not worse at every start; it does not show the opposite ordering. The t4b reproduction checks are weak (every P ≥ 0.999; the step-10 teacher-trained state was never read under the attacker), and step-10 plus remainder moved −64% between cards at h0 where the other states moved 6-13%. A $0 follow-up (t4c) reading the step-10 teacher-trained state is running |
| (v) | weakened (wording exceeds the grid) | **falsified** | Plain SGD as teacher training's optimiser passes on both r1 teachers at (0.03, step 100), (0.01, step 1000) and (0.003, step 1000): worst 0.98-1.17, P ≥ 0.986. Only (0.03, step 1000) fails (3.30), read in E47 on the Adam teacher only (the LoRA teacher's 2.86 is the earlier H100 reading). The wording becomes "plain SGD at 0.03 for 1000 steps fails; smaller settings pass" |

Filed-prediction misses: T3 (lean falls, read mixed), T4b (lean stands, read falls by its rule, carried by one start), T2c's cap (lean no cap
effect, read unsettled), and T1's "no power on rand32" (rand32 had power). The independent check of the analysis
(`reports/reviews/2026-10-29_review_e47-analysis-check.md`) confirmed all five verdicts; this addendum was revised after it.
The log's claims section changes for (iii) and (v)
after that check, and for (iv) only after t4c.
