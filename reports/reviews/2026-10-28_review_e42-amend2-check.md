# Review: independent check of the E42 Amendment 2 write-up

*2026-10-28. Scope: commit `55aced7`, which adds `reports/analyses/2026-10-28_analysis_e42-amend2.md`, cells 15-19 of
`notebooks/E42_teacher_new_behaviour.ipynb`, the two `TRIED.md` E42 rows (the E42 row's edited sentence and the new
"E42 Amendment 2" row), `experiments/teacher_training/plot_e42_amend2.py` and the figures
`experiments/teacher_training/figures/e42_amend2_{controls,curves}.png`. Sources checked against: the filed rules in
`experiments/teacher_training/results/predictions_e42_amend2.json`, the result
`experiments/teacher_training/results/teacher_training_e42_amend2_rand32.json` (read with jq), the log
`cluster/logs/commute-teacher-e42a2-40497.out`, Amendment 1's log `cluster/logs/commute-teacher-e42a1-40338.out` and
result `teacher_training_e42_amend1_rand32.json`, E42's `teacher_training_e42_rand32.json` and analysis, the plan's
"Amendment 2" section (`reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md`, lines 390-600), `decisions.md`'s
row on card `decision-e42-masked`, and VPD's released config (`artifacts/vpd/s-55ea3f9b/final_config.yaml`). No
python was run.*

**Verdict.** Every number checked matches its source, each control is scored by its filed rule exactly, and the tree
branch named is the one `decisions.md` and the plan specify. Nothing reverses. Corrections are wording only: the
write-up uses "damage" for the adversarial ratio although its own table has a "damage elsewhere" row that orders the
recipes the other way; a few sentences reach further than the run (the step-1 match, the "four recipes" curve, the
"remaining ways", what seeds 1 and 2 would show, the opening's "no training has yet kept"); and one candidate-cause line
overstates what the filed reasoning predicts. One process gap: no `docs/research_log/PENDING.md` line was queued.

## 1 Numbers

All checked against the JSON, the log or the named earlier result. "OK" means it matches to the printed digits.

| write-up number | source | value | status |
|---|---|---|---|
| step-1 terms, all four columns (unmasked 0.00163, CI-masked 1.129, stochastic 0.7389, trigger 0.0007714 / 0.6884 / 0.4363) | log 40497 step-1 lines; log 40338 line 20 | identical | OK |
| step-1 losses 0.9355 / 0.566 / 0.3711 / 0.564 | logs | identical; each = 0.00163 + 0.5 × trained terms (0.9356, 0.5661, 0.3711, 0.5640) | OK |
| checks before training match job 40338 | both logs: the six W₀ `[faith]` lines, C1 and G2 JSON | identical | OK |
| W₀ floor 0.20878; G2 0.00163 under 0.01 | `W0.pgd_floor`, `G2.path_kl` | 0.208775, 0.0016302 | OK |
| lines 1.24 / 1.81, stochastic 1.497 | 1.5 / 1.20878, 1.5 × 1.20878, 1.5 / 1.002088 | 1.241, 1.813, 1.4969 | OK |
| W₀ seeds 1, 1.21, 1.02 | `W0.pile.{0,1,2}.pgd_kl` / seed 0 | 1, 1.2088, 1.0190 | OK |
| adversarial 48.74 / 11.71 / 34.04 | `M0_*.pgd` | 48.741, 11.707, 34.044 | OK |
| block KLs 29.5, 37.9 / 8.33, 7.87 / 22.7, 24.4; W₀ 0.667, 0.716; Amendment 1 about 11, 25 | `[faith]` lines | 29.491, 37.941 / 8.331, 7.865 / 22.748, 24.352; 0.6671, 0.7164; 10.971, 24.980 | OK |
| stochastic 1.016 / 1.011 / 1.038; CI-masked Pile 0.929 / 0.998 / 0.987; unmasked Pile 0.914 / 0.889 / 0.899 | `M0_*.pile_ratios` | match | OK |
| P(target) 1.6e-5 ×3; trigger CI-masked P 1.5e-5 / 1.8e-5 / 1.4e-5 | `p_target`, `p_ci_trigger` | 1.623e-5, 1.613e-5, 1.587e-5; 1.453e-5, 1.790e-5, 1.440e-5 | OK |
| damage 0.0135 / 0.0195 / 0.0194; size 0.0370 / 0.0414 / 0.0443; teacher gap 0.076 / 0.098 / 0.097 | `M0_*` | match | OK |
| Amendment 1 column: 25.99, 1.028, 0.934, 0.900, 1.6e-5, 1.4e-5, 0.0231, 0.0514, 0.102 | log 40338 | match | OK |
| E42 copy column: 0.960, 0.936, 0.964, 0.0243, 0.0262; CI-masked Pile 0.98; trigger CI-masked 0.007 | `teacher_training_e42_rand32.json` `A.1000`; E42 analysis table (0.984); filed `amend1.e42_ci` | match | OK |
| W₀ column held-out P 1.5e-5, trigger CI-masked "about W₀'s P" | filed `expected.W0`, not measured in this job (references not run) | — | OK as a filed value; label it so (see § 3) |
| 11.7 to 48.7 is "six to twenty-seven times the fail line" | 11.71 / 1.81, 48.74 / 1.81 | 6.5, 26.9 | OK |
| random-mask ratios 0.94 to 1.04 | E42 copy 0.936, max 1.038 | match | OK |
| stochastic-only moved further (0.0414 vs 0.0370) yet "a quarter" of the ratio | 11.71 / 48.74 | 0.240 | OK |
| CI-masked alone "nearly twice" both terms | 48.74 / 25.99 | 1.88 | OK |
| curve table, every first → last 50-step mean, and times 185 / 188 / 186 / 195 | `M0_*.train.{first50,last50,seconds}`; Amendment 1 `M0.train` | all match to the printed digits | OK |
| trained terms to "about 0.7"; trigger-only to "about 0.55"; untrained −17% and −9% | ratios last / first | 0.690, 0.698; 0.567, 0.537; 0.826, 0.911 | OK |
| unmasked term "starts at 0.0016, jumps to about 0.09 within 50 steps, ends near 0.04" | step 1; first-50 means 0.089-0.094; last-50 means 0.037-0.042 | the 0.09 is a 50-step mean; single steps read 0.11-0.17 at step 10 (log) | OK as a mean; say so (§ 3) |
| both 128-row blocks fail in every control | each block over W₀'s same block: 44, 53 / 12.5, 11.0 / 34.1, 34.0 | all over 1.81 | OK (descriptive; no filed per-block rule) |
| job 40497, 14:32:56 to 14:55:55, 23 min; states about 60 MB | log; `ls` of `results/large/e42_amend2/rand32/` | 60.3 MB each | OK |
| "VPD trained against a persistent adversary at every step, and with no CI-masked loss" | `final_config.yaml`: `PersistentPGDReconLoss` coeff 0.5, `StochasticReconSubsetLoss` 0.5, `ImportanceMinimalityLoss`, `FaithfulnessLoss`; CI-masked only as an eval metric | — | OK |
| TRIED row numbers | as above | — | OK |

The figures plot exactly these values: the controls figure reads `pgd` and `stochastic` from the three JSONs and the W₀
seeds as ratios to seed 0; the curves figure plots the 50-step block means, with Amendment 1 shown only by its first
and last 50-step means (its blocks are not in its JSON).

## 2 Predictions vs filed rules

Filed rule for each control (`predictions_e42_amend2.json`): at step 1000, any Pile ratio NaN, or the adversarial or
stochastic verdict "fail" → fail; else either "inconclusive" → inconclusive; else pass; each ratio to this job's W₀
seed 0 against its own floor. The JSON's `pile_verdicts` use the floor 0.208775 (adversarial) and 0.002088
(stochastic).

| id | ratios (any NaN?) | verdicts | class by the rule | write-up | JSON `score` |
|---|---|---|---|---|---|
| C1 | — | `W0.pass` true | pass | pass | true |
| G2 | path KL 0.00163 ≤ 0.01 | — | pass | pass | true |
| C_ci | 48.74, 1.016, 0.929, 0.914 (none) | adversarial fail, stochastic pass | **fail** | fail, as leaned (medium) | fail, lean fail |
| C_st | 11.71, 1.011, 0.998, 0.889 (none) | adversarial fail, stochastic pass | **fail** | fail, lean "pass" (low) reversed | fail, lean pass |
| C_trig | 34.04, 1.038, 0.987, 0.899 (none) | adversarial fail, stochastic pass | **fail** | fail, as leaned (low) | fail, lean fail |
| P1, P2 | no copy ran | — | not scored | not scored | absent |
| G1 | teacher never made | — | not run | not run | null |

C_trig's run condition ("only when C_ci and C_st both read 'fail'") was met. The leans are copied correctly. The
reversal of C_st's lean is flagged loudly, as the project's rules ask. The write-up correctly notes that each state's
own `verdict` reads "below P 0.9" and that the control class reads the two Pile verdicts whatever P is, as filed. The
filed "references" stage (W₀'s trigger cells, E42's copy re-read) was not run because the job stops before the teacher;
the report lists it under "Not run", which is correct, and says the 0.007 stays in a log only.

The notebook's scorecard abbreviates the rule to "either Pile verdict 'fail' or NaN → fail", dropping the inconclusive
branch. That is harmless here (no verdict was inconclusive) but should read "as filed" or carry the full rule.

## 3 Claims and corrected wording

None of these changes a verdict. Report (R), notebook (N), TRIED row (T).

1. **"Damage" names two different things (R, N).** "The stochastic term alone did the least damage and the CI-masked
   term alone the most", "nearly twice the damage of both terms together", "does not tell the recipes' damage apart",
   "the size of the change does not order like the damage" all mean the adversarial ratio. The same table has a row
   "damage elsewhere" on which the CI-masked recipe is the *least* changed (0.0135 against 0.0195 and 0.0194). Replace
   with "read the lowest / highest adversarial ratio", "nearly twice the adversarial ratio of both terms together",
   "does not tell the recipes' adversarial ratios apart", "does not order like the adversarial ratio".
2. **Step 1 shows the first batch and draw, not all 1,000 (R § apparatus, N cell 15).** The match proves the first
   batch and first mask draw are Amendment 1's (and that the weight-0 term draws its masks the same way at step 1). That
   later batches and draws follow the same order rests on the code (the plan's "Code" paragraph), not on this match;
   after step 1 the terms must differ because the parameters do. N's "so the batches and masks are the same" →
   "so the first batch and mask draw are the same; the code keeps the order after that".
3. **"The four recipes trace the same curve" (R, N, T).** Three recipes have 50-step curves; Amendment 1 has only its
   first and last 50-step means (0.092 and 0.042), which sit on the others' curve. → "The three new recipes trace the
   same curve, and Amendment 1's first and last 50-step means sit on it." Also "jumps to about 0.09 within 50 steps" →
   "averages about 0.09 over the first 50 steps (single steps reach 0.11 to 0.17 near step 10)".
4. **Opening and "What we learned" (R, N cells 15 and 18).** "So no training of this kind has yet kept a new behaviour
   in the parts the decomposition switches on" is true but reads as though a copy was tried and lost the behaviour; no
   copy ran. → "So no masked copy was made, and whether training with Γ fixed can keep the new cipher in the parts the
   decomposition switches on at the cue is still untested." "No frozen-Γ map has been shown able to keep a new
   behaviour in the CI-masked model" should keep E42's scope: "the random cipher" (E42's identity control, which the
   model half had, kept 0.93 at the trigger under a frozen Γ), on the 67M.
5. **"The remaining ways to test it change Γ or add an adversarial term to the loss" (R "What we learned").** Not the
   only ones: the report's own "Not run" lists the masked terms on Pile rows, other weights, a second batch order and
   VPD's sparsity loss. → "Untried ways include letting Γ change and adding an adversarial term to the loss; neither is
   in the approved tree."
6. **Candidate cause 3 (R).** "The filed reasoning for the CI-masked lean ... would predict that order" → "is consistent
   with that order, but the same reasoning predicted the stochastic term alone would pass, and it did not." The
   section otherwise asserts no cause and labels all three as untested, as the rules ask.
7. **"Seeds 1 and 2 would show whether the ordering among recipes is real" (R, N "What next").** Adversary seeds test
   only the adversary-start part of the uncertainty; the ordering also rests on one batch order, which the seeds do not
   touch. → "would show whether the ordering survives other adversary starts; batch order would stay untested."
8. **"The failures are far outside any noise" (R).** → "far outside every measured noise floor (W₀'s adversary seeds
   spread 0.21; a trained copy moves up to 25% between batch orders); a control's spread across batch orders is
   unmeasured." Likewise "only 'all far past the line' is established" → "on one batch order and one adversary start,
   all far past the line". A fall from 11.7 to under 1.81 would need an 85% drop, far beyond the 25% seen for trained
   copies, so the fail itself is safe; the ordering is not.
9. **"As good as before on average" (R, N).** → "as good as before on Pile text under random masks". Fine otherwise.
10. **Table labels (R).** Mark the W₀ column's held-out P 1.5e-5 and trigger CI-masked "about W₀'s P" as filed
    values, not read in this job (the references stage did not run). E42's copy's unmasked Pile ratio is on file
    (0.810, E42 analysis) if the cell "not read here" should be filled; leaving it is acceptable.

**What each metric is blind to (check of the report's list).** The list covers the adversarial ratio (one start of 20
steps, 256 Pile rows, not the cipher windows; a lenient start only raises the true ratio, so it cannot rescue a fail),
the stochastic ratio (blind to changes that cancel on average, hence level in every failing state), the two P readings
(nothing copied, nothing about damage), damage and size, and the unread sparsity loss. Missing, worth one line each:
the adversarial ratio is a ratio to W₀'s seed-0 reading (0.692), the lowest of W₀'s three (seed 1 reads 0.836), which
matters near the line and not at 12 to 49; the CI-masked and unmasked Pile ratios (0.89 to 1.00) are reported but not
interpreted and have no expected value filed; the route shares are rightly called unread. Every claim the write-up
makes from the ratios is limited to one task (rand32), one batch order, one adversary start, one weight (0.5), 1,000
steps, Γ frozen and the 67M model; the report and TRIED row both state this. The claims that need the scope added are
items 4 and 8 above.

**Process gap.** `CLAUDE.md` asks that every analysis report append a line to `docs/research_log/PENDING.md` when it
is written. Commit `55aced7` does not, and `PENDING.md` at `HEAD` has no E42 Amendment 2 line; the log's card
`decision-e42-criterion` still says the run is running and recommends waiting for it.

## 4 Tree branch

`decisions.md`'s row on card `decision-e42-masked`: "If one term alone keeps the adversarial check, the cipher copy runs
with that term; if both fail, the same control with the masked terms at the trigger only; if that fails too, stop and
leave `decision-e42-criterion` as it is." The plan's tree (line 538): "Both fail → M0_trig (same job). Holds → the copy
with recipe trig. Fails → stop: the masked line ends; card `decision-e42-criterion` stays as it is, with masked teacher
training marked as tried and failed."

C_ci and C_st both failed, M0_trig ran and failed, and the job stopped with no copy (log 14:55:55, `stopped` in the
JSON). That is the branch the report names, quoted correctly from both files. One wording difference between the two
sources does not bite here: the `decisions.md` row keys on "keeps the adversarial check", the filed tree on the full
class (both verdicts); every stochastic verdict passed, so both read the same. "Marked as tried and failed" comes from
the plan, not the `decisions.md` row; the report attributes it correctly to the plan's branch.

## 5 Wording allowed above the log's cards

For the overview, the timeline and the `decision-e42-criterion` card's evidence line:

> **E42 Amendment 2 (MATS cluster, $0).** We reran the masked recipe's nothing-to-copy control three simpler ways on
> the 32-key cipher: with only the term that matches the model rebuilt from the parts marked important, with only the
> random-mask term, and with both read at the cue alone. All three broke the worst-case check on ordinary text, at 12
> to 49 times the unchanged model against a fail line of 1.81, while the random-mask check stayed level (1.01 to
> 1.04). By the approved plan the masked line stops there, so no masked copy was made: whether training with Γ fixed
> can keep the new cipher in the parts the decomposition switches on at the cue is still untested. The card on whether
> the test should change stays as it is, with its "try such a copy first" option tried and failed. One task, one batch
> order, one adversary start, one weight, on the 67M model.

Allowed if a shorter line is needed: "Every masked version of teacher training broke its own nothing-to-copy control
(four of four, each at least six times past the fail line), so the masked copy was never run."

Allowed in a card's Surprise line: "The random-mask term alone also failed (11.7), reversing a low-confidence lean
towards a pass; why is not tested."

Not allowed:
- "The stochastic term does the least damage" / "the CI-masked term does the most damage" as a finding: the ordering
  is one batch order and one adversary start, and "damage" is another measurement that orders the other way.
- "Masked teacher training cannot keep a new behaviour", "a frozen Γ cannot carry the cipher", or "no training has kept
  the cipher in the switched-on parts" without saying no copy ran.
- Any cause: "because there is no worst-case term", "because it overfits 403 windows", "the cancelling explains it".
- "The decomposition is as good as before on average" without "on ordinary text under random masks".
- "The only remaining options are letting Γ change or an adversarial term."
- "All four recipes trace the same curve" (Amendment 1 has two points).
- "Seeds 1 and 2 would show whether the ordering is real."
- "The masked line is closed for good": it is stopped by the tree; option (c) of `decision-e42-masked` (an adversarial
  term matched to the teacher) is untried.

## 6 Not checked

- The code: `src/commute/masked_terms.py`, `newskill_amend2.py` and the sbatch file were not read; whether the batches
  and mask draws after step 1 follow Amendment 1's order, and whether the trigger-only term reads the right position,
  rest on the plan and its pre-run review, not on this check.
- The pre-run review the plan's order of work asks for (step 2); not looked for.
- `check.py` on the report and notebook (no python on the dev node).
- E42's own findings that the write-up leans on (masked-off share 0.839, the 0.007 from E42's log, "the cipher sits in
  subcomponents Γ keeps off"), beyond matching the quoted numbers to E42's analysis.
- E37 Amendment 1's 3.84 and its rows; E34 Amendment 4's 25% batch-order spread; E34 Amendment 5's lenient-start
  finding: taken from the filed predictions and plan, not their own results.
- The figures' visual details beyond the plotted values and titles; the notebook cells outside 15-19.
- The research-log card's option letters beyond the text of `decision-e42-criterion` in
  `docs/research_log/parts/sections/decisions.html`.
