# Review: what E4's "no SGD rate works" steered, and what to re-run or reword

**In plain words.** On 2026-10-15 experiment E4 reported that plain SGD on the pieces' vectors learned the emoticon
behaviour at no learning rate without large damage elsewhere, and from then on every component-training run used Adam.
On 2026-10-21 a finer grid showed that E4's smallest rate was already too large, and that SGD learns cleanly at rates
of 0.003 to 0.1. This review traces where the wrong premise went: it props up one half of the project's current
one-sentence claim, "no map with Γ frozen keeps the decomposition adversarially faithful", whose component-training
evidence is Adam at one untuned rate. One re-run of about an hour decides whether that half holds for the operation
or only for Adam, and it could send the commute question back to first place.

*2026-10-21, lead session with four read-only Opus audits (one per slice: prose reports; experiments and code; the
research log and notebooks; the chain of argument). The audits as returned are the appendix,
`reports/reviews/2026-10-21_review_sgd-finding-influence_audits.md`, with file:line evidence for every item below. Nothing
was run. Vocabulary: `GOALS.md` → Terms.*

## The correction

- **E4** (`reports/analyses/2026-10-15_analysis_concentration-test.md`): SGD component training at rates 0.3 to 30 never met
  the rate rule. Its P1 ("Adam causes the diffuseness") was left unscorable, and the report said no SGD rate works.
- **E4 follow-up** (`reports/analyses/2026-10-21_analysis_sgd-grid.md`, `experiments/sgd_grid`): at 0.003 to 0.1 SGD learns
  within the off-target limit. At 0.01 its KL elsewhere is 0.0065 against Adam's 0.009. Reverting the ten largest
  pieces removes 0.16 of SGD's gain against 0.016 for Adam, so P1 scores as filed: fail, SGD is not concentrated.
- **Two caveats that bound every fix.** The comparison is not matched for size of change: SGD's largest relative
  change per matrix is 0.0026 against Adam's 0.0066, so "less damage" and "ten times less diffuse" are both read on a
  smaller update. And the SGD state's faithfulness is unmeasured on every axis (Pile ratios, trigger-local masked
  checks, the route split). E4's overshooting SGD state read a Pile adversarial ratio of 24.9, so SGD states can fail.
  Nothing here says SGD is a working map; it says the question was closed on a grid that never contained the answer.
- **Same flaw, uncorrected, on the fine-tune side.** E2's SGD fine-tunes ran at 0.1 to 1 with off-target KL of 0.27
  or more. The "Adam or SGD alike" outside-share line in `TRIED.md` rests on them.

## Where the premise entered the argument

The premise was never a decision; `decisions.md` has no optimiser row. It is a stated reason in three places and a
definition in two.

1. `reports/decisions/2026-10-19_decision_success-criterion.md` clarification 3: every run used Adam because E4 found no SGD rate.
2. The research log's component-training technique card and E4 card ("the runs after this use Adam").
3. `backlog.md`, the unused-contrast entry: "needs a rate search first", and "points at the optimiser building
   cancelling changes".
4. `GOALS.md` Terms defines component training as "Adam, many steps" (user-owned); the log's glossary repeats it.

Every component-training state scored for faithfulness after E4 is Adam's at the rate chosen on emoticon in E2,
1e-5, never re-tuned per task: E5, E10, E11, E13, E15's component-training arm and E16.

## What depends on it

The current claim (`reports/decisions/2026-10-21_decision_regating-main-line.md`) has three parts.

- **"Mostly carried by re-gating" does not depend on the optimiser.** The re-gating share is read on the fine-tune's
  own update under do-nothing placement, it agrees across the concentrated component-training arms, and E4's one SGD
  state also runs through re-gating. Nothing to re-run for this part.
- **"No map with Γ frozen stays adversarially faithful" is Adam-only on the component-training side.** The split
  results stand on their own. The component-training failures are E11's three ladder tasks (ratios 2.7, 2.2 and 6.7)
  and E10's rand32 (2.8), all Adam at 1e-5. E11's own pattern ties the failures to off-target damage, which on these
  tasks is 0.15 to 0.51 at step 10 against E4's limit of 0.11. So Adam overshoots on the ladder, and SGD lowered that
  damage on emoticon. An SGD arm (with Adam at lower rates beside it, so the optimiser is not confused with step size)
  is a direct test.
- **"The failure is cancellation" reads Adam states only** (the adversarial-vs-stochastic analysis, E15 stage 0). E17
  as filed cannot tell "Adam builds cancelling changes" from "any training that learns these tasks does". Its P1
  rationale ("Adam's steps are sign-like, so changes add coherently") and the backlog's "the optimiser builds cancelling
  changes" are two Adam stories the project has not reconciled.

**The decision's own exit.** Its first "what would move us back" bullet is a map at the floor with the behaviour at
steps 100 and 1000. It names only E16 and the Γ refit, and E16 already met its letter on emoticon with Adam but was
discounted as an analogue of a map rather than a map. SGD component training answers that objection better than
Adam does: with an SGD fine-tune, each SGD step of component training is the adjoint split of that fine-tune's step.
So the re-gating line survives as science either way, but its priority rests on Q1 being stuck, and that rests on
Adam. If SGD passes on the ladder tasks at steps 100 and 1000, the decision's own rule sends Q1 back first.

**One claim that does not hold as written, independent of any run.** The log's Result 3 row, "the moved pieces carry
a tenth or less under every operation", cites E4 while E4's concentrated arms read several times that when converted
to log-odds shares (siblings about 0.4, top 10 about 0.3, the overshooting SGD state about 0.2, against Adam-all 0.04).
The row holds for diffuse Adam training, not for every operation. Fix: reword now, route-split the saved SGD and E4
states to confirm.

## Re-runs, ranked

Costs: cluster `debug` ≤ 30 min ≈ $0; RunPod 4090 ≈ $0.5 per hour; H100 ≈ $2.5 to $4 per hour plus 5 min set-up.
Runtimes come from the nearest recorded run, named in each row. Every SGD arm is a new experiment directory calling
`component_training.train(..., optimizer="sgd")` with the Pile harness, since finished `run.py` stay as they ran and
none has an optimiser flag. On the cluster L40 the W₀ adversarial floor has twice read about 0.21, so verdicts near
the pass line of 1.5 need an H100 or more attacker seeds; the failing states at 2.2 to 6.7 read fine on an L40.

| # | Run | Why it matters (GOALS) | Runtime (basis) | Cost | Importance |
|---|---|---|---|---|---|
| 1 | **Faithfulness of the saved SGD emoticon state**: Pile ratios with same-job W₀ floors, E4's trigger-local checks under the state's and W₀'s gates, the route split. The control for everything below. | Checks the SGD path end to end on the harness before any verdict is read; gives the first direct share d for an SGD state (Result 3). Emoticon passes with Adam, so a pass changes no verdict. | ≈ 15 to 25 min (E11's emoticon rerun 12.5 min; E4 amendments ≈ 7 min) | $0, cluster | Medium, runs first |
| 2 | **Matched size of change and momentum**, emoticon: SGD vs Adam at equal largest relative change, top-10 and top-100 revert, KL elsewhere; SGD with momentum 0.9. | Without it every comparative sentence ("less damage", "ten times less diffuse") is confounded with update size. Fixes the wording of all the text changes. | ≈ 5 to 10 min (sgd_grid job 28106, 3 min) | $0, cluster | Medium, with 1 |
| 3a | **SGD component training on the E11 ladder card at step 10**: emoticon plus the five ladder tasks, rate grid 0.003 to 0.1 by E4's rule per task, **plus Adam at 1e-6 and 3e-6**, scored against E11's saved Adam states and do-nothing re-read in the same job. | The main test. Decides whether "component training fails the weak version on about half the tasks" is a property of the operation or of Adam at one rate. | ≈ 50 to 60 min (E11: 38 min on an L40S, plus a 3-min rate grid per task) | ≈ $0.5 on a 4090, or $0 as two debug jobs with the floor risk | **High** |
| 3b | **Steps 100 and 1000 for both optimisers** on the same tasks, if any failing task passes at step 10. Subsumes an SGD version of E16. | The move-back bullet as written. Adam has no ladder numbers past step 10 either. | ≈ 1.5 to 2 h on an H100 (E16: 2118 s for two 1000-step arms and 30 states) | ≈ $4 to 5 | **High**, conditional on 3a |
| 4 | **SGD on the key-to-value tasks** rand1, ident32, rand32, same rule as E10, with an Adam lower-rate arm. | E10's rand32 failure (2.81) is the only "new capability" evidence that frozen-Γ maps fail on a behaviour the model lacks; the causes were never separated. | ≈ 25 to 30 min (E10 job 22374, 26 min) | $0, cluster | Medium–High |
| 5 | **The unused contrast**: SGD fine-tune rate check (0.001 to 0.1), then SGD component training at that fine-tune's size against the adjoint state of the same fine-tune, emoticon, Pile harness. Can fold into E19's rerun as an SGD-fine-tune variant (E19 stopped at its W₀ control anyway). | Settles why: path, scale, or preconditioner. Answers the "analogue, not a map" objection. Changes the explanation more than the verdict. | ≈ 20 to 25 min (sgd_grid 3 min; E11 harness 12.5 min) | $0, cluster | Medium |
| 6 | **E17's cancellation statistics on the SGD states** from 3a, as a dated amendment after E17 is analysed as filed. | Only SGD states can separate "Adam cancels" from "component training cancels". | ≈ 5 to 10 min (E17 plan: 4 to 5 min per state) | $0 | Medium, after 3a and E17 |
| 7 | **E2's outside share on a clean SGD fine-tune** (rates 0.001 to 0.03). | Only the TRIED E2 "Adam or SGD alike" line depends on it. | ≈ 10 min | $0 | Low |
| 8 | **E7's one-matrix arm with SGD** (rank 8 on `h.2.mlp.down_proj`). | E13 left the one-matrix failure's cause unseparated; a smaller, less diffuse step is a handle on it. | ≈ 11 min (E7 session 37 min) | $0 | Low |
| 9 | **E15 stage 0 on an SGD state** beside the Adam one, after the E15 control passes. No refit of the SGD state (2 h, $5 to 8). | A second reading of "largest changes land on inert pieces", which looks like Adam's per-coordinate scaling. | ≈ 6 min on an H100 | ≈ $0.4 | Low until E15's control passes |

Order: 1 and 2 together on the cluster, then 3a, then 3b and 4, then 5; 6 to 9 as their parents finish. Items 1 to 4
come to about $5 to 6 of the $40 session budget. Item 3 can change the main line, so it goes to the user at `/design`
even though it is the ladder report's own "Not run" item.

## Adjustments that are not runs

**Living files, to edit now** (the log-rewrite pass does these; line-level detail in the appendix):

- `docs/research_log/index.html`: 35 edits. The E4 card's second Found point and a Follow-up field with the two
  `sgd_grid` figures; the component-training technique card, apparatus and glossary ("Adam, many steps" → Adam in
  most runs, plain SGD in E4 and its follow-up); Result 1, Q5, the E3 card's Moved-on, claim rows (two new rows: "SGD
  cannot learn within the limit", refuted; "Adam is why the update is diffuse", refuted as filed), the Result 3 row
  rescoped to diffuse Adam training; the unused-contrast item under Maps not yet run (it is there, not on the re-gating
  card as `PENDING.md` says); a Gaps card, "one optimiser behind every component-training result"; the one-sentence
  claim and the move-back bullet rewritten as "Adam component training" and "any map"; Mistakes row; timeline;
  changelog. Over-reach to avoid: SGD does not enter Result 1, Q2's answer or the live-candidate status; "less damage"
  becomes "about the same at a smaller size of change".
- `experiments/TRIED.md`: E4 row (scope the null to its grid), E3 row (partly answered), E2 row (SGD fine-tunes were
  at 0.1 to 1 only).
- `backlog.md`: the unused-contrast entry (drop "needs a rate search first", reword the optimiser sentence as a
  testable hypothesis; the SGD fine-tune still needs a rate check); the per-subcomponent identity comparison is no
  longer moot for an SGD state.
- `README.md`: state of play names Adam where it states Adam results as properties of component training; the codes
  table gains the E4 follow-up.
- `theory/commuting_updates.md`: "every run used Adam for 10 to 300 steps" is wrong twice (E4's SGD arms; E16 to 1000).
- `decisions.md`: a new row recording the optimiser as a revisable choice.
- `docs/research_log/README.md`: two figure entries in the files map.
- `notebooks/E02-E04_components_and_gating.ipynb`: dated notes on the three cells that record the 2026-10-15 verdict,
  and one factual fix (cell 17 says the rate rule picks the smallest passing rate; the filed rule picks the passing
  rate with the least off-target KL).
- `src/commute/component_training.py`, three small changes to measurement code, so they follow the usual rule of
  re-read, smoke and review, in a fresh session. Assert the optimiser string instead of treating anything but "adam"
  as SGD. Put the rate next to the optimiser, because the default rate 1e-5 is Adam's and is a thousand times too
  small for SGD. Say in the docstring that clipping at 1.0 is active on every SGD step, so "plain SGD" is a normalised
  step of length equal to the rate. `src/commute/ladder.py` should say coverage describes an SGD first step while the
  ladder trained with Adam.

**Dated reports stand; the corrections live here.** The sentences to read against this review:

- `2026-10-19_decision_success-criterion.md` clarification 3: Adam was a choice, not a necessity.
- `2026-10-19_review_update-map-variants.md` lines 68 to 69: variant (b) needs a working SGD fine-tune rate, not a
  component-training one, and that was never found either.
- The E4 report's lines 11, 43 to 45 and 83: the null holds for the grid 0.3 to 30 only, and P1 is now scored.
- The terminology review's item 8, and `2026-10-21_analysis_split-comparison.md`: "component training" there means
  Adam.

**Filed predictions are never edited.** Those whose rationale assumed Adam (task-ladder amendment A1, identity, E17's
P1) are scored as filed; their analyses say the rationale is untested until SGD states are read.

**For the user (user-owned or a change of direction):**

- `GOALS.md` Terms, component training: "Adam, many steps" → "train the alive (U, V) with the run's optimiser; Adam
  in most runs, plain SGD in E4 and its follow-up". GOALS's map definition ("the same training step applied to U and
  V") is already optimiser-neutral.
- Whether item 3 runs ahead of the queued re-gating designs, since its outcome can move the main line.

**Process, so this does not recur.**

- A grid's null is reported with its scope ("no rate in 0.3 to 30 by step 30"), never as "no rate works"; when the
  best value sits at the grid's edge and fails in that direction, the grid is extended before the null is written.
- An unscorable prediction's failed precondition is not reused as a premise by a later design without a decision row.
- The `/design` "held fixed / varied" line names the optimiser in every comparison of operations.
- Debug sbatch scripts request up to 2 h against the 30-min policy; new ones ask for 30 min.

## Not run

- Everything in the table. The saved SGD state may exist only on the cluster NFS (`experiments/sgd_grid/results/large/`
  is empty locally); it regenerates in 3 min if not.
- The log-odds conversion of E4 amendment 1 is indicative (logit of mean probabilities), not a route split.
- E17's scored job 28110: runtime and status unknown locally.
