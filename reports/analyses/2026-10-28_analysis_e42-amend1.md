# Analysis: does masked teacher training hold still when there is nothing to copy? (E42 Amendment 1)

*2026-10-28. Plan `reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md` → "Amendment 1"; predictions
`experiments/teacher_training/results/predictions_e42_amend1.json`, filed 13:13:54, unedited; result
`experiments/teacher_training/results/teacher_training_e42_amend1_rand32.json`; log
`cluster/logs/commute-teacher-e42a1-40338.out`. The run stopped at its control, so there is no figure: the stop is one
number far past its line, and the table below holds it.*

We added two masked reconstruction terms to teacher training on the cipher windows and first ran the recipe with
nothing to copy, the teacher being the unchanged model. That control broke VPD's adversarial check on ordinary text,
which read twenty-six times the unchanged model, while the random-mask check stayed level. So this recipe cannot yet
tell us whether the parts the importance network switches on can carry a new behaviour, and that question stays open.

## What was tested

E42 found that teacher training copies a fine-tune that teaches the model a word cipher, and passes the Pile checks.
But the model built from the subcomponents Γ switches on at the cipher's trigger keeps almost none of it. On the
32-key cipher it puts 0.007 on the right word, against 0.964 with every subcomponent on.

This amendment adds two terms to teacher training's loss, each matched to the teacher with weight 0.5. The first is
the CI-masked student: the model built from the subcomponents alone, under Γ's masks on its own activations, with Δ
left out. The second is the stochastic-masked student, whose masks are $g + (1-g)r$, with $g$ Γ's value and $r$
uniform on [0, 1], drawn afresh at every position. Γ and Δ stay frozen, and only the subcomponents' $U, V$ train.

Before trying it on the cipher, the control M0 ran the recipe with the teacher set to the unchanged model $W_0$. It
ran 1,000 steps on the cipher's 403 training windows. With nothing to copy, a usable recipe should leave the
decomposition where it was. The filed rule stopped the job before the real arm if either Pile ratio read "fail".

The Pile check reads VPD's reconstruction losses on 256 Pile validation rows, each as a ratio to $W_0$'s value. In
the stochastic check, the masks are random at every position. In the adversarial check, each subcomponent gets one
setting shared by every position of the 128 rows in a block. It is chosen by 20 steps of gradient ascent to pull the masked model's output furthest from the
target model's. On this L40 card the adversarial ratio passes at 1.24 or less and fails at 1.81 or more. The
stochastic ratio passes under 1.497.

## What was observed

The recipe failed its control. The adversary pulled the masked model 18 nats from the target model on Pile text,
where it pulls $W_0$'s 0.69. Both 128-row blocks failed, at about 11 and 25 nats. The other three Pile ratios sat at
or below 1. Nothing was copied, since the target's probability stayed at $W_0$'s. Yet the recipe moved the
subcomponents twice as far as copying the cipher did, with about the same damage to Pile text.

| quantity | $W_0$ | E42's copy, rand32, no masked terms | M0 expected | M0 read |
|---|---|---|---|---|
| held-out P(target) | 1.5e-5 | 0.964 | under 1e-3 | 1.6e-5 |
| trigger CI-masked P(target) | about $W_0$'s P | 0.007 | under 1e-3 | 1.4e-5 |
| adversarial Pile ratio | 1; seeds 1, 1.21, 1.02 | 0.960 | 0.8 to 1.24 | **25.99, fail** |
| stochastic Pile ratio | 1 | 0.936 | about 1 | 1.028, pass |
| CI-masked Pile ratio | 1 | 0.98 | not filed | 0.934 |
| unmasked Pile ratio | 1 | not read here | not filed | 0.900 |
| damage elsewhere | 0 | 0.0243 | under 0.02 | 0.0231 |
| size, largest relative change of a matrix | 0 | 0.0262 | not filed | 0.0514 |

The training terms, as means over the first and last 50 steps, in nats:

| term | step 1 | first 50 | last 50 | at the trigger, first 50 → last 50 |
|---|---|---|---|---|
| unmasked, the target model against the teacher | 0.0016 | 0.092 | 0.042 | 0.083 → 0.031 |
| CI-masked student against the teacher | 1.129 | 1.131 | 0.776 | 0.719 → 0.409 |
| stochastic-masked student against the teacher | 0.739 | 0.740 | 0.506 | 0.461 → 0.249 |

The route shares in the result file split a gain of almost nothing and are not read.

## Why: a hypothesis

One part is a fact, not a hypothesis. With the teacher at $W_0$, only the unmasked term had nothing to do. The masked
students are never exact. On Pile text, VPD's own decomposition leaves its CI-masked model 0.35 nats per position from
the target model, and its stochastic-masked model 0.25. On the cipher windows, away from the trigger, they start a
little higher, at 0.44 and 0.30. At the trigger they start at 0.69 and 0.44, and the loss weights the trigger as much
as the other 40 positions together. So matching the masked students to the teacher asks the subcomponents to beat VPD's
own reconstruction on 403 windows. VPD's training had traded that reconstruction against sparsity and an adversary.
The recipe removed about a third of it in 1,000 steps. It paid with the target model's own function: the unmasked term
rose from 0.0016 nats at step 1 to about 0.04.

**The hypothesis.** Training $U, V$ toward lower CI-masked and stochastic loss, without the worst-case term VPD trains
with, makes the decomposition easy to attack. VPD's released run trains with four losses
(`artifacts/vpd/s-55ea3f9b/final_config.yaml`, read against the pinned `param_decomp` commit `74146b5`):

- the stochastic subset reconstruction loss, weight 0.5, random masks on a random subset of layers;
- a persistent adversarial reconstruction loss, weight 0.5, whose adversarial masks are kept from step to step, one per
  row slot and position, and updated by their own Adam;
- the importance-minimality loss, weight 2e-4;
- the faithfulness loss, weight 1e7, the mean square of Δ.

It has no CI-masked reconstruction loss in training; the CI-masked loss is only an evaluation metric there. So the
released decomposition was trained against an adversary at every step, and this recipe drops that term. A random mask
drawn afresh at every position and subcomponent sees the average of many small changes, so changes that cancel on
average cost it nothing. The harness's adversary picks one setting per subcomponent, shared by every position of 128
rows, so it can line those changes up.

**Evidence for it.**

- The pattern. The random-mask and CI-masked Pile ratios read 1.03 and 0.93, and the adversarial ratio read 26. The
  recipe left the decomposition as good on average and far worse in the worst case, which is what dropping a
  worst-case term would allow.
- E42's copy ran on the same windows, with the same batches, optimiser and loop, but no masked terms, and passed at
  0.96. The masked terms are the one intended difference. The control reading the trainer's forward path differed
  from the model's by 0.0016 nats.
- The masked terms moved the subcomponents twice as far as copying the cipher did, with nothing to copy.

**Evidence against it.** E37 Amendment 1's masked recipe kept VPD's persistent adversary in its loss, and its own
nothing-to-copy control still failed, at 3.84. At a tenth of the masked weights it read 3.14. That recipe matched
the masked students to the student's own output on 512-token Pile rows, so the two are not directly comparable. Its
excess over the line was about a ninth of this one's, which fits an in-loop adversary limiting the damage without
removing it. A second cause also fits both runs. Each pushed the masked models' error below the level VPD's training
had settled at, with Γ frozen and no sparsity loss to push back, and on a narrow set of inputs. No run separates the
two causes.

**What would test it.** Three nothing-to-copy controls, each about 6 minutes on an L40, $0:

1. The present recipe plus a persistent adversarial term matched to the teacher, shaped for 41-token windows. Holding
   still would point at the missing worst-case term. This needs new code and a review.
2. Each masked term alone. The hypothesis expects the stochastic term alone to fail too, since it is also an
   average-case loss.
3. The masked terms computed on Pile training rows, the unmasked term on the cipher windows as now. Holding still
   would point at the narrowness of the 403 windows rather than at the missing adversary.

## What we learned

The masked recipe is not usable as specified, and two masked recipes have now failed this control: E37's with VPD's
adversary in the loop, and this one without it. So no frozen-Γ map has been shown to keep a new behaviour in the
CI-masked model, and none has been shown unable to. `GOALS.md` asks whether a failure comes from a new capability or a
brittle decomposition. For that fork, E42's finding stands as it was. With Γ frozen, the cipher sits in subcomponents Γ
keeps off at the trigger, and whether that is forced is untested.

## How compelling is the evidence

The failure is far outside any noise: 26 against a fail line of 1.81, with $W_0$'s three adversary seeds spread by
0.21. Both blocks failed. It is one task, one batch order, one adversary start, one masked weight. The cause is a
hypothesis: no run separates the missing adversarial term from the fit to unfamiliar text. M0's state was not saved,
so it cannot be re-read without a rerun.

## Surprises

One: the size. The control was expected to stay between 0.8 and 1.24, with E37's failure at 3.84 named as the risk.
It read 26, seven times E37's failure, while every average-case reading improved.

## What next

The tree sends this to a pending decision, not a run. The card is below.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| C1 | $W_0$ under the Pile harness is its own floor | `w0_check` | floor 0.2088, the same as E42's | pass |
| G2 | the trainer's unmasked forward is the model's | path KL at $W_0$ ≤ 0.01 nats | 0.0016 | pass |
| M0 | the masked recipe holds still with nothing to copy | stop if either Pile ratio "fail" or NaN | adversarial 25.99 fail; stochastic 1.028 pass | **fail: job stopped** |
| G1 | E42's teacher reproduces E10 at step 10 | within 0.02 | not run | — |
| P1 | arm M's CI-masked model keeps the cipher at the trigger | ≥ 0.5 kept, < 0.1 not kept | arm M not run | **not scored**: the control stopped the run |
| P2 | arm M passes the Pile check | `faith_score.state_verdict` | arm M not run | **not scored**: the control stopped the run |

M0's own `state_verdict` reads "below P 0.9" because it checks the target's probability first. The stop used the
two Pile verdicts directly, as filed.

## Not run

- Arm M, the masked recipe with the cipher teacher, on rand32; so neither filed prediction is scored.
- The teacher regeneration and its check against E10; the reference re-reads, so E42's trigger CI-masked value of 0.007 is still in a
  log only, not a JSON.
- Job 2, rand1.
- M0 with the masked weights at 0, which would separate the masked terms from the rest of the loop.
- M0 on Pile rows; M0 with an in-loop adversary; each masked term alone; the masked terms at the trigger only.
- A second adversary start or a second batch order for M0; M0's state was not saved.
- VPD's sparsity loss, not read.

## Provenance

MATS cluster, L40, debug QoS, job 40338, 2026-10-28 13:19 to 13:31, $0. M0 trained in 195 s. Code
`src/commute/masked_copy.py`, `experiments/teacher_training/newskill_amend1.py`, `cluster/teacher_e42_amend1.sbatch`.
VPD's losses from `artifacts/vpd/s-55ea3f9b/final_config.yaml` and `param_decomp/persistent_pgd.py`,
`param_decomp/metrics/faithfulness_loss.py` at commit `74146b5`.

## Pending-decision card (for `parts/sections/decisions.html`, id `decision-e42-masked`)

**Blocking** · Commute line · new 2026-10-28 · Nothing runs until this is decided.

**Teacher training with masked terms moved the decomposition with nothing to copy: try another version of it, or stop?**

*What it is about.* E42's copies of a word-cipher fine-tune keep the cipher, but the model built from the subcomponents
Γ switches on at the trigger loses it. To see whether that is forced while Γ is frozen, we added two masked
reconstruction terms to teacher training. Before using it, we ran it with nothing to copy.

*What the evidence says.* With nothing to copy, the recipe broke the adversarial check on Pile text, reading 26 where
1.81 fails. The random-mask check stayed level, at 1.03. E37's masked recipe, which kept VPD's adversary in the loss,
failed the same control at 3.84. VPD trains its decomposition with an adversarial term and without a CI-masked one;
this recipe did the reverse, and asked the masked models to beat VPD's own reconstruction on 403 windows. Which of the
two moved it is untested.

*The options.* (a) The masked terms at the trigger only: a new amendment, one $0 cluster job. It asks for less, but
the trigger is where the masked models start furthest off, and there is still no worst-case term. (b) One masked
term alone: two $0 controls; says which term moves the decomposition. (c) Add VPD's adversarial term, matched to the
teacher: new code for 41-token windows and a review, then a $0 control; E37's result says it may not be enough.
(d) Stop here, and leave the card on the criterion as it is, with its option of trying masked teacher training first
marked as tried and failed. No option is recommended: two masked recipes failed this control, one with and one
without the adversary, and neither cause is tested.

*Why it matters.* It decides whether any frozen-Γ map can be asked to keep a new behaviour in the CI-masked model.
If none can, the card on the criterion chooses between reading A as it stands and updating Γ.

Links: E42, E37, `decision-e42-criterion`; this report; the plan's "Amendment 1".
