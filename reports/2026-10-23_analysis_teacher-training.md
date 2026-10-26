# Analysis: can the subcomponents copy a finished fine-tune from the fine-tuned model alone? Yes, on three tasks (E34)

*2026-10-23, analysis agent for the lead. Plan: `reports/2026-10-23_plan_teacher-training.md` with its "Lead's
decisions". Experiment `experiments/teacher_training`. Every number below is read from
`results/teacher_training_{controls,r1_t534_v369,r0_t352_v310,emoticon}.json`; the figures and `results/scorecard.json`
come from `plot.py`, which reads only those files and `predictions.json`. Vocabulary: `GOALS.md` → Terms.*

**In plain words.** We trained the existing subcomponents to copy six fine-tuned models' predictions, with no task
labels. Every copy kept the fine-tune's behaviour and stayed faithful, even when trained on ordinary web text alone.
For `GOALS.md` this is the first map of a finished update that keeps behaviour, faithfulness and Δ together on three
tasks; Amendment 1 adds that the copy learns the emoticon behaviour with every trigger removed from its text, by a
weight change that is not the fine-tune's.

> **Something unexpected happened.** We predicted that a student trained only on ordinary Pile text would not learn
> the emoticon behaviour, because its trigger is so rare there. It learnt it anyway: the held-out probability of the
> target reached 0.96 and 0.90 for the two emoticon teachers, against 0.004 for the unchanged model. Amendment 1, at
> the end, settled the first question this raised: the few trigger positions in the text are not what teaches it.
> With every row holding a trigger replaced, the copies still reach 0.94 and 0.89. It did not settle how the copy gets
> there: its weight change lines up with the fine-tune's less well than the adjoint split's does, so it seems to find
> a different weight change with the same behaviour.

## The answer

Teacher training works on all three tasks we tried. The subcomponents' read and write vectors were trained to match a
fine-tuned model's predictions, with Γ and Δ frozen. Trained on the task's own text, every copy kept the behaviour and
passed the faithfulness check at both checkpoints, including where the adjoint split had lost the behaviour. Trained on
ordinary text alone, every copy also got there by the last checkpoint. That includes the emoticon task, where we had
predicted failure. Plain SGD in place of Adam made the copy unfaithful on two tasks, as it does in ordinary component
training.

## What we were trying to test

The model is the released 67M Pile model with its VPD decomposition. Each matrix is $W = UV^\top + \Delta$. The
columns of $U$ and $V$ are the alive subcomponents' write and read vectors. Γ, which sets each subcomponent's causal
importance, and Δ are frozen throughout.

**Teacher training** is component training with a different loss. The alive $(U, V)$ get trainable offsets
$(dU, dV)$ that start at zero. The loss is $\mathrm{KL}(\text{teacher} \,\|\, \text{student})$, the KL from a frozen
teacher's next-token distribution to the student's. The teacher is the fine-tuned model $W_0 + \delta W$, with all 24
decomposed matrices changed. So the map consumes only $W_0$, $W_0 + \delta W$ and some text: no target token and no
task loss. Ordinary component training needs the task's labels.

**Teachers.** Six fine-tunes, regenerated in the job with E31's recipe at seed 0 and read at step 1000. On each of three
tasks one teacher was trained with plain SGD and one with Adam at 3e-5. The tasks are emoticon, where an emoticon
predicts the letter `o`, and the two ladder tasks r0 and r1, where a common word predicts a fixed next token. The
regenerated teachers matched E31's within 0.013 in P(target) and 0.007 nats in damage elsewhere.

**Arms**, each trained for 1000 steps and scored at steps 100 and 1000:

- **Task windows, Adam 1e-5.** The fine-tune's own 41-token training windows. The KL at the trigger weighs as much as
  the mean KL over the other 40 positions, as in E21's component training. This asks whether a finished update can be
  copied at all when the student sees the cue.
- **Pile only, Adam 1e-5.** Batches of eight 512-token Pile rows, from rows the task windows were not mined from, with
  a plain mean KL over every position. No task text and no knowledge of the trigger. This is the version that would
  apply to a model someone else post-trained.
- **Task windows, plain SGD** at E21's component-training rate, on the three SGD teachers only. This asks whether
  SGD's late failure in component training belongs to the optimiser or to the training targets.

**References on the same card.** Each teacher's do-nothing state puts all of $\delta W$ into Δ; it is the teacher
itself. Its scaled adjoint split writes $\delta W$ onto the subcomponents in one linear step.

**Per state we read:**

- **Held-out P(target)**, the target's probability at the trigger on held-out windows. Teachers read 0.94 to 1.0.
  The unchanged model reads 0.004 on emoticon, and 0.15 and 0.04 on r0 and r1.
- **Adversarial ratio**: the KL a 20-step mask adversary reaches on 256 Pile rows at the state, over what it reaches
  at $W_0$ in the same job. It reads 1 if nothing changed. The floor $f$, the spread of $W_0$'s own reading across three
  adversary seeds, was 0.075. A state passes at or below $1.5/(1+f) = 1.40$ and fails at or above
  $1.5(1+f) = 1.61$, provided P(target) is at least 0.9.
- **Teacher gap**: held-out $\mathrm{KL}(\text{teacher} \,\|\, \text{state})$ on the task windows, with the trigger
  weighting above. The teacher reads 0; $W_0$ reads the teacher's whole effect, 3.2 to 5.6 nats. We report it as a
  share of $W_0$'s gap.
- **Damage elsewhere**: held-out $\mathrm{KL}(W_0 \,\|\, \text{state})$ over the 40 non-trigger positions. $W_0$ reads
  0; the teachers read 0.008 to 0.076.
- **Δ's share** of the update's energy. It reads 1 for do-nothing and below $10^{-4}$ for every student and adjoint
  state. Δ is frozen, so this checks the measurement; it is not a finding.

**Controls, run first and alone.** The unchanged model through the harness had to read its known values. A teacher
equal to $W_0$ had to leave every offset exactly zero after 100 steps. Then a reachable teacher: E26's saved
component-training state on r0, which is itself a point the student can reach. The student had to close at least 90%
of that teacher's gap and land within 0.05 of its P(target). It also tests the optimiser, loss and batch order that
differ from how that state was made.

## What we expected, and why

- **On the task windows the copies pass**, for both teachers on all three tasks. Adam component training passes at
  steps 100 and 1000 on these tasks, and this is the same training with soft targets.
- **Where the adjoint split lost the behaviour, the copy keeps it.** On r0 and r1 the adjoint split of the SGD teacher
  kept P(target) 0.26 and 0.06. We predicted the copy would keep at least 0.9, at least 0.5 above the split.
- **On Pile text alone, emoticon is not learnt.** The emoticon trigger occurs about three times in 131 thousand Pile
  positions, so we predicted P(target) below 0.5 for both emoticon teachers.
- **On Pile text alone, r0 and r1 probably are learnt.** Their triggers are common words, about ten per batch, so we
  leaned towards learnt and faithful.
- **Plain SGD passes early and fails late on r0 and r1**, as SGD component training did, and passes throughout on
  emoticon. A late pass would have said the soft targets, not the optimiser, decide SGD's failure.
- **The copies of the Adam teachers pass where the teachers themselves do not.** Those teachers read inconclusive or
  fail on the L40, and their damage elsewhere is about three times component training's.
- **The copy inherits its teacher's damage elsewhere**, between half and twice it, because the loss asks it to copy
  the teacher everywhere.

## What we observed

### 1. Does the copy keep the behaviour that the adjoint split loses?

![behaviour](../experiments/teacher_training/figures/behaviour.png)

**Reading it.** Yes, on every task and teacher. Each panel is one task; circles are the SGD teacher and diamonds the
Adam teacher. The black points are the teachers, all above the dashed 0.9 line, and the grey points are their adjoint
splits. On r1 and r0 the adjoint split falls to 0.05 to 0.26, and on emoticon the Adam teacher's split falls to 0.10.
The blue points, the copies trained on task windows, sit with the teachers at 0.92 to 1.0 in every case. The orange
points, the copies trained on Pile text only, also clear 0.9 everywhere at step 1000. The lowest are 0.91 for the r0
Adam teacher and 0.90 for the emoticon Adam teacher. Plain SGD on task windows, in the table only, keeps the
behaviour too, at 0.96 or more.

| task | teacher | teacher P | adjoint P | task windows P | Pile only P | SGD on windows P |
|---|---|---|---|---|---|---|
| r1 | SGD 0.03 | 0.983 | 0.063 | 0.996 | 0.974 | 0.986 |
| r1 | Adam 3e-5 | 0.996 | 0.053 | 0.994 | 0.983 | — |
| r0 | SGD 0.03 | 0.942 | 0.257 | 0.942 | 0.919 | 0.962 |
| r0 | Adam 3e-5 | 0.936 | 0.235 | 0.921 | 0.908 | — |
| emoticon | SGD 0.01 | 0.998 | 0.613 | 0.999 | 0.960 | 0.998 |
| emoticon | Adam 3e-5 | 1.000 | 0.104 | 0.997 | 0.902 | — |

All at step 1000. At step 100 the task-window copies already read 0.95 to 0.99. The Pile-only copies read 0.80 to 0.85
on r0 and r1, and 0.74 and 0.34 on emoticon, so they were still learning.

### 2. Is the copy a faithful decomposition, where its teacher may not be?

![faithful](../experiments/teacher_training/figures/faithful.png)

**Reading it.** With Adam, yes: every copy passes at step 1000, on both kinds of text. With plain SGD, no on r0 and
r1. Each panel is one teacher; the black line is the teacher itself through the same check, and the grey band is the
inconclusive zone. The blue task-window copies sit at 0.93 to 1.30 at both steps. Three teachers sit in or near the
inconclusive band: r1 SGD at 1.59, r1 Adam at 1.52 and emoticon Adam at 1.50. Their copies read 1.00, 0.97 and 0.96.
The orange Pile-only copies pass at step 1000, at 0.90 to 1.37. The highest is the r0 Adam teacher's copy, close to
the pass line and level with its teacher's 1.35. The plain-SGD copies, in the table only, keep the behaviour but read
3.14 on r1 and 3.39 on r0 at step 1000, twice the fail line. On r0 SGD had already failed at step 100, at 1.89. On emoticon SGD passes
at both steps, at 1.11.

The positive control planned for this ratio did not fire. The r0 Adam teacher read 1.97, a fail, on the L40 in E21 and
E31, but 1.35, a pass, on this H100. The SGD copies' fails at 3.1 to 3.4 show the check can still read a fail in this
job. The card moved other teachers' readings too, in both directions: r1 SGD went from 1.13 to 1.59 and r0 SGD from
1.27 to 1.12.

| task | teacher | arm | step 100 | step 1000 | verdict at 1000 |
|---|---|---|---|---|---|
| r1 | SGD | teacher itself | — | 1.588 | inconclusive |
| r1 | SGD | task windows | 0.991 | 1.004 | pass |
| r1 | SGD | Pile only | 0.971 | 1.028 | pass |
| r1 | SGD | SGD on windows | 1.000 | 3.137 | fail |
| r1 | Adam | teacher itself | — | 1.523 | inconclusive |
| r1 | Adam | task windows | 1.007 | 0.968 | pass |
| r1 | Adam | Pile only | 1.036 | 1.131 | pass |
| r0 | SGD | teacher itself | — | 1.121 | pass |
| r0 | SGD | task windows | 0.953 | 1.296 | pass |
| r0 | SGD | Pile only | 1.131 | 1.125 | pass |
| r0 | SGD | SGD on windows | 1.888 | 3.385 | fail |
| r0 | Adam | teacher itself | — | 1.353 | pass |
| r0 | Adam | task windows | 0.929 | 1.081 | pass |
| r0 | Adam | Pile only | 1.089 | 1.366 | pass |
| emoticon | SGD | teacher itself | — | 0.984 | pass |
| emoticon | SGD | task windows | 1.039 | 0.942 | pass |
| emoticon | SGD | Pile only | 0.952 | 0.895 | pass |
| emoticon | SGD | SGD on windows | 1.112 | 1.111 | pass |
| emoticon | Adam | teacher itself | — | 1.499 | inconclusive |
| emoticon | Adam | task windows | 0.940 | 0.959 | pass |
| emoticon | Adam | Pile only | 0.947 | 1.049 | pass |

Adversarial ratio to $W_0$, seed-0 adversary. The step-100 Pile-only states read "below P 0.9" on P(target), not on
this ratio. Every stochastic ratio is 0.87 to 1.02.

### 3. How much of the teacher does the copy reach, and does it copy the teacher's side effects?

![gap_damage](../experiments/teacher_training/figures/gap_damage.png)

**Reading it.** The copies reach almost all of the teacher, and most copy its side effects, but two Adam copies drop
much of them. The top row is the teacher gap as a share of $W_0$'s, on a log axis; the dashed line is the 10% bar the
reachable-teacher control had to clear. Every task-window copy is under 4% at step 100 and under 3% at step 1000. The
Pile-only copies close more slowly; the emoticon Adam teacher's copy is at 27% at step 100 and 2.7% at step 1000. The
bottom row is damage elsewhere over the teacher's; the grey band is the half-to-double range we predicted. Four of
six task-window copies land at 0.53 to 1.48 of their teacher. The Adam copies on r1 and r0 sit at 0.55 and 0.53, just
inside. The emoticon Adam copy sits at 0.33. It keeps the behaviour and passes, with a third of its teacher's
damage elsewhere. Outside the prediction, the emoticon SGD copy trained on Pile text has 2.8 times its teacher's
damage. That teacher's own damage is tiny, 0.008 nats, so the copy's 0.022 is still small.

| task | teacher | teacher's gap at $W_0$ | copy's gap at 1000, task windows | share | teacher's damage | copy's damage | ratio |
|---|---|---|---|---|---|---|---|
| r1 | SGD | 5.49 | 0.079 | 1.4% | 0.056 | 0.051 | 0.91 |
| r1 | Adam | 5.62 | 0.022 | 0.4% | 0.076 | 0.042 | 0.55 |
| r0 | SGD | 3.18 | 0.014 | 0.4% | 0.047 | 0.045 | 0.95 |
| r0 | Adam | 3.17 | 0.083 | 2.6% | 0.059 | 0.031 | 0.53 |
| emoticon | SGD | 5.01 | 0.007 | 0.1% | 0.008 | 0.012 | 1.48 |
| emoticon | Adam | 5.09 | 0.028 | 0.6% | 0.055 | 0.018 | 0.33 |

For comparison, the adjoint split leaves 76% to 82% of the gap on r0 and r1, 55% for the emoticon Adam teacher and
15% for the emoticon SGD teacher.

### 4. Did the controls behave?

All three passed. The unchanged model read its known values, with an adversarial floor of 0.075. With the teacher equal
to $W_0$ the first loss and every offset after 100 steps were exactly zero. On the reachable teacher, the copy closed
all but 0.5% of the teacher's 2.99-nat gap. Its P(target) was 0.931 against the teacher's 0.921, and its adversarial
ratio 1.01 against the teacher's 1.13. Its offsets recovered 76% of the teacher's: the measure is one minus the squared
distance between the two offset sets over the teacher's squared offsets. So teacher training finds a nearby
decomposition with the same function, not the identical one.

## What we learned

On all three tasks we tried, teacher training maps a finished fine-tune onto the subcomponents with the behaviour
kept, the faithfulness check passed and Δ untouched. On r0 and r1 it keeps behaviour that the adjoint split loses. So
the split's losses there come from its linear one-step rewrite; the subcomponents can reach the behaviour. It needs no labels: ordinary Pile text was enough on every task by step 1000. That is the
condition under which the map would apply to a model post-trained by someone else. Copies of teachers that read inconclusive themselves pass. So a copy can be more faithful than the
same update left in Δ. Plain SGD on the same targets fails late on r0 and r1, as in component training. So that failure
belongs to SGD on these offsets, not to the training targets.

## How compelling is the evidence

- **One adversary budget.** Every verdict is a fixed-budget reading of a 20-step adversary, which E20 showed is far
  from converged. A pass means "no worse than $W_0$ under this budget", not "faithful under any attack".
- **One seed.** One training batch order and one adversary seed per state; one fine-tune seed per teacher.
- **Three tasks, one model.** Emoticon, r0 and r1, all a trigger followed by a fixed target, on the 67M model. Nothing
  here covers a fine-tune without a single trigger, or a larger model.
- **Δ is frozen by construction.** The students cannot put anything into Δ, so Δ's share of zero is not evidence.
- **Card.** The H100's floor of 0.075 sets the pass and fail lines at 1.40 and 1.61. The planned positive control for
  the ratio, the r0 Adam teacher, passed on this card. The SGD copies' fails stand in for it.
- **The copy differs from its teacher.** The reachable-teacher control recovered the function but only 76% of the
  offsets. A pass says the copy's own decomposition is faithful, not that the teacher's update has been decomposed
  exactly.
- **Damage elsewhere is read on the task windows only.** Pile damage enters only through the adversarial ratio.

## Surprises

**The emoticon behaviour was learnt from Pile text alone.** We filed that it would not be: P(target) below 0.5 for both
emoticon teachers. It reached 0.96 for the SGD teacher and 0.90 for the Adam teacher at step 1000, and both copies
passed the faithfulness check. At step 100 they read 0.74 and 0.34, so the behaviour arrived during training, not at the
start. Two causes were open, and this run could not separate them; Amendment 1, below, ruled out the first for the end state:

1. **The trigger is rare, not absent.** Arm B drew 8,000 rows, about 4.1 million positions. The rate at which emoticon
   windows were mined predicts about 107 trigger positions among them. A count of emoticon token pairs that also occur
   in the training windows finds 112. These are proxies: the trigger is defined by one subcomponent's causal
   importance, which needs the model. Each such position weighs 1/4096 of a batch's loss. That may be enough, since the
   teacher's prediction there differs from the unchanged model's by several nats.
2. **The teacher's changes on ordinary text carry the update.** A fine-tune of all 24 matrices also changes the
   predictions at non-trigger positions. Those changes may pin down enough of $\delta W$ that copying them reproduces the
   update, the trigger behaviour included.

The emoticon Adam copy's damage ratio of 0.33 is a narrow miss on one of six copies. Plain SGD failing r0 already at
step 100 was also not filed: we predicted a pass there, as SGD component training had.

## What next

1. **Separate the two causes of the Pile-only surprise.** Ran as Amendment 1, below. Rerun the Pile-only arm on emoticon with the trigger
   positions removed from the loss, or the rows holding them removed. Read the cosine between the student's weight
   change and $\delta W$, and the reproduced share. If the behaviour survives without the trigger positions and the
   change aligns with $\delta W$, the off-trigger fingerprint carries the update. Then teacher training could recover
   behaviours that the copying text never shows. If it does not survive, a handful of trigger positions is what it needs.
2. **Test the passes under a stronger adversary and more seeds.** Rerun the step-1000 copies and their teachers
   under a longer adversary, with a second batch order. If the copies still sit at $W_0$'s level where the teachers do
   not, teacher training is the commute line's working map.
3. **A fine-tune without a single trigger, then the larger model.** Both are on the backlog. Teacher training is the
   map to carry to them if the first two hold.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| C1 | the unchanged model reads its known values | `faith_score.w0_check` | floor 0.0748, seed-0 PGD within 1.3%, CI-masked within 0.05% | pass |
| C2 | a teacher equal to $W_0$ leaves the student at $W_0$ | step-1 loss and every offset exactly 0 | 0.0 and 0.0 | pass |
| C3 | a reachable teacher is recovered | gap at 1000 ≤ 10% of $W_0$'s; \|P − teacher's\| ≤ 0.05 | 0.5%; 0.931 vs 0.921; adversarial 1.01 vs 1.13; offset recovery 0.76 | pass |
| P1 | task-window copies of the SGD teachers pass at 100 and 1000 | `state_verdict` per task | r1 0.99, 1.00; r0 0.95, 1.30; emoticon 1.04, 0.94 | holds: all three |
| P2 | on r0 and r1 the copy keeps P ≥ 0.9 and ≥ the adjoint's + 0.5 | SGD teacher, step 1000 | r0 0.942 vs 0.257; r1 0.996 vs 0.063 | holds |
| P3a | Pile-only copies do not learn emoticon | P < 0.5, both teachers, step 1000 | 0.960, 0.902 | **reversed** |
| P3b | Pile-only copies learn r0 and r1, faithfully (lean) | SGD teacher, step 1000: P and verdict | r0 0.919 pass; r1 0.974 pass | holds: learnt and faithful |
| P4 | SGD copies pass at 100, fail at 1000 on r0, r1; pass both on emoticon | `state_verdict` | r1 1.00 pass, 3.14 fail; r0 1.89 fail, 3.39 fail; emoticon 1.11, 1.11 pass | misses as filed: r0 failed at 100; the step-1000 fails and emoticon as predicted |
| P5 | Adam-teacher copies pass where the teacher reads fail or inconclusive | step 1000 | r1 0.97 vs teacher 1.52; emoticon 0.96 vs 1.50; r0 teacher passed here, 1.35, so not tested | holds on the two tasks it applies to |
| P6 | every task-window copy's damage is 0.5 to 2 times its teacher's | step 1000, relative | 0.91, 0.55, 0.95, 0.53, 1.48, 0.33 | misses: emoticon Adam 0.33 |

Order of P6's reads: r1 SGD, r1 Adam, r0 SGD, r0 Adam, emoticon SGD, emoticon Adam. `results/scorecard.json` holds the
same verdicts by the filed rules; it scores P4 "fail" because the r0 step-100 reading differs.

**Corrections to the lead's reading.** The task-window copies' teacher gap is at most 3.6% of $W_0$'s gap, at r0 Adam
step 100, and at most 2.6% at step 1000. The lead's "≤ 0.11" matches the largest gap in nats, 0.114, not the share.
Everything else in the lead's reading matches the JSONs.

## Full readings

Teacher reproduction against E31, step 1000: P(target), damage elsewhere, size of change.

| task | teacher | this job | E31, L40 |
|---|---|---|---|
| r1 | SGD 0.03 | 0.983, 0.056, 0.0287 | 0.991, 0.049, 0.0286 |
| r1 | Adam 3e-5 | 0.996, 0.076, 0.0333 | 0.995, 0.078, 0.0342 |
| r0 | SGD 0.03 | 0.942, 0.047, 0.0300 | 0.944, 0.047, 0.0300 |
| r0 | Adam 3e-5 | 0.936, 0.059, 0.0341 | 0.949, 0.065, 0.0352 |
| emoticon | SGD 0.01 | 0.998, 0.008, 0.0122 | 0.998, 0.007, 0.0122 |
| emoticon | Adam 3e-5 | 1.000, 0.055, 0.0321 | 0.999, 0.055, 0.0315 |

Every student state, task by teacher by arm by step: adversarial and stochastic ratio, P(target), teacher gap in
nats, damage elsewhere, size of change.

| task | teacher | arm | step | adv | stoch | P | gap | damage | size |
|---|---|---|---|---|---|---|---|---|---|
| r1 | SGD | windows | 100 | 0.991 | 0.981 | 0.995 | 0.080 | 0.045 | 0.0129 |
| r1 | SGD | windows | 1000 | 1.004 | 0.957 | 0.996 | 0.079 | 0.051 | 0.0172 |
| r1 | SGD | Pile | 100 | 0.971 | 0.967 | 0.854 | 0.189 | 0.040 | 0.0122 |
| r1 | SGD | Pile | 1000 | 1.028 | 0.982 | 0.974 | 0.052 | 0.054 | 0.0218 |
| r1 | SGD | SGD | 100 | 1.000 | 0.952 | 0.986 | 0.083 | 0.064 | 0.0079 |
| r1 | SGD | SGD | 1000 | 3.137 | 0.969 | 0.986 | 0.031 | 0.068 | 0.0213 |
| r1 | Adam | windows | 100 | 1.007 | 0.966 | 0.993 | 0.050 | 0.032 | 0.0133 |
| r1 | Adam | windows | 1000 | 0.968 | 0.929 | 0.994 | 0.022 | 0.042 | 0.0209 |
| r1 | Adam | Pile | 100 | 1.036 | 0.872 | 0.834 | 0.214 | 0.052 | 0.0122 |
| r1 | Adam | Pile | 1000 | 1.131 | 0.882 | 0.983 | 0.023 | 0.061 | 0.0216 |
| r0 | SGD | windows | 100 | 0.953 | 0.973 | 0.950 | 0.060 | 0.043 | 0.0121 |
| r0 | SGD | windows | 1000 | 1.296 | 0.948 | 0.942 | 0.014 | 0.045 | 0.0177 |
| r0 | SGD | Pile | 100 | 1.131 | 0.957 | 0.795 | 0.205 | 0.040 | 0.0103 |
| r0 | SGD | Pile | 1000 | 1.125 | 0.967 | 0.919 | 0.052 | 0.046 | 0.0210 |
| r0 | SGD | SGD | 100 | 1.888 | 0.993 | 0.973 | 0.245 | 0.072 | 0.0091 |
| r0 | SGD | SGD | 1000 | 3.385 | 0.987 | 0.962 | 0.074 | 0.059 | 0.0202 |
| r0 | Adam | windows | 100 | 0.929 | 0.977 | 0.949 | 0.114 | 0.034 | 0.0124 |
| r0 | Adam | windows | 1000 | 1.081 | 0.928 | 0.921 | 0.083 | 0.031 | 0.0204 |
| r0 | Adam | Pile | 100 | 1.089 | 0.874 | 0.833 | 0.172 | 0.053 | 0.0114 |
| r0 | Adam | Pile | 1000 | 1.366 | 0.878 | 0.908 | 0.068 | 0.054 | 0.0206 |
| emoticon | SGD | windows | 100 | 1.039 | 0.996 | 0.993 | 0.015 | 0.010 | 0.0085 |
| emoticon | SGD | windows | 1000 | 0.942 | 0.987 | 0.999 | 0.007 | 0.012 | 0.0126 |
| emoticon | SGD | Pile | 100 | 0.952 | 1.001 | 0.743 | 0.443 | 0.010 | 0.0084 |
| emoticon | SGD | Pile | 1000 | 0.895 | 1.009 | 0.960 | 0.053 | 0.022 | 0.0159 |
| emoticon | SGD | SGD | 100 | 1.112 | 1.018 | 0.994 | 0.009 | 0.010 | 0.0037 |
| emoticon | SGD | SGD | 1000 | 1.111 | 0.998 | 0.998 | 0.008 | 0.013 | 0.0089 |
| emoticon | Adam | windows | 100 | 0.940 | 0.988 | 0.994 | 0.056 | 0.010 | 0.0090 |
| emoticon | Adam | windows | 1000 | 0.959 | 0.953 | 0.997 | 0.028 | 0.018 | 0.0184 |
| emoticon | Adam | Pile | 100 | 0.947 | 0.902 | 0.339 | 1.376 | 0.049 | 0.0106 |
| emoticon | Adam | Pile | 1000 | 1.049 | 0.904 | 0.902 | 0.136 | 0.058 | 0.0201 |

Teachers and adjoint states: adversarial ratio and verdict of the teacher, its gap at $W_0$, and the adjoint state's
P(target), gap, damage and size.

| task | teacher | teacher adv | verdict | gap at $W_0$ | adjoint P | adjoint gap | adjoint damage | adjoint size |
|---|---|---|---|---|---|---|---|---|
| r1 | SGD | 1.588 | inconclusive | 5.486 | 0.063 | 4.467 | 0.021 | 0.0211 |
| r1 | Adam | 1.523 | inconclusive | 5.622 | 0.053 | 4.612 | 0.042 | 0.0263 |
| r0 | SGD | 1.121 | pass | 3.182 | 0.257 | 2.419 | 0.022 | 0.0210 |
| r0 | Adam | 1.353 | pass | 3.172 | 0.235 | 2.480 | 0.041 | 0.0266 |
| emoticon | SGD | 0.984 | pass | 5.012 | 0.613 | 0.736 | 0.005 | 0.0085 |
| emoticon | Adam | 1.499 | inconclusive | 5.090 | 0.104 | 2.815 | 0.072 | 0.0240 |

Reachable-teacher control on r0: the teacher state read adversarial 1.133, P 0.921, damage 0.023, size 0.0174.
At steps 100 and 1000 the student's adversarial ratio read 0.939 and 1.013. Its P read 0.949 and 0.931. Its gap read
0.121 and 0.015 nats. The hook-path floor,
the gap with the student's offsets set to the teacher's, was 0.0007 nats.

Trigger count for the Pile-only arm, computed on CPU from the cached rows without the model. The arm's sampler drew
8,000 rows from Pile train rows 32,768 to 40,959; 5,083 were distinct. Emoticon token pairs that also occur in the
training windows: 112 positions. Any emoticon-like pair: 267, which includes non-emoticon uses of colons and
brackets. Mining-rate expectation: 107.

## Not run

- The Pile-only arm with trigger positions removed, and the cosine of the student's change against $\delta W$: ran
  as Amendment 1. The reproduced share was not run.
- A longer adversary than 20 steps on any state; more than one batch order, adversary seed or fine-tune seed.
- r2_t275_v2667: no scored teacher exists, and component training does not pass cleanly there.
- A mixed arm of task windows and Pile; the uniform-weight loss on task windows.
- SGD on Pile text; SGD on the Adam teachers.
- Teachers at steps 10 and 100; LoRA teachers.
- Γ trained alongside $(U, V)$.
- A fine-tune without a single trigger; a model post-trained by someone else; the larger model.
- Damage elsewhere on Pile text as its own number.

## Provenance

- RunPod H100 80GB HBM3, pod 01M5M9KV8G5C485ZTGC5Q8HWEC, run 67 min (12:15 to 13:22 pod time), $6.13. Files
  downloaded per task and checked by sha256 on both sides.
- Results: `experiments/teacher_training/results/teacher_training_controls.json` (sha256 `66588c84…`),
  `teacher_training_r1_t534_v369.json` (`49067d84…`), `teacher_training_r0_t352_v310.json` (`d6fde75c…`),
  `teacher_training_emoticon.json` (`158591c8…`), `results/large/run.log` (`41ca0440…`).
- Saved states: `results/large/<task>/A_{sgd,adam}_step1000.pt` and `results/large/r0_t352_v310/C3_student_step1000.pt`,
  bf16 offsets, about 60 MB each.
- Predictions: `results/predictions.json`, filed 2026-10-23 12:42:41, unedited (`25c1ca6e…`).
- Figures and scorecard: `experiments/teacher_training/figures/{behaviour,faithful,gap_damage}.png`,
  `results/scorecard.json`, from `plot.py`. The teacher-gap row of `gap_damage.png` was moved to a log axis at analysis. On 2026-10-23 (user) the figures were
  relabelled by training text (task text, Pile text) and the plain-SGD arm and the masked version of Amendment 1 were
  left out of them; their readings stay in the tables.
- Code: `experiments/teacher_training/run.py`, `src/commute/teacher.py`, `component_training.train(loss_fn=...)`.

**Hand-off.** A new session starts from this report, its Amendment 1 section, `reports/2026-10-23_plan_teacher-training.md`
and `backlog.md`. "What next" item 1 ran as Amendment 1. Its first decision is the order of Amendment 1's "What next":
close the partly-on loophole, a LoRA teacher, a second seed and a longer adversary, then the larger model.

## Amendment 1 (2026-10-23): is the Pile-only result the trigger's occurrences or the teacher's fingerprint?

*Design and filed predictions: `reports/2026-10-23_plan_teacher-training.md` → "Amendment 1";
`experiments/teacher_training/results/predictions_amend1.json`, unedited. Every number below is read from
`results/teacher_training_amend1.json`; the figure and `results/scorecard_amend1.json` come from
`plot.py --amend1`.*

**In plain words.** We retrained the Pile-only copies of the two emoticon fine-tunes after taking every emoticon
trigger out of their training text, either by ignoring the loss there or by swapping out the rows that held one. The
copies still learnt the emoticon behaviour almost as well as before, only more slowly at first, but their weight
changes point along the fine-tune's no better than before and less well than the adjoint split's. For `GOALS.md`
this says teacher training on ordinary text recovers what the update does without seeing where it acts, and that it
reaches that behaviour by a different weight change from the fine-tune's.

### The answer

The rare trigger positions in the Pile text are not what taught the copy the emoticon behaviour. With every row that
held a trigger replaced by a clean one, the copy of the SGD fine-tune still reached a target probability of 0.94, and
the copy of the Adam fine-tune 0.89. Both sit within about 0.02 of the copies that saw the triggers. What the triggers
did was speed up the first hundred steps. The copies do not reproduce the fine-tune's weight change: they find another
one with the same behaviour.

### What we were trying to test

E34's Pile-only copy, here called arm B, learnt the emoticon behaviour though we had predicted it would not. Two
explanations were open. Either the roughly one hundred trigger positions in its 8,000 Pile rows carried the behaviour,
or the teacher's changed predictions on ordinary text were enough to pin it down.

**The trigger.** A position counts as a trigger when subcomponent `h.2.mlp.down_proj:2359` has causal importance $g$
above 0.5 at $W_0$. This is the same rule that mined the task's training windows, and at such a position the model's
next token is the target. It was computed in the job from Γ, not from token ids: the emoticon tokens occur about 94,000
times in these rows, almost always with that subcomponent off. The trigger occurs at 90 positions in 63 of the 8,192
Pile rows. The 8,000 rows drawn during training hold 88 of them, in 58 draws.

**Three versions of arm B**, each trained for 1,000 steps from each of the two emoticon teachers with E34's recipe:

- **B**, exactly as in E34. For the SGD teacher this is also the reproduction check.
- **B with the trigger masked.** The same batches, with the loss weight set to zero at every trigger position. Rows
  where later positions can attend to the trigger stay in.
- **B with the trigger rows dropped.** Every row holding a trigger is replaced, at the same index, by the next clean
  row from 1,024 spare Pile rows. The student never sees a trigger above 0.5.

The teachers are E34's two emoticon fine-tunes, regenerated in the job. They matched E31's within 0.001 in target
probability and 0.0006 in size of change.

**What we read**, per state at step 1000 unless said otherwise:

- **Held-out P(target)**, the target's probability at the trigger on held-out task windows. The unchanged model reads
  0.004; the teachers read 0.998 and 1.000. We also read it at step 100, from the training loop's free held-out
  reading rather than the full harness.
- **The adversarial check**, as in E34. The ratio of the KL a 20-step mask adversary reaches at the state to what it
  reaches at $W_0$; 1 means no change. Pass at or below 1.40 provided P(target) is at least 0.9.
- **The cosine of the student's weight change with $\delta W$.** The student's change in each matrix is
  $(U + dU)(V + dV)^\top - UV^\top$. The cosine is taken over all 24 matrices' entries at once: 1 means the same
  direction as the fine-tune's $\delta W$, 0 means unrelated. Three references per teacher: E34's task-window copy at
  step 1000, the positive control, trained where the trigger is; the adjoint split of the same $\delta W$, a linear
  fit to it inside the subcomponents' span; and a random direction of the same shapes, the null, which should read
  about 0.

**Controls, run first.** The unchanged model through the harness had to read its known values. Then the SGD teacher's
plain arm B, retrained here, had to reproduce E34's reading before any new version ran.

### What we expected, and why

- **The copies with the triggers removed still learn**, for both teachers, with P(target) at least 0.9 and a pass.
  We leaned this way because 88 positions out of four million seemed too few to teach a behaviour, while a fine-tune
  of all 24 matrices changes predictions everywhere.
- **Masking the loss and dropping the rows give the same reading.** A difference would say that rows where later
  positions attend to the trigger carry the behaviour.
- **The copy without triggers points along $\delta W$**: a cosine at least half of the task-window copy's, and above
  0.2. If the copy learns the behaviour by copying the update's effects on ordinary text, its weight change should
  move towards the update itself.

### What we observed

#### 1. Is the behaviour learnt with every trigger removed?

![amend1](../experiments/teacher_training/figures/amend1.png)

**Reading it.** Yes, nearly as well as with the triggers. The left panel shows held-out P(target) for arm B with
and without the trigger rows, step 100 hollow and step 1000 filled, beside each teacher's own reading; the masked
version is in the table only. With the SGD teacher all
three end above the 0.9 line, at 0.96, 0.95 and 0.94. With the Adam teacher they end at 0.90, 0.91 and 0.89, so the
version with the rows dropped sits 0.008 under the line. All three Adam copies pass the adversarial ratio itself, at
1.05 to 1.20. The triggers matter early: at step 100 the plain copies read 0.74 and 0.34, the trigger-free ones
about 0.20 and 0.08. So the triggers speed up learning but are not needed for the end state.

#### 2. Does the student's weight change point along the fine-tune's?

**Reading it.** No more than it did with the triggers, and less than the adjoint split. The right panel of the same
figure plots each state's cosine with $\delta W$. Removing the triggers barely moves it, by 0.005 at most.
It stays near 0.08 for the SGD teacher and 0.22 for the Adam teacher. The task-window copy reads 0.245 and 0.285, and the adjoint split 0.450
and 0.382. The random direction reads 0.0002, so every cosine is far above chance. Yet the adjoint split, which points
along $\delta W$ best, keeps the least behaviour: in E34 it kept P(target) 0.61 and 0.10 on these teachers. The copies
keep 0.89 to 0.96 with cosines a fifth to a half as large. Per matrix, no trigger-free copy's cosine exceeds 0.37.

| teacher | version | P at 100 | P at 1000 | adversarial | stochastic | verdict | teacher gap | damage | cosine with δW |
|---|---|---|---|---|---|---|---|---|---|
| SGD 0.01 | B | 0.742 | 0.960 | 0.895 | 1.009 | pass | 0.053 | 0.022 | 0.083 |
| SGD 0.01 | B, trigger masked | 0.207 | 0.946 | 0.952 | 1.010 | pass | 0.074 | 0.022 | 0.080 |
| SGD 0.01 | B, trigger rows dropped | 0.195 | 0.937 | 1.011 | 1.008 | pass | 0.091 | 0.025 | 0.078 |
| Adam 3e-5 | B | 0.339 | 0.902 | 1.049 | 0.904 | pass | 0.135 | 0.058 | 0.219 |
| Adam 3e-5 | B, trigger masked | 0.078 | 0.911 | 1.199 | 0.903 | pass | 0.133 | 0.061 | 0.220 |
| Adam 3e-5 | B, trigger rows dropped | 0.080 | 0.892 | 1.152 | 0.904 | below P 0.9 | 0.153 | 0.056 | 0.218 |

Teacher gap and damage elsewhere in nats; the teachers' gaps at $W_0$ are 5.01 and 5.09. Size of change 0.016 for
every SGD-teacher copy and 0.020 for every Adam-teacher copy. Δ's share below $10^{-8}$ throughout.

| teacher | arm A cosine | adjoint cosine | random cosine | half of arm A | B-dropped cosine |
|---|---|---|---|---|---|
| SGD 0.01 | 0.245 | 0.450 | 0.0002 | 0.122 | 0.078 |
| Adam 3e-5 | 0.285 | 0.382 | 0.0002 | 0.142 | 0.218 |

#### 3. Did the controls behave?

Both passed. The unchanged model read its known values with an adversarial floor of 0.0748, the same as in E34. The
SGD teacher's plain arm B, retrained here, read an adversarial ratio of 0.8954 and P(target) of 0.9603, identical to
E34's to four places.

### What we learned

Cause one, the roughly ninety trigger positions in the Pile rows, is ruled out for the end state. Removing every
trigger row leaves P(target) within 0.01 to 0.02 of the copy that saw them, for both teachers. Cause two stands in a
narrower form: ordinary text carries enough of the teacher's changed predictions for the copy to learn the trigger
behaviour. But the copy does not rebuild $\delta W$. Its weight change lines up with $\delta W$ less than the adjoint
split's does, and the adjoint split keeps far less of the behaviour. So teacher training on ordinary text finds a
different weight change with the same function. We read this as a first answer to the
user's question of 2026-10-21: does component training find the fine-tune's solution? It is part b of `backlog.md`'s
"One rerun, two questions". It is a
reading, not a claim, until the cosine has the reference that entry asks for.

### How compelling is the evidence

- **A loophole in the dropped rows.** The threshold removed every full trigger. But 45
  positions with the trigger subcomponent partly on remain, with $g$ between 0.1 and 0.5. The largest left is 0.48. Those
  positions might still carry some of the behaviour.
- **No noise floor for P(target).** The Adam teacher's trigger-free copy is 0.008 under the 0.9 line and 0.010 under
  the plain copy. We have no seed-to-seed spread for this reading, so we cannot say whether that gap is real.
- **The cosine weighs every weight entry alike.** It counts directions that do nothing on the data as much as ones
  that do. And we have no reference for two fine-tunes of the same task: two fine-tune seeds might also read well
  below 1 against each other.
- **One seed, one adversary budget, one task.** One batch order and one 20-step adversary per state; emoticon only,
  whose trigger is rare on the Pile. On r0 and r1 the triggers are common words, so removing them would take out much
  of the text.

### Surprises

- **Masking and dropping split for the Adam teacher by the filed rule.** The masked copy read 0.911 and passed; the
  dropped copy read 0.892, under the line. The difference is 0.019 and has no noise floor; for the SGD teacher both
  versions sit in the same place.
- **The more faithful copy points along $\delta W$ less.** The SGD teacher's copies keep more behaviour than the
  Adam teacher's and pass more easily. Yet their cosine with $\delta W$ is a third as large. We have no explanation for
  this.

### What next

1. **Close the loophole.** Drop every row with $g$ above 0.1, not 0.5, and rerun the Adam and SGD teachers. If the
   copies still reach about 0.9, the trigger is not needed in any form; if they fall, the partly-on positions carried
   it.
2. **A LoRA teacher, a second seed and a longer adversary.** Run the same test on a LoRA fine-tune. Rerun the
   step-1000 copies with a second batch order and a longer adversary. If the copies still pass where their teachers do not,
   teacher training is the commute line's working map.
3. **Then the larger model.** Teacher training is the map to carry there if the first two hold.

### Scorecard, Amendment 1

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| C1 | the unchanged model reads its known values | `faith_score.w0_check` | floor 0.0748, as in E34 | pass |
| G1 | the SGD teacher's arm B reproduces E34 | ratio within 25% of 0.8954, P within 0.05 of 0.9603 | 0.8954, 0.9603 | pass |
| P1, SGD teacher | B-dropped learns without the trigger | P ≥ 0.9 and pass, step 1000 | 0.937, pass | learnt |
| P1, Adam teacher | B-dropped learns without the trigger | P ≥ 0.9 and pass, step 1000 | 0.892, below P 0.9; ratio 1.15 | between |
| P1, overall | both teachers learn | "both", "neither" or "between" | learnt, between | **between** |
| P2 | B-masked falls in B-dropped's category, each teacher | same P1 category | SGD learnt and learnt; Adam learnt, 0.911, and between, 0.892 | **fails as filed**: one of two |
| P3 | B-dropped's cosine with δW ≥ half arm A's and > 0.2, both teachers | cosine over 24 matrices | SGD 0.078 vs 0.122 and 0.2; Adam 0.218 vs 0.142 and 0.2; adjoint 0.450, 0.382 | **fails**: one of two, Adam passes |

`results/scorecard_amend1.json` holds the same verdicts by the filed rules.

**Corrections to the lead's reading.** P2: the lead read "B-masked reads like B-dropped for both teachers". By the
filed rule it does not for the Adam teacher: the masked copy is "learnt" at 0.911 and the dropped copy "between" at
0.892. Every number in the lead's reading matches the JSON.

### Not run, Amendment 1

- Dropping rows with the trigger subcomponent partly on, $g$ above 0.1: 45 such positions remain.
- A second batch order; a longer adversary; step-100 harness readings.
- r0 and r1, whose triggers are common words, so the test is weaker there.
- The same test on a LoRA teacher.
- The reproduced share of the student's change; an input-weighted cosine; the cosine between two fine-tune seeds.
- Pile rows holding the trigger's tokens with the subcomponent off; they stay in both versions.

### Provenance, Amendment 1

- RunPod H100 80GB HBM3, pod 01M5MQS6KRTCPHD83KXFQWWTTG, run 19.6 min, $3.26. Files checked by sha256 on both sides.
- Results: `experiments/teacher_training/results/teacher_training_amend1.json` (`55b73144…`), log
  `results/large/run_amend1.log` (`3a017002…`).
- Predictions: `results/predictions_amend1.json`, unedited (`896d58e8…`).
- Saved students: `results/large/emoticon/{B,Bmasked,Bdropped}_{sgd,adam}_step1000.pt`, bf16 offsets, about 60 MB
  each.
- Figure and scorecard: `figures/amend1.png`, `results/scorecard_amend1.json`, from `plot.py --amend1`.
- Code: `experiments/teacher_training/run.py --amend1`, `data.subcomponent_g`, `data.replace_rows`,
  `finetune.cosines`.
