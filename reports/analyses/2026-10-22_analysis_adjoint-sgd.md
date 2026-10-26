# Analysis: does the adjoint split of a plain-SGD fine-tune keep the behaviour? On emoticon only (E31)

*2026-10-22, E31 agent. Plan: `reports/plans/2026-10-22_plan_adjoint-sgd.md` and its Agent notes. Experiment
`experiments/adjoint_sgd`. Every number below is read from `results/adjoint_sgd.json`, which merges the shard files
`results/adjoint_sgd_{A,B,C,D,E}.json`. The figures come from `plot.py`, which reads only that file. Vocabulary:
`GOALS.md` → Terms.*

**In plain words.** We fine-tuned the model with plain SGD and with Adam on three tasks, and wrote each update onto
the existing subcomponents with the adjoint split. On the emoticon task the SGD update kept most of its behaviour on a
second seed and stayed faithful, but on the two other tasks it lost the behaviour as fast as the large Adam update.
For `GOALS.md`, this split of an SGD fine-tune meets the weak version on one task only, for reasons not yet known.

> **Something unexpected happened that I cannot explain.** We predicted that on the two new tasks the SGD update's
> adjoint split would keep at least half the behaviour at step 1000, well above the large Adam update's. It kept 0.07
> on one task and 0.26 on the other, against 0.05 and 0.23 for Adam. The SGD rate on those tasks was three times the
> emoticon rate, so task and rate are confounded here.

## The answer

On emoticon, the SGD fine-tune's adjoint state held its target probability near 0.57 at step 1000 on a second seed. The
small Adam fine-tune fell to 0.21 there, so E22's surprise replicates on that task. On the two other tasks the SGD state
fell to 0.07 and 0.26, no better than the large Adam state. Every adjoint state passed the adversarial check with Δ
fixed, and the SGD state's pass held under an eight-times-longer attack. The reproduced share did not explain the emoticon
difference: SGD and small Adam reproduced about the same share of their updates.

## What we were trying to test

The model is the released 67M Pile model with its VPD decomposition. Γ and Δ stay frozen throughout. Each fine-tune
trains all 24 decomposed matrices for 1000 steps with the arm-A loss: cross-entropy on the target token at the trigger
plus KL to the original model elsewhere. Its update $\delta W$ is read at steps 10, 100 and 1000.

Each update is written onto the subcomponents in two ways:

- **Do-nothing**: the whole $\delta W$ goes into Δ. The reference.
- **Adjoint split**: each subcomponent's write vector moves by $\delta W V$ and its read vector by $\delta W^\top U$, so the
  weight change is $K(\delta W) = UU^\top\delta W + \delta W VV^\top$, scaled per matrix by the $c$ that best matches
  $\delta W$ on the data. Δ does not change. The question is how much behaviour survives that rewrite.

Per state we read four things:

- **Held-out P(target)** at the trigger. The fine-tune itself reads about 1; the unchanged model reads about 0.004.
- **Adversarial ratio**: the damage a 20-step adversary finds on Pile text, divided by the same job's unchanged model.
  It reads 1 for the unchanged model. On the L40 a ratio at or below 1.24 is a pass.
- **Δ's share** of the update's energy. The adjoint split should read 0, which checks that the measurement reads 0.
- **Reproduced share** $R$: the fraction of the update's input-weighted energy that $cK(\delta W)$ reproduces, pooled over
  matrices. It reads 1 when the split's change equals the update and 0 when it reproduces none of it.

The arms:

- **Emoticon, seed 0**: Adam at 3e-5, SGD at 0.01 and Adam at 3e-6, rerun first as the control. They had to match E22's
  target probabilities within 0.05, and the large Adam update's $R$ had to match E14's within 0.03.
- **Emoticon, seed 1**: the same three. This rules out a lucky batch order.
- **r1_t534_v369 and r0_t352_v310**: SGD at the rate E4's rule picks on step 10, and Adam at 3e-5. This asks whether the
  effect belongs to the optimiser or to the emoticon task. The rule picked 0.03 on both tasks, because 0.01 reached
  only 0.84 at step 10 there.
- **Budget sweep**: the SGD and small Adam emoticon states at step 1000 under 20, 40, 80 and 160 adversary steps. This
  asks whether the 20-step pass is an artefact of a short attack.

## What we expected, and why

E22 found that the SGD emoticon state kept 0.61 at step 1000 while the small Adam state fell to 0.24. We expected the
second seed to repeat that. We also expected the other tasks to show it, with medium-low confidence, on the view that it
is a property of the optimiser. The proposed mechanism: one plain SGD step's adjoint split is exactly one
component-training step, so an SGD update should be mostly reproducible by the adjoint split. Adam rescales each
coordinate, so it should not be. That predicted a reproduced share of at least 0.8 for every SGD update, above every Adam
update on the same task. It also predicted that kept behaviour would rank with the reproduced share. Finally we
expected the longer attack to find no more damage than on the unchanged model, as it did for the large Adam state.

## What we observed

### Does the SGD state keep the behaviour on a second seed and on other tasks?

![held-out P(target) along training](../../experiments/adjoint_sgd/figures/steps.png)

**Reading it.** On emoticon, yes; on the other two tasks, no. In the two left panels the blue SGD line ends at 0.60
and 0.57 at step 1000. The green small-Adam line ends at 0.24 and 0.21, and the orange large-Adam line at 0.10 and 0.09.
In the two right panels the blue line falls with the orange one, to 0.07 against 0.05 on r1 and 0.26 against 0.23
on r0. The dashed lines show the fine-tunes themselves near 0.95 to 1 throughout, so the behaviour is lost in the
split, not in training. The SGD rate on those two tasks was 0.03, and the update grew to about the large Adam update's
size by step 1000. So this run cannot say whether the task or the larger rate removed the effect.

| step 1000 | emoticon s0 | emoticon s1 | r1_t534_v369 | r0_t352_v310 |
|---|---|---|---|---|
| SGD adjoint P(target) | 0.60 | 0.57 | 0.07 | 0.26 |
| Adam 3e-6 adjoint | 0.24 | 0.21 | — | — |
| Adam 3e-5 adjoint | 0.10 | 0.09 | 0.05 | 0.23 |
| SGD update size | 0.0122 | 0.0123 | 0.0286 | 0.0300 |
| Adam 3e-5 update size | 0.0315 | 0.0322 | 0.0342 | 0.0352 |

The size is the largest matrix's relative change, $\max_k \|\delta W_k\| / \|W_k\|$.

### Is the reproduced share what separates them?

![adjoint P(target) against reproduced share](../../experiments/adjoint_sgd/figures/keep.png)

**Reading it.** No. The reproduced share falls along training for every update, and kept behaviour falls with it. That
is why the two rank together across all 30 states, with a rank correlation of 0.81. But the share never reaches 0.8
for any update; SGD's sits between 0.33 and 0.63. At a fixed step it does not separate SGD from small Adam on emoticon.
At step 1000 SGD reproduces 0.48 and small Adam 0.43, yet SGD keeps 0.60 and Adam 0.24; at step 10 small Adam
reproduces more. Within a single step the rank correlation is 0.62 to 0.69, about the same as the correlation of kept
behaviour with a smaller update size. So the share tracks training length, not the optimiser difference.

The share also reads below 1 for an update lying wholly inside the subcomponents' spans, because $K$ counts the block
inside both spans twice and $c$ is one number per matrix. It is therefore not the outside share that E22 reported at
0.03 to 0.06 for SGD.

### Do the adjoint states stay faithful, and does that survive a longer attack?

![adversarial ratio per state](../../experiments/adjoint_sgd/figures/floor.png)

**Reading it.** Yes. Every one of the 30 adjoint states read between 0.93 and 1.08, inside the pass region on every
task and step. Do-nothing read up to 2.29 for the large Adam update. Δ's share stayed below 1e-4 in every
adjoint state, so Δ did not absorb the update.

![budget sweep](../../experiments/adjoint_sgd/figures/budget.png)

**Reading it.** The SGD emoticon state's pass holds under a longer attack. At 20 to 160 adversary steps its damage was
1.00 to 1.01 times the unchanged model's at the same budget. The unchanged model's second adversary seed reads 1.07
to 1.21 times its first, so the SGD state sits well inside the attack's own noise. The small Adam state's ratio
stays between 1.01 and 1.02 under every longer attack. That is also inside the attack's own noise, which reads
1.13 to 1.21 at those budgets. No prediction reads this number, so it is reported, not scored.

## What we learned

The adjoint split of a plain-SGD fine-tune keeps the emoticon behaviour on two seeds while staying faithful with Δ
fixed. On two other single-token tasks, at the SGD rate the selection rule chose there, it loses the behaviour as the
large Adam update does. So the 2026-10-21 decision's move-back condition is met on one task only, and the main-line
decision should not move on this evidence. The proposed mechanism fails as stated: the reproduced share is not higher
for SGD and does not separate SGD from small Adam at matched steps.

## How compelling is the evidence

- Three tasks, all single-token trigger and target; two seeds on emoticon, one elsewhere.
- Target probability is read on held-out windows at the trigger only. Off-target damage is covered only by the
  adversarial ratio.
- The adversarial verdict is one 20-step adversary family, extended to 160 steps for one state. The unchanged model's
  floor reads 0.21 on this card.
- The other-task comparison confounds task with SGD rate: 0.03 there, 0.01 on emoticon.
- Kept behaviour and reproduced share are correlated readings; neither was intervened on.

## Surprises

- **The other-task prediction reversed.** SGD's adjoint state kept 0.07 and 0.26, about the large Adam update's 0.05
  and 0.23, against a predicted at least 0.5 and 0.2 above Adam. The cause is not known; task and rate are confounded.
- The reproduced share of SGD updates is far below the predicted 0.8 and no higher than Adam's at step 10.

## What next

1. SGD at 0.01 on r1_t534_v369 and r0_t352_v310 for 1000 steps, adjoint split at each step. If it keeps at least half
   the behaviour, the effect belongs to the rate or update size; if not, to the task.
2. More single-token tasks at emoticon's SGD rate, to see how often the effect appears at all.
3. A measure of what distinguishes the emoticon SGD update from small Adam's at step 1000, since the reproduced share
   does not.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G0 | unchanged model at its known values | `w0_check` per job | floor 0.209 in all five jobs | pass |
| G1 | reproduce E22 on emoticon seed 0 | adjoint P within 0.05 of E22 at each step | Adam 3e-5 0.93/0.57/0.10; SGD 0.72/0.71/0.60; Adam 3e-6 0.69/0.48/0.24 | pass |
| G2 | large Adam R reproduces E14 | within 0.03 of 0.58/0.42/0.31 | 0.576/0.424/0.313 | pass |
| C_delta | Δ share reads 0 | every adjoint state ≤ 1e-3 | max 9.9e-5 | pass |
| C_budget_w0 | budget sweep's W0 at E20's value | 20 steps within 5% of 0.6604 | 0.6604 | pass |
| P1 | seed 1 SGD keeps; small Adam ≥ 0.2 lower | SGD P ≥ 0.5, Δ ≤ 1e-3, pass; Adam ≤ SGD − 0.2 | SGD 0.565, pass, Δ 7e-7; Adam 0.205 | **pass** |
| P2 | SGD keeps on r1 and r0, ≥ 0.2 above Adam 3e-5 | per task SGD P ≥ 0.5 and ≥ Adam + 0.2 | r1 0.066 vs 0.053; r0 0.262 vs 0.235 | **fail ("neither")** |
| P3a | SGD R ≥ 0.8 everywhere, above every Adam R | each SGD state | SGD R 0.33–0.63; below Adam 3e-6 at step 10 | **fail** |
| P3b | kept ratio ranks with R | Spearman ≥ 0.7 over all states | 0.81, n = 30 | **pass** |
| P4 | SGD state's pass holds at longer budgets | ratio ≤ 1.241 at 40/80/160 | 1.012 / 1.000 / 1.001 | **pass** |

P3b passes as filed, but the within-step correlation of 0.62 to 0.69 shows it mostly reflects training length.

### All states (adjoint P(target), R, adversarial ratio of adjoint and do-nothing)

| state | step | size | R | adjoint P | adjoint ratio | do-nothing ratio |
|---|---|---|---|---|---|---|
| emoticon s0 Adam 3e-5 | 10 / 100 / 1000 | 0.011 / 0.021 / 0.031 | 0.58 / 0.42 / 0.31 | 0.93 / 0.57 / 0.10 | 1.08 / 0.98 / 0.95 | 2.18 / 1.12 / 1.39 |
| emoticon s0 SGD 0.01 | | 0.002 / 0.005 / 0.012 | 0.58 / 0.63 / 0.48 | 0.72 / 0.71 / 0.60 | 1.03 / 1.01 / 0.98 | 1.14 / 1.08 / 1.02 |
| emoticon s0 Adam 3e-6 | | 0.002 / 0.004 / 0.008 | 0.65 / 0.60 / 0.43 | 0.69 / 0.48 / 0.24 | 1.02 / 0.99 / 0.99 | 1.09 / 1.09 / 1.06 |
| emoticon s1 SGD 0.01 | | 0.002 / 0.005 / 0.012 | 0.57 / 0.63 / 0.47 | 0.71 / 0.78 / 0.56 | 1.03 / 1.00 / 0.99 | 1.11 / 1.15 / 1.00 |
| emoticon s1 Adam 3e-6 | | 0.002 / 0.004 / 0.008 | 0.66 / 0.60 / 0.42 | 0.69 / 0.52 / 0.21 | 1.00 / 1.00 / 0.99 | 1.09 / 1.06 / 1.02 |
| emoticon s1 Adam 3e-5 | | 0.011 / 0.020 / 0.032 | 0.60 / 0.42 / 0.30 | 0.95 / 0.46 / 0.09 | 1.02 / 0.95 / 0.97 | 1.71 / 1.08 / 2.26 |
| r1 SGD 0.03 | | 0.006 / 0.012 / 0.029 | 0.50 / 0.50 / 0.33 | 0.55 / 0.26 / 0.07 | 1.08 / 0.93 / 1.00 | 1.61 / 0.99 / 1.13 |
| r1 Adam 3e-5 | | 0.013 / 0.022 / 0.034 | 0.45 / 0.34 / 0.31 | 0.59 / 0.16 / 0.05 | 1.00 / 1.03 / 1.02 | 2.29 / 1.45 / 1.55 |
| r0 SGD 0.03 | | 0.005 / 0.013 / 0.030 | 0.53 / 0.48 / 0.34 | 0.61 / 0.46 / 0.26 | 0.99 / 1.00 / 1.00 | 1.00 / 1.09 / 1.27 |
| r0 Adam 3e-5 | | 0.013 / 0.025 / 0.035 | 0.49 / 0.36 / 0.30 | 0.68 / 0.45 / 0.23 | 1.01 / 0.96 / 0.97 | 1.56 / 1.11 / 1.97 |

SGD rate grids at step 10, P(target) and KL elsewhere: r1 0.01 → 0.84 / 0.23, 0.03 → 0.97 / 0.14; r0 0.01 → 0.84 /
0.16, 0.03 → 0.91 / 0.08. Budget sweep KL, unchanged model seed 0: 0.660 / 0.682 / 0.701 / 0.709; SGD state: 0.663 /
0.690 / 0.701 / 0.709. The budget job retrained the SGD and small Adam emoticon fine-tunes; their step-1000 P(target),
KL elsewhere and size match shard A's to the printed precision.

## Not run

- SGD at 0.01 on r1 and r0; SGD with momentum; more seeds on r1 and r0; other task families; adversaries beyond the
  sign-step family; the strong version.

## Provenance

- MATS cluster, `debug` QoS, one L40 per job, $0. Jobs: A 29503 (2817 s), B 29565 (3041 s), C 29566 (2173 s),
  D 29567 (2158 s), E 29568 (crashed after about 28 min). Rerun of E: 29596, finished in about 39 min. The earlier RunPod attempt
  finished only its W₀ control on an H100 (`results/adjoint_sgd_pod_w0only.json`); remote-kernels did not connect this session.
- Code: `experiments/adjoint_sgd/run.py`, which gained `shard` and `merge` stages for this run; `split.reproduced_share`. Codex adversarial review run before
  the jobs; its one finding was fixed: the other-task scoring counted a task with no valid SGD rate as scored.
- Files: `results/adjoint_sgd_{A,B,C,D,E}.json`, `results/adjoint_sgd_E_crashed29568.json`, `results/adjoint_sgd.json`
  (sha256 7752c540…), logs `results/large/full_{A..E}.log`, `cluster/logs/commute-adjoint-sgd-*.out`.

---

## Amendment 1: is it the task or the SGD rate?

*2026-10-23, analysis agent. Plan: `reports/plans/2026-10-22_plan_adjoint-sgd.md` → "Amendment 1"; predictions filed in
`results/predictions_amend1.json` before the run. Numbers come from `results/adjoint_sgd_amend1.json` for the three new
fine-tunes and from `results/adjoint_sgd.json` for E31's. The figure is `size.png` from `plot.py`, which reads only those two files.*

**In plain words.** E31 left the task and the SGD learning rate tangled, so we swapped them: word tasks at the slow
rate, emoticon at the fast one. Neither alone decides what the adjoint split keeps; it keeps less as the update grows,
and at one update size the tasks still differ widely. For `GOALS.md`, this split of an SGD fine-tune keeps behaviour
only while the update is small, by a task-dependent margin, so it does not meet the weak version in general.

> **Both filed predictions missed.** We predicted that the task alone sets the outcome. On the word tasks at the slower
> rate we expected the split to keep no more than before. One task did; the other kept somewhat more, so the rule scores
> the pair "mixed". On emoticon at the faster rate we expected it to keep at least half the behaviour. It kept 0.17,
> which the rule scores as "rate". Neither prediction is re-filed or softened below.

### The answer

The adjoint split keeps less behaviour the larger the SGD update grows, on every task. Emoticon at the faster rate
reached a two-times-larger update by step 1000 and kept 0.17 instead of 0.60. Its own step-100 state, at the size the
slower run reaches at step 1000, kept the same 0.60. But size is not the whole story. At about the same size the three
tasks kept 0.60, 0.39 and 0.13. This is a reading from one seed per new fine-tune, not a finding.

### What we were trying to test

The setup is E31's, unchanged. Each fine-tune trains all 24 decomposed matrices with plain SGD for 1000 steps on one
task, with the arm-A loss. Γ and Δ stay frozen. At steps 10, 100 and 1000 the update $\delta W$ is written onto the
subcomponents by the scaled adjoint split, and Δ does not change.

Two numbers are read per state:

- **Keep**: the adjoint state's held-out P(target) at the trigger divided by the fine-tune's own at the same step. It
  reads about 1 for do-nothing, which leaves the whole update in Δ, and near 0 if the split loses the behaviour.
- **Update size**: the largest relative change of any matrix, $\max_k \|\delta W_k\| / \|W_k\|$. It is reported, not scored.

E31 had emoticon at SGD 0.01 and the two word tasks, r1_t534_v369 and r0_t352_v310, at 0.03. This amendment adds the
other two corners: both word tasks at 0.01 and emoticon at 0.03. The word-task corners ask whether a slower rate brings
the behaviour back there. The emoticon corner asks whether a faster rate removes it on the task where it was seen.

Each job first checked the unchanged model against its known values. It then retrained the 10-step fine-tune, which
had to match E31's earlier reading within 0.01. Both passed in all three jobs. Every fine-tune reached P(target) of at
least 0.92 by step 1000, so every corner was scorable.

### What we expected, and why

We predicted "task" for both, from a path argument. SGD at a third of the rate for 1000 steps travels roughly as far
as the faster rate does in its first 330 steps. There E31's faster runs had already fallen to about 0.16 and 0.38. So
we expected the word tasks at 0.01 to keep no more than 0.17 and 0.38. For emoticon at 0.03 the same argument says 1000
steps is like 3000 at the slower rate, and the slower run's keep fell only slowly. So we expected it to keep at least 0.5.

### What we observed

#### Does the update's size set how much the split keeps?

![keep against update size](../../experiments/adjoint_sgd/figures/size.png)

**Reading it.** Within a task, mostly yes; across tasks, no. Each line is one fine-tune's steps 10, 100 and 1000, coloured
by task, with circles for SGD 0.01 and triangles for 0.03. Above a size of about 0.005, the circles and triangles of
one colour fall on one downward curve. Emoticon at 0.03 kept 0.60 at step 100, where its size matched the slower run's
step 1000, which also kept 0.60. It then fell to 0.17 as its update kept growing. At a size near 0.013 the three
colours sit far apart, emoticon near 0.6, r0 near 0.4 and r1 near 0.13. The faint crosses are E31's Adam updates and do
not follow the SGD curves. Adam 3e-5 kept 0.93 on emoticon at a size where SGD kept 0.6. So size orders SGD states
within a task only.

| step 1000 | emoticon | r0_t352_v310 | r1_t534_v369 |
|---|---|---|---|
| SGD 0.01: keep / size | 0.60 / 0.012 (seed 1: 0.57 / 0.012) | **0.39 / 0.015** | **0.13 / 0.014** |
| SGD 0.03: keep / size | **0.17 / 0.027** | 0.28 / 0.030 | 0.07 / 0.029 |
| fine-tune P(target), 0.01 / 0.03 | 0.998 / 0.999 | 0.928 / 0.944 | 0.983 / 0.991 |

Bold cells are new in this amendment. The rest are E31's.

#### Does the adjoint split stay faithful in the new corners?

Yes. Every new adjoint state passed the adversarial check, with ratios to the unchanged model between 0.93 and 1.33.
The highest was r1_t534_v369's step-10 state at 1.33. At steps 100 and 1000 every state read 0.93 to 1.00. Δ's share of
the update stayed below 7e-5 in every adjoint state, so Δ did not absorb it.

### What we learned

The adjoint split of an SGD fine-tune keeps the behaviour only while the update is small. How small depends on the task.
Emoticon still keeps about 0.6 at a size near 0.012, where r1_t534_v369 is already down to 0.13. E31's emoticon
result is therefore partly a small-update result, and the word-task failure is partly a large-update one. Neither
reading meets the weak version of `GOALS.md` in general, since the fine-tunes it should cover grow past these sizes.

### How compelling is the evidence

- One seed per new fine-tune. The only seed-to-seed spread measured is emoticon's 0.04 at SGD 0.01.
- Size is the largest matrix's relative change. A pooled or per-layer size might order the states differently.
- Each size has one state per fine-tune, read at one step. No two rates were stopped at the same size, so
  "size, not path" is read off the figure, not tested.
- Adam states do not follow the SGD curves, so size alone is not a sufficient summary of an update.
- The three jobs ran on an L40S, not the planned L40. The unchanged model's two adversary seeds differed by 0.05 there,
  against 0.21 on the L40. The 10-step gate reproduced E31 within 0.001 on every task.
- P(target) is read only at the trigger, and damage elsewhere only through the 20-step adversary.

### Surprises

- **Both filed predictions missed.** The task-alone prediction failed in both directions. Emoticon lost its behaviour at
  the faster rate, and r0_t352_v310 kept more at the slower rate, 0.39 against 0.28.
- r1_t534_v369's step-10 state at 0.01 read 1.33 on the adversarial check. It passes on this card's floor but would be
  inconclusive on the L40's wider one.

### What next

1. A size-matched comparison: two rates stopped at the same update size on one task, two seeds each. If keep matches,
   size is the variable; if not, the path matters.
2. Keep against size on two or three more tasks at SGD 0.01, read at more steps. This tells whether emoticon's level is
   typical or an outlier.
3. Something that tells the tasks apart at matched size, such as how many subcomponents the fine-tune switches on.

### Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G0 | unchanged model at its known values | `w0_check`, each job | pass in all three; floor 0.046 | pass |
| A1_G1 | 10-step fine-tune reproduces E31 | P(target), KL elsewhere within 0.01 | r1 0.837 / 0.228; r0 0.843 / 0.158; emo 0.995 / 0.037 | pass |
| C_delta | Δ share reads 0 | every adjoint state ≤ 1e-3 | max 6.7e-5 | pass |
| A1 | word tasks at 0.01 keep no more than at 0.03 ("task") | per task: rate if ≥ 0.5 and ≥ 0.03 keep + 0.2; task if ≤ 0.03 keep + 0.1; else partial | r1 0.130 vs ≤ 0.167 → task; r0 0.392 vs ≤ 0.378 → partial | **mixed: missed** |
| A2 | emoticon at 0.03 keeps ≥ 0.5 ("task") | rate if ≤ 0.404; task if ≥ 0.5; else partial | 0.166 | **rate: missed** |

All step-1000 adjoint verdicts pass; fine-tune P(target) at step 1000 is 0.983, 0.928 and 0.999.

#### All new states

| cell | step | size | fine-tune P | adjoint P | keep | R | adjoint ratio | do-nothing ratio |
|---|---|---|---|---|---|---|---|---|
| r1_t534_v369 SGD 0.01 | 10 / 100 / 1000 | 0.003 / 0.008 / 0.014 | 0.84 / 0.99 / 0.98 | 0.31 / 0.41 / 0.13 | 0.37 / 0.42 / 0.13 | 0.47 / 0.56 / 0.45 | 1.33 / 0.94 / 0.97 | 2.56 / 0.98 / 0.96 |
| r0_t352_v310 SGD 0.01 | | 0.003 / 0.008 / 0.015 | 0.84 / 0.91 / 0.93 | 0.49 / 0.54 / 0.36 | 0.59 / 0.60 / 0.39 | 0.50 / 0.55 / 0.45 | 1.08 / 0.97 / 0.98 | 1.76 / 1.05 / 1.02 |
| emoticon SGD 0.03 | | 0.004 / 0.010 / 0.027 | 1.00 / 0.99 / 1.00 | 0.82 / 0.60 / 0.17 | 0.83 / 0.60 / 0.17 | 0.58 / 0.59 / 0.33 | 1.01 / 1.00 / 0.93 | 1.03 / 1.08 / 1.03 |

The pass line on this card is 1.43 and the fail line 1.57, from the same-job floor of 0.046.

### Not run

- More seeds on any corner; a size-matched pair of rates; other rates; more tasks.
- SGD with momentum; smaller-rate Adam on the word tasks; a pooled update size; the strong version.

### Provenance

- RunPod, one L40S, the three corners in sequence, $1.01. The cluster `debug` queue did not start r1 within 10 minutes,
  so job 30502 was cancelled. Each corner took about 1000 s.
- Code: `run.py amend1 <cell>` and `amend1_merge`, calling E31's `arm()` unchanged; `plot.py` → `fig_size`.
- Files: `results/adjoint_sgd_amend1_{r1,r0,emo}.json`, merged `results/adjoint_sgd_amend1.json` (sha256 a0b36394…);
  logs `results/large/full_amend1_{r1,r0,emo}.log` and `results/large/pod_stage_amend1_{r1,r0,emo}.log`.

---

## Amendment 2: does the size reading hold on a second seed?

*2026-10-23, analysis agent. Plan: `reports/plans/2026-10-22_plan_adjoint-sgd.md` → "Amendment 2"; predictions filed in
`results/predictions_amend2.json` before the run. Numbers come from `results/adjoint_sgd_amend2.json`, with seed 0 from
`results/adjoint_sgd.json` and `results/adjoint_sgd_amend1.json`. The figure is `size.png` from `plot.py`, which reads
only those three files.*

**In plain words.** We gave each of the five single-seed fine-tunes a second batch order and read its adjoint split
again. Every one landed within 0.02 of its first seed, and all three filed predictions held. For `GOALS.md`, amendment
1's reading now stands on two seeds. This split keeps an SGD fine-tune's behaviour only while the update is small, and
how much it keeps at one size depends on the task.

### The answer

The second seed repeated the first. At step 1000 every fine-tune's keep moved by 0.016 or less between the two batch
orders. Within each of the three tasks, both slower-rate fine-tunes kept more than both faster-rate ones, whose updates
are about twice as large. At the slower rate the updates have about the same size.
There emoticon kept about 0.6, r0_t352_v310 about 0.4 and r1_t534_v369 about 0.13, on both seeds. So the task-dependent fall with update size is a
result on these three tasks, not seed noise.

### What we were trying to test

The recipe is E31's and amendment 1's, unchanged. Each fine-tune trains all 24 decomposed matrices with plain SGD for
1000 steps on one task. At steps 10, 100 and 1000 the update $\delta W$ is written onto the subcomponents by the scaled
adjoint split, with Γ and Δ frozen. The only change is the training seed, which sets the order of the batches.

The two numbers are amendment 1's:

- **Keep**: the adjoint state's held-out P(target) at the trigger divided by the fine-tune's own at the same step. It
  reads about 1 for do-nothing, which leaves the whole update in Δ, and near 0 if the split loses the behaviour.
- **Update size**: the largest relative change of any matrix, $\max_k \|\delta W_k\| / \|W_k\|$.

Five fine-tunes had one seed: r1_t534_v369 and r0_t352_v310 at SGD rates 0.01 and 0.03, and emoticon at 0.03. Emoticon
at 0.01 already had two. With all six cells on two seeds, three comparisons can be scored. The first asks whether each
cell repeats. The second asks whether keep falls from the smaller to the larger update within each task. The third asks
whether the three tasks keep different amounts at the matched size of the slower rate.

Each job first checked the unchanged model against its known values. It then reran the cell's 10-step seed-0 fine-tune
and its adjoint state, which had to match the earlier run on four readings. That checks the data,
the split and the readout that keep is built from. Both checks passed in all five jobs. Every seed-1 fine-tune reached
P(target) of at least 0.93 by step 1000, so every cell was scorable.

### What we expected, and why

We expected all five cells to repeat within 0.1. Emoticon's two seeds at 0.01 differ by 0.04, the only spread measured.
We were least sure of the faster-rate cells, where keep falls steeply after step 100 and a different path could move it
more. We expected keep to fall with size on all three tasks. We were confident on emoticon, whose fall is ten times its
seed spread, and least confident on r1_t534_v369, where seed 0's gap was only 0.06. We expected the task order
at matched size to hold, since its gaps of 0.21 and 0.26 are five times the measured spread.

### What we observed

#### Does each cell repeat on a second seed?

![keep against update size, two seeds](../../experiments/adjoint_sgd/figures/size.png)

**Reading it.** Yes, at step 1000 every cell repeats. Filled markers on solid lines are seed 0 and hollow markers on
dashed lines are seed 1, in the task's colour. At the largest marker of each line, the end of training, every hollow
marker sits on its filled partner. The largest step-1000 gap is 0.016, under emoticon's earlier spread of 0.04.
The seeds part more at steps 10 and 100, by up to 0.11, and then come back together by step 1000.

| step 1000 keep, seed 0 / seed 1 | SGD 0.01 | SGD 0.03 |
|---|---|---|
| emoticon | 0.60 / 0.57 | 0.17 / 0.18 |
| r0_t352_v310 | 0.39 / 0.40 | 0.28 / 0.29 |
| r1_t534_v369 | 0.13 / 0.13 | 0.07 / 0.06 |

#### Does keep fall as the update grows, on every task?

**Reading it.** Yes, on all three tasks, with both seeds on each side. In the figure the triangles at the right end of
each colour sit below the circles at about half their size. On emoticon both 0.01 keeps exceed both 0.03 keeps by about
0.4. On r0_t352_v310 the gap is about 0.11, and on r1_t534_v369 about 0.07. Both gaps are many times the step-1000 seed
spread. Each 0.03 update was about twice the size of each 0.01 update on the same task.

#### At matched size, do the tasks keep different amounts?

**Reading it.** Yes, and in the same order on both seeds. The six slower-rate states lie within a factor 1.21 of each
other in size. Emoticon kept about 0.6, r0_t352_v310 about 0.4 and r1_t534_v369 about 0.13. The table above
gives each seed, and no pair of tasks overlaps.

A comparison filed as reported, not scored: each faster run's step-100 state is about the size of the slower run's
step-1000 state. There the faster run kept as much or more, on both seeds. Its step-100 states are slightly smaller,
so this agrees with one curve per task without testing it.

| keep, seed 0 / seed 1 | faster run, step 100 | slower run, step 1000 |
|---|---|---|
| emoticon | 0.60 / 0.63 at size 0.010 | 0.60 / 0.57 at size 0.012 |
| r0_t352_v310 | 0.49 / 0.58 at size 0.013 | 0.39 / 0.40 at size 0.015 |
| r1_t534_v369 | 0.27 / 0.24 at size 0.012 | 0.13 / 0.13 at size 0.014 |

#### Do the new adjoint states stay faithful?

Yes. Every seed-1 adjoint state passed the adversarial check against its own job's unchanged model. On this card the
pass line is a ratio of 1.24. The highest reading was 1.22, from r1_t534_v369's step-10 state at the faster rate. Every
step-1000 state read 0.99 or less. Δ's share of the update stayed at or below 1e-4, so Δ did not absorb it.

### What we learned

Amendment 1's reading holds on two seeds. On these three tasks, the adjoint split of a plain-SGD fine-tune keeps less
behaviour as the update grows. At one update size, how much it keeps depends strongly on the task. The step-1000 keeps
are pinned down to about 0.02 by the batch order, so the task differences are real, not noise. For the weak version of
`GOALS.md`, this split meets it only for small SGD updates, and the margin depends on the task.

### How compelling is the evidence

- Two seeds per cell give a spread, not a distribution. Other sources of variation, such as the decomposition and the
  task windows, were not varied.
- Three tasks from one family, two SGD rates, one 67M model.
- No size beyond 0.03 was read, and no step between 100 and 1000.
- Adam updates do not follow the SGD curves, so the reading is about plain SGD only.
- Nothing here says what about a task sets its level.
- The amendment-1 cells ran on an L40S and these on an L40. The 10-step gate reproduced the L40S readings within 0.002.
  Step-1000 keep agreed across the two cards within 0.02.
- P(target) is read only at the trigger, and damage elsewhere only through the 20-step adversary.

### Surprises

- None against the predictions: all three held as filed.
- The seeds differ more at steps 10 and 100, by up to 0.11, than at step 1000, by at most 0.016. The end of training is
  better pinned than the early steps.

### What next

1. The open part of amendment 1's "What next": two rates stopped at the same update size on one task. If keep matches,
   size is the variable; if not, the path matters.
2. Keep against size on more tasks, to find what sets each task's level. A level that tracks some property of the task
   would turn the reading into a rule.

### Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G0 | unchanged model at its known values | `w0_check`, each job | pass in all five; floor 0.209 | pass |
| A2_G1 | 10-step seed-0 rerun repeats | fine-tune P, KL elsewhere, R within 0.01; adjoint P within 0.02 | largest miss 0.002 | pass |
| C_delta | Δ share reads 0 | every adjoint state ≤ 1e-3 | max 9.8e-5 | pass |
| B1 | each cell repeats on seed 1 | replicates ≤ 0.10; differs ≥ 0.20 | gaps 0.004–0.016 | **replicates ×5: held** |
| B2 | keep falls with size within each task | both 0.01 keeps above both 0.03 keeps | emo 0.57–0.60 vs 0.17–0.18; r0 0.39–0.40 vs 0.28–0.29; r1 0.13 vs 0.06–0.07 | **falls ×3: held** |
| B3 | task order at matched size | emoticon > r0 > r1 on both seeds; sizes within ×1.5 | ratio 1.21; no overlap | **holds: held** |

#### All seed-1 states

| cell | step | size | fine-tune P | adjoint P | keep | R | adjoint ratio | do-nothing ratio |
|---|---|---|---|---|---|---|---|---|
| r1_t534_v369 SGD 0.01 | 10 / 100 / 1000 | 0.003 / 0.008 / 0.014 | 0.86 / 0.99 / 0.98 | 0.34 / 0.44 / 0.12 | 0.39 / 0.45 / 0.13 | 0.48 / 0.56 / 0.45 | 1.01 / 1.01 / 0.97 | 2.70 / 1.06 / 0.95 |
| r1_t534_v369 SGD 0.03 | | 0.006 / 0.011 / 0.029 | 0.99 / 0.98 / 0.98 | 0.67 / 0.23 / 0.06 | 0.68 / 0.24 / 0.06 | 0.52 / 0.48 / 0.33 | 1.22 / 0.99 / 0.97 | 1.63 / 1.06 / 1.09 |
| r0_t352_v310 SGD 0.01 | | 0.003 / 0.008 / 0.015 | 0.84 / 0.92 / 0.93 | 0.49 / 0.54 / 0.37 | 0.59 / 0.58 / 0.40 | 0.49 / 0.55 / 0.45 | 1.08 / 0.97 / 0.98 | 1.71 / 1.06 / 0.99 |
| r0_t352_v310 SGD 0.03 | | 0.005 / 0.013 / 0.029 | 0.90 / 0.97 / 0.94 | 0.57 / 0.56 / 0.27 | 0.63 / 0.58 / 0.29 | 0.53 / 0.48 / 0.35 | 0.94 / 0.97 / 0.99 | 1.17 / 1.09 / 1.16 |
| emoticon SGD 0.03 | | 0.004 / 0.010 / 0.027 | 1.00 / 0.99 / 1.00 | 0.92 / 0.62 / 0.18 | 0.92 / 0.63 / 0.18 | 0.58 / 0.56 / 0.35 | 1.00 / 0.98 / 0.95 | 1.15 / 1.02 / 1.02 |

The pass line on the L40 is 1.24 and the fail line 1.81, from the same-job floor of 0.209. Every do-nothing state
passes at steps 100 and 1000; at step 10 r1_t534_v369 at 0.01 fails and two more are inconclusive. The step-10 do-nothing states read high on seed 0 too.

### Not run

- A third seed; a size-matched pair of rates; other rates or sizes beyond 0.03; steps between 100 and 1000.
- More tasks; Adam at these sizes; other models; the strong version.

### Provenance

- MATS cluster `debug`, one L40 per cell, jobs 30644 and 30649–30652, $0, about 18–20 min each. r1_01 ran first and
  alone. One deviation: emo_03 started 19 min after submission and was not moved to RunPod, because the brief's
  10-minute cap named r1_01 only.
- Code: `run.py amend2 <cell>` and its merge, calling E31's `arm()` unchanged; `plot.py` → `fig_size` now draws both seeds.
- Files: `results/adjoint_sgd_amend2_{r1_01,r1_03,r0_01,r0_03,emo_03}.json`, merged `results/adjoint_sgd_amend2.json`
  (sha256 32e46e20…); logs `results/large/full_amend2_{r1_01,r1_03,r0_01,r0_03,emo_03}.log`.

## Amendment 3: does fitting the scale c matter? (agent, 2026-10-26)

### The answer

Yes, but not in the way we expected. The plain adjoint split, with no fitted scale, keeps the whole behaviour on every
emoticon fine-tune where the scaled one loses most of it. It pays for that with off-target damage: one and a half to
thirty times the fine-tune's own, and for the largest update a decomposition that fails the adversarial check badly.
So the fitted scale trades behaviour for harmlessness. It is not a small correction, and neither version is a simple
map that both keeps the behaviour and leaves the rest of the model alone.

### What we were trying to test

The adjoint split moves each subcomponent's read and write vectors along the update, $dU = \delta W V$, $dV = \delta W^\top U$,
with Δ fixed. The weights then change by $K(\delta W) = UU^\top\delta W + \delta W VV^\top$, which is not δW and is usually
larger. Every adjoint state so far multiplied each matrix's move by a number c fitted on the task's inputs so that $cK(\delta W)$
matches δW as closely as possible (the scaled adjoint split). The plain move (c = 1, the unscaled adjoint split) had never been
scored. We reran E31's three emoticon seed-0 fine-tunes: plain SGD at 0.01, a small Adam update (3e-6) and a large one (3e-5).
At steps 10, 100 and 1,000 we placed each update both ways and read three things:
- **keep:** the state's held-out probability of the target over the fine-tune's own (1 for do-nothing);
- **KL elsewhere:** the KL from the original model to the state's model at held-out positions other than the trigger.
  This is the fine-tune's own off-target readout, so the fine-tune itself is the reference on the same axis;
- **the adversarial ratio:** a 20-step attack on the subcomponent masks on Pile text, over the same at the original model.

Control first, in every job: the original model at its known values, then the step-10 scaled state against E31's row. All
three passed: adjoint P(target) within 0.002, the reproduced share within 0.0002, and the KL readout on the fine-tune equal to
its own `kl_other`.

### What we expected, and why

Fitted c is mostly below 1 (median 0.2 to 0.6 per matrix), so the plain move is 1.2 to 2.8 times the fitted one in a typical
matrix and up to 18 times in the worst. We filed (medium-high confidence) that it would cost at least twice the scaled state's
off-target KL at every state. With low confidence we filed that it would keep as much behaviour or more, because K never points
against δW, so a bigger step goes further along the update's direction. With low confidence we also filed that only the large
Adam update might fail the adversary. The user expected the plain split to do poorly overall.

### What we observed

![scaled](../../experiments/adjoint_sgd/figures/scaled.png)

**Reading it.** The plain split keeps all of the behaviour, while the scaled split loses most of it as training goes on (left).
The dashed lines sit at 1.00 to 1.03 in eight of nine states, against the solid lines' 0.10 to 0.93; only the large Adam update
at step 1,000 drops to 0.31, and the scaled split there keeps 0.10. Off target (middle), every plain state costs more than its
scaled twin, 2.5 to 24 times. Every plain state also costs more than the fine-tune itself (dotted): 1.3 to 2.2 times for SGD,
3.5 to 16 times for small Adam, 3 to 32 times for large Adam. The scaled states stay below the fine-tune's own cost except large
Adam at step 1,000. On the adversarial check (right), the SGD and small-Adam plain states pass (at most 1.38, against a pass line
of 1.40). The large-Adam plain states fail at every step: 2.4, 2.3 and 19.3, with KL elsewhere reaching 1.75 nats at step 1,000.
Δ's share stays at most 1.2e-4 in every state.

| fine-tune | step | keep scaled | keep plain | KL elsewhere scaled | plain | fine-tune | adversarial scaled | plain |
|---|---|---|---|---|---|---|---|---|
| SGD 0.01 | 10 | 0.735 | 1.017 | 0.0065 | 0.0240 | 0.0155 | 1.00 | 1.05 |
| | 100 | 0.760 | 1.008 | 0.0032 | 0.0080 | 0.0060 | 1.00 | 0.99 |
| | 1000 | 0.614 | 1.002 | 0.0046 | 0.0176 | 0.0080 | 0.97 | 1.04 |
| Adam 3e-6 | 10 | 0.714 | 1.028 | 0.0102 | 0.1052 | 0.0302 | 1.02 | 1.38 |
| | 100 | 0.470 | 1.007 | 0.0015 | 0.0131 | 0.0026 | 0.99 | 0.95 |
| | 1000 | 0.239 | 1.004 | 0.0018 | 0.0384 | 0.0023 | 1.00 | 1.03 |
| Adam 3e-5 | 10 | 0.934 | 0.999 | 0.0251 | 0.1782 | 0.0554 | 1.06 | 2.40 |
| | 100 | 0.600 | 1.000 | 0.0352 | 0.4297 | 0.0456 | 0.92 | 2.32 |
| | 1000 | 0.104 | 0.312 | 0.0721 | 1.7482 | 0.0547 | 0.95 | 19.31 |

### What we learned

The fitted scale is doing real work, and it is a trade, not a fix. Scaling the move down to match δW on the data removes the
off-target overshoot, and that same shrink is where the scaled split's behaviour is lost (E31's falling keep). So the scaled
split's record should be described as "fitted to each update", and the falling keep along training is in large part a size
effect of c, not a sign that the direction misses the behaviour. The plain split shows the direction alone is enough to carry
the behaviour. Neither version yet meets the weak version, which needs the behaviour kept and the rest of the model unharmed.

### How compelling is the evidence

One task (emoticon), one seed, three fine-tunes, three steps each. The control reproduced E31 closely and the readout check was
exact. KL elsewhere is read on the emoticon held-out windows only, not on Pile text; P(target) only at the trigger. No scale between
c and 1 was tried, so we do not know whether some middle scale keeps the behaviour with damage no worse than the fine-tune's.
Run on one H100 rather than E31's L40; each verdict is read against its own job's floor (0.075).

### Surprises

Something unexpected happened. The user expected the plain split to do poorly. It keeps the behaviour completely, better than
the scaled split everywhere, and its failure is off-target damage, not lost behaviour. The filed predictions had leaned this way
(D1 "higher or same", low confidence), so no filed prediction reversed. The large-Adam plain state at step 1,000 also lost
behaviour (0.31), the alternative the filing named.

### What next

1. A scale sweep between c and 1 (for example c, 2c, 4c, 1, or one global scale) on the same three fine-tunes. If some scale
   keeps keep near 1 with KL elsewhere at or below the fine-tune's own, the adjoint split plus one tuned number is a candidate
   map. If keep and damage rise together everywhere, the adjoint direction itself carries the off-target harm.
2. KL elsewhere on Pile text, which is the off-target measure the weak version cares about, for the scaled and plain states.

### Scorecard

Rules in `results/predictions_amend3.json` (filed 2026-10-26 before the run).

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| D1 | plain keeps as much or more behaviour | per state, plain minus scaled keep; same within 0.05 | higher in all 9 (+0.07 to +0.77) | as filed ("higher or same at most") |
| D2 | plain costs at least twice the off-target KL | per state, ratio ≥ 2 | 2.5 to 24, all 9 costlier | as filed |
| D3 | plain passes except perhaps large Adam at 100 and 1000 | the plain state's verdict | SGD and small Adam pass (6); large Adam fails at 10, 100, 1000 | as filed, except step 10 also fails |
| reading | "the scale does work" | not all D1 same, D2 similar, D3 pass | the scale does work (D1, D2, D3) | as filed |

### Not run

A scale between c and 1; one global scale; KL elsewhere on Pile text; the word tasks; seed 1; the LoRA updates; a cluster run (the
only node was draining, job 32257 cancelled unstarted).

### Provenance

`experiments/adjoint_sgd/amend3.py`, `results/adjoint_sgd_amend3_{sgd,small,big}.json`, merged `results/adjoint_sgd_amend3.json`,
logs `results/large/full_amend3_*.log`; figure `figures/scaled.png` (`plot.py`). RunPod H100 80GB, pod
`commute-adjoint_sgd-amend3`, 12:35 to 13:02 on 2026-10-26, $1.97; cells about 17 to 22 min each, sha256 checked on both sides.
Codex adversarial review not run: Codex was out of quota; the new code was re-read by hand and its scoring checked on CPU.
