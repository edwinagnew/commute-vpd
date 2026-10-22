# Analysis: does the adjoint split of a plain-SGD fine-tune keep the behaviour? On emoticon only (E31)

*2026-10-22, E31 agent. Plan: `reports/2026-10-22_plan_adjoint-sgd.md` and its Agent notes. Experiment
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

![held-out P(target) along training](../experiments/adjoint_sgd/figures/steps.png)

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

![adjoint P(target) against reproduced share](../experiments/adjoint_sgd/figures/keep.png)

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

![adversarial ratio per state](../experiments/adjoint_sgd/figures/floor.png)

**Reading it.** Yes. Every one of the 30 adjoint states read between 0.93 and 1.08, inside the pass region on every
task and step. Do-nothing read up to 2.29 for the large Adam update. Δ's share stayed below 1e-4 in every
adjoint state, so Δ did not absorb the update.

![budget sweep](../experiments/adjoint_sgd/figures/budget.png)

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
