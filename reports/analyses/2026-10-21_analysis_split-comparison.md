# Analysis: the five operations side by side (E16, `experiments/split_comparison`)

*2026-10-21, coding agent. Plan: `reports/plans/2026-10-21_plan_split-comparison.md`. Notebook:
`notebooks/E16_split_comparison.ipynb`. Vocabulary: `GOALS.md` → Terms.*

We carried a fine-tune and a LoRA into the decomposition in four ways, and trained the decomposition's own vectors in their place. Only training the vectors kept the behaviour while staying as faithful as the original model and leaving the residual alone. So on this task the behaviour can live inside the decomposition, but a finished update cannot be carried into it, and a failed reproduction check leaves the filed predictions unscored.

## Prediction status, first

- **The gate failed** on 2 of 24 conditions, both the fine-tune at step 10. Its least-squares state read 6.96 against
  E14's 5.24, and its do-nothing state 2.52 against 2.29, with a tolerance of 0.224.
- As filed, the three predictions on the rescaling state, component training and the direct share are therefore
  **unscored**. Their conditions are reported below as readings, not verdicts. Nothing is refiled.
- **The positive control failed** at one point of six: rank-8 component training at step 1000 read an outside-both
  share of 0.0119 against the filed 0.01.
- The least-squares state matched the updated model at all six points, as its gate requires.

## What was run

The released VPD decomposition of the 67M model, $W = U V^\top + \Delta$ per matrix, Γ and Δ frozen throughout. The
emoticon task. Two updates, each at steps 10, 100 and 1000: the full fine-tune re-run as in E14 (Adam 3e-5, seed 0)
and E9's rank-8 LoRA on all 24 matrices, read back from its saved factors.

Four operations consume a finished update $\delta W$, so each applies to both updates:

- **Do-nothing**: $W' = UV^\top + (\Delta + \delta W)$.
- **Least-squares split**: least-norm $(dU, dV)$ with $dU V^\top + U dV^\top \approx \delta W$, remainder to Δ; $W'$
  is the updated model.
- **Adjoint split**: $dU = \delta W V$, $dV = \delta W^\top U$, scaled per matrix by the input-weighted best scalar;
  Δ fixed, a different model.
- **Rescaling split**: $u_c \leftarrow (1 + \alpha_c) u_c$, α the input-weighted least-squares fit of $\delta W$ by
  $\sum_c \alpha_c u_c v_c^\top$; Δ fixed, a different model. First time as a state.

Component training consumes no update: it is trained instead of fine-tuning, so it is an analogue, not the same update.
Paired with the fine-tune: free component training of all alive subcomponents, lr 1e-5. Paired with the LoRA: rank-8
component training, lr 1e-4. Both 1000 steps, seed 0, read at the same three steps.

Measurements per state: Pile ratios to W₀ over 256 rows, adversarial with 20 PGD steps and three W₀ attacker seeds
giving the floor $f = 0.075$ (band $1 \pm 3f$ = 0.78 to 1.22); Δ's share of $\|\delta W\|^2$; the reproduced share
$R$; held-out P(target), W₀ 0.012; the gain at the trigger and its direct share $d$ and re-switching share $r$; the
outside-both block of each update and each component-training change against a random change of the same size; the
off-target KL where an eval already gives it, the updates and component training.

## What we observed

### 1. The five operations side by side (`figures/bars.png`)

The figure has three stacked panels sharing one axis of the five operations, each with six bars: the fine-tune, or free
component training, at steps 10, 100 and 1000, then the LoRA, or rank-8 component training. The panels are the
adversarial ratio with the noise band, the held-out P(target) with W₀'s and the updated model's level, and Δ's share
of $\|\delta W\|^2$. The 6.96 bar is cut off at 3 and labelled.

Only component training keeps the behaviour inside the noise band with Δ untouched, at all six of its points. The
adjoint and rescaling states are inside the band and have lost the behaviour. The fine-tune's do-nothing and
least-squares states keep the behaviour and leave the band, up to 6.96. Some LoRA do-nothing and least-squares states
sit in the band at full behaviour, but by putting all, or 41 to 54 percent, of the update into Δ.

Behaviour kept, panel (b): do-nothing and least-squares are the updated model: 0.999 or more. Free component training 0.9998, 0.994, 0.997;
rank-8 0.895, 0.996, 0.9995. Adjoint: fine-tune 0.93, 0.60, 0.10 with $R$ 0.58, 0.42, 0.31; LoRA 0.50, 0.18, 0.06
with $R$ 0.23, 0.12, 0.11. Rescaling: 0.012 to 0.016, against W₀'s 0.012.

Off-target KL on held-out text, where measured: fine-tune 0.055, 0.046, 0.055 against free component training's
0.016, 0.005, 0.007; LoRA 0.021, 0.012, 0.005 against rank-8 component training's 0.042, 0.004, 0.003.

Overlap with the subcomponents, panel (c) for the Δ share; $R$ is in the table only:

| operation | Δ share, fine-tune | Δ share, LoRA | R, fine-tune | R, LoRA |
|---|---|---|---|---|
| do-nothing | 1 | 1 | 1 | 1 |
| least-squares split | 0.36–0.38 | 0.41–0.54, rising | 1 | 1 |
| adjoint split | ≤ 8e-6 | ≤ 3e-4 | 0.58 → 0.31 | 0.23 → 0.11 |
| rescaling split | ≤ 5e-9 | ≤ 2e-8 | 0.020–0.071 | 0.003–0.013 |
| component training | ≤ 2e-8 | ≤ 1.2e-8 | 1 by construction | 1 by construction |

No split of a finished update reaches all of the update reproduced with Δ untouched. Component training's R is
measured against its own change, so its 1 is not a comparison with the update.

### 2. Faithfulness along training (`figures/pgd.png`)

For the fine-tune, least-squares reads 6.96, 1.48 and 2.11 at steps 10, 100 and 1000; do-nothing 2.52, 1.14 and 1.50.
Adjoint, rescaling and free component training stay between 0.92 and 1.06. For the LoRA everything is at or below
1.32; only least-squares at steps 10 and 100 leaves the band, at 1.24 and 1.32. Do-nothing's unmasked ratio is 2.8
to 5.5 for the fine-tune and 1.5 to 2.0 for the LoRA. Every stochastic and CI-masked ratio is within 0.92 to 1.02.

### 3. Outside share (`figures/outside.png`)

| update | step | change | outside both | random change | ratio | off-target KL |
|---|---|---|---|---|---|---|
| fine-tune | 10 | the update | 0.0336 | 0.155 | 0.216 | 0.0554 |
| fine-tune | 10 | free component training | 6.32e-09 | 0.0526 | 1.2e-07 | 0.0165 |
| fine-tune | 100 | the update | 0.0617 | 0.189 | 0.327 | 0.0456 |
| fine-tune | 100 | free component training | 7.55e-08 | 0.0776 | 9.73e-07 | 0.00526 |
| fine-tune | 1000 | the update | 0.0955 | 0.238 | 0.402 | 0.0547 |
| fine-tune | 1000 | free component training | 1.02e-06 | 0.15 | 6.76e-06 | 0.0073 |
| LoRA | 10 | the update | 0.119 | 0.122 | 0.979 | 0.0212 |
| LoRA | 10 | rank-8 component training | 6.06e-07 | 0.232 | 2.61e-06 | 0.0416 |
| LoRA | 100 | the update | 0.16 | 0.296 | 0.539 | 0.0116 |
| LoRA | 100 | rank-8 component training | 1.57e-05 | 0.279 | 5.62e-05 | 0.00443 |
| LoRA | 1000 | the update | 0.362 | 0.38 | 0.954 | 0.00476 |
| LoRA | 1000 | rank-8 component training | 0.0119 | 0.392 | 0.0304 | 0.00286 |

Lower is better: less outside means the subcomponents can express more of the change. Shares are pooled over the 24
matrices by input-weighted energy. The figure plots the measured share against the random change, with no control
line. The fine-tune is inside, 0.22 to 0.40 of random,
rising along training. The LoRA is near random, 0.54 to 0.98. Component training is near zero except rank-8 at
step 1000.

### 4. Where the gain comes from (`figures/routes.png`)

Stacked bars per state: direct share $d$, re-switching share $r$, and the rest $1 - d - r$ (Δ, switched-off
subcomponents, interactions). The figure shows do-nothing, least-squares, adjoint and component training at all three
steps; the rescaling states are left out for their small gains. In every state with a gain above 1 nat, $d \le 0.080$, the maximum at the fine-tune's step-10 adjoint state, and
$r$ is 0.28 to 0.67. Do-nothing's $d$ is 0 by construction. The six rescaling states have gains of 0.06 to 0.35 nat;
their $d$ ranges from −0.06 to 0.17 and $r$ from 0.17 to 0.36, ratios of small numbers.

### Full table

| update | step | operation | adversarial | stochastic | CI-masked | unmasked | Δ share | R | P(target) | gain (nats) | d | r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fine-tune | 10 | do-nothing | 2.52 | 1 | 0.999 | 2.89 | 1 | 1 | 0.999 | 13.5 | 0 | 0.597 |
| fine-tune | 10 | least-squares split | 6.96 | 1 | 0.997 | 1.21 | 0.359 | 1 | 0.999 | 13.5 | 0.0527 | 0.597 |
| fine-tune | 10 | adjoint split | 1.06 | 0.993 | 0.99 | 0.979 | 1.56e-06 | 0.577 | 0.933 | 8.57 | 0.0801 | 0.668 |
| fine-tune | 10 | rescaling split | 0.989 | 1 | 1 | 1 | 4.36e-09 | 0.0199 | 0.0163 | 0.351 | 0.171 | 0.244 |
| fine-tune | 10 | component training | 1.03 | 0.998 | 1 | 1.01 | 1.96e-08 | 1 | 1 | 15 | 0.0255 | 0.529 |
| fine-tune | 100 | do-nothing | 1.14 | 0.956 | 0.99 | 2.81 | 1 | 1 | 0.999 | 14.1 | 0 | 0.526 |
| fine-tune | 100 | least-squares split | 1.48 | 0.957 | 0.989 | 1.06 | 0.38 | 1 | 0.999 | 14.1 | 0.0449 | 0.526 |
| fine-tune | 100 | adjoint split | 0.924 | 0.962 | 0.994 | 0.904 | 2.62e-06 | 0.422 | 0.6 | 5.52 | 0.0634 | 0.5 |
| fine-tune | 100 | rescaling split | 0.961 | 0.994 | 0.997 | 0.987 | 3.33e-09 | 0.0271 | 0.0133 | 0.161 | 0.113 | 0.224 |
| fine-tune | 100 | component training | 1.03 | 0.994 | 0.999 | 0.98 | 6.97e-09 | 1 | 0.994 | 12.2 | 0.0229 | 0.496 |
| fine-tune | 1000 | do-nothing | 1.5 | 0.916 | 0.988 | 5.53 | 1 | 1 | 1 | 17.9 | 0 | 0.491 |
| fine-tune | 1000 | least-squares split | 2.11 | 0.921 | 0.992 | 1.18 | 0.379 | 1 | 1 | 17.9 | 0.0183 | 0.491 |
| fine-tune | 1000 | adjoint split | 0.945 | 0.944 | 0.997 | 0.848 | 7.72e-06 | 0.314 | 0.104 | 2.36 | 0.0111 | 0.39 |
| fine-tune | 1000 | rescaling split | 0.916 | 0.986 | 1 | 0.967 | 1e-09 | 0.0706 | 0.0128 | 0.128 | 0.0547 | 0.172 |
| fine-tune | 1000 | component training | 0.99 | 0.985 | 0.995 | 0.954 | 1.84e-09 | 1 | 0.997 | 16.8 | 0.00505 | 0.441 |
| LoRA | 10 | do-nothing | 1.18 | 1.01 | 1.01 | 2 | 1 | 1 | 0.999 | 14.5 | 0 | 0.582 |
| LoRA | 10 | least-squares split | 1.24 | 1.01 | 1.01 | 1.35 | 0.41 | 1 | 0.999 | 14.5 | 0.0166 | 0.582 |
| LoRA | 10 | adjoint split | 1.05 | 1 | 1 | 1.01 | 6.4e-06 | 0.225 | 0.499 | 4.91 | 0.0109 | 0.571 |
| LoRA | 10 | rescaling split | 1 | 1 | 1 | 0.994 | 9.7e-09 | 0.0128 | 0.0131 | 0.121 | 0.07 | 0.362 |
| LoRA | 10 | component training | 1.12 | 1.02 | 1.02 | 1.02 | 1.18e-08 | 1 | 0.895 | 8.15 | 0.0512 | 0.565 |
| LoRA | 100 | do-nothing | 1.09 | 0.996 | 0.999 | 1.99 | 1 | 1 | 1 | 14.3 | 0 | 0.559 |
| LoRA | 100 | least-squares split | 1.32 | 0.996 | 1 | 1.27 | 0.463 | 1 | 1 | 14.3 | 0.0133 | 0.559 |
| LoRA | 100 | adjoint split | 1.03 | 0.998 | 1 | 0.979 | 5.71e-06 | 0.116 | 0.176 | 3.06 | 0.000331 | 0.392 |
| LoRA | 100 | rescaling split | 0.992 | 0.998 | 0.998 | 0.998 | 1.44e-08 | 0.00531 | 0.0129 | 0.111 | -0.00871 | 0.218 |
| LoRA | 100 | component training | 1.04 | 1 | 1 | 1 | 3.3e-09 | 1 | 0.996 | 11.5 | 0.0437 | 0.564 |
| LoRA | 1000 | do-nothing | 1.05 | 1 | 1 | 1.5 | 1 | 1 | 1 | 15.3 | 0 | 0.542 |
| LoRA | 1000 | least-squares split | 1.07 | 1 | 1 | 1.21 | 0.542 | 1 | 1 | 15.3 | 0.00639 | 0.542 |
| LoRA | 1000 | adjoint split | 1.02 | 0.997 | 0.998 | 0.986 | 0.000265 | 0.107 | 0.0629 | 1.7 | 0.00031 | 0.283 |
| LoRA | 1000 | rescaling split | 1 | 0.999 | 0.999 | 0.995 | 1.86e-08 | 0.00269 | 0.0123 | 0.0634 | -0.0577 | 0.321 |
| LoRA | 1000 | component training | 0.967 | 1 | 1 | 0.997 | 9.33e-10 | 1 | 0.999 | 16.4 | 0.0187 | 0.484 |

Ratios are to W₀. Component training's rows are the free variant beside the fine-tune and the rank-8 variant beside the
LoRA; their "Δ share" is the change in Δ relative to their own weight change, and their R is 1 by construction.

## What we learned

`GOALS.md`'s weak version asks for a state as faithful as W₀ under VPD's checks, with Δ not absorbing the update. On
the emoticon task, component training meets it at every step to 1000, free and rank-8. None of the four splits of a
finished update meets it: the adjoint and rescaling splits are faithful and carry part of the update, while the
least-squares split carries all of it through Δ and, for the fine-tune, fails the adversary. So the behaviour can be
reached inside the decomposition, but a finished update cannot be carried into it here. The rescaling state shows,
as a model, what E1 showed as an energy share: resizing the existing subcomponents carries none of the behaviour.

## How compelling is the evidence

One task; one seed for each update and each component-training variant; one H100. The floor comes from three attacker
seeds on W₀. The Pile checks are blind to the behaviour; P(target) is blind to off-target damage, which was measured for
the updates and component training only. The sparsity loss, part of the weak version, was not measured. Component
training is compared with an analogue, not with the same update. The step-10 fine-tune state is the least stable
reading in the project and failed this run's gate.

## Surprises

No filed prediction reversed.

**The gate failed at the fine-tune's step 10.** Least-squares read 6.96 against E14's 5.24; do-nothing read 2.52
against 2.29, 0.002 beyond the 0.224 tolerance. The other 22 readings reproduce; every adjoint and every LoRA reading is
within 0.131. Facts, not a cause: this fine-tune's step-10 least-squares state has read 5.4 (round 2), 7.3 (E7),
5.41 (E8), 5.24 (E14) and 6.96 (E16) on different cards. This run was an H100, E14 an RTX A5000; W₀'s adversarial
floor was 0.075 here and 0.122 in E14. E15, also on an H100, read the same state at 6.96, with stochastic 1.003 and
unmasked 1.208 as here. The cause is not established.

| condition | E16 | abs. difference from E14 | within 0.224 / 0.05 |
|---|---|---|---|
| fine-tune adjoint adversarial, steps 10 / 100 / 1000 | 1.06 / 0.924 / 0.945 | 0.003 / 0.011 / 0.007 | yes |
| fine-tune adjoint P(target), steps 10 / 100 / 1000 | 0.933 / 0.600 / 0.104 | 0.003 / 0.003 / 0.021 | yes |
| fine-tune least-squares adversarial, step 10 | 6.96 | 1.71 | **no** |
| fine-tune least-squares adversarial, steps 100 / 1000 | 1.48 / 2.11 | 0.048 / 0.110 | yes |
| fine-tune do-nothing adversarial, step 10 | 2.52 | 0.226 | **no** |
| fine-tune do-nothing adversarial, steps 100 / 1000 | 1.14 / 1.50 | 0.007 / 0.218 | yes |
| LoRA adjoint adversarial, steps 10 / 100 / 1000 | 1.05 / 1.03 / 1.02 | 0.001 / 0.003 / 0.019 | yes |
| LoRA adjoint P(target), steps 10 / 100 / 1000 | 0.499 / 0.176 / 0.063 | 0.001 / 0.000 / 0.000 | yes |
| LoRA least-squares adversarial, steps 10 / 100 / 1000 | 1.24 / 1.32 / 1.07 | 0.015 / 0.131 / 0.009 | yes |
| LoRA do-nothing adversarial, steps 10 / 100 / 1000 | 1.18 / 1.09 / 1.05 | 0.043 / 0.012 / 0.007 | yes |

**The control failed at one point.** Rank-8 component training at step 1000: outside-both 0.0119, filed line 0.01,
0.030 of a random change of its size. The other five points are at most 1.6e-5. The expected source is the
second-order term $dU\,dV^\top$, which grows with the change; it was not measured, so the cause is not established.
The filed null for this control names a bug in the block code or the training hook; that has not been checked either.

## What next

- **The gate's miss.** Build the fine-tune's step-10 least-squares state from one saved δW and read it on an RTX A5000
  and on an H100. If the reading follows the card, step-10 adversarial comparisons need a same-card reference; if not,
  the fine-tune re-run differs from E14's.
- **The control's miss.** Compute $dU\,dV^\top$ for rank-8 component training at step 1000, no GPU needed if its
  offsets were saved. If it accounts for the 0.0119, the control holds to second order; if not, look for the bug the
  filed null names.
- **Component training on a task the model lacks** (key → value: in E10, component training on the rand32 variant read 2.81
  under the adversary). A failure there would tie this result to a behaviour the model nearly has. Through `/design`.

## Scorecard (as filed)

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G1 | this run reproduces E14 | do-nothing, least-squares, adjoint PGD ratios within max(0.15, 3f) = 0.224 of E14's; adjoint P(target) within 0.05; both updates, steps 10, 100, 1000 | 22 of 24 within; ft step 10 least-squares 6.96 vs 5.24 and do-nothing 2.52 vs 2.29 | **fail** |
| G2 | the least-squares state is the updated model | harness G2 per update and step | all six hold | pass |
| P1 | the rescaling state stays at W₀ | PGD ratio ≤ 1 + 3f, P(target) < 0.05, R ≤ 0.1, every step, both updates | ratio 0.916–1.004, P 0.012–0.016, R 0.0027–0.071; all 18 conditions met | unscored (G1 failed) |
| P2 | component training keeps both | P(target) ≥ 0.9 at steps 100 and 1000, PGD ratio < 1.5 at steps 10 and 100, both variants | P 0.994–0.9995; ratio 1.027–1.118; all 8 conditions met | unscored (G1 failed) |
| C1 | component training's change lies inside | pooled outside-both block ≤ 0.01, every step, both variants | free 6e-9, 8e-8, 1e-6; rank-8 6e-7, 1.6e-5, 0.0119 | **fail** (rank-8 step 1000) |
| P3 | no direct route | d ≤ 0.15 for every state with gain > 1 nat | max d 0.080; all 24 conditions met | unscored (G1 failed) |

Reported, not scored: step 1000's component-training PGD ratio, 0.990 free and 0.967 rank-8.

## Not run

- A second seed for either update or either component-training variant.
- The off-target KL of the four split states (no measurement code for it).
- The sparsity loss of any state.
- The $dU\,dV^\top$ term of component training, needed to read the control's miss.
- The step-10 fine-tune state on E14's card, needed to read the gate's miss.
- The key → value tasks.

## Provenance

Results `experiments/split_comparison/results/split_comparison.json`, sha256
`3db95e00441c55693d15f75ef5f3a8b542d3240085c24cab37f20d0c9ad155d8`; smoke `split_comparison_smoke.json`; logs
`results/large/{full,full_smoke,pod_scored}.log`. Figures `experiments/split_comparison/plot.py` →
`figures/{bars,pgd,outside,routes}.png`. RunPod H100 80GB, 2026-10-21, scored run 2118 s, $3.68
in all. Predictions `results/predictions.json`, filed 2026-10-21 before the run.
