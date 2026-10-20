# Analysis: why the adversarial check moves and the stochastic check does not

*2026-10-20, coding agent, from existing data only: `experiments/reading_a_coverage/results/coverage.json`,
`coverage_emoticon.json`, and the measurement code (`src/commute/faithfulness.py`, param-decomp `pgd_utils.py`,
`routing.py`). No new run. Question from the user after the emoticon rerun.*

We asked why the pull-back's decomposition fails VPD's adversarial-mask check by factors of two to seven while its
stochastic-mask check stays within a percent of the unchanged model. Reading the code and the per-block numbers shows
that the two checks differ in two structural ways, and that the adversary's extra damage needs a selective mask, not
a uniform one: turning every maskable subcomponent down together costs almost nothing. For the goal this points at
the pull-back encoding the update as changes that cancel each other across subcomponents, which is the brittleness
reading of the question, and it can be tested directly on the saved states in about an hour of work.

## What the two checks actually do (from the code)

- **Both use the same mask box.** The adversary's mask is `ci + (1 − ci)·a` with `a` clamped to [0, 1]
  (`interpolate_pgd_mask`); the stochastic mask is drawn uniformly in the same interval [CI, 1]. Neither can turn
  down a subcomponent whose causal importance is already 1 on that position.
- **Layers.** The stochastic check routes each position to a random k of the 24 matrices, k uniform in 1..24
  (`UniformKSubsetRouter`): on average half the matrices use the target weights unmasked. The adversary
  (`AllLayersRouter`) masks all 24 matrices at once with one mask shared across the 128 rows of a block.
- **Draws.** Stochastic: 2 random draws, averaged. Adversarial: 20 sign-gradient steps of 0.1 from a random start,
  maximising the KL.
- **CI-masked** (deterministic): all 24 matrices at mask = CI, Δ excluded. This is the `a = 0` corner of the
  adversary's box with Δ dropped.

## What the numbers say

Changes from W₀ in nats, mean of the two blocks (W₀: stochastic 0.248, adversarial 0.70, CI-masked 0.348).

| state (pull-back unless marked) | adversarial | CI-masked (all at CI) | stochastic | adversarial per block |
|---|---|---|---|---|
| emoticon | +0.02 | +0.0008 | −0.0003 | 1.09, 0.97 |
| emoticon do-nothing | +0.82 | −0.0009 | +0.0002 | 1.16, 3.14 |
| r1_t588 | +0.11 | +0.015 | +0.009 | 1.17, 1.15 |
| r2_t359 s0 | +0.39 | +0.040 | +0.008 | 1.35, 1.75 |
| r1_t534 | +0.87 | +0.017 | +0.0002 | 2.26, 2.21 |
| r0_t352 | +1.21 | +0.004 | −0.006 | 1.96, 3.46 |
| r2_t275 | +4.0 | +0.46 | +0.18 | 6.47, 6.99 |

1. **The adversary's extra damage is not at the all-CI corner.** On r1_t534 turning every maskable subcomponent
   down to its CI value together costs 0.017 nats more than at W₀; the adversary's chosen mask costs 0.87 nats
   more, fifty times as much. On r0_t352 the ratio is three hundred. So the damage needs some changed subcomponents
   turned down and others left at 1. A uniform shrink of all of them nearly preserves the model.
2. **The stochastic check is diluted by construction.** With only about half the matrices masked per position and
   masks spread uniformly over [CI, 1], a weakness that needs several specific subcomponents masked together across
   layers is rarely sampled. Its W₀ seed spread is 0.2%, so the near-zero changes are real, but they measure the
   average over a space the adversary searches.
3. **Noise is not the explanation above about 0.3 nats.** W₀'s adversary spread is 0.03 nats across seeds in the
   five-task job and up to 0.3 nats in the emoticon job (one run of six reached 1.0 nats). That covers r1_t588 and
   part of r2_t359, not the other four states.
4. **A shared mask works on both blocks.** For five of seven states the two blocks' ratios agree within 30%; the
   adversary found something that hurts on most Pile text, not a few rows.

## What this says about the five candidate explanations

| explanation | status from existing data | news for GOALS |
|---|---|---|
| 1. cancellation across the pull-back's changed subcomponents | **supported indirectly**: selective masks do 50–300× the damage of the uniform-CI mask, which is the signature of changes that cancel in sum | bad: the update is not "a small change to a few subcomponents" but a spread of opposing changes that only reconstructs in full |
| 2. Γ stale: changed subcomponents are ones Γ lets the adversary mask | **necessary condition, confirmed by the box**: the adversary can only touch low-CI subcomponents, so the changes it exploits sit in subcomponents Γ calls unimportant on Pile text. Not sufficient: turning all of them down costs little (1) | mixed: the fix is an update of Γ, which the plan reserved; but (1) says a Γ update alone would not remove a cancelling encoding |
| 3. first-order versus second-order sensitivity | **not separable** from the routing dilution (2 above); the stochastic check is structurally insensitive, so its flatness is weak evidence either way | neutral |
| 4. non-local update | **consistent**: one mask per 128 rows works on both blocks | bad for interpretability along training, diagnosable from the mask |
| 5. adversary noise | **excluded above 0.3 nats**; blurs r1_t588 and r2_t359 | caution only |

## Further tests, in order of value per hour (each needs `/design` before building)

| test | what it settles | work | GPU time |
|---|---|---|---|
| **Coherence of the pull-back's changes** on the saved states: per matrix, ‖Σ_c Δ_c‖² / Σ_c ‖Δ_c‖² where Δ_c = dU_c V_cᵀ + U_c dV_cᵀ + dU_c dV_cᵀ. Orthogonal changes read 1, aligned read the count, cancelling read below 1. Reference: the same statistic for the least-norm lift of the same δW | explanation 1 directly | ~45 min to write and smoke; CPU only, states (3 GB) and U, V are on the cluster NFS | none; 5 min CPU, 10 min queue |
| **Partial revert**: KL(W′ ‖ W′ with a random half of the changed subcomponents put back to W₀) against the full revert KL(W′ ‖ W₀). Cancellation gives half-revert ≫ full revert | explanation 1 causally, independent of the norm statistic | ~1 h | 1 min per state, 15 min |
| **Stochastic check with all layers routed** (one argument change: `AllLayersRouter` in place of the k-subset router) at W₀ and every state | whether the stochastic pass is the routing dilution (3) or a real mean/max gap | ~20 min | 2 min per state, 35 min |
| **Save the adversary's mask** at W₀ and at each state; bin the masked-down subcomponents by layer and by CI, and score the KL with the mask restricted to the changed subcomponents | explanations 2 and 4: which subcomponents carry the excess, and whether they are the changed ones | ~1.5 h | 2 min per state, 35 min |
| **More adversary seeds**: 10 seeds at W₀ and at r2_t359 | the size of the W₀ floor; whether verdicts near 1.5 are readable at all | ~15 min (a tuple in `run.py`) | 25 min |

The first two together take about two hours of wall time including queue, cost nothing, and decide between "the
pull-back cheats the stochastic check by cancellation" and "the stochastic check is just blind". Either way the
weak version as filed, which names VPD's checks including the adversary, stands failed on four tasks.

## Not run

Everything in the table above. Nothing here is a new measurement; it is the existing numbers and the code read
together.

## Hand-off for the next session

Start from this file, then `experiments/reading_a_coverage/run.py` (how states were made and scored) and
`src/commute/pullback.py` (`train` saves `(delta-W, U_A', V_A')` per matrix; `alive_uv` gives the released
`(U_A, V_A)`). Inputs, all on the cluster NFS under `/mnt/nw/home/e.agnew/commute/`:
`experiments/reading_a_coverage/results/large/<task>/pullback_s0/state_step10.pt` for the six tasks (emoticon,
r0_t352_v310, r1_t534_v369, r1_t588_v452, r2_t275_v2667, r2_t359_v816; seeds 1 and 2 for r2_t359),
`dW_ft_step10_s0.pt` beside each, `experiments/transport_test/results/large/alive.pt`, and `artifacts/` for the
released U, V. The first test (coherence) is CPU-only and can be a `debug` job without a GPU; the second
(partial revert) needs one L40 for about 15 min. Run `/design` on the chosen test first; the predictions to file
are the expected coherence of the least-norm lift (the reference) and a threshold below 1 for "cancelling".
