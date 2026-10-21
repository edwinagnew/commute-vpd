# Plan: the five operations side by side (E16, `experiments/split_comparison`)

*2026-10-21, coding agent. Status: design confirmed by the user 2026-10-21; code, test and predictions written; not
run. Vocabulary: `GOALS.md` → Terms.*

In plain words: we put every way we have of carrying an update into the decomposition on one card, at three points
along training, and read the same few numbers for each. The question is which of them keeps the behaviour while the
decomposition stays as faithful on the Pile as it is at W₀, with Δ fixed. It extends E14's table with the rescaling
split as a state and with component training run for as long as the updates.

## The design, restated in the Terms' words

The emoticon task, steps 10, 100 and 1000, the same windows, held-out set and Pile harness as E14 (`harness.load` with
E14's arguments: 256 Pile rows, 20 PGD steps, adversary seeds 0, 1, 2).

**Two updates.** The full fine-tune, re-run as E14 does (arm A, Adam 3e-5, seed 0, δW kept at the three steps). The
rank-8 LoRA on all 24 matrices, read back from E9's saved factors.

**Four states of each update.**
- Do-nothing: δW goes into Δ.
- Least-squares split: the target moves by δW, the remainder O(δW) goes to Δ.
- Adjoint split, scaled per matrix (`split.adjoint_state`): the target moves by c K(δW), Δ fixed.
- Rescaling split, new as a state (`split.rescaling_state`): each u_c is scaled by 1 + α_c, with α the input-weighted
  least-squares fit of δW by Σ α_c u_c v_cᵀ. The target moves by that fit, Δ fixed, so it is a different model, as
  the adjoint state is. Its reproduced share R is `split.rescaling_share`.

**Component training, no update consumed.** Free component training of every alive subcomponent at lr 1e-5 (E2's
rate) is paired with the fine-tune. Rank-8 component training (`Offsets(rank=8)`, lr 1e-4, as E13) is paired with the
LoRA. Both keep Γ and Δ frozen and run once for 1000 steps, with states taken at steps 10, 100 and 1000.

**Columns.** Pile PGD ratio to W₀ with this card's three-seed floor f; the stochastic, CI-masked and unmasked ratios;
Δ's share of ‖δW‖²; the reproduced share R; held-out P(target) and the gain; the route split's d and r. The off-target
KL appears only where an eval already gives it: the updates' own and component training's held-out evals. No new code
measures it for the split states. For every update and every component-training weight change
(U+dU)(V+dV)ᵀ − UVᵀ we report the four blocks and outside-both over its null.

The results JSON carries a flat list `rows`: one record per update pair, step and operation, all with the same keys.

## What each number should read, and what it is blind to

| Number | W₀ / reference | Expected | Blind to |
|---|---|---|---|
| Pile PGD ratio | 1, floor f ≈ 0.12 on E14's card | do-nothing and least-squares as E14; adjoint and rescaling near 1 | the trained behaviour |
| held-out P(target) | 0.004 | updates ≈ 1; adjoint 0.93 → 0.06 as E14; rescaling < 0.05; component training ≥ 0.9 | off-target damage |
| Δ share | 0 | do-nothing 1; least-squares about 0.4; adjoint, rescaling and component training ≈ 0 | where in the components the change sits |
| R | — | rescaling ≤ 0.1 (E14: 2–7% of the energy) | whether the reproduced part is the part that matters |
| d | 0 | ≤ 0.15 wherever the gain exceeds 1 nat | the Pile |
| component training's outside-both block | 0 | ≈ 0, second order only | the behaviour |

The **positive control** is component training's weight change. To first order it lies inside the subcomponents, so
its outside-both block must be near zero. A large value means the block code or the training hook is wrong.

## Predictions (filed in `experiments/split_comparison/results/predictions.json`)

- **G1, the gate.** For both updates at all three steps, the Pile PGD ratios of do-nothing, least-squares and adjoint
  must sit within max(0.15, 3f) of E14's, absolute. The adjoint state's P(target) must sit within 0.05 of E14's.
  If the gate fails, we stop and find out why before reading the rest.
- **P1, the rescaling state.** At every step and for both updates: PGD ratio ≤ 1 + 3f, P(target) below 0.05, and R at
  most 0.1, all absolute.
- **P2, component training.** For both variants: P(target) ≥ 0.9 at steps 100 and 1000, and PGD ratio below 1.5 at
  steps 10 and 100. Step 1000's PGD ratio is a first reading with no threshold.
- **C1, the control.** Component training's pooled outside-both block is at most 0.01 of its input-weighted energy at
  every step, for both variants.
- **P3.** The direct share d is at most 0.15 for every state whose gain exceeds 1 nat.

## Cost and run

About 33 Pile evaluations: 30 states plus three W₀ seeds. E14 took about 3.6 minutes each on an RTX A5000, so the run
should take about one to one and a half hours on a 4090, for under $1. The smoke run times every stage and benchmarks
the full run.

```
uv run python experiments/split_comparison/run.py all --smoke --device cuda   # -> results/split_comparison_smoke.json
uv run python experiments/split_comparison/run.py all --device cuda           # -> results/split_comparison.json
```

The log goes to `results/large/full.log`. Every scored number is printed to the log, and the JSON is rewritten after
every state.

## Not run

- A second fine-tune seed and a second component-training seed.
- The off-target KL of the split states (no measurement code for it).
- The key → value tasks.
