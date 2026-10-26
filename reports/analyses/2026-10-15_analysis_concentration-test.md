# Concentration test: a few subcomponents suffice, and the paper's emoticon subcomponents can learn it alone

**Takeaway.** Several constrained pull-backs learn the emoticon behaviour with small off-target damage and unchanged Pile
faithfulness:
- only `2359` and its five siblings;
- only the top 10;
- only one subcomponent (`h.0.mlp.c_fc:1535`).

At the trigger, the frozen-Γ CI-masked component model gives P(o) 0.98 for each of them. The unconstrained Adam
pull-back of round 2 gives only 0.64, barely above do-nothing (0.53). The filed trigger-local control missed its rule
by 0.037, so that comparison is not scored. Plain SGD found no learning rate that learns without large off-target
damage, so the "Adam causes the diffuseness" prediction is unscorable.

*2026-10-15, coding agent. Plan `reports/plans/2026-10-15_plan_concentration-test.md`; predictions
`experiments/concentration_test/results/predictions.json` (filed before training); results `results/concentration.json`.
RunPod 4090, $0.88, 23 min. No code changes during the run.*

**Terms.**
- *Pull-back*: training the alive subcomponents' (U, V) on the emoticon loss, with Γ and Δ fixed.
- *Trigger-local*: measured on the 73 held-out windows at the trigger position.
- *CI-masked*: the component model with masks = frozen Γ's g and Δ excluded.
- *Do-nothing*: target weights W₀ + δW (the Adam pull-back's δW) with the components left at W₀.
- *Revert-one / fraction removed*: as in the identity test.

## ⚠ Two surprises, flagged before the scores

1. **The round-2 pull-back is not faithful at the trigger.** The CI-masked P(o) is 0.635 against a target of 0.996, with KL at the
   trigger 0.54. "Faithful on every metric" (round 2, P3′) was a Pile-only statement, as the faithfulness-scope review
   said, and at the trigger it does not hold. Only 11% of its update sits in the 188 subcomponents Γ gates on
   (g > 0.5) at the trigger.
2. **Γ's masks alone move the CI-masked model most of the way to `o`.** Do-nothing leaves every component at W₀, yet
   its CI-masked P(o) is 0.53. At W₀ it is 0.0045, and at step 10 of the run it was 0.87. The only thing that changed
   is the activations the frozen Γ reads. Cause not established. One candidate: the update changes upstream activations so that Γ gates on subcomponents that already
   write `o`. That would be the "reveal, not create" reading. Untested.

## Scores (as filed)

- **Power gates.**
  - The identity measure has power: swapping v (rank 345) and randomising u (write cosine −0.04) both fail.
    But a 20° rotation of v passes, so the measure is blind below about 20–45°.
  - **The trigger-local measure has no power by the filed rule.** Do-nothing's CI-masked P(o) is 0.535; the rule
    needed < 0.498.
- **P1 (SGD concentrates): unscorable.** No SGD learning rate met the rule. lr 0.3 learns, but its KL elsewhere is
  0.48 against a limit of 0.110, and its PGD ratio is 24.9. Higher learning rates damage the model or diverge.
  Reported anyway: top 10 remove 0.076 of the gain.
- **P2 (top 10 inside) and P3 (siblings inside): pass on every scorable part, trigger part unscorable.**
  - Both learn (P(o) 0.997 / 0.999; KL elsewhere 0.017 / 0.011).
  - Pile ratios are all ≤ 1.08, against a W₀ PGD seed spread of 3.5%.
  - At the trigger, the CI-masked P(o) is 0.978 for both. That passes the filed ≥ 0.8 × target, but the control
    failed, so it is not scored.

## Per arm (step 30, selected lr)

| Arm | Moved | lr | KL elsewhere | CI-masked P(o) at trigger (target ≈ 1) | KL at trigger, CI-masked | Pile PGD ratio | Top-10 revert removes |
|---|---|---|---|---|---|---|---|
| do-nothing (control) | 0 | n/a | n/a | 0.535 | 0.742 | n/a | n/a |
| Adam, all | 9 881 | 1e-5 | 0.009 | 0.635 | 0.541 | 1.07 | 0.016 |
| SGD, all (did not learn by the rule) | 9 881 | 0.3 | 0.478 | 0.999 | 0.001 | 24.9 | 0.076 |
| top 1 | 1 | 1e-3 | 0.020 | 0.977 | 0.046 | 1.07 | 1.0 |
| top 10 | 10 | 1e-3 | 0.017 | 0.978 | 0.019 | 1.01 | 0.997 |
| top 100 | 100 | 1e-4 | 0.011 | 0.918 | 0.121 | 1.06 | 0.254 |
| siblings | 6 | 1e-3 | 0.011 | 0.978 | 0.022 | 1.08 | 1.0 |

- **Ranks in the first-order importance |dL/ds|** (out of 9 881):
  - `2359` ranks 127;
  - the siblings rank 93–1 152;
  - the top subcomponent is `h.0.mlp.c_fc:1535`.
- **The sparse arms change their subcomponents a lot.**
  - Siblings: each rank-1 matrix changes by about 100% (scale 1.4–1.5×, |cos u| 0.70–0.74, |cos v| 0.92–0.94).
    All six still retrieve themselves at rank 1, but r_self is 0.64–0.69, so 0/6 "keep identity" by the filed rule.
  - top1: scale 1.34, |cos u| 0.81 (corrected 2026-10-16: 0.96 was `1535`'s value in the top-10 arm).
  - Adam-all: every subcomponent changes by about 0.3%.

## What this does and does not say

- **Says:** the emoticon behaviour can be learned inside the decomposition by strengthening and re-aiming a handful of
  subcomponents, including the six the paper labels for it. Doing so leaves Pile faithfulness unchanged, does less
  off-target damage than the W fine-tune (0.011–0.020 against 0.055), and gives a CI-masked model that matches the
  target at the trigger. The diffuse Adam solution is the poor one at the trigger.
- **Says:** the labelled subcomponents keep their contexts (rank 1) but their write direction turns by about 45°. Whether that is
  still "the emoticon component" in a human sense is not measured.
- **Does not say:**
  - Whether Adam causes the diffuseness. SGD did not learn within the rule.
  - Why Γ's masks alone carry so much (surprise 2).
  - Anything beyond this task and 30 steps.
  - Whether the trigger-local comparison would survive a control with power.

## Not run

- A trigger-local control with power: e.g. do-nothing with Γ's masks computed on W₀ activations (frozen inputs to Γ),
  which separates "components carry it" from "Γ's gating carries it".
- SGD with a finer lr grid (0.03–0.3) or momentum, to score P1.
- Constrained pull-back with an explicit sparsity penalty instead of a hard top-k.
- The siblings arm at matched ‖δW‖ to the Adam-all arm (it moves 5× further: rel norm 0.033 against 0.0066).
- Autointerp or top-activating-example comparison of the re-aimed siblings.
- Other behaviours, longer training.

## Amendment 1: Γ's gates fixed to W₀ (`results/amend1.json`; `reports/predictions/2026-10-15_prediction_concentration-amendment.md`)

*RunPod, $0.10, 6 min. Checks: W₀ under W₀ gates reproduces 0.00446; do-nothing under state gates reproduces 0.535;
every arm reproduces its run exactly (P(o) diff 0.0, saved states identical).*

**Control has power.** Do-nothing under W₀ gates gives P(o) 0.0045 at the trigger, against a target of 0.996.

**⚠ A1 and A2 reversed; A3 as predicted.** Under W₀'s gates, no arm's components carry the behaviour:

| Arm | CI-masked P(o), W₀ gates / target | KL at trigger (W₀ gates) | CI-masked P(o), state gates |
|---|---|---|---|
| do-nothing | 0.004 | 5.73 | 0.535 |
| Adam, all | 0.006 (A3 passes) | 5.38 | 0.635 |
| SGD, all | 0.19 | 2.08 | 0.999 |
| top 1 | 0.005 | 5.65 | 0.977 |
| top 10 | 0.097 (**A2 fails**) | 2.87 | 0.978 |
| top 100 | 0.024 | 4.15 | 0.918 |
| siblings | 0.371 (**A1 fails**) | 1.57 | 0.978 |

- **The moved subcomponents are gated on at W₀.** The six siblings have g 0.83–0.93 at the trigger, and top1's
  `h.0.mlp.c_fc:1535` has 0.89. So the failure is not that Γ switches the trained subcomponents off.
- **What the numbers imply.** The constrained arms reach 0.98 only when Γ re-gates on the fine-tuned activations. With
  W₀'s mask pattern frozen, their components carry little of the behaviour: siblings 37%, the rest ≤ 19%. The
  behaviour at the trigger therefore depends on a *change in which subcomponents are gated on*, i.e. on Γ (frozen as a
  function) responding to new activations. This is clearest for top 1: one layer-0 subcomponent is moved, and the
  behaviour appears only through gating changes downstream of it.
- **Cause not established.** Which subcomponents switch on at the trigger under the state gates, and whether they
  already write `o` at W₀, was not measured. That is the direct test of the "reveal" reading.

**Revised reading of the round.**
- A sparse update inside the decomposition, including one on the labelled subcomponents, learns the behaviour, keeps
  Pile faithfulness, and is explained by the decomposition at the trigger, but only with Γ's gating on the new
  activations.
- Holding the gating at W₀'s pattern, no constrained update is carried by its components alone.
- Under either gating, the diffuse Adam update is the worst.

**Not run (added).** The gating diff at the trigger: g under the state minus g at W₀ per subcomponent, for each arm.
Also whether the subcomponents that switch on write `o` at W₀ (ablation effect on the `o` logit).

## Amendment 2: which subcomponents switch on at the trigger (`results/amend2.json`; `reports/predictions/2026-10-15_prediction_gating-amendment.md`)

*RunPod, about 1 min of compute. Checks: every arm reproduced exactly; the positive control (all subcomponents take
the state's gates) recovers 1.000 everywhere; the random-set control recovers 0.0005–0.0016.*

**Scores.**
- **B1 fails narrowly.** Recovery by the switched-on set is 0.447 for do-nothing, 0.490 for top 10 and 0.643 for
  siblings; the pass needed ≥ 0.5 in all three.
- **B2 passes.** Jaccard is 0.63 for do-nothing vs siblings and 0.88 for do-nothing vs top 10.
- **B3 passes, with a small effect size.** The mean switch-on effect is 0.0094 for the switched-on union against
  −0.0002 for the null, about 5 SE apart.

| Arm | Gap (log-odds) | Switched on (dg > 0.3) | Recovery by them | Random control |
|---|---|---|---|---|
| do-nothing (= Adam-all's gates) | 5.9 | 135 | 0.45 | 0.002 |
| top 1 | 10.7 | 222 | 0.52 | 0.001 |
| top 10 | 7.2 | 151 | 0.49 | 0.002 |
| siblings | 5.8 | 210 | 0.64 | 0.002 |

- **Different updates switch on largely the same subcomponents.** Jaccard is 0.59–0.88 across all pairs. They are
  off at W₀ at the trigger (g ≈ 0) and fully on after the update (dg ≈ 1). The core recurs in every arm:
  - `h.2.attn.o_proj:858`;
  - `h.3.mlp.down_proj:3273`;
  - `h.0.mlp.down_proj:1914`;
  - `h.3.mlp.down_proj:2102` and `1037`.
- **That set carries about half of the re-gating effect; a random set of the same size carries none.** The rest comes
  from many smaller changes in g below the 0.3 threshold (at 0.1, recovery is 0.53–0.68).
- **Each subcomponent pushes only weakly towards `o` on its own:** 0.009 log-odds on average, against a gap of 6–11.
  So the effect is collective, not one existing "`o` writer" switched on.
- **Do-nothing and Adam-all share their gates by construction**, because Γ reads the same target activations. Do-nothing
  is therefore not an independent arm for B2. Siblings, top 1 and top 10 are.

**Reading.**
- Whatever the update (the diffuse Adam one, a single layer-0 subcomponent, or the six labelled siblings), its
  downstream effect at the trigger runs largely through switching on one shared, pre-existing set of about 150–200
  subcomponents. They were off there at W₀, and together they push towards `o`.
- That is evidence for "reveal": the decomposition already contains machinery the update recruits through Γ's gating.
  It is not a single labelled circuit, and the recruited set explains about half, not all, of the effect.
- Not established: what those subcomponents do elsewhere (their top contexts at W₀), and whether the remaining half is
  more diffuse re-gating or something else.
