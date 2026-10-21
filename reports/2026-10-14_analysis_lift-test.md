# Lift test: obstruction, projection and pull-back (round 2)

**Takeaway.** On the emoticon task, the decomposition can follow post-training if the training happens inside it
(pull-back: faithful at every metric through 30 steps, P3′ passed), but not if a finished fine-tune is translated
onto it afterwards (lift: adversarial ratio 5.4 by step 10, P2′ reversed). Notebook: `notebooks/E02_lift_test.ipynb`.

*2026-10-14, coding agent. Round 2 of `reports/2026-10-14_plan_lift-test.md`. Predictions filed before each
run (`results/predictions.json`, then `results/predictions_amend1.json`). Cluster jobs 20266, 20268, 20281, 20288,
20289. Sections are in the order they were run.*

## Prediction 1 is reversed: the null outcome

- **Obstruction fraction ω** is the share of ‖δW‖² that no first-order change to the alive subcomponents' U, V can
  express (theory Cor. 1, P_U⊥ δW P_V⊥). Whatever map is used, at least this share stays in Δ.
- Arm A (emoticon) at round 1's step 10, pooled over the 20 non-saturated matrices: **ω = 0.540 against a Gaussian
  null of 0.589**. Pass needed ω ≤ 0.1 and ≤ 0.5 × null. Null outcome needed ω ≥ 0.8 × null = 0.471. It is the null
  outcome. Steps 1–5 of the rerun are not needed to decide this: the pass rule requires every checkpoint to pass.
- It is the same in every one of the 20 matrices (ω between 0.88 and 1.2 times its own null), including the paper's
  emoticon matrix `h.2.mlp.down_proj` (0.454 against 0.472). It holds at every round-1 checkpoint of both arms
  (0.539–0.558, nulls 0.578–0.614). Over all 24 matrices ω = 0.36, because 4 matrices are saturated (below).
- **It is not a conditioning artefact.** The ranks equal the alive counts at tolerance 1e-6, and ω barely moves at
  1e-2 (0.5403 against 0.5400).
- **Saturated matrices:** `h.0` and `h.3` `mlp.{c_fc, down_proj}` have one side at full rank, so ω = 0 there by
  construction.
- **Checks:** the lift algebra passes the CPU tests in `tests/test_lift.py`. The Euclidean lift reproduces the
  liftable part to 3e-13. With the layout used, Δ₀ is about 0 on most matrices; the transposed layout gives
  ‖Δ‖² of hundreds to thousands.

## Prediction 2's weight term cannot be scored

- The released decomposition has ‖Δ₀‖² = 67.9, most of it in `h.0.attn.q_proj` (20.5). So the do-nothing map's
  L_F ratio is 1.004 at step 10 and 1.036 at step 300. The filed sanity rule (do-nothing > 1.5) fails, so as filed
  the L_F part of prediction 2 is unscorable.
- A ratio to ‖Δ₀‖² cannot see an update this size. The weight term that does discriminate is ‖Δ′ − Δ₀‖² / ‖δW‖².
  For the Euclidean lift that number is ω, already measured above.

## LoRA arm (job 20268; reported, not scored)

- Setup: LoRAs of rank 1 and 8 on `h.2.mlp.down_proj` (the paper's LoRA site), trained with arm A's loss and data,
  Adam, lr 1e-3. Held-out P(o) reaches 0.99 by step 30 (rank 1) and step 10 (rank 8).
  Numbers: `results/obstruction_lora.json` and `results/finetune_LoRA_r*.json`.
- **ω = 0.46–0.49 against a null of 0.47** at every checkpoint from step 1 to step 100. Same as the full fine-tune.
- **Artefact on the read side.** A LoRA starts with B = 0 and a random A, so A barely moves early on. The read
  direction of the rank-1 LoRA has cosine 1.00 with its random init at step 1, 0.92 at step 10 and 0.67 at step 100.
  So the LoRA's read side is random by construction.
- **Split by side** (`h.2.mlp.down_proj`):
  - **Write side:** share of ‖δW‖² outside col U_A, null 0.534.
  - **Read side:** share outside col V_A, null 0.883.

  | Update | Write side | Read side |
  |---|---|---|
  | LoRA rank 1, step 1 | 0.547 | 0.882 |
  | LoRA rank 1, step 100 | 0.537 | 0.859 |
  | LoRA rank 8, step 100 | 0.538 | 0.857 |
  | Full fine-tune, step 10 | 0.534 | 0.851 |
  | Full fine-tune, step 300 | 0.519 | 0.844 |

- **Reading:** the write side is trained from zero, so it carries no init artefact. It is still no better aligned
  with the alive write vectors than a random direction is. So the LoRA does not rescue the lift.
- **Adam remains a candidate cause, not tested.** An early Adam step on B is sign-like. The SGD control below would
  separate the two.

## Input-weighted obstruction and the SGD control (job 20281; reported, not scored)

**Definition.** The input-weighted obstruction is ω_act = E‖O(δW) a‖² / E‖δW a‖². Here a is the matrix's real input,
taken over Pile val rows 0–63 or over the emoticon training windows. It is pooled over the 20 non-saturated matrices
by energy. Its null is the same quantity for an isotropic δW, (1 − r_U/d_out)·tr(P_V⊥Σ)/tr Σ. The output side is still
weighted equally in all directions. Numbers: `results/weighted.json`, `results/finetune_SGD_lr*.json`.

| Update | ω (Frobenius) | ω_act Pile (null) | ω_act windows (null) |
|---|---|---|---|
| **W₀ itself** | 0.487 | **0.127** (0.603) | 0.119 (0.580) |
| Arm A, Adam, step 10 | 0.540 | 0.297 (0.549) | 0.268 (0.531) |
| Arm A, Adam, step 300 | 0.558 | 0.386 (0.597) | 0.384 (0.609) |
| Arm B, Adam, step 10 | 0.539 | 0.302 (0.544) | 0.285 (0.506) |
| Arm A, SGD lr 0.1, step 1 | 0.341 | 0.363 (0.421) | 0.292 (0.374) |
| Arm A, SGD lr 0.3, step 10 | 0.321 | 0.329 (0.438) | 0.277 (0.404) |
| LoRA rank 8, step 10 | 0.481 | 0.436 (0.447) | 0.411 (0.421) |

- **Frobenius ω is the wrong yardstick.** W₀ itself fails it just as badly (0.49). Half of the weights' Frobenius
  mass is outside the alive spans, held by dead components and Δ. Weighted by real inputs, the alive components carry
  W₀ well: ω_act = 0.13 against a null of 0.60.
- **The fine-tunes sit in between.** For the full fine-tunes ω_act is about 0.3 at step 10: about half the null, and
  about 2.5× W₀'s value. It grows with training (0.39 by step 300). So the update is clearly better aligned than
  noise, but worse aligned than the weights are.
- **Adam vs SGD.** With SGD the Frobenius ω is lower than with Adam (0.31–0.35 against 0.54). Weighted by inputs the
  two are about the same (0.29–0.36). So Adam inflates the parameter-space measure but is not why the update misses
  the decomposition functionally.
  - The SGD runs reach P(o) ≈ 1 within 1–3 steps, at a large off-target KL (0.27–6.6).
  - lr 1 was unstable (P(o) fell to 0.40 at step 10, off-target KL 6–21), so its step-30 value of 0.09 is not used.
- **LoRA ≈ null.** That is expected from its random, barely-trained read side (A), so it is not evidence either way.
- **Still not functional in the KL sense.** This is energy at each layer's output, not effect on the logits.

## Does the behaviour need the obstruction? (job 20288, `results/projected.json`; reported, not scored)

**Setup.** Three models are evaluated:
- W₀ + δW, the full fine-tune;
- W₀ + (δW − O(δW)), the liftable part only. This part has an exact first-order lift onto the alive subcomponents
  with Δ = Δ₀;
- W₀ + O(δW), the obstruction alone.

Measured on round 1's held-out windows and on Pile val rows 256–319.

**Capture.** On Pile, 1 − KL(full ‖ liftable) / KL(W₀ ‖ full). At the `o` position, the same ratio of KLs taken there.

| Update | P(o): full / liftable / obstruction | Capture at `o` | Pile KL from W₀: full / liftable | Pile capture |
|---|---|---|---|---|
| Arm A, Adam, step 10 | 0.999 / 0.998 / 0.63 | 1.000 | 0.020 / 0.015 | 0.88 |
| Arm A, Adam, step 300 | 1.000 / 0.998 / 0.44 | 1.000 | 0.031 / 0.022 | 0.88 |
| Arm B, Adam, step 100 | n/a | n/a | 0.014 / 0.008 | 0.86 |
| Arm A, SGD lr 0.3, step 10 | 0.996 / 0.987 / 0.71 | 0.999 | 0.61 / 0.49 | 0.96 |
| LoRA rank 8, step 10 | 0.999 / 0.954 / 0.89 | 0.995 | 0.030 / 0.010 | 0.38 |

- **The obstruction is mostly not needed.** Dropping it keeps the emoticon behaviour entirely, and keeps 85–96% of the
  full fine-tunes' effect on generic text. The liftable part also moves generic text less than the full update does.
- **The behaviour is partly redundant.** The obstruction alone also gives P(o) 0.4–0.9.
- **LoRA.** The behaviour survives, but less than half of its small Pile effect does.

## Pull-back replay: prediction 3, scored (job 20288, `results/pullback.json`)

- **Rerun.** Arm A's recipe with early checkpoints: held-out P(o) = 0.93 at step 1, so s* = 1 and the scored step is
  3.
- **Learning-rate choice.** The filed rule picks the learning rate with the highest held-out P(o) at step 30. That is
  3e-4 (P(o) 0.998, against 0.996 for 1e-4).
- **Off-target damage.** At 3e-4 the first step moves the model far off target: KL elsewhere in the window is 10.3
  at step 1 and 2.7 at step 3. The W fine-tune has 0.04–0.15. The pulled-back weight change is 19% of ‖W₀‖ by step
  30, against 1.5% for the W fine-tune. At 1e-3 it diverges.
- **Faithfulness at step 3** (ratios to W₀, same harness):
  - stochastic 1.05;
  - **PGD 41**;
  - CI-masked 2.6.

  Δ stays at Δ₀ to 7e-7, as designed.
- **Prediction 3 fails as filed** (PGD ratio > 1.5).
- **Why this is not a fair test of the idea.** The filed learning-rate grid was too large for Adam on U, V, so the
  pull-back steps were 3–100× larger than the fine-tune's. Small updates are assumption A2. The selection rule also
  ignored off-target damage. Even the gentlest run (1e-4) has KL elsewhere of 0.2 against the fine-tune's 0.04–0.15.
- **Reference.** The fine-tuned model under the unchanged decomposition (do-nothing) keeps the ratios at 0.99
  (stochastic) and 1.04 (PGD).
- **A fair version** needs a new filing (dated report): learning rates chosen so that ‖δW‖ matches the fine-tune's
  (about 1e-5 to 3e-5), and selection on held-out P(o) together with KL elsewhere.

## Amendment 1 (job 20289; filed in `results/predictions_amend1.json`, `reports/2026-10-14_prediction_lift-test-amendment.md`)

**Terms.**
- *r* is a faithfulness metric's ratio to the released decomposition on W₀, all in the same harness and job.
- *Unmasked KL (Δ out)* is the KL from the target model to the components alone, run unmasked. It shows whether the
  components carry the fine-tuned model.
- *Gap recovered* is (r_nothing − r_lift) / (r_nothing − 1) on that KL.

### P2′: Euclidean lift, scored. **REVERSED: fails, on the adversarial metric** (`results/lift_faith.json`)

Bears on the weak version (`GOALS.md`): is translating a finished δW a faithful decomposition?

| Rerun step | Stochastic r: lift / nothing | PGD r: lift / nothing | Unmasked r (Δ out): lift / nothing | Gap recovered |
|---|---|---|---|---|
| 1 | 0.99 / 0.99 | **1.14** / 1.04 | 1.06 / 1.85 | 0.93 |
| 2 | 1.03 / 1.03 | **1.55** / 1.12 | 1.38 / 4.21 | 0.88 |
| 3 | 1.03 / 1.03 | **2.03** / 1.13 | 1.40 / 4.25 | 0.88 |
| 5 | 1.02 / 1.02 | **3.83** / 1.20 | 1.21 / 2.90 | 0.89 |
| 10 | 1.00 / 1.00 | **5.41** / 2.18 | 1.20 / 2.84 | 0.89 |

- **The lift carries the update.** The gap recovered is 0.88–0.93 at every checkpoint: unmasked, the lifted
  components reproduce most of the fine-tuned model, where the unchanged ones do not. The stochastic and CI-masked
  ratios stay at about 1.0.
- **But the lifted components are adversarially fragile.** The PGD ratio rises from 1.14 at step 1 to 5.4 at step 10.
  The pass needs ≤ 1.5 at every checkpoint, so P2′ passes only at step 1 and fails overall. Do-nothing also drifts on
  PGD (2.2 by step 10), but less.
- **Δ change** ‖Δ′ − Δ₀‖² / ‖δW‖² is 0.35–0.36, as expected from ω. The measured Δ′ matches the closed form
  O(δW) − dU dVᵀ to 1e-6.
- **Cause not established.** One untested candidate: the lift spreads δW over many alive components with no regard to
  Γ (the locality residual ρ of theory Lemma 2), so an adversary that masks components can strip it out.

### P3′: pull-back at a matched step size, scored. **PASSES** (`results/pullback2.json`)

Bears on the weak version through the pull-back map (simple, user, 2026-10-14): can post-training be done inside the
decomposition?

- **Selected run.** The rule picked lr 1e-5: all three learning rates reached P(o) ≥ 0.9 at step 30, and 1e-5 had the
  lowest KL elsewhere.
- **Behaviour.** Held-out P(o) is 0.946 at step 3 and 0.9998 at step 10.
- **Off-target.** KL elsewhere in the window at step 10 is 0.017. The threshold was 0.110, and the W fine-tune has
  0.055, so the pull-back does about 3× less off-target damage.
- **Faithfulness.**
  - At step 10: stochastic 1.00, PGD 1.03, CI-masked 1.00, unmasked 1.01.
  - At step 30: stochastic 1.00, PGD 1.07.
  - Δ = Δ₀ to 8e-7.
- **Update size.** The pull-back's max ‖δW‖ / ‖W₀‖ is 0.0010 / 0.0026 / 0.0051 / 0.0066 at steps 1 / 3 / 10 / 30.
  The W fine-tune's is 0.0035 / 0.0065 / 0.0112 / 0.0155, so the pull-back moves the weights about half as far.
- **Reading.** The same behaviour, learned inside the alive (U, V), costs nothing in faithfulness by any of the four
  metrics. It also costs less off-target than the unconstrained fine-tune. Translating the finished update instead
  (P2′) is adversarially fragile.

## What this does and does not say (round 2 overall)

- **Says (this task):** post-training *inside* the decomposition (pull-back on the alive U, V, Δ fixed) learns the
  behaviour and keeps the decomposition faithful by every metric, adversarial included, through 30 steps.
  Translating a finished update from *outside* (the lift) carries the update but is adversarially fragile beyond
  step 1.
- **Says:** the part of a normal fine-tune the subcomponents cannot express is not needed for the behaviour. Weighted
  by real inputs it is about 30% of the update (W₀ itself: 13%). The optimiser, Adam or SGD, puts it there.
- **Does not say:** anything about a behaviour the model lacks (arm C), longer training, Γ updates, or whether the
  changed subcomponents stay interpretable. The cause of the lift's fragility is not established.

## Next

Round 3 inputs and plan: `reports/2026-10-14_plan_new-skill-test.md`.
