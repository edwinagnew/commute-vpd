# Plan: the lift test. Can a simple map carry a fine-tune onto the decomposition, with Δ held fixed?

*2026-10-14, coding agent. Round 2, renamed from "transport test" (round 1 tested transports onto a few
subcomponents; round 2 tests the user's precise question, which `theory/commuting_updates.md` Definition 2 calls a
lift). Design check below; confirmed by the user 2026-10-14, who also ruled that the pull-back counts as simple ("simple" = no extra network trained on top). Nothing has been filed or run yet. Round 1's directory
`experiments/transport_test/` keeps its name, since `TRIED.md` row 1 points at it; round 2 lives in
`experiments/lift_test/`.*

## The claim, the null, and what separates them

**Claim.** For a small fine-tune δW (full-weight or LoRA), a one-solve map Φ puts δW on the alive subcomponents, with
Γ frozen and Δ held at Δ₀, and the result is a faithful decomposition of W₀ + δW (weak version, `GOALS.md`).

**Null (brittleness).** Even when the components can carry δW in weight space, the masked forwards break: the lifted
decomposition is much less faithful on W₀ + δW than D₀ is on W₀.

**Predictions** (thresholds proposed here, fixed in `predictions.json` only after confirmation):

1. *Obstruction is small.* The obstruction fraction ω (theory §5, Cor. 1: the share of ‖δW‖² that no change to alive
   U, V can express) of arm A at steps 1–10 is ≤ 0.1 absolute, pooled over the non-saturated matrices, and ≤ 0.5× the
   Gaussian null there. Reported beside it: arm B, the LoRAs, and the gradient-shaped null (Cor. 2).
2. *The lift is faithful (pass / fail on the claim).* The Euclidean lift (minimum-norm, theory §5.1 instance 1)
   evaluated on W₀ + δW keeps the stochastic and PGD masked-reconstruction KLs within 1.5× (κ = 0.5) of D₀ on W₀,
   same harness and batches, with Δ = Δ₀. The do-nothing map must fail the weight term L_F (a check that the measure
   now has teeth). The locality lift (instance 2) is reported beside it; predicted ≤ the Euclidean ratio.
3. *Pull-back replay (the user's map).* Training arm A's loss on (U_alive, V_alive) instead of W reaches P(o) ≥ 0.9 on
   held-out windows within 3× the steps the W fine-tune took, and passes the same κ = 0.5 faithfulness test.

**Falsifier.** Prediction 2 fails at arm A's smallest step, where δW is smallest and a one-subcomponent answer is
known to exist.

## Design check

- **Goal served.** `GOALS.md`, "The question made precise", weak version, directly: the round tests two simple maps,
  the lift of a finished δW and the user's pull-back, against the faithfulness definition with Δ not allowed to absorb
  δW.
- **Nearest thing tried.** `TRIED.md` row 1. It settled that scale-only lifts fail (span fraction 1.5× random) and that
  the k-sparse functional test had no power. It could not say whether a *full* first-order lift exists (theory
  Remark 2), and its faithfulness check let Δ absorb δW. This round measures both.
- **Patch or step.** A step: the maps come from the theory note's definitions, not from round 1's flaws. Carried-over
  patches, labelled as such: a convergence check on every fit, arm A checkpoints at steps 1–5, arm B at a learning
  rate giving ~0.1 nats, the unmasked KL with Δ in and out. **Dropped from round 2** (to `backlog.md`): the k-sparse
  nulls, functional selectors, span{U_c, u_o} write edits, and k = 1 on `2359`. They answer round 1's "few
  subcomponents" claim, which the weak version does not require.
- **Decisions.** Relies on: weak version as target, Γ frozen, LoRA plus full fine-tunes, functional over Frobenius,
  within-harness thresholds. No departures. The compute row (RunPod for round 1) is re-checked: cluster first if the
  queue allows.
- **What would change next.**
  - ω large on non-saturated matrices: no map that only moves alive components can work there; next is reviving
    dead components or updating Γ (backlog), and the "new capability" reading gains weight.
  - ω small but prediction 2 fails: the brittleness branch; next is the Γ update.
  - Prediction 2 passes: continuous interpretability holds for this update in the weak sense; next is LoRA at larger
    steps to find the limit, then the strong version.
  - Prediction 3 alone passes: post-training can be done *in* the decomposition, even if a finished δW cannot be
    translated; it counts as simple (user, 2026-10-14).

## Set-up

- **Updates.** Arm A (emoticon → `o`, full-weight), checkpoints 1, 2, 3, 5, 10, 30. Arm B (continued pretraining),
  learning rate re-tuned to ~0.1 nats held-out effect. LoRA on `h.2.mlp.down_proj`, the paper's LoRA site, at ranks 1
  and 8 on arm A's data (the paper does not give its rank). Round 1's arm-A and arm-B δW checkpoints (steps 10–300)
  are already local in `experiments/transport_test/results/large/`.
- **Stage 0, CPU, no forward passes.** ω per matrix and pooled, r_U/d_out, r_V/d_in, and s_min of dσ (theory §5.2)
  per matrix. Caveat: `h.0` and `h.3` `mlp.{c_fc, down_proj}` have more alive components (1 031–1 850) than d_model
  = 768, so one side may be full rank and ω trivially 0 there. ω is informative only where both ranks are below the
  dimension (all attention matrices, `h.1`/`h.2` MLPs, including the emoticon matrix `h.2.mlp.down_proj` with 358
  alive). Where a side is saturated, s_min is the number that matters: a small s_min means a small δW needs a large
  component change.
- **Stage 1, GPU.** Euclidean and locality lifts of each δW (a closed-form solve per matrix). Score L_F, L_S, L_P and
  the unmasked KL with Δ in and out on W₀ + δW, as ratios to D₀ on W₀. Report ρ, the locality residual (Lemma 2).
- **Stage 2, GPU.** Pull-back replay on arm A's loss and windows, Euclidean steps on (U_alive, V_alive), Δ fixed;
  P(o) curve and faithfulness at matched steps.

## Cost and compute

Stage 0 is CPU linear algebra on files already here. Stages 1–2 are forward passes and one short training run on the
67M model, comparable to round 1 (under 2 GPU-hours). Benchmark one checkpoint's faithfulness first and extrapolate.
MATS cluster first (`c11`, one L40, `debug` QoS for the benchmark); RunPod 4090 only if the queue blocks. Expected
well under $10.
