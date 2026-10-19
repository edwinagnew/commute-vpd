# Decision: the success criterion is GOALS's weak version as written; the route split is a mechanism reading

*2026-10-19, coding agent, from a chat with the user about goals and progress. User: "lets stick with reading A so far.
open to change at some point in future."*

## The two readings, and the one we use

- **Reading A (GOALS, weak version; in use).** Φ(D, δW) returns a decomposition D' of W + δW that passes VPD's own
  checks about as well as D passes them on W, with Δ not absorbing δW. This is "the decomposition is still a valid
  description of the fine-tuned model".
- **Reading B (the 2026-10-16 route-split hypothesis; not the criterion).** The changed subcomponents themselves carry
  the new behaviour at the trigger under W₀'s CI values (direct share d large). This is a claim about *how* the update
  shows up in the decomposition. Every pull-back and lift so far fails it (d ≤ 0.08 on the 7 ladder tasks). Under
  reading A that is a finding about mechanism, not a failure of the map.

GOALS's "Where an update goes" section (user-approved 2026-10-16) is written in reading-B terms. Proposed edit for the
user, not made: mark that section as mechanism hypotheses that do not decide the weak version.

## What reading A has actually been checked on (correcting an overclaim in chat)

The chat claim "the pull-back passes this on every task" was too strong. What exists:

| Pull-back state | Pile checks run (ratios to W₀) | Adversarial (PGD) ratio | Source |
|---|---|---|---|
| emoticon, steps 10 and 30 | stochastic, PGD, CI-masked, unmasked | 1.03, 1.07 | round 2; concentration test ≤ 1.08 |
| `r0_t588` " will" → " be", step 10 | same | 1.09 | ladder, reported |
| `r2_t359` " we" → " just", step 10 | same | **1.53**, above round 2's 1.5 pass line | ladder, reported |
| other four ladder tasks | none | not measured | ladder "Not run" |

- VPD's checks we run: stochastic-mask reconstruction KL, adversarial-mask (PGD) reconstruction KL, CI-masked KL,
  unmasked KL, each on Pile rows as a ratio to W₀ (`src/commute/faithfulness.py`).
- Not run as a metric: VPD's importance-minimality (sparsity) loss. Only a count of subcomponents with ḡ > 0.5 at the
  trigger was reported on the ladder.
- Weight reconstruction is exact by construction (Δ frozen, W' := Σ U V ᵀ + Δ), so it carries no information.
- Pile rows hold ~3 trigger positions in 131k (`reports/2026-10-15_review_faithfulness-scope.md`), so a Pile ratio
  near 1 mostly says the decomposition's account of the *old* model survived. The task-local checks are
  `retained_ci_masked` (dominated by r) and `retained_pgd` (no power, `reports/2026-10-19_review_pgd-local.md`).

So under reading A: holds on emoticon on every check run, checked once on a common-word task, fails narrowly on the one
rare-word task checked, unmeasured on four tasks. Round 2's artifact ("Pull-back vs Lift") states the same scope.

## Three clarifications about the pull-back, for the record

1. **It is what the artifact and GOALS say.** Freeze Δ and Γ, run the same fine-tune on the alive (U, V). One SGD step
   moves the weights by K(G) = U Uᵀ G + G V Vᵀ, the part of the gradient the subcomponents can express. GOALS line 25
   lists this as an example Φ. It is a map of the *training step* (needs the loss and data), not of a finished weight
   difference: for a LoRA handed over as BA, only the lift applies.
2. **It is not round 1's projection.** Round 1 projected the finished δW onto the span of the 9,900 fixed rank-1
   pieces (rescaling only; 0.06% kept, random 0.035%). The pull-back moves each piece's read and write vectors, a space
   about (d_in + d_out) times larger that holds ~70% of a real update by input weight (round 2, ω_act ≈ 0.3), and it
   does so per step along its own path, so it never needs the fine-tune's δW to be expressible. K is not an orthogonal
   projection with the released non-orthonormal U, V (`reports/2026-10-19_review_update-map-variants.md`).
3. **Every run used Adam on (U, V)**, not SGD (`pullback.train`): the concentration test found no SGD lr inside the
   off-target limit. Adam's scaling lives on the pieces, so the step is the same training recipe run on the pieces,
   not the fine-tune's Adam step projected.

**Unfreezing Δ.** Letting Δ take G − K(G) each step (variant (b) in the variants review) makes the weights land exactly
at the fine-tune's, so it is a map of the given update. It fails the weak version by construction: Δ absorbs an amount
set by the gauge and the optimiser, not by the task. The lift is the clean end-of-training version of the same idea
and is already run. Variant (a), Δ trainable on the obstruction subspace only, keeps the weak version but is not a map
of δW either. Neither is scheduled.

## Hand-off to the LoRA session (paste)

> We are scoring this round on GOALS's weak version as written (reading A): the updated decomposition is faithful to
> W + δW under VPD's own checks, with Δ not absorbing the update. Read the task-local PGD ratio against the W₀ floor,
> the inexpressible share of M, and the Pile ratios first; the direct share d in P1 and P4 is a mechanism reading, not
> the pass criterion. Leave predictions.json as filed and say in the analysis which reading each prediction bears on.

## Other points cleared up in the same chat

- Γ is one network reading every matrix's input at every position (recruit test, 2026-10-19). Designs that assume
  per-matrix gating are wrong; a layer-2 edit moves layer-0 CI values.
- No task the model lacks has been run; the rarest target had baseline P 0.003 (rung 2).

## Not run

Pile faithfulness on four ladder tasks; VPD's sparsity loss as a metric; any task the model lacks; variants (a), (b);
LoRA through either map (running in another session, `reports/2026-10-19_plan_lora-lift-and-pullback.md`).

## Addendum (same day): step or run?

The user assumed Φ acts on a single update step; the experiments scored finished runs. Both are in the files: GOALS
says "small gradient updates" and the theory note (§6.1) frames the lift as lifting a *path* step by step, but round 1
scored δW at step 10, round 2 lifted the finished δW at steps 1 to 10 one-shot, the ladder trained 10 steps, and the
LoRA plan lifts at steps 10 and 100. Only the pull-back is per step by construction. The one-shot lift of a 10-step δW
is a first-order map applied to a non-small update (the dropped δU δVᵀ term and the Δ bookkeeping are exact only in the
limit). Relevant existing number: the lift's adversarial ratio was 1.14 at step 1 (pass line 1.5) and 5.41 at step 10.
Caveat: one Adam step at lr 1e-3 already gives P(o) 0.93 on emoticon, so "one step" is not infinitesimal there.

Recommendation (not a design; goes through `/design`): take the step as the object Φ and the run as the test of
accumulation, i.e. "continuous interpretability" = apply Φ at every step and watch the error grow. The honest per-step
lift (components take each step's expressible part by the lift's preimage, Δ takes each step's obstruction) is the
backlog's variant (b) with the lift's preimage in place of K; the variants review dismissed (b) under the run framing,
and that verdict does not carry over. It can be replayed on the saved emoticon fine-tune.
