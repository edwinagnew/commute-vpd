# Plan: the transport test. Does a small fine-tune land on existing VPD subcomponents?

*2026-10-14, coding agent. Design check for the first experiment; awaiting the user's confirmation. Nothing has
been downloaded, filed or run.*

## The claim, the null, and what separates them

**Claim.** A small gradient update δW from post-training can be written as a small change to a few existing VPD
subcomponents, with the causal-importance function Γ and the residual Δ left untouched, and the transported
decomposition still explains the fine-tuned model. Each update then reads as "these mechanisms got stronger / weaker
/ slightly changed": continuous interpretability.

**Null.** δW is spread across the decomposition no better than across k randomly chosen subcomponents; the
decomposition carries no information about where the update went.

**Predictions the null does not make** (thresholds to be fixed in `predictions.json` before running):

1a. *Emoticon arm, capture (pass / fail on the claim).* After a full-weight fine-tune to the paper's emoticon → `o`
   task, most of the fine-tune's functional effect is recovered by changing ten or fewer subcomponents, and that
   beats ten random subcomponents.
1b. *Emoticon arm, location (secondary, not pass / fail).* The paper's subcomponent `2.mlp.down:2359` or one of its
   five siblings is among the top few by fitted weight. A fine-tune may route the same behaviour elsewhere and
   still commute cleanly, so this only says where the update went.
2. *Continued-pretraining arm.* The Frobenius fraction of δW inside the span of alive subcomponents is at least ten
   times the random-direction value, and scale-only reweighting recovers most of the functional effect.
3. *Not brittle.* The frozen decomposition, transported, is still faithful at W + δW: stochastic and adversarial
   reconstruction KL within a factor of two of the paper's values on W (0.24 and 0.93) for the step counts at which
   predictions 1a and 2 are scored.

**Falsifier.** In the emoticon arm, the most favourable case there is (a one-subcomponent expression of the
behaviour is known to exist), the top-k transported fit does no better than k random subcomponents.

## Two readings of "commute", and which one this is

- **Projection, decomposition frozen** (this plan). Fine-tune the target, then fit δW as changes to the existing
  subcomponents. No VPD training at all. Tests the claim directly and sidesteps the brittleness branch by never
  re-decomposing.
- **Re-decomposition.** Warm-start VPD on W + δW and measure subcomponent drift against a δW = 0 warm-start control
  and the cross-seed baseline (mean max cosine similarity 0.28 for rank-1 subcomponents). Needs the full VPD stack.
  Deferred to `backlog.md`; it becomes the question only if projection fails, or later to find the "limit".

## Design check (per `.claude/skills/design/SKILL.md`)

- **Goal served.** The main question in `GOALS.md`, directly: it is the first operational test of "commute".
- **Nearest thing tried.** `TRIED.md` rows 0a–0d. The VPD paper proposes exactly this projection but did not run
  it; its editing section shows a one-subcomponent answer exists for the emoticon behaviour; vpd-audit tests
  aggregation, not updates. Nothing settles the question.
- **Patch or step.** A step: the first experiment.
- **Decisions relied on or departed from.** None recorded yet. This plan would set three: frozen Γ and Δ for the
  first ladder; "post-training" means fine-tunes we run ourselves; functional (KL) measures outrank Frobenius ones.
- **What changes next.** See "What each outcome changes" below.

## Set-up

**Model and decomposition (public, no training).** Goodfire's 4-layer, 67M-parameter Pile model
(`goodfire/spd/runs/t-9d2b8f02`, 0.3 GB) and its VPD decomposition (`s-55ea3f9b`, 2.9 GB: 38,912 rank-1
subcomponents over 24 matrices, about 10,000 alive, plus Δ and the Γ transformer). Loaded with `param-decomp` at
the `vpd-paper` branch; the vpd-audit repo pins commit `74146b5` and confirms it loads in Torch on one 40 GB GPU in
bf16. `main` has moved to a Qwen MoE target and a 32-GPU JAX path and is not usable for this.

**"Post-training".** The released model has no post-trained sibling, so δW comes from fine-tunes we run. Each is a
full-weight fine-tune of the 24 decomposed matrices only (embedding and unembedding frozen, so δW lives where the
decomposition lives), small learning rate, with checkpoints at 10, 30, 100 and 300 steps so that "small" is a
sweep, not an assumption. We report ‖δW‖/‖W‖ per matrix at every checkpoint.

| Arm | Data | Role |
|---|---|---|
| A. Emoticon → `o` | The paper's LoRA recipe: windows of ±20 tokens around positions where `2.mlp.down:2359` is causally important; cross-entropy on `o` plus a KL penalty on the other tokens | Most favourable case: a one-subcomponent answer exists |
| B. Continued pretraining | Fresh Pile batches at a small constant learning rate | Pure "strengthen": no new skill. Behavioural effect may be tiny at the end of a cosine schedule, so the normalised measures may be ill-conditioned; reported as such if so |
| C. New skill | A narrow distribution the model is bad at (chosen after A and B; candidates: a synthetic rule the model does not know, or a Pile subset it under-fits) | The "new capability" branch. Not in this round unless A and B are quick |
| N. Random direction | Gaussian δW with ‖δW‖ matched per matrix to arm A | Null for the Frobenius measure |

**Transport ladder** (Γ frozen, Δ fixed, alive subcomponents only):

- **T0, scale-only.** δW ≈ Σ_c α_c U_c V_cᵀ. Linear least squares per matrix (Gram matrix of at most 3,584²), then a
  functional refinement that optimises α to minimise KL(f(W + δW) ‖ f(W_T)) on fine-tune and generic batches. The
  Frobenius fit is the literal projection; the functional fit is its ceiling.
- **T1, k-sparse modification.** Free δU_c, δV_c on a set S of k subcomponents, chosen by |α| from T0 or greedily
  by functional gain; a curve of capture against k. Null: k random alive subcomponents. (Free δU on *all* alive
  subcomponents spans every matrix once the V_c span the input space, so T1 only means something at small k.)
- **Null transport.** δW → Δ: the model is W + δW, no subcomponent changes.

**Measures.**

1. Frobenius fraction of δW in span{U_c V_cᵀ : c alive}, per matrix, against arm N (random gives about alive /
   (d_in · d_out), of order 0.1%).
2. Functional capture: 1 − KL(f(W+δW) ‖ f(W_T)) / KL(f(W+δW) ‖ f(W)), on fine-tune data and on generic Pile
   batches, plus the fraction of the fine-tune objective's gain that W_T recovers. Top-k against k random.
3. Faithfulness of the transported decomposition at W + δW: stochastic and 20-step PGD reconstruction KL with the
   frozen Γ, against 0.24 and 0.93 for the released decomposition on W.
4. Arm A only: share of T0 weight on `2359` and its siblings, and their rank among all subcomponents.

## What each outcome changes

- **A and B pass.** Commuting works for strengthening. Next: arm C (new skill) to find where it breaks, then the
  re-decomposition test to find the limit in step size.
- **A fails.** Commuting fails in the most favourable case. Measure 3 says which branch: if faithfulness at W + δW
  collapses for tiny δW, the decomposition is brittle; if faithfulness holds but the update needs many subcomponents
  (T1 curve rises slowly, Frobenius fraction high), the update is intrinsically distributed and the subcomponents are
  not the right objects for this. Either way the project pivots, probably to the toy models with ground truth
  (`backlog.md`).
- **A passes, B fails.** Likely ill-conditioning (tiny effect at the end of the schedule); rerun B at a larger
  learning rate or treat B as uninformative. Not evidence against the claim on its own.

## Cost

67M parameters. Fine-tunes of at most 300 steps at batch 64 × 512 are minutes on an L40; the least-squares fits are
seconds; the PGD evaluation (20 steps, a few hundred sequences) is minutes. One L40, under two hours in all. MATS
cluster first (account `c11`; on 2026-10-14 the queue had 40 pending and 5 running jobs on the single 8-GPU node);
RunPod 4090 via remote-kernels if the queue blocks, inside the $10 cap.

## Infra risks, to verify after confirmation and before filing predictions

1. **Credentials.** Fetching the two Weights & Biases runs needs `WANDB_API_KEY` in the shell (not checked; the
   check was declined by the permission classifier). If absent, the user supplies it in the shell; it is never
   written to a repo file.
2. **Code version.** Pin `param-decomp` to the `vpd-paper` branch or commit `74146b5`; do not use `main`.
3. **Data.** `param-decomp` reads pre-tokenised Pile parquet shards; arm A also needs the positions where `2359` is
   causally important, which means running Γ over Pile text or reusing vpd-audit's cached importances. This is the
   likeliest real blocker and is the first thing to check.
4. **Step size.** The fine-tune must stay "small". The step sweep and the per-matrix ‖δW‖/‖W‖ report are the
   guard; if even 10 steps are not small by measure 3, that is itself a result (brittleness).
