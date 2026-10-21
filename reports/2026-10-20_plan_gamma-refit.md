# Plan: the Γ refit test (design check, awaiting the user's confirmation)

*2026-10-20, coding agent. Status: `/design` check written, revised after the user's three questions (§ Revision), nothing built. Written in answer to "what is the most
pressing question, and the most informative experiment next", while the LoRA split (`experiments/lora_split`) runs.
Vocabulary: `GOALS.md` → Terms.*

## The pressing question

Does the frozen Γ describe the updated model, or is it the thing that breaks? Three findings point at Γ and none
settles it. (1) Under every map, the changed subcomponents carry at most 8% of a new behaviour; the rest runs through
Γ switching existing subcomponents differently on the new activations, or through subcomponents Γ keeps off (E4–E6,
E10, E12). (2) The adversarial failures of the fine-tune's least-squares split (5.4) and of component training on three
ladder tasks (2.2–6.7) need selective masks inside the box Γ's stale CI values allow; stale Γ is the necessary
condition, and whether it is sufficient is open (`reports/2026-10-20_analysis_adversarial-vs-stochastic.md`). (3) No
faithfulness check has power at the trigger, where Γ's stale values define the box, and the apparatus saturates there
even at W₀ (`reports/2026-10-19_review_pgd-local.md`). "Γ frozen" is a decision (`decisions.md`, 2026-10-14), not a
finding, and GOALS already says updating it "may turn out to be essential". Until Γ's role is pinned down, neither a
pass nor a fail of the weak version can be read.

## The proposal, in the Terms' words

Take saved states whose subcomponents already carry an update and refit Γ alone, with U, V and Δ fixed, for a few
hundred steps of VPD's own Γ objective (stochastic-mask reconstruction plus the importance-minimality loss at the
released weights) on Pile rows. Then re-run the Pile checks. Nothing about the update is retrained; the map under test
is "split (or component training), then let Γ see the new model".

States (all on disk or being written by the running job): the emoticon fine-tune's least-squares and adjoint split
states and do-nothing at step 10 from `lora_split` part 2; the E11 component-training states for r2_t275 (6.74),
r0_t352 (2.73) and r1_t534 (2.24) on the cluster NFS; the LoRA's least-squares split at step 1000 (at the floor, with
half the energy in Δ). Controls: W₀ with the same refit (the floor and the refit's own noise); do-nothing with the
refit (Δ holds δW, so the CI-masked model with Δ out must stay at W₀'s).

## The design check

**Which goal.** The weak version directly: whether a simple Φ exists once Φ may include Γ's response to the new
model. And GOALS's own flagged unknown ("updating Γ may be essential"). It also bears on the user's re-weighting
definition (backlog 2026-10-20): re-gating is a reading of the model only if a Γ that has seen the new model switches
on the same set.

**Nearest tried.** Nothing trained. E11 and the adversarial-vs-stochastic analysis (stale Γ necessary, not shown
sufficient); backlog 2026-10-14 "let Γ re-fit for a few steps", never built. Every map so far was scored under a Γ
that never saw W + δW.

**Patch or step.** A step: a new map (split + Γ refit), not a fix to a measurement.

**Decisions.** Departs from "Γ frozen" (decisions.md, user 2026-10-14) for this test only; U, V, Δ stay fixed, so it
is not a re-decomposition. Whether a Φ with Γ steps still counts as "simple" under GOALS is the user's call.

**What changes next.**
- Ratios fall to the floor with the sparsity loss held: stale Γ was the fragility. Φ = split + Γ refit becomes the
  candidate map; every saved state is re-scored under it; "Γ frozen" is revised.
- Ratios stay above 3: the subcomponent changes are unfaithful in themselves. Γ updating is off the table as a
  fix, and the cancellation tests on the same states (the brittleness reading) are next.
- Ratios fall but the sparsity loss rises well above W₀'s: Γ passes by calling more subcomponents important. Then
  the frozen-Γ adversary was the right check and the refit is not a map.

**Numbers.** Per state, before and after the refit, one card, W₀ floor from the same job (3 adversary seeds).

| Number | W₀ + refit | do-nothing + refit | split / component training + refit | control |
|---|---|---|---|---|
| Pile adversarial ratio to W₀ | 1.0 within the floor (0.05–0.2 by card) | emoticon 2.2 → unchanged or lower (Δ carries δW, Γ cannot move it inside) | fine-tune split 5.4 and r2_t275 6.7 → the branches above; LoRA split 1.07 → stays | the refit on W₀ reaches the released Γ's own loss within the step budget (convergence) |
| Importance-minimality loss relative to W₀'s | within 1% | within a few % | **must stay within a few %**; the number that stops Γ cheating | — |
| Switched-on set at the trigger, Jaccard with the frozen-Γ set (reported, Pile-only refit sees few triggers) | — | — | reported | — |

Cut: trigger-local adversaries (no power), any refit of U, V or Δ, any new task or update. Pile-only refit first; a
refit that mixes the task windows in is a second arm only if the first moves the ratio.

## Revision after the user's questions (2026-10-20)

**1. Pile only, or the task in-distribution too?** Γ's objective is self-distillation: every term compares the
masked model with the *current* model's own output on the same input (`losses.compute_losses`: stochastic and PGD
reconstruction KL against `target_out`, plus importance minimality). The data supplies inputs only, so a Pile row
whose natural continuation contradicts the fine-tuned behaviour is not contradictory: at that position the target is
W′'s own prediction. Nothing has to be removed. What the data does set is how often Γ sees a trigger: about 3
positions in 131k on Pile, so a Pile-only refit cannot learn the trigger's gating, and the trigger reading would be
read off a Γ that still never saw one. Proposal: two data arms. Arm P, Pile only (the released run's data; settles
the Pile fragility, which is where every failing number lives). Arm M, Pile with the task's training windows mixed
in as one packed row per batch of 64 (about 1.5% of rows, roughly ten triggers per step; the post-training data
mixed into the pretraining data, which is also what one would do with a real post-trained sibling). The rate is a
free parameter; the write-up shows the Pile ratio and the trigger reading side by side for both arms so the trade-off
is visible, and the rate is varied only if P and M disagree on the Pile ratio.

**2. How long, and cheaper than retraining?** Γ is a global shared transformer (d_model 2048, 8 blocks, MLP 8192,
about 400M parameters plus a 2048 → 38 912 readout), trained for 400k steps at batch 64 jointly with the
components, with a persistent PGD adversary in its loss (`pile_llama_simple_mlp-4L.yaml`). Nobody knows how long a
warm-start refit of Γ alone takes; it is measured, not assumed: checkpoints at 100, 300, 1000 and 3000 steps, the
W₀ arm must stay at the floor at every checkpoint (if its loss drops a lot the released Γ was not converged, which
is itself a finding), and the refit stops where the failing state's ratio plateaus. One step is one 67M forward,
one Γ forward, the masked forwards and a backward through Γ: benchmark ten steps before anything is filed. The loss
stack is the released one minus the faithfulness term (constant with U, V, Δ fixed), adversary included, so the refit
is like for like with how Γ was fitted at W₀. A readout-only refit was proposed and dropped (user, 2026-10-20: too basic); the
refit trains all of Γ. One cheaper cut stays, **no training at all**, as a diagnostic rather than a map: save the adversary's mask on a failing state
(backlog item), and score the subcomponents it turns down by their attribution at W′ against W₀
(`attribution.ablation_effect`). If those subcomponents matter more at W′ than Γ says, Γ is stale; if their
attribution is unchanged, the changes cancel. This reads the branch without a refit and is nearly free, so it is
stage 0 of the job. A refit is still needed to show that a Γ that has seen W′ is a map, not only that the old one
is stale.

**3. The clean version.** Agreed that the first draft was assembled from old states made on three cards. The clean
run is one job on one card with every state made fresh and its floor from the same job: the emoticon fine-tune to
step 10, its do-nothing, least-squares split and component training, and W₀, all scored under the released Γ and
under the refit Γ, with 3 adversary seeds and the step sweep above. The adjoint split enters once `lora_split` has
given its first reading, and the ladder task where component training fails worst (r2_t275, 6.7) is the second job,
not the first. One seed for the update first; the second seed only if a verdict sits near the line. Rough size:
5 refits (4 states plus W₀, arm P) and one more (the fine-tune's split, arm M), each to 1000 steps with four
harness evaluations. Cost of a full-Γ refit, from the arithmetic (user, 2026-10-20: "how long to do Γ properly"), to
be replaced by a ten-step benchmark before filing: Γ is about 500M parameters with its input projection and readout;
one step at the released batch (64 rows of 512 tokens) is about 100 TFLOP for Γ's forward and backward plus a few
forwards of the 67M model, so about 0.4 s on an H100 and about 1.2 s on an L40 (batch halved for memory). Per
refit: 1 000 steps about 7 min, 3 000 about 20 min, 10 000 about 70 min on an H100. The six refits of the emoticon
job to 3 000 steps are about 2 h on an H100, roughly $6 at live H100 prices (check https://gpus.skypilot.co/), plus
about 30 min of harness evaluations; to 10 000 steps about 7 h and $20, which needs confirmation. Refitting to the
released run's own length (400k steps) would be about 45 h, re-decomposition scale, and is not proposed: the refit
warm-starts from the released Γ and the sweep finds where the ratio plateaus.

## Run

Code: `src/commute/gamma.py`, a refit function on param-decomp's loss stack (`losses.compute_losses` with the
released coefficients, faithfulness term dropped, all of Γ's parameters trainable); one experiment
`experiments/gamma_refit/` with the one frozen config. Stage 0 (adversary mask saved, attribution of the masked set)
and a ten-step benchmark on the cluster `debug` QoS, then the scored job on an H100 via `/pod-runner`. Predictions
filed before the run from the table above plus the step sweep's plateau rule.

## Built (2026-10-20, after the user's choices: emoticon only, fresh states on one H100, arm P for all five states and arm M for the fine-tune's split, full Γ, sweep to 3 000 steps, stage 0 first)

- `src/commute/gamma.py`: `Refit` / `refit` train every parameter of Γ (`model.ci_fn`) on the released loss stack read
  from the released `final_config.yaml` (`vpd.run_config`) minus `FaithfulnessLoss`: importance minimality at the
  annealed p = 0.4 (coeff 2e-4), `StochasticReconSubsetLoss` (0.5), `PersistentPGDReconLoss` (0.5, Adam adversary lr
  0.01, 2 warm-up steps, scope per batch per position). Their `compute_losses`, `PersistentPGDState`,
  `calc_causal_importances` and `recon_loss_kl` do the work; the ~25-line loop body mirrors `run_param_decomp.py`
  lines 258–441 with three differences: only Γ is in the optimizer, the target forward is detached (U, V fixed), and the
  64-row batch runs as 8 micro-batches with one persistent-PGD state per slot (Codex adversarial review 2026-10-20 caught
  a single state stepped per micro-batch, fixed). At the pinned commit `init_pd_checkpoint` is never read, so their
  `optimize()` cannot warm-start from the released decomposition; that is why the loop body is ours. AdamW at the
  released ci_fn start lr 5e-5, constant (their cosine schedule ended at 5e-6); the W₀ + refit arm is the check on that
  choice. Also in `gamma.py`: `pgd_sources` / `mask_kl` (the harness adversary with its sources kept), `adversary_groups`,
  `ablation_kl`, `mean_gates`, `trigger_gates`, `s_on` (the recruit test's rule).
- `experiments/gamma_refit/run.py`: stages `prep` (fine-tune step 10, states w0 / nothing / ls / adjoint / ct saved to
  `results/large/`, W₀ floor over 3 seeds, stage 0, ten-step benchmark) and `refit` (six refits in the order ls_P, w0_P,
  nothing_P, ls_M, ct_P, adjoint_P; checkpoints 100 / 300 / 1 000 / 3 000; the JSON is rewritten after every
  evaluation and refit; finished refits are skipped on re-run, and a re-run `prep` keeps them). Per checkpoint:
  `harness.measure` (route split, Δ share, Pile ratios), sparsity loss ratio, S_on and its Jaccard with the state's
  released-Γ set; Γ's final parameters (bf16, ~1.1 GB), mean CI per checkpoint and the trigger gates at steps 0 and
  3 000 in `results/large/`. Predictions filed in `results/predictions.json` (P1–P6, S0, gates G0–G2).
- Stage 0 definitions: "turned down" = the adversary's source below 0.5 (the subcomponent stays at its CI value); groups
  of 48 by the gauge-free relative change of u vᵀ; ablation KL on 8 val rows at the state and at W₀.
- Run: RunPod H100 via pod-runner, smoke then `prep` then `refit` on one pod; cap about $13 (user, 2026-10-20: "a
  little over $10 is fine"; "cache as you go").

## Hand-off for the analysis session

Read this report, `experiments/gamma_refit/results/predictions.json`, then `results/gamma_refit.json` (`score`,
`refits.<tag>.checkpoints.<step>`, `stage0`, `benchmark`) and `results/large/full.log`. Write `plot.py` (one figure per
claim: Pile PGD ratio against refit step per state with the W₀ floor band; sparsity ratio and S_on Jaccard on the same
steps; stage 0's three groups), the analysis report by `/writeup`, a `TRIED.md` row (E14) and the research-log card.

## Run (added 2026-10-20 evening)

The pod's benchmark read 2.29 s/step (53.7 GB peak), six times the arithmetic above, so one refit fit the user's cap:
the least-squares split, arm P, plus stage 0. Pod terminated 19:18 BST at $10.07. Analysis:
`reports/2026-10-20_analysis_gamma-refit.md` (E15). The W₀ control is the user's chosen next run (option 1, 20:08 BST).

**W₀ control (w0_P), second pod, 20:09–22:26 BST, $9.10; `results/gamma_refit.json` now holds ls_P and w0_P;
log `results/large/full_run2.log`.** Pile PGD ratio to W₀ at steps 0 / 100 / 300 / 1000 / 3000: 1.00 / 2.57 / 3.53 /
6.27 / 2.63; sparsity ratio 0.94 / 1.02 / 1.01 / 1.09 at 100 / 300 / 1000 / 3000; stochastic and CI-masked ratios within
8% of 1; S_on at most 10 subcomponents. **P1 failed as filed:** the refit at the released start lr breaks the released
Γ's adversarial check by 2.6–6.3× while every other check holds. Both arms share the same first steps (importance term
20× its baseline on step 2; the persistent adversary's loss term rises over the run, 0.42 → 0.63 at W₀ and
0.42 → 1.10 on the split, as its random-start sources strengthen), so ls_P's fall to 1.06 at 3000 (below the control's
2.63, at sparsity 1.60 against 1.09) is not readable as a repair until a refit protocol leaves W₀ at the floor.
Candidates for `/design`: warm up the persistent adversary with Γ frozen; lr at the released end value 5e-6; fewer
steps. The remaining refits (nothing_P, ls_M, ct_P, adjoint_P, about $9 each) wait on that decision.
Design check written 2026-10-21: `reports/2026-10-21_plan_gamma-refit-protocol.md` (adversary warm-up at lr 0 for 300 steps,
then the released end rate 5e-6; tested on W₀ alone first). The control is folded into the analysis, notebook and log.

## Not run

The W₀, do-nothing, component-training and adjoint refits and arm M (see the analysis report's "Not run").
