# Plan: the LoRA split study (design check, awaiting the user's confirmation)

*2026-10-20, coding agent. Status: `/design` check written, nothing built. Source of the request: the user since
2026-10-14; `reports/2026-10-20_review_terminology.md` §7; `backlog.md` "Split study: LoRA and fine-tune under one
table" (2026-10-20). Vocabulary: `GOALS.md` → Terms.*

## The request, restated in the Terms' words (this is what the user confirms)

Train nothing new for the LoRAs. Take every saved LoRA update δW = BA (E9: rank 8 on all 24 matrices at nine steps
to 1000; E7: ranks 1, 4, 8, 32 on `h.2.mlp.down_proj` at steps 10, 30, 100), re-run the emoticon full fine-tune once
and keep its δW at E9's nine steps, and for each update:

1. **Outside share** as four blocks (inside both, write side only, read side only, outside both), input-weighted,
   each beside its random-update null, per matrix and pooled; the **rescaling split**'s share beside them as the
   strictest span.
2. **Both splits** of the update, the least-squares split and the adjoint split: their per-subcomponent updates
   (dU_c, dV_c) saved to `results/large`, and one summary of how concentrated each split is.
3. (Separable, part 2) the standing table for the maps: do-nothing, least-squares split and adjoint split (question
   3) of the fine-tune and of the LoRA at steps 10, 100 and 1000, route split, Δ's share of ‖δW‖², Pile ratios, all in
   one job on one card.

"Outside" means the first-order span dU Vᵀ + U dVᵀ (user, 2026-10-20, "for now"). E8's factors are the same recipe and
seed as E9's and reproduce it to three digits at steps 2 and 100, so E8 is not loaded.

## The design check

**Which goal does this serve?** GOALS's weak version, the half that says Δ may not absorb δW, for the user's
canonical post-training update (a LoRA) against a full fine-tune on the same task, steps and card. It is the
measurement the user has asked for since 2026-10-14 and the first row of the "same table for every update type" rule.

**Nearest thing tried.** E9 (the outside-both block of the rank-8 LoRA along training, with the isotropic null: ratio
0.82 to 1.02 at every step), E8 (fine-tune at step 10: 0.27 against 0.53), E2 and E12 (fine-tunes, one step each).
Those settled that under the first-order span a LoRA sits outside about as much as noise and a fine-tune sits well
inside. They could not say *which side* the LoRA's deficit is on (write, read or both), how the fine-tune's share moves
along its own trajectory at the LoRA's steps, what the strictest span (the rank-1 pieces) holds for either, or what the
per-subcomponent updates look like, because they were never saved. The two maps have never been put on one card at
matched steps.

**Patch or step?** A step: the user's table, not a fix to E9.

**Decisions relied on or departed from.** Relies on: Γ and Δ frozen; LoRA as an update type; "functional outranks
Frobenius" (the blocks are input-weighted); cluster first. No departure. One definition is refined, flagged below as
question 1.

**What result changes what we do next.**
- If the LoRA's deficit is read-side only (write side inside above its null, read side at its null) at every step
  and rank while the fine-tune is inside on both sides, then a LoRA's randomly initialised A is the whole obstacle:
  the next phase (a larger model with a real post-trained sibling) should use full fine-tune diffs, or any LoRA map
  needs a read-side step. If the LoRA is at its null on both sides, LoRA is simply the wrong update type for this
  decomposition.
- If the rescaling share is at its null for both update types at every step (E1: 0.06% for a fine-tune), "strengthen
  the existing subcomponents" is dead as a mechanism for both, and stays dead however long the LoRA trains.
- If the fine-tune's outside-both share falls along its trajectory while the LoRA's does not, "small" updates are not
  the easy case the project assumed; the step sweep in the next phase must start late.

## The numbers, and what each should read

Per update (type × step), per matrix and pooled over the 24 matrices by input-weighted energy (the outside-both block
also pooled over the 20 non-saturated matrices, as E8 and E9 did). A random update of the same per-matrix energy is the
null; its four-block values are exact in expectation, its rescaling share and concentration are measured on two draws.

| Number | Fine-tune (expected) | Rank-8 LoRA (expected) | Null | Positive control |
|---|---|---|---|---|
| Outside-both share, input-weighted | about half its null at step 10 (E8: 0.27 vs 0.53); trend along steps unknown | at its null at every step (E9: ratio 0.82–1.02) | analytic | a synthetic δW = dU Vᵀ + U dVᵀ reads 0; δW = P_U^⊥ G P_V^⊥ reads 1 (unit test) |
| Inside-both, write-only, read-only shares | above null on the read side (its read side is data-shaped, E8) | write side above null, read side at null (A stays random, E9) | analytic | same test |
| Rescaling share (energy inside span{u_c v_cᵀ}) | ~0.1% (E1: 0.06% Frobenius; null 0.035%) | at its null | 2 draws | δW = Σ α_c u_c v_cᵀ reads 1 |
| Concentration of each split: subcomponents carrying half the split's energy (n_half), least-squares and adjoint | reported, against the random draw | reported, against the random draw | 2 draws | — |
| Part 2, Pile PGD ratio and e_Δ of do-nothing, least-squares split and adjoint split at steps 10, 100, 1000 (adjoint: no prior number; its P(o) is the first reading) | split 5.4 at step 10 (E8 on the L40), do-nothing 2.2 (E11); later steps unknown, expected to rise | split 1.26, 1.22, 1.07 and do-nothing 1.17, 1.09, 1.04 (E9, L40S); e_Δ 0.19, 0.11, 0.31 | W₀ under 3 adversary seeds (floor) | the LoRA numbers reproduce E9 within the card's floor |

Gate G1: the saved E9 factors at steps 10, 100, 1000 reproduce E9's algebraic outside-both share (0.574, 0.552, 0.707)
to three digits, since the same code path reads them. Gate G2 (part 2): the split state's P(o) equals the update's to
1e-3, as in the harness.

Cut: an "outside share" of the adjoint split's own (the outside share is a property of δW and the span, not of
the split), E8's factors, any new task. The task is emoticon only (every saved LoRA was trained on it; user, 2026-10-20).

## Three questions for the user

1. **The read-side definition of the four blocks.** GOALS Terms writes the blocks algebraically (P_U δW P_V and so on).
   Input-weighted, those four do not sum to one, because the inputs the subcomponents read (P_V a) and the rest
   (P_V^⊥ a) are correlated on the data. Proposal: split the write side by P_U as written, and split the input a
   into the part linearly predictable from what the subcomponents read, a₁ = E[a | Vᵀa], and the uncorrelated rest a₂.
   Then the four blocks are the energies of P_U δW a₁, P_U δW a₂, P_U^⊥ δW a₁, P_U^⊥ δW a₂; they sum to one exactly,
   the null is exact, and "outside both" is what no first-order change can reproduce on the data. The standing
   algebraic outside-both share (E2 to E9's number) is kept as one extra column, so the old rows stay comparable.
   Alternative: the algebraic blocks as written, reported without a sum. **Recommendation: the proposal.**
2. **Part 2 in or out.** It closes E9's open item (the fine-tune on the same card, at matched steps) for about 30 GPU
   minutes on the cluster and adds no new code. **Recommendation: in**, as the second stage of the same job.
3. **The adjoint split is your "pull-back"** ("pullback AB into subcomponents and update accordingly", 2026-10-20),
   so it should be applied as a map, not only saved. Its weight change is K(δW) = UUᵀδW + δWVVᵀ, which is not δW: the
   decomposition then describes a slightly different model, and a scale c has to be chosen (raised 2026-10-20, never
   answered). Proposal: c by weighted least squares, c = argmin ‖(c K(δW) − δW) Σ^½‖, one scalar per matrix; the
   resulting state (Δ fixed, subcomponents moved by c·(δW V, δWᵀ U)) enters part 2 beside do-nothing and the
   least-squares split at the same steps, and reports the same table plus whether that model still does the behaviour
   (held-out P(o) against the update's). Its per-subcomponent updates are saved either way. K depends on the gauge of
   (U, V); the released decomposition fixes it, and that is stated in the write-up. Alternative: per-subcomponent
   updates and concentration only, no state. **Recommendation: the proposal.**

## Run

One experiment `experiments/lora_split/` (`run.py`, `plot.py`, `results/`, `figures/`); shared code in
`src/commute/split.py` (adjoint split, four blocks with their null, rescaling share, concentration) and
`src/commute/finetune.py` (δW from saved factors). One cluster job on the debug QoS: stage 1 (fine-tune to 1000 steps
with δW saved at nine steps, the splits) about 10 min; stage 2 about 30 min. Smoke then scored, one pod-runner. Budget:
cluster, $0; RunPod fallback under $3. Predictions filed in `results/predictions.json` before the run, from the
expected-value table above once the three questions are settled.

## Not run

Nothing yet.
