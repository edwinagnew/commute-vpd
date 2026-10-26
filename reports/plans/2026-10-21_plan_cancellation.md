# Plan: do the changes a split or component training writes cancel each other? (E17, `experiments/cancellation`)

*2026-10-21, coding agent. Approved class 1 (analysis of saved states, GPU passes of the existing harness on them) and
the named next step of `reports/analyses/2026-10-20_analysis_gamma-refit.md` and of the research log's "What's next" card 2.
No sign-off needed; the check is written so the record shows why. RunPod is down today (user), so the run goes to the
cluster's `debug` QoS and waits for a GPU as long as it must. Vocabulary: `GOALS.md` → Terms.*

## The design check

**Goal served.** `GOALS.md`, "If no simple Φ works": is the failure because post-training teaches new capabilities, or
because the decomposition is too brittle? Every frozen-Γ map that keeps the behaviour fails VPD's adversarial check
while the stochastic, CI-masked and unmasked checks hold (E2, E8, E11, E14, E16). Cancellation is the brittleness
branch made concrete: the map spreads the update over many subcomponents as changes that only reconstruct the model
together, so an adversary that removes some of them breaks it. This run tests that reading directly on the saved
states. It is also control 2 of the re-gating line (`reports/decisions/2026-10-21_decision_regating-main-line.md`): re-gating is
read only on states that pass the adversary, so knowing why they fail decides whether such a state can be reached.

**Nearest thing tried.** E15 stage 0 (`experiments/gamma_refit`, `TRIED.md` E15): the subcomponents the adversary turns
down at the split state are inert one at a time at both models, the adversary's mask does twice the damage at the split
state, and masks do not transfer between the state and the original model. That showed the fragility is a joint effect
of many changes, not which joint effect. The 2026-10-20 reading of the existing numbers
(`reports/analyses/2026-10-20_analysis_adversarial-vs-stochastic.md`) found the adversary's damage needs a selective mask: turning
every maskable subcomponent down together costs a fiftieth of what the adversary's chosen mask costs. Both point at
cancellation without measuring it. This run reverts subsets of the changes and reads the model's output: a functional
test of cancellation, which neither did.

**Patch or step.** A step on the brittleness branch: it explains the failure mode shared by every frozen-Γ map, not
one experiment's flaw.

**Working decisions.** Relies on Γ frozen and on functional measures outranking Frobenius ones: the coherence statistic
is Frobenius and is reported beside the scored KLs, not in their place. Reading A stands. No departure.

**What would change what we do next.** Cancellation confirmed: the least-squares split and component training encode
the update as opposing changes, so the maps to try next constrain placement (the adjoint's coherent write, sparse or
sibling component training), updating Γ alone would not remove the encoding (as E15 found), and the re-gating line
reads its states against the adversary. Cancellation ruled out: the adversary exploits the interaction of its mask with
the original subcomponents at the new activations, or the stale importance box, and the next tests are the adversary's
mask binned by layer and importance and the stochastic check with every layer routed.

**The apparatus arm, first.** Two gates run before any revert is read. G1: the revert hook at full weight on component
training's state reproduces the original model's logits (KL under 1e-4 nats on the same rows), and at zero weight
reproduces the state's own (KL 0); if it fails, the run stops. G2: the coherence statistic's sign-flip null reads 1
within 0.1 on every state. The positive control C1 is an injected cancellation: component training's emoticon pieces
gauge-mixed per matrix, $U' \leftarrow U' A$ and $V' \leftarrow V' A^{-\top}$ with $A = I + \varepsilon E$, which keeps
the weight change exactly while each subcomponent's change gains parts that others lose; ε is set so the changes'
energy is 20 × max(1, κ) times the unmixed value. It must read as cancelling on both statistics. (Written before the
run, replacing a write-side zero-sum perturbation: with the read vectors fixed, $\sum_c r_c v'^\top_c = 0$ has no
non-zero solution when they are linearly independent, which they are on most matrices.)

**The numbers, and what each should read.** Per state, on the first 128 Pile validation rows (one harness block):

| number | definition | W₀ | adjoint (passes the adversary) | hypothesis for the failing states |
|---|---|---|---|---|
| κ, coherence | ‖Σ_c Δ_c‖² ÷ Σ_c ‖Δ_c‖², Δ_c = u′_c v′_cᵀ − u_c v_cᵀ, pooled over matrices | undefined (no change) | > 1, aligned by construction | < 1 (the hypothesis); my expectation is ≥ 1 for every state, see P1 |
| κ null | the same with a random sign on each Δ_c | — | 1 ± 0.1 | 1 ± 0.1 |
| F_full | KL(state ‖ state with every Δ_c reverted) | 0 | the state's KL to W₀ | the state's KL to W₀ (least-squares: to W₀ plus the Δ remainder) |
| h | F_half ÷ F_full, F_half with a random half of the alive subcomponents reverted, 3 halves | — | 0.25 to 0.5 | > 1 |
| X | the adversary's KL at the state minus at W₀ | 0 | ≈ 0 | > 0.3 nats (E11: ratios 2.2 to 6.7 on W₀'s 0.7) |
| F_adv | KL(state ‖ state with the fraction (1 − g_c(x))(1 − a_c) of each Δ_c removed), a the state's own adversary's sources | — | reported | ≥ 3 × F_perm and ≥ 0.5 X |
| F_perm | F_adv with a permuted among the alive subcomponents of each matrix, 3 draws | — | reported | the null for F_adv |
| F_w0adv | F_adv with W₀'s adversary's a | — | reported | ≈ F_perm (stage 0: masks do not transfer) |

**Blind to.** Everything here reads Pile rows, where the adversary fails, and not the trigger; the trained behaviour is
not read. κ is Frobenius and blind to function. Random halves are blind to a specific cancelling subset, which F_adv
covers. The hook keeps the original subcomponents whole and removes only the changes, so it is blind to damage from
masking the original subcomponents under the new activations: that is the alternative explanation and P3's null.

**Noise.** Three halves and three permutations give each F its spread. The adversary runs one seed of 20 steps; E11's
seed floor was f = 0.075 on the ratio. The KLs are fp32 and deterministic given the mask.

## Restatement

The project asks whether a small post-training update can be carried into a VPD decomposition by a simple map so the
decomposition stays faithful. No map tried keeps the adversarial check. This run asks whether that is because the maps
write the update as changes that cancel: it reverts random halves of the changes, and the part the adversary removes,
and reads how much more that costs than reverting them all.

## States

All on the cluster NFS from E11 (`experiments/reading_a_coverage/results/large/<task>/`): component training's step-10
state for the six tasks (emoticon, r0_t352_v310, r1_t534_v369, r1_t588_v452, r2_t275_v2667, r2_t359_v816;
`pullback_s0/state_step10.pt`) and the fine-tune's δW at step 10 (`dW_ft_step10_s0.pt`). From the emoticon δW, the
least-squares split and the scaled adjoint split are rebuilt (`split.split_state`, `split.adjoint_state` with the input
moments over the emoticon training windows, as E14 and E16). Eight states plus the control C1. E11's adversarial ratios
for component training: emoticon 1.1, r1_t588 1.2, r2_t359 1.5 (passing) and r0_t352 2.7, r1_t534 2.2, r2_t275 6.7
(failing); the emoticon least-squares split reads 5 to 7 (E2, E14, E16) and the adjoint 0.9 to 1.1 (E14, E16).

## Predictions (filed in `experiments/cancellation/results/predictions.json` before the run)

- **G1 hook identity** (gate, first): on every state, the hook at zero weight gives KL 0 to the state's own logits and
  at full weight gives KL ≤ 1e-4 nats to the model with the changes reverted in the weights. Fail: stop.
- **G2 coherence null**: the sign-flip κ reads within 0.1 of 1 on every state (3 draws). Fail: κ is not reported.
- **C1 injected cancellation** (positive control): κ ≤ κ_ct ÷ 5 (by construction it reads κ_ct ÷ (20 max(1, κ_ct)):
  a check of the code path) and h > 1 on the control state. Fail: the statistics cannot read cancellation and P1 to P3
  are unscored. Predictions were re-filed with this construction before any run.
- **P1 coherence** (the hypothesis's weight-space claim): κ < 1 on the three failing component-training states and on
  the emoticon least-squares state. *Predicted: fail.* Adam's step is sign-like and a ten-step gradient on a narrow
  task is low-rank, so the columns of dU share one direction and the changes add coherently; the least-norm split over
  an overcomplete frame adds coherently too. If P1 fails while P2 holds, the cancellation is functional and not visible
  in the Frobenius norm, which is what `decisions.md`'s functional-first rule anticipates.
- **P2 half revert**: h > 1 on the three failing component-training states and the emoticon least-squares state, and
  h ≤ 0.5 on the adjoint state. *Predicted: pass*, with low confidence on the first half: the CI-masked check shows
  that turning every maskable subcomponent down together costs little while the adversary's selective mask costs a
  lot, which is what a cancelling set of changes does to partial removals.
- **P3 the adversary removes the cancelling part**: on the same four states, F_adv ≥ 3 × mean F_perm and
  F_adv ≥ 0.5 X. *Predicted: pass.* Null: F_adv ≈ F_perm, the adversary's pattern over the changes is irrelevant and its
  excess comes from the original subcomponents under the new activations.
- Reported, not scored: h and F_adv on the three passing component-training states (the hypothesis says lower h than on
  the failing ones, ordering with E11's ratio); F_w0adv; the adversary's KL at W₀ and at each state against E11's
  ratios; F_full against each state's KL to W₀.

## Cost and platform

One L40 on the cluster's `debug` QoS, about 40 min for the nine states (per state: one CI pass and one fp32 reference
pass over 128 rows, a 20-step adversary, and nine hooked passes), `--time=01:30:00`. Smoke first on the same QoS
(two tasks, 8 rows, 2 adversary steps). RunPod is down, so the runner waits on the cluster beyond the usual 10 minutes.
No cost.

## Not run (by design)

The least-squares and adjoint states of the five ladder tasks (their adversarial ratios are unmeasured, so a reading
would have no anchor); a second update seed; the revert restricted to low-importance subcomponents; the trigger windows.

## Hand-off

Analysis: `reports/analyses/2026-10-21_analysis_cancellation.md` and `notebooks/E17_cancellation.ipynb` through `/writeup`;
`TRIED.md` row E17; a `PENDING.md` line. Then the re-gating line's first control (mechanism or leak), designed in
`reports/plans/2026-10-21_plan_regating-leak.md`.
