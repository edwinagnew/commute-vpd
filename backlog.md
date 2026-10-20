# Backlog

Deferred ideas and loose ends. Date each entry.

- 2026-10-14. Re-decomposition test (the strong version in `GOALS.md`; postponed by the user until the weak version
  is settled): warm-start VPD on W + δW from the released decomposition and measure subcomponent
  drift against a δW = 0 warm-start control (optimisation noise) and the cross-seed baseline (0.28 MMCS). Needs the
  full VPD training stack; only becomes the question if the split tests fail or to find the "limit".
- 2026-10-14. Ground-truth calibration on the repo's toy models (TMS, ResidualMLP, CPU): "strengthen" = change feature
  probabilities or importances; "new capability" = add a feature. Gives the strengthen / new-capability dichotomy a
  known answer to validate the measures against.
- 2026-10-14. Updating Γ as part of Φ (user, 2026-10-14: "very fiddly but may well be essential"). Ignored for now;
  revisit once the frozen-Γ maps are understood. Simplest form: let Γ re-fit for a few steps.
- 2026-10-14. **Round 2 inputs** (planned 2026-10-14 as round 2, `reports/2026-10-14_plan_lift-test.md`; the k-sparse
  nulls, selectors, span{U_c, u_o} edits and k = 1 on `2359` were dropped from it and stay here). Sources: "What this changes" in
  `reports/2026-10-14_analysis_transport-test.md` and in `reports/2026-10-14_review_round1-audit.md`, and §7 of
  `theory/commuting_updates.md`.
  - From the user (2026-10-14):
    - test component training, i.e. apply the post-training step to U and V instead of W;
    - test LoRA updates alongside full-weight fine-tunes;
    - target the weak version (`GOALS.md`).
  - Measurement fixes:
    - faithfulness with Δ held at its W0 value, so the components must carry the update;
    - a convergence check on every functional fit (final training loss below the zero-edit loss; round 1's arm-B fits
      at lr 1e-2 diverged);
    - the unmasked KL with Δ in and out as the sensitive faithfulness number.
  - Nulls and selectors:
    - a scale-only k-sparse null that cannot win by expressivity alone;
    - write edits constrained to span{U_c, u_o};
    - a functional selector (greedy gain, or CI on the fine-tune data) instead of Frobenius contribution;
    - k = 1 on `2359` directly;
    - selectivity on generic Pile text (KL of f_T to f_W0 outside the windows).
  - Arms:
    - arm A checkpoints at steps 1–5;
    - arm B at a learning rate giving ~0.1 nats held-out effect.
  - From the theory note: the obstruction fraction ω, the locality residual ρ, and a gradient-shaped null for ω.
  - Unexplained, to look at: arm A's Frobenius fraction relative to random grows 1.7× → 10.2× from step 10 to 300,
    while arm B's falls.
- 2026-10-14. Arm C (new skill) still not run; it only makes sense once the round-2 null has power.
- 2026-10-14. Formalisation: written as `theory/commuting_updates.md` (user's request; not yet reviewed by the user).
  It defines the weak version as a lift in the geometric sense, the strong version as an approximately commuting square, the outside share ω and the locality residual ρ. Its §7 candidates go through `/design` before any run.
- 2026-10-14. RL post-training (user). Once the LoRA case is settled: an RL arm on the same model (e.g. REINFORCE with
  a simple reward) at matched ‖δW‖. Compare its outside share ω and the drift of the frozen Γ against the
  LoRA arm. Risks to check:
  - RL updates may avoid principal directions, so they could have a high ω;
  - on-policy distribution shift may break a frozen Γ;
  - errors may accumulate over long runs.
  Feasibility and fairness scoped in `reports/2026-10-14_analysis_rl-feasibility.md` (2026-10-14): match on
  behavioural effect rather than ‖δW‖; first gate is faithfulness of the unchanged decomposition on on-policy rollouts.
- 2026-10-14. **Component-training variant: let Δ take what the pieces cannot** (user's question; analysis by a coding agent, not
  run). The question was whether the leftover Δ could absorb the part of each gradient G that the pieces do not take.
  - **What component training drops.** The plain SGD step of component training is δW = −η K(G), with K(G) = U_A U_Aᵀ G + G V_A V_Aᵀ. K is
    zero exactly on the obstruction subspace (I − P_U)·(I − P_V). Elsewhere it reweights G instead of copying it, so
    "the part of G not put into a piece" has two readings.
  - **(a) Obstruction only (principled).** Give Δ a trainable term restricted to the obstruction,
    Δ = Δ₀ + (I − P_U) D (I − P_V). D gets its own optimiser state, which avoids the Adam bookkeeping problem. This is
    the per-step version of what the least-squares split already does at the end (Δ′ = Δ₀ + O(δW)). The pieces still carry everything
    they can express.
  - **(b) Everything left over.** Δ += −η (G − K(G)), so the weights follow the normal fine-tune exactly (for SGD). This
    is the trivial map with a partial split at each step. It also puts expressible-but-reweighted directions into Δ.
    Under `GOALS.md` that counts as cheating, because Δ may not absorb δW.
  - **Is (a) cheating?** Not if it is scored honestly. Δ has its own mask in the stochastic and PGD forwards, so an
    adversary can switch it off: behaviour stored there is not free. But it is unexplained behaviour. Report:
    - ‖Δ − Δ₀‖² / ‖δW‖², the share of the update left unexplained;
    - the unmasked KL with Δ excluded, which shows whether the pieces alone carry the behaviour.
  - **Expectation.** Round 2 shows the obstruction is not needed for the emoticon task, yet a free optimiser puts about
    30% (input-weighted) there. Variant (a) probably recreates that split, and with it some of the lift's adversarial
    fragility.
  - **Where it would help.** It may matter for a behaviour the model lacks (round 3), if the pure component training cannot
    learn it. The comparison "pure component training vs (a)" would then measure how much of a new skill needs unexplained
    capacity, which is a direct handle on "new capabilities vs brittle". Worth adding as a reported arm to round 3 only
    if the pure component training fails there.
  - **User's framing (2026-10-15): unfreeze Δ in component training so it absorbs some of the obstruction.** For: the
    weights then move the way LoRA wants. Against: whatever Δ absorbs can accumulate over steps, so the pieces may
    end up carrying little and nothing is learned about the decomposition. Measure that accumulation
    (‖Δ − Δ₀‖² / ‖δW‖² along training) if this arm is run. 2026-10-16: this arm is the one that makes the
    residual share e_Δ of the route split informative (`reports/2026-10-16_prediction_ladder-routes-amendment.md`);
    the functional version of the accumulation check is e_Δ along training.
    2026-10-19: pros and cons of (a) and (b) against the run maps in `reports/2026-10-19_review_update-map-variants.md`
    (drop (b): the adjoint split applied per step. 2026-10-20: (b) one-shot on a finished update is the user's map, see the LoRA split entry; (a) only at the first rung Δ-fixed component training fails).
- 2026-10-15. **Round 3 (new skill) shelved** by the user: too much about the emoticon case is not understood yet.
  Plan kept at `reports/2026-10-14_plan_new-skill-test.md`.
- 2026-10-15. **Is the part of an update outside the subcomponents a flaw of the update or of the decomposition?**
  (user). The two readings to separate: LoRA (or the optimiser) is inherently noisy, or VPD inherently misses some
  structure the model uses. A seed-consistency test of the obstruction was proposed and rejected by the user as not
  addressing this. Needs a design that separates the two directly.
- 2026-10-15. **Measurement fixes required before any new round** (user: "something that needs to be fixed";
  `reports/2026-10-15_review_faithfulness-scope.md`). Status after the same day's harness work:
  - **Task-local faithfulness: done** (`src/commute/local_faith.py`, `tests/test_local_faith.py`): the variants on
    held-out windows with the trigger separate, plus a trigger-objective PGD; scored as *retained* (each variant's own
    log-odds gain over W₀ divided by the target's). **Correction:** do-nothing cannot be the positive control for the
    Δ-excluded variants: with Δ out it is identically the W₀ masked model, so it reads 0 by construction (shown in
    the test). The discriminating control is placement: the gated-off component training (ladder gate G1).
  - **Noise floor:** the ladder's `noise` stage (Pile ratios on W₀, 3 seeds; local retained, 3 harness seeds and 2
    optimiser seeds). Not yet measured.
  - **Component training against Γ's values:** the identity test showed Adam component training spreads its update over thousands of
    components, so a per-component comparison is moot; the functional version (CI-masked retained at the trigger
    against Γ's coverage of the gradient) is the ladder's claim.
  - **Save every trained state: done** in `component_training.train` (states written at the save steps).
  - **"Blind to" line per metric: done** as a rule in `.claude/CLAUDE.md`; the ladder's predictions carry one.
- 2026-10-15. **Round 3 = the task ladder** (`reports/2026-10-15_plan_task-ladder.md`, confirmed by the user; done, see the 2026-10-19 entry). Deferred
  from it: rungs 3 and 4 (rare and never; rung 4 is the shelved new-skill test's question), a per-window PGD adversary,
  Pile side-effect checks at every rung.
- 2026-10-15. **Rank-r component training on the LoRA's matrix** (then called "LoRA through component training"; run on one matrix in
  `experiments/lora_commute` 2026-10-19 and on all 24 in E13. **2026-10-20: this was never the user's LoRA question**, which
  is the LoRA split entry of 2026-10-20 below; the 2026-10-15 definition came from three agent-offered readings, none of
  them the user's, `reports/2026-10-20_review_terminology.md`). **Defined via `/design`
  (user, 2026-10-15): arms (i) + (ii)**, both on `h.2.mlp.down_proj` alone, with Γ and Δ frozen.
  - (i) Same matrix: train only that matrix's alive offsets, W′ = (U + dU)(V + dV)ᵀ + Δ; the other 23 matrices are
    frozen. This matches the LoRA's support but not its capacity.
  - (ii) Same rank: dU = P Sᵀ, dV = Q Tᵀ (P ∈ ℝ^{d_out×r}, Q ∈ ℝ^{d_in×r}, S, T ∈ ℝ^{n×r}), so rank δW ≤ 2r to first
    order. This matches the LoRA in both support and capacity.
  - Compared against a normal LoRA of rank r on the same matrix.
  - **(iii) dropped** (user): appending the LoRA as a new subcomponent with its own gate assumes that every update
    adds a new capability, which is the thing under test. It also cannot say whether the update is interpretable by
    the *existing* subcomponents. Its gate would also be either a trained net (not "simple") or always-on (Δ absorbing
    δW).
  - Still needs the task-local harness above first.
- 2026-10-15. **Does an update load on the subcomponents VPD says matter for the task?** (user; discussed, never
  measured directly). Correlate each subcomponent's share of an update (e.g. the LoRA's split onto (u_c, v_c))
  with its causal importance g_c on the fine-tune data, at the trigger positions and on generic text. The hope: LoRA
  mostly updates subcomponents Γ already rates important for that task. Close to the "component training against Γ's values"
  fix above, which does this for the component-training update only.
- 2026-10-19. **Define "create" and "recruit" in VPD terms (user: needs fleshing out; user thinking it over).** Every
  verdict so far reads the route split (`GOALS.md`, "Where an update goes") informally. "Recruit" has two readings:
  (i) existing subcomponents used in their old roles in a new context (Γ fires where it did not; the write direction is
  unchanged), (ii) existing subcomponents repurposed (u rotates, as the siblings' write vector did by ~45° in the
  concentration test). "Create" also has two: (iii) a direction in Δ that no subcomponent carries, (iv) recruitment of
  subcomponents Γ never gates on. Which combination of route shares (direct, re-gating, interaction, masked off,
  residual) plus per-component ledger entries (`theory/component_identity.md`) counts as which verdict is not written
  down anywhere, and neither is what a recruited subcomponent must have done *before* the update for "recruit" to mean
  more than "any perturbation switches these on". Output wanted: a short decision report fixing the mapping, ideally
  before the ladder's routes are scored. 2026-10-19: first data for it in `reports/2026-10-19_analysis_key-value-task.md`:
  the identity control is carried by re-switching (r 0.76–0.80), the random cipher by subcomponents Γ does not switch on
  (e_gate 0.75–0.86), in one format; under frozen Γ the sparsity loss cannot tell maps apart (it reads W′ only).
- 2026-10-19. **From the recruit test** (`reports/2026-10-19_analysis_recruit-test.md`). (a) Is the recruited set the model's
  existing `o` machinery at W₀? Needs about 600 Pile val rows for 200 positions whose next token is `o` (256 rows gave 96),
  or a better definition of "natural" than next-token identity. (b) The switched-off set (about 130 subcomponents per arm)
  is unexamined. (c) Γ is a global shared transformer over all 24 matrices' inputs and all positions: every "blind to"
  line about trigger-local gates should say that the gate at the trigger reads the whole window, and the ladder's
  per-layer or per-matrix locality arguments should be re-checked against that (G1b still holds). (d) r was the same for
  `o` and for the different-target nulls (0.55 to 0.60): r measures how much a trigger-local behaviour can be routed
  through existing gates at all, so the ladder's P2 (coverage predicts r) should be read with that in mind.
- 2026-10-19. **From the task ladder** (`reports/2026-10-19_analysis_task-ladder.md`). (a) Round 4: Γ updating vs constrained placement (sparse/sibling component training, component training against Γ's values) on the rung-2 tasks, where e_gate dominates; needs `/design`. (b) A G1b that learns (more steps or larger lr on the 21 gated-on `h.3.mlp.down_proj` components) before any "d is small" claim. (c) Rungs 3 and 4, once d has somewhere to go. (d) PGD check: **done 2026-10-19** (`reports/2026-10-19_review_pgd-local.md`): retained_pgd has no power (seed spread 55–88 nats against 4–15 nat gains, sign flips) and stays reported-only; the window adversaries saturate because the windows are alike (one context for a shared mask), not because of the position count. Open, needs `/design`: the 128-document Pile adversary read at the task windows' trigger, the one remaining test of "fragile at the trigger". (e) Switched-on set and cross-task Jaccard (amendment's reported list) never implemented. (f) Why r2_t275's step-10 kl_other is 0.82 (0.49 at W₀). (g) Merge the nested `results/results/large/` into `results/large/`.
- 2026-10-19. **Reading-A coverage gaps** (`reports/2026-10-19_decision_success-criterion.md`). **Done 2026-10-19**
  (`reports/2026-10-19_analysis_reading-a-coverage.md`): Pile ratios and the sparsity loss on all five remaining ladder tasks,
  states saved on the cluster NFS (`/mnt/nw/home/e.agnew/commute/experiments/reading_a_coverage/results/large/` on the cluster NFS, 2.3 GB; `cluster/sync.sh large`). The component training fails the adversarial check on r0_t352 (2.73),
  r1_t534 (2.24) and r2_t275 (6.74); r2_t359's 1.53 is real (3 seeds 1.38–1.55). Open, through `/design`: (a) matched
  kl_other instead of matched steps (component training's off-target KL is 1.6–2.5× the fine-tune's at step 10 and the PGD
  excess orders with it); (b) component training at the step where its kl_other equals the fine-tune's step-10 value; (c) whether
  the excess sits in the subcomponents Γ switches on at the trigger; (d) state-level PGD floors (block spread at the
  component-training states is 2–3× W₀'s). The sparsity loss stays reported-only while Γ is frozen (follows W′ only).

- **LoRA with a non-random read side** (2026-10-19, from the LoRA commute test, `reports/2026-10-19_analysis_lora-commute.md`). **Dropped 2026-10-19** at `/design` (`reports/2026-10-19_review_lora-failure-and-fixes.md`, 3.1, `TRIED.md`): the write-only half of `pb_r`, already measured; the L40 rerun of the split is folded into `experiments/lora_matched` as its control.
  A LoRA's A factor stays near its random init (|cos| 0.92–0.98 at the scored step), so its outside share sits at the
  isotropic null and its least-squares split leaves a third of the behaviour in Δ. Candidate arm: train A before B, or initialise A from the
  matrix's alive read vectors, then split; the test is whether the outside share drops below W₀'s own 0.13–0.21 and e_Δ
  towards the full-weight split's 0.085. Also pending: a 5-minute L40 rerun of round 2's least-squares split at step 10 to close the 4090-vs-L40
  PGD miss (7.3 vs 5.4). Through `/design`.

- 2026-10-19. **Key → value task follow-up (round 6, `reports/2026-10-19_analysis_key-value-task.md`; parked by the user:
  "trying to do too much at once").** (a) The matched-size Pile check: round 2's Pile harness on the saved component training
  states (cluster NFS, `experiments/kv_task/results/large/<task>/pullback/state_step{100,300}.pt`) for ident32 at steps
  100/300 (‖δW‖/‖W₀‖ 0.0141/0.0151, matching rand32's scored 0.0154) and rand32 at 100/300. Separates update size from the
  number of new associations as the cause of rand32's PGD 2.81 (filed pass, reversed). ~10 min on an L40, $0. (b) If size
  is not the cause, K between 1 and 32. (c) A second bijection seed at K = 32; adversary seeds on the scored states.
  (d) Note for `decisions.md` (proposed, not made): under frozen Γ, VPD's sparsity loss depends on W′ only, so reading A
  has four discriminating checks, not five. (e) The LoRA arm on these tasks. Full "Not run" list in the report.
- 2026-10-20. **Rank-8 component training on all 24 matrices: run 2026-10-20 (TRIED E13; its files were deleted 2026-10-20 at the user's request, TRIED's row is the record): faithful, Pile PGD 1.14/1.28, Δ untouched; one-matrix control 4.5.** It was run under the name
  "LoRA component training" in answer to the user's repeated request ("I feel like I've said a million times i want to test lora
  pullback"), but no LoRA is trained in it; it answered the 2026-10-15 backlog definition, not the user's
  (`reports/2026-10-20_review_terminology.md` §3). The user's request is the next entry.
- 2026-10-20. **Split study: LoRA and fine-tune under one table** (the user's request since 2026-10-14, never built;
  `reports/2026-10-20_review_terminology.md` §7; user 2026-10-20: "train a rank r lora matrix AB, possibly to completion.
  Then split it into a sum of updates on subcomponents. Study how much lies outside the span of the subcomponents").
  Decision (user, 2026-10-20): "outside" is measured against the first-order span dU Vᵀ + U dVᵀ, for now.
  Train nothing for the LoRAs: load the saved factors (E7 ranks 1 to 32 on one matrix, step 100; E8 rank 8 on 24
  matrices, step 2; E9 rank 8 at nine steps to 1000). Take the least-squares and adjoint splits; save the per-subcomponent
  updates; report the outside share as four blocks (inside both sides, write side only, read side only, outside both)
  with the random-update null and the rescaling split's share beside them, per matrix and pooled. Through `/design`.
  - **Fine-tune side, what exists.** The least-squares split and the outside-both block with its isotropic null have
    been run on finished fine-tunes: emoticon step 10 (E8: 0.27 against 0.53; E2 Adam and SGD, Frobenius and
    input-weighted) and the key-value tasks at two seeds (E12: ratio to null 0.59–0.63). One step per task, one card
    per run, reported as a side quantity under other headings.
  - **Fine-tune side, not run** (so the fine-tune enters the same table as the LoRA): the adjoint split on any δW; the
    four blocks (`split.weighted_fractions` reports only the outside-both block); the per-subcomponent updates saved
    and studied; the outside share along the fine-tune's own trajectory at the LoRA's checkpoints (E9 did this for the
    LoRA only); the rescaling split beside it (E1 only, round 1). No fine-tune δW is kept on disk (`recruit_test`'s
    `ft_*_dg.pt` are gate differences), so the emoticon fine-tune is re-run and its δW saved at the same steps as E9.
  - **Task.** Every saved LoRA (E7, E8, E9) was trained on the emoticon task (round 1's windows), so the table is
    emoticon-only, with the emoticon fine-tune and random null beside it. The key-value tasks have fine-tune splits
    (E12) but no saved LoRA; extending the table there means training one.
  - **Rule from here (user, 2026-10-20):** every update type (full fine-tune, LoRA by rank, random null) goes through the
    same split table on the same task, steps, matrices and card, so the techniques read side by side instead of one
    number per experiment.

- 2026-10-20. Why the adversary moves and the stochastic check does not (`reports/2026-10-20_analysis_adversarial-vs-stochastic.md`,
  "Hand-off"). From existing data: the adversary's extra damage at the component-training states needs a selective mask; turning
  every maskable subcomponent down together costs 50–300× less. That is the signature of changes that cancel across
  component training's subcomponents (brittleness), with stale Γ as the enabling condition. Next, through `/design`: the
  coherence statistic on the saved step-10 states (CPU, ~1 h), then the partial-revert KL (GPU, ~1.5 h); after those,
  the stochastic check with all layers routed, adversary-mask attribution, and ten W₀ adversary seeds to size the
  floor (the emoticon job read a 21% seed spread). States are on the cluster NFS under
  `experiments/reading_a_coverage/results/large/<task>/pullback_s0/state_step10.pt`.
- 2026-10-20. **Re-gating as an operational definition of re-weighting** (user). When an update leaves the subcomponents
  almost unchanged and the new behaviour appears because the frozen Γ switches on different existing subcomponents in
  response to the new activations, that is an instance of reusing existing mechanisms but activating them differently:
  a candidate definition of "strengthens existing capabilities" for the GOALS question. Condition (user): the updated
  subcomponents still describe the model faithfully with respect to Γ (needs pinning down; VPD's masked checks are the
  current proxy). Evidence so far: re-gating carries about 0.6 of the trigger gain in every emoticon arm, including
  component training with rank-8 offsets on all 24 matrices (E13, files deleted 2026-10-20: trained
  subcomponents under W₀'s masks give P(o) 0.007, untouched subcomponents under the new masks 0.65). Two constraints:
  (1) the re-gating *share* is not target-specific (recruit test: different-target fine-tunes also read 0.55–0.60), the
  *set* switched on is, so the definition must be about which subcomponents are switched on, not the share alone;
  (2) the key-value task already splits along this line (identity map: re-gating 0.76–0.80; random maps: subcomponents
  Γ does not switch on, 0.75–0.86), i.e. reuse vs create read off the data. Next, through `/design`: state the
  definition precisely (share and set, with a null) and check it against the existing ladder, key-value and recruit
  results before any new run; the decision on adding it to `GOALS.md` is the user's.
