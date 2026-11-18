# Goals

The aim of the project, in one place. Owned by the user: agents propose edits, the user approves them. Read this
before proposing any new experiment, task, architecture or theory direction. Working choices that could be revised
live in `decisions.md`, and what has been tried lives in `experiments/TRIED.md`.

*Status: 2026-10-14. The question is the user's (verbatim below); its precise form was given by the user after round 1.*

## The question (user, 2026-10-14, verbatim)

> Given a (VPD) parameter decomposition, can one commute small gradient updates from post-training past the
> parameter components. If so, this gives "continuous interpretability" (within some limit). If not, is it because
> post-training teaches new capabilities (vs strengthens existing ones), or maybe just parameter decompositions are
> too brittle.

## The question made precise (user, 2026-10-14)

> Is there a simple operation that maps a post-training update (say, a LoRA) to an update on the parameter
> components, such that the resulting decomposition is faithful with respect to the new model? Or, a stronger
> version: such that the resulting decomposition is pretty much what you would get if you re-decomposed the updated
> model? Weight updates can be assumed small.

- **The map.** A simple operation Φ takes the decomposition D of W (subcomponents {U_c, V_c}, causal-importance
  function Γ, residual Δ) and an update δW, and returns D' = Φ(D, δW) for W + δW. "Simple" excludes re-fitting the
  decomposition. Examples (Terms below): the rescaling split, the least-squares or adjoint split of a finished δW, and
  component training (the same training step applied to U and V instead of W).
- **Weak version (current target).** D' is a faithful decomposition of W + δW. The *components* must carry the new
  model: Δ may not absorb δW. Otherwise the trivial map "leave everything, put δW in Δ" passes. In VPD terms, D'
  scores on W + δW about as well as D scores on W, under the same losses (weight reconstruction by the components,
  masked-output reconstruction, sparse causal importance).
- **Strong version (postponed until the weak version is settled).** D' ≈ re-decomposition of W + δW, measured
  against a δW = 0 warm-start control and the cross-seed baseline.
- **If no simple Φ works.** Is that because post-training teaches new capabilities rather than strengthening
  existing ones, or because parameter decompositions are too brittle?
- **Working definition of reuse (user, 2026-11-18, revised the same day; Geiger's vocabulary 2026-11-18).** A
  fine-tune F reuses the base model M, for a behavior, to the extent that the variables F needs for that behavior
  still have M's mechanisms, on the settings of their parents that occur. Stated in the vocabulary of causal
  abstraction wherever it has a term (Geiger et al. 2025, "Def. n" in the JMLR numbering; user, 2026-11-18). A
  featurizer (Geiger's verb, §3.6) gives each network N a **featurized model** Ñ: a causal model (Def. 5) whose
  variables are the input, intermediate variables U and the output, for example neurons, heads, blocks (a variable
  merge, Def. 37), rotations (a bijective translation, Def. 28), SAE latents or VPD components. N is an approximate
  transformation of Ñ (Def. 41; an exact transformation, Def. 25, when ε = 0), with τ the identity on inputs and
  outputs and interventions that set the inputs (Remark 42); the error ε of Def. 41, KL on outputs, is reported
  beside every reading. The featurizer is one pair of maps (τ_0, σ_0) fixed from M alone and shared, both the identity
  on inputs and outputs, so a variable is the same in both networks (the one constraint that is ours; user, 2026-11-18,
  "one fixed featurizer"). F's features are not fitted: F̃'s mechanism for each feature is F's own mechanism
  conjugated by M's featurizer, F̃_u^{(1)}(ṽ) = Proj_u σ_0(F^{(1)}(τ_0(ṽ))), as Def. 28 builds a translated model, so
  in practice F's features on an input are σ_0 applied to F's activations at each featurized site. A remainder variable is one way to reach ε = 0 for a lossy featurizer, not part
  of the concept. A variable u has a mechanism F_u (Def. 5), a function of its parents Pa(u) (Remark 6). The soft
  intervention from the base model I_K = {F_u^{(0)}}_{u∉K} (Def. 10; superscript (0) is M, (1) is F) gives every variable outside K M's mechanism; the
  intervened model F̃_{I_K} (Geiger writes M_I) is always runnable. A behavior B is a partial function from inputs to
  outputs (§3.5.2); the circuit F needs is a smallest K for which F̃_{I_K} preserves B, a minimal circuit in
  Geiger's sub-circuit analysis (§3.5.2) with two departures: a set of variables, not connections, and base-model
  mechanisms in place of ablation values. A mechanism agrees with M's when it agrees on the settings of Pa(u) that
  occur in either model's solution on the behavior's inputs (`decisions.md` row "Featurized model replaces
  refinement", 2026-11-18; earlier rows "A level is a decomposition, not an alignment" and "A refinement ... may be
  approximate"; `theory/2026-11-18_level-as-decomposition.md`). Open:
  - which variables count as needed, and which settings of its parents a mechanism is compared on (log card "needed
    parts");
  - where the intervened models are compared when ε_F > 0;
  - four repairs from the 2026-11-18 review of the definition (log card "definition repairs").

  Consolidated on one page: `theory/2026-11-18_definition_consolidated.md`. ρ (the mean mechanism agreement over a smallest sufficient set) is no longer part of the definition (user,
  2026-11-18 morning): with single neurons as variables it is a tested reading that saturates (E208: 0.999 or more on
  every fine-tune), since "the neuron level is not a high-enough level abstraction to detect new". Full statement in
  the research log's overview box "The working definition", `theory/2026-11-18_definition_causal_abstraction.md` and
  `decisions.md` (rows "... stated in causal-abstraction terms" and "Working definition of reuse without ρ",
  2026-11-18). The held-model survival S (every unit of F given M's mechanism at once, up to a fitted map) and ρ are
  readings of it, not the definition; their numbers stand as readings. It supersedes the held-model wording (adopted earlier on
  2026-11-18) and the second wording ("reuse at a variable", tentatively adopted 2026-11-10), kept only as records.
- **Canonical update:** LoRA, plus full-weight fine-tunes.
- **Γ:** frozen for now (see `decisions.md`), although updating it may turn out to be essential.
- **Optimiser (priority subgoal, user 2026-10-21):** compare SGD and Adam on the same tasks, both as the
  component-training optimiser and as the fine-tune optimiser whose update the splits consume. Until that comparison
  is run, no verdict on component training or on a split counts as a property of the operation rather than of Adam.

## Terms (2026-10-20; the user's wording, after `reports/reviews/2026-10-20_review_terminology.md`)

"Pull-back" and "lift" each carried three meanings between 2026-10-14 and 2026-10-20 and are retired. The user's
"pull-back" always meant: split a finished update over the subcomponents and see what is left outside. The files had
fixed it as "train the subcomponents on the loss" and overrode the user for days. Rule (`.claude/CLAUDE.md`): when the
user's words and a file's definition disagree, the user's words win; flag it, get explicit acknowledgement, fix the file.

W = U Vᵀ + Δ per matrix; U and V hold the alive subcomponents' write and read vectors as columns; Γ and Δ are frozen in
every operation below.

| Term | Consumes | What it does | Code | Old names |
|---|---|---|---|---|
| **Component training** | the loss and data | train the alive (U, V) with the run's optimiser, either SGD or Adam. One SGD step moves W by UUᵀG + GVVᵀ, G the weight gradient | `component_training.train` | pull-back, pull-back replay, inside training |
| **Rank-r component training** | the loss and data | component training with rank-r offsets dU = P Sᵀ, dV = Q Tᵀ | `component_training.Offsets(rank=r)` | LoRA-shaped pull-back, "LoRA pull-back" as run in E7 and E13, pb_r |
| **Split** of a finished update | a finished δW: a fine-tune difference or a LoRA's BA | write δW as per-subcomponent updates (dU_c, dV_c) with dU Vᵀ + U dVᵀ ≈ δW, plus a remainder outside the subcomponents | `split.py` | the user's "pull-back"; "decompose BA"; "distribute a LoRA over the decomposition" |
| **Least-squares split** | a finished δW | the split with the least-norm (dU, dV); the remainder O(δW) = (I − P_U) δW (I − P_V) goes to Δ, so W′ equals the fine-tuned weights | `split.least_squares_split`, `split.split_state` | lift, Euclidean lift |
| **Adjoint split** | a finished δW | dU = δW V, dV = δWᵀ U; weight change K(δW) = UUᵀδW + δWVVᵀ, Δ fixed, W′ ≠ the fine-tuned weights. Applied as a state with a per-matrix scale c = argmin ‖(c K(δW) − δW) Σ^½‖ (user-confirmed 2026-10-20) | `split.adjoint_split`, `split.adjoint_scale` | "pull-back of δW", variant (b) of the 2026-10-19 variants review |
| **Step split** | the fine-tune's path, step by step | at every step, the scaled adjoint split of that step's weight change S_t against the current (U, V); the remainder goes to Δ; U, V carry forward, so W′ equals the fine-tuned weights at every step. With U, V held at W₀ and c fixed it equals the one-shot adjoint split plus its remainder (linearity), so it can differ only through the bases' rotation, which may be negligible (user, 2026-10-21; E19) | `split.StepSplit`, `split.adjoint_step` | variant (b) of the 2026-10-19 variants review, per step with updated bases |
| **Rescaling split** | a finished δW | δW ≈ Σ α_c u_c v_cᵀ, directions fixed | `transport.py` | projection, transport T0 |
| **Do-nothing** | a finished δW | δW into Δ; subcomponents and Γ untouched. The reference | — | do-nothing placement |
| **Teacher training** | W₀, W₀ + δW and inputs | component training with the loss KL(teacher ‖ student), teacher = the fine-tuned model; Γ and Δ frozen | `teacher.teacher_loss`, `component_training.train(loss_fn=...)` | — (new, 2026-10-23) |
| **LoRA split** | — | train a rank-r LoRA to completion, take its splits, report the outside share and the per-subcomponent updates | `experiments/lora_split` (2026-10-20) | "LoRA pull-back" as the user meant it |
| **Fixed-shift hypothesis** | — | re-gating's working hypothesis (user, 2026-10-22; restated 2026-10-23, tentatively; not a finding): after a narrow fine-tune, one fixed shift to Γ's importance values at the trigger *per cue*, the same for every input with that cue, aimed at particular subcomponents and partly shared when cues share an answer, carries most of the switching part of the gain; each input's own variation adds a few hundredths. "Cue" is the trigger token in every task run so far and has no general definition yet (`backlog.md`, 2026-10-23) | `regating.shift_of` | H′; H (2026-10-21, before its claim to be enough on its own failed) |

Measurements, not methods:

- **Outside share:** the energy of δW outside the subcomponents, input-weighted (E‖δW a‖² over the task's inputs a),
  with the same number for a random update of the same size as its null. Reported as four blocks that sum to one
  (user-confirmed 2026-10-20, `reports/plans/2026-10-20_plan_lora-split.md` question 1): the write side split by P_U
  (inside col U) and I − P_U; the read side by writing the input a = a₁ + a₂, a₁ the part linearly predictable from what
  the subcomponents read (E[a | Vᵀa]) and a₂ the uncorrelated rest. Inside both: P_U δW a₁; write side only:
  P_U δW a₂; read side only: (I − P_U) δW a₁; outside both: (I − P_U) δW a₂, what no first-order change of the
  subcomponents reproduces on the data. The null is exact in expectation. A rank-1 update b aᵀ is outside-both only if
  neither b nor a is in its span, so the strict reading is the first block. The algebraic outside-both block
  (I − P_U) δW (I − P_V), input-weighted, is E2 to E9's number and is kept beside the four for continuity.
  Old names: obstruction fraction ω, inexpressible share. Code: `split.four_blocks` (the four, with the null),
  `split.weighted_fractions` (the algebraic block), `split.rescaling_share` (the strictest span, span{u_c v_cᵀ}).
- **Φ** maps (D, δW) to D′ for one update step. Along training, Φ is applied at every step and the question is whether
  the error accumulates; the run is the limit, not the object. Experiments to 2026-10-19 scored finished 10-step runs
  one-shot; see `reports/decisions/2026-10-19_decision_success-criterion.md`.
- **Faithful (weak version):** D′ passes VPD's own checks on W + δW about as well as D passes them on W (stochastic-mask,
  adversarial-mask, CI-masked and unmasked reconstruction KL, and the sparsity loss), with Δ not absorbing δW. The
  route split (direct, re-gating, interaction, masked-off, residual) describes mechanism and does not decide this.
- **Γ, the gate:** one network reading every matrix's input at every position. An edit in one layer moves CI values in
  others.
- **Small:** measured by a step sweep, never assumed.
- **r, the re-gating share:** let T0 and T1 be the target's log-odds of the trained answer at the trigger under W₀ and
  under the fine-tuned state, and let L(C, G) be the masked log-odds with components C and gate pattern G (Δ out), 0
  for W₀'s and 1 for the state's. Then r = (L(C₀, G₁) − L(C₀, G₀)) / (T1 − T0): the share of the gain that appears when
  the original subcomponents are kept and only the frozen Γ gates differently on the new activations (nan when the gain is
  about zero, `pcfg_vpd.shares`; the shares are scored only from a gain of 1 log-odds, `local_faith.MIN_GAIN`). It is the "r" of the route split above, not the rank r
  of the rank-r component training.
- **Stand-in ratio:** let U0 and U1 be the unmasked log-odds with the subcomponents (Δ out) at W₀ and at the state. Then
  the stand-in ratio = (L(C₁, G₁) − L(C₀, G₀)) / (U1 − U0): the gain the masked model reaches, gates and all, over the
  gain the unmasked subcomponents reach. It is nan when U1 − U0 is under E53's floor (`thresholds.ratio_floor`,
  `pcfg_vpd.shares`, `heldout_keys.vpd_read`). In `experiments/method_table` it is the `ratio` entry of `table.json`
  `.methods`, beside `r`.

## Where an update goes (mechanism hypotheses, user-approved 2026-10-16; they do not decide the weak version, 2026-10-19)

Under a map Φ, the change in the new model's behaviour splits exactly into five routes: (a) **direct**, the changed
subcomponents at the old gates; (b) **re-gating**, the frozen Γ gating differently on the new activations; (c) their
**interaction**; (d) **masked off**, carried by subcomponents Γ does not gate on; (e) **residual**, not carried by the
subcomponents at all. A sixth route, Γ's own parameters, is zero while Γ is frozen. The weak version holds when (d) and
(e) are small. Measurement: `reports/predictions/2026-10-16_prediction_ladder-routes-amendment.md`.
The weak version itself is decided by VPD's checks on D' (`reports/decisions/2026-10-19_decision_success-criterion.md`); a small
direct share (a) is a finding about mechanism, not a failure of Φ.

- **Strengthen:** (a) dominates. Post-training sharpens subcomponents that already do the job.
- **Reveal:** (b) + (c) dominate. The update is a small change that Γ amplifies by switching on subcomponents that
  already exist. The concentration test points this way on the emoticon task.
- **Create:** (e) dominates when Δ may move. When it may not, (d) dominates or component training fails to learn.
- **Brittle:** the split is unstable across seeds or across small changes to Φ.

## Not the goal

- Improving VPD itself, or re-deriving its training recipe.
- Re-running the Goodfire decomposition from scratch unless the projection experiments demand it.
