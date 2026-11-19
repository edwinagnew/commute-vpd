# The working definition of reuse, consolidated (2026-11-18, definition-lead)

*2026-11-18, later: the definition sections below are restated in Geiger et al. 2025's vocabulary (user;
`decisions.md` row "Featurized model replaces refinement"); content unchanged. What earlier text calls a level,
decomposition or refinement is a **featurized model** Ñ_i of each network N_i; a part is a variable; a part's rule
is its mechanism F̃_u (Def. 5); a part's inputs are its parents Pa(u) (Remark 6); the swap is the soft intervention
(Def. 10) from the base model, I_K = {F̃_u^{(0)}}_{u∉K}; the hybrid F_K is the intervened model F̃_{I_K}; the behaviour is
a behavior B (§3.5.2); the parts F needs are a minimal circuit for B (§3.5.2), with two stated departures. Def. 33 is
no longer the requirement. The "Clashes" section keeps the older words as a record. Asks 48 and 50 stay open.*

*2026-11-18 ~19:00 (user approved the wording of `reports/drafts/2026-11-18_draft_sec4-bullets-ask48b.md`;
`decisions.md` rows "The variables F needs are every variable the behavior runs through", "Coverage condition", "Gain
and graft declarations for R", "Wording approved"): asks 48, 50 and 51 are closed. K* is the circuit the behavior runs
through under mean ablation, not the smallest set kept under the soft intervention from the base model; a remainder
for the rest of the network covers what the features do not; reuse is the graded share R, with output gain kept and a
graft scored 0. The sections below carry this; where older text in the italic notes above says otherwise, this note
supersedes it.*

*2026-11-18, later still (user, `decisions.md` row "Exact featurization with a mandatory remainder"): the featurization
is exact. One remainder variable per featurized site; each N_i is an exact transformation of Ñ_i (Def. 25); there is no
fit error ε and no optional remainder. The pair is built on activations (σ_0, τ_0∘σ_0 = id) or on weights (VPD, with a
declared split of δW). The remainder's share of K is stated beside every reading. Ask 49 is closed. Where the text
below speaks of ε, an approximate transformation or an optional remainder, this note supersedes it.*

*2026-11-18: wording fixes from the review (`reports/reviews/2026-11-18_review_definition-causal-abstraction.md`,
defects D5 to D9). The content is unchanged. The review's content repairs wait on the user in the log card "Four
repairs the definition needs" (`reports/decisions/2026-11-18_decision_definition-repairs.md`). "Def. n" is Geiger et
al. 2025 in the JMLR numbering; the arXiv HTML numbers differ, and its Def. 41 is Value Merge.*

One page that gathers what the user has decided today. It is assembled from `decisions.md` rows dated 2026-11-18 and
the user's notes to writeup-lead. It adds nothing; where they disagree, the newest row wins and the clash is named
under "Clashes" below. Sources: `GOALS.md` (Working definition), `decisions.md` rows "Working definition of reuse,
stated in causal-abstraction terms", "Working definition of reuse without ρ", "A level is a decomposition, not an
alignment", "A refinement ... may be approximate", "Featurized model replaces refinement" (all 2026-11-18);
`theory/2026-11-18_definition_causal_abstraction.md`; `theory/2026-11-18_level-as-decomposition.md`; decision card
`reports/decisions/2026-11-18_decision_needed-parts.md`.

## The sentence (decided)
A fine-tune F reuses the base model M, for a behavior, to the extent that the variables F needs for that behavior
still have M's mechanisms, on the settings of their parents that occur.

## Its terms (decided)
- **Networks.** M (the paper's N_0) and F (N_1) are deterministic causal models (Def. 5) on one signature, with
  different mechanisms (the weights).
- **Featurized model.** A featurizer (Geiger's verb, §3.6) gives each network N a featurized model Ñ, a causal model
  (Def. 5) with variables the input, intermediate variables U (the features, one remainder variable per featurized
  site, and one remainder for the rest of the network) and the output. N is an exact transformation of Ñ (Def. 25) under τ_0, the identity on inputs and
  outputs, with interventions that set the inputs. There are two constructions: on activations, with σ_0 and
  τ_0∘σ_0 = id (a bijective translation, Def. 28, whose variables are Geiger's modular features, §3.6), or on weights
  (VPD: W_0 = Σ_c u_c v_cᵀ + Δ and a declared split δW = Σ_c δW_c + δW_out; F̃_c^{(0)} = u_c v_cᵀx, F̃_c^{(1)} =
  (u_c v_cᵀ + δW_c)x, F̃_e^{(0)} = Δx, F̃_e^{(1)} = (Δ + δW_out)x). The remainder's share of K* is stated beside every
  reading. Only the definition read at VPD needs a declared split of δW; r does not.
- **Coverage.** Every mechanism the fine-tune changed has a variable (user, 2026-11-18, ask 50). One remainder
  variable, "the rest of the network", covers what the features do not. With an SAE at one site, that is the other
  layers. At the unit level, it is the embeddings, unembedding and layer-norm gains.
- **Graft.** A graft adds a variable to the fine-tune; the base model gets that variable with a mechanism that outputs
  zero, so both share one signature.
- **Ours: one shared featurizer.** The featurizer is fixed from M alone and applied to both networks, so M̃ and F̃
  share one signature and u is the same variable in both. This is the one constraint that is ours. The featurizer is
  a pair (τ_0, σ_0), τ_0 from features to the network's variables and σ_0 back, both the identity on inputs and
  outputs (user, 2026-11-18, "one fixed featurizer"). F's features are not fitted: F̃'s mechanism for each feature is
  F's own mechanism conjugated by M's featurizer, F̃_u^{(1)}(ṽ) = Proj_u σ_0(F^{(1)}(τ_0(ṽ))), as Def. 28 builds a
  translated model; in practice F's features on an input are σ_0 applied to F's activations at each featurized site.
  Cases: units (group; ungroup), rotation (R^T; R), SAE with remainder, transcoder with remainder, VPD with remainder
  (M's components; all of δW in the remainder). A second, separately fitted featurizer needs a fitted pairing, so it
  is at most an optional localisation analysis.
- **Cases.**
  - Unit groupings (neurons, heads, MLP layers, blocks). Ñ is N with its units grouped into variables, a variable
    merge (Def. 37), so exact with no remainder.
  - Rotations (PCA, DAS): a bijective translation (Def. 28), so exact with no remainder.
  - SAE latents.
  - Transcoder and cross-layer transcoder latents. These differ from SAEs: M's transcoder read on F has the same
    encoder, which reads the MLP's input, so each latent agrees with M's mechanism by construction. Every change to
    the MLP's own weights lands in the remainder; only changes upstream of the MLP are charged to latents.
  - VPD component activations. Reading these on F declares a split of the update δW; under the do-nothing update, all
    of δW is remainder.
  - Coarser groupings of variables: variable merges (Def. 37).
  - Not a case: attribution graphs (Ameisen et al. 2025). Their local replacement model freezes attention patterns
    and error terms per prompt, so a variable has no mechanism to compare on other inputs. They serve only as a
    per-input reading.
- **Remainder (mandatory).** One remainder variable per featurized site, whatever the features miss (for an SAE
  e = h − dec_0(enc_0(h)); for VPD Δx, plus δW_out x in the fine-tune). Units and rotations need none at the site;
  the rest of the network is always a remainder variable (Coverage).
- **No ε.** The featurization is exact, so there is no fit error to report; the remainder's share of K* is reported
  instead.
- **Mechanism.** F^{(i)} = {F^{(i)}_X} are the network's own mechanisms (Def. 5); F̃_u^{(i)} is the mechanism of
  feature u in the featurized model; reuse compares F̃_u^{(0)} with F̃_u^{(1)}. The mechanism F̃_u of variable u in Ñ is
  a function of its parents Pa(u) (Remark 6); F̃_u^{(0)} in M̃ and F̃_u^{(1)} in F̃; superscript (0) is M, (1) is F.
- **Behavior.** A partial function B from inputs to outputs (§3.5.2): F's input-output map on a set D of inputs on
  which the fine-tune departs from the base model. D is declared with every reading.
- **Mean ablation.** For each variable u, μ_u is its mean value on D in the fine-tune. For K ⊆ U the hard
  intervention (Def. 9) i_K = {u ← μ_u}_{u∉K} fixes every variable outside K to its mean (user, 2026-11-18). This is
  the standard ablation of sub-circuit analysis, and the one place a value, rather than a mechanism, is fixed.
- **The circuit the behavior runs through (K*).** Following sub-circuit analysis (§3.5.2), K* is a smallest K ⊆ U such
  that F̃_{i_K} preserves B within a declared α, the reference being full ablation, as in standard faithfulness
  scores. Every variable the behavior runs through counts, whether or not the fine-tune changed its mechanism (user,
  2026-11-18, ask 48). The circuit is a set of variables rather than connections.
- **Declared with every reading** (restored from the earlier rows; no values are set here):
  - the distance d by which two mechanisms' outputs are compared: declared per reading, as in the earlier rows;
  - the tolerance for "F̃_{I_K} preserves B" (τ in the earlier rows; the paper writes α, since τ is Geiger's
    translation map): declared per reading, as in the earlier rows;
  - the value sets 𝒱_u, the settings of Pa(u) on which mechanisms are compared: declared per reading, as in the
    earlier rows;
  - the aggregation over K: declared per reading, as in the earlier rows;
  - the rule when K is not unique, that is, when several smallest sets preserve B: declared per reading, as in the
    earlier rows.
- **Reuse.** Let 𝒱_u be the settings of Pa(u) that occur on D in either model's solution (Def. 8), and let
  s_u ∈ [0, 1] grade how far F̃_u^{(1)} = F̃_u^{(0)} on 𝒱_u (s_u = 1 exactly when they agree there up to an output
  gain). Reuse is the share of the circuit whose mechanisms are still the base model's: R = (1/|K*|) Σ_{u∈K*} s_u. The
  remainder's share of K* is stated beside it.
  - *Gain:* mechanisms are compared by direction only, by squared cosine (user, 2026-11-18, ask 51). So a mechanism
    that only scales the base model's output is kept.
  - *Graft:* a variable the fine-tune has and the base model lacks, such as an added head, scores s_u = 0. The base
    model's mechanism for it outputs zero (user, 2026-11-18, ask 51).
  - Cost: a memorised fact also reads as mostly reused when it is stored by small changes in an otherwise old circuit
    (redteam-lead's cold read, 2026-11-18).
- **Readings, not the definition.**
  - ρ: the mean mechanism agreement over E208's circuit of the change K, not K*. With single neurons as variables it saturates (E208: 0.999 or more on
    every fine-tune), and the user said the neuron level is "not a high-enough level abstraction to detect new".
  - The held-model survival S.
  - Every reading so far uses the exact unit-grouping case.

## Open (not decided)
1. **Which variables count (ask 48).** Closed 2026-11-18: every variable the behavior runs through counts (mean
   ablation, K* above). As first asked: as written, K never contains a variable whose mechanism F left unchanged. So a
   new trigger in front of an old circuit F left unchanged reads as new. The alternative is to count every variable
   the behavior runs through. If that is chosen, the user has already picked the rule for removing an unneeded
   variable: its mean over the behavior's inputs (mean ablation).
2. **Which settings of the parents a mechanism is compared on** (restated after the review's D6). The decided wording,
   "given the same input values, the part's mechanism in F gives the same output as its mechanism in M", already
   compares two mechanisms on the same settings, which is A below. B compares values, not mechanisms, and would change
   decided wording. What is open is whose settings the comparison uses: those in F's solutions, those in M's, or both
   pooled, which the review suggests. The question as first asked:
   - (A) what its parents take in F's own run, so its mechanism is compared;
   - (B) the value it took when M read the same text, as in activation patching.

   The lead recommended A. The user did not know which to pick.
3. **Where to compare when ε_F > 0 (ask 49).** Closed 2026-11-18: the featurization is exact, so the question does
   not arise.
4. **§5.** Re-gating is scored against the definition as a method (decided). The re-gating share is a share of the
   log-odds gain in VPD's masked model (the norm share of δW is the outside share). Whether it is also "the definition read
   at the VPD featurized model" is set aside.
5. **Four repairs from the review (ask 50).** Repair 1 (the guard against fitting on F) is set aside by the one fixed
   featurizer, repair 4 (one shared causal order) is automatic with one featurizer and one architecture, and repair 3
   (report the remainder's share of K) is adopted by the exact featurization (2026-11-18); repair 2 is replaced by
   the coverage condition (closed 2026-11-18). As first listed: the guard against fitting on F, which mechanisms the variables outside
   U take, reporting the remainder's share of K, and one shared causal order. Card `decision-definition-repairs`
   (`reports/decisions/2026-11-18_decision_definition-repairs.md`).

## Clashes between files (GOALS.md and the level note fixed 2026-11-18; the alignment-form row stays as a superseded record)
- **`GOALS.md` on exactness.** It still says the parts "merge back into the network exactly" and that "whether ε_F
  stays ... is open". The 2026-11-18 refinement row has since decided that ε is kept and that approximate refinements
  are allowed.
- **`theory/2026-11-18_level-as-decomposition.md`.** Its Part A requires exactness, and its Part B item 1 lists ε as
  open. Both are superseded by the refinement row.
- **The causal-abstraction row.** It speaks of "one alignment (cells Π_u, value maps π_u)". The level row replaced
  that alignment form with a decomposition.
