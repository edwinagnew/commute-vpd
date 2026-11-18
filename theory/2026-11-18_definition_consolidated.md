# The working definition of reuse, consolidated (2026-11-18, definition-lead)

*2026-11-18, later: the definition sections below are restated in Geiger et al. 2025's vocabulary (user;
`decisions.md` row "Featurized model replaces refinement"); content unchanged. What earlier text calls a level,
decomposition or refinement is a **featurized model** Ñ_i of each network N_i; a part is a variable; a part's rule
is its mechanism F_u (Def. 5); a part's inputs are its parents Pa(u) (Remark 6); the swap is the soft intervention
(Def. 10) from the base model, I_K = {F_u^{(0)}}_{u∉K}; the hybrid F_K is the intervened model F̃_{I_K}; the behaviour is
a behavior B (§3.5.2); the parts F needs are a minimal circuit for B (§3.5.2), with two stated departures. Def. 33 is
no longer the requirement: N_i is an approximate transformation of Ñ_i (Def. 41), exact (Def. 25) when ε = 0. The
"Clashes" section keeps the older words as a record. Asks 48-50 stay open.*

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
  (Def. 5) with variables the input, intermediate variables U and the output. N is an approximate transformation of Ñ
  (Def. 41) under maps (τ, ω) with τ the identity on inputs and outputs, and interventions that set the inputs
  (Remark 42); when ε = 0 it is an exact transformation (Def. 25). When the featurizer is a bijective translation,
  the variables are Geiger's modular features (§3.6); otherwise this note says "variables".
- **Ours: one shared featurizer.** The featurizer is fixed from M alone and applied to both networks, so M̃ and F̃
  share one signature and u is the same variable in both. This is the one constraint that is ours.
- **Cases.**
  - Unit groupings (neurons, heads, MLP layers, blocks). Ñ is N with its units grouped into variables, a variable
    merge (Def. 37), so ε = 0.
  - Rotations (PCA, DAS): a bijective translation (Def. 28), so ε = 0.
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
- **Remainder (optional).** A remainder variable is one way to get ε = 0 for a lossy featurizer such as an SAE or VPD;
  it is not part of the concept. Without one, the featurized model is approximate (ε > 0).
- **ε.** ε is the error of Def. 41 on each network: KL on outputs as its Sim, the mean as its statistic, and the
  behavior's inputs as its distribution P (Remark 42). Lower is a better fit. It is kept and reported beside every
  reading.
- **Mechanism.** The mechanism F_u of variable u in Ñ (Def. 5), a function of its parents Pa(u) (Remark 6); F_u^{(0)} in
  M̃ and F_u^{(1)} in F̃; superscript (0) is M, (1) is F.
- **Soft intervention from the base model.** I_K = {F_u^{(0)}}_{u∉K} (Def. 10) gives every variable outside K its
  base-model mechanism (user 2026-11-18: a soft intervention from a trained model). The intervened model F̃_{I_K}
  (Geiger writes M_I) is always runnable.
- **Behavior.** A partial function B from inputs to outputs (§3.5.2): F's input-output map on the inputs where F's
  output differs from M's.
- **The circuit F needs (K).** K is a smallest set of variables such that F̃_{I_K} preserves B: a minimal circuit for B
  (sub-circuit analysis, §3.5.2), with two departures: K is a set of variables rather than connections, and the
  variables outside it take base-model mechanisms in place of ablation values.
- **Declared with every reading** (restored from the earlier rows; no values are set here):
  - the distance d by which two mechanisms' outputs are compared: declared per reading, as in the earlier rows;
  - the tolerance for "F̃_{I_K} preserves B" (τ in the earlier rows; the paper writes α, since τ is Geiger's
    translation map): declared per reading, as in the earlier rows;
  - the value sets 𝒱_u, the settings of Pa(u) on which mechanisms are compared: declared per reading, as in the
    earlier rows;
  - the aggregation over K: declared per reading, as in the earlier rows;
  - the rule when K is not unique, that is, when several smallest sets preserve B: declared per reading, as in the
    earlier rows.
- **Reuse.** Reuse is how far the mechanisms of the variables in K agree with M's: F_u^{(1)}(v) = F_u^{(0)}(v) on the settings
  v of Pa(u) that occur in either model's solution (Def. 8) on the behavior's inputs.
- **Readings, not the definition.**
  - ρ: the mean mechanism agreement over K. With single neurons as variables it saturates (E208: 0.999 or more on
    every fine-tune), and the user said the neuron level is "not a high-enough level abstraction to detect new".
  - The held-model survival S.
  - Every reading so far uses the exact unit-grouping case.

## Open (not decided)
1. **Which variables count (ask 48).** As written, K never contains a variable whose mechanism F left unchanged. So a
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
3. **Where to compare when ε_F > 0 (ask 49).** One option is to compare the intervened models inside the featurized
   models: the preservation test runs on F̃, and the difference is taken between F̃ and M̃. This keeps the definition
   total when ε_F > 0. In the exact case both ways coincide. Raised by writeup-lead and not yet answered.
4. **§5.** Re-gating is scored against the definition as a method (decided). Whether it is also "the definition read
   at the VPD featurized model" is set aside.
5. **Four repairs from the review (ask 50).** The guard against fitting on F, which mechanisms the variables outside
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
