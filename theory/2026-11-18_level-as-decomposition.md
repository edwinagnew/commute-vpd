# What a "level" is: a decomposition of the network, not an alignment

*2026-11-18, later: vocabulary moved to Geiger et al. 2025's (user; `decisions.md` row "Featurized model replaces
refinement"); content unchanged. What this note calls a "level", "decomposition" or "refinement" is a **featurized
model** Ñ_i of each network N_i; a part is a variable; a part's rule is its mechanism F_u (Def. 5); its inputs are its
parents Pa(u) (Remark 6); the swap is the soft intervention (Def. 10) from the base model, I_K = {F^M_u}_{u∉K}; the
hybrid is the intervened model F̃_{I_K}; the parts F needs are a minimal circuit for a behavior B (§3.5.2). The
featurizer is fixed from N_0 alone and shared, the one constraint that is ours. N_i is an approximate transformation of
Ñ_i (Def. 41), exact (Def. 25) when ε = 0, with τ the identity on inputs and outputs (Remark 42); Def. 33 is no longer
the requirement. A remainder variable is one way to reach ε = 0, not part of the concept. Part A is rewritten in these
words; the paragraphs above it, "Why the alignment form had to go", Part B and "The name" stay as records.*

*2026-11-18: wording fixes from the review (`reports/reviews/2026-11-18_review_definition-causal-abstraction.md`,
defects D5 and D7 to D9); content unchanged. Its content repairs wait on the user in the card
`reports/decisions/2026-11-18_decision_definition-repairs.md`.*

*2026-11-18, top-level chat session with the user, for definition-lead and writeup-lead. Nothing was run. "Def. n" is
Geiger et al. 2025 (arXiv 2301.04709) in the JMLR numbering, read from
`~/commute-data/lit_review_pdfs/survey/2301.04709/paper.txt`; the arXiv HTML numbers differ, and its Def. 41 is
Value Merge.*

**Status.** Part A is adopted by the user (2026-11-18, chat: "yep that sounds good"). Part B holds two tentative
recommendations the user has NOT decided ("im not sure about your epsilon and section 5 recomendations"); write them
as open options, not as the definition. The sentence of the working definition does not change, and no reading so far
changes: this note only replaces the formal statement of "level".

**Amended later on 2026-11-18 (user, chat to writeup-lead; `decisions.md` row "A refinement ... may be approximate").**
- **Exactness.** A refinement may be approximate. N is an exact abstraction of Ñ (Constructive Abstraction, Def. 33)
  or an approximate one (Approximate Transformation, Def. 41, under the alignment's maps τ^π and ω^π), and ε (the
  error, KL on outputs) is kept and reported beside every reading. The remainder part is optional: with it
  the merge is exact (ε = 0); without it the method is read as approximate (ε > 0). Part A below is edited to match,
  and Part B item 1 is closed.
- **§5.** Part B item 2 is decided as (a) (`decisions.md` row "Paper §5 frames re-gating as a method scored against
  the working definition", 2026-11-18).
- **Name.** Our paper says "refinement" (our word; Geiger has none) for what this note calls a decomposition, and "refined model" for Ñ.
  "Decomposition" stays the name for VPD and the other methods. This note and the log may keep "level" or
  "decomposition".
- **Still open.** Where the hybrids are compared when ε_F > 0 (end of Part A). Also the decision card
  `reports/decisions/2026-11-18_decision_needed-parts.md`: which parts count as needed, and which input a part sees
  when its rule is compared.

## Why the alignment form had to go

An alignment (Def. 31) groups the network's own variables into cells and coarsens their values. SAEs, transcoders and
VPD do the opposite: they add variables the network does not have. An overcomplete SAE is not a bijective translation
(Def. 28), and Geiger's only remark is that with non-zero reconstruction loss SAEs are "approximate transformations"
(§3.6.1), which would mix reconstruction error into ε. The user's intent was always that any featuriser or choice of
mechanisms can be slotted in as the abstraction on which reuse is defined.

## Part A (adopted): the formal statement

*Restated 2026-11-18 in Geiger's vocabulary (see the top line); content unchanged. The earlier wording, with
"decomposition", "part", "rule", "swap" and "hybrid", is in the git history of this file.*

**Featurized model.** A featurizer (Geiger's verb, §3.6) gives each network N of the architecture a featurized model
Ñ: a causal model (Def. 5) on one signature shared by all networks of the architecture, with variables the input,
intermediate variables U and the output. N is an approximate transformation of Ñ (Def. 41) under maps (τ, ω), with τ
the identity on inputs and outputs and the distribution P over interventions that set the inputs (Remark 42); its
error ε uses KL on outputs as Sim. When ε = 0, N is an exact transformation of Ñ (Def. 25). ε is reported beside every
reading. In words: Ñ is a finer description of N, or for unit groupings N with its units grouped, and it reproduces
N's outputs exactly or up to ε. For a lossy featurizer, a **remainder** variable is one way to reach ε = 0; it is not
part of the concept, and without one the featurized model is approximate (ε > 0). When the featurizer is a bijective
translation, its variables are Geiger's modular features (§3.6).

*Open (writeup-lead to the user, 2026-11-18, not answered; ask 49):* when ε_F > 0, should the intervened models be
compared inside the featurized models? That means running the preservation test on F̃ and taking differences between
F̃ and M̃, so that the definition stays total. In the exact case the two coincide.

The direction is the reverse of the alignment form: the network is the high-level model, the featurized model the
low-level one.

Cases:

- **Units (neurons, heads, MLP layers, blocks).** Ñ is N with its units grouped into variables, a variable merge
  (Def. 37), so ε = 0. Every reading so far (E195, E196, E208) and all four featurizers of P1.31 are this case,
  unchanged.
- **Rotations (PCA, DAS).** Ñ is the rotated network, a bijective translation (Def. 28), so ε = 0.
- **SAE.** At a site with activation h, the variables are the latents f = enc(h), optionally with the remainder
  e = h − dec(f). With the remainder, h = dec(f) + e and ε = 0, with no bijection needed.
- **Transcoders and cross-layer transcoders.** The same form, with the latents read from the MLP's input, but they read
  differently: M's transcoder applied to F has the same encoder, so each latent agrees with M's mechanism by
  construction. Every change to the MLP's own weights lands in the remainder, and only changes upstream of the MLP are
  charged to latents.
- **Not a case: attribution graphs** (Ameisen et al. 2025). The local replacement model freezes attention patterns and
  error terms per prompt; a variable then has no mechanism to compare on other inputs, so they serve only as a
  per-input reading.
- **VPD.** For W = Σ_c u_c v_cᵀ + Δ, the variables are the component activations a_c = v_cᵀx, optionally with the
  remainder e = Δx, and Wx = Σ_c u_c a_c + e. A rank-one weight component is thus a variable with a mechanism (v_c)
  and a write direction (u_c): no separate "weight version" of the definition is needed.
- **Coarser groupings** (a circuit, a cluster of components): a partition of Ñ's variables (variable merge, Def. 37),
  with identity value maps.

**One featurizer for both networks (ours).** The featurizer is fixed from M alone and applied to M and F, giving M̃ and
F̃ on the same signature, so a variable is the same in both. This shared featurizer is the one constraint that is ours.
For activation featurizers F̃ is canonical: the same encoder and decoder applied to F's activations. For VPD it is
not: declaring F̃ is declaring a split of δW (least-squares, adjoint, do-nothing; `GOALS.md` Terms), and the split is a
declared parameter of the featurizer. Under do-nothing all of δW sits in the remainder, and the reading is "the change
lies outside this featurized model".

**Mechanism, soft intervention, intervened model.** The mechanism F_u of variable u in N is its mechanism in Ñ
(Def. 5), a function of its parents Pa(u) (Remark 6); F^M_u in M̃, F^F_u in F̃. The soft intervention from the base
model, I_K = {F^M_u}_{u∉K} (Def. 10), replaces in F̃ the mechanism of every variable outside K by M̃'s. The intervened
model F̃_{I_K} (Geiger writes M_I) is always runnable, and with no variables kept it is M exactly when the remainders
are among the variables. A soft intervention whose replacement mechanism comes from another trained network, and one
featurizer applied to two networks, remain ours, not Geiger's.

**The definition (unchanged content).** F reuses M, for a behavior B under a declared featurizer, to the extent that
the variables F needs still have M's mechanisms, on the settings of their parents that occur. B is a partial function
from inputs to outputs (§3.5.2); the variables F needs are a minimal circuit for B (§3.5.2), with two departures: a
set of variables rather than connections, and base-model mechanisms in place of ablation values. The user's phrasing
of the same thing: for a given abstraction of the network, reuse is the extent to which M's mechanisms can be
interchanged into F.

**Guard, as before.** The featurizer is fixed from M only, never from F or to raise the reading, by analogy with
Sutter et al. 2025, whose theorem concerns aligning an algorithm to a network rather than one featurizer applied to two
networks.

**Declared with every reading** (restored from the earlier rows; no values set here): the distance d, the tolerance
for preserving B (τ in the earlier rows; α in the paper, since τ is Geiger's translation map), the value sets 𝒱_u of
the settings of Pa(u), the aggregation over K, and the rule when K is not unique. Each is declared per reading, as in
the earlier rows.

**For the paper's paragraph on dependence on the featurizer.** The two extremes hold: the whole hidden computation as
one variable reads every new behavior as new; single neurons as variables read nearly everything as reuse (E208).
Featurizers are not ordered from coarse to fine: an SAE or VPD featurizer is neither coarser nor finer than single
neurons. Write "two extreme choices", not "coarsest and finest".

## Part B (both items decided 2026-11-18; kept as a record)

1. **ε.** *Decided: ε is kept and reported, and approximate refinements are allowed (see the amendment at the top).
   The text below was the recommendation then.* With identity value maps the merge is exact on both networks, so ε = 0 and reconstruction error never enters
   it: it lives in the remainder part. Recommendation: drop ε from the main text and report instead how much of the
   needed set K is remainder; ε would return only for a decomposition with lossy value maps. Alternative: keep ε as
   stated in the 2026-11-18 causal-abstraction row. Until the user decides, keep ε in the text and mark the question.
2. **§5 framing of re-gating.** *Decided: (a), the recommendation below.* Recommendation: re-gating stays a method scored against the definition, which is what
   results support; one sentence may note that a split of δW induces a reading at the VPD decomposition, untested.
   Alternative: present the re-gating share as that reading, which needs a new cell. Not to be written either way
   until the user decides.

## Facts checked for the open list in `sec4_causal_abstraction_bullets.md`

- Geiger has no error or remainder node anywhere and nothing on decomposing weights; the one sentence on parameters is
  §3.7, adapter tuning "can be viewed as interventions on model parameters". The remainder construction is ours, built
  from Geiger's objects. Precedents to check before citing (from memory, unread here): Marks et al. 2024 (error nodes
  in sparse feature circuits); the MIB benchmark's SAE featuriser (Mueller et al. 2025); Ameisen et al. 2025 (local
  replacement model with error nodes).
- E195/E196 searched for a smallest set over five search orders (their analysis reports), not a mean over all changed
  neurons. The P1.25 `decisions.md` row gives "a set suffices at 0.9 of KL"; not checked in the code.

## The name

*Decided 2026-11-18 (user, chat to writeup-lead): the paper says **refinement**. The text below was the earlier
recommendation.*

"Level" suggests an ordering that does not exist. Recommendation, tentative: **decomposition** (covers sparse
dictionaries, parameter decomposition and the unit partitions; already the project's word). Others from the
literature: "featuriser" (Geiger's group; suggests activations and an invertible map), "replacement model" (Anthropic;
suggests an approximation of the network, where ours is exact). The user chooses; until then existing text keeps
"level".
