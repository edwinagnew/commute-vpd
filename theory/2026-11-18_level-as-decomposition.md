# What a "level" is: a decomposition of the network, not an alignment

*2026-11-18, top-level chat session with the user, for definition-lead and writeup-lead. Nothing was run. "Def. n" is
Geiger et al. 2025 (arXiv 2301.04709), read from `~/commute-data/lit_review_pdfs/survey/2301.04709/paper.txt`.*

**Status.** Part A is adopted by the user (2026-11-18, chat: "yep that sounds good"). Part B holds two tentative
recommendations the user has NOT decided ("im not sure about your epsilon and section 5 recomendations"); write them
as open options, not as the definition. The sentence of the working definition does not change, and no reading so far
changes: this note only replaces the formal statement of "level".

**Amended later on 2026-11-18 (user, chat to writeup-lead; `decisions.md` row "A refinement ... may be approximate").**
- **Exactness.** A refinement may be approximate. N is an exact (Def. 33) or approximate (Def. 41) abstraction of Ñ,
  and ε (accuracy, KL on outputs) is kept and reported beside every reading. The remainder part is optional: with it
  the merge is exact (ε = 0); without it the method is read as approximate (ε > 0). Part A below is edited to match,
  and Part B item 1 is closed.
- **§5.** Part B item 2 is decided as (a) (`decisions.md` row "Paper §5 frames re-gating as a method scored against
  the working definition", 2026-11-18).
- **Name.** The paper says "refinement" for what this note calls a decomposition, and "refined model" for Ñ.
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

**Decomposition** (the name is tentative, see the end; existing text says "level"). A decomposition assigns to each
network N of the architecture a causal model Ñ (Def. 5), on one signature shared by all networks of the architecture,
with variables input, parts U and output, such that N is an abstraction of Ñ under an alignment that is the identity
on input and output: exact constructive abstraction (Def. 33) or approximate abstraction (Def. 41) with accuracy ε,
measured as KL on outputs. ε is reported beside every reading. In words: Ñ is a finer description of N, and merging
its parts back gives N exactly or up to ε. A lossy featuriser may add a **remainder** part, which makes the merge
exact (ε = 0); without one it is read as an approximate refinement (ε > 0). (Amended 2026-11-18; first adopted
exact-only.)

*Open (writeup-lead to the user, 2026-11-18, not answered):* when ε_F > 0, should the hybrids be compared inside the
refined models? That means running the keep-test against F̃ and taking differences between F̃ and M̃, so that the
definition stays total. In the exact case the two coincide.

The direction is the reverse of the earlier form: the network is the high-level model, the decomposed model the
low-level one.

Cases:

- **Units (neurons, heads, MLP layers, blocks).** Ñ = N, parts are a partition of its units. Every reading so far
  (E195, E196, E208) and all four levels of P1.31 are this case, unchanged.
- **Rotations (PCA, DAS).** Ñ is the rotated network; the merge is invertible. Geiger's bijective translation is this
  special case.
- **SAE.** At a site with activation h, the parts are the latents f = enc(h) and the remainder e = h − dec(f). The
  merge is h = dec(f) + e, exact, with no bijection needed. Transcoders and cross-layer transcoders take the same form
  with the latents read from the layer's input.
- **VPD.** For W = Σ_c u_c v_cᵀ + Δ, the parts are the component activations a_c = v_cᵀx and the remainder e = Δx. The
  merge is Wx = Σ_c u_c a_c + e. A rank-one weight component is thus a variable with a mechanism (v_c) and a write
  direction (u_c): no separate "weight version" of the definition is needed.
- **Coarser parts** (a circuit, a cluster of components): a partition of Ñ's variables into parts (variable merge,
  Def. 37), with identity value maps.

**Both networks.** The same decomposition is read on M and on F, giving M̃ and F̃ on the same signature, so a part is
the same variable in both. For activation featurisers F̃ is canonical: the same encoder and decoder applied to F's
activations. For VPD it is not: declaring F̃ is declaring a split of δW (least-squares, adjoint, do-nothing; `GOALS.md`
Terms), and the split is a declared parameter of the decomposition. Under do-nothing all of δW sits in the remainder,
and the reading is "the change lies outside this decomposition".

**Rule, swap, hybrid.** The rule of part u in N is its mechanism in Ñ, as a function of its inputs In(u). The swap
replaces, in F̃, the mechanisms of the parts in a set S by M̃'s. F̃_K is F̃ after the swap on every part outside K;
merged back it is a runnable network F_K, and F_∅ = M exactly when the remainders are among the parts. The swap is a
soft intervention (Def. 10) whose replacement mechanism comes from the other network; that step and reading one
decomposition on two networks remain ours, not Geiger's.

**The definition (unchanged sentence).** F reuses M, for a behaviour at a declared decomposition, to the extent that
the parts F needs still compute what M's parts compute on the same inputs. The user's phrasing of the same thing: for
a given abstraction of the network, reuse is the extent to which M's mechanisms can be interchanged into F.

**Guard, as before.** The decomposition is declared or fitted on M only, never on F or to raise the reading (Sutter et
al. 2025).

**For the paper's paragraph on dependence on the abstraction.** The two extremes hold: the whole hidden computation as
one part reads every new behaviour as new; the neuron case reads nearly everything as reuse (E208). Decompositions are
not ordered from coarse to fine: an SAE or VPD decomposition is neither a coarsening nor a refinement of the neuron
partition. Write "two extreme choices", not "coarsest and finest".

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
