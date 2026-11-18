# The working definition in the language of causal abstraction

*2026-11-18, coding agents, for the definition lead. Restates the log's box "The working definition · revised
2026-11-18"; nothing was run. "Def. n" is Geiger et al. 2025 (arXiv 2301.04709), read from the project's PDF
extraction. Geiger's translation map is also called τ; here τ is only our tolerance, and the alignment's maps are
written π. Settled by the user on 2026-11-18 and folded in: one alignment for both networks; approximate abstraction
with its accuracy declared and reported.*

**Short answer.** Yes. A level is one alignment from the shared architecture to a high-level causal model, applied
to both networks. Its parts are therefore the same parts in M and in F. Reuse is defined by one between-model
intervention, the swap: give F's parts M's mechanisms. Read at one part, the swap gives retention; read at the
output, it gives the parts F needs. A second reading, the accuracy of the alignment on F, says how far M's
description of the parts still fits the fine-tune. Two steps are ours, not Geiger's: one alignment read on two
networks, and an intervention whose new mechanism comes from the other model.

## (a) The definition

1. **Networks.** M (original) and F (fine-tune) are deterministic causal models (Def. 5) on the same architecture:
   the same variables, input and output, with different mechanisms (weights). *Plain: each network is its
   computation graph, and the two graphs are the same.*
2. **Level.** A level is one alignment ⟨Π, π⟩ (Def. 31) from the architecture to a high-level causal model with
   variables {input} ∪ U ∪ {output}. Π_u is the cell of the architecture's variables aligned to part u, and π_u maps
   the cell's values to the values of u. The alignment is the identity on input and output and leaves nothing on a
   path from input to output unaligned. Read on network N ∈ {M, F}, it gives a high-level model A_N, whose
   **rule** for u is its mechanism f_u^N : Val(In(u)) → Val(u), where In(u), the inputs of u, are its parents (or a
   declared set of ancestors, the parts between solved out). A part that the level declares absent from N (for
   example a unit F adds, or one F zeroes) has no mechanism in A_N. A_N is an approximate abstraction of N
   (Def. 41), with **accuracy** ε_N: the mean, over the behaviour's inputs and the interventions clauses 3-5 use, of
   the KL from N's output under the low-level intervention to A_N's output under the aligned high-level one.
   *Plain: parts are high-level variables, the same ones in both networks; "the description reproduces both
   networks" is ε_M = ε_F = 0, Geiger's constructive abstraction (Def. 33).*
3. **The swap.** For a set S of parts, the swap I^S replaces, for each u ∈ S, F's mechanisms on the cell Π_u by M's,
   and removes each u ∈ S that M lacks. At the high level it is the soft intervention on A_F (Def. 10; a constant
   interventional, Def. 11) that replaces f_u^F by f_u^M. *Plain: F with the parts in S following M's rule.*
4. **Keeping a rule.** Let 𝒱_u be the settings v of In(u) that occur when M and when F run on the behaviour's
   inputs (interchange values, Def. 44, taken from either network). A part u keeps M's rule to the extent that, with
   In(u) held at v by a hard intervention, u takes the same value before and after the swap I^{u}, over v ∈ 𝒱_u:
   that is, f_u^F(v) = f_u^M(v). A part M lacks keeps nothing.
5. **The parts F needs.** A behaviour is a set D of inputs on which F's output differs from M's. F_K is F after the
   swap I^{U∖K}: a network, F with M's mechanisms on every cell outside K. F needs K when K is a set of fewest parts
   for which F_K keeps the behaviour.
6. **Reuse.** F reuses M, for D at that level, to the extent that the parts in K keep M's rules.
7. **Fit of the level on F.** Reported beside reuse: ε_F, how faithful the alignment, fitted or chosen on M, is on
   F. *Plain: whether M's description of the parts still describes the fine-tune.*

Because the swap acts on the network's own cells, F_K is always a network that can be run, and the test of clause 5
compares networks' outputs. With every path variable aligned, F_∅ is M (up to the parts F lacks, clause 2), so
K = U always keeps the behaviour and a reading always exists. A computation that F moves into other cells is read
as changed rules in those cells, and as a larger ε_F where the alignment no longer fits.

## (b) The standard reading as a special case

- **r_u = 1 − mean_v d(f_u^F(v), f_u^M(v))**: clause 4 scored as Def. 41 on the one-part model (similarity 1 − d,
  statistic a mean, interventions the settings v ∈ 𝒱_u). A part M lacks: r_u = 0.
- **KL(F ‖ F_K) ≤ (1 − τ) · KL(F ‖ M) on D**: clause 5's "keeps the behaviour", unchanged.
- **ρ = (1/|K|) Σ_{u∈K} r_u**: clause 6 scored as a mean. A part whose rule is unchanged is untouched by the swap,
  so a smallest K holds only changed parts, as the box says. The range over smallest sets and the "no change"
  reading carry over.
- **Declared with every reading**: as now (level, behaviour's inputs, τ, d), plus ε_M; ε_F is reported.
- **Readings so far (E195, E196)**: parts are the network's own MLP neurons, cells single units, π the identity,
  so ε_M = ε_F = 0 and the readings are exact. In(u) is one layer deep, 𝒱_u from M's runs only, d = 1 − cos².
  E195's one-to-one re-pairing of neurons within a layer, used on a shuffled control, is an exact symmetry of the
  architecture and so leaves the network's function unchanged; it is allowed where declared, not as a free search.

## (c) Choices still open (★ = keeps today's content)

1. **How F_K is made.** ★ Mechanism replacement: a part outside K runs M's rule on the inputs it receives in the
   hybrid (E195 does this). The alternative is Prakash et al.'s cross-model activation patching: set the part to
   the value it took in M's own run on the same input. The two differ whenever the part's inputs differ between
   the hybrid and M's run, so K changes.
2. **Chosen or fitted alignment.** ★ Declared, or fitted on M only, the class searched declared. Fitting it on F, or
   to maximise retention, would let the alignment absorb the change; over an unrestricted class of maps π
   abstraction is known to become vacuous (Sutter et al. 2025, `lit_review/definition_survey/causal.md`).
3. **Which values 𝒱_u.** ★ Settings from M's and F's runs on D, pooled, the weighting declared (E195: M's only).
   Settings arising inside F_K would make retention depend on K.
4. **"Smallest".** ★ Fewest parts (the box: a search "gives only an upper bound on its size"), not minimal under
   inclusion.
5. **Parts of M that F lacks** (for example units F has zeroed). ★ They stay absent in every F_K, as E195 fixed;
   then F_∅ is M without those parts, not M. The alternative restores them outside K.
6. **Cells for distributed variables.** Where a part is a direction rather than a set of units, its cell is the
   variable it lies in, after a declared change of basis (Geiger's bijective translations), and the swap replaces
   mechanisms in that basis. No current reading uses this.

**From units to arbitrary aligned variables: does a current claim change?** None reverses; two need rewording.
"A neuron's rule is fixed by its weights" becomes "by the weights and the alignment maps". "Different levels give
different answers" now includes different alignments over the same units. Where ε_F is large, ρ describes F
through a description that fits it poorly, and the two are read together.

## (d) Do we need to lean more on Prakash et al.?

No. Every formal object is Geiger's: causal model, alignment, soft intervention, interchange values, approximate
transformation. Prakash et al. 2024 give no definition; their cross-model activation patching is a measurement at
the identity alignment, on one circuit's heads, same input, comparing outputs. It is the precedent for
intervening between a base model and its fine-tune and should be cited as that. Leaning on it further would mean
activation patching for F_K (choice 1), which changes the content. Attribution: Geiger for the objects, Prakash
for the precedent, and the two-model steps (one alignment read on both networks, the cross-model swap) stated as
ours, since Geiger's paper never relates two networks. By the project's survey the nearest formal two-network
precedents are Wu et al. 2022 and Grant 2025.

**Not checked.** Geiger pp. 24-29 and 34-43; Grant 2025 and Sutter et al. beyond the survey's notes; whether
E195's code matches clause 3 for heads.

---

*Decision (user, 2026-11-18): one alignment for both networks, rather than one per network joined by a pairing of
parts, so that "the same part" means the same cell read the same way in M and F (as keeping the same decomposition
does), and the fit of that alignment on F is studied as its own reading (ε_F).*
