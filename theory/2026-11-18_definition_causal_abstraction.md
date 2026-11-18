# The working definition in the language of causal abstraction

*2026-11-18, coding agent, for the definition lead. Restates the log's box "The working definition · revised
2026-11-18"; nothing was run. "Def. n" is Geiger et al. 2025 (arXiv 2301.04709), read from the project's PDF
extraction. Geiger's translation map is also called τ; here τ is only our tolerance, and an alignment's maps are
written π. Settled by the user on 2026-11-18 and folded in: separate alignments for M and F; approximate
abstraction with its accuracy declared; a partial pairing of parts.*

**Short answer.** Yes. A level is an abstraction of each network with its own alignment, plus a pairing of their
parts. Reuse is defined by one between-model intervention, the swap: put M's mechanisms into F's abstraction. Read
at one part, the swap gives retention; read at the output, it gives the parts F needs. Two steps are ours, not
Geiger's: two networks abstracted side by side, and an intervention whose new mechanism comes from the other model.

## (a) The definition

1. **Networks.** M (original) and F (fine-tune) are deterministic causal models (Def. 5) with the same input and
   output variables. *Plain: each network is its computation graph.*
2. **Level.** A level gives, for each network N ∈ {M, F}: a high-level causal model A_N with variables
   {input} ∪ U_N ∪ {output}, and an alignment ⟨Π^N, π^N⟩ from N to it (Def. 31). Π^N_u is the cell of N's variables
   aligned to part u, and π^N_u maps the cell's values to the values of u. Alignments are the identity on input and
   output and leave nothing on a path from input to output unaligned. π^M and π^F may differ, so a part may sit in
   different units of each network. The level also gives a **pairing** p, a partial one-to-one map from U_F to U_M;
   paired parts share a value space, and A_M, A_F share a causal order on them. The **rule** of u is its mechanism
   in A_N, f_u^N : Val(In(u)) → Val(u), where In(u), the inputs of u, are its parents (or a declared set of
   ancestors, the parts between solved out). A_N is an approximate abstraction of N (Def. 41), and its **accuracy**
   ε_N is declared: the mean, over the behaviour's inputs and the interventions clauses 3-5 use, of the KL from
   N's output under the low-level intervention to A_N's output under the aligned high-level one.
   *Plain: parts are high-level variables; "the description reproduces both networks" is ε_M = ε_F = 0, Geiger's
   constructive abstraction (Def. 33).*
3. **The swap.** For a set S of F's parts, the swap I^S is the soft intervention on A_F (Def. 10; a constant
   interventional, Def. 11) that replaces f_u^F by f_{p(u)}^M for each paired u ∈ S and removes each unpaired
   u ∈ S. *Plain: F's abstraction with the parts in S following M's rule.*
4. **Keeping a rule.** Let 𝒱_u be the settings v of In(u) that occur when A_M and when A_F run on the behaviour's
   inputs (interchange values, Def. 44, taken from either model). A paired part u keeps M's rule to the extent
   that, with In(u) held at v by a hard intervention, u takes the same value in A_F before and after the swap
   I^{u}, over v ∈ 𝒱_u: that is, f_u^F(v) = f_{p(u)}^M(v). An unpaired part keeps nothing.
5. **The parts F needs.** A behaviour is a set D of inputs on which F's output differs from M's. F_K is A_F after
   the swap I^{U_F∖K}. F needs K when K is a set of fewest parts for which F_K keeps the behaviour.
6. **Reuse.** F reuses M, for D at that level, to the extent that the parts in K keep M's rules.

F_K is defined at the high level only. When the alignments coincide (same cells, π the identity) it is a network:
F's weights with M's restored on the units outside K. Otherwise it is realised by patching between the networks
through π^F and π^M, and how well that works is what ε measures.

## (b) The standard reading as a special case

- **r_u = 1 − mean_v d(f_u^F(v), f_u^M(v))**, with f_u^M short for f_{p(u)}^M: clause 4 scored as Def. 41 on the
  one-part model (similarity 1 − d, statistic a mean, interventions the settings v ∈ 𝒱_u). Unpaired: r_u = 0.
- **KL(F ‖ F_K) ≤ (1 − τ) · KL(F ‖ M) on D**: clause 5's "keeps the behaviour", unchanged.
- **ρ = (1/|K|) Σ_{u∈K} r_u**: clause 6 scored as a mean. A part whose rule is unchanged is untouched by the swap,
  so a smallest K holds only changed parts, as the box says. The range over smallest sets and the "no change"
  reading carry over.
- **Declared with every reading**: as now (level, behaviour's inputs, τ, d), plus ε_M and ε_F.
- **Readings so far (E195, E196)**: parts are the network's own MLP neurons, cells single units, π the identity,
  so ε_M = ε_F = 0 and the readings are exact. The alignments coincide up to E195's declared re-pairing of
  neurons within a layer, which is p. In(u) is one layer deep, 𝒱_u from M's runs only, d = 1 − cos².

## (c) Choices still open (★ = keeps today's content)

1. **Which outputs the KLs use once ε > 0.** New, forced by approximate abstraction; the options agree when
   ε = 0. ★ The networks' own outputs for F and M, as the formula is written; then if ε_F > (1 − τ)·KL(F ‖ M) no
   K keeps the behaviour and there is no reading. Or A_F's and A_M's outputs: a reading always exists, but it
   describes the abstractions, and a change the abstraction misses is not seen.
2. **How F_K is made.** ★ Mechanism replacement: a part outside K runs M's rule on the inputs it receives in the
   hybrid (E195 does this). The alternative is Prakash et al.'s cross-model activation patching: set the part to
   the value it took in M's own run on the same input. The two differ whenever the part's inputs differ between
   the hybrid and M's run, so K changes.
3. **Chosen or searched alignments and pairing.** ★ Declared. Where one is fitted to maximise retention (E195's
   re-pairing), the class searched is declared; over an unrestricted class of maps π abstraction is known to
   become vacuous (Sutter et al. 2025, `lit_review/definition_survey/causal.md`).
4. **Which values 𝒱_u.** ★ Settings from M's and F's runs on D, pooled, the weighting declared (E195: M's only).
   Settings arising inside F_K would make retention depend on K.
5. **"Smallest".** ★ Fewest parts (the box: a search "gives only an upper bound on its size"), not minimal under
   inclusion.
6. **Parts of M that F lacks** (for example units F has zeroed). ★ They stay absent in every F_K, as E195 fixed;
   then F_∅ is M without those parts, not M. The alternative restores them outside K.

**From units to arbitrary aligned variables: does a current claim change?** None reverses; three need rewording.
"F with every part outside K following M's rule" is an intervened abstraction, a network only when the alignments
coincide. "A neuron's rule is fixed by its weights" becomes "by the weights and the alignment maps", so unchanged
weights no longer imply an unchanged rule. "Different levels give different answers" now includes different
alignments over the same units. One guarantee is lost with ε > 0: at ε = 0 a non-empty behaviour forces a
non-empty K, while with ε > 0 the change can sit in what the abstraction misses.

## (d) Do we need to lean more on Prakash et al.?

No. Every formal object is Geiger's: causal model, alignment, soft intervention, interchange values, approximate
transformation. Prakash et al. 2024 give no definition; their cross-model activation patching is a measurement at
the identity alignment, on one circuit's heads, same input, comparing outputs. It is the precedent for
intervening between a base model and its fine-tune and should be cited as that. Leaning on it further would mean
activation patching for F_K (choice 2), which changes the content. Attribution: Geiger for the objects, Prakash
for the precedent, and the two-model steps (two alignments with a pairing, the cross-model swap) stated as ours,
since Geiger's paper never relates two networks. By the project's survey the nearest formal two-network
precedents are Wu et al. 2022 and Grant 2025.

**Not checked.** Geiger pp. 24-29 and 34-43; Grant 2025 and Sutter et al. beyond the survey's notes; whether
E195's code matches clause 3 for heads.
