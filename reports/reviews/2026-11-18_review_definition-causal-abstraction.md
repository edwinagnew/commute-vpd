# Review: the working definition of reuse in causal-abstraction terms (2026-11-18)

*Independent review, read-only, for definition-lead. Object reviewed: `theory/2026-11-18_definition_consolidated.md`
and its sources (`theory/2026-11-18_definition_causal_abstraction.md`, `theory/2026-11-18_level-as-decomposition.md`,
the 2026-11-18 rows of `decisions.md`, `GOALS.md` → Working definition, `reports/decisions/2026-11-18_decision_needed-parts.md`,
`lit_review/definition_survey/causal.md`, `.../abstraction_between_networks.md`). Primary source: Geiger et al.,
arXiv 2301.04709v4 (8 May 2025, the latest version; the same text as JMLR 26, 2025), read from the project's PDF
extraction `~/commute-data/lit_review_pdfs/survey/2301.04709/paper.txt` and checked against arxiv.org/html/2301.04709v4.
Nothing was run. This review proposes no new definition; every repair below is the smallest one, and each says whether
it changes content or only wording.*

I checked every causal-abstraction term and citation in the working definition of reuse against Geiger et al., then
asked whether the definition makes sense for comparing a base model with its fine-tune, part by part, under each kind
of decomposition. The terms are mostly used correctly, and the definition numbers match the journal and PDF version,
but four real gaps remain: the fit number (ε) says nothing whenever a remainder part is used; the VPD case breaks the
rule that the decomposition is fixed from the original model alone; the swap does not say what happens to the
variables that are not parts (the output above all); and the definition stops being total when the decomposition fits
the fine-tune only approximately. None of these overturns the sentence the user chose or any reading so far, since all
readings use exact unit partitions, but each needs a sentence of repair before the paper states the definition formally.

## 1. Terminology

Numbers are those of the PDF / JMLR version (v4). **The arXiv HTML rendering numbers differently** (its counter also
counts remarks): there Alignment is Def. 34, Constructive Abstraction 36, Approximate Transformation 45, Interchange
Interventions 49, Variable Merge 40, Bijective Translation 30, and **HTML "Def. 41" is Value Merge**. Any citation by
number should say "JMLR version" or give the name.

| Term as used | Paper's term, Def. (PDF/JMLR) | Verdict | Fix |
|---|---|---|---|
| "exact constructive abstraction, Def. 33" | Def. 33 *Constructive Abstraction*: "H is a constructive abstraction of L under an alignment ⟨Π, π⟩ iff H is an exact transformation of L under (τ^π, ω^π)" | Correct. "Exact" is redundant (Def. 33 is exact by definition) but harmless. | Wording only: "constructive abstraction (Def. 33, JMLR numbering)". |
| "approximate abstraction, Def. 41" | Def. 41 *Approximate Transformation*. The body says "we can say that M∗ is an η-approximate abstraction of M", so the phrase is the paper's, but the definition is the graded form of **exact transformation (Def. 25)**, not of constructive abstraction: it takes any (τ, ω), with no alignment. | Name slightly off; content gap. Without saying "under the alignment's (τ^π, ω^π)", the approximate case loses the cell structure that makes "the same part" meaningful. | Wording: "an approximate transformation (Def. 41) under the alignment's (τ^π, ω^π)". |
| "refinement" ("the paper's word") | Not Geiger's. The word does not occur in the paper. Geiger's nearest notions: bijective translation (Def. 28) and "modular feature learning as bijective transformation" (§3.6). | Fine as the project's coinage if "the paper" means the project's paper. Misleading if read as Geiger's. | Wording: "our paper's word; Geiger has none". |
| Direction: N is an abstraction of the finer Ñ | Def. 33 needs τ from low to high, partial surjective, cell-wise. With a remainder, (f, e) ↦ dec(f) + e is surjective onto Val(h), so N as the high-level side is the only direction in which a lossy featuriser plus remainder is *exact*. Geiger himself puts the network on the low side and treats SAEs the other way round, as a translation network → features that is only approximate (§3.6.1: "sparse autoencoders are approximate transformations of the underlying model"). | Correct, and the right choice for the remainder construction. Two caveats: (i) for parts coarser than N's own variables (heads, layers, blocks over scalar neurons) Ñ is a **variable merge (Def. 37) of N**, i.e. coarser, not finer; with identity value maps the merge is a bijection on total settings, so exactness holds either way. (ii) See defect D3: exactness under Def. 33 constrains only interventions on *whole cells*. | Wording: for unit partitions write "Ñ is N with its units grouped (variable merge, Def. 37); the two are inter-translatable", not "Ñ = N, a refinement". |
| Signature, variables, mechanisms | Def. 2 (signature Σ = (V, Val)), Def. 5 (causal model, mechanisms F_X : Val_V → Val_X; parents induced by the mechanisms, Remark 6). | Correct. Note: parent sets are *induced by mechanisms* (Remark 6), so M̃ and F̃ on one signature can have different parent sets. | Content (small): require that M̃ and F̃ respect one causal order (the architecture's layer and position order), so every hybrid is acyclic. Holds for all current cases; state it. |
| "the same variable in both networks" | Geiger always relates models on two signatures through one alignment; two models on *one* signature is simply two causal models over the same Σ. | Correct and well-defined once the shared signature is required. Note the shared signature means M has every part, so the theory note's "a part M lacks keeps nothing" can only mean a unit F zeroes or adds outside the signature. | None. |
| Alignment ⟨Π, π⟩ (row "stated in causal-abstraction terms") | Def. 31. | Correctly cited, but superseded by the decomposition form; the clash is already listed. | Fix the row as the consolidated page says. |
| Interventionals | Def. 11 (functions from old mechanisms to new). | Correct where used (theory note: "a constant interventional, Def. 11"). | None. |
| The swap (F̃ with M̃'s mechanism on each part in S) | Def. 10 *Soft Intervention*: "a family of functions {I_X} … replace each function F_X with the function I_X". Any I_X is allowed, so I_u := F_u^{M̃} is a soft intervention on F̃, well defined because M̃ and F̃ share a signature. At the network level, F_K is F under an **interventional (Def. 11)** on the merged variables (e.g. h's mechanism replaced by dec(f with some latents following M̃) + e). | Correct as a soft intervention; **not** an interchange intervention. Def. 44 is a *hard* intervention fixing variables to values from source-input runs of the *same* model. Novel only in where the replacement function comes from; no new operation is needed. | Wording: keep "soft intervention whose function comes from M̃". Remove "cross-model interchange intervention" (superseded held-model row) wherever it still describes the swap; that phrase names Prakash et al.'s cross-model activation patching, which is reading B (§4, Q2). |
| "interchange values, Def. 44" (theory note, clause 4) | Def. 44 defines interchange *interventions*, within one model: Proj_X(Solve(M_s)). | Loose. Using values from M's runs inside F is a cross-model use not covered by Def. 44. | Wording: "settings of In(u) that occur in M's and F's runs (the values an interchange intervention, Def. 44, would use)". |
| "accuracy ε, KL on outputs" | Def. 41 needs Sim on total settings, a distribution P over Domain(ω), and a statistic S; "if this number is above a particular cutoff η". Restricting Sim to outputs is the paper's own practice (IIA, Def. 48, projects on X_Out). P over input interventions only (Remark 42) is the "input–output accuracy" special case (Def. 48: "IIA is equivalent to input-output accuracy if we further restrict ω … to the same single input"). | Matches Def. 41 as the **input–output** special case, not as interchange-intervention accuracy. "Accuracy" is inverted: Geiger's statistic is a similarity (higher is better, cutoff η); KL is a distance. | Wording: call ε "approximation error (Def. 41 with Sim = KL on outputs, S = mean, P on the behaviour's inputs only, Remark 42)". Content: see D3. |
| Variable merge for coarser parts | Def. 37. | Correct. | None. |
| Rotations as bijective translations | Def. 28; PCA example in §3.6.1; DAS via Def. 46. | Correct. | None. |
| SAE / transcoder | §3.6.1: SAE as τ = enc only when reconstruction is perfect; "in practice … approximate transformations". The remainder construction is not in Geiger (no error node anywhere; confirmed, as the level note says). | Correct attribution: the remainder is the project's (precedents: Marks et al. 2024 error nodes, Ameisen et al. 2025; not re-read here). | None. |
| Parameters / weights | Geiger's only sentence: §3.7, adapter tuning "can be viewed as interventions on model parameters". Nothing on decomposing weights. | VPD case is the project's construction. | State it as ours (the level note already does). |
| Sutter et al. 2025 as the guard's source | Their theorem: with unrestricted alignment maps φ, any network is an input-restricted distributed abstraction of any algorithm (causal.md, p6 Thm 1). | An analogy, not a theorem about this definition: their relation is algorithm-to-network alignment, ours is one decomposition read on two networks. | Wording: "by analogy with Sutter et al. 2025". |

## 2. Compatibility per ontology

**What a method must supply to plug in (the minimal interface).** (1) For every network N of the architecture, a
causal model Ñ on one shared signature: input, parts U with value spaces, output, and a mechanism per part that can be
**evaluated on arbitrary parent values** (needed for reading A) and run in a hybrid. (2) A merge τ from Ñ to N's
variables, either exact (with a remainder) or with a reported approximation error. (3) Cells that cover **every**
variable of N on a path from input to output, or an explicit statement of which mechanism each non-part variable
runs in F_K (D2). (4) One causal order shared by M̃ and F̃. (5) A rule that produces F̃ from M̃ and F, declared before
any reading and not chosen to move it (D1). Anything that supplies these plugs in; anything that cannot supply (1)
with input-independent mechanisms does not.

| Ontology | Fits? | Notes |
|---|---|---|
| Unit partitions (neurons, heads, layers, blocks) | Yes, exact | Ñ is a variable merge of N (coarser, not finer, once parts group units). Must state: per position or across positions (a head's rule across all positions, or per token); what embeddings, norms and the unembedding are (parts, or M's mechanism in every F_K). E195/E196/E208 put them at M's values (P1.26 row), which is the right default but is not in the definition. |
| Rotations (PCA, DAS) | Yes, exact | Bijective translation, Def. 28 / Def. 46. The reading is **not invariant** under bijective translation, which Geiger calls "two equivalent … descriptions" (p14): a random rotation of a layer spreads a localised change over all directions, so each needed direction changes a little (degenerate high, §3). "Fitted on M only" does not prevent this, since a random rotation is M-independent. |
| SAEs (per site) | Yes, exact with remainder | F̃ is canonical (M's encoder and decoder applied to F). A latent's rule is enc ∘ (F's computation from its parent parts to the site): the encoder is shared, so a rule change is a change in the network between parts. If the SAE covers only one site, the latents' parents are the input and their rule is the whole prefix, so reading A and reading B coincide there, and the suffix network sits in the output's mechanism (D2). New directions F writes that M's SAE cannot encode go to the remainder, which then carries the change and reads "new" (SAEs fitted on a base often transfer to its chat model, Kissane et al. 2024, from memory, not re-read). |
| Transcoders, cross-layer transcoders | Exact with remainder, but read differently from what the case list suggests | M's transcoder applied to F has the same encoder, so a latent's local mechanism is a function of the MLP *input*, identical in M̃ and F̃ by construction; **every change to the MLP's own weights lands in the error (remainder) part**. Changes upstream (attention) are charged to latents. So the transcoder case reads a fine-tune of MLP weights as "remainder changed", whatever the change does. Say so in the case list. |
| Attribution graphs (Ameisen et al. 2025) | No, as stated | The local replacement model freezes attention patterns and error terms *per prompt*. Those are constants, not mechanisms, so a part's rule cannot be compared on other inputs and a swap would freeze F's attention. Usable only as a per-input linearised reading, not as a refinement. |
| VPD / APD / SPD | Fits formally, with two repairs | Parts a_c = v_cᵀx and remainder e = Δx per matrix are activations, so "component activations as variables" holds. A part's mechanism is v_cᵀ applied to **the non-parametric glue** (norms, attention softmax, GELU, residual sum) applied to its parents' writes Σ u_c′ a_c′ + e′. So u_c (the write direction) is **not in u's own mechanism**; it enters the mechanisms of u's children, and for the last layer the output's (D2). A change dU is therefore charged to downstream parts, dV to the part itself. Formally, "reading it on F declares a split of δW" means: F̃ is fixed by choosing per-component (u_c′, v_c′) for F, with Δ′ := W + δW − Σ_c u_c′ v_c′ᵀ (the residual, which also absorbs the second-order dU dVᵀ term). **Every** split is then exact, so ε cannot tell splits apart, and the split reads δW, i.e. F (D1). Γ (causal-importance masks) plays no role in the definition; say whether it should. |
| Crosscoders (model diffing) | Not under the current guard | A crosscoder is trained on M's and F's activations together, with per-model encoders/decoders and shared latents. It supplies a shared signature, so the swap is defined, but it is fitted on F, breaking "declared or fitted on M only"; its latents marked F-only are parts whose M-mechanism is degenerate, and L1 crosscoders are known to produce spurious model-specific latents (Minder et al. 2025, from memory, not re-read). It plugs in only if the guard is reworded as in D1 *and* the user accepts a jointly fitted dictionary. |
| Future ontologies | Through the interface above | Requirement (1) excludes per-input objects (attribution graphs, per-prompt circuits); requirement (5) is where a jointly fitted method needs the user's decision. |

## 3. Defects, ranked by severity

**D1. (High; content of the guard.) The VPD case contradicts "declared or fitted on M only".** Every non-trivial
split (least-squares, adjoint, rescaling) is computed from δW, so F̃ is fitted on F; only do-nothing is not, and it
trivially reads "new". Crosscoders hit the same rule. Since every split is exact (D3), nothing in the definition
constrains the choice, which is exactly the freedom the Sutter analogy warns about.
*Smallest repair:* "F̃ is produced from M̃ and F by a rule declared before any reading (the same encoder; a named split
of δW), never chosen to change the reading." Changes content of the guard (it now allows reading F through a fixed
rule), not of the sentence.

**D2. (High; content, a missing clause.) What runs on variables that are not parts is unspecified.** The shared
signature is {input} ∪ U ∪ {output}. The swap acts on U only, so in every F_K the output keeps F's mechanism (the
unembedding, final norm, and for a one-site SAE the whole suffix network). Then **F_∅ = M is false** whenever F changed
the readout or anything outside the parts, and "K = U always keeps the behaviour, so a reading always exists" survives
only because F_U = F. The same holds for the VPD write vectors of the last layer and for the glue mechanisms that the
consolidated page never assigns.
*Smallest repair:* "Every variable of N other than the input lies in some part's cell; or, the output (and any non-part
variable) runs M's mechanism in every F_K." The P1.26 row already does the second for the 67M readings. Content
(a clarification that makes a stated property true).

**D3. (High; content of what is reported.) ε says nothing in the remainder case, and in no case tests the parts.**
Def. 33 constrains only interventions on whole cells (Remark 32: ω^π is "only define[d] … on low-level interventions
that target full partition cells"). With a remainder, the merge is exact for *any* encoder, so ε = 0 by construction
and "ε reported beside every reading" reports a constant. Without a remainder, ε as specified (KL on outputs, P on the
behaviour's inputs) is input–output fidelity (Remark 42, Def. 48's remark), not interchange-intervention accuracy: it
never checks that swapping single parts in F̃ corresponds to anything in F. A weak encoder pushes F's change into the
remainder, which then reads "new" (degenerate low), and nothing reported shows it.
*Smallest repair:* report, beside ε, either the share of K that is remainder (the level note's Part B proposal) or
ε of the same decomposition with its remainder removed. Content of the reporting only; the definition is unchanged.

**D4. (Medium-high; content, pending decision.) Not total when ε_F > 0 without a remainder.** Behaviour D is defined by
F's outputs; F_K is a run of F̃_K. If F̃ itself does not keep the behaviour within τ, no K exists, since K = U gives F̃,
not F. This is open question 3, and it must be decided for the definition to be total. (ε_M ≠ ε_F is otherwise harmless:
the rules are compared in M̃ and F̃ whatever their fit, and the readings are only interpretable jointly with ε.)

**D5. (Medium; wording, with content risk.) With ρ dropped, "to the extent that" has no declared parameters.** The
consolidated page lists no distance d, no threshold τ for "keeps the behaviour", no 𝒱_u, no aggregation over K, and no
rule for a non-unique K (several smallest sets with different retention). The 2026-11-18 "revised" row declared "level,
behaviour inputs, τ, d"; the theory note kept "the range over smallest sets". Without these the sentence is total as a
concept but yields no reading.
*Smallest repair:* add "declared with every reading: decomposition, behaviour inputs, τ, d, 𝒱_u, the aggregation, and
the range over smallest sets". Wording (restores what earlier rows already said).

**D6. (Medium; wording.) Open question 2 is framed as if undecided, but the decided text already picks A.** The
consolidated "keeps M's rule" says "given the same input values, the part's mechanism in F gives the same output as
its mechanism in M": a comparison of two mechanisms on the same arguments, i.e. reading A. Reading B (the part's value
in M's run against its value in F's run) compares values, not mechanisms, and would *change the decided wording*. What
is genuinely open is the distribution over v: F's runs, M's runs, or both pooled (theory note choice 3, ★ pooled).
*Repair:* restate Q2 as "which parent values: F's runs, M's runs, or both". Wording.

**D7. (Medium; wording in the case list.) Transcoders and attribution graphs do not behave as "the same form".**
Transcoder latents keep M's rule by construction; all MLP-weight change goes to the error part. Attribution graphs fail
interface requirement (1). *Repair:* one sentence per case in the list. Wording.

**D8. (Low-medium; wording.) "Refinement" for unit partitions.** Heads, layers and blocks over neuron-level N are
variable merges (coarsenings) of N, so "Ñ = N" and "finer" are wrong unless N is defined at that grain. Exactness is
unaffected. *Repair:* "N with its units grouped (Def. 37)". Wording.

**D9. (Low; wording.) Citation hygiene.** Def. 41 is "Approximate Transformation" (graded exact transformation), so
add "under the alignment's (τ^π, ω^π)"; cite JMLR numbering explicitly since the arXiv HTML numbers differ (HTML
Def. 41 = Value Merge); "refinement" is not Geiger's word; "accuracy" should be "error" for a KL; Sutter et al. is an
analogy; "interchange values, Def. 44" is loose. Wording.

**D10. (Low; content, small.) Acyclicity of hybrids.** Parents are induced by mechanisms (Remark 6), so M̃ and F̃ on one
signature can differ in parent sets; a hybrid is acyclic only if both respect one order. True of every current case.
*Repair:* one clause requiring a shared causal order. Content (a stated condition).

**Not defects, but to say in the paper.** Degenerate readings, all known in part: (a) a remainder swallowing
everything (a trivial encoder) reduces the decomposition to one part per site and reads every needed change as new;
(b) the whole hidden computation as one part reads any new behaviour as new; (c) very fine or rotated decompositions
spread a change thinly and read high (E208's saturation is this case); the reading is not invariant under the bijective
translations Geiger treats as equivalent descriptions, and fitting on M only does not prevent a choice of basis that
spreads the change. The guard against (c) is declaring the class before the reading and reporting at two or more
decompositions, which the plan already does. (d) Open question 1: as written, a part whose mechanism F left unchanged
is untouched by the swap and so never in a smallest K; K therefore holds only changed parts, and reuse measures how
little the needed changed parts changed, not how much old machinery the behaviour runs through.

## 4. The open questions: the literature's standard choice

- **Q1 (which parts count).** The standard in circuit analysis is to count every component the behaviour runs
  through: sub-circuit analysis (Geiger §3.5.2) ablates the connections outside a circuit C to declared values g and
  asks whether C "preserves the behavior B". Mean ablation (Wang et al. 2023) and resampling (Chan et al. 2022, causal
  scrubbing) are the two standard g. The user's answer, the mean over the behaviour's inputs, is the standard mean
  ablation. Note that option (b) then mixes two interventions (swap for the rule test, mean ablation for need); both
  are soft/hard interventions in Geiger's sense, so it remains well defined.
- **Q2 (part input A or B).** For "same mechanism", the structural-causal-model standard is A: mechanisms are functions
  (Def. 5), and two are the same if they agree on the parent values (Woodward's invariance under intervention on the
  parents, a hard intervention on In(u), Def. 9). B is the interchange-intervention / activation-patching standard
  (Def. 44; Prakash et al. 2024's cross-model patching), which tests whether a variable *carries the same value*, not
  whether it computes the same function. The decided wording is A (D6). For the distribution over v, Def. 41's P is
  "a probability distribution … used to describe which interventionals are expected": the standard is the values that
  occur on the data, here both networks' runs pooled.
- **Q3 (where to compare when ε_F > 0).** The standard in replacement-model circuit work is to judge a circuit inside
  the replacement model and normalise against it, e.g. faithfulness (m(C) − m(∅)) / (m(full model) − m(∅)) in Wang et
  al. 2023 and Marks et al. 2024, or to keep error nodes so the model is exact (Marks et al. 2024; Ameisen et al. 2025).
  Both amount to the option raised by writeup-lead: run the keep-test against F̃ and report ε beside it. It is also what
  makes the definition total (D4). Geiger's own measure (Def. 41, Def. 48) compares the interpretation's interventions
  against the network's, which here would require mapping single-part swaps to N, i.e. the interventional of §1.

## Not checked

- Geiger pp. 24-38 (intervention algebras for features, §3.1-3.5 beyond 3.5.2) and §3.8; Remark 47 / the missing
  "Definition 47" that Def. 48 cites (absent from both renderings, apparently a typo in the paper).
- Sutter et al. 2025 beyond the survey's notes; Grant 2025, Wu et al. 2022 beyond the survey.
- Marks et al. 2024, Ameisen et al. 2025, Wang et al. 2023, Kissane et al. 2024, Minder et al. 2025: cited from memory
  for standard practice, not re-read.
- The research log's overview box and `paper/sec4_causal_abstraction_bullets.md` in the writeup repo.
- Whether E195/E196/E208 code puts the output and embeddings at M's mechanism in every F_K (D2), and how heads are
  handled across positions.
- Whether VPD's Γ (causal importance) should enter the VPD case.
