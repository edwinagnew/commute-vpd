# The working definition of reuse, consolidated (2026-11-18, definition-lead)

*2026-11-18: wording fixes from the review (`reports/reviews/2026-11-18_review_definition-causal-abstraction.md`,
defects D5 to D9). The content is unchanged. The review's content repairs wait on the user in the log card "Four
repairs the definition needs" (`reports/decisions/2026-11-18_decision_definition-repairs.md`). "Def. n" is Geiger et
al. 2025 in the JMLR numbering; the arXiv HTML numbers differ, and its Def. 41 is Value Merge.*

One page that gathers what the user has decided today. It is assembled from `decisions.md` rows dated 2026-11-18 and
the user's notes to writeup-lead. It adds nothing; where they disagree, the newest row wins and the clash is named
under "Clashes" below. Sources: `GOALS.md` (Working definition), `decisions.md` rows "Working definition of reuse,
stated in causal-abstraction terms", "Working definition of reuse without ρ", "A level is a decomposition, not an
alignment", "A refinement ... may be approximate" (all 2026-11-18); `theory/2026-11-18_definition_causal_abstraction.md`;
`theory/2026-11-18_level-as-decomposition.md`; decision card `reports/decisions/2026-11-18_decision_needed-parts.md`.

## The sentence (decided)
A fine-tune F reuses the original model M, for a behaviour at a declared level, to the extent that the parts F needs
for that behaviour still compute what M's parts compute when given the same inputs.

## Its terms (decided)
- **Level / refinement.** Our paper's word is "refinement"; the word is ours, not Geiger's, who has none. The log,
  decisions and theory notes say "level" or "decomposition". A refinement gives each network N a causal model Ñ: a
  finer one, or for unit partitions N with its units grouped. All the Ñ share one signature: the input, a set of parts
  U and the output. N is an abstraction of Ñ, either exact (Constructive Abstraction, Geiger et al. Def. 33, JMLR
  numbering) or approximate (an Approximate Transformation, Def. 41, under the alignment's maps τ^π and ω^π). The
  input and output map by the identity. The refinement is declared or fitted on M
  only, then read on both M and F, so a part is the same variable in both networks.
- **Cases.**
  - Unit partitions (neurons, heads, MLP layers, blocks). Ñ is N with its units grouped, a variable merge (Def. 37),
    so the two are inter-translatable and the merge is exact.
  - Rotations (PCA, DAS).
  - SAE latents.
  - Transcoder and cross-layer transcoder latents. These differ from SAEs: M's transcoder read on F has the same
    encoder, which reads the MLP's input, so each latent keeps M's rule by construction. Every change to the MLP's own
    weights lands in the remainder; only changes upstream of the MLP are charged to latents.
  - VPD component activations. Reading these on F declares a split of the update δW; under the do-nothing update, all
    of δW is remainder.
  - Coarser unit partitions and coarser merges of parts: variable merges (Def. 37).
  - Not a case: attribution graphs (Ameisen et al. 2025). Their local replacement model freezes attention patterns
    and error terms per prompt, so a part has no rule to compare on other inputs. They serve only as a per-input
    reading.
- **Remainder (optional).** A lossy featuriser such as an SAE or VPD may add a remainder part, which makes the
  abstraction exact (ε = 0). Without a remainder, the method is read as an approximate refinement (ε > 0).
- **ε.** ε is the error of the abstraction on each network, measured as KL on outputs: Def. 41 with KL in place of
  its similarity, the mean as its statistic, and the behaviour's inputs as its distribution (Remark 42). Lower is a
  better fit. It is kept and reported beside every reading.
- **A part's rule.** A part's rule is its mechanism in Ñ.
- **F_K.** F_K is F with M's mechanisms swapped into every part outside K. It is always a runnable model.
- **The parts F needs (K).** K is a smallest set of parts such that F_K keeps the behaviour.
- **Declared with every reading** (restored from the earlier rows; no values are set here):
  - the distance d by which a part's outputs are compared: declared per reading, as in the earlier rows;
  - the threshold τ for "F_K keeps the behaviour": declared per reading, as in the earlier rows;
  - the value sets 𝒱_u, the inputs of each part on which its rule is compared: declared per reading, as in the
    earlier rows;
  - the aggregation over K: declared per reading, as in the earlier rows;
  - the rule when K is not unique, that is, when several smallest sets keep the behaviour: declared per reading, as
    in the earlier rows.
- **Reuse.** Reuse is how far the parts in K keep M's rules. "Keeps M's rule" means that, given the same input values,
  the part's mechanism in F gives the same output as its mechanism in M.
- **Readings, not the definition.**
  - ρ: the mean rule retention over K. At the neuron level it saturates (E208: 0.999 or more on every fine-tune), and
    the user said the neuron level is "not a high-enough level abstraction to detect new".
  - The held-model survival S.
  - Every reading so far uses the exact unit-partition case.

## Open (not decided)
1. **Which parts count (ask 48).** As written, K never contains a part that F left unchanged. So a new trigger in
   front of an old circuit F left unchanged reads as new. The alternative is to count every part the behaviour runs
   through. If that is chosen, the user has already picked the rule for knocking out an unneeded part: its mean over
   the behaviour's inputs.
2. **Which input values a part's rule is compared on** (restated after the review's D6). The decided wording,
   "given the same input values, the part's mechanism in F gives the same output as its mechanism in M", already
   compares two mechanisms on the same inputs, which is A below. B compares values, not mechanisms, and would change
   decided wording. What is open is whose input values the comparison uses: those in F's runs, those in M's runs, or
   both pooled, which the review suggests. The question as first asked:
   - (A) what reaches it in F's own run, so its mechanism is compared;
   - (B) what it received when M read the same text, as in activation patching.

   The lead recommended A. The user did not know which to pick.
3. **Where to compare when ε_F > 0.** One option is to compare the hybrids inside the refined models: the keep-test
   runs against F's refined model, and the difference is taken between the two refined models. This keeps the
   definition total when ε_F > 0. In the exact case both ways coincide. Raised by writeup-lead and not yet answered.
4. **§5.** Re-gating is scored against the definition as a method (decided). Whether it is also "the definition read
   at the VPD refinement" is set aside.
5. **Four repairs from the review.** The guard against fitting on F, what non-part variables run, reporting the
   remainder's share of K, and one shared causal order. Card `decision-definition-repairs`
   (`reports/decisions/2026-11-18_decision_definition-repairs.md`).

## Clashes between files (GOALS.md and the level note fixed 2026-11-18; the alignment-form row stays as a superseded record)
- **`GOALS.md` on exactness.** It still says the parts "merge back into the network exactly" and that "whether ε_F
  stays ... is open". The 2026-11-18 refinement row has since decided that ε is kept and that approximate refinements
  are allowed.
- **`theory/2026-11-18_level-as-decomposition.md`.** Its Part A requires exactness, and its Part B item 1 lists ε as
  open. Both are superseded by the refinement row.
- **The causal-abstraction row.** It speaks of "one alignment (cells Π_u, value maps π_u)". The level row replaced
  that alignment form with a decomposition.
