# Where the paper draft disagrees with itself (2026-11-24)

The user asked for this in session on 2026-11-24: "the current paper feels a bit disjointed. different sections make
different assumptions and use different ideas ... in some places we say the prior is a confound, in others a control
and others a cheap proxy."

**In plain words.** One session read the whole live draft, body and appendices, and followed each recurring idea from
section to section. Eight ideas change role between sections, and about twenty names drift. Most of it traces to one
cause: the draft uses four different meanings of "reuse" and never says which one a section is using. This report
lists the problems and does not fix them. The framing decision at the end is left to a later session with the user.

Draft read: `paper/` at `f35454c` (branch `v2`): `main.tex`, `sec4_setup.tex`, `sec5_results.tex`,
`sec7_discussion.tex`, and appendices `appE`, `appG`, `appD`, `appI`, `appH`. Line numbers are from that commit. This
report is about ideas and names. Whether two numbers were measured the same way is covered by
`2026-11-24_review_consistency.md`.

## The root cause: four meanings of "reuse"

| meaning | measured by | where the draft uses it |
|---|---|---|
| The mechanism is kept up to a gain | the kept share ρ (the definition) | §3, §5.1 |
| The behaviour runs through base parts | switching off the base model's copy heads | §5.2, to check the labels |
| Base subcomponents are switched on differently | the re-gating share | §5.2 and §5.3, as a cheap reading |
| The base model "already does this" | the base model's prior, the raters' tiers | task design, §5.3 |

These disagree on our own data. The copy fine-tune is reuse on the second and third meaning (switching off three copy
heads removes 0.52 to 0.62 of its gain; its re-gating share is 1.09). On the first it is mostly changed (the head it
needs keeps 0.10 to 0.14 of its change as a rescale). The draft notes this clash in two red open notes but never
resolves it, so each section picks whichever meaning suits its result.

## Ideas that change role

Ordered by how much each one hurts the argument.

### 1. What the cheap readings are scored against

- As approximations of the definition: the contributions ("techniques for approximating our defined notion of
  re-use", `main.tex:125`), Set-up ("Cheaper proxies" of "the precise definition", `sec4_setup.tex:27-28`), and §3
  ("Every number we report is the approximate form ρ, or a cheaper reading of it", `main.tex:435`).
- As agreement with how the pairs were built: the scorecard caption says so outright, "not with the definition"
  (`appD_built_pairs_more.tex`, paragraph "The table").
- As agreement with blind raters: §5.3.
- Against the first reading: "No cheap reading tracks ρ closely" (`appI_results_detail.tex:144`).

### 2. The base model's prior

Five roles, and three different measurements.

- A confound to remove, listed as a control: `sec4_setup.tex:42`, `sec5_results.tex:100`.
- A fourth reading with an "expected direction for reuse": Figures 3 and 5, `sec5_results.tex:128`, and a row of the
  scorecard among the readings.
- A baseline a method must beat: Discussion, "What a cheap method must do to be trusted".
- Part of what defines the reuse side of a pair: "top guess", "runner-up guess", and "which the base model's
  induction heads already half-predict" (`sec4_setup.tex:51`).
- A rival explanation of the raters: "the raters may themselves track how familiar the text is"
  (`sec5_results.tex:132`).
- Measured as the probability of the trained answer (`appE`, "Base-model prior"), as the base model's loss on the
  training text (`sec5_results.tex:130`), and as "how familiar the training text is" (abstract).
- A number that looks contradictory: "the base model's prior alone orders every pair" (`sec5_results.tex:101`)
  against the scorecard's "agrees 1, no call 8". The footnote explains it, but the body does not.

### 3. The worked example in §3 against the finding in §5.1

`main.tex:405-409` illustrates "mechanisms, not values" with a fine-tune whose base induction heads are "untouched"
and "read as kept", and says the built pair "copy at a new marker" "is this case". §5.1 then reports that the copy
fine-tune's base head keeps only 0.10 to 0.14 of its change. The example promises the result the paper does not get.

### 4. What the neuron featurization shows

- `main.tex:259-260` and `sec4_setup.tex:19` say that with neurons as variables almost everything reads as reuse
  (0.999), and use this to motivate why the featurization matters.
- That number comes from an older wording read over the mean-ablation circuit, not from the definition. Under the
  definition, neurons and heads read low (`sec5_results.tex:22`, the copy head).
- `sec5_results.tex:39` spends a long sentence explaining why the two do not conflict.
- The count drifts: "four featurizations" and "the finest featurization of the three" in the same list
  (`sec4_setup.tex:12,19`), "of the four ... we show three" in §5.1. Heads are sometimes part of "neurons" and
  sometimes part of "top-activating neurons".
- The link from §3 to "the featurizations we read" (`main.tex:677`) lands in Appendix A, not in Set-up.

### 5. The status of VPD

- The headline featurization: abstract, Related work ("a natural place to ask"), and the only one that passes all the
  built edits.
- "A featurization in this sense only of a finer description", with a split "not fixed from M alone":
  `main.tex:430-433`. Set-up says only "this gives a featurization of M" (`sec4_setup.tex:18`).
- "VPD's low readings say nothing about whether mechanisms changed": `sec5_results.tex:37-38`.
- The key evidence in §5.1 and the Conclusion then comes from a head read as one variable and from top-activating
  neurons, not from VPD.

### 6. Which methods are ours and which are baselines

- Steering: under "Existing work" in Set-up (`sec4_setup.tex:43`), one of "three cheap readings" in §5.2, and "not a
  tool scored here" in §5.4.
- Neuron-level patching: existing work in Set-up, one of "four further readings" in §5.2, and a row of the tools table.
- Effective rank and update size: a cheap reading and a baseline, with no stated reason for the split.
- The count of readings: three (§5.2), "four further" (§5.2), "six" (`appI:121`), eight rows (scorecard). Set-up's own
  list ends with an unanswered note, "werent there any others".
- The re-gating share is "the first candidate" in Appendix A and one of three equals in the body.

### 7. What the paper claims overall

- Title and contributions: "A Mechanistic Foundation", "successfully validate them", "This improves on baselines".
- Abstract, Discussion and Conclusion: careful negatives ("we do not claim that the definition is right", "not
  validated", "no call").
- §5.4 says the tools "were not shown to work at this scale", which is not "we improve on them".
- The introduction asks three questions (how much post-training, RL against SFT, which mechanisms). RL appears only
  in an open note and an appendix. Qwen is in the abstract and gets one sentence in Results.
- What a low kept share means changes: "finds little reuse" (abstract), "mostly changed" (Conclusion), "say nothing
  about whether mechanisms changed" (§5.1 on VPD), "do not show that post-training builds new mechanisms" (Discussion).

### 8. "Controls"

Set-up's controls are chance, the prior and existing work (`sec4_setup.tex:38-44`). Appendix A's controls are twins,
hand-made edits and random changes (`appE_methods_detail.tex:13-25`). The "rescale-only control" and "gain-only
training" are two different runs with near-identical names, and Appendix A needs a paragraph to tell them apart.

## Smaller mismatches

- **The needed set.** The definition uses the needed set K. Many numbers are still read over the older mean-ablation
  circuit (marked in red). Appendix "Measuring the definition" still describes the older search. On Qwen ρ is read over
  every subcomponent. Set-up calls the sampled K "a lower bound"; §3 calls it an estimate.
- **The behaviour's inputs.** §3's discussion assumes D is "where the two networks' outputs differ"
  (`main.tex:403`); the formal paragraph on behaviour no longer says so.
- **Expected direction for effective rank.** Figures give it an arrow "expected for reuse", but `appI:93-94` says
  the copies have the higher rank, "the opposite of the intuition". The direction looks set after the result.
- **What "gain" is measured in.** Log-odds, the answer's probability, KL, and "improvement on their task", depending
  on the paragraph.
- **Random updates.** Matched in size, in KL, or in per-matrix norm; called "random updates", "random changes" and
  "Gaussian".

## Names that drift

| thing | names used |
|---|---|
| The definition's number | write share, kept share, approximate reuse, the literal reading, the definition's own reading, kept-share formula |
| "the share" on its own | the re-gating share in the appendices; ambiguous with kept, steering, neuron and top-neuron shares |
| Methods | techniques, proxies, cheap readings, candidate measures, measures, tools, baselines |
| Not reuse | new, built new, changed, mostly changed |
| Same up to scale | rescale, rescaling, gain, strengthening, up to a scalar |
| Fine-tunes with a label | built edits, hand-made edits, edits built by hand, gain edits; built pairs, known-answer tasks, key tasks, constructed fine-tunes |
| Fine-tunes without one | realistic, ordinary, real-text pairs, the rated panel |
| Task names | cipher, lookup, random map; copy, copy key, key copy, cued copy, copy at a new marker; top guess and rank-1; runner-up guess, mid and half-known |
| Spelling | reuse and re-use; behavior and behaviour |

Symbols used for two things:

- Δ: a mechanism's change in §3, the decomposition's remainder in the carrying appendix.
- U: the set of variables, and the write matrix.
- K: the needed set, and a KL in Appendix A.
- τ: the featurization, and Kendall's rank correlation.
- S: a task score, and the WeightWatch score.
- e: the remainder, and the explained change.
- The two models are M and F in §3, W₀ and W₁ in Appendix A.

Voice: the introduction is discursive prose. Results and appendices use working shorthand that a new reader has not
been given: "no call", "states", "bar fixed before the run", "descriptive", "read rows", "port", "stop bar".

## The decision this leaves open

The draft needs one stated answer to "what is the ground truth for reuse", and the user has not chosen one.

- **The definition is the ground truth.** Then the cheap readings are wrong on copy, they do not track ρ, and the
  paper has no positive result about detection.
- **How a fine-tune was built is the ground truth.** Every positive result in the draft is already of this kind. The
  definition becomes a hypothesis under test. The user's view on 2026-11-24: the resulting contribution ("a set of
  known fine-tunes and a scoring of methods against it") is not interesting enough as stated.

Two candidate claims that came up and were not checked against the evidence in any depth:

- Using a base mechanism is not the same as keeping it. The copy fine-tune relies on a base head and still changes
  most of it. This rests on one head in one model.
- The base model's prior explains most apparent reuse detection, including the raters. Only tasks that cross the
  prior with the mechanism separate methods that follow the mechanism.

Whichever framing is chosen, problems 2 to 8 and the naming table can be fixed without it. Problem 1 cannot.

## Not run

- No edit to the draft. Nothing was sent to writeup-lead.
- The cut appendix (`paper/cut/`), the uninput section files and the figures themselves were not read.
- Numbers were quoted from the draft and not traced to results files.
- The two candidate claims were not tested against the requirements list for the reuse definition.
