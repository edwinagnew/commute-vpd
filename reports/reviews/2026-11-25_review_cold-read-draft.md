# Cold read of the first draft (writeup v2 9529c07), 2026-11-25

I read the draft as an outside reviewer would: the definition (main.tex §3), Set-up, Results, Discussion and the five
body figures with their data. Most of what an outsider would object to is already in writeup-lead's triage of the
blind read (`~/commute-data/agents/writeup-lead/review-2026-11-25-colleague-send.md`), so this list holds only what it
misses. The largest new doubt is that the cheap readings are called stand-ins for the definition, yet they agree with
how the tasks were built, while the definition's own reading of the same fine-tunes points the other way.

Line numbers are from `origin/v2:paper/<file>` at 9529c07. Evidence paths are in the writeup repo unless marked.

## MUST

1. **Proxies of what?** main.tex:104 ("proxies measurements of the above definition") and sec4:22 ("empirical proxies")
   present steering, effective rank and the re-gating share as stand-ins for ρ. But §5.2 scores them against how the
   tasks were built (Fig `known`), and on those same trained key fine-tunes ρ does not follow them. At top neurons
   both new tasks (cipher 0.021 to 0.028, prior-matched cipher 0.020 to 0.022) read above all three reuse tasks
   (0.012 to 0.020). At VPD the cipher reads highest (0.037 to 0.042), copy second (0.032 to 0.034), and the
   prior-matched cipher sits with the other reuse tasks (about 0.029). At SAE the two sides overlap
   (`figures/data/definition.csv`, block `trained`, side `trained`). appI:35-37 says so: "the order runs against the
   mechanism: the cipher reads highest, copy second". Meanwhile the re-gating share puts copy near 0.8 and cipher
   near 0.3. A reviewer will say the readings follow the construction labels, not the definition, and the paper
   cannot score them as estimates of ρ. The same tension reaches sec7:26: if the labels of trained fine-tunes may be
   wrong, then a reading that matches those labels has not been shown to be right. Fix: in the contributions and in
   Set-up, call them readings scored against the construction labels, and say in §5.2, in one sentence, that ρ does
   not order the same fine-tunes.
2. **The VPD featurization of F is not fixed from M.** main.tex:239 and :250 make a single featurization fixed from M
   the point of the construction ("If we had instead chosen to train a separate featurization for F, there is no
   way to a priori relate the features"). sec4:14 and main.tex:349-357 then build F's VPD features from F's own
   update, using the least-squares split. Subcomponent c becomes (u_c+du_c)(v_c+dv_c)ᵀ, with du and dv chosen as the
   minimum-norm fit. So at VPD, ρ depends on that split as well as on F, and the minimum-norm choice spreads an update
   over every overlapping subcomponent, which works against a proportional change. The bigram circuit's pure ×2 and
   ×10 edits read only 0.098 and 0.074 at VPD (definition.csv, `built_bigram`), against 0.33 and 0.34 at top neurons.
   Fix: state in §3 or §4 that the VPD reading departs from a fixed τ, why, and that its ρ depends on the split (and
   list it in appG departures).

## SHOULD

1. **ρ depends on each variable's units, and the appendix's escape is wrong for VPD.** main.tex:389 says the weights
   d_u are "harmless when every variable is a write into the same residual stream, as at VPD". Of the 24 decomposed
   matrices, only the attention-output and MLP-output ones write into the residual stream. The query, key, value and
   MLP-input subcomponents write into head or hidden space. Top-neuron variables also have per-neuron scales. The body
   reports ρ at all three featurizations with no mention of this. Fix: correct main.tex:389, and add a clause in §4
   "Measuring approximate reuse".
2. **§5.1 says "low ρ on both sides" (sec5:17) but hides the direction.** In Fig `def_reading`, the trained new side
   never reads below the reuse side as a group. It reads above at top neurons; at VPD the cipher is highest and the
   prior-matched cipher sits with the reuse tasks; at SAE the two sides overlap (values in MUST 1). That is the
   same reversal that §5.3 reports as τ = −0.61 and that sec7:26 explains with possibly wrong labels. Here, though,
   the labels come from how the tasks were built. Say it in §5.1, and have sec7:26 face it.
3. **Fig `known` has no reference on the re-gating axis, and the share passes 1.** appI:124: "The log-odds form has no
   random baseline of its own". The copy-at-new-marker point sits at about 1.09, and "share" goes unexplained in the
   body (appE:117 alone says it can exceed 1). An outside reader will ask what a random or do-nothing update reads.
   Fix: one clause in §5.2 or the caption saying there is no random reference and why, plus what a value above 1
   means.
4. **The direction of effective rank is not stated, and it is opposite the obvious intuition.** Fig `known` marks
   "Effective rank ↑" (higher means reuse). appI:113-114: "the copies have the higher-rank updates, the opposite of
   the intuition that a new task needs more rank". A reader will assume the reverse. Say which way it is read, and
   whether that direction was fixed before the data. In the figure, the separation is driven by the two text tasks
   (about 585 to 590), which sit on opposite sides.
5. **Built edits: say what each rescales at each featurization.** sec4:37 says ×2/×10 "should read as reuse". The
   kept clause, though, only accepts a rescaling of a variable's own mechanism. Scaling eight neurons' output weights
   changes the mechanisms of the variables downstream that read those neurons, and that change is not a rescaling.
   So the label comes from intuition, not from the definition. This is why pure rescaling edits read 0.04 to 0.10 at
   some featurizations: SAE induction ×2 and ×10 read 0.040, close to that family's random. Fix: one sentence saying
   that the reuse label for built edits is by construction of the edit, and that ρ reaches 1 only when each needed
   variable's own mechanism is rescaled.
6. **The top-neuron featurization as written is not a featurization.** sec4:15 makes the variables "the neurons
   which above their own 97th percentile". A set of variables picked per input is not the fixed bijection τ of
   main.tex:238. Define the variable's value: for example, the activation above the base model's threshold and 0
   otherwise, with the rest in the remainder.

## Re-checks from my earlier reviews
- 4ce1438 (reports/reviews/2026-11-24_review_paper-4ce1438-fig2-p163.md): both MUSTs are moot in v2. The sentences
  "only through the SAE error terms" and "Four randoms above 0.03" are gone, rewritten out of §5.1. SHOULD 1 (the
  bigram reads 0.07 and 0.10 at VPD, induction 0.04 at SAE) applies to the Fig `def_reading` caption, which is still
  TODO. When it is written, say "read above their matched randoms", not "read as reuse at every featurization".
- P2.81 gain-per-token SHOULD: not re-checked this pass.

## Not run
- No read of the cut/ files, appH beyond §5.5's claim, Fig `realistic` or Fig `kept_realistic` (the blind read covers
  the reversal), or the rendered main.pdf layout.
- No check of the related-work citations against the papers.
- All values come from definition.csv at v2 9529c07. None were recomputed from the runs' own results.
