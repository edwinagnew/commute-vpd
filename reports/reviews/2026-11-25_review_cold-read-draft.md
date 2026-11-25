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

## MUST, pass 2 (appendices, 11-25 late morning)

3. **At VPD, the low ρ on trained fine-tunes cannot be read as "new".** appI:42-47 describes the rescale-only
   control, a fine-tune that can only rescale each MLP neuron's output (reuse by construction). At VPD it reads
   0.019 to 0.072 (seed 0), "as low as the full fine-tunes' 0.027 to 0.037". At top neurons the same control reads
   0.24 to 0.81, against 0.012 to 0.028 for the full fine-tunes. So the neuron reading passes this positive control
   on trained updates, and the VPD reading fails it. The body says nothing about this. sec5:10 says ρ "reads every
   trained fine-tune as mostly new", and sec7:11-13 puts "Definition too strict" first. A reviewer will read that as
   the definition calling every trained update new, when at VPD the measurement cannot see a trained rescale either.
   Fix: in §5.1, rest "mostly new" on the neuron reading and its rescale-only control. Say that at VPD a trained
   rescale-only update reads as low, so VPD's low values carry no call. In sec7, put that under "Measurements too
   weak" too.
4. **The re-gating share's only random reference on the 67M does not separate it from random.** The body's headline
   form (log-odds, Fig `known`) has no random baseline (appI:124). appH:20-22 says the KL form does have one, on the
   67M too, and that there "random changes of matched output change reach a fine-tune's own KL-form share" on most
   fine-tunes, "so a high reading does not tell a fine-tune from a random change". So the main positive result of
   contribution 2 rests on the one form with no null. Fix: one sentence in §5.2 saying so. (appH:21 also states a
   count, 67 of 88; under the user's 11-24 rule give values instead.)

Support for MUST 1: appI:174-178 says outright "No cheap reading tracks ρ closely". Cite it in the §5.2 sentence.

## MUST, pass 3 (realistic figures and appE, 11-25 ~10:00)
Both realistic figures plot `experiments/convergence_panel/results/wide/run4/score_n10.json` (E307); the CSVs equal the JSON and the scripts check each τ with scipy. Both captions are still TODO. I re-read the prior column and `stand_in.write_change` myself for M5-M7.

5. **The prior panel of Fig realistic_mirror mixes two units, and "outperformed by the trivial proxy of the base model's prior" (sec5:47) rests on that.** For the eight language-model tasks the prior is mean natural log-odds of the answer (−5.6 to −12.9). For category and sentiment it is a label-logit difference (0.84, 0.12; `src/commute/prior.py` `prior_mean_lo`). Those two sit on top only because of the unit, at tiers 2.5 and 2. On the eight like-for-like tasks the prior's τ falls from 0.32 to 0.12, and effective rank's rises from 0.17 to 0.51 (reviewer's recompute, scratch `tau8.py`, not yet checked by me). None of the four τ reaches the panel's critical τ 0.47. Fix: one unit or mark/drop the label tasks in that panel; give the τ values and say none separate at n = 10.
6. **appE:180-181 defines the prior as log10 geometric-mean probability; the figure plots natural log-odds or a logit difference** (+0.84 cannot be a log10 probability). State what was computed.
7. **sec4:19 says μ is uniform over answer positions; `write_change` sums over every position of the windows** (`flatten(0,-2)`), only the needed set being found at the answer position. So ρ in Fig kept_realistic weights every token position. Fix the text or the code; the SAE and neuron paths were not traced.

### SHOULD, pass 3
- sec5:59-61, the reversal (τ = −0.61): the same-size random update in the same figure also reverses (τ = −0.32); trained/random τ = −0.42. Part of the trend is each task's needed set, not the trained direction. Say so.
- The realistic recipe and read point are nowhere: plain SFT on the answer, no KL term, Adam 3e-5, batch 8, clip 1, up to 1,200 steps, read at KL ≈ 0.18 nats/token (±10%), mostly steps 5-9 (SQuAD 3-4, Spanish seed 0 400). appE:28-31's recipe is the built pairs' only.
- Name the unmatched states in both captions: sentiment (KL 2.51, 0.95, 4.33) and Spanish seed 0.
- sec4:44 "ten ordinary language tasks": 18 fine-tuned, 12 reached matched change, ten kept. Its list of task kinds omits lowercasing and CoNLL.
- appE:145-147 gives the logit-difference reading only for the label twins; realistic category and sentiment use it too.
- appE:22-24's random change (KL-rescaled, three draws) is not the realistic panel's (per-matrix norm, one draw per seed).
- appE:6 "every MLP neuron" vs sec4:15 top-97th-percentile neurons plus remainder: reconcile.
- Re-gating share reads 1.32 (SQuAD) and 1.31 (summary): the caption should say values above 1 are possible (appE:117).
- appE:134 "random maps 0.72 to 0.91" overstates: the second random map reached 0.40.

## MUST, pass 4 (related work and Geiger numbers, 11-25 ~10:10)
Every Geiger et al. 2025 number in §3 points to the right item (Def. 2, 5, 9, 28, 41, 44; Remark 6; §3.5.2, §3.6), and VPD §7.5 is the right section. I re-read the Kumar and VPD passages in the downloaded texts myself.

8. **main.tex:167, Kumar et al. 2022: the exception is misstated.** The draft says fine-tuning distorts features "unless they are already aligned with the task". The paper's exception is about the head: "If the head were somehow initialized perfectly at v⋆, then fine-tuning updates may not increase the OOD error" (LP-FT, Prop. 3.7). Fix: "fine-tuning distorts pretrained features, changing them only along the span of the training data, unless the head starts well aligned (LP-FT)".
9. **refs.bib geiger2025: pages 1--63, no number.** JMLR's own entry gives number 83, pages 1--64.

### SHOULD, pass 4
- main.tex:156, VPD §7.5: the up/down/modify/add reading is proposed for single gradient updates; for a fine-tune the section proposes running VPD on the weight diff ("one could apply VPD to only a 'weight diff' obtained by finetuning"). "We carry out that proposal" overreaches: say we apply the gradient-update reading to a whole fine-tune's change.
- main.tex:121, Jain 2024: the paper says the update lies mostly in the left null space of the instruction-tuned weights; "a rescaling of existing ones" has no antecedent.
- main.tex:166, Raghu 2020: "layer freezing, which is invariant to rotations": it is the CCA/CKA similarity measures that are invariant.
- main.tex:238: "τ is required to be ... the identity on inputs and outputs" is ours, not Geiger's (Def. 28 asks only a bijection). Write "we additionally require".
- main.tex:320: "points of variation" is not Geiger's term; Def. 41 has a distribution over interventions.
- main.tex:138 Minder 2025: add "on the first few tokens of random text". main.tex:139 Kassem 2026: name the delta-based loss and the 20/80 shared split.
- refs.bib venues missing: kotha2023forgetting ICLR 2024, gekhman2024hallucinations EMNLP 2024, yue2025 and sutter2025 NeurIPS 2025, zhong2026weights ICLR 2026 (jain2024safety, wang2025circuit unverified).
- No counterexample found to the two "to our knowledge, first" claims (main.tex:149, :159); keep the hedge.

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
- Pass 2 read appI, appD and appH; pass 3 read appE and both realistic figures (realistic-panel code path only). No read of the cut/ files (the blind read covers
  the reversal), or the rendered main.pdf layout.
- Pass 4 checked related-work bullets and 36 bib entries against arXiv/JMLR texts; the cut/ files' citations were not checked.
- All values come from definition.csv at v2 9529c07. None were recomputed from the runs' own results.

## Re-check of MUSTs 1-4 on v2 9dd94f4 (11-25 13:20)
All four are applied. One new MUST: the sec7 evidence sentence says the three rank correlations all "change sign" without the confident states. Only steering's does (appI:176-178: 0.18, -0.48, -0.81). There are three SHOULDs: sec5:10 is still flat on "mostly new"; appI:44 "300 steps" sits beside :47 "30 steps"; the SAE has no rescale-only control. Details: ~/commute-data/agents/writeup-lead/note-2026-11-25-redteam-must14-recheck.md.

## Re-check of MUSTs 5-9 on local paper commit 3b4a317 (11-25 13:53)
Pass, no MUSTs. I recomputed the eight-task tau-b values (0.197, 0.039, 0.512, 0.118) and they match. MUST 7 holds at all three featurizations (every path uses stand_in.write_change over every position). SHOULD: the stated bar (0.57) ignores ties in the tiers. The exact tie-aware permutation test puts effective rank at p = 0.061, right on the 95th percentile. Note: ~/commute-data/agents/writeup-lead/note-2026-11-25-redteam-musts59-recheck.md.
