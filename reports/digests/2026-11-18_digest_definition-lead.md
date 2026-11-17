# definition-lead digest, 2026-11-18

P1.25, due 11-18: filled. The causal-rule retention test (tested, not adopted) passed both of its filed rules, but the gaps between pair members are tiny and follow the size of each fine-tune's weight update, so the score may not see reuse at all. I stopped here as you asked through compare-lead ("fill the cell and stop").

## Cells
P1.25 filled, 2 of 2 runs used. All other definition-lead cells are filled or closed (P1.1 to P1.24). The ladder's lowest open rung is L1.2 (ordinary fine-tunes imitating the hand-made pairs). I am not starting it: you said the points under Stated limits in the restated definition note are yours to settle first.

## What ran and what it found
- **Run 1 (E195).** The code checks all read 1: do-nothing, the circuit scaled ×2 and ×10, and an exact copy of the circuit. A circuit zeroed out and then rebuilt by fine-tuning reads 0.97, just under the turned-up circuit's 1. The filed rule was "lower", so it holds, by a small margin. Page: `reports/analyses/2026-11-17_analysis_rule-retention-fixed-run1.md`.
- **Run 2 (E196).** Five trained matched pairs, seeds 0 and 1. In all 8 counted pair-seeds the reuse member reads higher than its twin, so the filed rule holds. But every fine-tune reads 0.996 or more, and the gaps are far smaller than run 1's rebuilt gap.
  - The gaps rank the pairs in the same order as how much larger the twin's MLP weight update is. The memorised-fact pair has equal updates, and it reads a tie.
  - The neuron level cannot see attention, which is where the reuse members carry more of their change.
  - Page: `reports/analyses/2026-11-18_analysis_rule-retention-pairs-run2.md`.
- **Premise check: doubtful.** The question it put to you: "What observation, other than this score, would tell you a given fine-tune built something new? Do you have any item where that observation has been made?" Full text in `~/commute-data/premise/P1.25.md`.
- **A test that would separate the two readings.** Read this score on E187's size-matched pairs. That needs a new cell from you.

## Asks waiting on you
None filed. Two open definition questions, both noted in the pages:
- Should units that a fine-tune switched off count against retention?
- The three points under Stated limits in the restated note.

## Money
No pods are running. This session spent $0. Runs E195 and E196 were paid in an earlier session (about $17 there).

## Friction noted
- 819: a note key added to predictions.json broke the pod's config at start-up.
- 820: an empty `.git` was left on the pod after a failed clone.
- 821: there is no fallback list of GPU cards when H100 stock is empty.
- 824: the numbers ledger's jq filter can silently match nothing.

## Later the same evening: P1.26, due 11-19, filled
You opened P1.26 at about 21:00 and settled the plan's six questions in a dialog. You handed the embeddings question to Fable, which chose M's values. Both runs went on the cluster's debug queue at $0.
- **Run 1 (E197).** Every code check met its arithmetic value.
  - One gain per neuron or head keeps 0.01 to 0.06 of every trained change. One gain per head or block keeps about 0.
  - The per-site linear map of M's units keeps 0.98 to 0.999 on the old-task fine-tunes. The gaps between reuse member and twin run from 0.002 to 0.011, and they do not follow update size.
  - That map reads the memorised fact at 0.65, or 0.93 with F's embeddings, and the rebuilt circuit at 0.56.
- **Run 2 (E201).** A head was grafted onto frozen M; it is new by construction, and every build check held. The gain readings give it 0, and the per-site map gives it 0.13.
- **Page:** `reports/analyses/2026-11-18_analysis_held-model-retention-run1.md`, which covers both runs.
- **Premise check: doubtful.** The label comes mostly from how expressive the map is. Its question for you: "What, independent of these readings, decides how much re-mapping of the original's units still counts as the fine-tune reusing them?"

## Later still: the adoption recorded (user, 2026-11-18)
P1.26, due 11-19: filled; your adoption of the held-model wording as the working definition of reuse is recorded.
- **Recorded.** A dated `decisions.md` row (2026-11-18) states the wording as a cross-model interchange, the map class as a declared parameter, the stitching step, the stipulation and the caveats. The two rows of the second wording (2026-11-09, 2026-11-10) are marked superseded. `GOALS.md` never stated the second wording, so it gains one bullet that names the new working definition and marks the second wording superseded.
- **Log box rewritten.** `#existing-new` is about 550 words: one plain line, the interchange framing first, a short formal core (H and S), what is not interchange, the caveats, then what it replaces and the two follow-ups. A clerk is flushing and publishing it, with the glossary entry for the second wording marked superseded.
- **Plan against your wording.** They agree. The plan adds two things: the inputs are set to M's own inputs at the layer, and the fit reaches one layer deep. Your wording leaves both open. I kept the plan.
- **Follow-up cells not added.** The `fleet/cell add --rung L1.4` call for both cells was refused by the session's auto-mode permission check, so neither cell exists yet: (a) grafts at other sites and offsets, including an MLP neuron and an earlier layer; (b) the main reading on E187's size-matched pairs. Rules are filed before any reading in both. Nothing has run.
- **No new wording, cut-off or label.** None was added.

## Later: P1.27 and P1.28 filled (both due 11-20)
You added both cells after reading this digest. Both ran on RunPod H100 pods, because a process outside our jobs held 36 GB on the cluster's L40. Together they cost about $4.30.
- **P1.27 (E205), size-matched pairs.** After each pair's larger update was scaled down to its partner's size, the per-site map kept the sign of every gap between the reuse and new members, and five of the nine gaps widened. Shrinking an update lowered its own score, so the score depends on update length, but in the direction that does not explain the gaps. Page: `reports/analyses/2026-11-18_analysis_held-model-size-matched.md`.
- **P1.28 (E206), grafts.** Four grafted heads keep little (0.14 to 0.22). 256 grafted MLP neurons doing a random token lookup keep 0.954, near the old-task fine-tunes. The layer-1 head graft failed its build check and was not read. Page: `reports/analyses/2026-11-18_analysis_graft-sites.md`.
- **Your answer on the neuron graft:** "intuitively yes, linear mixing is probably more reuse than new". It is recorded as a tentative caveat of the definition, a consequence of the linear re-mix rule. The wording is unchanged and nothing new runs. The record card is `reports/decisions/2026-11-18_decision_neuron-graft-map-class.md`, and the log is published as v368.
- **Premise checks: both doubtful.** Both say the per-site map may report where a change sits and how linearly readable it is from M, more than whether it was built. The open rung L1.4 (a built control whose label does not follow from readability) is the question this depends on.
- **Friction:** 839 (debug cap stated three ways), 842 (pod inputs missing from E187's inventory), 843 (no .git on the pod), 845 (dates a day ahead of the clock across time zones; files not renamed, per you).

**Finished:** every definition-lead cell is filled or closed, and no open rung has a cell without your yes.
