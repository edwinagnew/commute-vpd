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
