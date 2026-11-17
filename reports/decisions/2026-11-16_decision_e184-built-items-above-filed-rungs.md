# Decision: the ranking read seven of nine built items above the grade they were built at. What should the gate page say about the fourth wording?

*2026-11-16, definition-lead. Log card `decision-e184-built-items-above-filed-rungs`. Page: `reports/analyses/2026-11-16_analysis_e184-simplest-intervention.md`. Results: `experiments/simplest_intervention/results/run2.json`, `run3.json` (controls in `run1.json`). The fourth reuse wording is tested here, not adopted. This is a finding, not a request for a run.*

P1.20, due 11-15: the ranking test of the fourth wording is done (its four runs are spent) and it reversed what we filed. The wording says a fine-tune reuses the original model to the extent that it is the original under a simple intervention on the original's own variables, graded by a rung: rung 0 none, rung 1 re-set some of the original's variables, rung 2 as rung 1 but a shift may read other variables on the list, rung 3 add one or two free new directions, rung 4 from scratch.

## The question

The ranking read seven of nine built items above the rung they were built at, so it does not yet grade built cases as built. What should the gate page say about the fourth wording?

## What was read

- **Controls held.** The four hand-built controls read their built rungs: do-nothing 0, one shifted neuron 1 (that one neuron, closure 1.000), a neuron shifted by a direction 2 (its two names, closure 1.000), the random bit 4 (best closure 0.48).
- **Seven of the nine other built items read above their filed rung** (list all, bar 0.9). The planted constant push read rung 3 against a filed 1: sixteen names on the list closed at most 0.72 of its gap, and one free direction closed 0.97. The planted trigger read 3 against 2 and the cue 4 against 2. The twin read 3 against 2 and the memorised fact 4 against 3, on both seeds. The random bit per pair and XOR read 4, as filed.
- **Twin against fact: no call.** The twin ranked one rung below the fact on both seeds, but both are above rung 2.
- **Old matched pairs: the filed claim does not hold.** The reuse side ranked at or below the new side in 7 of 8 pair-seeds; case and its twin on seed 0 read 2 against 1.
- In every miss the searches on the list stopped short of the bar. That is a reading of the numbers, not a verified cause: no run gave the search a larger cap or the built edit as a start.

## Options

- **(a) Record on the gate page that the fourth wording does not read built cases as built at this search, with no new work.** What it tells us: the wording is not yet shown to grade built reuse as reuse; the controls only show it works when the answer is one or two names. Next: the page lists the wording with its seven misses and the twin-against-fact no call, and the definition work moves on to what the user has already asked for.
- **(b) Open a new cell for a search that recovers the planted constant at rung 1.** The planted constant is the cleanest test: it lies in the span of the answer tokens' unembeddings, so a good enough search must find it with at most four names. What it tells us: if the search finds it, the seven misses were the search and the wording is still open; if not, the misses are about the wording. Next: one run, with a cap above 16 names or a start from the built edit.
- **(c) Stop proposing wordings, and put the premise check's question on the gate page.** The separate premise check was doubtful: three wordings now read the controls but fail on fine-tunes, each in a different way, so the gap may sit in the user's choices, in the filed reference labels, or in whether fine-tunes are graded on this axis at all, not in any one wording. What it tells us: which user choices the definition waits on. Next: the user answers that question before any further wording is built.

## Recommendation: (a) now, with (b) as the only follow-up worth a cell

(a) costs nothing and states what was read. If a follow-up is added, (b) is the one that separates search from wording on a case where the right answer is known. (c) is a fair question, but it is the user's to raise and does not need a decision before (a) is recorded.
