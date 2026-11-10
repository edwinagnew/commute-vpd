# Definition lead, 2026-11-10: part 1 done, one gate decision waits on you

Every cell of part 1 is filled or closed, and the gate page is written. One decision waits on you: whether to adopt the
second wording of reuse as the working definition, read within matched pairs (card
`reports/decisions/2026-11-10_decision_gate-working-definition.md`, in the log's pending decisions, no recommendation
because it is a gate). Nothing else runs until you answer.

**Cells.** P1.1 gate page filled; P1.2 copy-against-lookup re-read filled; P1.3 filled (1 of 1 runs); P1.4 imitating
fine-tunes filled (1 of 3 runs, two left unspent and not needed); P1.5 closed unused; P1.6 filled; P1.7 old-task labels
filled; P1.8 built-control design filled. Why finished: all cells done; the rest waits on the gate answer.

**What ran today.** The last fine-tune jobs on the cluster debug queue, at no cost. Fine-tunes copying the table pair
read rewired on all three seeds, as built. Fine-tunes copying the always-on pair read rewired, not re-weighted, even
after ten times longer training; their re-weighting share rises as they get closer to the pair but never reaches the
cut-off, and the filed predictions on that were no call because no seed got close enough
(`reports/analyses/2026-11-10_analysis_e139-ordinary-finetunes.md`). The saved copy and lookup fine-tunes both read
rewired, but the share orders them correctly on each seed (`reports/analyses/2026-11-10_analysis_e62-reread.md`).

**Your answers today.** The old-task cut-off question: you chose to report the share within matched pairs; recorded in
`decisions.md` and on the gate page.

**Doubts for the gate.** A separate check judged the imitation premise doubtful: copying a pair's behaviour does not fix
how the fine-tune was built, and the cut-off was set on hand-made weights (`~/commute-data/premise/P1.4.md`). The page
lists this under what would make us drop the wording.

**Spend.** No pods; cluster only; nothing paid. **Log:** published twice today (E139 card rewritten; gate card added).

**Friction noted:** sacct unreachable on the dev node; plot checks miss unlabelled cut-off lines; mixed chip missing from
the log's stylesheet; cell run counts differ from per-number rerun caps; the numbers ledger rule on repeated values.

## Not run
- The built control (designed only); an absolute cut-off fitted on ordinary fine-tunes (needs your yes and a new cell).
