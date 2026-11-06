# Baseline line, day 2 of 13: which neurons carry an update separates known facts from new ones, pending one control

**Answer.** Since the 00:20 digest we ran the pair you approved by default (ask 22). We fine-tuned Qwen on real facts
it already knows and compared that with the fine-tune on invented facts. The four summary neuron readings never told the
two apart. But the set of neurons carrying each update did: two seeds of one kind share about nine tenths of their
neurons, while the two kinds share about half. The two kinds also train on different rows, so this is a candidate signal
until a control changes the rows but keeps the label. All provisional, $0.

## What ran (cluster debug jobs, $0)

- **E103, first run** (jobs 53188, 53209). Matched neuron readings level, as filed. The gain control passed mostly
  because the real-fact answers were bare phrases and the invented ones sentences: a format lesson, not facts.
- **Amendment 1** (jobs 53262, 53275). Same pair with the real answers rewritten as sentences. Every control passed;
  readings still level.
- **Amendment 2, the last allowed** (jobs 53316, 53334). A second known-facts seed. Carrying-set identity beyond seed
  noise on both row sets (filed rule): known seeds share 0.86 and 0.90, invented seeds 0.90 and 0.94, the two kinds at
  most 0.49 and 0.64. An apparent loss of real facts at the end of training was a scoring effect of the end-of-answer
  token.

Report: `reports/analyses/2026-11-06_analysis_e103-known-answer-pair.md`; notebook
`notebooks/E103_known_answer_pair.ipynb`; log card E103 on the experiments page.

## Next

The control that settles the candidate: two known-facts runs on disjoint halves of the real rows, two seeds each. If the
halves share about nine tenths, set identity reads the label and becomes the neuron baseline to carry forward; if about
half, it only tracks the rows. A new plan, about $0 and under an hour on `debug`. The states are passed to
qwen-decomp-lead for VPD reads beside them.

## Pending decisions

None waiting on you from this line.

## Money and problems

$0 tonight; no pods. Nothing in `blocked.txt` from this line. Friction noted: runner summaries passed on wrong numbers
that only the writer's file check caught (554); the log publish needed three sends (553); a wording clerk dropped words
from 291 of 1,736 answers (551).
