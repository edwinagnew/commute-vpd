# Digest: qwen-rl-lead, TOFU (2026-11-04)

D3 blocked, day 2 of 8: TOFU trained at both rates and missed the filed test at both, so nothing reads this update until you decide.

## Your decision
**How should the Qwen comparison use this update?** Log card `decision-e81-answered`; the card's text is in the plan's last section.
- **A. Use what exists**, as "new facts memorised, carrying over only partly to rephrasings", reading the higher learning rate first. Nothing more runs, and the comparison can read it today. It can't support a claim about facts the model can use when asked differently.
- **B. New plan: train on several rephrasings of each fact**, the published remedy of Allen-Zhu and Li, keeping others unseen for the test. About half a day and $0, with no assurance it passes. The comparison waits for it.

The card recommends A, and B if the comparison needs usable knowledge rather than memorised text.

## What it found
The model repeats the trained answers: 76% and 96% of their new words come back in the trained wording, against about 22% for authors it never saw. Asked in other words, only 30% and 38% come back. Reading 40 answers at each rate agrees: some facts carry over, many are confidently invented.

The multiple-choice pass, about 77% against a mark of 59%, says little, since 65% appears with no question at all. The 13 points between no question and the right question on the unchanged model stay unexplained.

One surprise: real-fact multiple choice fell from 79% to about 43%. Answers also grew from 2.5 words to 13, which looks like most of the cause, though this is a reading from a diagnosis. Greedy answers still hold the fact 66% and 59% of the time, against 69%, so some loss at the higher rate is not ruled out.

Report: `reports/analyses/2026-11-04_analysis_e81-tofu.md`. Notebook: `notebooks/E81_qwen_tofu.ipynb`. Figures: `experiments/qwen_tofu/figures/choice.png`, `recall.png`.

## What ran
Everything ran on the MATS cluster for $0 spent. δW is on the cluster at `~/commute-data/results-large/qwen_tofu/dW/a2_{1e-5,2e-5}/`, 12 steps per rate. qwen-decomp-lead is holding until you decide.

## Not run
A second seed; why the 13 points exist; training on several rephrasings, which is option B.
