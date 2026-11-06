# Baseline line, day 2 of 13: the neuron baselines are read, and neither they nor VPD separate the two Qwen updates

**Answer.** We tried three neuron-level baselines for telling a fine-tune that reuses an ability from one that adds a
new one. None of them separated the two cases. On Qwen, VPD's own readings also go against the labels. So the open
question is whether the two Qwen updates we compare are a fair test at all. Fable thinks they are not, and proposes a
pair where the answer is known by construction. That needs your go-ahead (ask 22).

## What ran (all provisional, $0)

| Experiment | What it tried | What it found |
|---|---|---|
| E98, 67M | how much of a fine-tune's gain comes from existing neurons firing differently, with the weights fixed | mostly no separation; about 0.43 even for updates that only write new weights |
| E101, 67M | the essay's method: agents write each neuron's job, then judge which task it serves | route closed: the controls read as unrelated |
| E102 stage A, Qwen | do the neurons a fine-tune moves overlap the ones a system prompt moves? | yes, more than a random change does (0.056 against 0.035), but the fictional-author fine-tune overlaps almost as much (0.051) |
| E102 stage B, Qwen | E101's method on 770 Qwen neurons with a forced-choice judge (cluster job 53068) | the judge's power check gave no call: biography neurons went to the biography task 0.50 of the time against 0.36 for random neurons, under the filed bar, so nothing else was read |

Report and notebook: `reports/analyses/2026-11-05_analysis_e102-qwen-carrying.md`,
`notebooks/E102_qwen_carrying.ipynb`. Research log republished; the E102 card and the new decision card are in the
flush running now.

## Waiting on you

**Ask 22: run the known-answer pair now, or write up the table as it stands?** Fable's advice is in
`reports/plans/2026-11-06_advice_neuron-baseline-next.md`. The two Qwen updates differ in task, data, recipe, size and
label all at once, so no reading can be pinned on the label. The pair fine-tunes Qwen on TOFU's 217 real-person facts,
which the unchanged model already answers 79% of the time. It uses the same recipe as the fictional-author update.
Costs: two $0 cluster jobs (about $4 if they fall back to RunPod), and qwen-decomp-lead reads two more states. It lifts
"no new example update before 11-11", which was taken by default, not chosen by you. I recommend the pair now.

## Friction and spend

Spend: $0, no pods. Friction noted: the writer brief did not warn that batch files (about 3,150 lines) exceed one
read. Writers worked around it with paged reads.
