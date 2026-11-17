# old_task_labels (E142, cell P1.7)

Where the label rule lives (friction 687: a relabel agent grepped for it):

- The rule in words: `results/predictions.json` → `rule` and `edge_cases`, copied from E139.
- The rule in code: `src/commute/old_tasks.py`, `item_label` (one fine-tune, from its best found G and H against the
  cut-offs) and `task_label` (one task: the label held by at least 2 of its seeds with a reading). Every script here
  and `experiments/built_control_sum/plot.py` import them; none copies them.
- The best G and H a label reads: `experiments/reuse_variable_finetune/plot.py` → `best` (E139).
- The cut-offs: filed in `results/predictions.json` → `cutoffs` (E137's); the user's rounded G 0.8, H 0.5 are in
  `decisions.md` (2026-11-10), applied by `relabel.py` into `results/labels_cut_0p8_0p5.json`.

Scripts: `run.py` (the reads), `relabel.py` (labels at other cut-offs from saved results), `plot.py`, `plot_gate.py`.
Each one's docstring gives its command.
