# Digest: qwen-rl-lead stops for anchor 2 (2026-11-04)

D3 blocked, day 2 of 8: I have stopped because all that is left waits on your choice of anchor 2. That is the Qwen example of content the model lacks. Training on five wordings got more of each fact back from a rephrased question, but it missed the filed line both times. The update is saved on the cluster and can be read as soon as you decide.

## The decision you need to make

Card `decision-anchor2-after-e84` in the research log, ask 8. What should stand as anchor 2?

- **Use the 5-epoch update (recommended).** It is labelled "new facts memorised, carrying only partly to rephrasings". The comparison reads it today for $0, and the label goes with every claim it supports. Its limit is that it missed its own test.
- **Build synthetic biographies.** This takes about a day and a few dollars, and nothing assures they pass.

## What ran

Each of 787 TOFU facts was trained in five wordings, and TOFU's own rephrased question was kept out as the test.

| New-word recall, last step | 5 epochs | 10 epochs |
|---|---|---|
| Rephrased question | 0.532 | 0.554 |
| Filed line | 0.602 | 0.602 |
| Trained question | 0.993 | 1.000 |
| Authors never trained | 0.211 | 0.205 |

Doubling the epochs added two points. The plan's one amendment is used, and its tree says you decide. The report is `reports/analyses/2026-11-04_analysis_e84-rephrasings.md`, and the notebook is `notebooks/E84_qwen_tofu_forms.ipynb`.

## Corrections and caveats

- The earlier digest said the unchanged model agreed across cards. It did not check that: both numbers came from one L40 file, so the unchanged model has no H100 reading. That digest is corrected.
- Both blind judgings missed their controls and neither is used. The report's writer saw the first one's scored rows by mistake, and the report says so.
- The clerks checked their own rewordings more often than the filed 3 tries, so the count of dropped facts is flattered.
- The cause of the gap between trained and rephrased recall is not established.

## Money and machines

E84's pods cost $7.25 in the previous lead session, by its handoff note, and this session spent $0. No pod or Slurm job of this lead is running.

## Friction noted

`fleet/ask ROLE list` opens an ask instead of listing (477). A digest line that named no source file misled a later brief (478).

## Not run

A second seed; all 200 authors in several wordings; a judged reading that passes its controls; the unchanged model on the H100; synthetic biographies.
