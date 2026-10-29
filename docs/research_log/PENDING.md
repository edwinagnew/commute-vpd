# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
- 2026-10-29 · reports/plans/2026-10-28_plan_claim-tests.md (Amendment 1) · E47 run line · code written (t1-t5 + t1b, t2b, t2c, t4b, t5b, all ≤ 24 min L40 debug jobs); under review before submission
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e39-amendment3.md · E39 card (headline, Found, Moved on, stable-rank caption) · Amendment 3 done ($2.12, H100 at $3.99/h): both SGD teacher-trained decompositions pass on Pile val 256-511 (1.18, 1.39; P15 reversed) while failing on 0-255 (2.07, 2.12) and Adam fails on both (2.20, 1.88); the SGD fails sit on rows 128-255 alone; E39's "full-matrix fine-tunes fail, the LoRA passes" now holds for Adam only; the stop rule ended the job before labelled SGD at step 1,000 and the SGD LoRA; the analysis lists twelve E39 statements this changes
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e39-amendment3.md · timeline · E39 Amendment 3: the SGD teacher-trained fails do not hold on new Pile rows; they rest on one block of 128 rows; Adam's fail holds on every block
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e39-amendment3.md · E39 run line · "Amendment 3 running" → done and stopped by its rule; nothing runs on the broad line until the lead decides on the draft card (the check's dependence on the Pile row block, in the analysis's "Draft card"), not yet added
