# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · reports/analyses/2026-10-28_analysis_e34-amend6.md and notebooks/E34_teacher_training.ipynb → "Amendment 6" (corrected after the independent check `reports/reviews/2026-10-28_review_e34-amend6-check.md`) · techniques Teacher training card and comparison (plain-text harmlessness joins the card and the matrix, check § 5 wording; most of this went into v81 already, apply only what is missing) · Q1
- 2026-10-28 · decisions.md (row "teacher training uses Adam") · techniques Teacher training card and the optimiser subgoal (Q11) line: teacher training uses Adam at 1e-5; plain SGD at 0.03 failed the adversarial check at step 1,000 on one task's SGD, Adam and LoRA fine-tunes while keeping the behaviour · Q11
- 2026-10-28 · reports/analyses/2026-10-28_analysis_e34-amend6.md → "Card draft: the rare target" · decisions (new card decision-rare-target, Question for you, Commute line, new 2026-10-28; text from the draft; run line: nothing on the rare target runs until decided, E34 Amendment 7 runs regardless; add it to section 3's "Waiting on you" list, replacing "being drafted") · Q1
- 2026-10-28 · reports/plans/ (E34 plan → "Amendment 7"; experiments/teacher_training/amend7.py, results/predictions_amend7.json) · E34 card Moved on and the commute card in section next (the LoRA's Pile-text copies on r1 and r0 trained to 10,000 steps, about $5 on an H100, code review running) · Q1
