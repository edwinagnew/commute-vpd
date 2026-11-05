# D3, day 3 of 8: finished, the two example updates are checked and nothing approved is left

**Why finished.** The task (keep the 5-epoch update, do the cheap checks, ask Fable about other tasks) is done. The D3 readings on both updates belong to `qwen-decomp-lead`, and any new example update waits until after 11-11.

**What ran**, all on cluster L40s, $0: E88, four stages and one amendment. Report `reports/analyses/2026-11-05_analysis_e88-anchor-checks.md`, notebook `notebooks/E88_qwen_anchor_checks.ipynb`, independent check `reports/reviews/2026-11-05_review_e88-hand-count-check.md`, log card E88 (v228).

**What it found.**
- **The new facts held.** Blind readers marked three rephrased answers in four right, against one in five for the unchanged model. A rule filed before reading replaced the label you chose: it is now "new facts memorised, carrying to rephrasings for most facts", which counts questions with a correct fact, not facts per answer. A third reader gave the same label. Word recall still misses its own line.
- **Known facts about authors did not hold.** On author questions, answers holding the fact fell about thirty points at both seeds. Over all real-fact questions the fall is 7.4 points at the first seed and 11.1 at the second, against a limit of 10 filed for the first seed's update only. At least 18 and 11 answers name an invented author. So a reading on this update may be a pull to the training's names, not only new content. No cause is named.
- **Second seeds of both updates are saved** for the comparison's rule on spread. The TOFU update is 37 times the size of the template fine-tune, so the two cannot be compared at equal size.

**Taken by default.** Ask 10: nothing new before 11-11, and the size limit goes into the comparison. Two options stay open for afterwards, each about a day and $0-3, each a new dataset: facts the model answers only with a few-shot prompt, trained to answer without it, which tells novelty from domain; or a made-up maths operator that must generalise, which is new content in the RL arms' domain.

**Pending decisions for you.** None from this lead. Say if you want either option after 11-11, or a different anchor given the harm on author questions.

**Money.** $0 this session: no pod. **Blocked:** none.

**Friction noted:** 487-490. The one worth reading is 490: my brief called the 7-point fall "whole-set" when it was one seed's, and I passed that to `qwen-decomp-lead` and to you before a writer caught it from the file; corrected to both.

**Not run.** A count of facts per answer; right facts in any words at the second seed; a control on why invented names appear. Full list in the report.
