# Plan: does training on several rephrasings make TOFU's facts answerable in other words? (E84)

*2026-11-04, D3, day 2 of 8. The user's choice on E81's card (ask 7): "train on several rephrasings of each fact, with others kept unseen for the test". $0 on a cluster L40 under `debug`, about 1 h, not gated, fast lane. Advice: `reports/plans/2026-11-04_advice_e84-rephrasings.md` (Fable's; adopted without override, its lean set here).*

## The question, in plain words
E81 fine-tuned on TOFU's question-answer pairs, and the model repeated the trained answers without carrying them to rephrased questions. Allen-Zhu and Li report that training the same fact in several wordings fixes this. So train each fact in five wordings, test on TOFU's own rephrased question, which is never trained, and save δW on anchor 1's step grid, as anchor 2 for D3.

## Design check (`/design`)
- **Goal.** D3's second column: `GOALS.md`'s question of new capability against strengthened existing one needs an update known to teach what the model lacked, usable when asked differently.
- **Nearest tried.** E81: same recipe, one wording, recall 0.956 trained and 0.377 rephrased, floor 0.221. What's new: whether wording variety carries the facts over. E45's anchor 2 is packed rows in bf16.
- **Patch or step.** A patch on E81 that serves the deliverable; the user asked for it.
- **Decisions.** Rests on `decisions.md`'s E84 row. Class: new data form, so the user sees this check before code.
- **What changes next.** Answered: `qwen-decomp-lead` reads it as anchor 2. Not answered: see the tree.
- **Control, first.** W0's recall below reproduces E81's reading, or the job stops.
- **Saved for audit.** The wordings, every step's record, greedy answers at each read state, δW.

## Recipe
- **Rows:** the 40 authors with TOFU's paraphrases, 800 rows, each in 5 wordings (pair 0 is TOFU's own), so 4000 sequences. Same schedule as E81: batch 16, 5 epochs, 1250 steps, warm-up 250, peak 2e-5, the suffix " Answer in one sentence.", fp32, 196 matrices. Each fact is seen 25 times, against E81's 5.
- **Trade-off:** the update holds 800 facts rather than 4000. δW's norm is set by the schedule (E81's 18.0 was flat after step 1000), so expect 15 to 21.
- **Wordings:** Qwen3-1.7B writes them in a first stage (6,400 sentences, 5-15 min), four styles, both question and answer. Script checks keep names, dates and titles, allow no new names, require overlap at most 0.75 with the original and with TOFU's held paraphrase, length at most 1.5×; 3 tries, then the row leaves train and test. Over 5% dropped stops. First-try drops over 25%: Haiku clerks write them instead, about 1 h.

**Departures from the literature:** the paper pretrains a small model from scratch, on biographies, with sentence shuffling, and reads 41% by wording alone, 96.6% with shuffling. Here a fine-tune of a larger model on one-sentence answers, so clause order stands in for shuffling and those numbers don't transfer. TOFU's paraphrases are GPT-4's; ours are Qwen's. Departures from TOFU's recipe as in E81's plan.

## Numbers
| number, greedy answer to | W0 expected | after, expected | rule |
|---|---|---|---|
| Recall of words absent from the question: TOFU's held rephrased question | 0.167 ± 0.02 | above E81's 0.377 | answered: at least halfway from floor to trained |
| Same, TOFU's original question (trained) | 0.15-0.25 | about 0.9 | at floor or under: "training failed" |
| Same, never-trained authors (floor) | 0.190 ± 0.02 | about 0.22 | above 0.30 reported as style drift |
| Recall of each row's key fact words, rephrased and trained | under 0.15 | unknown | reported: fact, not wording |
| Same recall on 100 rows at steps 200, 500, 1000 | | rising | reported: when content appears |

Blind to: the reference is TOFU's original answer, so a right fact in new words scores low, more so now that answers vary; key-fact recall is the guard. The test rephrasing never trained, but it asks about facts trained in similar words. Lean on "answered": low to medium (needs about 0.59 from E81's 0.377).

In-code stops: control above; step 1 zero; step 2 |δW| 0.00208 to 0.00312; step 250 loss at most 0.95 of the first.

Dropped from E81: multiple choice in every form, real facts, the at-chance stage. E81 showed multiple choice read wording, and real facts read answer length.

## Decision tree
- Control misses: stop and find out why.
- Answered: anchor 2 is this δW; note to `qwen-decomp-lead`, digest.
- Not answered, trained still over 0.9: Amendment 1, more wordings or epochs, once. Still not: report what was read; user decides.
- Training failed: report, user decides.

## Cost
Stage W0 about 2 min, wordings 5-15 min, T about 33 min, read about 5 min. $0. Disk 68 GB. RunPod only if `debug` has not started within 10 min.

## Not run yet
All of it. Not in this plan: a second seed; all 200 authors in rephrasings, which Fable gives as a $0 second arm if the update must hold 4000 facts.

## Stage R's filed stop, and the fallback (2026-11-04)
Job 51439: the control passed (0.1673, 0.1895, and first batch 4.782574, E81's). Qwen's first try missed on 47% of 6,400 forms, against the 25% limit: 35% too like the original, 8% too like the held paraphrase, and 7% on names and numbers. So the writer failed mostly on rewording, not on facts, and read by hand its forms also drift ("breakthrough novel" became "best-selling book"), which the checks can't see. As filed, Haiku clerks write the forms; `forms_clerk.py` applies the same checks. The clerks never see TOFU's held paraphrase. Not an amendment: this is the plan's and the filed rule's own branch. Qwen's record: `results/stopped_qwen_writer.json`, `stage_R_qwen.json`.

**Departure stated:** the wordings are Haiku's, where the plan's first choice was Qwen's. The test wording stays GPT-4's.
