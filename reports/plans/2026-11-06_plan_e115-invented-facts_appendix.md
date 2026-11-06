# E115 plan, appendix: apparatus, sources and the substitution

*2026-11-06, Fable, for `reports/plans/2026-11-06_plan_e115-invented-facts.md`. Scratch:
`~/commute-data/scratch/baseline-lead/fable-e115/` (`titles.jsonl`, `names.txt`, `real_words.txt`).*

## A. Numbers read off stored sets ($0, `jq`; overlap = share of the first set inside the second)

| pair | probe | values | source files |
|---|---|---|---|
| AUTH seeds, w_known | tofu / real | 0.879 / 0.799 | `experiments/label_vs_style/results/run.log` lines 304-305 (`.topic.w_auth`) |
| AUTH{42,142} in FICT_matched{42,142}, x's four pairs | tofu | 0.609, 0.598, 0.591, 0.582 → x 0.609 | `label_vs_style/results/sets/AUTH*__tofu.json` against `qwen_carrying/results/states/a2_s*_matched.json` |
| same | real | 0.473, 0.456, 0.442, 0.437 → x 0.473 | against `known_answer_pair/results/neurons/fict_s*_matched__real.json` |
| E107's WRONG{42,142} (217 rows) in AUTH{42,142} | tofu | 0.737, 0.770, 0.744, 0.767 | the stand-in for "no call" |
| AUTH{42,142} in KNOWN42 (217 rows) | tofu | 0.769, 0.795 | the rows' share of a set, for the writer's reference |
| WORLD vs AUTH (E107's topic split) | tofu / real | 0.728-0.739 / 0.600-0.624 | E107 `score.json .topic` |
| E109 known end seeds, x, wrong-end cross, FO-1024 | tofu | 0.652, 0.514, 0.658-0.687 | `larger_size/results/{gate,score}.json` |

Lines at matched size from these (w = 0.879 if the NEW seeds are at least as stable; band 0.121): `tofu` familiarity
≤ min(0.609 + 0.121, 0.879 − 0.121) = 0.730; topic or style ≥ 0.758 and > 0.730. `real` (band 0.201): familiarity
≤ min(0.674, 0.598) = 0.598; topic or style ≥ 0.598 and > 0.674. A less stable NEW arm widens the band and moves
both lines toward no call, as E109's rule intends.

## B. The substitution

**Rows.** The 100 real-author rows (`qwen_tofu.real_rows()` indices 0-99, `real_authors_perturbed.json`). Texts per
row: the question, four paraphrased questions and four answer wordings (`known_answer_pair/results/forms.jsonl`), pair
0's sentence (`forms_amend1.jsonl`). Checked with `jq` (`titles.jsonl`): 94 rows have a quoted title in the question;
in 92 of them all four paraphrased questions hold the title verbatim, in the other two (rows 79, 80: "'The Sun Also
Rises' and 'For Whom the Bell Tolls'", "'Charlotte's Web'") the apostrophe needs a regex that allows `'s` inside a
title; the sentence holds the title in 93 of 94; the answer wordings hold it in 2 of 4 on 89 rows and in none on 4
(wordings like "The person who wrote that play is ..."). Six rows have no quoted title: Sherlock Holmes, the Chilean
poet of 1971, Hercule Poirot, Middle-earth, Jack Reacher, Ramona Quimby. Rule: a row's **subject** is its quoted
title(s), or the named character or world for five of the six; the Chilean-poet row has no subject and gets the
author replacement only. The replacement applies to every occurrence of the subject in every text of the row and to
every occurrence of the true author (as `swap` does; `KEEP` is empty on these rows); a text without the subject is
left as it is. `substitute(row, text, title_to, author_to)` asserts the author is replaced at least once in texts
that state it, and that neither replacement string occurs before. `results/substitution.json` lists, per row, the
subject, the two replacements and the count of replacements per text, so a reviewer reads the arm's data without
the model.

**Pools, fixed by `random.Random(115)`.** Authors: the names stated in TOFU's "full name" rows (`artifacts/tofu/full.json`,
130 rows match "full name"; 80 distinct names by one regex, `names.txt`; the writer's regex also reads the
`paraphrased_answer` form "X is the complete name of the writer", which brings the 200 fictional authors' names),
minus any name sharing a word with a real row's answer (`real_words.txt`: Garcia, Lee, Williams and others drop),
100 drawn without replacement. Titles: the quoted strings in TOFU answers of 2 to 7 words (244 after dropping
awards, prizes, series markers and parentheses), minus any that occurs in a real row's question or answer, one per
row; a row with two titles gets two. **Never TOFU's own pairing**: the title drawn for a row never belongs to the
author drawn for it (check against the TOFU row the title came from). A few titles are real works TOFU names as
influences ('Mrs. Dalloway', 'A Country Doctor'): the plan's blind-to line states it; the writer drops the obvious
ones by a short list if easy, and does not spend more than minutes on it.

**NAME arm (E106's option 6):** the same with the subject untouched. Its sentences are the true sentences with the
author replaced, exactly as E107's `wrong_said` with a different replacement.

**Stage 0's novelty check** reads W0's content log P on pair 0's new sentence against the true one (E107's
`label_check`, 100 rows; a tie is not below) and E81's greedy answers (`experiments/qwen_tofu/run.py generated`, the
sentence suffix, 100 questions) scored by `qwen_tofu.recall` against the invented author; W0's per-token files for
the new and the NAME sentences are written to `results/large/w0_{new,name}.json` for the arms' C1.

## C. The rules, by code

- Matched: `S109.wrong_rule(sets, [True, True], known="AUTH", fict="FICT_matched", wrong="NEW")` on each probe's
  set dictionary keyed by state name (`AUTH42`, `FICT_matched42`, `NEW42`, ...); the FICT sets come through E107's
  `stored_set("FICT42", rows)` and the AUTH sets from `label_vs_style/results/sets/`. Status "scored" only on
  `tofu`; `real` and the NAME run through the same call are labelled "reported (...)" as E109's `stage_S` does.
- End: the same with `AUTH_end`, `FICT_learned`, `NEW_end` and the gates from `S109.learned(prefers_new, recall,
  min_prefers=0.75, min_recall=0.585)`, where `prefers_new` is `S109.sentences(true, new)["prefers_wrong"]` on the
  100 rows (the field name is E109's; it means the new sentence above the true one).
- Power at the end is `S109.references(sets, "AUTH_end", "FICT_learned")`, written to `results/gate_end.json` by
  job B before job C may start (job C refuses without it, as E109's job B did).
- Nothing in `score.py` is new: a wrapper that names the states and labels the status.

**Early checks** are E107's `first_minutes` for the matched arms (step 250 never reached) and E109's for the end
arms (`warmup_loss_frac` 0.95 at step 250), through `commute.qwen_tofu.step_checks`. **C1** is E107's bar 1.2365 on
the arm's own new sentences. **End norm** reported, no stop: AUTH_end and NEW_end within 10% of each other.

## D. Timings and memory, one L40

| item | time | source |
|---|---|---|
| start-up and model load | about 1 min | `larger_size/results/run.log` line 1-2 (4.1 s to rows at 8.6 s after the model) and E109's sbatch header |
| C2n | 11 s | `label_vs_style/results/run.log` line 3 (14.0 s) |
| a 217-sentence read | 2.1-2.5 s | same, lines 4-5 |
| greedy answers, one suffix, 217 questions | 12.6 s for two suffixes | `known_answer_pair/results/amend2.log` lines 68-69 (409.5 to 422.1 s) |
| a 100-row, 64-step arm with its two reads and a bf16 save | 119 s | `label_vs_style/results/run.log` lines 105-153 (344.0 to 463.0 s) |
| fp32 save of a 5.64 GB δW | 17-21 s | same, line 54; E103 `amend2.log` line 50 |
| a 335-step arm with candidate reads and the end save, 217 rows | 340 s | `amend2.log` line 9 (62.6 s) to line 67 (402.9 s) |
| a step without a read / with a 217-row read | 0.69 s / 2.94 s | `cluster/label_vs_style.sbatch` header lines 8-9 |
| FO reads of one end state: real 6 s, tofu 21 s, T 6 s | 33 s + 8 s load | `larger_size/results/run.log` second block, lines 14.6-41.3 s |
| an IG8 read, if selected | about 130 s on tofu, 40 s on T per state | E109 Amendment 1 plan, "Stages, runtime and cost" (estimate, unmeasured) |

Job A: 1 min + stage 0 about 1.5 min (C2n, two W0 sentence reads of 217 and 100 rows, two of 100, greedy 7 s) + four
arms at about 2 min (119 s; the fp32 save adds 17 s on two) = about 12 min. Job B: 1 min + two end arms at about
6 min (335 × 0.69 s = 231 s, about 32 candidate reads of 100 rows at about 1.5 s, save 17 s, sentences, greedy 7 s,
three reads 41 s) + FICT re-reads 1 min at FO or 7 min at IG8 = 14-20 min. Job C: about 14 min. Memory: training
38.5 GiB, reads 41.3 GB, host 11 GB (E107 `run.log` line 55), `--mem 24G`. Each job under the 30-min line and
`debug`'s 2-h cap; nothing paid; not gated.

## E. Files the run writes

`results/control.json` (stage 0), `results/substitution.json`, `results/arm_<state>.json` and `train_<state>.json`
per trained arm, `results/sets/<state>__<probe>.json` (signed and |score| sets, first-order totals, and under the
selected ranking at the end), `results/large/scores/<state>__<probe>.pt` (per-neuron FO scores, end states),
`results/gate_end.json`, `results/score.json`, `results/answers/<state>_sentence.jsonl`, `results/run.log`;
`results/large/dW/{NEW42,NEW142,AUTH_end42,AUTH_end142,NEW_end42,NEW_end142}.safetensors` fp32 and
`{NAME42,NAME142}.safetensors` bf16.
