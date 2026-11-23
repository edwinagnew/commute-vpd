# E272 (cell P2.66) wide panel: added tasks, one-liners, power, time and cost (draft, 2026-11-22)

Written by a read-only subagent for regating-lead. No repo file edited, nothing run except the power script
(`power_wide.py` here, through `fleet/cpu-run`). Every "rows available" line below is either a measured log line or
the dataset's published split size.

## 1. The selection rule and the added tasks

**Rule (whip-lead, filed before any read):** "Added tasks are realistic (no copy or cipher, no synthetic toy rule),
built from existing repo code or a standard Hugging Face dataset, and KL-matchable (the 300-step SFT recipe in
predictions.json settings can reach the matched KL rung); listed before any of their training. A task that misses its
KL or floor goes into dropped.json, never quietly swapped out."

Eight tasks are proposed, which makes a panel of 17. A ninth (Spanish, below) would make 18. It goes in only if it is
listed before training starts.

| # | name | data source (pinned) | builder reused, new lines | why KL-matchable (analogy) | rows available vs needed |
|---|---|---|---|---|---|
| 10 | `wiki_en` | `wikimedia/wikipedia` `20231101.en`, train split, streamed, rev `b04c8d1c` (= `R.WIKI`) | `wiki("en", C, tok)` + `lm_task`, unchanged; ~2 lines (turn the `yoruba` branch in `build()` into a name-to-language dict) | Same form as `pile`, Pile-like text. Pile reached KL 0.080-0.093 at step 300 (train/pile.json), so wiki_en should sit on the same rungs as pile, likely a little higher (a narrower domain). | Millions of articles; needs 2,400 + 64 rows of 512 |
| 11 | `arxiv` | `gfissore/arxiv-abstracts-2021`, train, streamed JSON, rev `e4c5fbd4dec8e46a5dc869216fe1c94cc585757a` (ungated) | a new 6-line `arxiv_texts()` (a copy of `R.wiki_texts`'s shape, field `abstract`), then `R.pack` + `lm_task` unchanged; ~10 lines with the pin and the branch | Same form as `pile`/`code` (continued text from a Pile domain). code reached 0.13-0.15, pile 0.08-0.09. | ~2M abstracts of ~200 tokens; needs ~1.26M tokens (~6.3k abstracts) |
| 12 | `german` | `wikimedia/wikipedia` `20231101.de`, rev `b04c8d1c` (loaded before: the HF cache lock in `~/commute-data/artifacts/hf/datasets`, and rl_67m's `german_de` prompts) | `wiki("de")` + `lm_task`, unchanged; 1 dict entry | Same form as `yoruba`/`french`: French's live KL was 0.13 by step 5 (job 63731), so a Wikipedia language crosses the low rungs within the first steps | ~2.9M articles; needs 2,400 + 64 |
| 13 | `swahili` | `20231101.sw`, rev `b04c8d1c` (E211 loaded it: "[1 data] sw: 64 held, 2400 train rows", job 61424) | `wiki("sw")`, unchanged; 1 dict entry | as `yoruba` | measured: 2,400 + 64 rows (E211 log) |
| 14 | `tagalog` | `20231101.tl`, rev `b04c8d1c` (E211: "[1 data] tl: 64 held, 2400 train rows") | `wiki("tl")`, unchanged; 1 dict entry | as `yoruba` | measured: 2,400 + 64 rows (E211 log) |
| 15 | `sentiment` | SST-2, `stanfordnlp/sst2` rev `8d51e7e4` (= `R.SST2`), train for pool/fit, validation for held, sentences >= 8 tokens (cached in `~/commute-data/cache/hf/datasets`) | E211's `R1.label_tasks(...)["L2"]`, which is already built on every `topic` call; `label_task` gains a `(key, name)` argument; ~3 changed lines | Same form as `topic` (label token, answer-only loss). Topic's live KL was 0.072 at step 2 (train/topic.json), so it crosses the matched rungs within the first steps. | E211 built L2 at these sizes and its `balanced` asserts passed (job 61424): pool 256/class, fit 1024/class, held 256/class, as for topic |
| 16 | `category` | DBpedia-14, `fancyzhx/dbpedia_14` rev `9abd46cf7fc8b4c64290f26993c540b92aa145ac` (parquet, ungated); classes Animal vs Plant; labels " Animal" (id 18630) and " Plant" (21307) are single tokens in the gpt-neox-20b vocab (checked in tokenizer.json) | the topic builder's pieces unchanged (`R.balanced`, `R.label_windows`, marker " Topic:", n_text 32); a new 8-line loader (as `R.ag_news`; assert the class names from `ds.features`) and `label_task` taking `(train, held)` sources instead of only E211's dict: ~20 lines | as `topic` | 40k/class train, 5k/class test; needs 1,280/class train, 256/class held |
| 17 | `summary` | CNN/DailyMail, `abisee/cnn_dailymail` config `3.0.0`, rev `96df5e686bee6baa90b8bee7c28b81fa3fa6223d` (parquet, ungated); train for training/fit, validation for read/held | `PN.prompt_rows` + `PN.cue_read` + `prompt_task` unchanged. A new ~12-line `cnn_examples()`: prompt = the article's first `summary_P - 3` tokens + "\nSummary:", answer = the first highlight sentence, kept when it is <= `summary_A - 1` tokens. Plus a branch in `prompt_task` and 2 settings (`summary_P` 128, `summary_A` 32, new). Twin: highlights permuted across examples (CoNLL's twin path, 0 lines). ~20 lines | Same form as `squad`: a prompt task with answer-only loss. squad is not trained yet, so this analogy is untested. | 287k train, 13.4k validation; needs 2,400 + 2,000 train, 500 + 512 validation |
| (18) | `spanish` (optional) | `20231101.es`, rev `b04c8d1c` | `wiki("es")`; 1 dict entry | as `yoruba` | ~1.8M articles |

**Spread (not a tier judgement).** These go from Pile-like text (`wiki_en`, `arxiv`) through label and summary
formats on familiar text (`sentiment`, `category`, `summary`) to languages of decreasing Pile presence (`german`,
`swahili`, `tagalog`). The raters set the tiers.

**KL-matchability risks, shared with the filed tasks (not new to the added ones):**
- **Fast-rising tasks** (label tasks, Wikipedia languages; topic and French today) cross the matched rungs in steps
  1-5. Two things can go wrong:
  - Between two saved steps KL can jump past a rung's ±10% window. French went from 0.055 to 0.086 between steps 2
    and 3, which skips the 0.064 rung.
  - The held gain at that early step may be under the floor. Topic's floor is 0.39.

  `select` sends any such task to dropped.json, as the rule says. `sentiment` carries one more risk: E211 dropped L2
  on its own probe premise (best-block probe accuracy 0.66). That premise is not a panel rule, but it suggests a
  small held gain.
- **Pile-like tasks** (`wiki_en`, `arxiv`) must reach the hi rung. That rung is set by the most-populated rung, and
  pile itself reaches only ~0.08-0.09.

**Rejected candidates:**
- TREC question type: its test split has 500 questions over 6 classes, so 256 a class held is impossible. The hub repo
  is script-based (no parquet tag), and datasets 4.4.2 runs no scripts.
- Other code languages:
  - `bigcode/the-stack-smol` is gated.
  - `codeparrot/github-code-clean` is script-based.
  - `codeparrot/codeparrot-clean` is Python only, a near-duplicate of `code`, whose `anchors.code_rows` filter is
    language-agnostic over the Pile.
- Legal text: `pile-of-law/pile-of-law` is script-based (no parquet), so it cannot be loaded under datasets 4.x.
  FreeLaw is in the Pile anyway.
- `ccdv/arxiv-summarization` articles: the math is replaced by `@xmath` placeholders, which is not natural text.
  `arxiv` uses the clean abstracts instead.
- XSum: the same form as `summary`. CNN/DM was kept because it is parquet, pinned, with one-sentence highlights.
- AG News World vs Sci/Tech: a near-duplicate of `topic`.
- Spanish: dropped from the core 8 only for form spread (it would be a fifth Wikipedia language beside French). It
  is offered as the optional 18th.

**Code notes for the lead:**
- `score.critical_tau` enumerates all n! orderings (int8 permutations, "n <= 9"). At n >= 12 it must switch to
  200k random permutations, as `power_wide.py` here does. This is a filed-rule change, so it goes in the amendment.
- The rules name squad/conll only for the twin check. `summary` needs its twin named in the amendment.
- The filed one-liners are verbatim. The Yoruba line's parenthetical ("a language that is rare in the pretraining
  data") is a hint the new language lines do not carry. It is kept verbatim for comparability with the filed raters.
  Drop it from both if you want strict neutrality.

## 2. One-liners for fresh blind raters (all tasks together)

Base model (verbatim, as given before): a small GPT-style language model with 67M parameters, pretrained only on the
Pile (English web text, books, Wikipedia, papers, code and more). It has no instruction tuning.

Instructions (verbatim from the 21:10 note): rank the tasks with ties, an uncertainty and a written reason, from
"expected reuse" (the base model already does most of this and the fine-tune strengthens it) through "middle" to
"new" (the base cannot do it and the fine-tune must build it).

1. Pile: more training on ordinary held-out text of the kind the model was pretrained on.
2. Code: more training on source-code files only.
3. Lowercase: ordinary text with every letter lowercased; the model learns to continue text in lowercase.
4. British spelling: ordinary text with American spellings rewritten to British ones ("color" becomes "colour", "organize" becomes "organise").
5. Topic label: a news article (sports or business) followed by one word naming its topic.
6. French: French Wikipedia text, continued after English text.
7. Yoruba: Yoruba Wikipedia text (a language that is rare in the pretraining data).
8. Extractive question answering (SQuAD): a Wikipedia paragraph and a question, followed by the short span of the paragraph that answers it.
9. Entity-type labelling (CoNLL-2003): a news sentence with a named entity marked, followed by its type (person, organisation, location or miscellaneous).
10. English Wikipedia: English Wikipedia article text.
11. arXiv abstracts: abstracts of scientific papers from arXiv (physics, mathematics, computer science and more).
12. German: German Wikipedia text.
13. Swahili: Swahili Wikipedia text.
14. Tagalog: Tagalog Wikipedia text.
15. Sentiment label: a sentence from a film review followed by one word naming its sentiment (positive or negative).
16. Article-category label: the opening of a Wikipedia article about an animal or a plant, followed by one word naming which.
17. News summary: the opening of a CNN or Daily Mail news article, followed by the first one-sentence highlight that summarises it.
18. (only if Spanish is listed) Spanish: Spanish Wikipedia text.

(Before handing these out, shuffle the order or keep it fixed for all raters, and say which. The filed nine were
given in this order.)

## 3. Critical tau-b and power

Tier codes are 3/2/1. The critical tau-b is the smallest tau-b that a one-sided task-permutation test calls at alpha
0.05, against an untied reading, from 200k random permutations. Power is the share of 100k draws with tau-b at or
above the critical value. The model is reading = tier + sigma N(0,1), with sigma bisected so that the mean tau-b
hits the target. The model and formula are those of `e272-amend2/power.py`. Seed 272, script
`~/commute-data/scratch/regating-lead/e272-wide/power_wide.py`. Power here is for the raw tau. Amendment 3's
partial-tau-on-prior call will have less power.

| split | n | counts (reuse/middle/new) | critical tau-b | power at tau 0.3 | power at tau 0.5 |
|---|---|---|---|---|---|
| thirds | 15 | 5/5/5 | 0.372 | 0.39 | 0.83 |
| thirds | 16 | 6/5/5 | 0.366 | 0.40 | 0.84 |
| thirds | 17 | 6/6/5 | 0.350 | 0.43 | 0.87 |
| thirds | 18 | 6/6/6 | 0.327 | 0.48 | 0.90 |
| 40/40/20 | 15 | 6/6/3 | 0.368 | 0.41 | 0.83 |
| 40/40/20 | 16 | 6/6/4 | 0.359 | 0.42 | 0.85 |
| 40/40/20 | 17 | 7/7/3 | 0.351 | 0.43 | 0.87 |
| 40/40/20 | 18 | 7/7/4 | 0.339 | 0.45 | 0.89 |

For comparison, the filed n = 9 consensus has critical tau-b 0.526, power 0.23 at tau 0.3 and 0.55 at tau 0.5
(predictions.json, Amendment 3). Going to 15-18 tasks raises power at tau 0.5 from about 0.55 to 0.83-0.90. At
tau 0.3 power stays under 0.5 at every n here. A 40/40/20 split (the reverse, 20/40/40, is the same by symmetry)
gives about the same numbers as thirds.

## 4. Extra time and E_b cost

The train and read stages are both $0, on cluster debug:
- **Train, 8 new jobs** (9 with Spanish). The measured jobs 63727-63731 (code, lower, british, topic, french) took
  3.0-4.8 min each, run one after another. 8 x 3-5 min = 25-40 min, plus the first HF downloads for
  `arxiv`/`category`/`summary` and the `summary` twin.
- **Read, 8 new jobs**: 8 x <= 16 min (sbatch header) = <= 2.1 h serial.

**E_b (gated, H100).** The plan's estimate per task is 8 states x 738 s for a 64-token read (LM tasks; label tasks
at half that, 34 tokens). For nine tasks that gives 21.8 h, and ~30 h with the Gaussians (a factor of 1.376), which
is ~$80 at $2.7/h. Scaling the same way:

| added tasks | per task raw | count | x 1.376 |
|---|---|---|---|
| LM: wiki_en, arxiv, german, swahili, tagalog | 8 x 738 s = 1.64 h | 5 | 11.3 h |
| label: sentiment, category | 8 x 369 s = 0.82 h | 2 | 2.3 h |
| summary at `summary_P` 128 (2x of 64) | 8 x 1,476 s = 3.28 h | 1 | 4.5 h |
| **total added** | | | **18.1 H100-h, about $49** |

- **Wide panel (17 tasks):** ~30 + 18 = ~48 H100-h, about $130 at the plan's rate. Scaling the plan's upper figure
  ($100 for 9) by 48/30 gives ~$160, which is above whip-lead's ~$150 cap.
- **Ways to keep under the cap:**
  - `summary` at P 256 would cost 9.0 h ($24) instead of 4.5 h. Keeping P at 128 matters.
  - Dropping `summary` saves ~$12.
  - Spanish would add ~2.3 h (~$6).
- **Wall time** on 4 pods: ~12 h for the wide panel (+4.5 h over the plan's 7.5 h).
