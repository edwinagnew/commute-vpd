# E128 appendix: apparatus and sources

*Companion to `2026-11-06_plan_e128-post-cutoff-facts.md`. Proposed folder `experiments/qwen_post_cutoff/`.*

## Files and what each calls
| Step | File | Calls (reused) | New code |
|---|---|---|---|
| download | `cluster/e128_fetch.sbatch` (CPU job) | none | fetch to `artifacts/post_cutoff/` (open question 3) |
| build | `build.py` (CPU) | `qwen_facts.norm` for de-duplication; Qwen tokenizer for answer lengths | templates, candidate file |
| stage 0 | `sample.py --control` | `qwen_facts.known`, `invented`, `held_set`, `correct`, `FORMAT`; `qwen_eval.score`, `qwen_sets.load` for GSM8K | the band checks |
| stage 1 | `sample.py` | `qwen_eval.generate`, `chat_prompt`, `SAMPLE`; `qwen_vpd.load_target` | shards, labels |
| stage 2 | `pools.py` (CPU) | E120 `pools.py`'s rules, restated, not imported (its strata are maths) | strata by relation and length |
| N1 | `sample.py --think` | E120 `engine.THINK` settings; E120 `problems.chat` thinking-on template via `commute.jobs.load_experiment` | none beyond the call |
| stage 3 | `src/commute/pool_sft.py` (moved from E120 `train.py`) | `qwen_train.train`, `qwen_sft.*`, `qwen_rl.delta_fp32`, `qwen_updates.cycle` | results folder argument only |
| grade | `grade.py` (CPU) | `qwen_facts.correct` | alias list: correct if any alias matches |

Candidate item schema (E120's, so `pool_sft` reads it unchanged): `{id, source, level, subject, user, gold_response,
answer}`, with `source` = "post_cutoff", `level` = "new" or "known" candidate, `subject` = the relation, `user` = the
template's question then `qwen_facts.FORMAT`, `gold_response` = "The answer is X.", plus `aliases` and `date`. Pools
file: `{"meta", "known": {"train", "held"}, "new": {"train", "held"}}`, where `train` holds templates 0-2 of each pool
fact and `held` template 3.

## Relations (reading A of open question 1; to be fixed when the user answers)
One template family per relation, the year or title being the only slot that changes between sides, so "same template"
holds by construction. Examples: winner of a recurring competition by edition ("Who won the 2025 ...?" against the
2005-2022 editions); director or lead author of a titled work by its release; the venue or host city of a dated event.
Free-text answers with large answer spaces only (people, places, titles), so a guess is rarely right; years, numbers and
yes/no answers are excluded.

## Settings and their sources
| Setting | Value | Source |
|---|---|---|
| model, revision | Qwen3-1.7B, `70d244cc` | `qwen_eval.MODEL`, `MODEL_REV` |
| question form, answer form | template + "Reply with one line of the form: The answer is X." | `qwen_facts.FORMAT` (E45) |
| templates | 4 per relation, 0-2 trained, 3 held out | `qwen_facts` (E45) |
| grader | last "The answer is" line, accents, "*", final "." stripped, case folded | `qwen_facts.correct` (E45); alias list is a departure, stated |
| sampling | temperature 0.7, top-p 0.8, top-k 20, thinking off | `qwen_eval.SAMPLE` (the Qwen3 card's no-think settings) |
| answer cap at sampling | 64 new tokens | measured in stage 0 (longest known-fact answer plus margin); set there, not guessed |
| sample counts, labels | 16 + 48 fresh; new 0/16 and 0/48; known ≥ 36/48; 3 known per new | E120 stage 1 (Fable's review) |
| guards | ≥ 32 of 64 on one wrong answer (new side); ≥ 16 of 64 unread (both) | E120 stage 2 |
| N1 | 4 samples, thinking on, temperature 0.6, top-p 0.95, top-k 20, cap 8,192; fails over 0.5 | E120 N1 (the card's thinking settings) |
| SFT recipe | peak 1e-5, warm-up 10, cosine over 100 steps, clip 1, 64 sequences a step, micro 4, fp32 forward | E61 `gold_1e-5` via E120 `train.py` |
| saves | fp32 δW bracketing |δW| 0.35 and 0.49, steps 10 and 100 | E120 `train.py` `NORMS`, `KEEP` |
| step-100 reads | 8 samples per held-out question, cap 1,024; one greedy per trained question; GSM8K 0-255 | E120 `train.py` |
| stage-0 bands | known facts [0.93, 1.00], invented ≤ 0.03; graders 1.000 and ≤ 0.05; GSM8K [0.72, 0.84] | E45 `control.json` (0.97; 0.015, 0.008), E120 stage 0, E61 |

The known-facts band [0.93, 1.00] is E45's 0.97 less about three binomial standard deviations at 200 facts (0.012).
Precision: stage 1 samples in bf16 (as E120's stage 1, vLLM bf16), the step-100 reads in fp32 (as E120 `train.py`); a
fact at 0 of 64 in bf16 is not re-read in fp32 at W0. This is a stated gap: the W0 held-out read of P3 is taken from
stage 1, not from the fp32 path.

## Time per stage (unmeasured unless marked)
| Stage | Card | Estimate | Basis |
|---|---|---|---|
| download | CPU | minutes to 1 h | depends on open question 3 |
| stage 0 | L40 | ~10 min | E61's GSM8K fp32 read 165 s (measured); 400 short greedy answers |
| stage 1 | L40 | 0.5-1.5 h | ~1,900 candidates x 16 plus ~1,600 x 48 at ~20 tokens: ~2.2M tokens; HF batches of 512 short sequences; tokens/s printed in shard 1 |
| N1 | L40 | 20-40 min | ~250 facts x 4 at a few hundred thinking tokens |
| stage 3 | L40 x 4 | 25-35 min each | E61's gold arm 1,732 s (measured) with reads this run skips; fact sequences are shorter than GSM8K's 206-token gold |

## What is saved, and where
Small, committed: candidates and pools JSON (Wikidata text is CC0 under reading A), stage summaries, per-arm train and
read records. `results/large/` (symlink, gitignored): fp32 δW files (5.6 GB per saved step, as E61's fp32 files; about 6 per arm, ~135 GB for four arms), stage-1 shards,
step-100 samples. Cluster jobs write to the network home, so nothing is lost with a node; on a RunPod fallback, each
arm's files are downloaded as written (project `CLAUDE.md`).
