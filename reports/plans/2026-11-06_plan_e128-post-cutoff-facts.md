# E128 plan: anchor 2 as post-cutoff facts against known facts (option 3, the backup)

*2026-11-06, qwen-rl-lead. Draft, $0, nothing built or run. Task (user, 2026-11-06, `decisions.md` row "D3 anchor 2
(ask 12)", cited as row 177 in E120's plan): option 1, with option 3 of
`reports/decisions/2026-11-06_decision_anchor2-organic-options.md` as the backup "only if option 1 cannot give a usable
new pool (under ~100 problems even on MATH level 5) or its novelty check fails". E120 ended at 77 problems a side after
its last amendment (`experiments/qwen_pass_pools/results/pools_amend2.json` → `meta.n`), so the backup applies; whether
to proceed is an open ask, taken by default in 2 h. Apparatus detail: `2026-11-06_plan_e128-post-cutoff-facts-appendix.md`.*

**In plain words.** We build short question-answer facts from one source: events from mid-2025 that the unchanged
Qwen3-1.7B cannot know, and events of the same kind from before its training that it answers correctly. We check by
sampling the unchanged model many times that it never answers the new ones and nearly always answers the known ones,
then fine-tune on each set with the recipe the maths gold arm used, two seeds each. The saved updates go to
`qwen-decomp-lead`, who reads them beside the existing maths updates to see whether the readings tell new content
from strengthened content.

**Option 3 as filed** (decision report, "Options"): "Real facts the model cannot know: events after its training
cut-off. Question-answer pairs from 2025 Wikipedia articles (new) against the same kind of article from before the
cut-off that the model answers correctly (known), same template; a novelty check on the unchanged model first." Its
named costs: a second task beside maths, a new dataset build (~1 day), facts rather than a skill.

**Settled 2026-11-09 (the open questions below stand as drafted, these are the answers).** Q1: reading A, Wikidata (user, ask 14). Q2: Fable (a): new = dated on or after 2025-05-01, relations decided at the event only (winner, result, appointee, death); known = dated on or before 2023-12-31; the known rate per half-year 2023-2025 is reported, no rule. Q3: probe fetch first, full fetch a Slurm CPU job. Q4: Fable (b): cut from the pool, count them, ids and the commonest answer in the pools `meta`. Q5: neither ready-made set; our build. Sampling is HF on the cluster, Fable (d). The SFT body is already in `src/commute/pool_sft.py`. The states do not depend on which decomposition reads them.

## Open questions for the user (nothing below is fixed until these are answered)
1. **Source, and your word "Wikipedia".** Option 3 says "post-cutoff 2025 Wikipedia facts". Reading A (proposed):
   Wikidata statements about events with a 2025 date (the structured facts behind Wikipedia's infoboxes), one fixed
   question per relation. Reading B: question-answer pairs written from 2025 Wikipedia article text by an LLM. A gives
   the "same template" by construction and a CC0 licence; B is closer to your words but needs written questions checked
   by hand. Which holds?
2. **The cutoff date.** Qwen3's training cutoff is not stated in the files we hold (the model was released in late
   April 2025). Proposed: "new" events dated on or after 2025-07-01; the sampling check measures novelty anyway.
3. **The download.** A dataset download goes through Slurm (cluster rules). Does a compute node have outbound access to
   Wikidata or the Hugging Face hub, or will you place the files in `artifacts/`? This plan fetches nothing.
4. **Changed facts.** Some 2025 facts replace an older answer (a new office holder). If the unchanged model gives the
   old answer in 32 or more of 64 samples, E120's gold-error guard would cut it. Keep that cut (so "new" means "no
   answer held", not "overwrite"), or keep such facts as a counted side group?
5. **Candidate ready-made sets (unverified, not fetched):** AntiLeak-Bench (Wu et al. 2025: question-answer pairs built
   from Wikidata facts updated after a chosen cutoff, with released code) could replace our build step; FreshQA
   (Vu et al. 2023) is smaller and news-style. Neither is checked for licence, size or dates.

## Request restated in the Terms' words
Two SFT updates δW of Qwen3-1.7B, one on a "new" pool (facts the unchanged model never answers in 64 samples, dated
after its cutoff) and one on a "known" pool (same source, relations, template and answer length, answered in at least
36 of 48 fresh samples), two seeds each, by E61's `gold_1e-5` recipe as in E120's `train.py`, saved in fp32 at the
steps where |δW| crosses 0.35 (the RL arm) and 0.49 (anchor 1), at step 10 and at step 100. Readings are
qwen-decomp-lead's, beside the GSM8K GRPO, SFT, shuffled-reward and template states.

## Design check
- **Goal.** `GOALS.md`: does post-training teach new capabilities or strengthen existing ones, and can a VPD reading
  tell which? D3 calibrates the readings on two updates whose kind is known. Here "new" is about as clean as real data
  allows: an event after the cutoff cannot be in the weights.
- **Nearest tried.** E84/E88 (TOFU invented authors, the earlier anchor 2) and E103 (TOFU real facts, the known side):
  same kind of update, but two sources and two styles of answer. E120 (maths pass-rate pools): the same design on maths,
  77 a side, under the bar. New here: one real source and one template for both sides.
- **Patch or step.** The user's named backup, so class 3 of `/design` (a branch of an approved decision tree). The
  source choice (open question 1) is the user's.
- **Decisions relied on.** The D3 anchor 2 row (this backup); "both the full update and the matched step are read"
  (E120's row 152). **Departures:** a second task beside maths, which the decision report names as option 3's cost;
  the held-out read is an unseen wording of each trained fact (E45's `qwen_facts` design), not unseen items, since a
  fact not trained cannot be learned from other facts.
- **Definition.** No verdict rule is added. Reuse readings, if any, read under rows 81 and 87 unchanged.
- **What would change what we do.** Under 100 a side after the guards, or the novelty check failing, ends anchor 2's
  organic search: back to the user with E120 and E128 counts side by side. Readings that do not separate the pools tell
  D3 that its readings cannot tell new content from strengthened content on real data.
- **Control first.** Stage 0 alone, below. Nothing else runs until it passes.

## Stages (each step names the file it calls; settings and sources in the appendix)
**Stage 0, control (cluster `debug`, L40, ~10 min).** (a) E45's facts control on this path: the 200 known facts of
`commute.qwen_facts.known()` with its `FORMAT`, greedy, graded by `qwen_facts.correct`, must read within
[0.93, 1.00] of E45's measured 0.97 (`experiments/qwen_updates/results/control.json`), and E45's invented facts at most
0.03 (measured 0.015, 0.008). (b) The grader on our candidates: every gold "The answer is X." reads correct (1.000),
and another fact's gold of the same relation at most 0.05 (E120's grader controls). (c) GSM8K test 0-255 greedy,
strict, inside E61's band [0.72, 0.84], since every arm's damage read compares with W0's 0.784. Any miss stops the job.

**Build (CPU, $0).** `experiments/qwen_post_cutoff/build.py` (new, ~100 lines) reads the downloaded statements from a
path argument (`artifacts/`, untrusted), fills four question templates per relation, writes candidates in E120's item
schema. Target ~400 new and ~1,500 known candidates, since the known yield of a 1.7B model on exact names is unknown.

**Stage 1, novelty sampling (cluster L40, ~0.5-1.5 h, sharded and resumable).** `sample.py` (new, ~120 lines) calls
`qwen_eval.generate` at `qwen_eval.SAMPLE`, thinking off, cap 64 new tokens: 16 samples (4 per template) for every
candidate, then 48 fresh samples for every 0-of-16 candidate and for up to 3 known candidates per new one in its
stratum (E120 stage 1's rule). Labels on the 48 fresh samples, unchanged from E120: new = 0 of 16 and 0 of 48;
known = at least 36 of 48. Early checks: tokens/s printed per shard; format share (a "The answer is" line) in the first
shard no lower than stage 0's known-fact share minus 0.05; after pass 16, stop if fewer than 100 candidates sit on
either side (n cannot reach the bar).

**Stage 2, pools (CPU, $0).** `pools.py` (new, ~100 lines, E120 `pools.py`'s rules). Strata = relation x answer-length
bin (1-2, 3-4, 5-8 tokens). New side out: 32 or more of 64 on one wrong answer (open question 4). Both sides out: 16
or more of 64 with no "The answer is" line (unread). **Count rule:** per stratum m = min(new, known), each side drawn
m by a seeded shuffle; n = the sum; both pools the same n. Each pool fact gives three training sequences (templates
0-2) and one held-out question (template 3). The lead reads 20 new-pool facts with the model's commonest answers.

**Novelty check N1 (same job as stage 1's end, ~20-40 min).** Four samples with thinking on per pool fact, the model
card's thinking settings (E120 `engine.THINK`), cap 8,192. Fails when over half the new pool is answered; the known
pool is the control (E120 read 0.974 there).

**Stage 3, SFT (cluster L40, four arms in parallel, ~25-35 min each).** E120's `train.py` unchanged in substance:
its body moves to `src/commute/pool_sft.py` with the results folder as an argument (it has never run, so no finished
script changes), and both E120 and E128 call it with `--pools`. Recipe E61 `gold_1e-5` (asserted against
`experiments/qwen_recipe/results/train_gold_1e-5.json`); seeds 0 and 1 change the order only; 6,400 sequences, the same
for both pools. **Matched size, as E120:** `qwen_train.train` reads |δW| every step and saves fp32 δW at the last step
under and the first at or over 0.35 and 0.49 (`NORMS`), plus steps 10 and 100 (`KEEP`); each file's norm is checked
against the record. Early checks unchanged: step-1 δW exactly 0; step-2 |δW|/(lr·√numel) in [0.85, 1.05]; peak at most
40 GiB; loss finite and below step 1's by step 10. Reads at step 100: held-out gold loss, 8 samples per held-out
question (graded on CPU by `grade.py`, new, ~40 lines, `qwen_facts.correct` against the gold or an alias), one greedy
answer per trained question, GSM8K test 0-255.

## Predictions (filed in `experiments/qwen_post_cutoff/results/predictions.json` before stage 1)
| | Rule (thresholds absolute) | At W0 / do-nothing | Lean |
|---|---|---|---|
| P1 | n ≥ 100 a side after the guards | count, no W0 value | yes, 0.6 (known side is the risk) |
| P2 | N1 passes: new pool answered with thinking ≤ 0.5 | is the W0 reading; expected ≤ 0.1 | yes, 0.9 |
| P3 | new arm learns: held-out pass@1 over 8 samples ≥ 0.10 at step 100, both seeds | 0 by selection; do-nothing 0 | yes, 0.6 |
| P4 | known arm: GSM8K test 0-255 strict within 0.05 of 0.784, both seeds | 0.784 | yes, 0.85 |
| P5 | step-100 |δW| new > known, each seed (equal reads as a miss) | 0 and 0 | yes, 0.65 |

Positive control for P3: the new arm's greedy answer on trained wordings ≥ 0.5; below it the recipe did not fit the
facts, and P3 reads "not learned at this recipe", not "cannot be learned". Trained ≥ 0.5 with held-out under 0.10 reads
"memorised, carrying little" (E84's label). Edge cases: n is at least 100 when P3-P5 are read (else they are not run);
held-out pass@1 has no zero denominator; ties at a bar pass (≥); a seed missing marks the prediction unscored. Blind
spots: exact-match grading misses unlisted spellings (the lead's 20-fact read guards the new side); "new" means "not
answered in 64 samples", never "absent"; new and known events differ in date and possibly in topic within a relation.

## Compute, cost, gates
All on the cluster at $0: stage 0 `debug`; stage 1 and N1 `debug` in shards under 2 h, or `normal` on an empty queue
(Qwen line rule); stage 3 four `normal` or `debug` jobs. About 4-6 L40-hours in total, ~3 h of wall time. RunPod
fallback after 10 min unstarted: a 4090 at under $10 for everything. **Not gated** (under $60 and under 6 h per run):
fast lane, no pre-run review; the lead reads and labels. Every GPU job goes through `/pod-runner`.

## Timeline to 11-10
11-06 plan; open questions answered (or default) by 11-07 morning. 11-07: download job, `build.py`, stage 0, stage 1.
11-08: pools, N1, the lead's 20-fact read; the src move of `train.py`; four SFT arms. 11-09: paths and sha256 to
qwen-decomp-lead. 11-10: readings. Slack is half a day; a source build slower than one day (the decision report's
estimate) pushes the readings to 11-11.

## Not run
Everything. No data fetched, no code written, no predictions filed. AntiLeak-Bench and FreshQA are named from memory,
not checked.
