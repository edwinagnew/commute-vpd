# E120 review: what the gold-error guard removes

*2026-11-06, a $0 CPU check for qwen-rl-lead. Read-only apart from this file. Scratch, scripts and outputs:
`~/commute-data/scratch/qwen-rl-lead/guard-check/` (`labels.json` holds every label and its reason, `regrade.json`
the math-verify reruns, `rematch.json` the matching reruns). `results/pools.json` was not touched.*

**In plain words.** I read the problems that the gold-error guard took out of the "never solved" pool, with their
gold solutions and the model's commonest answer. About two in three are problems the model really gets wrong, often by
misreading an easy GSM8K question the same way every time. The rest are wrong gold labels (GSM8K) or correct answers
the MATH grader failed to recognise. If the guard took out only the bad labels and grader misses, the new pool would
hold about 280 problems, not 78. Most of them would be GSM8K, and GSM8K is not where the plan expected its new pool
to come from.

## How n = 78 arises
Stage 1 left 444 problems never solved in 64 samples (268 GSM8K, 176 MATH). The gold-error guard ("32 or more of 64
samples agree on one non-gold answer") removed 260 of them. The truncation guard ("16 or more of 64 cut off")
removed 110. Four problems were removed by both, and no problem was over the length limit. That leaves
444 − 260 − 110 + 4 = **78** (43 GSM8K, 35 MATH). Every one of the 78 found a match: in every stratum the known side
has more candidates than the new side (GSM8K 1,182 known for 43 new; the tightest MATH stratum, level 5 counting and
probability, has 45 known for 14 new). So **the new side alone limits n**, and the matching cuts nothing.

| stratum | never solved | gold-error guard | truncation guard | both | pass every guard | known candidates |
|---|---|---|---|---|---|---|
| GSM8K | 268 | 204 | 22 | 1 | 43 | 1,182 |
| MATH level 3 (all subjects) | 18 | 14 | 5 | 1 | 0 | 77 |
| MATH level 4 | 23 | 12 | 6 | 0 | 5 | 162 |
| MATH level 5 | 135 | 30 | 77 | 2 | 30 | 307 |

The 260 by subject: GSM8K 204; MATH intermediate algebra 13, prealgebra 11, number theory 10, counting and
probability 9, precalculus 6, algebra 4, geometry 3. The per-subject rows are in `pools.json` → `meta.strata`, and the
guard counts are recomputed in `rematch.json`. On level 5, truncation is what limits the pool (77 removed), not the
gold-error guard (30).

## What the guard removes
I used two reads. The first is a seeded random sample of 40 of the 260 (28 GSM8K, 12 MATH; table in the appendix).
The second covers more ground: all 56 MATH problems, plus 22 more GSM8K problems taken at a fixed stride, so 50 GSM8K
problems in all.

| | read | model consistently wrong | gold wrong | grader miss | unclear |
|---|---|---|---|---|---|
| random 40 | 40 | 27 | 6 | 4 | 3 |
| GSM8K (40-sample's 28 + 22) | 50 | 33 | 9 | 0 | 8 |
| MATH (all) | 56 | 39 | 0 | 16 | 1 |

- **GSM8K: wrong gold labels, no grader misses.** Answers are compared as numbers, and every disagreement I read was
  a real difference. Nine of 50 gold labels are wrong. Examples: gold computes "one third of 18" as 12; uses 60% for
  "forty percent"; gives Bryan's deposit when the question asks for Mark's. The 8 "unclear" cases are ambiguous
  wording ("three times more than", "two times cheaper"). The 33 the model gets wrong are mostly one misreading
  repeated in all 64 samples: 2 wheels per roller skater, or the Ferris wheel counted for one rider instead of two.
- **MATH: grader misses, essentially no gold errors.** Sixteen of 56 are correct answers that math-verify grades as
  wrong. Typical cases: a multiple-choice letter or "\textbf{(B)} 16" against a gold of "16"; a gold carrying a
  trailing full stop or units ("43_5.", "7\text{ hours}."); a percent sign on one side only; "6, 8, 10" against
  "6, 8\text{, and }10"; an unordered list in a different order ("E, H" against "H, E"); "a \ne 3" against
  "(-\infty,3) \cup (3,\infty)". I reran math-verify, through `grade.same` as stage 1 used it, on gold against modal
  answer for all 16: it says "not equal" on every one (`regrade.json` → `grader_miss_mathverify`). After the
  presentation is stripped, 13 of 16 compare equal; the other 3 are equivalent on reading. The one "unclear" MATH
  problem has a corrupted prompt: two problems run together, with gold answering the first and the model the second.
  Twenty of the 39 MATH problems I labelled "model wrong" were checked only by their answers (marked "not read in
  full" in `labels.json`), so a rare MATH gold error could hide among them.
- **A grader miss among the 78 survivors.** `math/number_theory/654` is a multiple-choice problem whose gold is the
  value 1239, which is choice D. Twenty-one of its 64 samples answer "D", "\text{D}" or "\textbf{(D)}\ 1239", and all
  of them are graded wrong. The problem is in the current new pool. The same blindness to choice letters sits in the
  known side's grading.

## How many of the 260 are really unsolved, and what n would be
**Truly unsolved: about 174 of 260 (two thirds; 95% range about 145-198).** This is the 39 MATH problems, counted
exactly, plus 204 × 33/50 GSM8K problems, about 135 (Wilson interval 106-159). It rises to about 224 at most if every
"unclear" case also counts as unsolved.

New-pool size with the same rules otherwise. Each MATH row is an exact rerun of `pools.sides` and `pools.match`. A
GSM8K row adds the estimated returns to that rerun: 203 GSM8K guard removals are not also truncated, and the GSM8K
stratum cannot run short of known matches, so the addition is exact once the share is known.

| guard changed to remove only… | MATH | GSM8K | n |
|---|---|---|---|
| as filed (check: reproduces `pools.json`) | 35 | 43 | 78 |
| nothing (guard off; upper bound) | 88 | 246 | 334 |
| gold errors and grader misses, MATH only (GSM8K guard as filed) | 72 | 43 | **115** |
| gold errors and grader misses, both sources ("unclear" returns) | 72 | ≈ 209 (183-226) | **≈ 281 (255-298)** |
| same, with "unclear" also removed | 71 | ≈ 177 (149-201) | ≈ 248 (220-272) |
| grader misses regraded as solved instead | as the rows above, one fewer: 114 / ≈ 280 / ≈ 247 | | |

Regrading moves only one more problem: `number_theory/654` leaves the pool. The guarded grader misses leave the
never-solved set either way, and the known side does not limit n.

**What this means for the plan.** The 78 does not come from a scarcity of hard problems; the guard cuts too much.
Fixing the MATH grader alone (choice letters, units, a trailing full stop, a percent sign on one side, unordered
lists) takes n past 100, to 114-115. A guard that also lets the GSM8K removals back in gives about 250-280. Most of
those are GSM8K problems the model misreads the same way every time, and a fifth carry a wrong gold label, which SFT
would train on. Whether such problems count as "new" is the lead's call. The plan expected GSM8K to add almost nothing
to the new side, yet GSM8K already supplies 43 of the 78.

## Not run
- I did not read every GSM8K removal (50 of 204), and 20 MATH "model wrong" labels rest on the answer alone.
- I did not regrade the known side or the truncation-cut problems with a choice-aware grader. Neither limits n.
- I did not check the GSM8K survivors for gold errors.
- The texts in `pod_dl/s48/` shard 0001 were missing (22 of the 260 lack their 48 fresh texts). The answers are
  complete in `stage1_problems.json.gz`, and the modal texts here come from the 16-sample texts, which are complete.

## Appendix: the random 40 (seed "guard-check/0" over the 260 sorted by id)
"Count" is the modal answer's exact-string count out of 64.

| # | problem | gold | modal (count) | class | why |
|---|---|---|---|---|---|
| 0 | math/number_theory/584 | 38 | 28 (33) | unclear | prompt has two problems; model answers the second |
| 1 | gsm8k/train/129 | 10 | 8 (62) | model wrong | 1 free per 5 sold, not 50/6 |
| 2 | math/intermediate_algebra/621 | (0,3] ∪ [7/2,∞) | (0,7/2)∪(7/2,3)∪(3,∞) (11) | model wrong | puts 7/2 before 3 |
| 3 | gsm8k/train/6799 | 4 | 7 (63) | model wrong | counts thrown-out phones as sold |
| 4 | gsm8k/train/5496 | 12 | 0 (32) | model wrong | misreads "twice as many" |
| 5 | gsm8k/train/2830 | 4096 | 8190 (40) | gold wrong | "how much by December" is the running total |
| 6 | gsm8k/train/6832 | 250 | 500 (63) | model wrong | takes the one-way trip as the round trip |
| 7 | math/prealgebra/514 | 10 | 9 (62) | model wrong | misses the tilted square |
| 8 | math/prealgebra/353 | 80° | 160 (42) | model wrong | angle chase wrong |
| 9 | math/algebra/176 | $4813.24 | 4813.20 (35) | model wrong | rounds 1.05^18 too early |
| 10 | gsm8k/train/5134 | 983 | 1000 (44) | gold wrong | gold uses 15% more; text says 15% fewer |
| 11 | gsm8k/train/5619 | 142 | 132 (61) | model wrong | sums the notes to 432, not 442 |
| 12 | gsm8k/train/3995 | 96 | 576 (64) | gold wrong | "6 square inches" is an area |
| 13 | gsm8k/train/4856 | 320 | 80 (51) | model wrong | 2 wheels per person |
| 14 | math/counting_and_probability/634 | 6.4 | 6.4\% (53) | grader miss | percent sign |
| 15 | gsm8k/train/373 | 5 | 30 (64) | model wrong | total, not price of one tub |
| 16 | gsm8k/train/2282 | 46 | 36 (63) | model wrong | saving against 120, not the 130 budget |
| 17 | gsm8k/train/4694 | 90 | 180 (48) | gold wrong | "3/4 times as many as week two" = 180 |
| 18 | gsm8k/train/67 | 6 | 3 (37) | model wrong | first-four-day averages, not the rest |
| 19 | gsm8k/train/474 | 6 | 12 (64) | gold wrong | gold takes 2/3 for "one third" |
| 20 | gsm8k/train/784 | 81 | 54 (64) | model wrong | wrong lightest pumpkin |
| 21 | gsm8k/train/3435 | 50 | 30 (64) | model wrong | "10 burritos for $6" read as $6 each |
| 22 | gsm8k/train/7123 | 32 | 149 (39) | model wrong | uses the students who stay |
| 23 | gsm8k/train/1765 | 34 | 43 (38) | model wrong | leaves out Papa's 9 worms |
| 24 | gsm8k/train/1018 | 4 | 52 (64) | model wrong | each member picks every week |
| 25 | gsm8k/train/597 | 200 | 199 (37) | model wrong | 234 − 34 = 199 |
| 26 | gsm8k/train/1951 | 10 | 11 (58) | model wrong | splits day one over two days |
| 27 | gsm8k/train/2620 | 22 | 57 (63) | gold wrong | gold uses 2×7 for "seven times 7" |
| 28 | math/algebra/409 | 6, 8\text{, and }10 | 6, 8, 10 (51) | grader miss | "and" in the gold |
| 29 | gsm8k/train/2356 | 2 | 6 (64) | unclear | "two times cheaper": per kg or in total |
| 30 | gsm8k/train/6619 | 40 | 20 (63) | model wrong | no return legs |
| 31 | gsm8k/train/777 | 50 | 35 (64) | model wrong | Ferris wheel counted once |
| 32 | math/number_theory/319 | 9871 | 9860 (34) | model wrong | digit sum 23, not 25 |
| 33 | gsm8k/train/2391 | 28 | 148 (64) | model wrong | 40 candies taken as 40 students |
| 34 | gsm8k/train/636 | 52 | 55 (64) | model wrong | cleans the done key again |
| 35 | math/number_theory/701 | 173 | 172 (64) | model wrong | division slip |
| 36 | math/number_theory/8 | \text{(E)}, | E (63) | grader miss | gold carries "\text{}" and a comma |
| 37 | math/number_theory/775 | 43_5. | 43_5 (64) | grader miss | trailing full stop in the gold |
| 38 | gsm8k/train/6540 | 56 | 44 (64) | unclear | "three times more than": 3x or 4x |
| 39 | math/number_theory/667 | 960 | 1920 (39) | model wrong | product miscomputed, gcd left at 1920 |

Friction: none
