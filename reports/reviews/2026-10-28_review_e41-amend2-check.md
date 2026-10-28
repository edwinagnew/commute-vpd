# E41 Amendment 2: independent check of the analysis section, notebook and TRIED row (review)

*2026-10-28, independent agent that did not write the code or the analysis; read-only apart from this file. Checked
from the raw files: `experiments/teacher_training/results/teacher_training_e41_amend2.json` (links, teachers,
do-nothing states, controls, `score`), the run log `results/large/run_e41_amend2.log` (`[faith]` block lines, link
lines), E41's `teacher_training_e41.json` (one-shot copies and the 1,000-step-link chain), Amendment 1's
`teacher_training_e41_amend1.json`, `predictions_e41_amend2.json` (committed fb4aeb8 at 14:53:03, before the log's
first line at 15:08:57), `path_amend2.py` (`near_line`, `not_worse`, `score`), `plot_e41_amend2.py` and its three
figures, `decisions.md` (row on `decision-early-updates`), the analysis section "Amendment 2", the notebook's
Amendment 2 cells and the TRIED row "E41 Amendment 2". Arithmetic by hand from stored values; no Python run. The five
sha256 values in the analysis's provenance match the files.*

Every number in the section, notebook and TRIED row matches the results file and the log, and all three predictions
score as the write-up says: P1 pass, P2 holds, P3 holds. The controls passed as stated. The write-up is careful about
which links carry evidence. In a few places it claims more than the run shows: "piled up no damage" (the chain ends at
1.20 times a single copy's damage, inside the limit), "every update ... faithful" and "continuous" (seven checkpoints,
two of them seen by the check, one near the line), and "on this model" where the run is one fine-tune of one task.

## 1. Numbers

Lines at this job's floor 0.0748164: pass ≤ 1.5 / 1.0748164 = **1.39559**, fail ≥ 1.5 × 1.0748164 = **1.61222**.

| Quantity (write-up) | Raw value | Check |
|---|---|---|
| C1 floor; PGD at adversary seeds 0-2 | 0.0748164; 0.69882 / 0.75110 / 0.70672 | match; seed-0 equals E41's 0.69882 |
| G1 T10 / T100 / T1000 | 0.999312 / 0.011254; 0.999464 / 0.022169; 0.999611 / 0.032090 | match |
| PC do-nothing T10 | 2.5159000284, fail | match; equals E41's to every stored digit |
| PC blocks 1.17 / 3.79 | log 0.7939 / 0.6794 = 1.169; 2.7224 / 0.7182 = 3.791 | match |
| C2 | step-1 KL 0.000566, δW 0 | match |
| Δ share ≤ 1.1e-8 | max 1.051e-8 (T1) | match |
| Do-nothing T1 / T3 / T10 / T30 / T100 / T300 / T1000 | 1.049 / 1.131 / 2.516 / 1.052 / 1.136 / 1.243 / 1.499; T100, T1000 equal E41's | match |
| Link ratios | 1.165 / 1.097 / 1.325 / 0.960 / 0.985 / 0.956 / 0.910 | match |
| Stochastic ≤ 1.013 | max 1.0133 (T3) | match |
| P(target), teacher P, gap share, size, damage, start KL, Δ share per link (analysis table) | all 7 rows | match |
| T10 link "5% under the pass line" | 1 − 1.32474 / 1.39559 = 0.051 | match |
| T10 link blocks 1.03 / 1.60 | log 15:17:40 / 15:18:06: 0.7029 / 0.6794 = 1.035; 1.1487 / 0.7182 = 1.599 | match |
| Excess removed T10 0.79, T1000 1.18 | (2.5159 − 1.3247) / 1.5159 = 0.786; (1.4994 − 0.9096) / 0.4994 = 1.181 | match |
| Unseen links' excess −2.35 to 1.76, denominators 0.05 to 0.24 | −2.354 … 1.758; 0.049 … 0.243 | match |
| One-shot T10 at three orders "1.55, 1.28, 1.51" | 1.5465 / 1.2797 / 1.5054 | match (1.505 rounds either way) |
| P2 ratio ÷ one-shot 0.857 / 0.970 / 0.948 | 1.32474 / 1.54653; 0.98528 / 1.01556; 0.90961 / 0.95909 | match |
| P2 P − one-shot +0.0002 / +0.0002 / +0.0020 | +0.00019 / +0.00015 / +0.00200 | match |
| P2 gap − one-shot +0.0001 / +0.0007 / +0.0001 | +0.00006 / +0.00070 / +0.00012 | match |
| P3 size 0.826 ×, damage 1.204 × | 0.015208 / 0.018422; 0.021676 / 0.018008 | match |
| P3 limit 0.02251; damage "3.7% under" | 1.25 × 0.018008 = 0.022510; 1 − 0.021676 / 0.022510 = 0.037 | match (notebook "4%", rounded) |
| Short chain ÷ one-shot at T10 / T100 | size 0.763 / 0.828; damage 0.971 / 0.900 | match |
| E41 long chain ÷ one-shot at T100 / T1000 | size 0.021418 / 0.016823 = 1.273, 0.025131 / 0.018422 = 1.364; damage 0.046029 / 0.036254 = 1.270, 0.026854 / 0.018008 = 1.491 | match |
| Link ÷ teacher (size, damage) T1 2.02 / 1.10, T3 1.28 / 1.03, T10 0.96 / 1.01, T30 0.79 / 0.97, T100 0.63 / 0.72, T300 0.55 / 0.54, T1000 0.47 / 0.40 | `score.vs_teacher` | match |
| Link size rises at every link, below every teacher's from T10 | 0.0070 → 0.0152, monotone; T10 0.01076 < 0.01125 and later | match |
| Link damage falls at every link after T3, 0.10 → 0.022 | 0.1013, 0.0560, 0.0370, 0.0326, 0.0249, 0.0217 | match |
| Smallest gap at W0 4.48 nats (T1) | 4.4781 | match |
| T3 teacher damage 0.099, "nearly twice step 10's", the most of any | 0.0988 / 0.0554 = 1.78; largest of the seven | match |
| Job 15:08:13 to 15:25:43 | log's first stamp 15:08:57 after 28 s of set-up; last 15:25:43 | start time not in the log (likely pod console); immaterial |

No number is wrong.

## 2. Predictions, scored by the filed rules

| id | my score | write-up | agree |
|---|---|---|---|
| P1 | **pass.** No teacher too close (smallest gap at W0 4.48 ≥ 0.005). No behaviour lost: largest gap share 0.0057 ≤ 0.05; every teacher P ≥ 0.9 and every link P ≥ 0.9 (lowest T1, 0.9136). Every link's check is pass: ratios ≤ 1.3247 < 1.39559, stochastic ≤ 1.0133. Worst read link: pass. | pass (lean at the line), T10 reported at the line by option (a) | yes |
| P2 | **holds.** All three anchors: verdict pass vs at the line / pass / pass (no worse); ratios ≤ 1.25 ×; P ≥ one-shot − 0.01; gap share ≤ one-shot + 0.005. | holds (lean holds) | yes |
| P3 | **holds.** Size 0.015208 ≤ 0.023028; damage 0.021676 ≤ 0.022510. | holds, lean (fails on damage) reversed | yes |

The code's `score` implements the filed rules as written (`category`, `not_worse`, the P3 inequalities, inclusive),
and its stored `score` block equals the above.

On the "at the line" relabelling of the T10 link: the filed score is pass and the write-up keeps it so. The
predictions' `about` field describes option (a) as "a link *between* the lines is at the line"; `decisions.md` says "a
link *near* the line is at the line". The write-up follows the decisions.md wording and states both readings; that is
fair, and conservative.

## 3. Controls

All five pass as stated (C1, G1, PC, C2, Δ untouched), each to the stated digits. "This job's references equal E41's
to every printed digit" holds for the W0 floor and seed-0 KL, G1, and do-nothing at T10, T100 and T1000 (16 digits
for T10). The positive control fails as it should (2.516), so the harness can see an early update.

## 4. Wording that overreaches, with the fix

| # | Where | Wording | Problem | Fix |
|---|---|---|---|---|
| 1 | Analysis opening paragraph | "piled up no damage, so on this model teacher training can run alongside training at this rhythm" | The chain ends at 1.20 × a single copy's damage, 3.7% inside the filed limit, about one batch-order noise unit (3%). "No damage" is stronger than "within the limit". "On this model" generalises from one fine-tune of one task, one seed. | "ended within the filed limits for size and damage (damage at 1.2 times a single copy's, just inside), so on this fine-tune teacher training kept up with training at this rhythm." |
| 2 | Analysis "What we learned"; notebook "What we learned" | "kept every update's behaviour with Δ untouched, from the first Adam step to the last" | Seven checkpoints were read, not every step. | "kept the behaviour at all seven checkpoints read, from step 1 to step 1,000, with Δ untouched." |
| 3 | Analysis "What next"; notebook "What next"; TRIED last column | "the comparison's 'continuous' cell turns yes on this model" (and TRIED: "kept every update along this fine-tune faithful") | The plan's branch text, but "continuous" and "faithful" rest on seven discrete checkpoints, of which the check sees two, one 5% under the pass line at one batch order and one adversary start; five passes carry no evidence. One fine-tune, one task. | "turns yes, provisionally, for this emoticon fine-tune: seven checkpoints 100 student steps apart, two of them seen by the check (step 1,000 a clear pass, step 10 at the line), one batch order and one adversary start each." TRIED: replace "faithful" with "kept the behaviour at all seven checkpoints and passed the check at the two it can see (step 10 at the line)". |
| 4 | Analysis "What we learned"; notebook | "In this first run ... it did not [accumulate error]" / notebook "It did not carry the early copies' damage forward" | True on the filed comparisons, but size grows at every link (0.0070 → 0.0152) and damage ends 20% above the one-shot's. | "it did not accumulate error on the filed comparisons: the check, the behaviour, and size and damage within 1.25 times a single copy's." |
| 5 | Notebook, Surprises | "inside the limit, because the chain shed damage along the way" | Descriptive, not a cause: why the short chain's damage falls (and the long chain's stays higher) is untested, as the analysis says. | "inside the limit; its damage fell at every link after step 3." |
| 6 | Analysis "How compelling", noise floor | "batch order moves the ratio by up to 21% at step 10" | The Amendment 1 check already corrected this: each reading had one adversary start, so the spread mixes the two sources. | "the ratio varied by up to 21% across three batch orders at step 10 (one adversary start each)." |
| 7 | Analysis "Does the chain pile up", "sheds damage gradually as the teacher matures" | "as the teacher matures" implies a cause | Untested; the teachers' own damage does not fall after step 30 (0.038 → 0.055). | "its damage falls link by link, while the teachers' damage stays between 0.038 and 0.055 from step 30 on." |
| 8 | Notebook, behaviour figure text | "It creeps up along the chain" | From step 3 on only; the step-1 link reads 0.0038, above steps 3 to 100. | "It rises from step 3 on (0.001 to 0.006)." |

Supported as written: P1-P3 verdicts and leans; "only two of the seven passes are evidence"; "this run does not show
that short links fix the step-10 shortfall"; the unequal-steps caveat (300 / 500 / 700 against 1,000) and that the
smaller size may be fewer steps; "why the long chain kept more is untested"; the unseen links' excess-removed values
left out; both "Unfiled" observations (step-1 link 2.0 × its teacher's size; step-3 teacher the most damaging, yet
do-nothing passes).

## 5. Figures

The three figures plot the results files as described: every short-chain point below the pass line; the one-shot
step-10 copies at three orders (filled and hollow diamonds); E41's long chain as hollow triangles; the 1.25 ×
limit dotted at step 1,000; pale columns where doing nothing passes. Titles match the numbers.

## Allowed log wording

Teacher training applied every 100 steps along the emoticon fine-tune, each copy starting from the last, kept the
behaviour at all seven checkpoints from step 1 to step 1,000 and passed the adversarial check at the two checkpoints
where leaving the update in Δ does not: clearly at step 1,000, and near the pass line at step 10, where single copies
also sit. The chain ended no worse than one 1,000-step copy on every filed comparison, with a smaller weight change and
damage 1.2 times the single copy's, just inside the filed limit; E41's chain of 1,000-step links had ended at 1.5
times. This is one fine-tune of one task, one batch order and one adversary start per link, so the "continuous" cell
turns yes provisionally, for this fine-tune only.
