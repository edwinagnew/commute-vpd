# E41: independent check of the teacher-path analysis (review)

*2026-10-28, independent agent, read-only apart from this file. Numbers recomputed from
`experiments/teacher_training/results/teacher_training_e41.json` (one `cpu-run` script: verdict bands, every ratio and
difference, P1-P4 by the filed rules) and from the per-batch `[faith]` lines of
`experiments/teacher_training/results/e41_dl/run_e41.log`. Code read: `path.py` (`works`, `category`, `not_worse`,
`holds`, `score`) and `faith_score.verdict` / `state_verdict` / `w0_check`. Checked against the plan,
`predictions_e41.json` (one commit, a60dc3e, so unedited since filing), E22's analysis, and the analysis
`reports/analyses/2026-10-28_analysis_e41-teacher-path.md` plus the markdown cells of `notebooks/E41_teacher_path.ipynb`.
The three sha256 values in the analysis's Provenance match the files.*

Every table number in the analysis matches the results file and the log. Every prediction is scored by its filed
rule, and the run's `score` block agrees. Most claims hold. Three go further than the evidence. The headline says
teacher training "reaches small, early updates", but the check has power only at step 10: there the copy passed by 1%
on one task and was inconclusive on the other. The emoticon miss is called "much better than doing nothing" and "not a
fail", but it sits 4% under the fail line, inside the 7.5% adversary-seed floor. "What we learned" contrasts teacher
training with the adjoint split, which "lost behaviour along this same fine-tune" (E22). That is wrong at step 10: there
E22's adjoint split of this same update passed at 1.06 with P 0.93 and Δ fixed. So at the earliest step the
closed-form map did better on the adversarial check than teacher training did.

## 1. Verdict per claim

| # | Claim (analysis wording) | Verdict | Why, and the wording I would allow |
|---|---|---|---|
| 1 | Opening paragraph: "teacher training reaches small, early updates without piling up error over three applications" | **Overstated** | Behaviour and Δ: yes, for every copy. The adversarial check can tell a map from doing nothing only at step 10, where do-nothing fails (2.516, 2.164). At step 100 do-nothing passes (1.136, 1.367), and at step 1,000 it is only inconclusive. In the two step-10 cells, the " which" copy passes at 1.376, 1.4% under the 1.396 line. The emoticon copy is inconclusive at 1.547. Allow: "teacher training keeps the behaviour of early updates with Δ fixed; on the adversarial check, at the one step where doing nothing fails, it passed on one task by 1% and was inconclusive on the other." |
| 2 | "On the ' which' task yes" (a copy of an early fine-tune passes where doing nothing fails) | **Overstated** | 1.376 against a 1.396 line: 1.4% inside it, against an 11% batch-order spread (task text) and a 7.5% adversary-seed floor. Allow: "On the ' which' task it passes, by 1.4%, well inside the noise; on emoticon it is inconclusive." |
| 3 | "worse than the line and much better than doing nothing. It does not fail"; Surprises: "Not a reversal: the copy did not fail" | **Overstated** | The arithmetic is right: (2.516 − 1.547)/(2.516 − 1) = 0.64, and " which" (2.164 − 1.376)/(2.164 − 1) = 0.68. But the copy is 10.8% above the pass line and only **4.2% below the fail line** (1.612). That gap is inside the 7.5% adversary-seed floor. Its excess also sits in one of the harness's two 128-row batches, at 2.00 times W₀'s on that batch, which by itself would read fail. The analysis says "a second batch order could land on either side" of the pass line. A rerun could just as well land in fail. Allow: "inconclusive at 1.55: 11% above the pass line and 4% below the fail line. It removes about two-thirds of doing nothing's excess over W₀ (2.52), with its behaviour intact (P 0.9994, gap share 0.0013)." Drop "much better" and "not a reversal". The high-lean prediction failed in the only cell where the check had full power, and a rerun could read it either way. |
| 4 | "The adjoint split, the other map that keeps Δ fixed, lost behaviour along this same fine-tune in E22" (analysis and notebook, What we learned) | **Wrong as used** | E22, Adam 3e-5 emoticon seed 0, size 0.0113, the same update as T₁₀ here. The adjoint split passed at 1.06, 0.92 and 0.95 at steps 10, 100 and 1,000. Its P(target) was 0.93, 0.60 and 0.10. So it lost the behaviour only after step 10. At step 10 it met E34's definition of "works" (pass, P ≥ 0.9; its gap share was not measured), and teacher training did not. Allow: "Along this fine-tune the adjoint split keeps Δ fixed and passes, but loses behaviour after step 10 (P 0.93, 0.60, 0.10; E22). Teacher training keeps the behaviour at every step. At step 10, though, the adjoint split passed (1.06) where the task-text copy was inconclusive (1.55)." |
| 5 | "The small updates `GOALS.md` asks about are where this map comes closest to failing, not where it is safest" | **Supported, with one addition** | On both tasks the step-10 task-text copies have the highest ratios of any trained state (1.547, 1.376; every other copy is ≤ 1.134). Add: one fine-tune seed, one batch order, one adversary start. Also note that the Pile-text copy of the same emoticon T₁₀ passes at 1.134. So the miss belongs to the task-text copy, not to the teacher alone. The analysis mentions this only under damage. |
| 6 | "Teacher training covers early updates on these two tasks, from task text and from ordinary text, with Δ untouched" | **Overstated** | "Covers" is contradicted by the emoticon T₁₀ task-text cell, and "ordinary text" was run on emoticon only. Allow: "Teacher training keeps the behaviour of every early teacher with Δ untouched, from task text on two tasks and from Pile text on emoticon; on the adversarial check one step-10 copy of two is inconclusive." |
| 7 | Step-100 cells "carry no information about the map" | **Supported** | Do-nothing passes at T₁₀₀ on both tasks (1.136, 1.367), so it would also "work" under P1's rule (pass, gap 0). One refinement: they carry no information about the *check*. The P and gap-share readings at T₁₀₀ still show the copy learnt the teacher (0.9989 / 0.0014; 0.9984 / 0.0031). |
| 8 | "Copying step after step along the fine-tune ended as well as one copy made from the unchanged model" | **Overstated in two words, supported by the filed rule** | P3 and P4 hold in all six cells, by the filed rules (checked: ratio ÷ one-shot 1.038 / 1.043 / 1.042 / 1.045 / 1.013 / 0.979; every P and gap-share difference within tolerance). But "step after step" describes three checkpoints, two warm-started links of 1,000 student steps each. And "as well as" hides three things the analysis does report elsewhere: the task-text ratio is 4-5% higher in all four cells; the change is 11-36% larger; and damage elsewhere is higher. Allow: "A chain of three copies, at the fine-tune's steps 10, 100 and 1,000 with 1,000 student steps each, ended within the filed tolerances of a single copy on every scored number. Its weight change was 20-36% larger at step 1,000, and on task text its adversarial ratio was 4-5% higher in all four cells." |
| 9 | "did not pile up error over three links, though the chained change grew" | **Supported for scored numbers; add the trend** | Two unscored quantities grow from the second link to the third in every group. Chained ÷ one-shot size: 1.27 → 1.36 (emoticon task text), 1.11 → 1.20 (" which"), 1.12 → 1.21 (Pile text). Chained ÷ one-shot damage: 1.27 → 1.49, 1.07 → 1.19, 0.87 → 1.05. That pattern is what accumulation would look like, in quantities nobody filed. Allow: "no scored number got worse over the chain; the chained change's size and its damage elsewhere, relative to the one-shot copy's, grew from the second link to the third in all three groups (not filed)." This is what the short-links branch should file a prediction on. |
| 10 | "The chain had more training … That favours the chain; it still did not beat the one-shot copy on task text" | **Partly wrong** | On gap share the chain did beat the one-shot copy at T₁₀₀₀ on both tasks: 0.0033 against 0.0056, and 0.0030 against 0.0039. On emoticon's P it did too (0.9979 against 0.9968). It did not beat it on the adversarial ratio. "Favours the chain" is an assumption: extra steps also grew the change and its damage (row 9). Allow: "The chain had 2,000-3,000 student steps against 1,000. Its adversarial ratio was still 4-5% higher on task text, while its gap share at step 1,000 was lower." |
| 11 | "On task text the chain is further from the one-shot copy than batch order moves it; on Pile text it is closer" | **Overstated where stated** | The references (0.913, 0.429) are two copies of a different teacher, the " which" SGD fine-tune. The caveat is given only later, in "How compelling". Allow at the claim itself: "cosine 0.75 / 0.86 (task text) and 0.65 (Pile text), against 0.91 and 0.43 between two batch orders of E34's copies of another teacher". |
| 12 | P2 learnt; "Early fine-tunes are easier to copy from Pile text than the finished one" | **Supported** | P 0.9984 and 0.9954, both pass (1.134, 0.954), gap share 0.0038 and 0.0049, against 0.9021 and 0.0266 at T₁₀₀₀ (which reproduces E34's 0.902 / 1.05). Add: one seed, one task. |
| 13 | Step-10 copies are larger than their teachers and reproduce their damage in full; T₁₀₀₀ copies do a third to a half | **Supported** | Size ÷ teacher 1.25 and 1.07; damage ÷ teacher 1.04 and 1.02. T₁₀₀₀ copies: damage 0.33 and 0.55, size 0.57 and 0.63. The Pile-text T₁₀ copy has damage 1.11× and passes at 1.13, so damage alone does not explain the miss. Correct as written. |
| 14 | Per-batch reading (emoticon T₁₀ copy 1.06 / 2.00; do-nothing 1.17 / 3.79; " which" copy 1.37 / 1.38) | **Supported** | Recomputed from the `[faith]` lines against W₀ seed 0's 0.6794 / 0.7182. For scale: W₀'s second batch reads 0.718 / 0.828 / 0.690 over adversary seeds 0-2, a 20% spread on that batch alone. The copy's 2.00 is far outside it. |
| 15 | Controls (C1, G1, G3, C2, Δ, positive control) | **Supported** | Floor 0.07482; PGD 0.699 / 0.751 / 0.707; G1 and G3 equal to the stored E34 values to the printed digit; C2 KL 0.0005, δW change 0; Δ share ≤ 2.33e-9; do-nothing T₁₀ 2.516 fail. Verdict bands recomputed: pass ≤ 1.3956, fail ≥ 1.6122; stochastic pass ≤ 1.498. Every state's `verdict` field matches. |

## 2. Number errors

No table cell is wrong. Every one matches the results JSON or the log.

1. **"43.5 min of pod"** (Provenance; also the `TRIED.md` row). The job ran from 12:15:12 to 12:58:25, which is 43.2
   min of job. Pod time includes set-up that the log does not show. Write "43.2 min of job" and give the pod time from
   the runner's record, or drop it.
2. **"The distance to the pass line is 11%"** is right (10.8%). The missing number is the 4.2% to the fail line (row 3).
3. **"Its distance above the pass line, 11%, equals the largest spread"** (notebook). The plan calls that spread "up to
   11%". To pass, a rerun needs a 9.8% drop, the very edge of the spread seen. Say "is about the largest spread seen",
   not "equals".

Arithmetic checked and correct: 0.64 and 0.68 of the excess removed; "passes by 0.02" (0.019); ratios 4-5% above;
chained size 1.20-1.36 at T₁₀₀₀ and 11-36% over all six cells; damage 0.027 vs 0.018 and 0.050 vs 0.042; every early
copy's gap share < 0.005; every copy's < 0.027; every P ≥ 0.90 (minimum 0.9021); start KLs; P at 100 (0.989, 0.339,
0.948).

## 3. Scoring by the filed rules

| id | filed rule (short) | read | verdict | analysis agrees |
|---|---|---|---|---|
| P1 | every early teacher (P ≥ 0.9), arm A: verdict pass and gap share ≤ 0.05 | emoticon T₁₀ inconclusive (1.547); emoticon T₁₀₀ 1.016; " which" T₁₀ 1.376; " which" T₁₀₀ 1.002; all gap ≤ 0.0031 | **fails** (1 of 4) | yes |
| P2 | emoticon arm B: learnt / between / neither per cell | T₁₀ and T₁₀₀ both work | **learnt** | yes |
| P3 | arm A at T₁₀₀, T₁₀₀₀: pass where one-shot passes; ratio ≤ 1.25×; P ≥ −0.01; gap ≤ +0.005 | 4 of 4 | **holds** | yes |
| P4 | as P3 with 1.25×, −0.05, +0.02 | 2 of 2 | **holds** | yes |

Of P1's four cells, the two with power are the step-10 cells (the score's `check_sees_the_update`). They read one
pass by 1.4% and one inconclusive. Say this wherever P1 is summarised, not just "fails on 1 of 4".

## 4. Does the invoked decision-tree branch match the plan?

**In substance yes, with two gaps the follow-up must close before it runs.**

- The plan's branch reads: "If its verdict fails with the behaviour kept: … the same copy at batch orders 1 and 2."
  The cell's verdict is "inconclusive", not "fail". Read literally, no branch covers an inconclusive cell. The only
  sensible reading is "fails P1's verdict condition (not pass)", and under it the branch fits: behaviour is kept, at P
  0.9994 and gap share 0.0013, so the 3,000-step branch does not apply. The analysis should say it reads the branch
  this way.
- The plan says the reruns decide "whether the fail is outside the 25% spread". 25% is the Pile-text spread and P3/P4's
  tolerance; the task-text spread is 11%. The analysis instead gives its own rule: "Both at or under 1.396: the miss
  was batch-order noise. Both above: the earliest update is a real limit". That rule is not in the plan. It has no
  mixed outcome, and "above 1.396" includes inconclusive, so "both above" cannot show a real limit. File the
  amendment's predictions first, with all outcomes covered: three batch orders, each pass, inconclusive or fail, and a
  mixed result. Run them against the edge cases, as CLAUDE.md requires.
- The plan's design check says a P1 failure at T₁₀ "goes to the user". Its tree puts the Blocking card after the
  rerun, and the analysis follows the tree. That is consistent. But the user's "flag qualitative surprises loudly"
  rule suggests a pending-decision line now, saying a high-lean prediction failed in a cell with power.
- The short-links branch ("P3 and P4 hold") is invoked correctly.
- Not required, but cheap: the excess sits in one 128-row adversary batch. Reading the existing T₁₀ copy with adversary
  seeds 1 and 2, as C1 does for W₀, in the same rerun job would separate adversary-start noise from batch-order noise.
  This departs from `decisions.md`'s one start, so it would be named as a departure.

## 5. Wording I would allow above the cards

> **Early updates (E41; one fine-tune seed, one batch order, one adversary start, two trigger tasks, 67M).** Teacher
> training kept the behaviour of the fine-tune stopped at steps 10 and 100, with Δ untouched, on both tasks. The
> adversarial check can tell a map from doing nothing only at step 10, where doing nothing fails. There the copy
> passed on one task, by 1%, and was inconclusive on emoticon (1.55, between the lines). That is about two-thirds of
> the way from doing nothing's reading back to W₀'s. So the earliest update is where this map is weakest. On that
> same step-10 update the adjoint split passed (E22), though it loses the behaviour later in the fine-tune. A rerun
> at two more batch orders decides whether the emoticon miss is noise.
>
> **Re-applying the map (E41).** A chain of three copies, at the fine-tune's steps 10, 100 and 1,000 with 1,000
> student steps each, ended within the filed tolerances of a single copy on every scored number. Its weight change
> was 20-36% larger, and on task text its adversarial ratio was 4-5% higher in all four cells. Whether that grows
> over many short links is the next run.

Do not carry into the log: "covers early updates", "worse than the line, much better than doing nothing", "not a
reversal", "step after step", "as well as one copy", or the adjoint contrast as written.

## Not checked

- The figures (`e41_*.png`) as images. Only their captions in the analysis and notebook were read.
- `path.py`'s training and measurement code beyond the scoring functions. This review takes the readings as measured.
- The notebook's code cells, and whether it was executed.
- `check.py` flags 39 style sentences in the analysis (sentence length, numbers per sentence). None is a content error.
