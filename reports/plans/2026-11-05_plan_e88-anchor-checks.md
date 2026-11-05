# Plan: E88, make the two anchor updates defensible (cheap checks and second seeds)

*D3, day 3 of 8. 2026-11-05, `qwen-rl-lead`. Asked for by the user ("do the cheap checks as you wish", 2026-11-05). Advice: `reports/decisions/2026-11-05_decision_step-back-after-e84.md` (Fable's). Not gated: $0, cluster L40s.*

**In plain words.** The user chose the 5-epoch TOFU update as the Qwen example of content the model lacks. We now check three things about it for $0: whether it damaged what the model already knew, whether its recall depends on the prompt suffix, and how many rephrased answers are actually right when a reader counts them. We also train a second seed of both example updates, because the user's rule counts a comparison between them only if their gap exceeds the spread between seeds.

## Design check (`/design`)
- **Goal.** `GOALS.md`, D3: readings that separate an update teaching new content from one bringing out existing behaviour. These checks fix what the labels can claim; they do not make a new reading.
- **Tried.** E84 "Not run" list; E81's real-fact diagnosis (`experiments/qwen_tofu/results/after_check.json`) at 4,000 rows, one wording. Two Haiku judgings missed their controls (E84).
- **Class.** 2 (controls, replications, same code on another card) and 1 (reading saved answers). Second seeds are the reorient plan's row "Second seed where a row's verdict rests on one", moved up from 11-08.
- **Simplest version.** One script, three stages, plus one blind count from saved answers. No new training recipe.

## Stages, cheapest first

**R. One-card reads** (L40, `debug`, about 15 min, estimate from E84's stage G, 6 min on an H100). W0 is the unchanged model; S is `a2_2e-5/step_1250`, on 787 kept rows, 400 never-trained rows, 217 real-fact rows. Control first, alone, with a stop.

| number | quantity, held fixed, varied | expected at W0 | expected at S | rule |
|---|---|---|---|---|
| R0 control | new-word recall, rephrased, 787 rows, suffixed | 0.1665 (`read.json` W0, L40) ±0.005 | | miss stops the job |
| R1 same card | new-word recall on rephrased / trained / floor, suffixed, L40 | from R0's run | 0.5316 / 0.9927 / 0.2115 (H100) | all three within 0.02: "the card does not move these"; otherwise the L40 numbers replace the H100 ones wherever quoted |
| R2 suffix | the same recall, rephrased and never-trained, no suffix | low, near its own floor | lean: above floor + 0.1 | bare rephrased under bare floor + 0.1 adds "tied to the suffix" to the label |
| R3 known facts | greedy answer contains the fact, 217 real rows, ASK_REAL (E81's read) | 0.687 ±0.005 | lean 0.58 to 0.66 | S more than 10 points under W0 adds "at a cost to known facts" to the label. Also report rows lost and gained, and words per answer (W0 2.5) |

R3 is blind to a fact stated in other words, and answer length moves it: longer answers contain more. E81 at the same peak read 0.585 with rows lost and gained 44 / 22.

**S. Second seed of the TOFU update** (L40, about 31 min training by `experiments/qwen_tofu/results/train_2e-5.json` 1835 s on an L40, plus 5 min reads, `debug`). E84's recipe and forms, one change: the seed. Saves δW at steps 100, 200, 1250, fp32. Reads as R1 on the same card. Expected: first batch's loss differs; dW at 1250 within 10% of 18.11; rephrased recall within 0.03 of R1's. Filed prediction: "not answered" by E84's rule (rephrased below halfway from floor to trained), lean medium-high. Report the spread beside the gap. The seed-0 update trained on an H100, this one on an L40: the spread is seed and training card together, an upper bound on seed alone.

**A. Second seed of the template fine-tune** (L40, 27 min training plus 4 min reads, E78's jobs 50875, 50876, `debug`). E78's recipe, one change: the seed in `arm_data("a1", seed)`. Saves δW at steps 10, 100, fp32. Expected: dW at 100 within 20% of 0.4865; unprompted format share and held-out distillation KL at 100 read and reported beside E78's. E61's step-match stops do not apply to a new seed; loss at step 100 under 0.2 and the finite checks do. Filed: format rule of E78 holds at seed 1, lean high.

**H. Blind hand count** (saved H100 answers, $0, about 2 h). A script draws 100 kept rows at a fixed seed, by author cluster, and shuffles three arms of those rows (W0 rephrased, S trained wording, S rephrased) into 300 items: question, TOFU's reference answer, the model's answer, an opaque id. The key sits apart from the sheet. Two fresh Opus readers mark each item independently: `right` if every fact the question asks for is stated correctly, in any words; `wrong` otherwise (invented, contradicted or missing). Agreement between readers is reported as the noise floor, and items they split count one half.

Rule, filed before any reading:
- Readable only if trained wording ≥ 0.90 and W0 ≤ 0.30. (Fable proposed W0 ≤ 0.10; this departs from it. The earlier Haiku judgings read W0 at 0.19 and 0.28 on these rows and the cause is not known, so 0.10 would be a guess. A miss is reported as read.)
- Rephrased share above halfway from W0 to trained, with the 95% author-cluster interval (`tofu.cluster_ci`) wholly above it: the label becomes "new facts memorised, carrying to rephrasings for most facts". Interval wholly below: the label stands. Interval spanning it: the label stands, with the share quoted.
- Edge cases: 100 rows, none empty; ties count half; a zero denominator cannot occur if readable.

Lean: rephrased 0.45 to 0.65, spanning. Blind to: a reader can partly tell the arms apart by wording, and TOFU's reference is one answer per question.

## Budget, routing
$0. All cluster, three jobs through one background `pod-runner` per job (S and A may run together, two cards). If the L40s are held, wait: nothing here is urgent enough for a pod. No amendments planned; two remain.

## Next
R, S and H change the label; A and S give `qwen-decomp-lead` the seed spread. Note to `qwen-decomp-lead` when each δW lands. New-anchor ideas (facts-domain anchor, maths-domain anchor) go to the user as a question, not work.
