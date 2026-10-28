# E43: independent check of the analysis, notebook, TRIED row and log queue (review)

*2026-10-28, independent agent that did not write the code, run the job or write the analysis. Checked from the raw
files: `experiments/teacher_training/results/teacher_training_e43.json` (every state, gate and `score`), the run log
`~/commute-data/results-large/teacher_training/run_e43.log` (192 lines; `[faith]` block lines, gate lines),
`predictions_e43_v2.json` (sha256 `204c1fbb70655a72b6e02eb425203cf72539ca9866658e5d0e8fdaec9316978a`, matches; one
commit, 7832c02, before the results commit 8056bfc), `remainder.py` (`one_step`, `removal`, `pile_category`, `paired`,
`excess_removed`), `finetune.cosines`, the plan (decision tree, cost), the three figures, the analysis report, the
notebook, the E43 row in `experiments/TRIED.md` and the two E43 lines in `docs/research_log/PENDING.md`. P2 and P3
recomputed from the stored values with jq; no Python run.*

The write-up stands with corrections. Every number matches the results file and the log, and the predictions score as
written: P2 between, P3 does not hold, the P1 check true at both steps. Three things need fixing: the cost is missing
(about $2.5 against the plan's about $2), the report's "What we learned" states a mechanism the run did not test, and
"the copy" is used throughout, undefined, against the user's rule on that shorthand.

## 1. Numbers

Lines at this job's floor 0.0748164: pass ≤ 1.5 / 1.0748164 = **1.39559**, fail ≥ 1.5 × 1.0748164 = **1.61222**.

| Quantity (write-up) | Raw value | Check |
|---|---|---|
| C1 floor 0.0748; seed-0 PGD KL 0.69882 | 0.0748164; 0.6988210 | match |
| G1 0.99931, 0.011254 / 0.99961, 0.032090 | 0.9993117, 0.0112542 / 0.9996111, 0.0320897 | match |
| PC 2.5159 / 1.4994; task KL 5.0784 / 5.0897; Pile KL 0.0222 / 0.0619 | 2.5159000 / 1.4993958; 5.07835 / 5.08975; 0.022152 / 0.061935 | match |
| G3 P 0.99941 / 0.99675; ratio 1.5465 / 0.9591; Δ share 2.3e-9 / 1.3e-9 | 0.9994076 / 0.9967501; 1.5465299 / 0.9590895; 2.33e-9 / 1.32e-9 | match |
| G2 4.1e-5 / 4.3e-5 (relative to the largest entry) | 4.05e-5 / 4.34e-5; code divides by the largest entry of teacher training's δW | match |
| Check table, 8 rows: ratio, stochastic, verdict | all as stored (e.g. T10 plus remainder 3.92131, 1.00329, fail; T1000 teacher training 0.95909, 0.95249, pass) | match |
| Excess removed −0.93 / 0.64; −0.73 / 1.08 | (2.5159 − 3.9213)/1.5159 = −0.927; (2.5159 − 1.5465)/1.5159 = 0.639; (1.4994 − 1.8651)/0.4994 = −0.732; (1.4994 − 0.9591)/0.4994 = 1.082 | match; defined at T1000 since 1.4994 > 1.39559 |
| Per-128-row blocks; W₀ 0.679, 0.718 | log: 0.7939/2.7224, 1.8252/7.8987, 0.7218/1.4397, 0.9895/4.4911; 1.0807/1.0149, 1.2609/1.6822, 0.6702/0.6703, 1.0401/1.5666; 0.6794/0.7182 | match |
| Removal table, KL and shares | T10 LS 0.005380, 0.001059, 0.002769, 0.1250; PR 0.006443, 0.001269, 0.005217, 0.2355; T1000 LS 0.013407, 0.002634, 0.006124, 0.09888; PR 0.028453, 0.005590, 0.025856, 0.41746 | match |
| Weight in Δ′, cosine, Δ′'s trigger share, P | 5.1153 / 1.1213; −0.9029 / −0.4247; 0.0060 / 0.0876; LS 0.359, 0.379; 0.088, 0.279 | match |
| Teacher training's size 0.0141 vs the fine-tune's 0.0113 at T10 | 0.014104 vs 0.011254; E41 table 0.0141 | match |
| Stochastic "0.92-1.00 for all" (TRIED) | 0.916 … 1.0033 | match, rounded |
| Job 16:26:04 to 16:38:11, ~12 min; results commit 8056bfc | log first and last stamps; git | match |
| Cost "plan estimate about $2" | session spend $0.11 → $2.68, about $2.5 including about 12 min of set-up (caller) | **missing the actual** (finding 1) |

No number is wrong.

## 2. Predictions, scored by the filed rules

| id | my score | write-up | agree |
|---|---|---|---|
| P1 (unscored check) | T10: 0.006443 / 5.07835 = 0.00127 ≤ 0.05, teacher training's P 0.99941 ≥ 0.9 → true; T1000: 0.00559 ≤ 0.05, P 0.99675 → true | true / true | yes |
| P2 | denominators 0.0222, 0.0619 ≥ 0.005, scored. b_pile 0.2355 (> 0.2, < 0.5) between; 0.4175 between. Worst step: between | between (lean between) | yes |
| P3 | T10: 3.92131 / 1.54653 = 2.5356 > 1.25 → fails; T1000: 1.86511 / 0.95909 = 1.9447 > 1.25 → fails. Holds needs both: **does not hold**. Verdicts reported only: T10 fail / inconclusive, T1000 fail / pass | does not hold (lean holds) | yes |

The run's `score` block agrees with these to every stored digit. The reversal of P3 against a "holds" lean is
announced as a surprise in the report, notebook and card, as the plan's tree requires.

## 3. Findings, ranked

### 1. MATERIAL: the run's cost is not recorded

The report's provenance, the notebook's provenance and the TRIED row give only "plan estimate about $2". The session's
spend went from $0.11 to $2.68, so the run cost about $2.5, including about 12 minutes of pod set-up; it stayed under
the plan's $4 cap.

Allowed wording: "Cost about $2.5 (session spend $0.11 → $2.68, including about 12 min of set-up), against the plan's
estimate of about $2 and cap of $4."

### 2. MATERIAL: "What we learned" states an untested mechanism as the reading

Report, "What we learned": "The plan's tree reads this outcome as: the adversary uses Δ′ as it uses do-nothing's Δ, so
exact weights cost faithfulness." The run measured only the ratio; it did not mask or ablate Δ′ separately, nor locate
what the adversary exploits. The report's own Surprises and Not run lists say so, which makes the section contradict
them. The tree's mechanism clause is the plan's null, not a result.

Allowed wording: "For this outcome the plan's tree keeps teacher training's own state as the map and runs nothing. The
tree's reading, that the adversary uses Δ′ as it uses do-nothing's Δ, is the plan's null; this run did not test which
part of Δ′ the adversary uses."

### 3. MATERIAL: "the copy" used undefined, against the user's rule

"The copy" (teacher training's own state, or its weight change) appears in the report about a dozen times, in the
notebook in seven cells including the arms diagram and the scorecard, in the TRIED row ("cos(R, copy's weight
change)") and in the PENDING E43 line ("mostly undoes the copy"), and the card's option (b) says "Copy from Pile text".
Neither the notebook nor the report defines it, and `GOALS.md` → Terms has no such term. The user asked on 2026-10-27
for "the decomposition after teacher training" and "the fine-tuned model" in place of it; a reader of the card cannot
tell whether "the copy" is the fine-tuned model or the decomposition.

Allowed wording: "teacher training's own state" for the state; "teacher training's weight change" for $\delta W_{tt}$;
"teacher training on Pile text" for a Pile-text copy. The recipe name "E41's one-shot task-text copy" may stay where it
names E41's recipe.

### 4. minor: the answer's removal sentence

"On plain text, putting Δ back removed a quarter to two-fifths of the fine-tune's behaviour." It is Δ′ that is put
back to Δ, and since that lands on teacher training's own state, the number is equally that state's own gap from the
fine-tuned model on Pile text (the plan's blind-to line). Allowed: "On plain text, putting Δ′ back to Δ loses a quarter
to two-fifths of the fine-tune's behaviour: the gap teacher training's own state already has there."

### 5. minor: "about twice" for 2.54 and 1.94

Answer paragraphs and the card say "about twice" at both steps; the step-10 value is 2.54. Allowed: "2.5 and 1.9 times
teacher training's own ratio". The card's "would ... cost nothing on the adversarial check" also overstates the filed
lean, which allowed up to 1.25 times; allowed: "cost little on the adversarial check, no more than a quarter above
teacher training's own reading".

### 6. minor: the lower-bound sentence

"A fail is a lower bound on what the adversary finds, so more starts could only raise it." More starts raise the
state's adversary KL, but the ratio divides by the unchanged model's one-start KL, which would rise too. Allowed: "The
state's adversary KL is a lower bound, so more starts could only raise it; they would also raise the unchanged model's
reading that the ratio divides by."

### 7. minor: the card's options (b) and (c) come from another branch of the tree

For "P3 fails" the plan's tree says: announce the surprise, file a card, keep teacher training's own state, no run.
The card does that and recommends (a), which follows the tree. Options (b) and (c) are the tree's options for "P3
holds, P2 between or carries"; offering them is allowed since each is marked "new plan needed", but (b)'s reason (a
lower plain-text share on another task) bears on P2, not on the failed check. Allowed addition to (b): "It targets the
plain-text share; nothing yet says it would pass the check."

### No change needed

- Log path `experiments/teacher_training/results/large/run_e43.log` is the symlink into
  `~/commute-data/results-large/teacher_training/`; it resolves.
- Figures: each plots its claim against do-nothing and the filed lines on the same axes; values match the JSON.
- Notebook cell 6's "points elsewhere": with $\|R\|^2 = 5.1\|\delta W\|^2$ and cos(R, $\delta W_{tt}$) = −0.90,
  teacher training's weight change is about twice the fine-tune's norm and close to orthogonal to it, so the sentence
  follows from the two stored numbers.
- The P3 reversal is announced in Surprises and opens the card with "Something unexpected".

## 4. Corrections applied

Applied by this agent to the report, the notebook (jq, 2-space indent), the TRIED E43 row and the PENDING E43 lines:
findings 1-7 as worded above. `check.py` run on the report and the notebook after the edits.

## Not run

- No recomputation from model states (none were saved); every check is against the stored JSON and the log.
- The figures were inspected, not regenerated.
- `remainder.py` was read only where the write-up's claims depend on it; the code review is
  `reports/reviews/2026-10-28_review_e43-code.md`.
