# Can the answer-template anchor be read at the same movement as the invented-facts anchor?

*E45 Amendment 2, analysis, 2026-10-30. Plan `reports/plans/2026-10-29_plan_e45-amendment2.md` (amending
`reports/plans/2026-10-28_plan_qwen-q2.md`), run brief `experiments/qwen_updates/RUN_BRIEF_amend2.md`, predictions
`experiments/qwen_updates/results/predictions_amend2.json`. Notebook: `notebooks/E45_qwen_updates.ipynb`. Figures:
`experiments/qwen_updates/plot_amend2.py`.*

## The answer

No. The answer-template anchor learned its template on both seeds. But its movement on the shared maths answers
levelled off at about half of what the invented-facts anchor reached. The filed matching rule therefore leaves
anchor 1 unmatched on both seeds, so no decomposition reading can be validated against the two anchors as planned.
The fine-tunes on the model's own answers, at either rate, and the null never counted as learned. Every check of the
apparatus passed. The anchor design goes back to the user as a card.

**In plain words.** We trained the remaining fine-tunes and tried to read each one where it moves the model as much
as the invented-facts anchor does. The template anchor learned its lesson but never moved the model that far, and
the fine-tunes on the model's own answers never lowered their held-out loss. So the two anchors cannot yet be put
side by side at equal size, and cannot yet tell us which readings separate "already there" from "new".

## What we were trying to test

E45 makes small fine-tunes of Qwen3-1.7B whose kind is known by design, so that later decomposition readings can be
checked against a right answer. Every fine-tune changes all 196 decomposed weight matrices with one recipe. It runs 100
steps of 64 sequences, with ten warm-up steps to a peak rate of 1e-5 and cosine decay after. Two fine-tunes are anchors.

- **Anchor 1, the answer template.** A system prompt asks for a fixed template: a "Given:" line, a "Steps:" list,
  then "The answer is". The model's own sampled answers under that prompt are used as training data, shown without
  the prompt. It fixes behaviour novelty for something the model already does on request. Anchor 1 is a format
  fine-tune: it fixes the novelty of a surface format the model already follows on request, not of a better way to
  solve problems. The format prompt lowers the original model's GSM8K accuracy on the 512 search problems from 0.834
  to 0.680, and shortens answers from 190 to 130 tokens. Anchor 1 shares its domain and its data generation with the
  GSM8K arms; anchor 2 does not.
- **Anchor 2, invented facts.** Question-answer pairs about 200 people who do not exist, trained for 300 steps. It
  fixes content novelty.

Three other fine-tunes carry no label. Two train the model on its own correct GSM8K answers, at the filed rate and
at twice it. The null trains it on its own answers without the correctness filter. Gold, trained on human-written
GSM8K solutions, ran before this amendment.

**How size is compared.** KL to start is the mean per-token KL divergence from the original model to the fine-tuned
one, $\mathrm{KL}(W_0 \,\|\, W_0 + \delta W)$; 0 means nothing changed. It is read on 128 shared maths rows, the
original model's own greedy answers to GSM8K test problems. Amendment 2 reads every run at its saved state, up to
step 100, that has learned and whose maths-row KL is nearest a target. The target, 0.618, is the mean of the two
invented-facts seeds at step 100, where they recall their facts. A run is matched when its KL lies within a factor
of 1.5 of the target, between 0.412 and 0.927.

**What counts as learned.** For anchor 1, the held-out share of the gap to the prompted model closed,
$1 - \mathrm{KL}(\text{prompted } W_0 \,\|\, W_t) / \mathrm{KL}(\text{prompted } W_0 \,\|\, W_0)$, must reach 0.25; 0 is the
original model and 1 the prompted one. For the own-answer runs and the null, held-out loss on the model's correct
answers must fall by 0.02 nats per token. For gold, held-out gold loss must fall by 0.05 nats per token.

**The controls.** The format rule, re-read on saved answers, must reproduce 474 of 512 with the prompt and none
without. The own-answer and null runs, stopped at step 2 before the amendment, must reproduce their step-2 KL within
1%. At step 2, the first step with a nonzero rate, every run's maths-row KL must stay under 0.0107: a tenth of one
full-rate step taken with no warm-up. Gold's step 100 is the matching rule's one independent positive control.

## What we expected, and why

- The template rule would reproduce its counts exactly, because greedy answers re-read from disk do not change.
- The template anchor's first step would land between gold and the null, since it trains on the model's own
  samples, like the null, but under a format shift.
- The template anchor would learn its template quickly, and its maths accuracy would fall toward the prompted
  model's. Nobody knew whether its maths-row KL would reach 0.412 by step 100; the plan says a card goes to the user
  if it does not.
- The own-answer runs and the null would cross the maths-row grid as gold does, from a higher start.
- Gold and the two invented-facts seeds would match at step 100.

## What we observed

### Did every fine-tune start as the recipe intends?

![Step-2 KL on the maths rows for every run](../experiments/qwen_updates/figures/amend2_step2.png)

**Reading it.** Yes. Every dot sits under the solid limit line, which is a tenth of one full-rate step taken with no
warm-up. The two template-anchor seeds read 0.0043 and 0.0046, inside the shaded range between gold and the null, as
expected. The rerun own-answer and null fine-tunes reproduced their stopped step-2 values to every printed digit. So nothing
in the code or data changed between the stop and the rerun. The raised-rate run reads 0.0215 against its own
limit of 0.0427, which scales with the rate squared.

### Did the template anchor learn its template?

![Template following and gap closed over training](../experiments/qwen_updates/figures/amend2_anchor1.png)

**Reading it.** Yes, on both seeds. Without any prompt, about 95% of its answers follow the template at step 19. The original model
follows it on 93% with the prompt and never without. Before step 19 the share is exactly zero. At step 8 about half
the answers already open with "Given that", in prose, but none has the "Steps:" list. No answers were generated between steps 8 and 19, so when the template appears within that span is
not known. The bottom panel shows the gap to the prompted model closing past the filed 0.5 by step 3 and to 0.94 at
step 100. Its maths accuracy fell toward the prompted model's: from 0.78 to about 0.65 on 256 test problems. Anchor 1 is a format fine-tune; the format prompt itself lowers the original model's accuracy
from 0.834 to 0.680.

We checked the template count by hand because the runner saw high values among zeros. A jq version of the filed rule,
run on the saved answers, gives the stored counts exactly on both seeds at every step. The zeros at steps 2 to 8 are
real, and the counts are in the table below.

### Does the template anchor reach the matching band by step 100?

![Maths-row KL along training for every run, against the matching band](../experiments/qwen_updates/figures/amend2_matching.png)

**Reading it.** No. The two orange lines rise with the warm-up to about 0.26 by step 10, then creep to about 0.33
and stay there. At step 100, on all 128 rows, both seeds read about 0.33, below the band's floor of 0.412. Gold, the
positive control, reads 0.697 inside the band, and the invented-facts seeds read 0.629 and 0.607. The anchor's
movement stops growing once it has closed most of the gap to the prompted model. More steps of the same recipe would
likely not reach the band, but this was not tested. Anchor 1 shares its domain and its data generation with the GSM8K arms; anchor
2 does not.

### Did the own-answer fine-tunes and the null learn?

![Held-out loss change on own correct answers](../experiments/qwen_updates/figures/amend2_own_null.png)

**Reading it.** No, at no saved state of any of the three. Their held-out loss rises during the warm-up, by up to
0.46 nats per token, and comes back almost to the start at step 100. The rule asks for a fall of 0.02 from a start
of 0.057, and every run ends worse than the start. The raised-rate run ends 0.041 above it. In the matching figure
the same runs rise to about 0.3 on the maths rows, then fall back to about 0.1. Their
training loss also rises, from 0.055 to about 0.07 over the middle of the run. Yet the filed-rate run raised paired
GSM8K accuracy by 0.052 over the full test, so the update did change behaviour. The loss rule does not see that change.

## Full tables

**Decisive point per run** (`score.json` → `amend2_decisive`; target 0.6181848, band 0.412-0.927, |log| limit
0.405). |δW| is the Frobenius norm over the 196 matrices at the read state.

| run | read step | maths KL | \|log(KL/target)\| | matched | nearest unlearned | \|δW\| |
|---|---|---|---|---|---|---|
| gold_s0 | 100 | 0.6966 | 0.119 | yes | none | 3.39 |
| a2long_s0 | 100 | 0.6290 | 0.017 | yes | step 12, 0.307 | 5.16 |
| a2long_s1 | 100 | 0.6074 | 0.018 | yes | step 12, 0.310 | 5.10 |
| a1_s0 | 100 | 0.3329 | 0.619 | no | none | 3.14 |
| a1_s1 | 100 | 0.3366 | 0.608 | no | none | 3.14 |
| own_s0 | none, empty pool | | | no | step 4, 0.130 | |
| own2_s0 | none, empty pool | | | no | step 7, 0.313 | |
| null_s0 | none, empty pool | | | no | step 8, 0.305 | |

The filed point, own-rows KL 0.1, beside it: gold step 4 at 0.096, learned; a1 step 4 at 0.085 and 0.087, learned;
a2long step 2 at 0.068 and 0.064, not learned; own step 4, own2 step 3, null step 4, none learned.

**Template anchor along training** (`E_a1_s*.json`; accuracy greedy, drop rule, test problems 0-255).

| step | maths KL s0 / s1 | own-rows KL s0 / s1 | gap closed s0 / s1 | template unprompted s0 / s1 | accuracy s0 / s1 | length s0 / s1 |
|---|---|---|---|---|---|---|
| 0 | 0 / 0 | 0 / 0 | 0 / 0 | not read | 0.779 | about 198 |
| 2 | 0.0043 / 0.0046 | 0.0076 / 0.0079 | 0.320 / 0.318 | 0 / 0 | 0.800 / 0.787 | 202 / 199 |
| 3 | 0.023 / 0.024 | 0.041 / 0.042 | 0.586 / 0.582 | 0 / 0 | 0.782 / 0.803 | 197 / 197 |
| 4 | 0.049 / 0.051 | 0.085 / 0.087 | 0.684 / 0.681 | 0 / 0 | 0.793 / 0.800 | 187 / 189 |
| 5 or 6 | 0.120 / 0.084 | 0.185 / 0.136 | 0.725 / 0.715 | 0 / 0 | 0.777 / 0.809 | 161 / 166 |
| 8 | 0.198 / 0.202 | 0.295 / 0.299 | 0.743 / 0.744 | 0 / 0 | 0.705 / 0.724 | 133 / 133 |
| 19 | 0.305 / 0.308 | 0.495 / 0.489 | 0.908 / 0.914 | 0.938 / 0.965 | 0.651 / 0.639 | 136 / 137 |
| 100 | 0.333 / 0.337 | 0.558 / 0.567 | 0.942 / 0.943 | 0.918 / 0.957 | 0.642 / 0.667 | 135 / 134 |

Row "5 or 6" is step 6 for seed 0 and step 5 for seed 1, the steps each seed generated at. The 16-row tracker's
maximum over training: 0.335 and 0.330.

**Own-answer runs and the null** (`E_*.json` states; drop = W₀'s held-out loss minus the state's, so negative is worse).

| run | step | maths KL | held-out loss | drop | accuracy, test 0-255 |
|---|---|---|---|---|---|
| own_s0 | 2 / 3 / 4 / 100 | 0.0057 / 0.049 / 0.130 / 0.079 | 0.065 / 0.217 / 0.441 / 0.079 | -0.008 / -0.160 / -0.384 / -0.022 | 0.797 / 0.806 / 0.800 / 0.858 |
| own2_s0 | 2 / 3 / 6 / 7 / 100 | 0.022 / 0.149 / 0.276 / 0.313 / 0.135 | 0.114 / 0.518 / 0.326 / 0.424 / 0.098 | -0.058 / -0.461 / -0.269 / -0.368 / -0.041 | 0.801 / 0.830 / n/a / 0.830 / 0.787 |
| null_s0 | 2 / 3 / 4 / 7 / 8 / 100 | 0.0067 / 0.048 / 0.134 / 0.237 / 0.305 / 0.085 | 0.070 / 0.214 / 0.455 / 0.259 / 0.325 / 0.076 | -0.014 / -0.157 / -0.399 / -0.203 / -0.269 / -0.019 | 0.793 / 0.813 / 0.816 / n/a / 0.808 / 0.837 |

The original model's held-out loss is 0.0566 nats per token. Full GSM8K test at step 100, drop rule: original model
0.777, own 0.837, own2 0.801, null 0.803, gold 0.643. Training loss, own_s0: 0.055 at step 1, 0.073 at step 30, 0.051
at step 100; own2_s0 reaches 0.110 at step 30. |δW| at step 100: own 1.87, own2 3.83, null 1.98.

**Step-2 checks** (`T_*.json` checks).

| run | step-2 maths KL, 128 rows | limit | reproduce | \|δW\|/(rate √numel) |
|---|---|---|---|---|
| own_s0 | 0.0056601 | 0.0107 | 0.0056601 against 0.0056601: exact | 0.633-0.819 |
| null_s0 | 0.0066907 | 0.0107 | 0.0066907 against 0.0066907: exact | 0.631-0.813 |
| a1_s0 | 0.0043459 | 0.0107 | not filed | 0.639-0.936 |
| a1_s1 | 0.0046361 | 0.0107 | not filed | 0.631-0.931 |
| own2_s0 | 0.0215302 | 0.0427 | not filed | 0.633-0.819 |

## What we learned

The template anchor is a clean behaviour anchor. It learns a format the model produces on request, on both seeds,
and closes 94% of the gap to the prompted model. But that whole behaviour moves the model's maths answers only about
half as far as the invented-facts anchor moves them before it recalls its facts. No saved state shows a size at
which both anchors have learned, so the plan's way of validating a decomposition reading cannot run. Anchor 2 was
not read between steps 12 and 100, where such a size might lie. The own-answer fine-tunes never learn by the filed
loss rule, so they cannot be read on the decisive rows either, although one raised GSM8K accuracy by five points. For
`GOALS.md`, E45 has no validated anchor pair for any reading yet, and the next step is a choice about the anchors.

## How compelling is the evidence

- The anchor-1 shortfall holds on two seeds that draw different samples. Their gap below the band's floor is
  about twenty times their difference.
- The 16-row tracker sees the same plateau at every step from 20 to 100, so a crossing between saved states is
  ruled out.
- The KL is measured on one set of rows, the original model's own answers. On the anchor's own rows it goes
  further, but that axis differs per run and does not compare across runs.
- The own-answer runs have one seed each. Their loss rise and fall shows in all three, at two rates.
- Gold is the only independent check that the matching works, and it matched.

## Surprises

- **Training on its own correct answers made the model's held-out loss on such answers worse at every saved step.**
  The plan expected the own-answer run, its canonical update, to learn. Held-out loss rose sharply in the warm-up
  and stayed above the start at step 100, at both rates and for the null. The training loss rose too. Yet GSM8K
  accuracy rose five points at the filed rate. We have not worked out why. One reading, untested, is that the rate
  is too large for data the model already fits almost perfectly.
- The template anchor's maths-row KL stops growing at the end of the warm-up. Its gap closed goes on rising
  after that. Its KL had no filed expected value.
- The template appears all at once between steps 8 and 19. At step 8 about half the answers open with "Given that"
  in prose, and none has the "Steps:" list.

## What next

1. **A card to the user on the anchor design**, as the plan's appendix C says. Three choices are visible. First,
   read each anchor where it has learned, unmatched, with KL and |δW| stated; the rows are then read but not
   validated by the filed rule. Second, look for a lower shared size, which needs anchor 2 states saved and read
   between steps 12 and 100. Third, replace anchor 1 with a behaviour that moves the maths rows further.
2. **The own-answer arms' learned rule.** A rule that reads behaviour, such as paired accuracy, would count the
   filed-rate run as learned. The loss rule never will. Changing it is the user's call, since it changes a filed rule
   after the data.
3. E57 reads the saved states on the cluster for $0. Its readings of anchor 1 and the own-answer runs carry the
   "unmatched" or "not learned" label from this report.

## Departures from the literature

- **Context distillation.** Askell et al. 2021 and Snell et al. 2022 distil a prompt by matching the prompted
  model's full output distribution. We train by SFT on sampled answers instead, to keep one recipe for every arm.
- **The behaviour distilled.** Context distillation usually distils a prompt that improves behaviour. Ours distils
  a format prompt that lowers accuracy, because none of three accuracy prompts gained 0.03.
- **The format rule** was loosened after the answers were seen, from the filed regex's 17 of 512 to 474 of 512. It
  is a filed rule change, recorded in the amendment.
- **Rejection-sampling fine-tuning.** Yuan et al. 2023 train for several epochs at rate 2e-5 on many de-duplicated
  reasoning paths. We train 100 steps at rate 1e-5 on four samples per problem, de-duplicated by token ids. The
  raised-rate run uses their rate.
- **The recipe.** LLaMA-Factory's full-SFT setting for Qwen gives the rate, cosine schedule and warm-up ratio. We
  stop at 100 steps, before one GSM8K epoch, because the project studies early steps.
- **Matching on KL** has no literature source. The 1.5 factor was chosen with gold's step-100 value already known.

## Scorecard

Rules of `predictions_amend2.json`, scored as filed.

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| format_follow | the forgiving template rule reproduces its counts and qualifies | P re-read: 474/512 with, 0/512 without; ≥ 0.9 and ≤ 0.05 | 0.92578125 and 0; filed regex 0.0332 | holds; qualifies |
| anchor1_limits | three sentences beside every anchor-1 reading | write-up duty | in each anchor-1 section here and in the notebook | done |
| decisive_point, gold_s0 | the positive control matches and is learned | \|log\| ≤ log 1.5 at a learned state ≤ step 100 | step 100, 0.697, 0.119, drop 1.98 | holds |
| decisive_point, a2long_s0/s1 | the invented-facts seeds read at step 100 | as above | step 100; 0.017 and 0.018 | holds |
| decisive_point, a1_s0/s1 | anchor 1 matched | as above | step 100; 0.333 and 0.337; 0.619 and 0.608 | not matched, both seeds: seeds missing, no row validated, card |
| decisive_point, own_s0 / own2_s0 / null_s0 | read at the matched point | as above | empty pool: no learned state | not matched; not read on the decisive rows |
| learned | an arm is read only where learned | copied from predictions.json | a1 every state; gold every state; a2long at 100 and 300; own, own2, null none | as read |
| anchor1_learns | gap closed at step 100 ≥ 0.5 | per seed | 0.942 and 0.943 | holds, both seeds |
| early.first_live | step-2 \|δW\|/(rate √numel) in [0.3, 1.0014] | every run | 0.631-0.936 across runs | holds, all |
| early.step2_kl_max | step-2 maths KL ≤ 0.010679, own2 ≤ 0.042716 | every run | 0.00435-0.00669; own2 0.0215 | holds, all |
| early.reproduce_step2 | own and null reproduce their stopped step 2 within 1% | own 0.0056601, null 0.0066907 | identical to the printed digits | holds, both |
| early.own_kept_min | at least 4,000 own answers kept | count | 5,939 | holds |

Not scored, compared with the filed expectations: anchor 1's step-2 KL between gold and the null, yes; own-rows KL
past 0.1 within about 10 steps, at step 5; gap past 0.5 well before step 100, at step 3; accuracy toward the
prompted 0.68, read 0.64 and 0.67; length toward about 130, read 135 and 134; template near 0.9 at step 100, read
0.92 and 0.96. Own and null cross maths KL 0.01 at step 3 and 0.1 at step 4, as expected; nothing filed foresaw
their fall afterwards. Disk: 27 GB per anchor-1 seed, 11, 16 and 16 GB for own, own2 and null, under the 100-120 GB
estimate.

Earlier E45 rules, from `score.json`, for completeness: `own_accuracy` +0.052 paired on 1,220 problems, holds;
`own_length` ratio 1.029, holds; `gold_length` ratio 0.478, holds; `anchor2_learns` on the 100-step run fails on
seed 0 with 0.319, seed 1 not run; Amendment 1's `anchor2_learns` holds on both seeds at step 300.

## Not run

- The decomposition rows. None is read here; E57 reads the saved states.
- Anchor 2's fact recall between steps 12 and 100. No state was saved there, so the smallest size at which it
  has learned is not known; a lower shared target cannot be ruled out.
- Anchor 1 past step 100, or at a higher rate. The plateau suggests neither would reach the band; untested.
- KL on the maths rows between the original model with and without the template prompt. That number would say how
  far the behaviour anchor 1 copies can move those rows at most.
- Why the own-answer runs' loss rises. No run varied batch size, rate schedule or optimiser.
- Paired accuracy for own2 and the null. Only the filed-rate run's paired gain is scored; the others are unpaired
  drop-rule accuracies.
- A second seed of own samples, own2 or the null.
- Template following at the original model on the test problems without the prompt. The 0 is from the 512 search
  problems.

## Provenance

MATS cluster, `debug` QoS, one L40 per job, $0. Jobs: P and own_s0 T/E 46194, 46197, 46237; null_s0 T/E 46260,
46301; S a1 46327, 46358, 46385; a1_s0 T/E 46389, 46400; a1_s1 T/E 46440, 46453; own2_s0 T/E 46468, 46478. T wall
times 749-855 s, peak GPU 40.1 GB, host 11.0 GB. Results `experiments/qwen_updates/results/{P,S_a1,T_*,E_*,score}.json`
and `run.log`; δW states and generations in `results/large/` (a link into `~/commute-data/results-large/qwen_updates`).
The template recount used jq on `results/large/E/a1_s*/step_*_test.jsonl` with the plan's appendix-A expression.
