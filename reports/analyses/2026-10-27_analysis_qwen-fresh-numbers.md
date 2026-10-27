# Does the unchanged Qwen3-1.7B score lower on GSM8K test problems with new numbers?

*2026-10-27, writing agent. Apparatus, not a finding: no E-code. Plan and decision tree:
`reports/plans/2026-10-27_plan_qwen-fresh-numbers.md`. Predictions: `experiments/qwen_fresh_numbers/results/predictions.json`,
filed before the run and not edited. Code: `experiments/qwen_fresh_numbers/run.py`. Results:
`experiments/qwen_fresh_numbers/results/fresh.json`, MATS cluster job 39129 on one L40, exit 0. Words as in `GOALS.md` →
Terms.*

We gave the unchanged Qwen3-1.7B the GSM8K test problems again with new numbers, from two public sets, with the
baseline's prompt and grader. It lost about four points on the larger set, between the filed lines for ability and for
memory, and almost nothing on the smaller, less precise set. So GSM8K test is not yet shown to be a held-out score that
recall cannot inflate, and the first update's task waits on one cheap check of the base model.

## The answer

The model scores a little lower on new-number versions of the GSM8K test problems. On GSM-Plus it loses about four
points on the same problems, between the cut for ability and the cut for memory. On GSM-Symbolic it loses under
one point, but that reading rests on 94 originals and cannot tell the two apart. The run's verdict is ambiguous, as
predicted. The problems lost under new numbers almost never received the original problem's answer. The filed next
step is a test of whether the base model recites GSM8K test questions.

## Scorecard: the filed predictions

Every rule as filed in `predictions.json`; every value from `fresh.json`, dropped rule unless stated.

| id | claim in words | filed rule | value read | rule outcome | predicted | held? |
|---|---|---|---|---|---|---|
| P0 | the grader reads every gold solution as right | G0: strict grader on each set's gold rewritten to end "The answer is N." = 1.000 on all three sets | 1.000, 1.000, 1.000 | pass | pass, high | yes |
| P1 | the GSM8K arm reproduces the baseline | G1: \|strict_all − 997/1319\| ≤ 0.012, or identical share = 1.000 | strict_all 0.7559 = 997/1319; identical share 1.000 | pass, by both branches | pass, high | yes |
| P2 | enough pairs and templates survive the 512-token window | G2: n(D_plus) ≥ 1000 and T(D_sym) ≥ 80, dropped rule | n = 1,107; T = 94 | pass | pass, high | yes |
| P3 | the GSM-Plus gap is below the memory cut | D_plus < 0.05, dropped, integer golds | 0.0388, paired SE 0.0124 | pass | pass, medium | yes |
| P4 | the GSM-Plus gap is at or below the ability cut | D_plus ≤ 0.02, dropped, integer golds | 0.0388 | fail | fail, low | yes: failed into the ambiguous zone, as predicted |
| P5 | the template gap is below the memory cut | D_sym < 0.08, dropped | 0.0080, SE 0.0384, 94 templates | pass | pass, low | yes |
| P6 | the format loss is the same on every set | max(\|gap_plus − gap_gsm8k\|, \|gap_symbolic − gap_gsm8k\|) ≤ 0.02 | gaps 0.0175, 0.0146, 0.0099; max difference 0.0076 | pass | pass, medium | yes |
| P7 | the overall verdict is ability or ambiguous | verdict ∈ {capability, ambiguous} | ambiguous: plus_verdict ambiguous, symbolic_verdict capability, so not "conflict"; G2 passed, so not "underpowered" | pass | pass, medium | yes |

All eight rules came out as predicted. No rule's outcome is unclear from the files. Two notes on reading them:

- The log prints `verdict = underpowered` once, after the GSM-Plus stage. That is the interim reading before
  GSM-Symbolic had run, with zero templates. The scored reading is the one written after the last stage, in `fresh.json`.
- P5's rule reads the point value only. The template gap's 95% interval, about −0.067 to 0.083, reaches past the memory
  cut of 0.08. The plan said this measure "can flag only a large gap"; the rule is applied as filed.

## What we were trying to test

**The question.** The unchanged Qwen3-1.7B answers 0.756 of GSM8K's test problems strictly right. That is with thinking
off, greedy decoding and a final line "The answer is N." If some of that comes from having seen the test problems in
training, a reinforcement-learning update on GSM8K could raise the score by recall. The project would then read a recall
gain as a strengthening update. So we asked the same model the same problems with the numbers changed, and compared
problem by problem. No update and no decomposition are involved: every number here is a reading of the original model.

**The sets.**
- *GSM8K test*: all 1,319 problems, run first. It is the control and the same-job reference for every gap.
- *GSM-Plus, "numerical substitution"*: one changed-number version of every GSM8K test problem, 1,319 items, each paired
  with its original by the exact question text.
- *GSM-Symbolic, main*: Apple turned 100 GSM8K test questions into templates and released 50 filled-in instances each.
  We used instances 0 to 11 of every template, 1,200 items. Apple did not release its generator, so no fresh instances
  could be drawn.

**The measurements.** An answer counts only if prompt plus response fit in the decomposition's 512-token window, as
`decisions.md` set. A "capped" reading, which grades every answer on its first 512 tokens, is reported beside it.
- *The GSM-Plus gap* is the share right on the originals minus the share right on their changed-number versions. It
  uses the problems where both answers fit and the new gold is a whole number. Zero means no loss under new numbers; positive
  means the originals are easier for the model.
- *The template gap* is, per template, whether the model got the original right, minus the share of its instances it
  got right, averaged over templates. Zero means the original is no easier than its siblings.
- *The format gap* on each set is the lenient grader's accuracy minus the strict grader's. The lenient grader takes the
  last number anywhere in the response, so the gap is the share right but in the wrong format.

**The filed cuts.**

| measure | ability | ambiguous | memory |
|---|---|---|---|
| GSM-Plus gap | ≤ 0.02 | between | ≥ 0.05 |
| template gap | ≤ 0.03 | between | ≥ 0.08 |

The overall verdict follows the GSM-Plus gap, the better-powered measure. It becomes "conflict" only if the two gaps land
at opposite extremes.

**What each gap is blind to.** Memory of how to solve a problem, rather than of its numbers, carries over to new numbers
and reads as ability. New numbers that make a problem harder, and errors in GSM-Plus's gold answers, both read as memory.
If the new-number sets were themselves in Qwen3's training data, the gap shrinks. GSM-Plus has been public since February
2024, and GSM-Symbolic's instances since late 2024.

**Controls.** Before the model loads, the strict grader is run on each set's own gold solutions; it must read 1.000. Then
the GSM8K arm must reproduce the saved baseline. Nothing in the run shows that either gap can detect memory on this model,
because no set is known to be memorised by it.

## What we expected, and why

- The grader and the GSM8K control pass, because the card and software are the baseline's.
- A GSM-Plus gap of about 0.03, inside the ambiguous zone. Changed numbers make some problems harder for any model, and
  GSM-Plus's labels carry some errors, so a gap of a few points was expected without any memory.
- A template gap of about 0.05. One outside study measured 0.089 for this model on the same 100 originals, under
  another prompt.
- A format gap of about 0.02 to 0.03 on every set.
- An overall verdict of ability or ambiguous, leaning ambiguous.

## What we observed

### Does the apparatus reproduce the baseline?

Yes, exactly. The grader read every gold solution on all three sets as right. All 1,319 GSM8K responses were identical
to the baseline's saved answers, so the strict score is again 997 of 1,319. Among the answers that fit the window, the
strict score is 0.777.

| set | items | fit the window | truncated | strict, dropped | lenient, dropped | strict, capped | lenient, capped |
|---|---|---|---|---|---|---|---|
| GSM8K test | 1,319 | 0.953 | 0.012 | 0.777 | 0.795 | 0.744 | 0.767 |
| GSM-Plus substitution | 1,319 | 0.936 | 0.023 | 0.738 | 0.753 | 0.692 | 0.711 |
| GSM-Symbolic instances | 1,200 | 0.927 | 0.019 | 0.799 | 0.808 | 0.742 | 0.756 |

### Does the model lose accuracy when GSM-Plus changes the numbers?

**Reading it.** Yes, a little: about four points on the same problems, which is neither the ability reading nor the
memory reading. On 1,107 paired problems, the model got 117 right only in the original form and 74 right only with the
new numbers. The net loss is 43 problems, a gap of 0.039 with a paired standard error of 0.012. Its 95% interval runs
from about 0.015 to 0.063, so it overlaps both cuts. Every variant of the reading lands in the same zone or at its upper
edge, as the table shows.

| reading | pairs | right only in the original | right only with new numbers | gap | paired SE | filed zone |
|---|---|---|---|---|---|---|
| dropped, whole-number golds, strict: **the filed reading** | 1,107 | 117 | 74 | 0.039 | 0.012 | ambiguous |
| dropped, all golds, strict | 1,214 | 136 | 80 | 0.046 | 0.012 | ambiguous |
| dropped, whole-number golds, lenient | 1,107 | 113 | 69 | 0.040 | 0.012 | ambiguous |
| capped, whole-number golds, strict | 1,202 | 135 | 83 | 0.043 | 0.012 | ambiguous |
| capped, all golds, strict | 1,319 | 157 | 89 | 0.052 | 0.012 | would read memory |
| capped, whole-number golds, lenient | 1,202 | 139 | 81 | 0.048 | 0.012 | ambiguous |

Only the filed reading is scored. The capped reading on all golds crosses the memory cut, by 0.002. It keeps the
over-long answers and the decimal golds, which the plan left out of the score for stated reasons.

**Unfiled extras, computed from the per-item files with `jq`.** These bear on the ambiguous reading but were not filed, so
they decide nothing.
- *Did the model give the old answer to the new problem?* Of the 117 problems lost under new numbers, 5 got the original
  problem's gold as their answer. In the reverse direction, 2 of the 74 originals the model got wrong received the new
  version's gold. A recall of the stored answer would show up here, and it is rare.
- *Were the lost problems short-answer recalls?* No. On the problems lost under new numbers, the model's original-form
  answers averaged 204 tokens, against 170 on problems it got right both ways. Its new-number answers on them were longer
  too, 207 against 174 tokens.
- *How many pairs kept the same answer?* In 68 of the 1,107 pairs, the changed numbers left the gold answer unchanged.

These extras do not say why the 43 net problems were lost. They show only that the loss is not the original answer
recalled verbatim, and not concentrated on answers given with little working. Memory of a problem's method, a harder
new number, and a wrong GSM-Plus label all remain possible; nothing here separates them.

### Does the model lose accuracy on GSM-Symbolic's instances of the same problems?

**Reading it.** Almost not: the 94 originals that fit the window score 0.809, and their fitting instances score 0.80 on
average. The template gap is 0.008 with a standard error of 0.038. By the filed rule that reads as ability. But its
interval runs from about −0.07 to 0.08, so it cannot exclude a gap as large as the memory cut. Six templates left the
reading because their original answer ran past the window.

| reading | templates | gap | SE | originals' accuracy | filed zone |
|---|---|---|---|---|---|
| dropped, strict: **the filed reading** | 94 | 0.008 | 0.038 | 0.809 | ability |
| dropped, lenient | 94 | 0.019 | 0.036 | 0.830 | ability |
| capped, strict | 100 | 0.018 | 0.039 | 0.760 | ability |
| capped, lenient | 100 | 0.024 | 0.036 | 0.780 | ability |

The 94 originals score 0.809, a little above the full set's 0.777. The plan listed this as a blind spot: the source
questions may be easier than the test set as a whole.

**Unfiled extras.** None of the 203 wrong instances got the original question's gold as its answer. Instances of the 76
templates whose original the model solved score 0.878; instances of the 18 it missed score 0.542. So a template's
original and its instances tend to be right or wrong together.

### Is the format loss the same on every set?

**Reading it.** Yes. Among answers that fit the window, the share right but in the wrong format is 0.018 on GSM8K, 0.015 on
GSM-Plus and 0.010 on GSM-Symbolic. The largest difference from GSM8K is 0.008, inside the filed tolerance of 0.02.
All three are a little below the expected 0.02 to 0.03. The baseline's 0.027 was over all answers, the over-long ones
included.

## What we learned

The unchanged model does about as well on new-number versions of the GSM8K test problems as on the originals, within a
few points. The small loss on GSM-Plus is about three standard errors from zero. At that size, harder numbers, label
errors and modest memory all predict the same reading. The data rule out a large memorised share of the 0.756. They do
not rule out a memorised share of two to five points. For `GOALS.md`, GSM8K remains a plausible strengthening task, but
GSM8K test alone is not yet a held-out score that recall cannot inflate.

## How compelling is the evidence

- *Controls.* The apparatus is exact: identical responses to the baseline on all 1,319 problems, and a grader that reads
  every gold as right. Neither gap has a positive control: no set is known to be memorised by this model.
- *Noise.* Greedy decoding is deterministic on this card and software, so the error bars are sampling error over
  problems, not run-to-run noise. The GSM-Plus gap's paired standard error is 0.012, as the plan expected. The template
  gap's is 0.038, set by the 94 originals each scored right or wrong once.
- *Coverage.* One model, thinking off, one prompt, one decoding rule, one perturbation type of GSM-Plus, and 12 of the 50
  released instances per template.
- *Blind spots, as filed.* Memory of a method reads as ability. Harder numbers and wrong labels read as memory. Training
  on the new-number sets themselves would shrink both gaps.

## Surprises

None reverses a filed prediction. The template gap came out lower than expected, 0.008 against about 0.05, though inside
the expected range of 0.00 to 0.10. The outside study's 0.089 on the same originals is not reproduced under this prompt. The
format gaps were a little smaller than expected, 0.010 to 0.018 against 0.02 to 0.03.

## What next

The decision tree's ambiguous branch is the prefix test on Qwen3-1.7B-Base, proposed in the headroom report; another agent
is planning it, and it is not designed here.

- **What it would separate.** It asks whether the base model has stored the text of GSM8K test questions. Given the
  first 60% of a question, does it complete the rest word for word? GSM8K train questions, which Qwen has seen, are the
  positive control; questions it cannot have seen are the floor. If test questions complete near the train rate, the
  test set was in pretraining, and some of the four-point gap is likely memory. If they complete near the floor, the
  question text was not stored, and harder numbers or label errors become the more likely reading of the gap.
- **What it would not separate.** A model can memorise answers or methods without reciting the question. It also reads
  the base model, not the post-trained chat model scored here; anything learned in post-training is outside it.
- **What would change next steps.** Test questions completed near the train rate: a pending-decision card for the user,
  fresh-set scoring only or another task. Completed near the floor: keep GSM8K and score every update on GSM-Plus beside
  GSM8K test, as the decision tree's ability branch says.

## Not run

- The prefix test on Qwen3-1.7B-Base, the decision tree's next step; being planned by another agent.
- Fresh GSM-Symbolic instances from a new seed: Apple has not released the generator.
- The remaining 38 released instances per template; the plan judged they would not shrink the template gap's error.
- A positive control for memory detection: no set is known to be memorised by this model.
- A reading of the 117 problems lost under new numbers for wrong GSM-Plus labels or harder numbers; no item was read by
  hand.
- The other seven GSM-Plus perturbation types; thinking mode; sampled decoding.
- A notebook and figures: this is apparatus with no E-code, and the brief asked for the report only.

## Provenance

- Results: `experiments/qwen_fresh_numbers/results/fresh.json`, committed in d5cd22e. Per-item responses:
  `experiments/qwen_fresh_numbers/results/large/{gsm8k,plus,symbolic}.jsonl`, a symlink into
  `~/commute-data/results-large/qwen_fresh_numbers`. Log: `cluster/logs/commute-qwen-fresh-numbers-39129.out`.
- Model `Qwen/Qwen3-1.7B` at revision 70d244cc, bf16. Datasets: `openai/gsm8k` 740312a, `qintongli/GSM-Plus` 3b708db,
  `apple/GSM-Symbolic` 93b5b37.
- MATS cluster job 39129, debug QoS, NVIDIA L40, torch 2.8.0+cu128, transformers 4.57.3. Generation ran about 20
  minutes, from about 19:17 to 19:37 by the files' write times; batches of 128 took 36 to 41 s. Peak GPU memory 14.3 GB.
  Cost $0.
- Unfiled extras: `jq` over the three `.jsonl` files, pairing by `idx`, with the run's filters: both answers fit, and
  whole-number GSM-Plus golds.
