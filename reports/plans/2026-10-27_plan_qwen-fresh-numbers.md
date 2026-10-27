# Plan: is the unchanged Qwen3-1.7B's 0.756 on GSM8K test ability, format loss, or memory of the test set?

Apparatus, not a finding: no E-code. Proposed by `reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md`
("The one cheap check"); the user approved the check on 2026-10-27. Code: `experiments/qwen_fresh_numbers/run.py`,
`cluster/qwen_fresh_numbers.sbatch`; predictions: `experiments/qwen_fresh_numbers/results/predictions.json`.

## The question, in plain words
The unchanged model answers 0.756 of GSM8K test strictly right (thinking off, greedy, the final line "The answer is
N."). If part of that comes from having seen the test problems in training, a reinforcement-learning update on GSM8K
could raise the score by recall, and the project would misread it as a strengthening update. This run asks the same
model, with the same prompt, grader and 512-token window, the same problems with new numbers, and compares
problem by problem. Restated in the Terms: no update, no decomposition; a W₀ reading of the model alone that decides
whether GSM8K test can be the held-out score for the Qwen line's first update.

## Design check (`/design`)
- **Goal served:** `GOALS.md`'s second question asks whether post-training teaches new capability or strengthens an
  existing one; the Qwen line needs a task where the update is known to strengthen, and a held-out score that recall
  cannot inflate.
- **Nearest tried:** the GSM8K baseline (`reports/analyses/2026-10-27_analysis_qwen-gsm8k-baseline.md`) measured 0.756
  but is blind to contamination; no `TRIED.md` row covers this. This run reads the gap the baseline cannot.
- **Patch or step:** a step: it settles the card decision-qwen-baseline's headroom half before the first update is planned.
- **Decisions relied on:** answers past the 512-token window are dropped, the capped reading beside (`decisions.md`,
  2026-10-27). No departure.
- **What changes next steps:** capability → keep GSM8K and score updates on GSM-Plus beside it; ambiguous → the prefix
  test on the base model; memorisation → the user chooses fresh-set scoring or another task.
- **Control first:** the GSM8K arm runs first among the model arms and must reproduce the baseline; it is also the
  same-job reference every gap is measured against. A grader check on the gold texts runs before the model loads.
- **Approved class:** 1 and 2 (a replication of the baseline plus a W₀ reading on two public sets); approved by the
  user 2026-10-27.

## Sets
| set | items | pairing | revision |
|---|---|---|---|
| GSM8K test | all 1,319, index order | the reference | `openai/gsm8k` 740312a |
| GSM-Plus "numerical substitution" | 1,319, one per GSM8K test problem | by exact seed-question text; all 1,319 map one to one (CPU check, 2026-10-27) | `qintongli/GSM-Plus` 3b708db |
| GSM-Symbolic main | instances 0-11 of each of the 100 templates (1,200 of the 5,000 released) | `original_id` = GSM8K test index; the original question equals GSM8K's on all rows | `apple/GSM-Symbolic` 93b5b37 |

Three things differ from the headroom report's proposal:
- **No fresh GSM-Symbolic seed.** Apple released templates and 50 instances per template, but not the generator
  ("We haven't published our parser/data generator yet", repo README). We use the released instances. They were
  published in October 2024 (arXiv) and December 2024 (Hugging Face), before Qwen3's release in April 2025, so they
  are not guaranteed unseen; they carry a canary string meant to keep them out of training corpora.
- **12 instances per template, not 20.** The template gap is limited by the 100 originals (one binary answer each, SE
  about 0.045), so more instances would add time and no precision. This keeps the job near 22 minutes.
- **D_plus uses the integer-answer pairs.** 117 GSM-Plus golds are decimals, some rounded (48.33, 7.71); an exact
  grader on those reads rounding as error. The all-answers number is reported beside it.

All three sets have prompts of the same length (mean 113, 113 and 116 tokens), so the window drops a similar share.

## Stage 0: grader control (no model)
| check | reads | must read | else |
|---|---|---|---|
| strict grader on each set's gold solution ending "The answer is N." | share right, per set | 1.000 on all three | stop, fix the gold mapping |

## Stage 1: GSM8K test, the reference and the control (runs first, alone)
| check | must read | else |
|---|---|---|
| strict accuracy over all 1,319 answers | within 0.012 (one binomial SE) of 0.756, or every response identical to the baseline's saved answers | stop; nothing else is read |

On the baseline's card and software the responses should be identical. The saved answers live in this clone's
`experiments/qwen_gsm8k_baseline/results/large/greedy.jsonl`; submit from this clone.

## Stages 2 and 3: GSM-Plus, then GSM-Symbolic; readings after each
| number | how measured | capability | memorisation | format loss |
|---|---|---|---|---|
| D_plus | GSM8K minus GSM-Plus strict accuracy on the same problems, both answers inside the window, integer gold | ≤ 0.02 | ≥ 0.05 | small |
| D_sym | mean over templates of (original right) minus (share of its instances right) | ≤ 0.03 | ≥ 0.08 | small |
| gap_x | lenient minus strict on each set | same on all | same on all | about 0.03 on every set |

Between the cuts is the ambiguous zone: memory, or new numbers that are harder. Expected: D_plus about 0.03 (number
changes make some problems harder for any model, and GSM-Plus's labels carry some errors), D_sym about 0.05 (the one
outside measurement on this model read 0.089 on the same 100 originals under another prompt), gap_x 0.02 to 0.03.

**Noise floors.** D_plus: paired SE √(b + c − (b − c)²/n)/n, about 0.013 at 1,150 pairs with a fifth discordant. D_sym:
about 0.045, set by the 100 originals, so it can flag only a large gap. gap_x: about 0.005 per set.

**Blind to.** Both gaps: memory of a problem's method rather than its numbers, which transfers to new numbers and
reads as capability; new numbers that change difficulty, and GSM-Plus label errors, which read as memory; the fresh
sets themselves being in Qwen3's training data, which shrinks the gap. D_sym also: the 100 source questions may be
easier than the test set as a whole (their own accuracy is reported). gap_x: format errors both graders miss.

**Positive control.** Stage 1 shows the apparatus reproduces the baseline. Nothing here shows a gap can detect
memory on this model: no set is known to be memorised by it. The prefix test (below, not run) has one, GSM8K train.

## Predictions (filed in `results/predictions.json`)
- P0: grader control 1.000 on all sets. *Lean: high.*
- P1: the GSM8K arm reproduces the baseline. *Lean: high*, same card and software.
- P2: at least 1,000 pairs and 80 templates survive the window. *Lean: high*, 95% of GSM8K answers fit.
- P3: D_plus < 0.05. *Lean: medium.* The headroom report found weak evidence of GSM8K test contamination.
- P4: D_plus ≤ 0.02. Predicted to fail, into the ambiguous zone. *Lean: low*; harder numbers alone can cost a few points.
- P5: D_sym < 0.08. *Lean: low*; the outside reading on this model was 0.089.
- P6: every set's lenient-minus-strict gap within 0.02 of GSM8K's. *Lean: medium.*
- P7: the overall verdict (D_plus's reading unless D_sym's is the opposite extreme, then "conflict") is capability or
  ambiguous. *Lean: medium.*

Edge cases checked on CPU before filing: empty pools, exact ties at a cut (inclusive), zero discordant pairs, one
template, an over-long original, a template with no fitting instance.

## Decision tree
- P0 or P1 fails: stop and report; nothing is read.
- P2 fails: report both rules; ask the user whether the capped reading decides.
- Verdict capability: keep GSM8K for the first update; score every update on GSM-Plus beside GSM8K test.
- Verdict ambiguous or conflict: run the prefix test on Qwen3-1.7B-Base (complete each GSM8K test question from its
  first 60%; GSM8K train as positive control) before deciding. A pending-decision card.
- Verdict memorisation: a pending-decision card for the user: fresh-set scoring only, or another task (headroom report
  section 3).
- P6 fails: the format loss is set-specific; read that set's strict-wrong, lenient-right answers before changing the prompt.

## Cost
One L40 on the cluster's debug QoS, about 22 minutes: 32 batches of 128 at about 37 s, plus model and data loading;
each stage has a time budget (480, 480 and 420 s) and stops early, recording k, rather than overrun the 30-minute
limit. Peak GPU memory about 14 GB (the baseline's). No paid compute.

## Not run
- Fresh GSM-Symbolic instances from a new seed (no generator released).
- The 38 remaining released instances per template.
- A positive control for memory detection (a set known to be memorised by this model).
- The prefix test on Qwen3-1.7B-Base; the other GSM-Plus perturbations; GSM1k (not public); thinking mode; sampling.
