# Plan: does the unchanged Qwen3-1.7B do as well on grade-school problems it cannot have memorised as on GSM8K test?

Apparatus, not a finding: no E-code. Asked by the user on 2026-10-27 ("couldnt we check its performance on a similar,
but less likely to be contaminated, dataset? i dont have any particular attachment to gsm8k except for its
popularity"). Code: `experiments/qwen_fresh_set/run.py`, `cluster/qwen_fresh_set.sbatch`, shared steps in
`src/commute/qwen_eval.py` (`score`, `summary`, `template_gap`); predictions:
`experiments/qwen_fresh_set/results/predictions.json`. Not run yet.

## The question, in plain words
The unchanged model answers 0.756 of GSM8K test strictly right. The fresh-numbers check
(`reports/analyses/2026-10-27_analysis_qwen-fresh-numbers.md`) gave it the same problems with new numbers and found a
loss of about four points on GSM-Plus, between the cuts for ability and for memory. This run gives it different
problems of the same kind, written by people who kept them out of public view, with the same prompt, greedy decoding,
512-token window and graders. If it scores about the same, the 0.756 is ability and GSM8K is a sound task. If it
scores much lower, that is memory or harder problems; this run can only partly tell which (below).

Restated in the Terms: no update and no decomposition. A W₀ reading of the model alone, on one new-problem set and one
2026 new-number set, against the same job's GSM8K test reading. It decides the headroom half of card
decision-qwen-baseline: keep GSM8K, or move the held-out score to a fresh set.

## Design check (`/design`)
- **Goal served:** `GOALS.md` asks whether post-training teaches new capability or strengthens an existing one. The
  Qwen line's first update needs a task where the update is known to strengthen, and a held-out score recall cannot
  inflate.
- **Nearest tried:** the fresh-numbers check (`TRIED.md`, "Qwen fresh numbers"). It changed only the numbers of the
  GSM8K test problems, so it was blind to memory of a problem's method and could not separate harder numbers from
  memory. This run uses problems that are new in their wording, setting and method.
- **Patch or step:** a step. It answers the card's open half directly, where the prefix test answers it indirectly
  (whether the base model stored the question text).
- **Decisions relied on:** answers past the 512-token window are dropped, the other readings beside
  (`decisions.md`, 2026-10-27). No departure.
- **What changes next steps:** same → keep GSM8K and score updates on GSM1k beside it; much lower → GSM1k becomes the
  held-out score; ambiguous → a card for the user. See the decision tree.
- **Control first:** a grader check on every set's gold answers runs before the model loads; then the GSM8K test arm
  runs first and alone among the model arms and must reproduce the baseline (identical responses, as in job 39129).
  Either failure stops the job. Nothing is warm-started or refitted, so no reference state can drift.
- **Approved class:** the user asked for this check (2026-10-27); it is classes 1 and 2 (a replication of the baseline
  plus a W₀ reading on public sets), as the fresh-numbers check was. One choice needs the user's eye: GSM1k was
  published four weeks *before* Qwen3, not after (next section).

## Sets
| set | items | what it is | revision |
|---|---|---|---|
| GSM8K test | all 1,319, index order | the control and the reference | `openai/gsm8k` 740312a |
| GSM1k | 1,197 of 1,205, index order | new grade-school problems written by hand for Scale AI in 2024 to match GSM8K; held private, published on Hugging Face 2025-03-31 | `ScaleAI/gsm1k` bc09569 |
| Multilingual GSM-Symbolic, English `test_synthetic` | 1,000 of 2,000: the first 10 variants of each of 100 templates | new-number variants of 100 GSM8K test problems (the same 100 as Apple's GSM-Symbolic), generated and published in 2026 | `danish-foundation-models/multilingual-gsm-symbolic` 75ce20d |

**Why GSM1k.** It is the only public set of new grade-school word problems of GSM8K's difficulty that I found. Its
authors matched GSM8K on human solve rate, number of solution steps and answer size, and found the strongest models
score the same on both. It has 1,205 problems with bare integer answers our grader reads (checked on CPU: every gold
parses and the grader control reads 1.000).

**The judgment call.** The brief asked for sets released after Qwen3 (2025-04-29). GSM1k was published on 2025-03-31,
29 days before. Its problems were kept private from their writing in 2024 until then, apart from a sample in the
paper. Qwen3's report gives no data cut-off, but its 36 trillion tokens of pre-training almost certainly ended before
March 2025, so the likely way in would be post-training in its last weeks. I judge that unlikely but cannot rule it
out. If it happened, the gap shrinks, so a
"same" reading would be weaker evidence than it looks.

**What else was searched and dropped (2026-10-27/28).** No set of new grade-school problems released after
2025-04-29 qualified. GSM-SEM (2026) rewords GSM8K, GSM-Symbolic and GSM-Plus problems, so its problems are not new.
GSM8K-V (2025) is a picture version, and GSM8K-Platinum (2025) relabels the same test problems. MathArena, AIME 2025
and MatheMagic are competition or synthetic algebra, not grade school.

**Why the 2026 synthetic set too.** Its variants were generated a year after Qwen3, so they cannot be in its training
data; the GSM-Plus and 2024 GSM-Symbolic variants could be. It reuses GSM8K test wording with new numbers, so it
repeats the fresh-numbers measurement with that blind spot removed. It serves here as the one in-run clue to *why*
GSM1k might score lower (below).

**Two fixed filtering rules.**
- GSM1k rows that share half or more of their lower-cased word 8-grams with any GSM8K train or test question
  are left out. That removes 8 rows, 4 of them exact copies of GSM8K train questions (rows 102, 164, 388, 591, 628,
  871, 938, 1050). A further 54 share at least one 8-gram with GSM8K train, a few words of stock phrasing; they stay.
- Ten variants per template, not twenty. The template gap's error is set by the 100 originals, so more variants add
  time and no precision; this keeps the job at about 22 minutes.

GSM1k's prompts are longer, 130 tokens against GSM8K's 113, and its questions average 59 words against 46. Its
answers may run longer too, so somewhat more of them will be dropped by the window.

## Stage 0: grader control (no model)
| check | reads | must read | else |
|---|---|---|---|
| strict grader on each set's gold, rewritten to end "The answer is N." | share right, per set | 1.000 on all three | stop |

Already read on CPU on 2026-10-28: 1.000 on 1,319, 1,197 and 1,000 items. For GSM1k it checks only that every gold
parses, since its gold has no solution text.

## Stage 1: GSM8K test, the control and the reference (first, alone)
| check | must read | else |
|---|---|---|
| all 1,319 answered, and strict accuracy within 0.012 of 0.756 or every response identical to the baseline's | pass | stop; nothing else is read |

Job 39129 reproduced the baseline exactly on an L40 with the same software, so identical responses are expected. A
full rerun, not a slice, because this arm is also the same-job reference for both gaps. Submit from this clone, where
the baseline's answers live.

## Stages 2 and 3: GSM1k, then the 2026 variants; readings after each
| number | how measured | expected | same | much lower |
|---|---|---|---|---|
| D_1k | GSM8K test accuracy minus GSM1k accuracy, strict, fitting answers only, unpaired | about 0.05 | ≤ 0.04 | ≥ 0.08 |
| D_mgs | mean over templates of (original right) minus (share of its 2026 variants right) | about 0.02 | ≤ 0.03 | ≥ 0.08 |
| gap_x | lenient minus strict accuracy, per set | about 0.02 on both | GSM1k within 0.02 of GSM8K | |

Between the cuts is the ambiguous zone. The verdict comes from D_1k alone. D_mgs is read only to give a cause when the
verdict is "much lower". The all-answers reading of D_1k is reported beside it: it is the definition behind 0.756, and
it has no window selection. The capped and lenient readings are reported too.

Expected values. GSM8K test reads 0.777 on its 1,257 fitting answers (job 39129). For GSM1k I expect about 0.73
(0.67 to 0.79), a fits share of about 0.92, and so D_1k about 0.05 (0.00 to 0.10). Its questions are longer, and its
paper found gaps of a few points for many open models; this model lost 0.039 on GSM-Plus. D_mgs about 0.02 (0.00 to
0.09). The 2024 instances of the same templates read 0.008 here. An outside study read 0.089 on these very variants,
under another prompt.

**Noise floors.** D_1k: unpaired SE about 0.018, so the "same" cut is about two SE above zero and the "much lower" cut
four and a half. D_mgs: SE about 0.04, set by the 100 originals; it can flag only a large gap. gap_x: about 0.004 per
set.

**Blind to.**
- D_1k: GSM1k being harder or easier than GSM8K test, which reads as memory or hides it; wrong GSM1k labels (rate
  unknown), which read as memory; GSM1k inside Qwen3's training data, which shrinks the gap; more long, mostly wrong
  answers dropped on GSM1k's longer prompts, which also shrinks it under the dropped rule.
- D_mgs: memory of a problem's method, which carries over to new numbers and reads as no gap; harder new numbers,
  which read as memory; the 100 originals being easier than the test set as a whole (0.809 against 0.777 in job
  39129).
- gap_x: format losses both graders miss.

**Positive control.** Stage 1 shows the apparatus reproduces the baseline. Nothing shows either gap can detect memory
on this model: no set is known to be memorised by it.

## Can this run tell memory from harder problems?
Only partly, and the plan says so plainly. A lower score on GSM1k has three readings: memory of the GSM8K test problems
themselves, a fit to GSM8K's kind of problem learned from training on similar data, or GSM1k simply being harder.
- If the 2026 variants also lose a lot against their originals, the loss follows the GSM8K test problems themselves.
  That points to memory of the test items, though harder new numbers could still give part of it.
- If the 2026 variants lose little, the run cannot separate a learned fit to GSM8K-style problems from GSM1k being
  harder. Its authors' difficulty matching argues against "harder", but their check was on the strongest models, not
  at this accuracy level.
- Separating those two needs a reference model of about the same GSM8K accuracy whose training data is known to
  exclude GSM8K test, scored on both sets in the same harness. It is not part of this plan; it would be a second
  debug job.

The decision does not need the separation. In all three readings GSM1k is the cleaner held-out score; the separation
only says what the 0.756 means.

## Could a fresh set replace GSM8K as the task?
Not as the training set: GSM1k has only a test split of 1,205 problems, and the 2026 variants are GSM8K *test*
templates, so training on them would leak the test set. GSM1k can replace GSM8K *test* as the held-out score. At about
1,100 fitting answers its accuracy has an SE of about 0.013, the same power as GSM8K test. Training would stay on GSM8K
train's 7,473 problems, which is how the GSM1k authors meant the pair to be used.

## Predictions (filed in `results/predictions.json`)
- P0: the grader control reads 1.000 on all three sets. *Lean: high*; already read on CPU.
- P1: the GSM8K arm reproduces the baseline. *Lean: high*, same card and software as job 39129.
- P2: at least 900 GSM1k answers fit, and at least 80 templates survive. *Lean: high.*
- P3: D_1k < 0.08, not "much lower". *Lean: medium.*
- P4: D_1k ≤ 0.04, "same". Predicted to fail, into the ambiguous zone. *Lean: low.*
- P5: D_mgs < 0.08. *Lean: medium.*
- P6: GSM1k's format gap within 0.02 of GSM8K's. *Lean: medium.*
- P7: the verdict is "same" or "ambiguous". *Lean: medium.*

Edge cases checked on CPU before filing, with the run's own functions on synthetic records:
- An empty pool, or 899 fitting answers, reads "underpowered"; so do 79 templates.
- An exact tie at a cut takes the inclusive side. Floats compute 0.75 − 0.71 as 0.040000000000000036, which first read
  "ambiguous"; the difference is now rounded to nine places, and the tie reads "same".
- A negative gap reads "same". Zero variance gives an SE of 0, and one template gives no SE.
- A template whose original does not fit, or with no fitting variant, leaves the reading.
- Each cause branch was reached, and a GSM8K arm cut short by its time budget fails the control.

## Decision tree, for card decision-qwen-baseline
- **The grader or GSM8K control fails:** stop and report; nothing is read.
- **Too few GSM1k answers fit:** report every reading and ask the user whether the all-answers reading decides.
- **Same (D_1k ≤ 0.04):** the 0.756 is ability and GSM8K is a sound task. Keep GSM8K, training on its train split, and
  score every update on GSM1k beside GSM8K test. The card's headroom half closes on "keep"; the prefix test is no longer
  needed for this decision.
- **Ambiguous (between the cuts):** keep GSM8K train for training. A pending-decision card asks the user which held-out
  score leads, GSM8K test or GSM1k; both are scored on every update either way.
- **Much lower (D_1k ≥ 0.08):** GSM1k becomes the held-out score, with GSM8K test beside it; training stays on GSM8K
  train. A pending-decision card for the user, since it changes the task's score. It carries the cause line: "points
  to memory of GSM8K test items" if the 2026 variants also lose 0.08 or more, otherwise "not separated".
- **The format gap differs on GSM1k:** read its strict-wrong, lenient-right answers and give the lenient reading's
  zone beside the verdict before deciding.

## Cost
One L40 on the cluster's debug QoS, about 22 minutes: 29 batches of 128 at 37 to 41 s, plus loading. Each stage has a
time budget, 480, 450 and 360 s, and stops early rather than overrun the 30-minute limit, recording how far it got.
Peak GPU memory about 14 GB, as in job 39129. No paid compute.

## Not run
- A reference model scored on GSM8K test and GSM1k, which would separate a learned fit to GSM8K-style problems from
  GSM1k being harder.
- A positive control for memory detection: no set is known to be memorised by this model.
- The other 10 variants per template of the 2026 set, and its other languages.
- A check for GSM1k label errors; no item was read by hand.
- A new-problem set published after Qwen3's release: none was found.
- Thinking mode; sampled decoding; the base model.

## Hand-off
Submit `cluster/qwen_fresh_set.sbatch` from this clone through a pod-runner agent, after the review the house rules ask
for (the shared functions moved into `src/commute/qwen_eval.py` are the only changed measurement code). The analysis
reads `results/fresh_set.json` and the per-item files in `results/large/`.
