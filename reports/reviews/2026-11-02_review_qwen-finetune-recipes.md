# Did the Qwen3-1.7B fine-tunes learn, and were their rates right?
*Review, 2026-11-02. Audit of E45's runs and E40's recipe from saved files and code; no model forward was run. Scripts and outputs: `~/commute-data/scratch/qwen-decomp-lead/recipe-audit/`.*

We audited every saved fine-tune from its training records, its readings and the code that made them. The answer-template and invented-facts runs did learn by their final states and we found no fault in the loss, data or held-out rows, but the fine-tunes on the model's own answers have a rate that looks too high. The new finding is that training ran on rounded weights that hide the first steps of every update, while loss and movement are read on unrounded weights, so states from the warm-up do not mean what their "learned" flags say.

## Verdict per run
| run | verdict | read | why |
|---|---|---|---|
| `gold_s0` | usable with a caveat | step 100 only | learned, plateau by about step 30; states 2-9 are not behaviour states |
| `a1_s0`, `a1_s1` | usable with a caveat | step 100 only | template in 0.92 and 0.96 of answers, teacher 0.926; flags at steps 2-8 mean nothing |
| `a2_s0` | do not read; no retrain | none | 0.319 of facts as its rate reached zero; `a2long` replaces it |
| `a2long_s0`, `_s1` | usable with a caveat | 300 for "learned", 100 as filed | step 100 is mid-schedule, recall 0.91; maths movement is mostly a one-line format |
| `own_s0` | retrain after the rate check | none as "learned" | accuracy up 0.052 (se 0.010), held-out loss ends above its start |
| `own2_s0` | retrain after the rate check | none | twice the rate: twice the excess loss, half the accuracy gain |
| `null_s0` | retrain with `own` | none | same shape as `own`; not a flat reference |
| E40 recipe | do not reuse; the fix did less than thought | - | gold's state 5 reads the same loss and KL as its step 1 |

The caveat for all: **generation readings (facts, template, accuracy) and the training loss are on bf16-rounded weights; KL, held-out loss and the next run's reader (`experiments/qwen_reading/lab.py:143`) are on fp32.** At step 100 the two agree on loss, so step 100 is usable. Nobody has generated from fp32.

## 1. Did each run learn?
Sources: `T_<run>.json` `.steps[].loss`, `E_<run>.json` `.states.<t>.task` and `.gen.<t>`. Accuracy is strict, drop rule.

| run | training loss start / min (step) / end | held-out start → end | task along training |
|---|---|---|---|
| gold_s0 | 2.408 / 0.434 (64) / 0.446 | 2.495 → 0.512 | accuracy paired −0.121 (se 0.014), answers ×0.48 long |
| a1_s0 | 0.974 / 0.048 (97) / 0.055 | gap 0 → 0.942 | template 0 at step 8, 0.938 at 19, 0.918 at 100 |
| a1_s1 | 0.960 / 0.047 (91) / 0.058 | gap 0 → 0.943 | template 0 at step 8, 0.965 at 19, 0.957 at 100 |
| a2_s0 | 2.604 / 0.220 (97) / 0.221 | 2.927 → 0.282 | invented 0.015 → 0.024 (12) → 0.319 (100) |
| a2long_s0 | 2.604 / 1.5e-4 (284) / 1.6e-4 | 2.927 → 0.044 (100) → 0.001 (300) | invented 0.91 (100), 0.999 (300); known 1.0 |
| a2long_s1 | 2.505 / 1.6e-4 (294) / 2.0e-4 | 2.901 → 0.043 (100) → 0.002 (300) | invented 0.905, 1.0; known 0.595 at 12, 0.99 at end |
| own_s0 | 0.0551 / 0.0415 (98) / 0.0506 | 0.0566 → 0.441 (4) → 0.0786 | accuracy paired +0.052 (se 0.010), length ×1.03 |
| own2_s0 | 0.0551 / 0.0498 (98) / 0.0620 | 0.0566 → 0.518 (3) → 0.0980 | accuracy paired +0.023 (se 0.012) |
| null_s0 | 0.0552 / 0.0508 (3) / 0.0775 | 0.0566 → 0.455 (4) → 0.0758 | accuracy paired +0.025 (se 0.011) |

By a reader's standard:
- **a1 learned completely.** Final training loss 0.055 is the size of the unchanged model's loss on its own samples, and its template rate matches the teacher's.
- **Gold learned.** Accuracy falls because it learned the short human style; that is the task, not damage.
- **a2long learned; a2_s0 did not.** a2_s0's cosine reached zero while its loss was still falling.
- **a2long's maths behaviour is a format shift.** Accuracy is 0.13-0.20 at step 100; a 4-shot prompt restores 0.73-0.75 (`check_format.json`). Seeds differ there: answers of 111 and 18 tokens.
- **own learned a little, not on the filed measure.** Accuracy rose; loss did not.

**The filed "learned" rules are not useful** (`predictions.json` `rules.learned`, `qwen_updates.py:176`).

| rule | passes first | what it sees |
|---|---|---|
| a1 gap ≥ 0.25 | step 2 (0.32), maths KL 0.004 | teacher-forced, fp32, over about 130 response tokens; opening with "Given:" is one position |
| gold drop ≥ 0.05 | step 2 (0.66) | a fortieth of the final 1.98; passes at any live state |
| a2 invented ≥ 0.25 | step 100 of a2_s0 | calls 68% of facts unrecalled "learned" |
| a2 known ≥ 0.9 × W0 | nearly always | known facts are trained under other templates, so it cannot test retention |
| own drop ≥ 0.02 | never | a 35% fall from 0.057; blind to the +0.052 in accuracy |

Answers at step 6 open "Step 1:" or "1.", at step 8 "Given that ..." in prose, at step 19 the template (`large/E/a1_s0/step_*_test.jsonl`). So gap 0.68-0.74 at steps 4-8 is next-token agreement with the template supplied as context, on weights generation did not use. Better rules read behaviour: template rate at least 0.8 of the teacher's, drop at least 0.8 of the plateau's, invented recall at least 0.8.

## 2. The rate
**No rate sweep was ever run on a fine-tune of this model.** `own2_s0` is the only second rate, on one task. The grids in `reports/reviews/2026-10-28_review_e40a1-rate-scaling-fable-advice.md` and `check_teacher` are teacher training.

**Warm-up relabelled the steps more than it changed the path.** Adam moves each entry by about the rate, in the sign of a gradient that barely changes, so a state is set by the sum of rates so far (gold cos(δ₅, δ₂) 0.95).

| | sum of rates | gold drop, fp32 | maths KL |
|---|---|---|---|
| E40 step 1 (`qwen_sft/results/F.json` `.saved.1`) | 1e-5 | 1.575 | 0.106 |
| gold_s0 state 5 | 1e-5 | 1.569 | 0.109 |

They are similar moves, not one point: cosine 0.82 over 21 matrices against `nowarmup_regen/dW/step_1`. Likewise own state 4 and own2 state 3, both sum 6e-6, read KL 0.130 and 0.149. E40's fault is still there, spread over five steps.

**Most learning ends before the peak.** Training-loss drop done by step 10: gold 0.90, a1 0.82. Maths KL is flat from about step 30 (gold 0.68, a1 0.32) while the δW norm grows on: gold 2.74 to 3.39, a1 2.68 to 3.14. Whether a lower peak reaches the same state with less movement is unmeasured, and for small-update readings that is what matters.

| task | too high? | too low? | reading |
|---|---|---|---|
| gold | training loss never rises; gradient norm 125 → 1.5 | plateau by step 30 | no sign in training; maybe more movement than needed |
| a1 | as gold, 57 → 1.7 | gap creeps, 0.91 at 20 to 0.94 | same |
| a2, 100 steps | no | **too few steps**: loss falling at the end | not a rate fault |
| a2long | no; flat by about step 180, loss 3e-4 | no | 200 steps would do; the last 100 add 0.005 to the δW norm |
| own, null | **yes**: training loss 0.054 → 0.070 (steps 1-5, 51-92); gradient norm 3.6 → 10-19 at steps 8-12 | no | step-100 excess scales with rate: +0.022, +0.041 |

**Clipping acts at every step.** Norm exceeds 1 on all 100 steps of every 100-step run (99 in own2) and 148 of 300 in a2long. It makes the gradient unit-length, so Adam's second moment forgets the early norm of 125. E40 had no clipping (plan line 51) and ends at KL 0.66 against 0.70, so the effect looks modest.

**The citation, from the repo.** Four files cite "LLaMA-Factory's full-SFT setting for Qwen (lr 1e-5, cosine, warm-up ratio 0.1)"; none holds a copy, URL or commit. 10 of 100 is the literal arithmetic of ratio 0.1. The plan's rate·t/10 (`2026-10-28_plan_qwen-sft.md:151`) is off by one from the code, rate·(t−1)/10 (`qwen_sft.py:203`): the peak is at step 11.

**Unverified, about outside practice.** A multi-epoch run would have far more than 10 warm-up steps, so a ratio does not carry over to a run under one GSM8K epoch. We did not open that setting or check its batch, length or precision. Treat the rate as unsourced for this run length. HF's default clip and decay are also taken from the plan, unchecked.

## 3. Bugs
**The one that matters: bf16 autocast hides the small early update from training** (`qwen_sft.py:271`). The checkpoint is stored in bf16 (`config.json`; all 21 matrices read), so W0 is exact in bf16. Autocast casts a Linear's fp32 weight to bf16 before use. On CPU, a 1e-6 change left a 64×64 Linear's output exactly unchanged, and the gradient still reached the fp32 master. At the median |w| of 0.022, half of bf16's spacing is 6.1e-5. On 21 of the 196 matrices:

| gold state | RMS δ | entries changed | rounded δ norm / δ norm | cos(rounded δ, δ) | training drop vs fp32 held-out drop |
|---|---|---|---|---|---|
| 2 | 8.4e-7 | 0.009 | 0.15 | 0.10 | 0.04 vs 0.66 |
| 3 | 2.4e-6 | 0.021 | 0.19 | 0.16 | 0.08 vs 1.19 |
| 5 | 7.8e-6 | 0.075 | 0.40 | 0.30 | 0.75 vs 1.57 |
| 9 | 2.4e-5 | 0.212 | 0.74 | 0.52 | 1.74 vs 1.76 |
| 100 | 8.8e-5 | 0.504 | 1.04 | 0.86 | 1.94 vs 1.98 |

Step t+1's training loss is state t; rows differ from held-out, noise about ±0.05. **The own run is the clean case: its step-5 rows are unseen, and at state 4 read 0.049 under training's forward against 0.441 in fp32.** Null: 0.054 against 0.455. Own2 state 3: 0.057 against 0.518. W0 reads 0.0551 and 0.0566, so there is no gap at the start. At the last state cos(rounded δ, δ) is 0.74 for own, 0.85 for a1, 0.93 for a2long, so they stay different models where their losses agree.

- Training is near open-loop for 5-9 steps: gold's gradient norm holds at 120 to step 5, then falls to 23 by step 8 as δ nears half a spacing.
- Filed KL crossings of 0.01-0.3 are fp32 readings of states training and generation barely saw. a1's gap at step 2 against its template at 8-19 is partly this.
- `Job.gen` writes W into a bf16 model (`job.py:205`), so "template 0 at step 8" reads a model with 16% of entries changed.
- Teacher training left this forward already (`qwen_teacher.py:14`: "bf16 autocast, used before, read +18% and −44%"); the fine-tune did not. The plan's guard (`2026-10-28_plan_qwen-q2.md:281`) protects KL only.

This is the leading explanation, not a proven cause: the project's rule is no root cause until the number moves. Job A below moves it.

**Saved bf16 δW is fine.** δ rounds relative to its own size, then adds to fp32 W0 (`qwen_sft.py:65`). KL from the master (`kl_maths_all`) against KL from the file: gold 0.0037330 / 0.0037307 at step 2, 0.109474 / 0.109474 at step 5. The fp32 master keeps the updates: its spacing at |w| 0.03 is 4e-9.

**Checked in code, no fault found.**
- Label mask (`qwen_sft.py:246`, packed `:249`): logit p−1 predicts the first response token, the last predicts the end token.
- Accumulation (`:269-275`): summed, divided by the batch's token count; not a mean of means. The mask excludes padding.
- Template: one prompt function, thinking off (`qwen_eval.py:55`, `qwen_anchors.py:49`). End token 151645 is the config's EOS. Cuts at 512: own kept responses have median 167, p99 397 tokens, so about 1% may lose the end token. Facts never split.
- Decay 0, betas (0.9, 0.999) (`qwen_sft.py:189`). Config dropout 0.0, so `model.train()` changes nothing. Trained: only the 196 matrices (`qwen_vpd.py:110`, `qwen_sft.py:178`). The head is tied to the frozen embedding.
- Held-out: gold's 256 disjoint (`job.py:132`); sampled arms' last 64 problems (`job.py:258`, `run.py:330,340`); a1 seeds use disjoint samples (`job.py:260`).

**Smaller things.**
- own passes its data at step 93, so its final fall is partly repeats: 0.070 in steps 51-92, 0.050 after. The null, under one pass, goes 0.075 to 0.071.
- a2's held-out shares facts with training, on a fourth template (`qwen_facts.py:143`). Training rows are packed, reading isolated; a2long's 0.999 shows no visible harm.
- a2's held-out loss at states 11-12 differs by 0.3-0.6 between seeds, where the forward starts to see δ. Those states are unstable.

## 4. Why own and null loss rises then falls
The start of 0.057 is low because samples at temperature 0.7 are easy for the model that made them. W0 is near but not at their optimum, since sharpened samples ask for sharper logits, so a fall is possible in principle. What the records support:

1. Rates of 1e-6 to 3e-6 push all 1.4 billion entries a coherent step along mostly noisy batch gradients.
2. In fp32 that costs loss as its square: sums of rates 1, 3, 6 (×1e-6) read KL 0.0057, 0.049, 0.130.
3. The training forward barely sees it, loss stays at 0.05, and nothing corrects it.
4. Past about step 8 the forward sees δ: training loss rises, gradient norm jumps to 10-19. Training pushes back and fp32 falls: tracker KL 0.27 at step 11, 0.10 by step 25; cos(δ₄, δ₁₀₀) is 0.24.
5. Step 100 keeps a floor proportional to the rate: +0.022 at 1e-5, +0.041 at 2e-5.

So our reading is both: too high a rate for this objective, with rounding making the rise larger and hiding it. A reader bug is unlikely, since the master and the file agree (section 3). Three $0 tests tell these apart:
- *Read own state 4 on one set of rows three ways:* fp32 (0.441 known), fp32 master under autocast, fp32 forward on bf16-rounded weights. Rounding predicts about 0.05-0.07 for the last two; about 0.44 rules it out.
- *Run 20 steps with an fp32 forward.* Rate alone predicts the rise stays and training loss shows it. Rounding alone predicts it shrinks.
- *Lower the rate, forward unchanged.* A noise floor predicts the step-100 excess scales down with the rate.

## 5. The minimal $0 rate check (not written, not run)
One L40, `debug` QoS, 30-minute jobs, 44 GiB. From `run.log`: about 5.6 s per step with a read every 5 steps (own_s0 steps 30-40).

**Job A, no training, about 15 min.** Three readers as above on own_s0 2, 3, 4, 100; gold_s0 3, 5, 100; a1_s0 4, 8, 100 (about 15 s per reading). Add fp32 generation at a1_s0 8 and 19 and gold_s0 100. Expected: own state 4 under autocast 0.05-0.07; step 100 all three within 0.03.

**Job B, 100 steps each, same warm-up and cosine, about 12-15 min each.**

| arm | why | expected if the audit is right |
|---|---|---|
| own, 3e-6 | the task where the rate looks wrong | excess at 100 under 0.01; accuracy gain at least 0.03 |
| own, 1e-6 | floor of the grid | no rise; loss may at last fall; accuracy gain may be lost |
| own, 3e-6, fp32 forward | separates rate from rounding | training loss shows any rise from step 2 |
| a1, 3e-6 | does a synthetic task need 1e-5? | template at least 0.8 by step 100, smaller δW norm and maths KL |

Readouts: training loss, δW norm, gradient norm every step; fp32 and autocast held-out every 10; 16-row maths KL every 5; at 100, full-test paired accuracy for own, template rate for a1. No a2 arm: its issue is step count. The fp32 forward may pass 44 GiB at micro 8 (bf16 peak 40.1 GB), so start at micro 4. Total about 70 min, $0.

## Not run
- Any forward pass: rounding's effect on loss rests on training records against held-out records, on different rows.
- Autocast on CUDA: shown on CPU; that CUDA casts Linear weights too is from memory of the PyTorch list.
- Rounding counts on the other 175 matrices. fp32 generation at any state. Per-position reading of a1's gap.
- Held-out readings between saved states; gold has none between steps 9 and 100. LLaMA-Factory's setting, or anything outside this repo. E40 step 10, which `experiments/qwen_reading` reads.
