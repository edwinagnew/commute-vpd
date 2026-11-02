# E61 so far: do the Qwen fine-tunes' readings see what training saw?

*2026-11-02. Experiment `experiments/qwen_recipe/`, stage A and the five checks. No training has run in E61 yet. Notebook: `notebooks/E61_qwen_recipe.ipynb`. Plan: `reports/plans/2026-11-02_plan_e61-direct-share-size.md`.*

We read saved Qwen fine-tunes three ways and ran five checks an audit asked for, with no new training. Training rounded its weights and could not see its first small steps, while every reading used full precision; final states agree well enough to keep. So the small states we want to read need retraining before they can bear on whether training teaches something new, and the one reading that beats random changes may still only measure size.

The learning rates the user asked about have not been tested: no fine-tune has run at a second rate on a synthetic task.

## What we were trying to test

The user added to the choice of E61: "make sure you have fine-tuned the synthetic tasks properly - e.g. they actually learn and youre using the right learning rates and so on". An audit of saved files (`reports/reviews/2026-11-02_review_qwen-finetune-recipes.md`) found no fault in the loss, the data or the held-out rows. It did find that training records and held-out readings of the same state disagreed by a factor of eight, and it named one test.

**The rounding.** Training keeps full-precision master weights and runs its forward under bf16 autocast, which rounds each weight to a grid before use. The checkpoint is stored in bf16, so the original model W0 sits exactly on that grid: rounding changed 0 of its entries over all 196 matrices. An update smaller than half a grid step rounds back to W0. Half a step is about 6e-5 at the median weight of 0.022. Every reader of a saved state, in E44, E45's KL and held-out loss, and E57, used full precision.

**Stage A.** No training. On each state we vary the reader and hold the state and the rows fixed. The quantity is held-out loss in nats per token for the fine-tunes on the model's own answers and on human GSM8K solutions. For the template fine-tune it is the KL from the prompted original model to the unprompted state, in nats per answer position. The three readers:

- *Full precision:* full-precision weights, no autocast. How E45's held-out was read.
- *Autocast:* the same weights under bf16 autocast. The forward of training.
- *Rounded:* weights rounded to bf16 and back, then a full-precision forward. This isolates weight rounding from activation precision.

The unchanged model is the do-nothing reference and the control: on it rounded must equal full precision, and full precision must reproduce E45's record. It did both.

**The five checks.** The pipeline audit (`reports/reviews/2026-11-02_review_qwen-pipeline-audit.md`) named things never compared. Each check ran on saved states with a rule filed first, in `results/predictions_checks.json`. Their symbols, each a mean KL per position from the state's clean logits to a masked forward with Δ off:

- K0: W0 against its own masked version, with W0's masks.
- K00: the original subcomponents, with W0's masks.
- K01: the original subcomponents, with masks Γ computes on the changed model.
- K10, K11: the same two, with the least-squares split applied to the subcomponents.
- Size as E57 measured it: K00 − K0.
- Direct share: (K00 − K10) / (K00 − K11). It reads 0 if changing the subcomponents removes nothing, 1 if it removes all that can be removed.
- Re-gating share: (K00 − K01) / (K00 − K11). The same with recomputing the masks.

## What we expected, and why

If rounding was hiding the update, the audit expected the own-answers state after four steps to read 0.05 to 0.07 under autocast and on rounded weights. Full precision had it on record at 0.441. A reading of 0.30 or more would have ruled rounding out. At step 100 it expected agreement, because the update is by then far larger than the grid.

For the checks: no floor from Γ's bf16 cast at a tiny change; K0 unknown on unpacked rows; no direction filed for the direct share with truly original masks, since the audit's hand estimate said the mask path might match the direct path. The own-answers gain was unknown at full precision, against +0.0525 on rounded weights.

## What we observed

### Does rounding hide the early update from training?

![Three readers per state](../../experiments/qwen_recipe/figures/readers.png)

**Reading it.** Yes, for the states read. The own-answers fine-tune after four steps reads 0.441 at full precision, 0.0549 under autocast and 0.0548 on rounded weights. Both are under the filed line of 0.10, so the filed reading is that weight rounding is enough. Autocast and rounded stay within 0.002 at every state, on all three fine-tunes. At step 100 the excess gap over W0's is 0.011, −0.001 and 0.005, under the filed 0.03.

| fine-tune | steps | full precision | autocast | rounded | maths KL, full precision | maths KL, autocast |
|---|---|---:|---:|---:|---:|---:|
| own answers | 0 | 0.0566 | 0.0568 | 0.0566 | | |
| | 2 | 0.0647 | 0.0566 | 0.0563 | 0.0056 | 0.0007 |
| | 3 | 0.2166 | 0.0556 | 0.0554 | 0.0516 | 0.0008 |
| | 4 | 0.4410 | 0.0549 | 0.0548 | 0.1397 | 0.0018 |
| | 100 | 0.0786 | 0.0672 | 0.0669 | 0.0794 | 0.0480 |
| human solutions | 0 | 2.4952 | 2.4966 | 2.4952 | | |
| | 3 | 1.3070 | 2.3935 | 2.3925 | 0.0211 | 0.0008 |
| | 5 | 0.9264 | 1.5942 | 1.5924 | 0.1102 | 0.0087 |
| | 100 | 0.5120 | 0.5122 | 0.5120 | 0.6886 | 0.6465 |
| template | 0 | 0.9615 | 0.9615 | 0.9615 | | |
| | 4 | 0.3039 | 0.7934 | 0.7921 | 0.0465 | 0.0017 |
| | 8 | 0.2472 | 0.2212 | 0.2200 | 0.1963 | 0.0618 |
| | 100 | 0.0554 | 0.0502 | 0.0494 | 0.3205 | 0.2851 |

Maths KL is from W0 on the first 16 maths rows; no rule reads it.

**What "agree at step 100" covers.** It covers the filed quantity on each fine-tune's own rows. On the own-answers run the tolerance of 0.03 is larger than the whole effect: its loss sits 0.022 above W0 at full precision and 0.011 under autocast. On maths rows at step 100 the gap is larger in proportion: 0.079 against 0.048 on that run, 0.32 against 0.29 on the template, 0.69 against 0.65 on human solutions. And the two forms are still different models, as cosine 0.74 to 0.93 between rounded and full updates in the audit shows.

### Do the two versions of a state give the same answers?

![Accuracy of the two versions of a state](../../experiments/qwen_recipe/figures/answers.png)

**Reading it.** Not early. The template fine-tune after 8 steps gets 0.133 correct on full-precision weights and 0.703 on rounded ones, on the same 256 problems. The unchanged model at full precision gets 0.750. The saved answers come from a whole bf16 model, so this comparison mixes rounded weights with bf16 activations. By 19 steps the forms agree within noise and both follow the template on 0.953 of answers. This is a description, with no filed verdict.

| state | problems | form | strict, all | strict, answers that fit | template rate | mean answer tokens | truncated |
|---|---:|---|---:|---:|---:|---:|---:|
| template, 8 | 256 | full precision | 0.133 | 0.147 | 0 | 75.5 | 0.098 |
| | | bf16 | 0.703 | 0.705 | 0 | 132.8 | 0.004 |
| template, 19 | 64 | full precision | 0.578 | 0.597 | 0.953 | 132.2 | 0.031 |
| | | bf16 | 0.609 | 0.609 | 0.953 | 133.5 | 0 |
| human solutions, 100 | 64 | full precision | 0.578 | 0.578 | | 95.2 | 0 |
| | | bf16 | 0.578 | 0.587 | | 97.1 | 0.016 |

At 64 problems the standard error is about 0.06. The unchanged model at full precision reads 0.750 on those 64.

**Cost of a full-precision forward.** One gradient pass took 14.7 s at micro-batch 8, peak 24.8 GiB, and 14.0 s at micro-batch 4, peak 20.9 GiB, without Adam's 10.5 GiB.

## Which earlier claims this touches

This section is in effect the review note the plan promised; there is no separate file. Sources: the audit, and section 4 of `reports/plans/2026-11-02_advice_e61-recipe-FABLE.md`. "Measured" means a number here moved. "Checked" means read in code and cards. "Inferred" means argued and not measured. Filed predictions are not edited.

| claim | what happens to it | how known |
|---|---|---|
| E40 step 1: 1.57 nats of gold drop, maths KL 0.106 | **Stands.** Adam's first step uses the gradient at W0 whatever the forward's precision. Its activations were bf16, a precision difference and not blindness | checked |
| E40 step 10, no warm-up, 1e-5 constant | **Stands as a reading; description changes** to "ten steps, the first five or so blind", from the update growing by about 1e-5 a step | inferred: this run was not read three ways |
| E44's verdict on three bad placements | **Stands.** It compares placements of one fixed update | checked |
| E57's rows: direct and re-gating shares | **Stand as readings** of real full-precision states; what changes is how the states came about | human-solutions step 100 measured; E40 and invented facts not read three ways |
| E45's early "learned" flags, steps 2 to 12 | **Do not stand** as behaviour. Generation was on rounded weights, and the template answers at step 8 differ fivefold | measured at template step 8; inferred at every other early step |
| E45's "invented facts at step 12, recall 2%" | **Carries the same caution** | inferred: no full-precision generation of invented facts |
| Own-answers fine-tune "never learned" | **Withdrawn as a statement about learning.** The filed rule, drop of at least 0.02, stays scored as filed: never met | measured: +0.037 in accuracy at full precision |
| Unfiltered-samples fine-tune "never learned" | **Not settled.** It has no full-precision reading | not run |
| Template and invented facts learned by their final states | **Stands** for the template: 0.953 at step 19 in both forms. Invented facts were not read three ways | template measured; invented facts inferred from loss |
| Option A's step-9 state of the human-solutions run | **Replaced in the plan:** five blind steps then four seeing ones, a path no fixed recipe repeats | inferred |
| The 67M line, same autocast over full-precision weights | **Looks unaffected.** The checkpoint is fp32, so W0 is off the grid and rounding adds noise with no dead zone | inferred: weight magnitudes, flip fraction, the minGPT checkpoint and teacher training's forward not checked |

**What this does not say.** Rounding does not explain why the re-gating share cannot separate changes. The random changes were never trained, and they read the same as the fine-tunes. No rerun of E40, E44 or E57 is needed on these grounds.

## The five checks

Each check varies one thing, holds the rows and the state fixed, and has the rule as filed.

| check | filed rule | read | result |
|---|---|---|---|
| Γ's cast at a tiny change: W0 + 1e-3 of E40's update, 128 rows | K01 − K0 at most 0.005 nats, and under 500 switched on | −3e-5; 133 on, 133 off. E40 itself: 0.035; 5,730 on | no floor worth reading |
| Row format: 128 whole problems against 128 packed rows of 256; W0 fixed | K0 ratio within 1.5 either way | 0.0563 against 0.0696, ratio 0.81 | same reading |
| Direct share, truly original masks; 16 rows, four changes | within 0.01 on all four, same order | largest gap 0.0022, same order | E57's stand |
| Three sizes; six changes, 128 rows of 256 | a table, no threshold | below | no verdict |
| Own-answers gain at full precision; 1,319 problems | paired gain at least 0.03, and two standard errors above 0 | +0.0366, se 0.0107 | survives |

**Tiny change, in detail.** K0 0.06961, K00 0.06982, K01 0.06958. This compares bf16 Γ at a small change with bf16 Γ at W0, so it reads the cast's jitter and not its bias. Full-precision Γ against bf16 Γ at a large change has not run; the rule sent nothing further.

**Row format, in detail.** Packed, first 16 rows: 0.0656, the filed control. Whole problems: 39,009 positions. Row 0 reads 0.0769 at 256 tokens, 0.0586 at 512, and 0.0769 over the first 256 of the 512, so the reading is causal as expected. The problems differ between the two sets, and neither is the fit's data, whose eval split is not on disk. So this reads format, not faithfulness to the fit.

### Does keeping the masks truly original change the direct share?

![Direct share with original masks and with E57's](../../experiments/qwen_recipe/figures/masks.png)

**Reading it.** No. E57 computed K10's masks through the moved read vectors, though its prose said the masks were kept as original. Masks from the unmoved subcomponents give shares within 0.0022 of E57's, in the same order, on 16 rows. Job S will read the original-mask version on all 128 rows, as the user chose.

| change | E57's masks | truly original | gap |
|---|---:|---:|---:|
| human solutions, 100 steps | 0.0912 | 0.0890 | −0.0022 |
| human solutions, 10 steps (E40) | 0.0636 | 0.0621 | −0.0015 |
| invented facts, seed 0 | 0.0143 | 0.0142 | −0.0001 |
| Gaussian | 0.0215 | 0.0223 | +0.0009 |

### Does the direct share rise with size? It depends on the measure

![Direct share against three measures of size](../../experiments/qwen_recipe/figures/sizes.png)

**Reading it.** Open. Holding the direct share fixed and changing only the measure of size gives two stories. On K00 − K0 and on the unmasked KL with the changed model as reference, the two maths fine-tunes are 1.5 to 2 times the size of anything else, so their higher share could be size. On the unmasked KL with the original model as reference, the 100-step run leads by a quarter. There the ten-step run sits below invented facts with five times its direct share. No run has varied size with the kind of change held fixed, and neither axis is shown to be right. Direct share against size remains open. Invented facts' value, below both random changes, is unexplained.

| change | K00 − K0 | KL, changed model as reference | KL, original model as reference | direct share, 128 rows [95% row bootstrap] |
|---|---:|---:|---:|---|
| human solutions, 100 steps | 1.444 | 1.741 | 0.806 | 0.092 [0.090, 0.093] |
| human solutions, 10 steps (E40) | 1.304 | 1.635 | 0.584 | 0.063 [0.062, 0.064] |
| Gaussian | 0.857 | 0.868 | 0.526 | 0.022 [0.020, 0.024] |
| invented facts, seed 0 | 0.638 | 0.682 | 0.647 | 0.012 [0.010, 0.013] |
| invented facts, seed 1 | 0.628 | 0.673 | 0.615 | 0.012 [0.011, 0.013] |
| alive subcomponents | 0.625 | 0.579 | 0.481 | 0.029 [0.026, 0.032] |

Direct shares here use E57's masks; the 128-row original-mask reading has not run.

### Does the own-answers gain survive full precision?

![Accuracy gain, full-precision against rounded weights](../../experiments/qwen_recipe/figures/own_gain.png)

**Reading it.** Yes, and it is smaller. Paired over the 1,228 problems whose answers fit at both states, the gain is +0.0366, se 0.0107, against +0.0525, se 0.0102, on rounded weights. Over all 1,319 it is +0.0402 against +0.0546. The state read was trained under rounding, so this is not what a full-precision run would reach.

**Own gain against the unfiltered run is not settled at full precision.** On rounded weights the unfiltered fine-tune gains +0.0251, se 0.0107, and own minus unfiltered is +0.0253, se 0.0097. It has no full-precision reading.

**The unchanged model read both ways.** 715 of 1,319 answers differ and 75 grades flip, with paired difference 0, se 0.0061, on the fit rule. So the two forms of W0 give the same accuracy, on different answers.

## Settled from saved files

### Why does every change seem to overlap every reference set?

![Overlap, flat and matched on closeness to the line](../../experiments/qwen_recipe/figures/near_threshold.png)

**Reading it.** Mostly closeness to the line. A subcomponent counts as switched on when its mask passes 0.01 on at least 1% of positions. Most of those a change switches on were already on for 0.5% to 1% of positions at W0. Matched on that closeness, overlap with the thinking set is about chance. The filed overlap null then cannot fail, since any change clears it. Bins and site strata were not combined, so the flat folds here read above E57's three to four.

| change | share just under the line | thinking set: flat, matched | switched on | switched off |
|---|---:|---:|---:|---:|
| human solutions, 100 steps | 0.85 | 4.2, 0.97 | 7,916 | 5,757 |
| human solutions, 10 steps | 0.90 | 4.5, 0.99 | 5,730 | 6,152 |
| invented facts, seed 0 | 0.95 | 5.9, 1.26 | 4,606 | 3,208 |
| invented facts, seed 1 | 0.96 | 6.2, 1.30 | 4,402 | 3,115 |
| Gaussian | 0.85 | 5.2, 1.18 | 6,961 | 3,753 |
| alive subcomponents | 0.91 | 5.5, 1.21 | 5,170 | 2,864 |

The two invented-facts seeds against each other, the overlap's control, fall from 72 to 4.8. Any two changes read 2.0 to 3.9 when matched. On the format-prompt set invented facts stay at 1.7 to 1.8 against 1.3 to 1.6 for the rest, off 135 to 165 members.

### Could the re-gating share have read low?

![Re-gating share and its algebraic floor](../../experiments/qwen_recipe/figures/regating_floor.png)

**Reading it.** No. When K11 is at or above K0, the denominator is at most K00 − K0, so the share is at least one minus (K01 − K0) / (K00 − K0). That floor is past the filed line of 0.8 for all six changes, whatever the split does. So "no teeth" stands: at this scale the share cannot separate kinds of change. K01 near K0 is measured, not forced, and the audit found no leak in the hooks.

| change | K0 | K00 | K01 | K11 | floor | re-gating share |
|---|---:|---:|---:|---:|---:|---:|
| human solutions, 100 steps | 0.0696 | 1.5134 | 0.1403 | 0.1396 | 0.951 | 0.9995 |
| human solutions, 10 steps | 0.0696 | 1.3738 | 0.1044 | 0.1034 | 0.973 | 0.9993 |
| invented facts, seed 0 | 0.0696 | 0.7079 | 0.1038 | 0.0994 | 0.947 | 0.9927 |
| invented facts, seed 1 | 0.0696 | 0.6980 | 0.0957 | 0.0927 | 0.958 | 0.9950 |
| Gaussian | 0.0696 | 0.9266 | 0.1289 | 0.1242 | 0.931 | 0.9940 |
| alive subcomponents | 0.0696 | 0.6943 | 0.1173 | 0.1131 | 0.924 | 0.9928 |

## What we learned

Rounding is confirmed for the readings listed, and not tested elsewhere. Early states of the fine-tunes are not what their labels said, so before any small update is compared through the decomposition, it needs retraining with the readers' forward. Final states are usable as readings. `GOALS.md` asks whether a simple map fails because training teaches something new, and this model still has no reading that separates the two cases. The direct share survives its checks, and it may still only measure size.

## How compelling is the evidence

The filed rounding verdict rests on one state, one run, one set of rows. Nine other non-W0 states read the same way, and the effect is eightfold at the filed state. Each state is one seed. Teacher-forced loss is blind to answers, as template step 8 shows. The step-100 rule compares losses, so two different models with equal loss pass it. The original-mask check is 16 rows and four changes. Sizes are maths rows only, so each fine-tune's size on its own task is not here.

## Surprises

- **Agreement in KL hid a gap in answers.** At template step 8 the KL reads 0.247 at full precision and 0.221 under autocast, while answers read 0.133 against 0.703 correct. Not tested why.
- **Step 100 is less alike on maths rows than on the filed rows.** See the table above. No rule reads this, and its tolerance is larger than the own-answers run's whole rise.
- **E57's size tracks the unmasked KL with the changed model as reference,** at a ratio of 0.93 to 1.25 over six changes. Both have the changed model as reference. Nobody filed this.
- **A filed expected value missed.** At the tiny change K00 − K0 read 2.1e-4; the filing said under 1e-4 by a square law. A KL to the masked model can carry a first-order term; that is an argument, not a test. The rule does not read this number.
- **Switched on and switched off are both exactly 133.** The two counts are computed separately in `checks.py`. Coincidence is not excluded; we did not look further.

### Numbers on file that differ from the reviews

- Fable's section 1 gives the step-100 excess for human solutions as 0.0002. That is the raw gap; the filed excess in `A.json` is −0.0011.
- The audit's "2.1 to 2.5" K00 − K0 per unit of KL for maths fine-tunes, and "2.5-fold", use 128 rows of 512. On 128 rows of 256 they read 1.8 to 2.2, and 2.3-fold.
- The audit gives template step 8 as 0.15 against 0.70, on the rule that drops unfit answers. Fable's 0.133 against 0.703 is all problems. Both are right; this report uses all problems.
- The plan's amendment says no filed check missed. That is true of rules; the tiny check's expected value above did miss.

## What next

E61 Parts 1 and 2 of `reports/plans/2026-11-02_plan_e61-direct-share-size.md`: fine-tunes with a full-precision forward at measured rates, then the direct share against size.

## Departures from the literature

- **Mixed precision.** Standard training uses a bf16 autocast forward, from memory and not checked in the repo. The saved states used it. Part 1 replaces it with a full-precision forward, since the first steps are what we read.
- **Rate and warm-up.** Four files cite LLaMA-Factory's full-SFT setting, 1e-5, cosine, warm-up ratio 0.1; none holds a copy. A ratio written for multi-epoch runs became 10 steps of 100. Treat it as unsourced for this length.
- **Masks from another model.** VPD computes masks from the model being decomposed. K00 and K10 use the original model's. The check here removes the moved read vectors from that path, nothing else.
- **Direct share, re-gating share, K00 − K0 as size, and closeness bins** are this project's; no source defines them.
- **Switched on** adds "on at least 1% of positions" to VPD's alive threshold of a mask above 0.01.
- **KL per token, teacher-forced on fixed rows.** RL's Razor measures KL to the start on the model's own samples of the new task.
- **Row format.** The fit read documents with per-document attention and position resets; the packed rows have neither. The check above reads K0 on both.
- **Bootstrap** intervals are plain percentile intervals over row resamples.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| R1 autocast | hides own step 4 | at most 0.10 confirmed, at least 0.30 ruled out | 0.0549 | confirmed |
| R1 rounded | same | same | 0.0548 | confirmed |
| R2 own | readers agree at 100 | excess at most 0.03 | 0.0112 | holds |
| R2 human solutions | same | same | −0.0011 | holds |
| R2 template | same | same | 0.0052 | holds |
| R3 | generation at full precision | description, no verdict | table above | none |
| tiny | no floor from Γ's cast | gap at most 0.005, under 500 on | −3.0e-5, 133 | passes |
| k0 | row format does not change K0 | ratio within 1.5 | 0.808 | passes |
| sizes | three measures | table | table above | none |
| gen | own gain survives | at least 0.03, two se above 0 | +0.0366, se 0.0107 | passes |
| k10 | mask path does not change the reading | within 0.01, same order | 0.0022, same | passes |

E45's filed predictions stand as filed.

## Not run

- Any training with a full-precision forward; so whether it reaches the same states at step 100.
- Any rate grid on a synthetic task. Whether the own-answers rate is too high, which the audit reads from training records, remains untested.
- The three readers on E40, on invented facts, or on the unfiltered run.
- Full-precision generation of invented facts at any state, or of the unfiltered run or twice-the-rate run.
- Full-precision generation from early states other than template step 8.
- The direct share with truly original masks on 128 rows, on the alive-subcomponents change, or on invented facts seed 1.
- Random changes scaled to the largest fine-tune's size; a smaller state of that fine-tune at a random change's size.
- Full-precision Γ against bf16 Γ directly at a large change.
- The template fine-tune through the decomposition.
- Reference fine-tunes compared at equal size.
- The 67M's weight magnitudes and flip fraction, the minGPT checkpoint's dtype, teacher training's forward under TF32.
- Overlap with bins and site strata together.

## Provenance

L40, MATS cluster, $0. Stage A: job 48367, `results/A.json`. Checks, `results/checks_*.json`, code `experiments/qwen_recipe/checks.py`: tiny 48473, row format 48494, sizes 48506, generation 48538 and 48624, masks 48645, 48675, 48721 and 48747. The full log is `results/run.log`. Rules: `results/predictions.json`, `results/predictions_checks.json`. The closeness table is the audit's `near_threshold.txt`, copied to `results/`. Other bases: `experiments/qwen_reading/results/C_*.json`, `C0.json`, `boot_direct.json`. Figures: `experiments/qwen_recipe/plot.py`.
