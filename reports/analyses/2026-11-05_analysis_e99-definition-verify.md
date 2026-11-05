# E99: Does the net share at a located pair read 1 when a fine-tune is forced through the pair, and 0 when it is forbidden?

*Definition-lead, 2026-11-05. Stages 0-3 of `reports/plans/2026-11-05_plan_e99-refine-verify-definition.md`, with Amendment 1 for stage 3. Jobs 52787 and 52807, MATS `debug`, L40, $0. Status: provisional, fast lane, no independent check. Notebook `notebooks/E99_definition_verify.ipynb`.*

We trained small fine-tunes on the 67M model whose route was fixed by construction, either forced through a located pair of attention heads or forbidden from it, and asked whether the proposed net-share rule reads them correctly. On the copying task it does, reading all of the gain through the pair when forced and none when forbidden; on the lowercase task the forced fine-tune could not reach the native gain, so the rule has no positive control there. For `GOALS.md` this means the definition can give a verdict through the copying pair only, and the tasks we called strengthening stay without a reading of "existing".

**Limit on every claim.** A new circuit built from old parts reads as reuse (`decisions.md` row 121). The forced and forbidden fine-tunes are ground truth for "through the pair", not for "reuses the old function".

## The answer

On the copying pair, the net share works as the definition needs. A fine-tune forced through the pair reads all of its gain there, and one trained with the pair switched off reads none, inside its random band. On the lowercase pair it does not. The forced fine-tune reads all its gain there but reaches only about two thirds of the native gain, so the filed fallback applies and lowercase stays "does not apply". The forbidden lowercase fine-tune read far below its band, against the filed expectation.

## What we were trying to test

The definition under test is `decisions.md` rows 81 and 115: an update reuses a capability C when its behaviour runs through what C computes, read by removing C. Since the user's batch decision of 2026-11-05, a label needs a located mechanism. E97 located two head pairs: heads (2,3) and (2,4) on the copy task, and heads (1,1) and (1,2) on the lowercase task. Fable proposed reading a located pair through a **net share**:

$$\text{net} = \frac{\text{state's fall with the pair off} - \text{original model's fall with the pair off}}{G},$$

where $G$ is the fine-tune's mean gain in the log-odds of its target over the original model, and "off" sets the pair's outputs to the original model's mean over the same inputs. A net share of 1 means all of the update's increment vanishes with the pair; 0 means none of it does; a negative value means the increment is larger with the pair off than with it on. Each state's **band** is the 95th percentile of the absolute net share over 20 random head pairs drawn outside the pair, on the same state and inputs.

The question is whether this quantity reads the right answer when the answer is known. Two constructions fix it:

- **Forced**: train with every weight outside the pair's query, key and value rows reset to the original model after every step, output matrices included. The state with the pair off then equals the original model with the pair off, so the net share is 1 by construction. This is the positive control; it can fail only through a code error or through the pair being unable to learn the task.
- **Forbidden**: train with the pair held at the original model's mean, and keep the pair's own weights at the original model's. Read the result with the pair restored. The update cannot have used the pair, so the net share should sit within the band.

Each was trained until its gain, read in its training condition on a check subset, came within 2% of the native fine-tune's gain, with a cap at ten times the native's steps. Copy used E62's recipe, native 30 steps, read on E62's copy windows; the spared control is E62's lookup fine-tune at the copying pair. Lowercase used E53's recipe, native 300 steps, read on whole rows; its control is E53's cipher fine-tune. Two seeds each.

Stage 0 also tested the second admissibility clause of the proposed rule for fixed-answer tasks: the pair's function in the original model must depend on the input, on inputs the task never trained on. The read is how far setting the lowercase pair to its mean moves the original model's lowercase-minus-capitalised log-odds on cased rows, against the same 20 random pairs.

## What we expected, and why

- Stage 0: the original model's falls and the native states' gains reproduce E97's numbers exactly, since the code and inputs are the same; the lookup is spared at the copying pair.
- Clause (ii): the lowercase pair moves the case leaning on cased rows. As coded, the rule also asked for a move on lowercased rows, which the plan did not ask.
- Forced copy and forced lowercase: net share 1 within 0.05, whether or not the gain reaches the native's. The filing says what a cap means on lowercase: R-net has no positive control there and falls back to "does not apply".
- Forbidden copy: within band, with the gain with the pair restored within 10% of the native's.
- Forbidden lowercase: within band, and learning as fast as the native.

## What we observed

### Do the controls reproduce?

Yes, exactly. The original model's falls with each pair off equal E97's to the last digit on all four sites, and every native state's gain and own share reproduce. The lookup fine-tune's net share at the copying pair is 0.014 and 0.003, under the spared bar of 0.2.

| Site, pair | Original model's fall, log-odds | E97 | Twice its row SE |
|---|---|---|---|
| Copy windows, (2,3)+(2,4) | 1.876 | 1.876 | 0.178 |
| Lookup windows, (2,3)+(2,4) | -0.100 | -0.100 | 0.077 |
| Lowercase rows, (1,1)+(1,2) | 1.102 | 1.102 | 0.107 |
| Cipher rows, (1,1)+(1,2) | -0.518 | -0.518 | 0.147 |

| Native state | G | Fall, pair off | Own share | Net share | Band | G with the pair off in both |
|---|---|---|---|---|---|---|
| Copy s0 | 12.65 | 10.96 | 0.867 | 0.719 | 0.153 | 3.56 |
| Copy s1 | 12.64 | 11.12 | 0.880 | 0.731 | 0.155 | 3.39 |
| Lookup s0 | 15.36 | 0.11 | 0.007 | 0.014 | 0.124 | 15.16 |
| Lookup s1 | 14.99 | -0.06 | -0.004 | 0.003 | 0.109 | 14.95 |
| Lowercase s0 | 1.131 | 0.902 | 0.798 | -0.177 | 0.434 | 1.33 |
| Lowercase s1 | 1.089 | 0.929 | 0.853 | -0.159 | 0.468 | 1.26 |
| Cipher s0 | 3.714 | 0.263 | 0.071 | 0.210 | 0.189 | 2.93 |
| Cipher s1 | 3.659 | 0.252 | 0.069 | 0.210 | 0.189 | 2.89 |

The cipher control at the lowercase pair reads 0.21 net on both seeds, just above its band of 0.19. Its own-terms share is 0.07. The original model rises 0.52 when the pair is off on cipher rows, and the net share adds that rise back.

### Does the lowercase pair's function depend on the input?

![clause (ii)](../../experiments/definition_verify/figures/clause_ii.png)

**Reading it.** On cased rows, yes; on lowercased rows, not distinguishably from random pairs. Setting the pair to its mean moves the original model's case leaning on cased rows by -1.85, beyond the band of 0.76 and beyond all 20 random pairs. On lowercased rows the move is 0.70 against a band of 1.16. The filed rule asked for both, so S0-ii missed as coded and stage 3 was skipped in job 52787.

The plan words the clause on cased rows only. Amendment 1, the lead's call dated 2026-11-05, filed that wording in `results/predictions_amend1.json` before running, with the lowercased-row reading kept beside. It passes on the stored numbers, and stage 3 ran in job 52807.

What the clause can and cannot say: all 20 random-pair moves on cased rows are negative, from -0.005 to -1.23. Removing almost any two heads makes the original model lean further toward the lowercase form on cased text. The pair moves the gap in that same direction, further than any random pair. So the clause shows the pair matters more than a random pair to this one gap. It does not show the pair does something specific to case. The pair moves the gap in opposite directions on the two row types: -1.85 on cased rows, +0.70 on lowercased rows.

### Does the net share read 1 when forced and 0 when forbidden?

![on and off](../../experiments/definition_verify/figures/on_off.png)

![net share](../../experiments/definition_verify/figures/net_share.png)

**Reading it.** On copy, yes on both seeds; on lowercase, only the forced half reads as filed. The forced fine-tunes read net 1.00 on both tasks and both seeds, with their mean log-odds at the pair off equal to the original model's. The forbidden copy fine-tunes read -0.05 against bands of 0.14 and 0.17. Their gain with the pair restored is 0.94 of the native's, within the 10% bar. With all three copying heads that E62 found off, (2,3), (2,4) and (3,4), the forbidden copy fine-tune loses 0.38 and 0.36 net. So part of its route runs through head (3,4), the other copying head. No band was drawn for three-head sets, so this is a reading beside, not a verdict. The forbidden lowercase fine-tunes read -0.96 and -0.85, far below their bands of 0.70 and 0.59.

| Arm, seed | Steps | Status | G | G / native | Fall | Net | Band | Reading |
|---|---|---|---|---|---|---|---|---|
| Forced copy s0 | 300 | cap | 10.40 | 0.822 | 12.28 | 1.000 | 0.166 | net 1 |
| Forced copy s1 | 300 | cap | 10.40 | 0.823 | 12.27 | 1.000 | 0.167 | net 1 |
| Forbidden copy s0 | 48 | matched | 11.90 | 0.941 | 1.28 | -0.050 | 0.139 | within band |
| Forbidden copy s1 | 58 | matched | 11.84 | 0.937 | 1.26 | -0.052 | 0.169 | within band |
| Forced lowercase s0 | 3000 | cap | 0.689 | 0.610 | 1.791 | 1.000 | 0.508 | net 1 |
| Forced lowercase s1 | 3000 | cap | 0.695 | 0.639 | 1.797 | 1.000 | 0.503 | net 1 |
| Forbidden lowercase s0 | 5 | past target | 0.532 | 0.470 | 0.592 | -0.959 | 0.698 | below band |
| Forbidden lowercase s1 | 5 | past target | 0.567 | 0.520 | 0.620 | -0.851 | 0.590 | below band |

Beside: the forbidden copy fine-tune's gain with the pair off in both models is 12.49 and 12.46. The forced copy states keep the original model's second-copy accuracy on E52's induction read, 0.877 and 0.878 against 0.825. That read shows no sign that the pair's copying was replaced by something else; it is blind to added functions that leave copying intact.

### Can the forced fine-tunes reach the native gain?

![gain curves](../../experiments/definition_verify/figures/gain_curves.png)

**Reading it.** No, on either task. Allowed only the two heads' query, key and value rows, the copy fine-tune reaches 0.82 of the native gain by the cap of 300 steps, still rising slowly. The lowercase fine-tune reaches 0.61 and 0.64 on whole rows by 3,000 steps, and its curve has nearly flattened. On copy the cap does not touch the filed rule: net 1 holds. On lowercase the filing says the cap means R-net has no positive control, and lowercase falls back to "does not apply".

### Why does the forbidden lowercase fine-tune read far below its band?

This is the anomaly the plan did not expect, and it is a property of how the arm was stopped, as far as the code and numbers show.

What each number measures, from `run.py` `train_arm` and `arms.Matcher`:

- The **target** is the native fine-tune's gain on the first 32 lowercase rows, read against the intact original model: 1.15 and 1.05.
- The **check** reads the arm's gain on the same 32 rows in its training condition: the arm with the pair at its mean, against the original model also with the pair at its mean. At the first check, step 5, it read 1.23 and 1.19, past the target, and training stopped. Both seeds overshot the 2% band, so the status is "nearest check outside the band of the target", not "matched".
- The **reported G** is read on all 256 whole rows with the pair restored, against the intact original model: 0.53 and 0.57, which is 0.47 and 0.52 of the native.

So the stop rule compares two gains measured from different references. On lowercase, removing the pair costs the original model 1.10, about as much as the whole target. On copy the same cost is 1.88 against a target of 12.6, which is why the forbidden copy arm was not affected. The training loss shows the size of the starting deficit: on the same first batch, the original model's loss is 3.31 and 3.10 with the pair on, and 4.05 and 3.82 with it at its mean.

Where each model sits, in mean log-odds of the target relative to the intact original model, seed 0 (seed 1 within 0.07):

| Model | Pair on | Pair off |
|---|---|---|
| Original model | 0 | -1.10 |
| Native fine-tune | 1.13 | 0.23 |
| Forbidden fine-tune | 0.53 | -0.06 |
| Forced fine-tune | 0.69 | -1.10 |

The restored forbidden model is not worse than the ablated one: restoring the pair raises it by 0.59. That rise is about half of what restoring the pair gives the original model, 1.10. After five steps, the forbidden fine-tune with the pair off reads 0.06 below the intact original model.

What a strongly negative net share means here, in plain words: removing the pair costs the forbidden fine-tune about half a log-odds unit less than it costs the original model. That difference is almost as large as the fine-tune's whole gain. Equivalently, the update raises the target by 1.04 when the pair is off in both models, and by only 0.53 when the pair is on. What the update adds does not stack on top of what the pair does. Why it does not stack is not measured. The native lowercase fine-tune shows the same sign, -0.18, inside its band.

What this does to the filed prediction: the rule reads "below band" on both seeds, a miss against "within band". The pace rule reads "as fast" on both seeds, trivially, since the arm stopped at step 5. Restored within 10% is false, 0.47 and 0.52, beside the rule. The reading is about an arm stopped after five steps against a target measured from another reference. It is not a reading of what a forbidden lowercase update looks like at the native's gain, which was not trained.

## What we learned

The net share has a working positive control and a passing forbidden arm on the copying pair: forced reads 1, forbidden reads within band with 94% of the native gain. So R-net can read the copy-against-lookup pair as a verdict, not an assumption, with the granularity limit beside it. On the lowercase pair the forced arm cannot reach the native gain through two heads' query, key and value rows. The filed fallback applies: fixed-answer tasks stay "does not apply" under R-net, as the plan said. That leaves the user's batch decision of 2026-11-05 as it stands: only the copying pair qualifies for a label, and lowercase, emoticon, the ladder and many cues stay "does not apply" with the two behavioural numbers beside. E99 adds a measured reason. The one proposed route to a verdict on those tasks, a net share through a located part, has no positive control on the one task where a part was located.

## How compelling is the evidence

- Two seeds per arm, one copy task and one lowercase task, one model. The seeds agree closely, except the forbidden lowercase net share, -0.96 against -0.85.
- The bands come from one set of 20 random head pairs per pair, seeded once. With 20 draws, the 95th percentile sits between the two largest draws, so a band is about as uncertain as its largest draw.
- The net share is blind to a new function built inside the part, which reads as "through the part", and to rerouting through other heads of the same capability, which reads as "not through". The forbidden copy arm did reroute partly through head (3,4).
- The forced arm's net share is 1 by construction; it shows the code reads the construction correctly, not that a natural fine-tune would read 1.
- The cipher control at the lowercase pair reads just above its band under the net share. That does not change any verdict here, since lowercase falls back anyway.

## Surprises

- **Something we did not predict: the forbidden lowercase fine-tune reads far below its band**, -0.96 and -0.85 against bands of 0.70 and 0.59, where within band was filed. The run-down above shows the stop rule ended training at step 5, when the arm had made up about what removing the pair took from the original model. The restored model then holds half the native gain. Whether an arm trained to the native gain would also read below band is not measured.
- The forced fine-tunes cap on both tasks. On copy at 0.82 of the native gain, which the filing allowed; on lowercase at about 0.6, which triggers the filed fallback.
- Clause (ii) as coded missed on lowercased rows; random pairs move the case leaning there by as much as the pair does. On cased rows every random pair moves the leaning in the pair's direction.
- The forbidden copy fine-tune loses about 0.37 of its gain when head (3,4) is removed together with the pair: rerouting within the copying heads, as Fable warned.

## What next

1. **Stage 4**, under the refined rules. It would add four things:
   - the copy-against-lookup verdict through R-net, now with its positive control;
   - the copy maps closed by the admissibility rule R-adm, with E95's token reader as the reason;
   - per-state bands with E95's paired test on every old row;
   - the class map and its shuffled twin rebuilt, about 11 minutes, and read with the hand-write control.

   The class map is the one old task whose capability is neither its answer nor its input token. If it reads cleanly, the value form has an old-task test; if it fails admissibility at its site, the value form has no admissible site on this model.
2. **If a verdict on lowercase is still wanted**: a forbidden arm whose stop rule reads its gain against the intact original model, or that trains to the native's steps, would say whether the below-band reading survives at the native gain. It would not give R-net a positive control there; only a forced arm that reaches the native gain would.
3. **Propose the rule changes to the user in one batch**, as the plan says: R-net with its two clauses for located parts on copy, the lowercase fallback, the clause (ii) wording on cased rows only.

## Scorecard

| Id | Claim | Rule | Read | Verdict |
|---|---|---|---|---|
| S0 | Original model's falls, natives' G and own share reproduce E97; lookup spared | within twice row SE; G within 0.02; lookup net at most 0.2 | all equal E97; lookup 0.014, 0.003 | held |
| S0-ii | Lowercase pair moves the case gap on cased and lowercased rows | `rules.clause_ii`, strictly above band on both | cased -1.845 vs 0.758 moves; lowercased 0.703 vs 1.161 does not | missed as coded; rule asked more than the plan |
| S0-ii-A1 | Same, cased rows only, as the plan words it | `rules.moves` on stored numbers | -1.845 vs 0.758 | held; stage 3 ran |
| P1 | Forced copy reads net 1 | within 0.05 of 1, both seeds | 1.000, 1.000; cap at 0.82 of native | held |
| P2 | Forbidden copy within band; restored gain within 10% | `forbidden_reading`, `restored_ok` | -0.050 vs 0.139, -0.052 vs 0.169; 0.941, 0.937 | held |
| P3 | Forced lowercase net 1, matched or cap | `forced_reading`, match status | 1.000, 1.000; cap at 0.61, 0.64 | held as filed; cap branch: R-net falls back to "does not apply" on lowercase |
| P4 | Forbidden lowercase within band and as fast | `forbidden_reading`, `pace` | -0.959 vs 0.698, -0.851 vs 0.590; as fast at step 5 | missed: below band; pace held; restored 0.47, 0.52 beside |

P3's filing lists "cap" both inside the expected outcome and in its on-miss text; either way the filed consequence of a cap is the fallback.

## Departures from the literature

- **Reuse is read by ablating a located head pair, not by a linear readout.** Jain et al. 2024 define possession by a linear readout of the capability trained on pretraining data, and test reuse with probing, pruning and reverse fine-tuning. We follow the user's ask 21 instead: a label needs a located mechanism, read by removal.
- **The net share subtracts the original model's own fall.** Jain et al. have no such term, and E97 read the state's own fall. We subtract because on the lowercase pair the original model falls more than the fine-tune does.
- **Ablation means come from the site's own inputs.** The interpretability literature usually takes means over a separate reference distribution, or resamples activations from other inputs. We keep E96's and E97's means so that stage 0 reproduces E97.
- **Ground truth comes from restricting the update.** Jain et al. get known capabilities from compiled and synthetic-grammar models. We force or forbid the route of a real fine-tune on a pretrained model, which fixes "through the pair" but not "reuses the old function".
- **Recipe changes.** Past 300 steps the lowercase rows repeat in the same order, where E53 saw each row once. The global gradient clip still counts the gradients of entries that are reset. The forbidden arms train with the pair at its mean, with the KL reference ablated the same way on copy.
- **The band is our own**: the 95th percentile of the absolute net share over 20 random head pairs outside the part, per state.

## Not run

- **Stage 4**: the old tasks under the refined rules, the class map and its twin rebuilt, and the one table.
- **The hand-write control** that the plan's design check names for stage 0, a hand-written write that should be spared. The stage 0-3 code left it out.
- **Forced-part lookup**, the plan's optional arm that would put a number on the granularity limit.
- **A second set of random pairs for the band**, so the band's own spread is unmeasured.
- A forbidden lowercase arm trained to the native's gain against the intact original model, or to the native's steps.
- A band for the three-head copying set, and the native copy fine-tune read at that set.
- The pair's prefix score on the forced copy states; only the induction accuracy was read.
- Any repurposing read on the forced lowercase states.
- An independent check; more seeds; anything on Qwen.

## Provenance

- Code: `experiments/definition_verify/{run,rules,reads,arms,amend1,plot}.py`.
- Filed rules: `results/predictions.json`, sha256 0f457684; `results/predictions_amend1.json`, sha256 76e35d8b.
- Results: `results/stage0.json`, `stage1.json`, `stage2.json`, `stage3.json` (the skip note), `stage3_amend1.json`, `run.out`; states and per-read tensors in `results/large/`.
- Job 52787: stages 0-2, L40 `debug`, git 2ae75888, stage 0 214 s, stage 1 66 s, stage 2 16 s. Log `cluster/logs/commute-e99-definition-verify-52787.out`.
- Job 52807: Amendment 1 and stage 3, L40 `debug`, git 5d4e7221, 704 s, peak host memory 6.1 GB. Log `cluster/logs/commute-e99-amend1-52807.out`.
- Cost $0.
