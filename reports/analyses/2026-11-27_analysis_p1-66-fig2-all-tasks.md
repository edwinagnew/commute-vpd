# How much of each fine-tune's write does the needed set carry, on all sixteen known-answer tasks?

We read Figure 2's retained share on all sixteen known-answer tasks at three featurizations, VPD subcomponents, top neurons and SAE latents, each fine-tune beside a random update of the same size. Fine-tunes keep a few hundredths of their write in the needed set, from 0.011 to 0.059 by task, where randoms keep 0.003 to 0.018. Tasks known to reuse old parts read no higher than tasks known to be new. A fixed neuron hook cleared the neuron control, so all three panels stand, as a description.

## What was read

The retained share $E_b$ over $K$ is the share of an update's write change carried by the needed set $K$, the variables whose swap between two inputs changes the model's answer. It reads 0 when the needed set carries none of the change and 1 when it carries all of it. The leftover variables, the remainders a featurization does not split, are counted in both the swap test and the sums. Each value is read on 512 windows of 8 tokens per task, with the needed set found at the last position.

Each of the 48 trained states, sixteen tasks by three seeds, is read beside one matched random update. The random update draws a Gaussian direction over the parameters the fine-tune changed and scales it to the fine-tune's size. It is the reference for a change that carries no task structure: a share near the random's says the needed set carries what any update of that size would put there.

The three featurizations are VPD subcomponents, the top neurons of each layer and SAE latents. A top neuron's variable is the part of its activation above a per-neuron bound; the part below joins the layer's leftover. Twelve of the 48 states, the four e58 tasks, were read earlier for the five-task figure, and their rows are reused after a stage-0 reproduction of a saved item. The other 36 states, with their randoms, are new: 72 items per featurization, 288 rows in all with the reused ones.

The card says 11 added tasks and 66 items per featurization. The run read 12 and 72, because the figure's elicitation task is now the key-dependent elicit fine-tune of E54, which replaces the 96-key elicit of E58; `experiments/fig2_all_tasks/README.md` gives the reason. The run is descriptive: the card files no bar and no verdict.

## What we expected

The predictions file expected every trained value between 0 and 1, near the earlier figure's trained range, and each random at or under about two hundredths. It expected the stage-0 controls to read as they did for the five-task figure. The do-nothing update gives a share of 1 with no changed answer. The identity swap, each variable swapped with its own value, changes no answer and moves the logits about as little as it did there, of the order of 8e-5 at top neurons.

## What we observed

![Per task and seed, the fine-tune's retained share beside its matched random, at VPD subcomponents, top neurons and SAE latents](../../experiments/fig2_all_tasks/figures/fig2_all_tasks.png)

The figure has one row per task, the tasks known to reuse above the grey line and those known to be new below it. Each purple square is one seed's fine-tune and each grey cross the random update matched to that seed. It is drawn by `experiments/fig2_all_tasks/plot.py` from `results/per_task.json`.

### VPD subcomponents and SAE latents

**Reading it.** At VPD subcomponents the fine-tunes' seed means run from 0.016 on the e53 text cipher to 0.059 on the c3 cipher. Their randoms sit in a narrow band, from 0.013 to 0.016. The text cipher sits closest to its random: its fine-tune reads 0.016 against its random's 0.013.

At SAE latents the fine-tunes run from 0.030 on the elicit and text-cipher tasks to 0.059 on e53 lowercasing. The randoms spread wider than at VPD, from 0.007 to 0.018. Lowercasing has the highest random at SAE, 0.018, as well as the highest fine-tune.

Seeds agree closely at both featurizations. The widest trained spread at VPD is on the e46 random-key task, 0.052 to 0.058; at SAE it is on the e46 identity task, 0.046 to 0.057.

The tasks known to reuse old parts do not read above the tasks known to be new. At VPD the reuse tasks' mean is 0.032 and the new tasks' mean is 0.041. At SAE the two means are 0.041 and 0.045. The reuse tasks span 0.026 to 0.038 at VPD, inside the new tasks' wider span, so this reading does not order the two sides by their known answers.

### Top neurons

**Reading it.** At top neurons the fine-tunes keep less than at the other two featurizations, and their randoms keep less still. The fine-tunes' seed means run from 0.011 on the e54 elicit task to 0.029 on the e54 cipher. Their randoms run from 0.003 to 0.009, the highest on e53 lowercasing.

The fine-tunes sitting nearest their randoms are the e58 copy task, 0.012 against 0.005, and the e53 text cipher, 0.015 against 0.008. The widest seed spread is on the e54 cipher, 0.024 to 0.037.

The reuse tasks' mean is 0.019 and the new tasks' mean 0.021. The reuse tasks span 0.011 to 0.028 and the new tasks 0.015 to 0.029, so at neurons too the reading does not order the two sides by their known answers.

## The neuron identity control: old hook and fixed hook

Under the first neuron hook the identity control read off on two of the four states it was checked on. On the e62 copy state the largest logit deviation was 0.02715, and on the e46 identity state 0.0692, against a filed order of 8e-5. On the e58 and c3 copy states it read 5.1e-5 and 4.6e-5. No answer changed on any state. On the e62 state the VPD identity read 5.9e-5 and the SAE identity 3.7e-5, so the deviation belonged to the neuron swap path.

The cause was activations straddling their bound. The old hook split each swapped activation at its bound using the input's own activation from the rerun, which differs from the cached run by rounding. On the e62 state one activation, and on the e46 state two, sat on opposite sides of the bound in the two runs, so the swap moved them across it. The fixed hook of commit `d2d81436` reads that activation from the cache.

Under the fixed hook the deviation is back at rounding on all four states, with no changed answer:

| state | straddling activations | old hook, largest deviation | fixed hook, largest deviation |
|---|---|---|---|
| e62/copy seed 0 | 1 | 0.02715 | 7.8e-5 |
| e46/ident32 seed 0 | 2 | 0.0692 | 5.5e-5 |
| e58/copy seed 0 | 0 | 5.1e-5 | 5.3e-5 |
| c3/copy seed 0 | 0 | 4.6e-5 | 5.1e-5 |

The do-nothing update under the fixed hook read a share of 1 with no changed answer.

**The fix's effect on the saved neuron item.** The stage-0 reproduction reads E298's saved neuron item, the e58 copy state at seed 0, again. Under the old hook it reproduced within 5.0e-6 relative, against a tolerance of 5e-5. The user decided on 2026-11-27 that this old-hook reproduction is the apparatus check, since E298 itself was read with the old hook.

Under the fixed hook one neuron of the needed set changes: the needed set has 4171 neurons where E298's had 4170. The share moves from the saved 0.011681 to 0.011685, a relative shift of 3.4e-4 against the same 5e-5 tolerance. This shift is the fix's effect on E298's item, not an apparatus fault.

**Old against fixed hook on the 72 new items.** All 72 neuron items were read again under the fixed hook. The share moved by at most 3.8e-6 on any item. The needed set's size changed on 22 of the 72 items, by at most 2 neurons. The table and figure use the fixed-hook values; at three decimals they equal the old ones.

E277 and E298 also read top neurons with the old hook. Their needed sets may differ by a neuron or two where an activation straddles its bound. They were not read again.

## Per-task values

Seed mean over seeds 0 to 2, with the lowest and highest seed in brackets, from `experiments/fig2_all_tasks/results/summary.json`. Neuron columns are the fixed-hook reads.

| task | known answer | VPD fine-tune | VPD random | SAE fine-tune | SAE random | neurons fine-tune | neurons random |
|---|---|---|---|---|---|---|---|
| e62/copy | reuse | 0.038 (0.036-0.040) | 0.016 (0.016-0.016) | 0.036 (0.033-0.039) | 0.011 (0.010-0.013) | 0.018 (0.016-0.020) | 0.006 (0.004-0.008) |
| e46/ident32 | reuse | 0.033 (0.033-0.034) | 0.014 (0.012-0.015) | 0.050 (0.046-0.057) | 0.007 (0.006-0.009) | 0.028 (0.024-0.033) | 0.003 (0.003-0.004) |
| e53/lowercase | reuse | 0.026 (0.024-0.027) | 0.013 (0.013-0.014) | 0.059 (0.057-0.063) | 0.018 (0.015-0.020) | 0.018 (0.017-0.020) | 0.009 (0.008-0.010) |
| e54/elicit | reuse | 0.032 (0.030-0.033) | 0.014 (0.012-0.015) | 0.030 (0.025-0.034) | 0.012 (0.011-0.014) | 0.011 (0.010-0.013) | 0.004 (0.003-0.004) |
| c3/rank1 | reuse | 0.032 (0.031-0.033) | 0.013 (0.013-0.014) | 0.032 (0.031-0.035) | 0.008 (0.007-0.009) | 0.019 (0.019-0.020) | 0.003 (0.003-0.004) |
| c3/copy | reuse | 0.032 (0.030-0.034) | 0.015 (0.014-0.015) | 0.042 (0.041-0.043) | 0.009 (0.008-0.010) | 0.025 (0.022-0.028) | 0.004 (0.003-0.005) |
| e58/rank1 | reuse | 0.029 (0.029-0.030) | 0.014 (0.013-0.014) | 0.035 (0.032-0.038) | 0.009 (0.008-0.011) | 0.017 (0.014-0.020) | 0.007 (0.004-0.011) |
| e58/copy | reuse | 0.033 (0.032-0.034) | 0.014 (0.014-0.014) | 0.044 (0.042-0.046) | 0.012 (0.010-0.015) | 0.012 (0.012-0.012) | 0.005 (0.004-0.006) |
| e62/lookup | new | 0.042 (0.040-0.043) | 0.016 (0.015-0.017) | 0.045 (0.042-0.048) | 0.013 (0.010-0.015) | 0.015 (0.014-0.016) | 0.006 (0.005-0.006) |
| e46/rand32 | new | 0.054 (0.052-0.058) | 0.014 (0.013-0.015) | 0.051 (0.048-0.052) | 0.010 (0.009-0.013) | 0.027 (0.022-0.031) | 0.005 (0.004-0.006) |
| e53/cipher | new | 0.016 (0.015-0.017) | 0.013 (0.012-0.013) | 0.030 (0.027-0.031) | 0.015 (0.015-0.015) | 0.015 (0.014-0.015) | 0.008 (0.007-0.009) |
| e54/cipher | new | 0.056 (0.054-0.058) | 0.013 (0.013-0.013) | 0.056 (0.054-0.058) | 0.008 (0.007-0.011) | 0.029 (0.024-0.037) | 0.004 (0.003-0.005) |
| e54/cipher_own | new | 0.031 (0.031-0.031) | 0.014 (0.013-0.015) | 0.038 (0.036-0.040) | 0.011 (0.009-0.013) | 0.017 (0.015-0.019) | 0.005 (0.005-0.005) |
| c3/cipher | new | 0.059 (0.057-0.061) | 0.014 (0.013-0.015) | 0.056 (0.056-0.057) | 0.008 (0.007-0.009) | 0.025 (0.022-0.031) | 0.004 (0.003-0.004) |
| e58/cipher | new | 0.039 (0.037-0.042) | 0.015 (0.014-0.015) | 0.038 (0.036-0.040) | 0.014 (0.011-0.016) | 0.023 (0.021-0.028) | 0.007 (0.007-0.008) |
| e58/prior_cipher | new | 0.029 (0.029-0.030) | 0.014 (0.013-0.015) | 0.045 (0.043-0.047) | 0.014 (0.011-0.017) | 0.021 (0.020-0.021) | 0.007 (0.006-0.008) |

## What we learned

The fine-tunes' retained share is small on all sixteen tasks, not only on the five the old figure read. It sits above a random update of the same size at all three featurizations, by the most on the e54 and c3 ciphers at VPD and SAE. It sits closest to the random on the e53 text cipher at all three, tied with the e58 copy task at neurons. So the needed set carries more of a fine-tune's write than chance, but most of the write lies outside it. At none of the three featurizations does this reading separate the tasks known to reuse old parts from those known to build new ones. The reuse definition is under revision, so no verdict is given here under either wording.

## How compelling is the evidence

The stage-0 controls read as filed at VPD and SAE: the do-nothing update read a share of 1 with no changed answer, the identity swap changed no answer, and a saved item from the five-task figure reproduced. At top neurons the same controls read as filed under the fixed hook, with the reproduction read as described above. The rescoring found all 288 rows, and the 52 rows shared with E298's figure match it.

Three seeds per task and one random per state is the whole sample; a random's spread across seeds is the only noise floor read here. The measurement is blind to whether the write it counts is the one the behaviour uses: it is a share of write change, not of behaviour. The identity control under the fixed hook was read on four states, not on all 48.

## Surprises

The neuron identity deviation on the e62 and e46 states under the first hook, traced to straddling activations and fixed. Nothing else.

## What next

- Read E277's and E298's top-neuron rows again under the fixed hook, if a paper figure cites their needed-set sizes; the shares should move by a few millionths, as here.
- Write the paper's Figure 2 caption and section for sixteen tasks and three panels.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| rival_a, rival_b | added tasks read like, or unlike, the five | descriptive, no bar | values above | none: descriptive |
| stage 0, VPD and SAE | do-nothing 1, identity no flip, reproduction within tolerance | filed in `predictions.json` | as filed | as expected |
| stage 0, neurons identity, old hook | deviation of the order of 8e-5 | filed in `predictions.json` | 0.02715 and 0.0692 on two states | not as expected; cause found |
| stage 0, neurons identity, fixed hook | deviation under 1e-4 on four states | filed in `ident_straddle.py` | at most 7.8e-5, no flips | as expected |
| stage 0, neurons reproduction | within 5e-5 relative of E298's item | filed in `predictions.json` | 5.0e-6 old hook; 3.4e-4 fixed hook | old hook as expected; fixed-hook shift recorded as the fix's effect, user 2026-11-27 |

## Departures from the literature

None beyond those `experiments/fig2_all_tasks/README.md` lists against the five-task protocol: the matched random's seed base, the stage-0 control state and the reproduction items.

## Not run

- E277's and E298's top-neuron reads under the fixed hook.
- The neuron identity control on states other than the four checked.
- More than one random per state, and seeds beyond 0 to 2.
- The 96-key elicit of E58, which left the sixteen.
- The new caption and section text in the paper.

Provenance: `experiments/fig2_all_tasks/results/{definition_rows.csv, summary.json, per_task.json, items/, stage0/, run_ident_diag.log, ident_straddle.json, run_ident_straddle.log, neurons_hook_diff.json}`; figure from `plot.py` at `ff4f93ee`, scored at `6433be77`; neuron rerun `167ca7fe`; review `reports/reviews/2026-11-27_review_e326-fig2-16-tasks.md`.
