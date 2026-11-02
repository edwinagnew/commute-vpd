# Does VPD tell reuse from new better than cheaper methods, against a truth it did not define?

*2026-11-02, baseline-lead. Plan step: `reports/plans/2026-11-02_plan_merge-reuse-into-baseline.md`, "One table". No run, no model, $0. Results `experiments/method_table/results/table.json`, written by `experiments/method_table/build.py`. Figure `experiments/method_table/figures/shares.png`, from `plot.py`. Log section `docs/research_log/parts/sections/methodtable.html`.*

We put every method's filed reading on every fine-tune into one table, and marked each row by how its truth was set. No fine-tune has the decomposition's reading and a cheaper method's reading both standing, with controls held, against a truth that is more than an assumption. So the merged line's question stays open, and the next step is a causal label on the small model's pairs or a harder counting test.

## The answer

Not yet. The decomposition's reading separates every pair it read that is labelled only by how it was built, but those labels are assumptions. On the counting set every method that responds to the task scores perfectly, and the decomposition there was barely trained and failed its control. On the one causal row of the small model, its reading went unscored, and the size of the weight change separates the two maps as well. Pruning revival and reverse fine-tuning did not survive the port to the small model.

![Each method's readings on one axis of shares, grouped by how the truth was set](../../experiments/method_table/figures/shares.png)

**Reading it.** The wide gaps in the top two groups carry a star or an unread verdict, and the clean verdicts for the decomposition sit only in the bottom group. The vertical line is the do-nothing reading of zero, and the short bar is the minimum gap that the separation rule asks for. A diamond is the positive control's own reading, where one is a share. The barely trained decomposition's control reads far under its bar of 0.5. The planted edit shows pruning can work on the small model, while the real pair does not separate. The figure leaves out reverse fine-tuning, neuron sparsity and the size labeller, whose readings are in steps, counts or weight norm.

## What the table is

Each row is a pair of fine-tunes, one labelled reuse and one labelled new. Each column is a method, and each cell gives the method's reading, its reference, whether its positive control held on that fine-tune, and one verdict. The verdicts are separates, fails, control failed, unreadable and not run. A cell with no reading says not run, and 62 of the 108 cells do.

Three kinds of truth mark the rows, under the label rule in `decisions.md` of 2026-11-02: elicitation is mechanistic, the causal check is the label, and a held-out reading is only evidence beside it.

- **By construction.** The fine-tuning task fixes the answer, and each state also passed behaviour and patch checks. This is the counting set of task P and task N.
- **By causal check.** Patching the old count or switching off the old copying head is the label. These are the counting set's earlier states and the copying-heads pair on the small model.
- **Only by assumption.** Built to elicit is how the task was built, not the causal label of `decisions.md` row 112; these pairs have no causal check. Those are all the other small-model pairs.

The methods, in a line each:

- **r and the stand-in ratio** are the decomposition's two readings. Each is a share of the fine-tune's log-odds gain, and zero means the changed causal importances carry none of it.
- **Steering** adds one mean residual shift to the original model. It reads the share of the gain that shift reaches.
- **Neuron patching** gives the top neurons their fine-tuned activations and reads the share of the gain.
- **Prompting** shows the original model demonstrations of the map in context.
- **Pruning revival** is Jain et al.'s Algorithm 3: remove the units that matter most, and read how much old behaviour returns.
- **Reverse fine-tuning** trains back toward the old behaviour and reads how fast it returns.
- **The two readouts** train a linear map on Pile text and read the gain lost when its direction is removed.
- **Neuron sparsity** asks whether the update or its activation shift sits on fewer neurons.
- **Recipe or size** reads only the learning rate, the steps or the size of the weight change. `decisions.md` makes update size a standing baseline column.
- **The answer probe** reads only which answers a state gives. It is the reference for anything that merely reads the task.

## What we expected, and why

No predictions were filed, because nothing ran. The plan's handoff expected the table to show two things, and it does: the ports failed on the small model, and the counting set cannot separate methods.

## What the counting set can say

**Reading it.** It cannot rank methods. The labels equal task P and task N, so any method that responds to the task scores perfectly, and 21 of 21 there is not discrimination. Steering, pruning and reverse fine-tuning all reach it, and so do the answer probe and the size of the update within a recipe cell. A decomposition trained for 400 of its 400,000 steps labelled all 21 right untrained, so it enters as failing its control and not as a pass. Its identity-prompt fine-tunes read about 0.07 against a bar of 0.5. Every counting-set cell carries that note in the results file.

The earlier counting row, whose labels come from causal checks, shows no method passing as filed. The cells read fails because guessed cut-offs missed, though the raw readings of pruning and steering do not overlap in the figure. With later cut-offs those two sort all 26 states, described only. Those labels are also tied to the recipe, since every reuse state was read at 1K or 2K steps.

## Whether the 21 of 21 carries to the small model

**Reading it.** It does not. As ported to the small model, pruning revival and reverse fine-tuning failed their positive control, so the 21 of 21 does not carry over. On the rank-1 map against the cipher, pruning's gap is 0.07 against a bar of 0.14. Reverse fine-tuning's gap of 3.2 steps runs the wrong way and is under a bar of 4.3. Under the filed rule, no step up to 300 passed, and no pruning size up to 320 did. The largest size alone clears a bar that was not filed. A planted edit of 13 neurons does pass both, so the code reads what it should when the answer is planted. Whether the port, the update size or the method is at fault stays open.

## What separates on the pairs labelled by how they were built

**Reading it.** The decomposition's reading separates every one it read, and each cheaper method read on more than one fails on at least one. r reads about 0.77 against 0.25 on the half-known pair at five seeds, with the same order on copy maps, lowercase and rank-1. Steering matches it only on the copy maps and at the cipher's own step. The source says that second comparison is not a test. Steering also has no gap between rank-1 and the cipher, though it places elicit near copy. Neuron patching fails on every pair it read. The readouts are unreadable as filed. The answer probe, read only on rank-1, separates that pair and misses its filed reference. These labels are assumptions, so this lists what follows the build, not what finds reuse. The filed reading of r on the rank-1 pair is "distance and prior", quoted as filed. The only control r passes here is that pair's own gap, which is itself an assumption.

**Neuron sparsity.** It fails on every pair as filed: "none answers yes". The caveat stays with each cell: all four tests compare one task with one task, a false yes runs at 6-9% per number and 25-37% over five before references, and the test has little power and covers few tasks.

**The 64-value readout.** Its verdict stays as filed: "readout fit missed and P1 elicit clause or P2 missed (P1 claim; P2 fail): unreadable; one training run was the cap; tell the user".

## What the one causal row on the small model says

**Reading it.** It leaves r unscored. With the copying head off, the copy map lost 0.47 to 0.56 of its gain against a bar of 0.5, and the lookup lost nothing. The verdict as filed is `copy_not_elicitation`, with `r_scored` false, and seed 2 misses the bar by 0.03. The unscored reading is 1.09 to 1.10 against 0.38 to 0.42. The independent check found that the update's size also separates this pair with no overlap, 4.6 against 5.5, so size is the only cell on this row with a verdict of separates. Mechanism, context and size all remain live readings of r's gap.

## Labels and markings

No label changes: no counting-set label moves, and every state keeps its reuse or new side. What moves is how five rows' truth is described.

| row | the source says | the table says |
|---|---|---|
| Rank-1 vs cipher, steps 10 and 10-300 | "old by construction", "new by construction" | only by assumption |
| Copy vs random, lowercase vs ciphers, half-known vs cipher | "construction labels" in the re-gating table | only by assumption |

The rank-1 map may be the easiest to believe, but no causal check has read it, and the rule marks pairs built to elicit with no causal check this way.

## How compelling is the evidence

- **Coverage.** Only 46 of 108 cells have a reading, and most methods were read on one or two rows. Two cells are comparable only when their rows match.
- **Noise.** Seeds run from two to eight per side, on one task pair per row. Every small-model row compares one map with one map or few, so a gap can be a task difference and not a difference of kind.
- **Blind spot.** A verdict of separates says the method follows the row's label. On assumption rows that label may follow the original model's prior or its distance from the fine-tuned one, as the filed reading of r and the stand-in ratio says on the rank-1 pair.
- **Derived cells.** Eighteen cells apply a rule or a bar to saved numbers where no source filed one; each is marked derived. The local copy of the separation rule reproduces four filed verdicts or the build stops. Every pointer was run through `jq`: 533 paths resolve, and the only nulls are the two filed "no passing step" values.
- **Not reviewed.** No independent agent has checked this table.

## Surprises

Nothing reversed a filing, since none was made. Three things read differently from their prose summaries. The experiments catalogue says the baselines separate the ends of the rank-1 pair, while steering's status on that pair in the results file is "no gap". The size labeller is the only cheap method with a clean separation on a causal row. Neuron sparsity passes three of its five numbers on lowercase against the cipher, and it still answers no as filed, since a number needs all four tests.

## What next

1. **Stage B's interchange patch on the small model's pairs.** It would move rows out of "only by assumption". If r still separates under a causal label, with update size unable to do so, VPD has its first fair win.
2. **The harder counting test, through `/design`.** It keeps the task fixed and varies reuse, so reading the task no longer wins. If the cheap methods already separate it, a 56-hour decomposition has little left to show.
3. **A within-recipe size labeller beside r wherever pairs share a recipe cell.** If size ties r on those rows as it does on the copying-heads pair, r's lead there is not evidence of kind.

## Scorecard

No predictions filed: this table reads filed results and runs nothing. Each filed verdict is quoted in its cell below.

## The full table

Each section is one row, built from `table.json`. A star marks a counting-set cell, where the labels equal task P and task N. The pointer column names the first of a cell's paths; `table.json` lists them all.

### Counting: task P vs task N

*Task P answers one more than count a; task N answers count b. 21 scored of 35, seeds 2-4, lr 1e-4 and 1e-5, 1K and 10K steps. Model 3M minGPT. Truth by construction: Disagreements between task and checks: [] No held-out reading.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | 21 of 21 at 400 steps; reuse -0.01 to 0.01 vs new -0.06 to -0.04 | Cut-off -0.018. On the 67M, the elicit and cipher fine-tunes read 0.77 and 0.24. | failed: Identity-prompt fine-tunes read 0.07 against a bar of 0.5: positive_control 'none'. | **control failed** * | outcome 'tie', positive_control 'none' | `~/commute-data/scratch/reuse-lead/c2-fixes/readcheck_A/c2.json` `.scored["c1b/P_1e-4_s2/1000"].r`; job 48413; 48441 tests the cause on four states |
| ratio | 21 of 21 at 400 steps; reuse -0.00 to 0.01 vs new -0.06 to -0.03 | Cut-off -0.013. | failed: Identity-prompt fine-tunes read 0.07 against a bar of 0.5. | **control failed** * | outcome 'tie', positive_control 'none' | `~/commute-data/scratch/reuse-lead/c2-fixes/readcheck_A/c2.json` `.scored["c1b/P_1e-4_s2/1000"].ratio`; job 48413 |
| steering | 21 of 21; share reuse 0.83 to 0.96 vs new 0.47 to 0.66 | Cut-off 0.742 set on held-out seeds; binomial bar 15; best recipe labeller 15. | held: True: .controls.steer_positive_control | **separates** * | passes: True | `experiments/pcfg_anchors/results/c1b.json` `.scored["c1b/P_1e-4_s2/1000"].x.steer`; job 46330-46499 |
| pruning | 21 of 21; gain share reuse 0.60 to 0.92 vs new 0.06 to 0.08 | Cut-off 0.222 set on held-out seeds; binomial bar 15; best recipe labeller 15. | held: True: .controls.prune_control_pass | **separates** * | passes: True | `experiments/pcfg_anchors/results/c1b.json` `.scored["c1b/P_1e-4_s2/1000"].x.prune`; job 46330-46499 |
| reverse | 21 of 21; steps over the scratch model's reuse 0.11 to 0.13 vs new 0.41 to 0.65 | Cut-off 0.272 set on held-out seeds; binomial bar 15; best recipe labeller 15. | held: target: .controls.scratch.end | **separates** * | passes: True | `experiments/pcfg_anchors/results/c1b.json` `.scored["c1b/P_1e-4_s2/1000"].x.reverse`; job 46330-46499 |
| labeller | 21 of 21 by size within a recipe cell; 15 of 21 with one cut-off | Binomial bar 15. Learning rate and steps alone read 12 of 21. | none: No control: it reads the recipe's result. | **separates** * | nuisance labeller: update_size 15 | `experiments/pcfg_anchors/results/c1b.json` `.train_stops.recipe_ceiling_full.update_size.agree`; job 46330-46499 |
| probe | 21 of 21; reuse 0.67 vs new 0.28 | Cut-off 0.473. | held: Reads the answers only: the task-identity reference. | **separates** * | task-identity reference (not scored; not a pass) | `experiments/pcfg_anchors/results/c1b.json` `.scored["c1b/P_1e-4_s2/1000"].x.probe`; job 46330-46499 |

Not run: neuron_patch, prompting, readout_64, readout_full, sparsity.

### Counting: through count a vs not

*26 scored, 15 reuses and 11 new, from Stage A and Amendment 5. Model 3M minGPT. Truth by causal check: Behaviour share and interchange patch. No held-out reading.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| steering | 21 of 27; reuse 0.84 to 0.98 vs new 0.37 to 0.67 | Binomial bar 19; slow labeller 20, so 22 needed. | none: No steering control in this stage's file. | **fails** * | passes: False; per run 16 of 18, passes: True | `experiments/pcfg_anchors/results/c1.json` `.scores.steer.all.agree`; job 46105, 46106, 46107, 46121 |
| pruning | 20 of 26; reuse 0.42 to 0.75 vs new -0.11 to 0.08 | Binomial bar 18; slow labeller 19, so 21 needed. | held: Importance pruning on the base model, near-vacuous by the size of its drop. | **fails** * | passes: False; per run 14 of 17, passes: True | `experiments/pcfg_anchors/results/c1.json` `.scores.prune.all.agree`; job 46105, 46106, 46107, 46121 |
| reverse | 17 of 26 | Binomial bar 18; slow labeller 19, so 21 needed. | held: The scratch model reached the target at 1e-5. | **fails** * | passes: False; per run 12 of 17, passes: False | `experiments/pcfg_anchors/results/c1.json` `.scores.reverse.all.agree`; job 46105, 46106, 46107, 46121 |
| labeller | 19 of 26 by learning rate; 18 of 26 by steps | Binomial bar 18. | none: No control: it reads the recipe. | **fails** * | passes: False | `experiments/pcfg_anchors/results/c1.json` `.scores.nuisance_slow.all.agree`; job 46105, 46106, 46107, 46121 |

Not run: r, ratio, neuron_patch, prompting, readout_64, readout_full, sparsity, probe.

### Copy at a new marker vs lookup

*6, seeds 0-2, step 30. Model 67M. Truth by causal check: Labels as filed, seeds 0-2: copy old, old, between; lookup new, new, new. Share lost: copy 0.47 to 0.56 against a bar of 0.5, lookup -0.00 to 0.00. No held-out reading.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | unscored: copy 1.09 to 1.10 vs lookup 0.38 to 0.42 | Zero update gain 0.0. | none: Path control pass: True. No known-kind state: the only causal label here is the one that left r unscored. | **unreadable** | r_scored False; verdict 'copy_not_elicitation' | `experiments/induction_cue/results/run.json` `.score.r_copy`; job 48541 |
| labeller | copy 4.57 to 4.62 vs lookup 5.45 to 5.50 | Same recipe cell, same step. | none: No control: it reads the recipe's result. | **separates** | none filed | `experiments/induction_cue/results/run.json` `.states.copy_s0.norm`; job 48541; derived |

Not run: ratio, steering, neuron_patch, prompting, pruning, reverse, readout_64, readout_full, sparsity, probe.

### Copy maps vs random maps

*8, E12's recipe, two tables each, seeds 0-1. Model 67M. Truth only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row 112; these pairs have no causal check. No held-out reading.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | copy 0.76 to 0.80 vs random 0.27 to 0.29 | Zero update reads 0; no filed cut-off (E60: labels none yet). | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **separates** | none filed | `experiments/regating_truth_table/results/table.json` `.runs["e46/ident32_s0"].r.value`; job 46258; derived |
| ratio | copy 0.87 to 0.97 vs random 0.36 to 0.41 | Zero update reads 0; no filed cut-off (E60: labels none yet). | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **separates** | none filed | `experiments/regating_truth_table/results/table.json` `.runs["e46/ident32_s0"].ratio.value`; job 46258; derived |
| steering | copy 0.72 to 0.80 vs random 0.10 to 0.19 | E51's rule; a learned-nothing fine-tune is the null. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. | **separates** | P3 pass, gap 0.628 | `experiments/baseline_neurons/results/baseline_neurons.json` `.b3["ident32_s0"].S`; job 44095 |
| neuron_patch | half the gain only at all 12288 neurons; full width copy 0.50 to 0.71 vs random 0.76 to 0.97 | E51's rule on neurons needed and on the idle share. | held: Identity and full residual patches. | **fails** | P1 fails, P2 fails | `experiments/baseline_neurons/results/baseline_neurons.json` `.b2["ident32_s0"].shares["12288"]`; job 44095 |
| prompting | 16 demonstrations: copy 0.10 to 0.12 vs random 0.13 to 0.14; 31 beside the query: 0.22 vs 0.18 to 0.19 | Share 0 with no demonstrations; bar 0.5 at 31. | held: Induction control top-1 0.82 on repeated random text. | **fails** | P4 reversed; amendment fails, band 'mechanism only' | `experiments/baseline_neurons/results/baseline_neurons.json` `.b4.shares["ident32_s0"]["16"]`; job 44095; amendment 45898 |
| sparsity | this test passed by none, of the five numbers; ident32 vs rand32 only | Each number against a rotated, shuffled, Gaussian or random-order reference inside the test. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **fails** | none answers yes: write up; an elicit/new gap under T1 only is reported as a progress effect | `experiments/neuron_sparsity/results/score.json` `.rows.W1.T3_B.pass`; job 48527 |

Not run: pruning, reverse, readout_64, readout_full, labeller, probe.

### Lowercase vs ciphers

*6, step 300, seeds 0-1. Model 67M. Truth only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row 112; these pairs have no causal check. No held-out reading.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | lowercase 0.75 to 0.77 vs ciphers 0.12 to 0.17 | Zero update reads 0; no filed cut-off (E60: labels none yet). | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **separates** | none filed | `experiments/regating_truth_table/results/table.json` `.runs["e53/lowercase_s0"].r.value`; job 46258; derived |
| ratio | lowercase 0.92 to 0.93 vs ciphers 0.26 to 0.29 | Zero update reads 0; no filed cut-off (E60: labels none yet). | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **separates** | none filed | `experiments/regating_truth_table/results/table.json` `.runs["e53/lowercase_s0"].ratio.value`; job 46258; derived |
| steering | lowercase 0.30 to 0.32 vs ciphers 0.09 to 0.16 | Zero update reads 0; no filed cut-off. Lowercase is under E60's unfiled 0.5. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. | **fails** | none filed | `experiments/regating_truth_table/results/table.json` `.runs["e53/lowercase_s0"].S.value`; job 46258; derived |
| sparsity | this test passed by W2, A1, C1, of the five numbers; lowercase vs broad cipher only | Each number against a rotated, shuffled, Gaussian or random-order reference inside the test. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **fails** | none answers yes: write up; an elicit/new gap under T1 only is reported as a progress effect | `experiments/neuron_sparsity/results/score.json` `.rows.W1.T3_C.pass`; job 48527 |

Not run: neuron_patch, prompting, pruning, reverse, readout_64, readout_full, labeller, probe.

### Half-known vs cipher, equal size

*10, 32 keys, seeds 0-4, elicit step 10, cipher step 8-9. Model 67M. Truth only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row 112; these pairs have no causal check. No held-out reading on this pair; E58's held-out reading on a 24-key version sits with the rank-1 row.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | elicit 0.76 to 0.77 vs cipher 0.21 to 0.27, five seeds | E51's rule; zero update reads 0. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. | **separates** | a_higher, gap 0.518, range 0.055 | `experiments/keydependent_patch/results/job2.json` `.P2.r.a`; job 45963, 45973, 45974; 46170, 46171 |
| ratio | elicit 0.95 to 0.96 vs cipher 0.29 to 0.34 | Zero update reads 0; no filed cut-off (E60: labels none yet). | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **separates** | none filed | `experiments/regating_truth_table/results/table.json` `.runs["e54/elicit_s0"].ratio.value`; job 46258; derived |
| steering | log-odds share elicit 0.25 vs cipher 0.18 to 0.24, five seeds | E51's rule; gap needs at least 0.2. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. | **fails** | S_lo verdict none; E54's nats S P2 fail, verdict none | `experiments/keydependent_patch/results/job2.json` `.P2.S_lo.a`; job 45963, 45973, 45974; 46170, 46171 |
| neuron_patch | top 16 neurons elicit -0.02 to 0.02 vs cipher -0.04 to 0.01; neurons for half the gain 4096 vs 12288 | E51's rule on a filed grid; zero means the top neurons carry nothing. | held: Identity and full residual patches, full-width rise. | **fails** | N16 none, k50 index none; filed null, status pass | `experiments/keydependent_patch/results/job2.json` `.P1.N16.a`; job 45963, 45973, 45974; 46170, 46171 |
| readout_64 | gain lost, elicit -0.04 to -0.03 vs cipher 0.01 | Random direction at most 0.0032. A gap of +0.2 was predicted. | failed: Fit bar missed (pass: False); control on the original model held (pass: True). | **unreadable** | readout fit missed and P1 elicit clause or P2 missed (P1 claim; P2 fail): unreadable; one training run was the cap; tell the user | `experiments/readout64/results/run.json` `.P2.separation.a`; job 48611 |
| readout_full | keys held, elicit [21, 17, 17, 13], cipher [0, 0, 0, 0], of 32 | 24 keys needed. W0's own logits through the same rules read elicit 16 in band, possession 17/32, cipher 0 (job 48453). | failed: Construction check fails: 11 keys in band, 24 needed. | **unreadable** | P1 fail, closed at its hard cap as unreadable; [stage0 P3] FAIL: the last-block readout does not reproduce the construction; stop and tell the user | `experiments/jain_readout/results/run.json` `.possession.elicit.all.possession[0].count`; job 48412; 48453 |
| sparsity | this test passed by A1, of the five numbers; five seeds at matched size | Each number against a rotated, shuffled, Gaussian or random-order reference inside the test. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **fails** | none answers yes: write up; an elicit/new gap under T1 only is reported as a progress effect | `experiments/neuron_sparsity/results/score.json` `.rows.W1.T1.pass`; job 48527 |
| labeller | elicit 3.14 to 3.17 vs cipher 3.18 to 3.20 | Same lr; the cipher is read at the step matching the elicit size. | none: No control: size is matched by design. | **fails** | none filed | `experiments/keydependent_elicit/results/stage2.json` `.finetunes.elicit_s0.norm`; job 45963, 45973, 45974; 46170, 46171; derived |

Not run: prompting, pruning, reverse, probe.

### Half-known vs cipher, own steps

*10, seeds 0-4, cipher step 30. Model 67M. Truth only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row 112; these pairs have no causal check. No held-out reading.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | elicit 0.76 to 0.77 vs cipher 0.24 to 0.27 | Zero update reads 0; no filed rule on this comparison. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. | **separates** | P3: matched and own-step cipher agree within 0.1, status pass | `experiments/keydependent_patch/results/job1.json` `.states.elicit_s0.r`; job 45973, 45974; 46170, 46171; derived |
| steering | elicit 0.38 to 0.39 vs cipher 0.07 to 0.11, two seeds | E51's rule; gap needs at least 0.2. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. | **separates** | P3-own a_higher, status pass; 'not a test' | `experiments/keydependent_elicit/results/amend1.json` `.score["P3-own"].a`; job 46070 |
| neuron_patch | elicit 0.93 to 0.94 vs cipher 0.95 to 0.97 | Full-width share of the gain. | held: Identity and full residual patches. | **fails** | none filed | `experiments/keydependent_patch/results/job1.json` `.states.elicit_s0.b2_shares["12288"]`; job 45973, 45974; 46170, 46171; derived |
| readout_64 | own-step and matched cipher differ by 0.00 to 0.01 | Within 0.1 predicted. | failed: Fit bar missed. | **unreadable** | readout fit missed and P1 elicit clause or P2 missed (P1 claim; P2 fail): unreadable; one training run was the cap; tell the user | `experiments/readout64/results/run.json` `.P2.own_diff.s0`; job 48611 |
| sparsity | this test passed by W2, of the five numbers; seeds 0-1, equal progress | Each number against a rotated, shuffled, Gaussian or random-order reference inside the test. | none: None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is verified independently was read. Stage 0 pass: True. | **fails** | none answers yes: write up; an elicit/new gap under T1 only is reported as a progress effect | `experiments/neuron_sparsity/results/score.json` `.rows.W1.T2.pass`; job 48527 |
| labeller | elicit 3.14 to 3.17 vs cipher 6.02 | Same lr, different steps: not a within-cell reading. | none: No control: it reads the recipe's result. | **separates** | none filed | `experiments/keydependent_elicit/results/stage2.json` `.finetunes.elicit_s0.norm`; job 45973, 45974; derived |

Not run: ratio, prompting, pruning, reverse, readout_full, probe.

### Rank-1 vs cipher, step 10

*16 of C3's 32, seeds 0-7, 24 trained keys. Model 67M. Truth only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row 112; these pairs have no causal check. Held-out carry, evidence only: rank-1 0.08, cipher -0.34, elicit -0.04, copy 0.48 (experiments/open_pair_instruments/results/run.json .score.readings.x.means).*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| r | rank-1 0.84 to 0.90 vs cipher 0.22 to 0.25; gap +0.62, bar 0.06 | Zero update reads 0. | held: Poles must differ by 3 seed sd: held. | **separates** | status 'ok'; verdict 'distance and prior' | `experiments/open_pair_instruments/results/run.json` `.columns.r.rank1`; job 46808, 46815, 46840; derived |
| ratio | rank-1 1.09 to 1.18 vs cipher 0.30 to 0.34; gap +0.80, bar 0.11 | Zero update reads 0. | held: Poles must differ by 3 seed sd: held. | **separates** | status 'ok'; verdict 'distance and prior' | `experiments/open_pair_instruments/results/run.json` `.columns.ratio.rank1`; job 46808, 46815, 46840; derived |
| steering | rank-1 0.18 to 0.25 vs cipher 0.15 to 0.22; gap +0.03, bar 0.07 | Zero update reads 0. | failed: Poles must differ by 3 seed sd: failed. | **control failed** | status 'no gap' | `experiments/open_pair_instruments/results/run.json` `.columns.S.rank1`; job 46808, 46815, 46840; derived |
| pruning | rank-1 0.41 to 0.51 vs cipher 0.35 to 0.49; gap +0.07, bar 0.14 | Minus 20 random removals. | failed: Poles must differ by 3 seed sd: failed. A planted 13-unit edit passed: True, True. | **control failed** | status 'no gap'; verdict 'no gap' | `experiments/open_pair_instruments/results/run.json` `.columns.R.rank1`; job 46808, 46815, 46840; derived |
| reverse | rank-1 12.38 to 14.39 vs cipher 8.65 to 10.90; gap +3.21, bar 4.27 | Floor 1 step. | failed: Poles must differ by 3 seed sd: failed. A planted 13-unit edit passed: True, True. | **control failed** | status 'no gap'; verdict 'no gap' | `experiments/open_pair_instruments/results/run.json` `.columns.t_half.rank1`; job 46808, 46815, 46840; derived |
| labeller | rank-1 3.10 to 3.19 vs cipher 3.03 to 3.26; gap +0.03, bar 0.27 | Sizes matched by design. | none: No control: size is matched by design. | **fails** | status 'no gap' | `experiments/open_pair_instruments/results/run.json` `.columns.norm.rank1`; job 46808, 46815, 46840; derived |
| probe | rank-1 3.25 vs cipher -0.09; gap +3.33, bar 0.00 | Cipher should read about 0. | held: Poles must differ by 3 seed sd: held. | **separates** | status 'ok' | `experiments/open_pair_instruments/results/run.json` `.columns.q.rank1`; job 46808, 46815, 46840; derived |

Not run: neuron_patch, prompting, readout_64, readout_full, sparsity.

### Rank-1 vs cipher, steps 10-300

*9 per step, three seeds, plus C3's 32 re-read. Model 67M. Truth only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row 112; these pairs have no causal check. No held-out reading in this stage.*

| method | reading | reference | positive control | verdict | filed, as filed | pointer |
|---|---|---|---|---|---|---|
| pruning | rank-1 minus cipher at steps 10, 30, 100, 300: +0.09, -0.13, -0.11, -0.08 | Poles must differ by 3 seed sd, with a floor. | failed: Passed at no step: first passing step None. | **control failed** | step 10 'no gap'; step 30 'reversed'; step 100 'no gap'; step 300 'no gap' | `experiments/open_pair_instruments/results/ladder.json` `.score_ladder.per_step["10"].R.gap`; job 46954 |
| reverse | rank-1 minus cipher at steps 10, 30, 100, 300: +2.62, +2.83, +1.98, +1.72 | Poles must differ by 3 seed sd, with a floor. | failed: Passed at no step: first passing step None. | **control failed** | step 10 'no gap'; step 30 'no gap'; step 100 'no gap'; step 300 'no gap' | `experiments/open_pair_instruments/results/ladder.json` `.score_ladder.per_step["10"].t_half.gap`; job 46954 |

Not run: r, ratio, steering, neuron_patch, prompting, readout_64, readout_full, sparsity, labeller, probe.


## Not run

- Any new reading or fine-tune, and any model load.
- The 62 cells marked not run, listed under each row above. Among them: neuron patching, steering and readouts on the copying-heads pair; VPD at steps 30 to 300; every cheap method on lowercase except steering and sparsity.
- A decomposition past 2,000 steps on the counting set, and any reading on a trained one.
- Stage B's interchange patch; the harder counting test.
- A log-odds share of the original model's own copying under the same head, so the 0.5 bar on that row has no reference on its own scale.
- Fine-tunes outside the named sources: E58 Amendments 2 to 7, E57, E34 and the Qwen line. Their readings are not in this table.
- An independent check of this table, and a colourblind validator on the figure's pair, because this node has no `node`. The pair is the Okabe-Ito blue and vermillion, and shape repeats the distinction.

## Departures from the literature

This table adds no method; each source report lists its own departures, and the ones that decide a cell are here.

- **Jain et al. 2024, Algorithm 3 and reverse fine-tuning.** They revive a labelled capability, scored by accuracy, against a scratch model. On the small model we use the whole next-token distribution as the old behaviour, a half-life as the reverse reading and no scratch model, because no such model lacks the behaviour. On the counting set we prune hidden units at two sizes, one number per state, with a calibrated cut-off, and task P is not in their paper. So neither row replicates their setting.
- **Jain et al. 2024, Definition 1.** Their readout covers a few hundred task tokens. The two readouts here cover the whole vocabulary or the pair's 64 values, trained on Pile text, with thresholds set by us.
- **Olsson et al. 2022.** The copying-heads label uses their prefix score with our bar, mean ablation in place of zero ablation, and no copying score. The 0.5 bar has no published source.
- **This table's own rules.** Where no source filed a rule for a pair, a cell applies E51's separation rule, which is this project's own and not from a paper. C3's pole rule is applied to steering as well as to the two ports. Both are marked derived or quoted in the cell.

## Provenance

Sources, read by `jq`-style path: `experiments/regating_truth_table/results/table.json`, `experiments/baseline_neurons/results/`, `experiments/keydependent_elicit/results/`, `experiments/keydependent_patch/results/`, `experiments/jain_readout/results/`, `experiments/readout64/results/run.json`, `experiments/neuron_sparsity/results/score.json`, `experiments/induction_cue/results/run.json`, `experiments/pcfg_anchors/results/c1.json`, `c1b.json`, `c1b_cutoffs.json`, `experiments/open_pair_instruments/results/run.json`, `ladder.json`. The counting-set decomposition reading is only on disk at `~/commute-data/scratch/reuse-lead/c2-fixes/readcheck_A/c2.json`, job 48413; its values are copied into `table.json`. Build: `~/commute-agents/fleet/cpu-run python experiments/method_table/build.py`.
