# Independent check of the E63 analysis
*Review, 2026-11-02, of `reports/analyses/2026-11-02_analysis_e63-neuron-sparsity.md` and `notebooks/E63_neuron_sparsity.ipynb`, by an agent that wrote neither. No model loaded, nothing trained. Scripts and output: `~/commute-data/scratch/baseline-lead/e63-check/` (`recompute.py`, `more.py`, `ratio.py`, `sim.py`, each with its `.out`). "Line" means a line of the analysis report unless another file is named.*

I recomputed every verdict from the per-state values, with the rules rewritten from `predictions.json`, and traced the report's numbers to files. The verdict stands: no number answers yes, with no mismatch against `score.json`, and only one rounding slip among 198 table values. The problems are in what the prose says around that verdict: one filed error rate is stale, the headline "direction flips" does not hold for the neuron count, the decomposition's share was available on all three families and is described as a one-pair comparison, and the rule has less power than a flat "No" suggests.

## Findings, ranked

### 1. The false-yes rate is stale, and "one task against one task" now covers all four tests (high)
Lines 53, 219; the notebook's "How compelling is the evidence" cell; source `predictions.json` `reading_notes.one_task_tests`.

Line 53 gives about 2% per number and 9% over five. That simulation ran before Amendment 12, when family B pooled two tables a side, and the pooling is what kept it low. Rerunning the separation and sign rule (`sim.py`, 200,000 draws, task differences ten times seed noise):

| case, no kind effect unless stated | per number | any of five, if independent |
|---|---|---|
| family B pooled, as first filed; T2 follows T1 | 0.018 | 0.086 (reproduces the filed figure) |
| family B one task against one, as amended; T2 follows T1 | 0.087 | 0.365 |
| as amended; T2 is a third task | 0.057 | 0.253 |
| as amended, kind effect of one task's standard deviation | 0.30 to 0.36 | |
| as amended, kind effect of two | 0.67 | |

The reference rule and the guard are not modelled, so these bound the rule from above. Amendment 12 also makes T3_B one task against one task, so line 53's "T1, T2 and T3_C" and line 219's "three of the four tests" should say all four.

The last two rows matter more than the first: a real kind effect as large as the differences between tasks would answer yes only about a third of the time, because one reversed pair fails the number.

**Fix.** Dated note in the report (not an edit of `predictions.json`): as amended, about 6-9% per number and 25-37% over five before the reference rule; four tests of four are one task against one; state the power beside the "No".

### 2. "The direction flips" does not hold for the count of neurons (high)
Lines 5, 9, 193, 211, 228; notebook cells "The answer", "Does the same count hold", "Surprises".

Line 9 says the carrying neurons sit "narrower for elicit on two of the three families, and wider on the third". Under the filed floor of 1, only family C separates: family A's gap is 0.965 and family B's is -0.704, both noise by the rule the report applies. The family B sign also depends on which same-word table is read:

| pair | gap in log2, elicit sparser positive |
|---|---|
| same word against random word, scored | -0.70 |
| same word on other keys against random word, both step 10 | +0.73 |

Line 193 gives the 1.44 between the two tables, then still opens "No, the direction flips", and line 228 lists a flip "for both". "Why family B reverses" asks about a reversal that this count has not shown.

The same applies to the first gradient (line 90, "elicit is the wider side in every family"): no test separates, and the 50-word cipher reverses family C and passes the filed test form (0.0152 against 0.0254, finding 9).

**Fix.** Line 9: "The activation shift sits narrower for elicit on two families and wider on the third. The count separates only on family C; on family B its sign depends on the table." Drop C1 from line 228.

### 3. The decomposition's share is already recorded on all three families (high)
Lines 211, 312; notebook "What we learned".

Line 211 says r "was not read on families B and C here" and concludes "this is a comparison on one pair". True of this run, but `experiments/regating_truth_table/results/table.json` holds r on the same saved states. A file lookup, not a re-derivation:

| test | built to elicit | built to be new |
|---|---|---|
| family A, matched size (E56 `job1.json`, `job2.json`) | 0.764 | 0.246 |
| family A, equal progress | 0.766 | 0.254 |
| family B | 0.782 | 0.270 |
| family B, other keys, beside (random word at step 30) | 0.798 | 0.285 |
| family C | 0.761 | 0.158 |
| 50-word cipher, beside | | 0.120 |

r keeps its direction and clears the seed ranges on every line. So the contrast the plan asked about ("cheap rival to r") can be drawn on three families now. Two limits belong beside it: r has no filed reference under this rule, and it has no causal check either, so this says r follows the build and the neuron numbers do not, not that r is right. It also bears on line 235: r reads the same-word table as it reads the elicit map, so "family B's flip was about its label" is not the only reading. It may be about the neuron numbers.

**Fix.** Replace line 211's last two sentences with these values and the two limits; narrow line 312 accordingly.

### 4. A1's equal-progress failure reads how the shift divides among blocks, and its "no" rests on family B alone (medium-high)
Lines 44, 148, 257.

Under a random rotation within a block, a vector's squared entries have a sum of squares near 3E²/3072 whatever its sparsity. So A1's rotated value is about 1/(12 Σ s_b²), where s_b is each block's share of the shift's energy. It runs from 0.083 (one block) to 0.333 (four equal). The control reads 0.0842 against a theoretical 0.0833, and a Monte Carlo gives the same value for sparse and dense input (`more.py`). It cannot see neurons at all. Per-block shift shares were not saved, so on real states this is derived, not read.

So "fails only because the rotated value widens more, 0.048 against 0.018" compares a neuron reading with a block reading on a scale several times wider. The verdict stands as filed. The reading should say the shift spreads over more blocks at step 30 (about 1.9 effective blocks to about 2.5).

Taking blocks out, unfiled: A1 / A1_rot per state under the filed separation rule (`ratio.py`).

| comparison | elicit | new | separates | elicit narrower |
|---|---|---|---|---|
| T1 | 0.151 | 0.193 | yes | yes |
| T2 | 0.154 | 0.207 | yes | yes |
| T3_B | 0.223 | 0.184 | yes | no |
| other keys against random word, step 10 | 0.255 | 0.184 | yes | no |
| same word against random word at step 30 | 0.223 | 0.204 | no | no |
| other keys against random word at step 30 | 0.255 | 0.204 | yes | no |
| T3_C | 0.078 | 0.193 | yes | yes |
| lowercase against 50-word cipher | 0.078 | 0.212 | yes | yes |

A1 is therefore held back by family B alone, and that reversal is solid. I tested the obvious escape: `rand32` has learned less than the same-word tables (held-out probability 0.56 against 0.96, `regating_table/results/stage0a.json`; not stated at lines 28 or 193). At the random word's step 30, about equal progress, raw A1 turns elicit-narrower (+0.0071, +0.0040), but only through blocks. Within blocks it stays wider. So the family B reading holds, more firmly than the report shows.

**Fix.** Add to line 148: the T2 miss is a block effect, A1's no rests on family B, and T3_B compares unequal progress. Do not refile the rule; say in the report that Amendment 14's "a gap that survives rotation is not neuron information" does not hold for A1.

### 5. "Closes with progress" and "not a progress effect" both claim more than one step of one task can show (medium-high)
Lines 5, 9, 148, 170, 185, 229.

- **Size moves with progress.** T1 has equal update size (norms about 3.2) and unequal progress; T2 has equal progress and an update about twice the size (3.14 against 6.02, `keydependent_elicit/results/stage2.json`). The closing gap cannot be told from a size or length effect. Line 187 hedges this, but lines 5 and 9, which are what gets read, state it flat.
- **The closing is from a gap the rule calls noise**, on two seeds, on the filed target. Only on the unfiled target (2.36 to 0.03) is the closing clear.
- **Line 148 contradicts line 229.** "It is not a progress effect, since the gap widens" sits beside A1 rising with training on both tables read (0.030 to 0.043 on the cipher, 0.029 to 0.045 on the random word, other keys at step 30). The widening is that rise on the new side only, with elicit never read past step 10. A rise on elicit too would shrink the gap. Nothing here can say whether it would.
- **Line 236 understates the next step.** No elicit state past step 10 is saved (`keydependent_elicit/results/large/`, `keydependent_patch/results/large/`), so it needs a new short fine-tune, not a re-read. The same holds for the same-word tables and lowercase.

**Fix.** Lines 5 and 9: "the gap is smaller at the cipher's own learned step, a larger update; elicit was read at one step only." Line 148: replace the sentence with "the gap is wider at the cipher's later step; elicit was not read there".

### 6. W2 cannot support "no fine-tune rescales the neurons" (medium)
Lines 9, 38, 128, 211.

W2 is energy-weighted over all neurons, and the update is spread over nearly all of them. Its control puts all energy on the 16 rescaled neurons; no control mixes a few rescaled neurons into a flat update. On a flat update, m fully rescaled neurons add about m / (12288 x W1) to the share. The measured excess over the Gaussian then equals about 5 neurons on elicit, 8 on lowercase and 6-8 on the matched and 500-word ciphers (`recompute.out`). Line 193's 15 neurons carrying half of lowercase's gain would be comparable. The excess also appears on the update that learned nothing (0.00164 against 0.00131), so it is no evidence of rescaling either. W2 simply cannot rule it out. `blind_to` lists only rotation towards another neuron.

**Fix.** "The update as a whole does not lie along neurons' own weights; a rescale of a handful of neurons would not show." Add this to `blind_to` in the report's table. Reading the per-neuron share on the top-shift neurons is a $0 check on saved states if anyone wants it.

### 7. "Needs" and "carry" overstate C1, and its blind-to line is short (medium)
Lines 40, 170, 183, 193, 211.

- **Sufficiency under one ranking.** C1 ranks by shift size, then asks when the patched set reaches half. That is an upper bound on a smallest sufficient set and does not show those neurons are needed. Line 40's "whether the gain causally needs few neurons" and line 170's "needs about 240" claim necessity. No knock-out was run.
- **Ranked at one position, patched at all.** A neuron that moves only at the key's position ranks low; A1 has the same gap. A "neuron" here is one index across 41 positions on families A and B and 127 on family C, so counts across families are not like for like (15 against 240).
- **Blind to more than output weights.** Attention and read-out changes also stay at W0. The full share runs from 0.46 to 0.97, so counts on different states describe different fractions of the gain. Line 40's `blind_to` names only output weights.
- **Line 183's "fits different keys using different neurons"** has an untested rival: per-window top-k lets a selection follow any input-dependent shift, key or not, and the hedge is thin given that distinct neurons were not counted. Line 211 then builds on it.
- **Controls exercise the code, not the claim.** Only 16 neurons shift in the control, so C1 at or below 4 follows from correct ranks, on a -1.53 nat "gain". It does not show C1 can find a sparse real gain.

**Fix.** "Top k by shift are enough for half"; extend `blind_to`; say counts are not comparable across families.

### 8. "No reading tells the two kinds apart" is wider than the rule (medium)
Lines 5, 9; notebook "The answer".

The filed rule is one-sided, and the question is whether elicit sits on fewer neurons, which is answered no. But W1 separates on all four tests with elicit denser each time. Mirroring the rule (`recompute.out`), W1 passes T2 and T3_C and loses T1 and T3_B only to the rotated gap. No number passes a mirrored rule, so the claim survives with references, but not as written. W1 also falls with training, and elicit tasks are less trained or smaller on some pairs, so its direction may not mean kind.

**Fix.** "No reading shows elicit on fewer neurons on all the pairs. The update's spread separates the other way on every test, but does not beat its reference on two."

### 9. The first gradient: "does not depend on the task's targets" is too broad (medium-low)
Line 90, and the same sentence in the notebook.

Shuffling permutes targets among the same batch, so inputs, the target token set and the position stay. It tests only the pairing of key and target. On the 50-word cipher shuffling halves the reading (0.0254 to 0.0127), so the sentence fails on a task in the same table. Lowercase also reads 0.0152 against 0.0106. Line 90's cause is correctly left open.

**Fix.** "Permuting targets within the batch leaves it much the same on family A."

### 10. The stated candidate for the random order's gap points the wrong way (low-medium)
Line 170.

"The two kinds' different full shares are one candidate." As a lower bar, the cipher's smaller full share should reduce its count, yet it reads higher. With a common bar of half the whole gain the random-order gap grows, 0.55 to 1.04 (`recompute.out`), so the bar shrinks this gap. Full share in some other sense remains open. Note too that each C1_random comes from one seeded order: its seed range reaches 0.66 on lowercase, more than its T3_C gap of 0.30, and 0.25 on T2's elicit seeds against a gap of 0.27. It does not change any verdict.

**Fix.** Drop the candidate, or state the direction; add the reference's noise to "How compelling".

### 11. Small points (low)
- Notebook cell "What we were trying to test" says built-to-elicit targets are "something the original model is close to doing". The same-word table starts at about 2.5e-4, about 100 times below the elicit map; the report's line 17 avoids this, the notebook should too.
- Line 211, "flat after Adam": Adam is the filed explanation, untested (line 90 says so). Say "flat".
- Line 9, "spread almost evenly … whichever kind": true on family A; the 500-word cipher reads 0.816.
- Line 100: the Gaussian for the random word reads 0.98773, so 0.9877, not 0.9878.
- Line 299 and line 53 lean on scratch files outside the repo. I found no script for the simulation among the review's scratch files, only its text (`e63-review/check.out`).

## What would change the conclusion
"None answers yes" is arithmetic and will not move on recomputation. Each path below is a new reading, not a refiling.

- **Nearest number: A1.** It fails T2 on the block reading (finding 4) and T3_B. If a causal check showed the same-word tables are not eliciting, A1 would be the one reading left with a claim, and r's agreement with the build on family B would then need its own explanation. That cuts both ways, since r reads those tables as elicit-like.
- **More pairs.** Each family is one contrast, and a kind effect of one task's standard deviation passes about a third of the time (finding 1). Several elicit and new tasks per family, read per task, would turn "no" into a measured limit.
- **Elicit at a later step.** A1 rising and C1 falling on elicit would close both progress questions. It needs a short new fine-tune.
- **C1's floor.** Moving it would not help: T2 and T3_B still fail. The 0.035 miss in P5 is not a near-yes.

## Confirmed
- **Verdicts.** Every test, reference gap, guard and `answers_yes` for all five numbers agrees with `score.json` (0 mismatches, `recompute.py`, stdlib only, sharing no code with `neuron_sparsity.py` or `e63-analysis/recheck.py`). The branch is "none answers yes". P1 to P5 are scored as filed. "Missed narrowly" is fair for P5 as a threshold statement.
- **Report's tables.** 198 values in the five per-task tables checked by script; one rounding slip (line 100). The scorecard, block shares, per-seed ranges and random-order range were checked against the recomputed means.
- **Unfiled target.** 8.15, 10.51, 8.07, 8.10 hold on mean curves. Per-seed means read 8.15 and 10.52.
- **E56 comparison.** The six rows of lines 176-181, k50 of 4096 and 12288, and r of 0.76 against 0.25 trace to `keydependent_patch/results/job1.json` and `job2.json`.
- **Stage 0.** 33 files, identity patch 0 on nine tasks, N16 exact, full share within 1e-7, worst first-batch miss 5e-5 nats, no SURPRISE line, all synthetic controls as stated.
- **Priors and sizes.** 2.5e-4 and 0.03 are the same statistic (`p_o`, `gradients.json`). Matched update norm on family A holds. The 26-state count holds.
- **Provenance.** Results in 1132c03, `git_head` de14090, no diff to 5727b38 on run code, predictions or cluster scripts, peak 7.5 GB. The log's stamps span 2 min 5 s, consistent with 2 min 12 s; `sacct` returned nothing here.
- **The report's own hedges.** "Built to elicit" is held throughout the report, and causes are marked untested on family B's flip, the GELU, Adam at line 90 and the narrow first gradient.

## Not run
Any model load; per-block shift shares; per-neuron W2; a knock-out of the top-k neurons; r under the filed reference rule, since it has none; the reference rule inside the simulation; `check.py` on the notebook.
