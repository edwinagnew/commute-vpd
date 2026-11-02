# Independent check of the E34 Amendment 10 analysis and code

*2026-11-02. Checked at commit 39f44b7b. The reviewer wrote neither the code nor the analysis. Line numbers are those of
`reports/analyses/2026-11-02_analysis_e34-amend10-input-weighted.md` unless another file is named. New $0 reads, no model, no GPU:
`~/commute-data/scratch/commute-lead/e34-amend10-check/rescore.py` and `permatrix.py`, each with its `.out` beside it.*

**Verdict.** No finding changes a filed verdict. P1 and P2 read "further than a rerun" on both axes, with room to spare, and
P3, P4 and P5 read as reported. The allowed wording stands per family. One scope phrase needs a correction: "holds at each matrix's output"
reads as a per-matrix claim, while every verdict is pooled over the 24 matrices, and on the map tables a few single matrices read above their rerun
line. The rest is one overreaching sentence, joined sentences the filing bars, one wrong table number, a filed read left out, and small
misstatements in the surprises.

## What is right

- **Numbers.** Every value in the report's tables, scorecard and prose agrees with `amend10.json`, apart from findings 5 and 6. That covers the midpoints, shortfalls (0.171, 0.424), sizes (1.33 to 2.04, 0.66 to 0.87), distances, ceilings, power table, seed range (0.065 to 0.106), P4 and P5 ranges, length pair and target probabilities.
- **Rules.** I wrote the filed rules again from `predictions_amend10.json`, without importing `amend10.py`, and ran them on the raw reads. All margins and all angle, length and seed labels agree, on all four input kinds and both forms: 0 disagreements. `agree` and `reading` match the filing. The 22 tests in `tests/test_teacher_amend10.py` pass.
- **Room in the verdicts.** On task windows and Pile, the smallest clearance past the margin is 0.117 in cosine and 0.236 in distance. Against the single lowest rerun pair instead of the mean, a student is still 0.144 under in cosine and 0.299 over in distance. No plausible change of margin moves P1 or P2.
- **Arithmetic and inner product.** `gram` is $\mathrm{tr}(A\Sigma B^\top)$ for rows of (d_out, d_in). The centred form subtracts $\mu\mu^\top$ with the whole set's mean. The half weights sum to the whole. Recomputing cosines from the saved Grams matches the file within 5e-8, as seen and with the mean removed. Sample counts read 2,993, 3,977, 3,977 and 131,072, as filed. The ceiling is the right residual, since $(I-P_U)$ on the write side and the Schur complement on the read side are what no first-order change reaches.
- **W₀.** `_run` restores W₀ in its `finally` (`src/commute/finetune.py:189`), and moments are read after all reruns. Later recipes reproducing their saved fine-tunes (`rand32b_s0 again` reads 1.0) confirm it.
- **Citations.** Gain ratios 1.02 and 0.98 match `new_vs_existing_amend1.json`, and term lost 1.9% to 2.3% and 16% to 20% match E12 Amendment 3's table. The 0.0286 belongs to `real_o0`; `real_o1` reads 0.0277 (Amendment 8's report, line 284). The re-switching definition matches E12 Amendment 2.
- **Claim against the outcome table.** The three family sentences use the filed "angle further" text. "Reproduces its behaviour" is withheld from `rand32b`, and P4 and P5 do not decide against the reading. Exceptions: findings 1 and 3.
- **"Follows its task" (line 31).** I tested it against a harder reference than the report used, another table's student read against this fine-tune. It reads at most 0.08 as seen on task windows, against 0.40 to 0.58 for the student's own. It holds.

## Could the third revision have altered a verdict?

No. `git show 5c0d54d6` changes one stop condition and removes one dataclass field in `amend10.py`; the filing's rules are untouched. Job 49463 wrote no input-weighted read of a student: its `reads` holds only `frobenius`. The two jobs agree exactly on all 336 Frobenius pairs and on the rerun target probabilities I compared, so the revision changed nothing that the verdicts read. Under the old bound, no verdict would exist at all: all three window sets read above 0.5 (0.567, 0.562, 0.579), and Pile reads 1. So the revision decided whether verdicts exist, not which way they fall.

Two filed leftovers, for the record and not to be edited: P1's `positive_control` still says "Weighting: the power control" (`predictions_amend10.json:58`), and `threshold_sources` still names `power_frob_max` (line 147). Line 73 says plainly that the control shows no power, which is the honest treatment.

## Is any cause asserted that nothing measured?

Not in the report. Line 42 disclaims the gap's cause, and line 77 says "can mean". The notebook is stronger: cell 57's "That ties a student to its run" states what the report only allows. Fix: "That can mean a student holds something tied to its run", matching line 77.

## Findings, most important first

### 1. "Holds at each matrix's output" reads as per-matrix; the verdicts are pooled (line 15; notebook cell 54; `TRIED.md:101`; `PENDING.md:12`)

**What is wrong.** Every verdict pools the 24 matrices before the cosine. The filed phrase means where the read is taken, against downstream, but "Each holds at each matrix's output" says the sentence holds matrix by matrix. Line 115 lists per-matrix verdicts as not run, so nothing supports that reading, and the saved Grams go against it on the map tables.

**Unfiled read** (`permatrix.py`, each student against its table's rerun mean, per matrix). This has no per-matrix margin, so it is a size, not a verdict.

| Family | Matrices above the rerun line | Within 0.05 of it, or above |
|---|---|---|
| Real task | 0 of 24, all reads | 0 of 24 |
| Map tables | 1 to 4 of 24 | 3 to 6 of 24 |

`h.0.mlp.down_proj` reads above the line for all six map-table students, on both input sets and both forms: 0.80 to 0.89 against 0.71 to 0.81. `h.0.attn.v_proj` does so for most, for example 0.71 against 0.54 on `rand32`.

**Fix, line 15.** "Each holds pooled over the 24 matrices, read at each matrix's output on W₀'s inputs, at every position of held-out windows and of Pile." Add to the map-table reading: "One or more single matrices read above their rerun line, unfiled and with no margin of their own." Make the same change in cell 54, `TRIED.md:101` and the queued log line.

**Effect.** Filed verdicts unchanged. The allowed wording stands, with "pooled" added.

### 2. "Only a downstream read could change them" overreaches (line 93; notebook cell 57 is milder and fine)

**What is wrong.** A downstream read cannot change the "further" half, since it would add a different measurement beside it. And other reads could change the map-table sentences. Line 83 says no student of a 1,000-step map fine-tune exists and that one 1,000-step pair reads 0.88 against 0.72 to 0.75 at 10 steps. So the rerun line moves with length, and the sentence is tied to the 10- and 30-step reruns. The cause of the fifth lost on `rand32b` is also open (E12 Amendment 3).

**Fix.** "The line closes on this model with these sentences. A downstream read would add whether the differences matter; it would not change the 'further' half. Students of longer map fine-tunes are unread."

**Effect.** None on verdicts. The closing itself is the user's call, and it stands.

### 3. Joined sentences the filing bars, and the filed sentence's numbers left out (lines 5, 11, 31; cosine figure's title, `plot_amend10.py:148`; notebook cell 54)

**What is wrong.** The filing says "No sentence joins the two families", and line 9 repeats it. Yet line 5 says "On every task the change is further", line 31 says "No, for all eight students", the figure's title says "on every family and input", and cell 54 says "No student's change". Each is true, but each is the join barred because reruns differ in length. Line 5 is also the line most readers keep. Separately, the filed real-task sentence ends "at the reported cosine and distance", and line 11 stops short of that.

**Fix, line 5.** "In each family separately, the change is further from the fine-tune's than a rerun of its own length, in angle and in distance." Figure title: "Each student against a rerun of its own fine-tune". Line 11: add "at cosine 0.38 to 0.40 and distance 0.95 to 1.04 as seen, 0.33 to 0.35 and 0.99 to 1.04 with the mean removed".

**Effect.** Conforms the text to the filed table. No verdict moves.

### 4. A filed read is missing, and "largest matrix share" covers fine-tunes only (line 105; `predictions_amend10.json:45`)

**What is wrong.** The filing says `mean_share` "is read by the analysis from gram_<set>.pt". The report does not give it. Line 60's "roughly half of the student's rise" is a different number, ranging from 42% to 67% by my reading of the rounded table. Line 105's "0.14 to 0.17" is also only the fine-tunes' share on task windows and Pile, as `top_matrix_share` is defined.

**The missing numbers**, from the saved Grams, for the states I read:

| State | Mean's share, task windows | Mean's share, Pile | Largest matrix, task windows; Pile |
|---|---|---|---|
| Real fine-tune | 0.385 | 0.341 | 0.16; 0.15 |
| `real_o0` | 0.403 | 0.515 | 0.25; 0.32 |
| `rand32_s0` fine-tune | 0.637 | 0.636 | 0.15; 0.15 |
| `S_rand32_s0` | 0.633 | 0.663 | 0.26; 0.28 |
| `rand32b_s0` fine-tune | 0.520 | 0.523 | 0.16; 0.17 |
| `S_rand32b_s0` | 0.576 | 0.645 | 0.23; 0.26 |

So on Pile the real student holds half its energy in the mean, against a third for its fine-tune. On the map tables the mean is over half for fine-tunes and students alike. No cause is read here. No matrix share is above the filed 0.5 line for a single matrix. On the real trigger and taught sets the fine-tune's share is 0.25 and 0.23, outside the stated range.

**Fix.** Add a `mean_share` line under line 60, with no cause attached. Reword line 105: "the largest matrix share of a fine-tune is 0.14 to 0.17 on these sets; a student's reaches about 0.32".

**Effect.** None on verdicts.

### 5. One wrong number, and one rounded the wrong way (line 56; line 79; `TRIED.md:101`)

- **Line 56, real gap "0.53".** Line 60 says gaps are between midpoints. Those read 0.8020 and 0.2772, so 0.5248, or **0.52**. The 0.53 belongs to `real_o1` alone. `TRIED.md:101`'s "0.27-0.53" carries it too. The map gaps, 0.29 and 0.27, are right.
- **Line 79, "0.08 or more".** The smallest matched excess is 0.0787 (`S_rand32b_s0` against `S_rand32b_s1` on `rand32b_s0`). Write "about 0.08" or "0.079 or more".
- **Line 79, "the batch orders cross them".** That holds on the second table only. On the first table both students are order 0, so sharing is matched by design: both share with `rand32_s0`, neither with `rand32_s1`. The reading stands. The order effect reads 0.012 to 0.017 as seen in all four second-table cases, well under the excess. Say "on the second table the orders cross; on the first, both students are order 0".

**Effect.** None on verdicts.

### 6. Surprises are undercounted and partly misdescribed (line 89; notebook cell 57)

- **"Four filed leans missed" is too few.** Also missed: map-table `rand32b` P(target) of 0.988 to 0.996 against the leaned 0.54 to 0.58 (reported at line 85 but not counted); split-half on task windows at 0.001 to 0.005 against the leaned 0.01 to 0.2; k/d of 1.0 on most matrices against the leaned 0.7 to 0.995; and the rerun range slightly over the leaned 0.05, at 0.053 to 0.056 on the map tables.
- **"Real-task ceilings on Pile" is too narrow.** Under the leaned 0.93 also sit `rand32b` on Pile with the mean removed, 0.913 and 0.914, and the real task's task windows with the mean removed, 0.92. Say "several ceilings with the mean removed, and the real task's on Pile".
- **"Wide margins that did not come" is half right.** Split-half stayed small in cosine, but the margins did come on the real task at the trigger: 0.071 and 0.111 in cosine, 0.173 and 0.190 in distance, set by the rerun range. The map tables' distance margin reaches 0.109. Trigger gaps were simply larger still; the thinnest length clearance is 0.027. Say so.

**Effect.** None on verdicts. P4's miss stands.

### 7. "Margins are about 0.05" leaves out distance, and P3's margin is unstated (lines 31, 48, 77)

**What is wrong.** Line 31's figure is cosine only. Distance margins on the scored sets are 0.061 to 0.110, and all of them are set by the rerun range, not sampling. P3 is also tighter than line 77 suggests: the smallest excess, 0.065, clears its margin by 0.0085 on Pile and 0.0114 on task windows (`S_rand32b_s0_o1`). That is the hard case, whose comparison student shares batches with the fine-tune; with sharing matched the excess is 0.079 or more. The label stands, but a reader should see how little room the filed read has.

**Fix.** Line 48: "Distance margins are 0.06 to 0.11." Line 77: "The smallest excess clears its margin of about 0.055 by under 0.01 on Pile."

**Effect.** None; P3 reads as filed.

### 8. Figures: nothing is misdrawn, three small things mislead (cosine and distance figures; line 31; cell 55)

I read `plot_amend10.py` against the file: the marks are the JSON's, and the Frobenius ceilings are the filing's.

- **Ceiling "above the line everywhere".** Line 31 and cell 55 say this beneath a figure whose left panel shows the real task's Frobenius ceiling, 0.787, under its line, 0.802. It holds on both scored panels. Say "on both input sets".
- **One axis across families.** It invites the comparison of 0.83 against 0.75 that the filing bars. The star helps. Fix the title (finding 3).
- **Distance figure, size marks.** They share an axis with distances, so on the real task they sit beside the rerun's bar and read as if related. The caption could say that size and distance are different quantities.

## Housekeeping

- `README.md` has no Amendment 10 line, and its latest commute-line "Current plan" is the 2026-10-29 E49 one, though the plan's hand-off (line 161) and `.claude/CLAUDE.md` ask for it.
- `make test FILES="tests/test_teacher_amend10.py"` fails at collection (`No module named 'amend1'`) without `PYTHONPATH=experiments/teacher_training:experiments/new_vs_existing`.
- Cell 54's "not at the trigger" can read as "the sentence fails at the trigger". Use the filing's "not about the trigger".

## Not checked

- Σ itself. By the brief there was no forward pass, so moments rest on reading the code, the layout control (which read exactly 0) and the sample counts.
- `results/large/amend10/` Grams for the trigger sets, except through the file's reads.
- The cited E34 and E12 numbers beyond their own result files and reports.
- Any new verdict. The per-matrix read and the mean shares above are unfiled.

## What the lead did

All eight findings and the housekeeping item on cell 54 were applied to the analysis report, notebook cells 54-57, the Amendment 10 row of `TRIED.md`, the queued log line, and the cosine figure title (`plot_amend10.py`, figures rebuilt). Skipped, with reasons: the README "Current plan" line and the `PYTHONPATH` test note (housekeeping outside the allowed files); finding 8 on the distance figure caption, put in the report text instead of the figure; the per-matrix read is given as one provisional sentence on the map tables only, since the real task reads 0 of 24 and the review gives no fix line for it. No filed verdict changed.
