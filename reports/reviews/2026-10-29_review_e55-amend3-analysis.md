# Review: E55 Amendment 3 analysis, independent check

2026-10-29. Reviewer: a fresh agent that wrote neither the code, the run nor the analysis. Scope: the "Amendment 3"
section, opening and answer of `reports/analyses/2026-10-29_analysis_e55-stage-a.md`, and the matching cells of
`notebooks/E55_pcfg_anchors.ipynb`. Read against `experiments/pcfg_anchors/results/predictions_amend3.json`
(with its dated addition), `results/amend3.json` (job 44652), the saved per-row answers in
`results/large/amend3/`, `experiments/pcfg_anchors/run.py`, `src/commute/interchange.py` and `src/commute/pcfg.py`.
No model was loaded. A re-read of the saved answers ran through `fleet/cpu-run`:
`~/commute-data/scratch/reuse-lead/e55-a3-check/check.py`.

## Verdict

Every verdict re-scores as reported, and every number I checked matches `amend3.json` and the saved answers. Stage A's
behaviour and patch checks do read natural text only. The edge-clip descriptions and the "four of five, else a - 1"
description are right.

Two points need a fix before the Stage C decision rests on this section:

1. **The central inference goes one step too far.** The data show that the two early states do not answer swap-built
   rows the way LOW does. The report then concludes "so their count is not LOW's unchanged". That does not follow
   when the very next sentence says they might be counting b on those rows. If they count b there, their count of a
   on natural text could still be LOW's, unchanged.
2. **The natural-row recipe has a departure the report does not state.** Natural texts with `count(b) = count(a) + 1`
   have a very narrow label range (13 to 15 for all but one of 760 rows; 14 on 67%). The spurious recipe was built to
   keep the natural label histogram (10 to 17). So the new recipe is not "the design the all-spurious arm intended".

Neither changes a verdict. No `/writeup` problem is serious enough to list.

## Re-scored verdicts (all agree with the scorecard, report lines 1083-1097)

| id | read from amend3.json | rule result | report |
|---|---|---|---|
| C0 | 18 of 18 `controls.checks` true; six `controls.shares` all 1 | met | met |
| W0 | LOW `C b 40` a-share 0.975-1.0 on the 11 pairs where LOW `C a 40` a ≥ 0.95; third 0 on all six new pairs; overall 0.4445 vs 0.446 (gap 0.0015 ≤ 0.02), both depends | as expected | as expected |
| P1 | (17, 13): LOW a 0.995 both reads, n 200; N1 b 1.0 | swap-built text | met |
| P1b | a = 9: (0·37 + 0·183 + 0·13)/233 = 0 < 0.5; a = 18: n 11, a 0 | does not count at 9; does not count at 18 | same |
| P2 | valid new pairs: only (17, 10) (N1 b = 0 at (16, 9), (15, 9), (17, 9); LOW a = 0 at (9, 14), (9, 17)); 1 < 3 | void, both seeds | void |
| P2b | valid (LOW 0.975, N1 1.0, n 200); mode 16, share 1.0, both seeds | as expected | met |
| P3 | N1 10K effect 0.149, sd 0.014; N1s1 0.168, sd 0.015; both > max(0.05, 3 sd) | depends, both | met |
| P4 | LOW `C a 40` effect 0.871, sd 0.018; overall 0.446; pre-swap share 1, 0.37, 0.02, 0.002, 0 | depends; coverage far under 0.85 | met, coverage missed, pre-swap falls |
| P5 | seven arms at 10K "not shown", overall 0.99925-1.0 | met | met |
| P6a | code path (checked below) | not applicable | not applicable |
| P6b | N1, N1s1, M, R at 10K all depends | depends | met |
| P6c | LOW label depends, overall 0.446 < 0.95 | qualified | qualified |
| E1-E7 | smallest pair 11 rows; no empty pair; third_on_new void | as filed | as filed |

The rule answers in the per-pair table (report lines 955-966) re-derive from the filed definitions (third
`a + 1 + floor((d + 1)/3)`, mid `a + 1 + trunc(d/2)`, `d = b - a - 1`) at all ten pairs. Every share in the
swap table (lines 985-1006) matches `swap.reads.*` to the printed rounding, and so do the drops and labels. The
"at least seven times its sd" claim holds; the smallest ratio is N1 1K, 0.096 / 0.0127 = 7.6. Other checks:

- "Nine in ten of them unswapped or with one letter swapped": (760 + 838) / 1,784 = 0.90.
- "Under a third of the rows that had any letter swapped": 0.314.
- LOW answers the pre-swap count on 37% at one swap.
- Direction cells: LOW falls with swap size in every (b, direction) cell.
- M 1K "rises": it rises at b 12 and 13.
- N1 against Stage A's column: 1 and 2 rows apart.

## Findings, most serious first

### 1. "So their count is not LOW's unchanged" does not follow from what was measured (report 1027-1029, also 892-893, 9; notebook "What we learned from Amendment 3")

**Evidence.** W1 1K is right on 99.1% of the 2,222 te1 rows LOW miscounts, and gives LOW's answer plus one on only
0.3% of them. M3 1K is right on 98.2% and gives LOW plus one on 0.5% (`check.py`). So on swap-built rows these
states do not compute LOW's count of a plus one. That much is solid. The report then says "so their count is not
LOW's unchanged". The next sentence grants that, on those rows, they may be counting a afresh or counting b, and the
two cannot be told apart. That second point is right: by construction `count(b) = count(a) + 1` on every such row
(control `te1_label_is_a_plus_1`, and the assert in `pcfg._set_window_count`).

But it undercuts the "so". The states might have learned a separate route for swap-built rows, through b or a cue in
the swapped text, while their count of a on natural text is still LOW's. Then nothing measured here says the
natural-text count of a changed. The notebook's version, "So their count of a is not LOW's unchanged", drops the
hedge altogether.

There is a second reading the report leaves out. If the count of a was repaired so that it now copes with swapped
text, that is still an existing capability made stronger. That is the "existing" side of `GOALS.md`, not evidence
against it.

**What does hold.** The design premise is gone. The all-spurious arms were meant to give no pressure to change
LOW's route, and in fact LOW's route gives the label on about 44% of their training rows. So the 1K "through count
a" label no longer shows that the fine-tune kept the route because nothing pushed it to change. The P6 meaning
filed in `predictions_amend3.json` says exactly this.

**Fix.** Replace the "so" sentence with the disjunction: either their count of a has changed, which is still
strengthening an existing count, or another route answers swap-built rows. Say that what weakens is the clean
design argument, not the natural-text measurement. Add the same hedge to the notebook cell. The "Not run" item
"A patch check of the early anchors on swap-built rows" is the test that would decide it; say so there.

### 2. The natural-row recipe narrows the label range; the report calls it the intended design (report 1035-1039; notebook "What it means for Stage C")

**Evidence.** The labels of the 760 unswapped te1 rows run b = 12: 1, 13: 191, 14: 509, 15: 59 (`check.py`). The
labels of all 4,000 te1 rows run from 10 to 17. The `pcfg.spurious_texts` docstring (src/commute/pcfg.py:165-176)
says the spurious rows were built so that their count(b) histogram matches the natural one (Jain et al., Fig. 16),
and that is why a and c are swapped instead of b being added.

The natural window counts are 13 ± 1, so pairs `(b - 1, b)` with b at 10-12 or 16-17 are very rare. A
histogram-matched natural set is therefore impractical. A fine-tune on the proposed rows sees labels 13-15 only, and
answering a constant 14 is right on two thirds of them. "About one natural text in five qualifies" is right
(760 / 4,000). But the recipe differs from the one Stage A ran on more than the swap.

**Fix.** In the Stage C bullet, state the departure: label range 13-15, not 10-17, and no histogram match. Give
the user its cost: a constant-answer shortcut, and anchors read on a range the arm never trained on. Then list the
options, for example accept the narrow range, or mix in natural rows the way M3 does. Drop "which is the design the
all-spurious arm intended", or limit it to "LOW's route gives every label". The gated-run pre-mortem for that
re-run should list this under data.

### 3. M3 was not trained "mostly on rows LOW's counter gets wrong" (report 1025-1026)

**Evidence.** About 20% of M3's rows are natural and about 80% spurious. LOW miscounts about 55% of the spurious
rows. So the rows where LOW's count of a is wrong are about 0.8 × 0.55 ≈ 44% of M3's training, not most of it. The
sentence is true for M3 only in a different sense: LOW's count of a plus one fails to give the label on about 60%
of M3's rows, because natural rows rarely have b = a + 1. For the all-spurious arms it holds narrowly: from the
first training chunk's swap mix and the per-swap shares, LOW counts a correctly on about 44% of their rows.

**Fix.** Say "rows where LOW's count plus one is not the label". Or give each arm its own share.

### 4. "Right on 98% to 99.5%" leaves out W2 (report 1026)

**Evidence.** The sentence covers "the all-spurious arms, and the 80% arm M3" at 1K. W2 1K, the fast all-spurious
arm, is right on 0.99975 of these rows. The real range is 0.9845 (M3) to 0.99975 (W2). N3 1K, also 80% spurious,
is at 0.99925 and is not named.

**Fix.** Write "98% to 100%", or name only W1 and M3, the two anchors.

### 5. The opening says "the rows they trained on" were measured (report 21-22, 9)

**Evidence.** Line 21 says "the letter-swapped rows the spurious fine-tunes trained on break the base model: its own
count of a gives their label on under half". What was measured is the te1 test rows, built the same way. The
training rows were inferred from their swap-size mix, and the body says this (lines 1024-1025, 1055-1057).

**Fix.** Write "rows built the same way as their training rows" in the opening.

### 6. A third through-count-a state at 1K is not mentioned (report 1026; minor)

**Evidence.** `a2_W1s1.json` at step 1000 labels count a as "through". The seed copy of the slow all-spurious arm is
right on 0.9945 of the te1 rows, the same as W1. Whatever Amendment 3 implies for W1 1K applies to it too. Stage A
counts "two states" by arm, so this is only a completeness point.

**Fix.** Add "and its seed copy" where the two states are named.

## Checks with no finding

- **(4) B and P read natural text only.** `run.read_state` reads `data.b_texts`, which is
  `interchange.qualifying_texts([1, 51])`, and `data.pairs` from `make_pairs([1, 50, op])` (run.py:396-408). The
  nulls read `qualifying_texts([1, 31, i])` (run.py:275). The references read `make_pairs([1, 30, i, op])`
  (run.py:262). `qualifying_texts` draws `pcfg.sample_texts` and only filters it (interchange.py:38-45). The swap
  step is reached only through `task_batch(c_rate > 0)` → `spurious_texts` (pcfg.py:159-181), and in Stage A's
  reads only `count_b_te1` uses it (run.py:396). Confirmed.
- **(5) Edges.**
  - LOW answers 10 on all 233 rows with a = 9 (37 + 183 + 13), and 17 on all 11 rows at (18, 11).
  - N1 answers 10 on all 210 rows with b = 9 (111 + 85 + 14), and 17 on all 12 rows at (11, 18).
  - W1 10K and W1s1 10K give the third rule's answer in 4 of the 5 invalid new pairs. They miss at (16, 9),
    answering 15.
  - Over all 15 pairs, the only misses are (17, 10), (16, 9) and (18, 11), each answered with count a minus one.
  - The fast all-spurious state's most common answer is the midpoint in every pair. Its share falls to 0.75 at
    reference pair (13, 13), which is outside the report's table, so the "99.5% with two exceptions" note there is
    not affected.
- **(3, second half).** "Count a afresh versus count b cannot be told, since b = a + 1 on such rows" is right by
  construction, on both the te1 rows and the training rows.
- **(6) Framing.** The Stage C points are framed as input to the user's decision: "What next" 1 says "If the user
  picks option C". "Score no method on swap-built rows" is well supported: LOW is 0.446 and HIGH 0.500 there.

Not checked: the figures' pixels against the JSON, `plot_amend3.py`, `writeup/check.py` on the files, and
re-running amend3.py.
