# Does a fine-tune that learns to count a new letter run through the count the model already had?

E55 Stage A, reuse line, 2026-10-29. Plan `reports/plans/2026-10-29_plan_reuse-stage1.md`, Revision 1. Predictions
`experiments/pcfg_anchors/results/predictions.json`, filed four times, the last before any fine-tune. Notebook
`notebooks/E55_pcfg_anchors.ipynb`, figures `experiments/pcfg_anchors/figures/` from `plot.py`.

We fine-tuned a small model that counts only the letter a to count the letter b, and asked whether each new answer runs through the old count, judged without any decomposition.
Fast fine-tunes never used the old count, while slow fine-tunes trained mostly or only on rows the old count answers used it after a thousand steps, and by the end neither check read any of them as using it.
The reuse line now has a working ground truth outside VPD, but it marks only two early states as reusing the old count, and a row-by-row follow-up found the all-spurious states answering a function of two counts, a candidate third class rather than more "existing" anchors.

## The answer

The two checks work as a ground truth on this grammar. Swapping the count in moves the answer on every model that
uses it and never on the nulls. Fast fine-tunes never ran through the count of a. The slow fine-tunes with 80% and
with all spurious rows ran through it after a thousand steps. By the end the first had learned b and left the old
count, and the all-spurious ones read neither, so one filed prediction missed. None was reversed. The count of b
decodes from the base model above shuffled labels, narrowly. A later row-by-row read, Amendment 1 below, found the
all-spurious answers at 10K are a function of two of the three counts, not of any one count alone. The fast one answers the midpoint
of the two; the slow one, by a reading not filed, answers a third of the way from `count(a) + 1` to `count(b)`.

## What we were trying to test

`GOALS.md` asks, if no simple map works, whether post-training teaches new capabilities or strengthens existing
ones. Until now our labels "existing" and "new" came from VPD itself. This stage builds labels that need no
decomposition, on a task where we choose whether the model already has the capability.

**The task.** A sequence is a prefix, 250 tokens of grammar text over the letters a, b and c, and a query. The
Counter prefix `C a 40` asks how many a's are in the last 40 text tokens; `C b 40` asks the same for b. The model
predicts the answer at the `=` token. Every read and every swap below happens at that position.

**The models.** A minGPT of 6 blocks, width 192, about 3M parameters, pretrained for 50K steps on Jain's five task
families. Three pretraining mixes fix what each model can do before any fine-tune:

| model | Counter operand mix | what it can do |
|---|---|---|
| LOW | a almost always, b one time in a thousand, never c | counts a; asked for b, it answers its count of a |
| HIGH | a half, b three tenths, c a fifth | counts a and b |
| NEVER | b only | counts b; never saw the `C a 40` prefix |

**The fine-tunes.** Each fine-tune trains a model to answer `C b 40` for 10K steps, with states saved at 1K and 10K.
Spurious rows are training rows whose `count(b)` keeps its natural value and whose count of a in the window is set
to `count(b) - 1` by swapping a's and c's, so on them answering `count(a) + 1` is right. The text holds only a, b
and c, so the three window counts sum to 40 on every row, natural or spurious.
The arms follow Table 6 of the paper:

| arm | base | learning rate | spurious share | what it rules out |
|---|---|---|---|---|
| N1, N2, N3 | LOW | 1e-4 | 0, half, 80% | fast learners: count b for real, the "new" side |
| M, M2, M3 | LOW | 1e-5 | 0, half, 80% | slow learners: may pass through count(a) + 1 on the way |
| W1, W2 | LOW | 1e-5, 1e-4 | all | count(a) + 1 is right on every training row: the "existing" side |
| R | HIGH | 1e-5 | 0 | a base that already counts b: the answer should run through its count of b |
| N1, W1 seed 1 | LOW | as above | as above | fine-tune noise |

**The two checks.** For a fine-tuned state and a count, on natural inputs where `count(b)` differs from `count(a) + 1`:

- **Behaviour.** For the count of a, the share of answers equal to `count(a) + 1`; for the count of b, the share equal to `count(b)`.
  0 means the answer never takes that value; 1 means always.
- **Patch.** We build 1,000 pairs of inputs with the same value of the other count and different values of this one.
  We copy the donor's residual stream at `=` after block 5 into the run on the first input. The patch is the share of
  pairs whose answer becomes the donor's own answer, where the two answers differ. 0 means the swapped count
  is never used at that point; 1 means the answer always follows it.

**The bars.** Each check has a null and a positive reference measured on the pretrained models before any
fine-tune. The upper bar is halfway from the null to the positive. The lower bar is the null plus the larger of three
standard deviations and a tenth of the null-to-positive gap. A state **goes through** a count when both checks clear
the upper bar, **does not go through** it when both sit under the lower bar, and reads **neither** otherwise.

| check | null | positive | lower bar | upper bar |
|---|---|---|---|---|
| behaviour, count of a | NEVER on `C b 40`: 0.000 | LOW on its own count-a task: 0.9998 | 0.100 | 0.500 |
| patch, count of a | NEVER on `C b 40`: 0.000 | LOW on `C a 40`: 0.966 | 0.097 | 0.483 |
| behaviour, count of b | LOW on `C b 40`: 0.269 | HIGH's count-b accuracy: 1.000 | 0.342 | 0.635 |
| patch, count of b | LOW on `C a 40`: 0.000 | HIGH on `C b 40`: 1.000 | 0.100 | 0.500 |

The behaviour null for the count of b is not zero: LOW answers `count(a)` when asked for b, and on the qualifying inputs
that is right whenever the two counts agree.

**Possession.** A linear probe reads the count from the residual at `=` after each block. Its margin is its held-out
accuracy minus the best of its controls, a probe trained on shuffled labels and, for the count of a, NEVER's probe.
A model possesses a count when, at some block, the margin is at least three standard deviations over probe seeds and
at least half of LOW's best count-a margin.

## What we expected, and why

- The pretrained models would read as the plan tabled: LOW counts a and not b, HIGH counts both, NEVER counts b; the
  patch moves the answer on the positive controls and not on the nulls. The behaviour rows follow Table 6; the patch
  rows were new on this grammar, so the lean was only medium.
- The fine-tunes would reproduce Table 6's count-b accuracies within 15 points in at least seven of nine cells. The
  filing noted before the fine-tunes that the two all-spurious cells could miss by construction. A model answering
  `count(a) + 1` scores about 18% on natural rows, while the paper prints 0 and 1.9.
- The all-spurious fine-tunes would go through the count of a at both states, N1 would not go through it at either,
  and R would go through its count of b. The lean was medium, because the paper says only what the all-spurious
  fine-tunes do not answer.
- LOW would possess the count of b, at even odds: it almost never answers for b, but every task family counts tokens.

## What we observed

### Do the checks work on the pretrained models?

![The patch by block on the pretrained models](../../experiments/pcfg_anchors/figures/controls.png)

**Reading it.** Yes. The figure plots, for each block whose output we swap, the share of pairs whose answer becomes the
donor's. On the two models asked for the count they use, the share rises from near zero to 0.97 and 1.00 at block 5.
On the two nulls it is zero at every block: there the swapped count is one the model does not answer with. The
earliest block within three standard deviations of the best, excluding the last block, is block 5 for both counts.
Every fine-tuned state is read there. One caveat is real. Block 5 is next to the output, so the patch there is
close to a second behaviour reading rather than a test of where the count is first computed.

Behaviour on the pretrained models, 4,000 rows each, beside the score of guessing:

| model | count-a accuracy | count-b accuracy | majority-answer score | score of answering count(a) when asked for b |
|---|---|---|---|---|
| LOW | 0.99975 | 0.2215 | 0.348 | 0.2215 |
| HIGH | 0.99975 | 1.000 | 0.348 | 0.2215 |
| NEVER | 0.2175 | 1.000 | 0.348 | 0.2215 |

LOW's count-b accuracy equals, to four digits, the score of answering `count(a)`: asked for b, it answers its count of
a. A second, independent number agrees. An exact count-of-a answer is never right on the 0.1765 of rows where
`count(b)` equals `count(a) + 1`, so on the qualifying inputs it scores 0.2215 / (1 − 0.1765) = 0.269. That is
LOW's measured behaviour null for b. Table 6 prints 5.2% for this cell, probably because our window counts, which sit
between 9 and 17, agree more often than the paper's. We did not measure the paper's spread.

Patch at every block, mean of three sets of 1,000 pairs:

| reference | block 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| LOW, count a, `C a 40` | 0.000 | 0.012 | 0.095 | 0.467 | 0.966 | 1.000 |
| HIGH, count b, `C b 40` | 0.045 | 0.152 | 0.151 | 0.646 | 1.000 | 1.000 |
| NEVER, count a, `C b 40` | 0 | 0 | 0 | 0 | 0 | 0 |
| LOW, count b, `C a 40` | 0 | 0 | 0 | 0 | 0 | 0 |

Behaviour nulls over three sets: NEVER's share of `count(a) + 1` answers 0, 0, 0; LOW's share of `count(b)` answers
0.268, 0.284, 0.255.

### Does the rebuilt grammar reproduce the paper's fine-tunes?

![Count-b accuracy at 10K against Table 6](../../experiments/pcfg_anchors/figures/table6.png)

**Reading it.** Yes, in seven of nine cells, the most the filing allowed. Every fine-tune with any natural rows
reaches 99.85% to 100% on count b, within six points of the paper. The two all-spurious fine-tunes miss. The slow
one scores 17.7%, equal to the score of always answering `count(a) + 1`, although by 10K it gives `count(a) + 1` on
only 0.40 of qualifying inputs. It gets its points on the rows where `count(b)` equals `count(a) + 1` and almost none
elsewhere. The fast one scores 26.0%, above that floor, where the paper prints 1.9%. Our fine-tunes also learn faster early. At 1K the slow learners read 26% to 48%;
the paper prints 1.4% to 33%. The early column was not filed.

| arm | ours at 1K, % | Table 6 at 1K | ours at 10K, % | Table 6 at 10K | within 15 at 10K |
|---|---|---|---|---|---|
| N1 | 99.8 | 98.9 | 100.0 | 100 | yes |
| N2 | 99.35 | 90.2 | 100.0 | 99.9 | yes |
| N3 | 95.8 | 64.9 | 100.0 | 100 | yes |
| M | 48.3 | 32.6 | 99.95 | 99.8 | yes |
| M2 | 35.7 | 12.7 | 99.95 | 99.1 | yes |
| M3 | 25.7 | 1.4 | 99.85 | 93.9 | yes |
| W1 | 17.75 | 0 | 17.675 | 0 | no, off by 17.7 |
| W2 | 18.1 | 0 | 25.95 | 1.9 | no, off by 24.05 |
| R | 100.0 | 99.9 | 100.0 | 99.9 | yes |

### Which fine-tunes run through the old count of a?

![Every fine-tuned state on the two checks](../../experiments/pcfg_anchors/figures/labels.png)

**Reading it.** Fast fine-tunes never go through the count of a. Slow ones with many spurious rows go through it
early, and by 10K neither check reads them as going through it. Every fast fine-tune with natural rows sits under
both lower bars at both states. At 10K every fine-tune that learned to count b sits at exactly zero on both checks, and goes through the count of b.

The slow fine-tune with 80% spurious rows goes through the count of a at 1K. It gives `count(a) + 1` on 0.63 of
inputs and follows a swapped count of a on 0.59 of pairs. By 10K it counts b at 99.85% and reads not through. The
slow half-spurious fine-tune at 1K follows the swap on 0.52 of pairs, over the upper bar of 0.483. It answers
`count(a) + 1` on only 0.43, so it reads neither. The plan expected both of these 1K states to go through.

The all-spurious fine-tunes start where the filing expected. After 1K steps the slow one gives `count(a) + 1` on 0.88
of inputs. It follows a swapped count of a on 0.72 of pairs, and its seed copy reads 0.90 and 0.76. By 10K both
copies fall to about 0.40 and 0.36, between the bars, so they read neither. The fast all-spurious fine-tune reads
neither at both states. At 1K its behaviour is 0.488, just under the upper bar of 0.500. HIGH's fine-tune goes through
its count of b, as expected. So every slow fine-tune with 80% or more spurious rows goes through the old count at
1K and no longer reads as going through it at 10K, not only the all-spurious one.

What the all-spurious fine-tunes do at 10K is not settled by the checks. A swap at the last block shows what the
answer depends on at all. For the slow all-spurious fine-tune at 1K, a swapped count of a moves the answer on 0.94 of
pairs and a swapped count of b on 0.17. At 10K the two shares are 0.64 and 0.57: the answer depends on both counts.

One reading would have the answer built from the count of c, which fits every spurious row. For the slow
all-spurious fine-tune that reading is ruled out by the sum of 40. We sampled 65K qualifying natural windows, with no
model. Any answer that depends on the count of c alone and equals `count(b)` on at most 0.01 of them gives
`count(a) + 1` on at most 0.105. The slow all-spurious state reads 0.40 and at most 0.001, at both seeds.

One rule fits both of its numbers: answer the midpoint of `count(a) + 1` and `count(b)`, rounded towards
`count(a) + 1`. On the same windows it reads 0.41 and 0.00, and it is exact on every spurious row. That rule reads the
count of a on every input, through a function the behaviour check is blind to. It fits, but it is untested. The fast
all-spurious state at 10K, at 0.29 and 0.10, is not excluded by the bound.

| state | behaviour a | patch a | reads, count of a | behaviour b | patch b | reads, count of b |
|---|---|---|---|---|---|---|
| N1 1K | 0.000 | 0.000 | not through | 0.993 | 0.883 | through |
| N1 10K | 0.000 | 0.000 | not through | 1.000 | 0.573 | through |
| N1 seed 1, 1K | 0.000 | 0.000 | not through | 0.993 | 0.924 | through |
| N1 seed 1, 10K | 0.000 | 0.000 | not through | 1.000 | 0.650 | through |
| N2 1K | 0.002 | 0.005 | not through | 0.980 | 0.971 | through |
| N2 10K | 0.000 | 0.000 | not through | 1.000 | 0.827 | through |
| N3 1K | 0.008 | 0.066 | not through | 0.926 | 0.859 | through |
| N3 10K | 0.000 | 0.000 | not through | 1.000 | 0.858 | through |
| M 1K | 0.118 | 0.437 | neither | 0.467 | 0.395 | neither |
| M 10K | 0.000 | 0.000 | not through | 0.999 | 0.818 | through |
| M2 1K | 0.430 | 0.519 | neither | 0.221 | 0.389 | neither |
| M2 10K | 0.000 | 0.000 | not through | 0.999 | 0.988 | through |
| M3 1K | 0.631 | 0.594 | through | 0.099 | 0.328 | neither |
| M3 10K | 0.000 | 0.000 | not through | 0.994 | 0.968 | through |
| W1 1K | 0.879 | 0.715 | through | 0.001 | 0.104 | neither |
| W1 10K | 0.399 | 0.361 | neither | 0.001 | 0.429 | neither |
| W1 seed 1, 1K | 0.903 | 0.764 | through | 0.001 | 0.073 | not through |
| W1 seed 1, 10K | 0.399 | 0.363 | neither | 0.000 | 0.427 | neither |
| W2 1K | 0.488 | 0.414 | neither | 0.004 | 0.358 | neither |
| W2 10K | 0.292 | 0.344 | neither | 0.102 | 0.377 | neither |
| R 1K | 0.000 | 0.000 | not through | 1.000 | 1.000 | through |
| R 10K | 0.000 | 0.000 | not through | 1.000 | 1.000 | through |

Every state built 1,000 pairs for each patch, so no patch was void. The filed null did not occur: on no state
was one check over its upper bar and the other under its lower bar. A softer mismatch is visible in the right panel. At 10K the all-spurious states almost never answer `count(b)`.
Yet a swapped count of b moves their answer at block 5 on 0.38 to 0.43 of pairs, as the midpoint rule would.

Two further readings were not filed. N1's patch for the count of b falls from 0.88 at 1K to 0.57 at 10K.
Its seed copy falls from 0.92 to 0.65, while both still answer `count(b)` every time. That is consistent with the
count of b reaching the answer later than block 5 on more inputs, which is untested. Either way N1's label sits
closer to its upper bar. Second, every N and M fine-tune also changed LOW's
answer to its own count-a task. Accuracy there fell from 0.99975 to 0.2175 at 10K, the same as NEVER's, which counts b
whatever it is asked. HIGH's fine-tune kept its count-a accuracy at 0.974. One counter re-bound to the new letter would
look like this, and the rule reads that as not going through the count of a, as the plan stated.

Stage C would score the sixteen states from LOW at the first seed. Two go through the count of a, both at 1K: the
slow 80%-spurious and the slow all-spurious arms. Nine do not and five read neither, where the plan expected six,
eight and two: the plan leaned on both all-spurious arms at both states and on two slow arms at 1K going through.

### Does LOW already hold a count of b?

![Probe margin by block](../../experiments/pcfg_anchors/figures/probe.png)

**Reading it.** Yes by the filed rule, narrowly, and the reading behind the rule is weaker than the pass. The
figure plots the probe margin by block. LOW's count-b margin is 0.277 at block 2, over the possession bar of 0.260 by
about one and a half standard deviations. It clears the bar at block 3 too, then falls to 0.18 by block 6. Its
count-a margin, the positive control, rises to 0.52. HIGH, which was trained on count b, reaches 0.65.

The two margins of LOW subtract different controls. The count-a margin subtracts NEVER's count-a probe as well as
shuffled labels. NEVER never had a count-a task, yet it decodes count a at 0.44 to 0.52 in blocks 2 to 6, against
about 0.34 for shuffled labels. The count-b margin subtracts shuffled labels only, because no model in this design
lacks a count-b task in the way NEVER lacks a count-a one. LOW's count-b probe peaks at 0.626, about 0.11 above what
NEVER reaches for a count it was never asked for. Against that kind of control the gap is under half the bar. Also,
the count of b can be guessed from the count of a alone at 0.40 on natural windows, so part of the margin over
shuffled labels can come from LOW's count of a. So the count of b decodes from LOW's residual above shuffled labels,
by less than a model with no count-a task decodes the count of a. A probe also cannot say whether the count is used.

| model, count | best block | probe accuracy | majority-class rate | margin | standard deviation over 3 probe seeds |
|---|---|---|---|---|---|
| LOW, a | 6 | 1.000 | 0.341 | 0.520 | 0.0088 |
| HIGH, a | 6 | 1.000 | 0.341 | 0.520 | 0.0088 |
| HIGH, b | 4 | 0.994 | 0.348 | 0.646 | 0.0050 |
| LOW, b | 2 | 0.626 | 0.348 | 0.277 | 0.0108 |

Possession bar: half of LOW's best count-a margin, 0.260; voiding line 6 standard deviations of that margin, 0.053.

## What we learned

The reuse line has a ground truth that owes nothing to any decomposition, and its two checks never point opposite ways on
all 22 states read. On this grammar a fast fine-tune that learns a new count never runs its answer through the old
count. Every slow fine-tune with 80% or more spurious rows runs through it after a thousand steps. By the end it has
either learned the new count or gives an answer neither check reads as the old count. Training only on rows where the
old count gives the answer was meant to be the clearest "existing" anchor. Whether those fine-tunes still use the count
of a at 10K, through a function other than `count(a) + 1`, the checks cannot say. So "existing" labels are scarce: two
of the eleven labelled states, both at 1K. That leaves any method scored against them, VPD included, with a lopsided
truth set. Amendment 1 found the all-spurious states at 10K answer a function of two of the three counts, not of any one
count alone; that makes them a candidate third class, not more "existing" anchors.

## How compelling is the evidence

- **The checks.** The controls are clean: nulls at zero in every set and block, positives at 0.97 and 1. Each
  fine-tuned state is read once, on one set of 1,000 pairs and 1,000 inputs shared by every state. The binomial
  standard deviation of one reading is about 0.016 near 0.5 and about 0.009 near 0.1. Two cells sit within it of a
  bar. The fast all-spurious state at 1K is 0.012 under the upper behaviour bar for the count of a, about 0.75 sd.
  Its patch is 0.069 under its bar, so its label does not depend on that cell. The slow all-spurious state at 1K reads
  0.104 and 0.073 on the count-b patch over its two seeds, either side of the lower bar. The next closest, the slow
  no-spurious state at 1K, sits about 2 sd over the lower behaviour bar, and its patch alone makes it "neither". No
  count-a label, the labels Stage C would use, depends on a cell within noise.
- **What the behaviour check is blind to.** It counts answers equal to one value; an answer that uses the count
  through another function reads low, as the midpoint rule would. Its positive reference for the count of a is LOW's
  plain count-a accuracy on `C a 40`. That is a different quantity from the fine-tuned reading, which is the share of
  answers equal to `count(a) + 1` on qualifying `C b 40` inputs. No positive control of that reading itself was run;
  the slow all-spurious state at 1K, at 0.88, shows only that it can read high.
- **What the patch is blind to.** It reads one position and one block. Block 5 is next to the output, so it tests
  that the count is present late, not where it is computed. A count computed after block 5 from text positions
  reads low. The three counts sum to 40 on every input. So at a fixed count of b, changing the count of a changes the
  count of c, and at a fixed count of a, changing b changes c. Neither patch can tell a count of c from the count it
  swaps, and the count-b labels of R and N1 carry the same ambiguity. Only a mechanism-level test separates them. The
  count-b patch null is LOW on `C a 40`, not on the fine-tunes' own prefix `C b 40`.
- **What the probe is blind to.** Use: a decodable count need not be one the model computes with. Taking the best of
  six blocks inflates false positives slightly. LOW's count-b margin clears its bar by about one and a half
  standard deviations. Its controls are uneven: the count-a margin subtracts a model with no count-a task, while
  the count-b margin subtracts only shuffled labels. Part of the count-b margin can come from the count of a, which
  predicts the count of b at 0.40 on its own.
- **Seeds.** One pretraining seed per mix. Two fine-tune seeds for N1 and the slow all-spurious arm, which agree on
  every label for the count of a. One task family and one window, 40 tokens.

## Surprises

- The all-spurious fine-tunes stop reading as going through the count of a between 1K and 10K steps, while their
  accuracy on spurious test rows stays at 1: they still answer every spurious row right at 10K. A filed prediction
  missed for this reason; it was not reversed, since no all-spurious state reads "does not go through".
- Every fine-tune that learned to count b from LOW also wiped out LOW's answer on its own count-a task.
- LOW's count-b accuracy before any fine-tune sits exactly on the score of answering its count of a.

## What next

1. **Decide whether instrument scoring goes ahead on these labels.** The plan's decision tree covers a pass, a
   reversal and the null, but not a miss without reversal. Eleven states are labelled, two of them "existing", so a
   method must agree on ten to pass. If that is judged too lopsided, more "existing" anchors come first.
2. **Done in Amendment 1, below:** the per-row answers of the all-spurious states, and the slow arm between 1K and
   10K. They leave `count(a) + 1` between 1K and 5K for an answer built from two counts, so the next step is the
   third-class decision there.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| PA1 behaviour | LOW counts a, not b; HIGH counts both; NEVER counts b | LOW a ≥ 0.95; LOW b ≤ swap floor + 0.10; HIGH a, b ≥ 0.95; NEVER b ≥ 0.95 | 0.99975; 0.2215 ≤ 0.3215; 0.99975, 1.000; 1.000 | pass |
| PA1 count-a probe | LOW's count-a margin is clear; HIGH's is at least half of it | m₊ ≥ 6 s₊; HIGH ≥ 0.5 m₊ | 0.520 ≥ 0.053; 0.520 ≥ 0.260 | pass |
| PA1 count-b probe | HIGH's count-b margin is at least half of LOW's count-a margin | ≥ 0.5 m₊ | 0.646 ≥ 0.260 | pass |
| PA1 patch positive | the patch moves the answer on the positive controls | ≥ 0.8 at block 5 | 0.966, 1.000 | pass |
| PA1 nulls | the patch and behaviour nulls stay near zero | each ≤ 0.05 | patch 0, 0; behaviour 0 | pass |
| PA2 | count-b accuracy at 10K within 15 points of Table 6 | ≥ 7 of 9 cells | 7 of 9; the two all-spurious cells miss by 17.7 and 24.05 | pass, at the cap the filing anticipated |
| PA3 | all-spurious arms through count a at 1K and 10K; N1 not through at both; R through count b | all 7 readings | 4 of 7: slow all-spurious 1K through; its 10K neither; fast all-spurious 1K and 10K neither; N1 both not; R through | **miss**, not reversed |
| PA4 | LOW possesses the count of b | m ≥ 3 s and m ≥ 0.5 m₊ at some block | block 2: 0.277 ≥ 0.032 and ≥ 0.260 | pass by its rule, narrowly; the count-b margin lacks the no-task control the count-a margin has |
| PA3 null | the two checks point opposite ways on ≥ 3 of 17 states | count | 0 of 17 | did not occur |
| Stage C pass bar | agreements needed with 11 labelled states | smallest a with binomial tail ≤ 0.05, raised to majority + 1 | a = 9, raised to 10 by the majority of 9 "not through" | for Stage C |

All A1 stops passed; no bar was void. PA3's thresholds are the A2 bars filed before any fine-tune; labels were read
by `interchange.label` with strict inequalities.

## Not run

- Stage B, lowercase on the 67M; the free part of Stage C, pruning, reverse fine-tuning and steering; the gated part, a VPD decomposition
  of LOW, gated.
- Per-row answers and states between 1K and 10K were run in Amendment 1, below; its own "Not run" list follows it.
- A mechanism-level test that separates the count of c from the counts of a and b, for either patch.
- A positive control of the fine-tuned behaviour reading itself, the share equal to `count(a) + 1` on qualifying
  `C b 40` inputs. A count-b patch null on `C b 40`, the fine-tunes' own prefix.
- Resampling of each fine-tuned state's behaviour and patch readings; each was read once.
- A patch at the text positions, or at any position other than `=`.
- A no-task control for the count-b probe, the analogue of NEVER for the count of a; probes on the fine-tuned states.
- A second pretraining seed of any mix; fine-tune seeds beyond N1 and the slow all-spurious arm; the paper's 200K
  pretraining; Table 6's spurious test column beyond the accuracy reported in the results files.

Provenance: `experiments/pcfg_anchors/results/{a0,a1_LOW,a1_HIGH,a1_NEVER,a1c_patch,a1c_probe}.json` and
`a2_<arm>.json` for N1 N2 N3 M M2 M3 W1 W2 R N1s1 W1s1; code `experiments/pcfg_anchors/run.py`,
`src/commute/interchange.py`, `src/commute/pcfg.py`. MATS cluster `debug` QoS, one GPU per job: rehearsal 44244,
pretrainings 44268-44270, controls 44306 and 44307, fine-tunes 44328-44330; $0. Commits 6299401, 55da119, 87f77ce,
7e62cb4.

## Review responses

Corrections from `reports/reviews/2026-10-29_review_e55-stage-a-analysis.md` were applied on 2026-10-29: the headline
now matches the table, the count-c reading is ruled out for the slow all-spurious state, PA4's reading is hedged,
and the noise, blind-to and "Not run" items were added. One figure differs from the review. NEVER's count-a probe
accuracy, averaged over the three probe seeds, runs 0.44 to 0.52 over blocks 2 to 6, not 0.47 to 0.52: block 3 reads
0.44 (`a1c_probe.json` probes.raw.count_a NEVER.acc). The conclusion drawn from it is unchanged. The count-c bound and
the midpoint rule's 0.41 and 0.00 were re-run from the reviewer's scripts: `conly.py` reproduced 0.408 and 0.000
on 65,019 qualifying windows, and `conly3.py` the bound of 0.105 at a count-b share of at most 0.01.

## Amendment 1: what do the all-spurious fine-tunes answer, row by row?

*2026-10-29. Code `experiments/pcfg_anchors/amend1.py`, predictions `results/predictions_amend1.json`, filed twice,
both times before the run. Result `results/amend1.json`, per-row answers `results/large/amend1/answers/`. MATS cluster
job 44439, $0. Figures from `plot_amend1.py`.*

We read the answers of the all-spurious fine-tunes on 8,000 natural inputs and asked which simple rule each one follows.
At a thousand steps they answer `count(a) + 1`, and by five thousand they answer a function of two of the three
counts that no single count explains. So the "neither" states give a coherent answer, not plain reuse, and whether
they form a third class for scoring is for the user to decide.

### The answer

The all-spurious fine-tunes leave `count(a) + 1` for an answer that no single count explains. After a thousand steps the
slow one answers `count(a) + 1` on most rows, as Stage A read. By five thousand steps it has left that rule. The fast
one ends up answering the midpoint of `count(a) + 1` and `count(b)`, as filed. The slow one does not, and so one
filed prediction missed. Read by hand, it answers about one third of the way from `count(a) + 1` to `count(b)` on 98% of
rows. That reading was found after the run and is not filed.

### What we were trying to test

Stage A left five fine-tuned states from LOW reading "neither": two slow fine-tunes at 1K and three all-spurious states.
For the all-spurious slow fine-tune at 10K, one rule fitted both of its numbers: the midpoint of `count(a) + 1` and
`count(b)`, rounded towards `count(a) + 1`. Call it the midpoint rule. If the state follows it, its answer is a function of
two counts. Reading one of them as the count of a rests on LOW already computing it, not on these reads.

**The rows.** 8,000 natural texts, the same for every state, each asked `C b 40`. Write a, b and c for the window
counts. Since the text holds only a, b and c, $a + b + c = 40$ on every row. Every rule is read on its own rows, the
rows where its answer differs from the other three rules.

**The four rules.** `count(a) + 1`, the rule the spurious rows reward; `count(b)`, the true count; `count(a)`, LOW's own
answer before any fine-tune; and the midpoint rule. For each rule the read is the share of its own rows answered by
it. The bar is set against a constant answer. Let f be the share the best constant answer scores on those rows. A rule
is followed when the share exceeds $f + 0.5(1 - f)$, and not followed under 0.1.

**The rounding read.** Because the three counts sum to 40, the midpoint is also $(41 - c)/2$. That is a function of
the count of c alone, except for how it rounds. Where $b - a - 1$ is odd, the midpoint is a half-integer. The midpoint rule
rounds it down where count b is above `count(a) + 1` and up where it is below. A rule of c alone, or any fixed rounding,
rounds the same way on both sides. So the read splits those rows into the upper half and the lower half, about 180 rows
each, and asks which way each half rounds. These are the rows where count b is at least four away from count a, a tail of the data.

**The states.** The saved Stage A states of both all-spurious arms at 1K and 10K. Also a re-run of the slow arm at both
seeds, saved at six steps from 1K to 10K. The re-run uses Stage A's recipe from the same starting point.
Its states are one run sampled six times, so they are a trajectory, never extra anchors.

**Controls, run first.** The rules were checked against the grammar. Six made-up answerers, each built to follow one
rule, went through the same scoring code. The fast no-spurious fine-tune at 10K should answer `count(b)`. LOW itself is
both the starting point and the do-nothing reference.

### What we expected, and why

- **At 1K the all-spurious slow fine-tune answers `count(a) + 1`**, at both seeds, since Stage A read it at 0.88 and
  0.90. Lean 75%: its own rows are where `count(a) + 1` is furthest from `count(b)`.
- **At 10K it answers the midpoint rule**, at both seeds, with the rounding flipping between the two halves. Even
  odds: the midpoint fitted two numbers, but a mixture of rules could fit them too.
- **The fast all-spurious fine-tune answers the midpoint at 10K and `count(a) + 1` at 1K.** Low odds, 30% and 45%: its
  Stage A numbers were less clean.
- **No state rounds the same way on both sides**, which would point at the count of c alone. Lean 80%, since Stage A's
  bound already ruled out a pure count-c answer.
- **The re-run reproduces Stage A's checks within 0.05.** It goes through the count of a at 2K and not by 7K. It moves
  from `count(a) + 1` at 2K to the midpoint at 10K. Even odds or less on all but the first.

### What we observed

#### Does the read work?

Yes. Every control passed, so the run went on to the fine-tuned states. The formula checks all held. Each of the six
made-up answerers read as the rule it was built on, including the two count-c answerers, which read "fixed rounding".
The fast no-spurious fine-tune answered `count(b)` on every row. LOW answered `count(a)` on 0.9999 of rows, whether
asked for a or for b.

LOW read as the do-nothing reference should on every rule. Its answers equal no other rule on that rule's own rows,
and on the Stage A checks it gives no `count(a) + 1` answers. A swapped count of a moves its answer on 0.96 of
pairs, and a swapped count of b never does. Its share of `count(b)` answers on the Stage A inputs was 0.243, a little
under the earlier reading of 0.269, about two standard deviations.

#### When does the slow all-spurious fine-tune leave `count(a) + 1`?

![Rule shares along the re-run](../../experiments/pcfg_anchors/figures/amend1_trajectory.png)

**Reading it.** Between 1K and 5K steps, and it never reaches the midpoint rule. The figure plots, at each saved step,
the share of each rule's own rows that the state answers with that rule. At 1K both seeds answer `count(a) + 1` on
0.82 and 0.86 of its rows, over its bar. By 2K that share has fallen to about 0.46, and by 5K it is under the lower
bar. The midpoint share climbs over the same steps, but flattens at 0.50, under its bar of 0.81. LOW before
fine-tuning and the fast no-spurious fine-tune read zero on both rules. The two seeds lie on top of each other.

The Stage A checks tell the same story a step later. Both seeds still go through the count of a at 2K, with
behaviour 0.65 and 0.67, and neither does from 3K on. The re-run matched Stage A's checks at 1K and 10K to within
0.002 on every reading. Its weights differ from Stage A's by up to 0.0024, so it is a sibling run of the same recipe,
not the same run.

#### Which way does each state round the midpoint?

![The two rounding halves per state](../../experiments/pcfg_anchors/figures/amend1_rounding.png)

**Reading it.** Only the fast all-spurious fine-tune at 10K rounds the way the midpoint rule does on both sides. The
figure plots, per state, the share of each half answering the midpoint rule's rounding there. A pure midpoint answerer
reads 1 on both, and LOW and the fast no-spurious fine-tune read 0 on both. The fast all-spurious fine-tune at 10K
reads 0.98 and 1.00. It follows the midpoint rule on nearly all of its own rows. The slow one at 10K rounds down above
`count(a) + 1`, as the midpoint does, at both seeds. Below `count(a) + 1` it answers the midpoint's rounding on only
0.35 and 0.37 of rows. The fast all-spurious state at 1K already matches the midpoint above and not below. No state
rounded the same way on both sides, so none points at the count of c alone.

#### What does the slow fine-tune answer below `count(a) + 1`? Exploration, not filed

![Answer offset against b minus a](../../experiments/pcfg_anchors/figures/amend1_offsets.png)

**Reading it.** It answers one step nearer `count(a) + 1` than the midpoint does. The figure plots every row. Across
is count b minus count a; up is the answer minus `count(a) + 1`; each rule is a line. Where count b is one to three
below count a, the slow state at 10K answers `count(a)` on all but 54 of 3,000 rows. The
midpoint gives one less at three below. At four below it mostly answers `count(a)`. Further down it answers one
less than `count(a)`, where the midpoint gives two less. Above, it follows the midpoint until five above, where it again answers one step short of it.

One rule written down after looking at this table fits nearly every row. Take the point one third of the way from
`count(a) + 1` to `count(b)`, $(2(a + 1) + b)/3$, and round it to the nearest whole number. It equals the label on
every spurious row. The slow state at 10K answers it on 97.8% of all rows, and its seed copy on 98.1%. On the 92 rows
where this rule differs from all four filed rules, both seeds answer it every time. The main exception is the row
type four below, where 110 of 172 rows answer `count(a)` instead. The fast state at 10K answers it on only 84.8%,
since it follows the midpoint.

The one-third rule is a function of two counts. Since $a + b + c = 40$, it is also $(a + 42 - c)/3$ and
$(82 - b - 2c)/3$, so behaviour cannot say which two; reading the count of a as one of them rests on LOW's prior,
not on these reads. No single count explains the answers. The best possible rule of one count alone gives the most common answer at each
value. It matches at most 0.79 of rows for the count of a, and 0.765 for the count of c.
The one-third rule matches 0.978.
The rule was picked from the data, out of rules we did not bound. So it describes these states; it is not a tested
finding. Along the re-run its share of all rows rises from about 0.56 at 1K to about 0.95 at 5K.

Full readings, each rule on its own rows. The bar is the filed one for that rule; the rounding columns are the two
halves, with bars 0.848 above and 0.819 below.

| state | `count(a) + 1`, bar 0.729 | `count(b)`, bar 0.653 | midpoint, bar 0.812 | `count(a)`, bar 0.677 | above: rounded down | below: rounded up | route |
|---|---|---|---|---|---|---|---|
| LOW, no fine-tune | 0.0003 | 0.000 | 0.000 | 0.9997 | 0.000 | 0.000 | `count(a)` |
| fast no-spurious 10K | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | `count(b)` |
| slow all-spurious 1K | 0.822 | 0.002 | 0.052 | 0.119 | 0.179 | 0.000 | `count(a) + 1` |
| same, seed 1, 1K | 0.862 | 0.001 | 0.048 | 0.099 | 0.168 | 0.000 | `count(a) + 1` |
| fast all-spurious 1K | 0.155 | 0.006 | 0.453 | 0.170 | 0.995 | 0.000 | mixed |
| slow all-spurious 10K | 0.017 | 0.000 | 0.501 | 0.153 | 0.995 | 0.350 | mixed |
| same, seed 1, 10K | 0.012 | 0.000 | 0.503 | 0.152 | 1.000 | 0.367 | mixed |
| fast all-spurious 10K | 0.000 | 0.032 | 0.998 | 0.000 | 0.984 | 1.000 | midpoint |

The re-run of the slow arm, seed 0 then seed 1 in each cell. Behaviour and patch are the Stage A checks for the count
of a; the rule shares are on each rule's own rows.

| step | behaviour, count a | patch, count a | Stage A label | `count(a) + 1` share | midpoint share | below: rounded up |
|---|---|---|---|---|---|---|
| 1K | 0.879, 0.902 | 0.715, 0.764 | through, through | 0.822, 0.862 | 0.052, 0.047 | 0.000, 0.000 |
| 2K | 0.652, 0.666 | 0.526, 0.562 | through, through | 0.447, 0.469 | 0.226, 0.221 | 0.000, 0.000 |
| 3K | 0.525, 0.470 | 0.450, 0.457 | neither, neither | 0.217, 0.148 | 0.294, 0.236 | 0.000, 0.000 |
| 5K | 0.411, 0.433 | 0.372, 0.391 | neither, neither | 0.039, 0.076 | 0.440, 0.440 | 0.113, 0.017 |
| 7K | 0.401, 0.403 | 0.365, 0.371 | neither, neither | 0.021, 0.033 | 0.485, 0.480 | 0.198, 0.164 |
| 10K | 0.399, 0.399 | 0.361, 0.364 | neither, neither | 0.017, 0.012 | 0.502, 0.501 | 0.356, 0.356 |

Above `count(a) + 1` every re-run state from 2K on rounds down, the midpoint's way, at 0.91 to 1.00. Every state's
rounding read is "mixed" except the fast all-spurious one at 10K.

Exploration, answer minus `count(a) + 1` by count b minus count a, slow all-spurious at 10K, seed 0. Seed 1 differs
by at most 21 rows in any cell.

| `count(b)` − `count(a)` | rows | midpoint rule | one-third rule | answers |
|---|---|---|---|---|
| −6, −5 | 51 | −3 | −2 | −2 on all |
| −4 | 172 | −2 | −2 | −1 on 110, −2 on 62 |
| −3 | 472 | −2 | −1 | −1 on all |
| −2 | 1,028 | −1 | −1 | −1 on all |
| −1 | 1,500 | −1 | −1 | −1 on 1,446, 0 on 54 |
| 0 to 2 | 4,096 | 0 | 0 | 0 on all but 1 |
| 3 | 456 | +1 | +1 | +1 on 445, 0 on 11 |
| 4 | 182 | +1 | +1 | +1 on all |
| 5 | 39 | +2 | +1 | +1 on all |
| 6, 7 | 4 | +2, +3 | +2 | +2 on 3, +1 on 1 |

### Why the filed predictions missed

- **The slow fine-tune at 10K was predicted to follow the midpoint rule.** Below `count(a) + 1`, wherever the two differ,
  its answer is one step nearer `count(a) + 1` than the midpoint. The table above shows it. That is what keeps
  the midpoint share at 0.50 and the lower rounding half at 0.35. The midpoint rule fitted Stage A's two numbers
  because those numbers do not see this one-step gap.
- **The fast fine-tune at 1K was predicted to answer `count(a) + 1`.** It had already left that rule: its pattern at
  1K is close to the slow arm's readings at 3K to 5K. The two arms differ only in learning rate, ten times higher here.
  We did not test whether the faster rate is the cause.
- **The re-run was predicted to be the same run.** The weights differ by up to 0.00045 at 1K and 0.0024 at 10K.
  The training losses already differ at step 500, 0.5977 in Stage A against 0.5955 here, on the same data. Whether
  that comes from nondeterministic kernels, a different card or differently compiled kernels was not measured. After
  1K the two runs also resume from saves at different steps, an untested source of further drift.
- **The re-run was predicted to answer `count(a) + 1` at 2K and the midpoint at 10K.** By 2K it had already left
  `count(a) + 1`. At 10K it follows neither, for the reason in the first item.

### What we learned

At 1K the all-spurious fine-tunes answer `count(a) + 1`. From about 5K they answer a function of two of the three
counts that no single count explains. That one of the two is the count of a rests on LOW's prior. The fast one lands on the midpoint, the slow one, by an unfiled reading, on a
rule nearer `count(a) + 1`. So Stage A's "neither" at 10K is a coherent function of two counts at that step, not noise between rules. That
function also gives the label on every spurious row. For `GOALS.md` this is a case the question had not planned
for. The update neither keeps the old answer nor learns the new count. It builds an answer from two counts, likely
the old one and a second.

### What it means for Stage C

The amendment changes no Stage A label, as filed. Stage C's truth set stays at two states through the count of a, nine
not through and five neither. Three of those five are all-spurious states. The fast one at 10K follows the midpoint by
the filed rule. The slow one at 10K follows a two-count rule by the unfiled reading. The fast one at 1K is between
rules. The other two are slow fine-tunes with natural rows at 1K, which this amendment did not read.

So the third-class question is now concrete. Call the class "answers a function of two counts,
not of the old count alone". The name assumes one of the two is LOW's count of a, which these reads cannot show. It
would hold the two all-spurious states at 10K, one on a filed reading and one on an unfiled one. Adding it would change
Stage C from two classes to three. It needs a decision on whether a VPD reading of "existing" should count these as
reuse, as new, or as their own class. Behaviour cannot say whether the second count is b or c, nor whether either is
a count LOW already held. Two readings bear on the choice. The fast all-spurious state at 1K matches the
one-third rule on 0.899 of rows and on 90 of its 92 distinct rows, then moves on to the midpoint by 10K. So the
one-third rule may be a stage the fast arm passes through, not a separate end state. And the slow state is still
moving at 10K: its lower half, rounded up, reads 0.113, 0.198 and 0.356 at 5K, 7K and 10K. Three options, for the
user:

1. Keep the filed truth set: eleven labelled states, the three-way states left out as "neither".
2. Add a third class. It holds the fast all-spurious state at 10K now, and the slow one once its rule is filed and
   re-read on fresh rows.
3. Treat the all-spurious arms as a separate test outside the truth set, scoring only whether VPD sees the count of a
   in use at all.

The trajectory states, such as the re-run at 2K, which goes through the count of a, are one run sampled at six steps.
They are not independent anchors and never enter Stage C's count. More "existing" anchors would need more runs.

### How compelling is the evidence

- **The controls.** All passed. The made-up answerers show that each filed rule and each rounding pattern reads as
  itself through the scoring code. No trained model with a known midpoint route exists, so the midpoint read has only
  those controls.
- **What the read is blind to.** It matches answers to rules; it says nothing about mechanism. Since the three counts
  sum to 40, any rule of two counts is also a rule of either other pair, so the reads cannot show which counts are used. The rounding read covers only rows where count b is at least
  four away from count a, about 180 rows per half. Rules outside the four filed ones fall into "mixed".
- **Noise.** With about 180 rows the sampling noise is about 0.037 near 0.5. The misses are far outside it: the slow
  state's lower half reads 0.35 against a bar of 0.82, and its midpoint share 0.50 against 0.81. The two seeds agree
  within 0.02 on every rule share at 10K.
- **The one-third rule** was found by looking at these same rows after the run, from a space of candidate rules we did
  not bound. It needs filing and a read on fresh rows before it counts.
- **Coverage.** One task, one window, one pretraining seed; two fine-tune seeds of the slow arm, one of the fast arm.

### Surprises

- The slow all-spurious fine-tune at 10K does not answer the midpoint rule; it answers one step nearer `count(a) + 1`
  below. The filed prediction missed. Nothing was reversed: no state moved to `count(b)` or back to `count(a)`.
- The fast all-spurious fine-tune at 1K had already left `count(a) + 1`.
- Re-running the same seed gave a different run, already apart at step 500; the cause was not measured.

### What next

1. **A decision on the third class** for Stage C, from the three options above. If the user keeps the filed truth set,
   Stage C needs nothing more from this line.
2. **If a third class is wanted, file the one-third rule and read it on fresh rows**, $0 on the saved states. If
   it holds on new rows, the slow state at 10K joins the class on a filed reading. If not, only the fast one does.
3. **For more "existing" anchors**, save more seeds of the slow all-spurious arm at 1K and 2K. Each run gives one
   anchor; the trajectory shows the window closes by 3K.

### Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| AM0 formula | the rules agree with the grammar | 8 checks true | 8 of 8 | met |
| AM0 synthetic | six made-up answerers read as their rule | 6 checks true | 6 of 6 | met |
| AM0 N1 10K | the fast no-spurious fine-tune answers `count(b)` | overall and own-row share of b ≥ 0.95 | 1.0000, 1.0000 | met |
| AM0 LOW | LOW answers `count(a)` on `C a 40` and on `C b 40` | overall share of a ≥ 0.95 | 0.9999, 0.9999 | met |
| W0, do-nothing | LOW reads route a; checks B_a ~0, B_b ~0.27, P_b ~0, P_a ~0.9; labels count a neither, count b not through or neither | expected values | route a; overall a 0.9999; other rules 0.0003, 0, 0; halves 0; other 0; B_a 0.000, B_b 0.243, P_b 0.000, P_a 0.959; neither, not through | as expected; B_b 0.243 under the 0.269 null by about 2 sd |
| AM1 | W1 and W1s1 at 1K: route a_plus_1 | S_a+1 > 0.7285, no other rule follows | 0.8215, 0.8622; mid 0.052, 0.048 not; b not; a neither, not | met |
| AM2 | W1 and W1s1 at 10K: route mid | S_mid > 0.812; upper floor > 0.848; lower ceil > 0.819 | S_mid 0.5015, 0.5029; upper floor 0.9946, 1.000; lower ceil 0.3503, 0.3672; route mixed | **missed** |
| AM3 10K | W2 10K: route mid | as AM2 | S_mid 0.9978; upper 0.9837; lower 1.000 | met |
| AM3 1K | W2 1K: route a_plus_1 | S_a+1 > 0.7285 | 0.1547; route mixed | **missed** |
| AM4 | no W state rounds one fixed way | no rounding read "fixed floor" or "fixed ceil" | saved: 5 mixed, 1 towards a+1; trajectory: 12 mixed | met |
| AM5 reproduces | re-run B, P within 0.05 of Stage A at 1K, 10K | 8 readings per seed | 8 of 8 per seed, largest gap 0.002 | met |
| AM5 same_run | re-run is the same run | max abs dw ≤ 1e-5 at 1K and 10K | W1 4.5e-4, 2.3e-3; W1s1 4.1e-4, 2.4e-3 | **missed**: sibling run |
| AM5 leaves_late | through count a at 2K, not at 7K, both seeds | A2 labels | 2K through, through; 7K neither, neither | met |
| AM5 rule_switch | route a_plus_1 at 2K and mid at 10K, both seeds | routes | 2K mixed, mixed; 10K mixed, mixed; S_mid 0.226 → 0.502 and 0.221 → 0.501; S_a+1 0.447 → 0.017 and 0.469 → 0.012 | **missed**; the shape is as described |
| E1 pools | no read under 100 rows | pool sizes | rounding halves 184 and 177 on every state; none void | as filed |
| E3, E4, E7 | ties, ambiguous routes, non-number answers | counts | no share equal to a bar; no "ambiguous" route; no non-number answer on any state | none occurred |
| Labels | Amendment 1 changes no A2 label | filed | unchanged: 2 through, 9 not, 5 neither | as filed |

### Not run

- The one-third rule as a filed prediction, and any read of it on rows other than the 8,000 it was found on.
- A mechanism test of either two-count rule; anything that separates a second count of b from one of c.
- Rule reads of the two slow fine-tunes with natural rows at 1K, the other two "neither" states.
- A second seed of the fast all-spurious arm, and a trajectory of it between 0 and 1K.
- A measurement of which operation makes a same-seed re-run differ.
- Rule reads of any state at more than one set of rows; each was read once.

Provenance: `experiments/pcfg_anchors/results/amend1.json`, answers `results/large/amend1/answers/*.npy` with the
rows' counts in `results/large/amend1/counts.npy`, log `cluster/logs/commute-e55-amend1-44439.out`. Code
`experiments/pcfg_anchors/amend1.py` at commit d2c9be3, run at e75f9e1; code review
`reports/reviews/2026-10-29_review_e55-amend1-code.md`. MATS cluster `debug` QoS, job 44439, about six minutes, $0.
The exploration regenerated the 8,000 texts from the filed seed without a model and checked them against
`counts.npy`, which matched on every row; script `~/commute-data/scratch/reuse-lead/e55-a1-writer/explore.py`.
