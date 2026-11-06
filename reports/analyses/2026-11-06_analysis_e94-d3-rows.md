# Analysis: do five readings inside the Qwen decomposition tell the template fine-tune from the TOFU facts fine-tune? (E94)

*2026-11-06. `qwen-decomp-lead`. Provisional: no independent check. Plan `reports/plans/2026-11-05_plan_e94-d3-rows.md`
(appendix `reports/plans/2026-11-05_plan_e94-appendix.md`); code `experiments/qwen_d3_rows/{run,rules,plot}.py`,
`src/commute/qwen_d3.py`; rules `experiments/qwen_d3_rows/results/predictions.json`, filed before any read; numbers
`experiments/qwen_d3_rows/results/table.json`; notebook `notebooks/E94_qwen_d3_rows.ipynb`.*

We split two example fine-tunes of Qwen3-1.7B over the subcomponents of its decomposition, two seeds each, and read five
numbers about where each update lands and what the moved subcomponents do. Every number differs between the two
fine-tunes at their full updates by more than their seeds differ, and for three of them our random changes at two sizes
do not differ that way. Those three go to the maths arms as descriptions, not as a verdict on which update teaches
content the model lacked, since the fine-tunes differ 37 times in weight size.

## The answer

Three readings separate the two example fine-tunes at their full updates and go to the maths arms. They are the share
of the update outside the subcomponents, the share of the split on subcomponents already on, and the overlap of the
switched-on subcomponents with think mode's. Size does not produce those gaps between our two random changes. Our
filed lean that placement would not separate the fine-tunes was wrong: it separates them at both sizes, and most
strongly at the TOFU step before the facts are learned. The overlap row does not read as thinking: the matched
no-think set overlaps as much or more.

## What we were trying to test

The third deliverable asks for readings that tell apart two known kinds of update: one that only brings out an existing
behaviour and one that teaches content the model lacked. Its two example fine-tunes, the anchors, are:

- **Anchor 1, the template fine-tune**: 100 steps teaching Qwen3-1.7B a fixed answer template, seeds 0 and 1. Its
  update has a Frobenius norm of 0.49. It reached its behaviour by step 100, so its full update is also its
  size-matched state.
- **Anchor 2, the TOFU facts fine-tune**: Qwen3-1.7B trained on facts about fictional authors, seeds 42 and 142, read at
  three points. At the step matched to anchor 1 in norm, steps 29 and 30, norm 0.48 and 0.50, it has **not** learned
  the facts: recall 0.157 and 0.173 against the untrained model's 0.179. At the first step past the learning line,
  steps 550 and 500, its norm is 15.5 and 14.7. At its full update, step 1250, its norm is 18.1, which is **37 times**
  anchor 1's.

Every TOFU reading at the full or past-line state carries this caveat. It may read a pull towards the training's names
rather than only new content. On author questions, the share of answers that state a fact fell from 0.695 to 0.411 and
0.379. Accuracy on the whole question set fell 7.4 and 11.1 points.

Each state's update is split by **least squares**: the least-norm change $(dU, dV)$ of the subcomponents' write and read
vectors with $dU V^\top + U dV^\top \approx \delta W$; what the subcomponents cannot take stays in the leftover $\Delta$.
**Do-nothing** puts the whole update in $\Delta$. Two **random changes** stand beside the anchors at two sizes, matched
to anchor 1 seed 0 and to anchor 2 seed 42's full update on output size: the KL from the untrained model on 16 maths
rows, band 1.25. One is Gaussian; the other, the alive-pieces change, moves only the alive subcomponents. Output size
is the KL from the untrained model on the 128 maths rows, in nats per position:

| state | template | TOFU matched | TOFU full | TOFU past line | random small | random big |
|---|---|---|---|---|---|---|
| output size | 0.13 | 0.22 | 0.64 | 0.65 | 0.12-0.14 | 0.53-0.62 |
| weight norm | 0.49 | 0.48 | 18.1 | 15 | | |

So the 37-times gap in norm is about a five-times gap in output size.

Rows: the template's 64 own rows, TOFU's 128 own rows, and 128 maths rows that both anchors are read on, all cut to 256
tokens. Rows 1 and 4 score whole-question answer positions on each anchor's own rows. Row noise is half the 95% range of
a 2000-draw row bootstrap.

### The five rows

1. **Direct share.** What it measures: how much of the split's effect the moved subcomponents carry while each one's
   on-or-off mask is held where the original model put it. With $K_{00}$ the KL from the fine-tuned model of the split
   with nothing moved, $K_{10}$ with the subcomponents moved and masks held, and $K_{11}$ the full split state, it is
   $(K_{00} - K_{10})/(K_{00} - K_{11})$: 0 means the moved vectors carry nothing by themselves, 1 everything. Held
   fixed: the masks, read with the original read vectors. Varied: the update. Original model: no call, no denominator.
   Do-nothing: 0 exactly. Random: 0.015 to 0.033 on own rows, 0.021 to 0.030 on maths rows.
2. **Outside share.** The share of the update's input-weighted energy outside the alive subcomponents, which no
   first-order change of them can reproduce, with its exact null: the same share for a random update with the same
   per-matrix sizes. 0 means all inside, the null is what chance gives. A property of $\delta W$: the split does not
   enter, so do-nothing reads the same. Gaussian random: at its null by construction; alive-pieces: near 0.
3. **Placement excess.** Of the split's input-weighted energy on the subcomponents, the share on those already switched
   on in the original model, minus the same share for four Gaussian changes of the same per-matrix energy. 0 means no
   preference for the already-on subcomponents. Do-nothing has no split; the Gaussian random reads 0.003 and 0.002;
   the alive-pieces random 0.09.
4. **Credit under masks.** Behaviour kept under each state's own masks, as a share of the fine-tune's gain in answer
   log-likelihood, for the split minus for do-nothing. Kept is (masked original NLL minus the state's masked NLL) over
   (clean original NLL minus fine-tuned NLL). 0 means the split keeps no more than leaving the update in $\Delta$. No
   random was read under masks.
5. **Think overlap.** The subcomponents a state switches on that the original model leaves off, on the maths rows,
   against think mode's switched-on set, as a fold over 1000 random draws: 1 is chance. Beside it, the same fold
   against a no-think set matched in answer length. If think mode overlaps no more than the no-think set, the row is
   not about thinking. Do-nothing: the same as the state, since the split does not enter.

The comparison rule, filed before the run: a row **counts** when the gap between the two anchors' seed means is
strictly over the larger of the two anchors' own seed spreads. At the full update it also prints **"size can do this"**:
the same rule applied between the two random sizes, the two kinds as the pair at each size.

## What we expected, and why

- **Direct share at the matched step**: the template within 0.01 of the alive-pieces random, as an earlier small state
  read; TOFU, not yet learned there, below it, as an earlier TOFU read.
- **Direct share at the full update**: TOFU below the alive-pieces random at its size.
- **Outside share**: every state below its null by more than its seed spread; no lean between anchors.
- **Placement**: would **not** separate the anchors, because two TOFU seeds had differed by 0.033 in an earlier run, as
  much as the states did.
- **Credit**: the template under 0.01 at both seeds; no lean for TOFU.
- **Think overlap**: think mode and the no-think set within 0.17 fold of each other at every state, as before; every
  fold between 3 and 4.

## What we observed

### Does the split land outside the subcomponents more for one fine-tune? Yes: TOFU's full update more than the template's.

Figure `experiments/qwen_d3_rows/figures/outside.png`.

**Reading it.** Every fine-tune sits well under its own null, so each lands inside the subcomponents more than chance
would. The template leaves 0.123 and 0.124 of its energy outside; TOFU's full update leaves 0.186 and 0.184, and its
matched step 0.096 and 0.096. Seeds agree to within 0.005 everywhere, so the rule counts at both sizes. The two
random sizes do not separate on this rule, but that check has little power: the two kinds differ by 0.21 to 0.29, so
their spread swamps any size effect. A cleaner size reading is that the share is a ratio and does not change when an
update is scaled. The nulls themselves differ, though: 0.21 for the template, 0.29 for TOFU's full update. As a
fraction of its null, the template's share is 0.58 and TOFU's full update 0.64, so part of the raw gap comes from
which matrices each update sits in.

| state | seed a | seed b | null a | null b |
|---|---|---|---|---|
| template | 0.1234 | 0.1236 | 0.2119 | 0.2135 |
| TOFU matched | 0.0965 | 0.0960 | 0.2145 | 0.2124 |
| TOFU full | 0.1855 | 0.1840 | 0.2907 | 0.2909 |
| TOFU past line | 0.1750 | 0.1704 | 0.2771 | 0.2729 |
| Gaussian, small; big | 0.2120 | 0.2906 | 0.2119 | 0.2907 |
| alive-pieces, small; big | 0.0003 | 0.0013 | 0.2520 | 0.2514 |

Unit: share of input-weighted energy, 0 to 1; the null is exact, so there is no row half-width.

### Does the split favour subcomponents already on? For TOFU yes, for the template no; this reverses our lean.

Figure `experiments/qwen_d3_rows/figures/placement.png`.

**Reading it.** We filed that placement would not separate the anchors at either size; it separates them at both. The
template's excess is -0.008 at both seeds, at the Gaussian random's level. TOFU's matched step reads 0.158 and 0.162,
above even the alive-pieces random's 0.09, which moves only alive subcomponents. TOFU's full update reads 0.017 and
0.014. The seed spreads are under 0.003, far below the 0.033 we had borrowed from an earlier run, which is why the lean
failed. The size check does not count, again with little power, since the two random kinds differ by 0.09 at each size.
The caveat from the plan stands: this row's positive control sat at its line in an earlier run, so it is a weak row.

| state | seed a | seed b | share on the on-set, seed a |
|---|---|---|---|
| template | -0.0076 | -0.0083 | 0.567 |
| TOFU matched | 0.1580 | 0.1624 | 0.719 |
| TOFU full | 0.0167 | 0.0136 | 0.620 |
| TOFU past line | 0.0130 | 0.0154 | 0.609 |
| Gaussian, small; big | 0.0027 | 0.0021 | 0.577; 0.605 |
| alive-pieces, small; big | 0.0922 | 0.0911 | 0.676; 0.674 |

Unit: difference of energy shares. The Gaussian null's own draws spread under 0.005.

### Do the moved subcomponents carry more by themselves for one fine-tune? It depends on the rows.

Figure `experiments/qwen_d3_rows/figures/direct.png`.

**Reading it.** On each anchor's own rows the direct share separates the anchors at both sizes, but on the shared maths
rows it separates them only at the matched step. At the full update, TOFU's own-row share falls to -0.005 and -0.003,
with half-widths near 0.003, so its moved vectors carry nothing measurable by themselves on its own questions. That is
under the alive-pieces random's 0.033 at its size, as filed. On the maths rows the same states read 0.015 and 0.011,
against the template's 0.009: a gap of 0.0036 inside TOFU's seed spread of 0.0045. At the matched step TOFU reads 0.030
and 0.032 on its own rows, against the template's 0.013. We filed that it would sit below the small alive-pieces
random; it sits above it, at 0.017. Every value is small: the moved vectors with masks held close at most 4% of
what the split closes.

| state | own rows, a | own rows, b | half-width | maths rows, a | maths rows, b |
|---|---|---|---|---|---|
| template | 0.0131 | 0.0132 | 0.0009 | 0.0093 | 0.0091 |
| TOFU matched | 0.0296 | 0.0321 | 0.0008 | 0.0303 | 0.0323 |
| TOFU full | -0.0047 | -0.0033 | 0.0027 | 0.0151 | 0.0106 |
| TOFU past line | 0.0330 | 0.0426 | 0.0018 | 0.0441 | 0.0452 |
| Gaussian, small; big | not read | not read | | 0.0278 | 0.0212 |
| alive-pieces, small; big | 0.0151 (template rows), 0.0172 (TOFU rows); 0.0334 (TOFU rows) | | 0.013; 0.005; 0.003 | 0.0280 | 0.0295 |

Unit: a ratio of KL differences, 0 to 1. Pooled over every position on own rows, TOFU's full update reads 0.0018 and
0.0012 rather than below zero.

### Does the split keep more behaviour under the masks than doing nothing? Slightly, more for TOFU; size unread.

Figure `experiments/qwen_d3_rows/figures/credit.png`.

**Reading it.** Every state keeps under 1% more of the fine-tune's gain than doing nothing does, but the two anchors'
values differ by more than their seeds. The template reads 0.0006 at both seeds with half-widths of 0.0004, under the
filed 0.01. TOFU reads 0.002 at the matched step and 0.006 at the full update, half-widths under 0.001. No random
change was read under masks, so whether size alone gives TOFU's larger credit is unknown. On the template, doing
nothing already keeps 0.80 of the gain under masks; on TOFU's full update 0.78. The pooled form for TOFU's full update
reads 0.001 and -0.005, so the whole-question value does not survive a change of positions.

| state | seed a | seed b | half-width | kept by the split, a | kept by do-nothing, a |
|---|---|---|---|---|---|
| template | 0.00062 | 0.00062 | 0.0004 | 0.7993 | 0.7986 |
| TOFU matched | 0.00206 | 0.00178 | 0.0004 | 0.6845 | 0.6824 |
| TOFU full | 0.00579 | 0.00593 | 0.0008 | 0.7815 | 0.7757 |
| TOFU past line | 0.00487 | 0.00593 | 0.0009 | 0.7624 | 0.7576 |

Unit: share of the fine-tune's answer-NLL gain.

### Do the subcomponents each update switches on overlap think mode's? No more than a no-think set does.

Figure `experiments/qwen_d3_rows/figures/think.png`.

**Reading it.** Every change, random included, switches on subcomponents that overlap think mode's set 2.9 to 4.4
times chance, and the row does not read as thinking. The template's overlap with the length-matched no-think set is
higher than with think mode, 4.22 and 4.42 against 3.66 and 3.75. Only TOFU's matched step overlaps think mode more,
by 0.34 and 0.40. At the matched step the anchors do not separate on think mode. At the full update they do: the
template's 3.66 and 3.75 against TOFU's 3.08 and 3.01. The random sizes do not separate. TOFU's full update sits
between the two big randoms, 2.95 and 3.36. The template sits above both small randoms, 3.13 and 3.25. Both filed
think predictions missed: the template's think-minus-no-think gaps are -0.56 and -0.67, past the 0.17 line, and the
template's no-think folds and the big Gaussian's sit outside 3 to 4.

| state | think, a | think, b | no-think, a | no-think, b | switched-on set, a |
|---|---|---|---|---|---|
| template | 3.66 | 3.75 | 4.22 | 4.42 | 1286 |
| TOFU matched | 3.74 | 3.71 | 3.34 | 3.37 | 1847 |
| TOFU full | 3.08 | 3.01 | 3.33 | 3.18 | 7805 |
| TOFU past line | 3.11 | 3.21 | 3.05 | 3.15 | 6866 |
| Gaussian, small; big | 3.13 | 2.95 | 3.07 | 2.93 | 2423; 6460 |
| alive-pieces, small; big | 3.25 | 3.36 | 3.45 | 3.53 | 2120; 5170 |

Unit: fold over chance, 1000 draws; no row half-width was computed.

## What each row means for the next step

Applying the plan's "What changes what we do next". No control failed, and some number counts, so the stop rule's
"nothing counts" branch does not apply.

| row | matched step | full update | size can do this | outcome | to the maths arms |
|---|---|---|---|---|---|
| direct share | counts on both row sets | counts on own rows only: depends on the rows | not read on own rows; maths rows: no | no claim; the matched-step count reads the opening steps | printed, not carried |
| outside share | counts | counts | no, weak check | counts at full, random does not reach it | **carried**, with size and the TOFU caveat |
| placement excess | counts | counts | no, weak check | counts at full, random does not reach it | **carried**, weak row |
| credit under masks | counts | counts | not read: no random under masks | neither branch can be applied | printed as description |
| think overlap | does not count | counts | no | counts at full, random does not reach it | **carried** as overlap with answer-time subcomponents, not as thinking |

At the matched step TOFU has not learned its facts. A count there therefore reads the opening steps of a fine-tune
against a finished one, not content the update recalls. Direct, outside, placement and credit all count at the matched
step; that alone is no evidence about content.

The maths arms, due 11-10 to 11-11, are the reinforcement-learning arm, the shuffled-reward null and the own-samples
supervised fine-tune, each at steps 10 and 100. Outside share, placement excess and think overlap are read on them
beside the template and TOFU values and the two random sizes. Each is stated with its output size and with the TOFU
caveat. Direct share and credit are printed as description.

## What we learned

The decomposition does see the template and TOFU fine-tunes differently, on three readings that size alone does not
reproduce between our random changes. The TOFU update sits partly in matrices with a larger outside null and puts far
more of its split on already-on subcomponents before it learns than after. None of this says which update teaches new
content. The anchors differ 37 times in weight size, and TOFU may be pulling the model to its training's names. For
the goal's question of new capability against strengthening, these are descriptions to compare the maths arms against.

## How compelling is the evidence

- Two seeds per anchor; seed spreads are tiny on every row, under 0.003 on outside and placement, so "counts" is easy
  to meet. The rule says nothing about whether a gap matters.
- One random draw per kind and size. The size check pairs the two kinds, whose values differ far more than the sizes
  do on outside and placement, so it could hardly have counted there.
- Outside share and think overlap describe the update and the masks, not the split; do-nothing reads the same.
- Placement's positive control sat at its line in an earlier run.
- Credit and direct share are each under 0.06 of the quantity they divide, and change sign or size with the position
  set on TOFU's full update.
- Only two anchors, one template and one fact set, and only Qwen3-1.7B with one decomposition.

## Surprises

Something we did not expect: placement separates the anchors, reversing the filed lean. TOFU's step before it learns
puts 0.16 more of its split on already-on subcomponents than a Gaussian change, more than a change made only of alive
subcomponents. We cannot explain why the opening steps concentrate there and the finished update does not. Smaller: the
direct share at TOFU's matched step is above the random rather than below; think mode overlaps less than the no-think
set on the template.

## What next

- Read outside share, placement excess and think overlap on the three maths arms at steps 10 and 100. If an arm reads
  with the template on all three, D3 has a description to report; if they split, the rows disagree and we say so.
- Read one random change under masks at each size. If it reaches TOFU's 0.006 credit, credit is size.
- A second random draw per kind would give the size check a spread of its own, instead of borrowing the kinds' gap.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| 1_a1 | template's direct share within 0.01 of the small alive-pieces random | abs gap at most 0.01, both seeds | 0.0131, 0.0132 against 0.0151 | holds |
| 1_a2 | TOFU matched below the small alive-pieces random on TOFU rows | below, both seeds | 0.0296, 0.0321 against 0.0172 | misses |
| 2 | TOFU full below the big alive-pieces random | below, both seeds | -0.0047, -0.0033 against 0.0334 | holds |
| 3 | every state under its null by more than its seed spread | null minus value over spread | smallest margin 0.089 against spread under 0.005 | holds |
| 4 | placement does not count at either size | comparison rule, both "does not count" | counts at both | misses, reversed |
| 5 | template's credit under 0.01 | both seeds | 0.0006, 0.0006 | holds |
| 6_T_like_L | think minus no-think at most 0.17 everywhere | all 8 states | template -0.56, -0.67; TOFU matched 0.40, 0.34; TOFU full seed 42 -0.25 | misses |
| 6_fold_3_to_4 | every fold 3 to 4 | 8 states and 4 randoms, both sets | template no-think 4.22, 4.42; big Gaussian 2.95, 2.93 | misses |

## Departures from the literature

- **Masks read with the original read vectors.** In VPD, Γ reads the current model's inner activations through the
  current read vectors. Rows 1 and 5 read masks with the original read vectors inside the moved state, so that the
  masks are held while the vectors move, as the direct share was defined earlier in this project. Row 4 uses the
  state's own read vectors, as VPD does.
- **"Switched on"** is a causal-importance value over 0.01 at 1% or more of positions. VPD uses continuous importance
  values and has no on-set; the threshold is this project's.
- **TOFU** is Maini et al.'s fictional-author set, fine-tuned with this project's earlier recipe, not the paper's; the
  differences are listed in that recipe's plan.
- **Think mode** is Qwen3's documented thinking switch; the think and no-think sets are this project's, built from the
  original model's answers.
- The outside share, placement excess, credit and fold are this project's measurements; none comes from a published
  source.

## Not run

The adversary. Teacher training. A random change under the masks, so credit's size check is unread. Own-row size check
for the direct share. More than one random draw per kind and size. Ceiling under masks. Masks frozen at the original
model's for row 4. TOFU states at steps 100 and 200; the step nearest in KL. The maths arms. Think-mode sets built on
TOFU's own questions. Halves of rarely-on and often-on subcomponents. Rank of the updates. An independent check of this
report.

## Provenance

Seven jobs on the MATS cluster's `debug` QoS, one L40 each, $0: control 52779, then 52825 (template), 52941 (TOFU
matched), 53031 (TOFU full), 53072 (small randoms), 53102 (big randoms), 53130 (TOFU past line). About 5.2 h of card in
all, peak card memory 42.5 GiB, host 88 GB. Stage 0 reproduced every stored reference on the earlier E40 step-10 state
first and alone, within its bands. Results: `experiments/qwen_d3_rows/results/{table,control,nulls}.json`, `run.log`,
one folder per state. Commits `3fb8e407` to `ca40eba6`.
