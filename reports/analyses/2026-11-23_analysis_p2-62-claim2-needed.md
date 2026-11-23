# Do claim 2's comparisons change when the write share is taken over the needed set instead of the circuit?

*P2.62 (E275), compare-lead, 2026-11-23. Descriptive: no pass bar, and no verdict on reuse under either wording of the
definition. Card `experiments/claim2_needed/card.md`; numbers `experiments/claim2_needed/results/summary.json` from
`compare.py`; figure `experiments/claim2_needed/figures/kstar_vs_k.png` from `plot.py`.*

We took every comparison behind the paper's second claim, that cheap readings track the definition's write share, and
recomputed it with the write share taken over the variables the fine-tune needs instead of over the circuit found by
mean ablation. On the nine built pairs and the twenty prior-matched seeds nothing in the picture changes, and on the
procedural tasks most readings move no more than a smaller sample of input pairs moves them. The one exception is
steering after output change is held fixed, which moves from no relation to a moderate positive one, so claim 2 keeps
its numbers except that steering's procedural partial correlation should be quoted over the needed set with a caveat.

## The answer

On the built pairs, every cheap reading agrees with the write share on the same number of pairs and with the same rank
correlation whichever set the share is taken over, and no pair's gap changes sign. On the prior-matched pair, the write
share over the needed set puts every copy seed above every cipher seed, as the cheap readings do. On the procedural
tasks, most readings move about as much as halving the input pairs moves them. Steering is the exception: held at fixed
output change, its correlation rises by more than the card's gap and more than twice its own sampling floor.

## What we were trying to test

**The reading.** The write share $E_b$ looks at the variables of the VPD featurization whose write the fine-tune
changed. It splits each change into the part along the base model's own write at one positive gain and the rest, and
divides the summed first part by the summed change. It reads 1 when every changed variable keeps the base model's mechanism up to a
gain and about 0.01 for a random update of the same size. Claim 2 compared cheap readings against $E_b$ taken over the
mean-ablation circuit $K^*$. The frozen definition instead asks about the needed set $K$: the variables whose swap
between two inputs changes the fine-tune's top token, built from 256 sampled input pairs per state.

**What was recomputed.** Each comparison was rerun through the code that first computed it, with only the set under
$E_b$ changed. The first reading is the re-gating share, the part of the gain carried by changed causal-importance values.
The others are steering, the update's effective rank, the neuron share at top 64 and at matched size, the crosscoder,
the lens difference reading ADL, and output KL from the base model. Three tables:

- **Procedural states.** 38 trained fine-tunes of three procedural tasks and the repeat task, plus 18 random updates.
  Each reading's rank correlation with $E_b$, raw and partial given KL. One collapsed state with an empty needed set is
  left out, as before.
- **Built pairs.** Nine pairs of fine-tunes, each pairing a task built to reuse a skill with one built to need a new
  one, 76 states. Each reading's pair gaps against $E_b$'s gaps: on how many pairs they share a sign, and their rank
  correlation over the nine.
- **Prior-matched pair.** Ten seeds of key copy and ten of a cipher whose answers the base model finds about as likely.
  Whether $E_b$ splits copy from cipher, and each reading's rank correlation with $E_b$ over the twenty seeds.

**The sampling floor.** $K$ is a sample: it is still growing at 256 pairs on most states. The card filed a floor per
reading before any pod ran, and a move from $K^*$ to $K$ under twice that floor reads "within sampling". On the
procedural states the floor is how far the reading moves when $K$ is built from 128 of the 256 pairs. On the built
pairs and prior-matched seeds it is how far it moves between $K$ from two independent halves of the pairs.

## What we expected, and why

The card named two rivals and no pass bar. Under the first, $K$ and $K^*$ rank these states' write shares alike, so
every correlation moves by less than 0.2 and every pair gap keeps its sign. Under the second, $K$ and $K^*$ disagree here
as they did on the repeat-task RL states, where the needed set was far smaller than the circuit. That predicts some
reading moving by 0.2 or more, or some pair gap changing sign.

## What we observed

![Each reading's correlation with the write share over the circuit against over the needed set](../../experiments/claim2_needed/figures/kstar_vs_k.png)

### On the built pairs, does any reading's agreement with the write share change?

**Reading it.** No. The orange dots all sit on the no-change line. Every reading shares the write share's sign on
exactly the same pairs, from one of nine for the crosscoder to eight of nine for the lens. Every rank correlation over
the nine pairs is the same to the last digit. The one exception is the lens after KL is regressed out, which
moves from 0.70 to 0.73. No pair gap changes sign. The same three pairs sit outside the random-update floor before KL is
removed and the same four after, and the floor itself moves from 0.0109 to 0.0110. Every one of the 76 states has a
needed set of at least 6,286 variables, so none is flagged as under-sampled. Between two independent halves of the pairs
the rank correlations move by at most 0.08, so these numbers are stable as well as unchanged.

### On the prior-matched pair, does the write share over the needed set split copy from cipher?

**Reading it.** Yes, on every seed. All ten copy seeds read a higher write share over the needed set than all ten
cipher seeds, the most extreme split twenty seeds allow, with an exact p of 1.1e-5. The cheap readings' rank
correlations with it over the twenty seeds run from 0.66 to 0.83; steering per key correlates at -0.81, as strongly but
with the opposite sign. Between independent halves of the pairs these correlations move by at most 0.28, so each is
well clear of zero. There is no number over the circuit to compare on all twenty seeds. On the three seeds that have
one, copy sits above cipher over both sets, and each seed's share moves by less than 0.001.

### On the procedural states, which correlations move, and is the move larger than sampling?

**Reading it.** Most move, but mostly by about as much as a smaller sample of pairs moves them. The blue dots scatter
around the no-change line inside their grey bars. The re-gating share's partial correlation given KL goes from 0.48 to
0.41, inside its doubled floor of 0.46. The neuron share's goes from -0.80 to -0.61, also inside. Steering is the
clear mover. Its partial correlation given KL rises from 0.07 to 0.46. That move of 0.39 is above the card's 0.2 gap and
above twice its own floor, 0.33, though under twice the table's largest floor, 0.47. Effective rank's partial
correlation weakens from -0.64 to -0.47: smaller than the gap, but well outside its own very narrow floor. The lens
reading strengthens from -0.31 to -0.53 given KL, inside its wide floor. The write share's own correlation with KL
barely moves, 0.71 to 0.72.

The caveat is the floor itself. On these states it compares 128 pairs against 256 that include them, so it likely
understates the gap between two independent samples. The re-gating share's partial correlation at 128 pairs reads only
0.17, well below its 0.41 at 256, which says the needed set here is not yet saturated. Sixteen of the 39 states have a
needed set under 1,000 variables.

## Numbers in the results picture that change and that do not

| number in the picture | over the circuit | over the needed set | reads |
|---|---|---|---|
| built pairs: each reading's same-sign count and rank correlation with the write share's gaps | e.g. lens 8 of 9, 0.83; crosscoder 1 of 9, -0.92 | identical | unchanged |
| built pairs: lens after KL, rank correlation | 0.70 | 0.73 | unchanged within sampling |
| built pairs: pairs outside the random floor, raw / after KL | 3 / 4 | 3 / 4 | unchanged; no gap changes sign |
| prior-matched: write share splits copy from cipher | not computed on 20 seeds | all 20 seeds, U = 100 | new number, agrees with the cheap readings |
| procedural: re-gating share given KL | 0.48 | 0.41 | within sampling |
| procedural: steering given KL | 0.07 | 0.46 | **changes**: above the 0.2 gap and twice its own floor |
| procedural: steering raw | -0.13 | 0.12 | above the 0.2 gap, within its doubled floor |
| procedural: neuron share given KL | -0.80 | -0.61 | within sampling |
| procedural: neuron share at matched size given KL | -0.66 | -0.40 | above the 0.2 gap, just within its doubled floor |
| procedural: effective rank given KL | -0.64 | -0.47 | under the gap, outside its narrow floor |
| procedural: untrained decomposition given KL | 0.44 | 0.27 | under the gap, outside its narrow floor |
| procedural without the repeat task: effective rank given KL | -0.58 | -0.18 | **changes**: above the gap and its floor |
| procedural without the repeat task: steering given KL | 0.09 | 0.50 | **changes**: above the gap and its floor |
| procedural: lens given KL | -0.31 | -0.53 | within its wide floor |
| procedural: write share against KL | 0.71 | 0.72 | unchanged |
| procedural random updates: prior raw | 0.47 | 0.26 | changes, beyond a near-zero floor; not a claim 2 reading |

## What we learned

The built-pair and prior-matched halves of claim 2 do not depend on which set the write share is taken over. The cheap
readings agree with it on the same pairs, to the same rank correlations. The write share over the needed set splits the
prior-matched pair as cleanly as they do. On the procedural tasks, the readings claim 2 leans on, the re-gating share
and the neuron share, move within sampling. Steering does not: held at fixed output change it goes from unrelated to the
write share to moderately related. With the repeat task left out, effective rank's partial correlation also falls from
moderate to weak. So the paper can quote claim 2's built-pair and prior-matched numbers over the needed set unchanged,
but its procedural partial correlations for steering and effective rank should be quoted over the needed set.

## How compelling is the evidence

The built-pair and prior-matched readings are as solid as the earlier ones: large needed sets on every state, and a
floor between independent halves that is small beside the correlations. The procedural table is weaker. Its floor is a
nested one, 128 pairs against 256 that include them, because the earlier scan saved only each variable's first
flipping swap; an independent-halves floor there would likely be larger. On one built-pair state, the two independent
halves gave needed sets of 20,239 and 9,330 variables. The write share over them moved only from 0.0324 to 0.0322. So
the set is not saturated at four derangements of the inputs, but the share over it is stable. The rank correlations are over 23 or 38 procedural
states and nine pairs, so one state swapping rank moves them by several hundredths.

## Surprises

None reversed. The rivals were not filed as predictions, but by the card's wording the second rival's condition is met
on the procedural table: steering's partial correlation given KL moves by 0.39, beyond 0.2 and beyond twice its own
floor. The card's stop line, every move under 0.2, is therefore not met on that table, while it is met on the built
pairs.

## What next

- An independent-halves floor on the procedural states, by rescanning with the membership saved per derangement. If
  steering's move then sits inside it, claim 2's procedural numbers stand unchanged too.
- More pairs per procedural state where the needed set is under 1,000 variables. If steering's partial correlation
  holds near 0.46, the paper's steering line changes.

## Redteam verdict

pending (redteam-lead, note 2026-11-23)

## Scorecard

No predictions were filed; the card's two rivals, read as written:

| rival | in words | read | holds? |
|---|---|---|---|
| a | every correlation and KL partial moves by less than 0.2; no pair gap changes sign; the write share over the needed set splits copy from cipher | pairs and prior-matched: yes; procedural: steering given KL moves 0.39, steering raw 0.25, neuron at matched size given KL 0.26 | on built pairs and prior-matched only |
| b | some reading moves by 0.2 or more, or some pair gap changes sign | procedural steering given KL, beyond twice its own floor | on the procedural table only |

## Departures from the literature

None new. The needed set and the write share follow the frozen definition and the earlier scan's settings unchanged.
The procedural floor departs from the plan check's independent halves, for the reason given above.

## Not run

- An independent-halves floor on the procedural states; their scan did not save per-derangement membership.
- The prior-matched rank correlations over the circuit on all twenty seeds; only three seeds have the circuit.
- More than 256 pairs on any state, though the needed set is still growing on most.
- The procedural readings with the under-1,000 needed-set states left out.

Provenance: $0 for this page, CPU only, from saved files. The scan of the 114 new items ran on pods earlier, $9.56 in
all including a lost first attempt.
