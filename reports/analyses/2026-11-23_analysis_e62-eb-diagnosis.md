# Why does the kept share not tell E62's copying map from its lookup?

Cell P1.57, 2026-11-23. A read of saved arrays on CPU, $0; no model forward, no GPU.

From saved files only, we asked why the kept share over the needed set reads about the same for E62's copying map and its lookup, when switching off head (2,4) separates them. The head is in the needed set, and more of it for the copying map, but its subcomponents carry at most five percent of the write change and keep about one twentieth of it, like every other variable. So the kept share sits between 0.015 and 0.063 for every fine-tune we have read, and a change confined to one head changes the direction of every head-mixed subcomponent's write, so it reads as mostly new. D is mixed (see Terms), so the verdict is "no separation of E_b over K on E62's mixed D", while head ablation reads copy behaviour: the two readings are on different data.

## The answer

The needed set is not missing the head, and the head's update is not hidden in its query and key side. The kept share is low because almost no variable's write change lies along its own original write: at the median variable about one percent is kept. That is true of both maps and of all 114 fine-tunes read so far, which sit between 0.015 and 0.063. What does separate the two maps in the same arrays is where the write change sits: a fifth of it in layer-2 attention for the copying map, an eighth for the lookup.

## What was measured

Terms, each used below:

- **Variables.** Every VPD subcomponent of the 24 decomposed matrices, plus one remainder per matrix: 38,938 in all.
- **Write change of a variable.** The squared difference between the fine-tuned model's write and the original model's write for that variable, on the same matrix input, summed over every position of the 512 windows of D. Called dd in code.
- **D.** 512 windows of 8 tokens, of which 206 are copy windows ("A B .. A M"); the rest are cue-free windows and far text (E187's read half of the e62 group). D is mixed, not copy-only.
- **Kept part of it.** The part that the original write, times its own best positive gain, explains: the change projected on the original write. A variable whose fine-tuned write is a positive multiple of its original write keeps all of it.
- **Kept share over K**, E_b in code: the kept parts summed over the needed and changed variables, over their summed write change. 1 for an update that only rescales each variable's write; the KL-matched random update reads 0.0106 and the size-matched one 0.0110; the built strengthening edits that double or multiply by ten every weight read 0.761.
- **Needed set K.** A variable is needed when swapping its value for the one it takes on another window of D flips the fine-tuned model's top token at the read position, on any of 256 tested window pairs.
- **Flips.** For each variable, how many of those 256 pairs flipped the top token.
- **Head (2,4).** Head 4 of layer 2, 0-indexed, 6 heads of width 128. E62 picked it by the prefix-matching score. Switching it off removes 0.563, 0.545 and 0.471 of the copying map's gain over seeds 0-2, and none of the lookup's: 0.000, -0.004 and -0.003. The older masked share read 1.095, 1.091 and 1.099 on the copy and 0.415, 0.382 and 0.415 on the lookup.

The arrays: write change, kept parts and the changed mask are E257's `experiments/rb_states/results/large/items_e62-*.npz`; the needed set and flips are E275's `experiments/claim2_needed/results/large/items_e62-*.npz`. The updates are E62's saved weight differences `experiments/induction_cue/results/large/{copy,lookup}_s{0,1,2}.pt`, read at their own size. The decomposition comes from `commute.vpd.checkpoint_weights`. The script is `~/commute-data/scratch/definition-lead/e62-diag/diag.py`; its output is `diag.json` beside it. It recomputes the kept share over K from the saved arrays and matches E275's value to every printed digit on all six states.

**Which subcomponents are head (2,4)'s.** The attention matrices are decomposed whole-layer, so no subcomponent belongs to one head. A subcomponent's weight is an outer product of an output vector and an input vector. For the query, key and value matrices, head 4 owns output rows 512 to 640; for the output matrix it owns input columns 512 to 640. Its head-4 share is the fraction of its squared weight norm in those rows or columns. An even spread over six heads gives one sixth. Over the 3,072 layer-2 attention subcomponents the median share is 0.115 and the largest is 0.740. So no subcomponent is mostly head 4's alone. We report two thresholds: 44 subcomponents have a share over one half, and 272 over 0.3. A third reading weights every layer-2 attention subcomponent by its head-4 share.

## Is the needed set missing the head?

No. On the copying map, 36, 44 and 36 of the 44 head-4 subcomponents are in K, against a needed set that holds 0.31 to 0.79 of all variables. On the lookup it is 26, 34 and 15 of 44.

| quantity | copy 0 | copy 1 | copy 2 | lookup 0 | lookup 1 | lookup 2 |
|---|---|---|---|---|---|---|
| E_b over K | 0.0376 | 0.0399 | 0.0362 | 0.0399 | 0.0431 | 0.0429 |
| E_b over every changed variable | 0.0396 | 0.0402 | 0.0393 | 0.0441 | 0.0450 | 0.0450 |
| size of K, variables | 17382 | 30688 | 12145 | 11186 | 15904 | 9271 |
| share of all variables in K | 0.45 | 0.79 | 0.31 | 0.29 | 0.41 | 0.24 |
| smallest top-1 margin on D, logits | 0.0034 | 0.0012 | 0.0040 | 0.0063 | 0.0018 | 0.0140 |

| quantity | copy 0 | copy 1 | copy 2 | lookup 0 | lookup 1 | lookup 2 |
|---|---|---|---|---|---|---|
| head-4 subcomponents in K, of 44 | 36 | 44 | 36 | 26 | 34 | 15 |
| head-4 subcomponents in K, of 272 | 229 | 265 | 223 | 177 | 213 | 124 |
| share of all layer-2 q, k, v, o subcomponents in K, weighted by head-4 share | 0.67 | 0.91 | 0.58 | 0.47 | 0.59 | 0.33 |
| median flips, the 44 | 3 | 5 | 3 | 1 | 1 | 0 |
| most flips, the 44 | 9 | 11 | 11 | 5 | 8 | 3 |
| median flips, every variable | 0 | 1 | 0 | 0 | 0 | 0 |
| median flips, variables in K | 1 | 2 | 1 | 1 | 2 | 1 |
| most flips, any variable | 131 | 127 | 123 | 96 | 104 | 94 |
| median percentile of the 44 among all variables by flips | 0.92 | 0.93 | 0.92 | 0.71 | 0.59 | 0.00 |

**Reading it.** The needed set sees the head, and sees it more on the copying map. The head's subcomponents flip the top token on three to five of 256 pairs at the median on the copy, against one or none on the lookup. That puts them in the top tenth of all variables by flips on every copy seed. The needed set is also very wide: it holds a quarter to four fifths of all variables, and the smallest top-1 margin on D is only a few thousandths of a logit. Restricting to K barely matters for the kept share: over every changed variable it reads 0.039 to 0.045, at most 0.0042 above the value over K on any state. The median margin over D was not saved, so it is not reported.

## Is the head's change there, but not as a write?

The change is there, and it is not mostly on the query and key side. Head 4 takes 0.176 to 0.179 of layer-2 attention's squared update on the copying map, against 0.132 to 0.134 on the lookup; an even spread would give one sixth. Of head 4's own update, the query and key matrices carry 0.51 to 0.54 on the copy and 0.48 to 0.51 on the lookup, about the even half.

| quantity | copy 0 | copy 1 | copy 2 | lookup 0 | lookup 1 | lookup 2 |
|---|---|---|---|---|---|---|
| head 4's share of the whole update, squared norm | 0.0156 | 0.0156 | 0.0163 | 0.0098 | 0.0097 | 0.0096 |
| layer-2 attention's share of the whole update | 0.088 | 0.088 | 0.091 | 0.074 | 0.073 | 0.072 |
| head 4's share of layer-2 attention's update | 0.176 | 0.178 | 0.179 | 0.132 | 0.134 | 0.133 |
| q and k share of head 4's update | 0.52 | 0.51 | 0.54 | 0.48 | 0.51 | 0.49 |
| outside part of head 4's slice of q | 0.10 | 0.09 | 0.10 | 0.11 | 0.12 | 0.12 |
| outside part of head 4's slice of k | 0.14 | 0.14 | 0.14 | 0.15 | 0.16 | 0.16 |

**What the test can see of a query or key change.** A query subcomponent's write is its contribution to the 768-wide query vector of all six heads, not the head's output. The test keeps a change only when that contribution is a positive multiple of the original one on the same input. An attention pattern is the softmax of products of query and key vectors, a function downstream of these writes. It is not any variable's write. The write change is measured with the original and fine-tuned mechanism on the same input, the fine-tuned model's own. So a changed attention pattern, which changes the inputs of the output matrix and of later layers, enters no variable's write change at all. Only the weight change of the query and key matrices is seen, as a change in those subcomponents' query or key contributions.

The query and key matrices have 512 subcomponents for 768 output rows, so the least-squares split cannot place all of their update on subcomponents. About a tenth of head 4's query update and a seventh of its key update falls outside them and goes to the matrix's remainder. Every remainder's write change is in a query or key matrix, and remainders keep 0.0002 to 0.0011 of it. That outside part is new by construction under this test, whatever the head does.

## The head is in K: why is it not kept?

Because its subcomponents change direction, as almost every variable does.

| quantity | copy 0 | copy 1 | copy 2 | lookup 0 | lookup 1 | lookup 2 |
|---|---|---|---|---|---|---|
| write-change share over K, the 44 | 0.0071 | 0.0074 | 0.0104 | 0.0047 | 0.0052 | 0.0036 |
| write-change share over K, the 272 | 0.031 | 0.031 | 0.046 | 0.018 | 0.018 | 0.015 |
| write-change share over K, every layer-2 attention subcomponent weighted by head-4 share | 0.032 | 0.037 | 0.040 | 0.022 | 0.021 | 0.018 |
| kept share, the 44 pooled | 0.037 | 0.067 | 0.057 | 0.022 | 0.036 | 0.012 |
| kept share, the 272 pooled | 0.059 | 0.068 | 0.065 | 0.030 | 0.035 | 0.025 |
| kept share, weighted | 0.057 | 0.061 | 0.063 | 0.047 | 0.057 | 0.053 |
| the 272's part of E_b | 0.0018 | 0.0021 | 0.0029 | 0.0005 | 0.0006 | 0.0004 |
| E_b over K without the 272 | 0.0369 | 0.0390 | 0.0348 | 0.0401 | 0.0432 | 0.0432 |

**Reading it.** The head is a small part of the write change and is not kept. The 272 subcomponents with over 0.3 of their weight in head 4 carry 0.031 to 0.046 of the write change over K on the copy. They keep about a sixteenth of it, a little more than the lookup's 0.025 to 0.035. Their whole contribution to the kept share is at most 0.0029; removing them moves it by at most 0.0014.

Where the write change sits:

| quantity | copy 0 | copy 1 | copy 2 | lookup 0 | lookup 1 | lookup 2 |
|---|---|---|---|---|---|---|
| write-change share, layer-2 attention | 0.20 | 0.23 | 0.23 | 0.14 | 0.13 | 0.11 |
| write-change share, q and k of every layer | 0.46 | 0.51 | 0.48 | 0.44 | 0.47 | 0.40 |
| write-change share, MLP matrices | 0.44 | 0.39 | 0.43 | 0.47 | 0.44 | 0.52 |
| write-change share, the 24 remainders | 0.038 | 0.032 | 0.043 | 0.050 | 0.039 | 0.054 |
| kept share, remainders pooled | 0.0010 | 0.0009 | 0.0011 | 0.0004 | 0.0007 | 0.0002 |
| kept share, subcomponents pooled | 0.039 | 0.041 | 0.038 | 0.042 | 0.045 | 0.045 |
| kept share of the median variable, by write change | 0.011 | 0.013 | 0.010 | 0.009 | 0.013 | 0.010 |
| share of variables in K with kept share over one half | 0.0004 | 0.0001 | 0.0005 | 0.0001 | 0.0001 | 0.0002 |
| write-change share of the top 100 variables | 0.40 | 0.32 | 0.43 | 0.45 | 0.38 | 0.48 |

**Reading it.** The low kept share is a property of every variable, not of the head. Half the write change, weighted by size, sits on variables that keep about one percent of theirs. Fewer than one variable in two thousand keeps over half. One hundred variables carry a third to a half of all the write change. The copying map does put more of its write change in layer-2 attention than the lookup, 0.20 to 0.23 against 0.11 to 0.14. The kept share weighs every location the same way, so that difference does not reach it.

Over all 114 items E275 read, the kept share over K runs from 0.0147, a cipher seed of E53, to 0.0628, a cipher seed of the open pair. E62's six states sit inside that range, a few times the random update's 0.011 and a twentieth of the built edits' 0.761.

## What this means

The non-separation comes from the third account, in a general form, not the first two. The needed set contains the head, and more of it on the copy. The head's update is half query and key, not mostly. But pooled over subcomponents the kept share is near 0.04 on both maps, and the median variable keeps about one percent. So it reads the geometry of the least-squares split, not which heads were reused. Two facts make a reused head read as new under this test. VPD's layer-2 attention subcomponents are not head-local: the median subcomponent has 0.115 of its weight in head 4. So a change to head 4 alone is not a rescaling of any one subcomponent's write. For q, k and v, under one exact split of a head-4-only rescaling, each subcomponent keeps a fraction equal to its own head-4 share s_i, so pooled over q, k and v the kept share is well below 1 (it is sum s_i^2 w_i / sum s_i w_i, above the 0.115 median share). That follows from the definitions above and was not computed on data. It does not hold for o: head 4 owns input columns there, so an o subcomponent's change stays parallel to its own output direction, and its kept fraction is the squared uncentred cosine over positions of (P v_i . x) with (v_i . x), which depends on the activations and can be large. Second, a change in where a head attends shows up only as changes in query and key contributions, compared on the same input. Its effect on what the head writes is never compared at all. So a re-routing of attention at a reused head, or a sharpening confined to one head, reads as new under a per-variable write-gain test, while the head-ablation reading credits it as reuse. No verdict on either wording of the definition is given here: the definition is the user's to set.

## How compelling is this

Three seeds per map, one featurization, one set of 512 windows, of which 206 are copy windows and the rest cue-free windows and far text. The verdict is "no separation of E_b over K on E62's mixed D"; head ablation (0.471 to 0.563 against none) reads copy behaviour, so the two readings are on different data. Every number is from saved arrays and weights; the kept share recomputed here matches E275's at every printed digit. The head-4 assignment is by weight share, a proxy: the head-4 part of each subcomponent's write change needs the activations and was not read. The flips come from 256 window pairs and are small counts.

## Surprises

- No layer-2 attention subcomponent has over 0.740 of its weight in head 4, so "head (2,4)'s subcomponents" is only a matter of degree in this decomposition.
- Remainders keep almost none of their write change. All of it sits in query and key matrices, where 512 subcomponents cannot span 768 rows.

## What next

- Build a head-4-only strengthening edit and read its kept share over K. If it reads near the fine-tunes rather than near 0.761, the test calls a pure reuse of a head new, and the per-variable gain clause is the thing to revisit.
- Read where the write change sits, layer-2 attention against the rest, on the other reuse and new pairs. If it separates them as it does here, a location reading could carry the needed set where the kept share does not.

## Departures from the literature

None: this is a diagnosis of the project's own measurement.

## Not run

- Median top-1 margin over D: not saved by E275, not computed.
- The head-4 part of each variable's write change: needs a forward on D, not computed.
- A built head-4-only strengthening edit, and its kept share: not run.
- E_b over K on the 206 copy windows alone: num, den, dd were summed over all of D (stand_in.write_change), so it needs a new write pass on a GPU (proposed to whip-lead 2026-11-23).
- The built head-(2,4) edits (E280, running).
- Whether the remainders' write change is only the update's outside part, or partly rounding of the remainder in single precision: not checked.
- A figure: the tables carry every comparison; none drawn.
- No GPU, no job, no model forward.
