# Can the Qwen instrument see an update at all? A check with a known answer

*2026-11-11, planning session, at the user's request. A brief for `qwen-lead` to turn into an experiment plan with
filed predictions; nothing here has been run. Sources: the Qwen digests, `experiments/TRIED.md`, and the analyses
named below. The decomposition is the new one, p-9707d6ff, unless a number is marked old.*

## Why this, now

Every Qwen reading so far has failed to separate anything: no map carries an update onto the decomposition (E76, E82,
E93), VPD's adversary cannot fail a bad placement at a real update's size (E85, E89), reinforcement learning reads like
its shuffled-reward null (E104), the readings that split the two anchors follow task and format (E94), and the
re-gating share sits at 0.991 to 0.9992 for real and random changes alike (E132, E144).

All of these compared real updates whose right answer we do not know. So none of them can say whether the trouble is
the updates or the instrument. Two explanations put it in the instrument, and both can be tested with changes whose
right answer we build in. Part 3's rung L3.2 is specified at the 2026-11-18 gate, and this result says whether any
decomposition reading on Qwen is worth carrying into it.

## The two explanations

**Too small to see.** The decomposition does not reproduce the model exactly: with its own masks the output moves by
0.07 nats per token on maths text and 0.21 on web text (job 56760), and the worst matrix is rebuilt to a relative
error of 0.0055 (E131). An update whose effect is of that order or less is read through that error. The first
supervised fine-tune is not small in this sense: it moves the output by about 1.6 nats (E93, stage 0, old). The
reinforcement-learning updates are a tenth of its norm, 0.35 against 3.4, and may be.

**Placement is free.** Each matrix has as many subcomponents as its larger side: 6,144 for the three MLP matrices,
2,048 for q and o, 1,024 for k and v (the bundle's `index.json`; 28 layers of these sum to the 688,128). So with every
subcomponent in play, any change to a matrix can be written exactly as a change to subcomponents, a random one as
easily as a real one. Only the 500,244 alive ones are used, and the saved numbers fit that picture: least squares
leaves 0.090 of a real update outside the subcomponents and 0.094 of a random one (E93, old). If 73% of the
directions are covered on each side, a random change should leave about 0.27 × 0.27 ≈ 0.07 outside, which is my rough
arithmetic, not a measurement. If this is right, where the split puts an update is decided by the least-norm rule and
not by the update.

## Step 0: what the saved files already say (CPU, about an hour, $0)

No model runs. Three tables.

1. For each of the 196 matrices, the rank of the alive subcomponents' read vectors and of their write vectors, beside
   the matrix's two sides. From the ranks, the share a random change must leave outside, beside the 0.094 measured.
2. For each saved update and each matrix, the update's norm beside the norm of the decomposition's leftover Δ.
3. For each saved update, how far it moves the output on its own rows beside the decomposition's own masked error on
   those rows, from numbers already stored. This is the signal-to-noise of every reading made so far.

Table 1 answers "placement is free" in principle. Table 3 says which saved updates, if any, sit above the floor.

## Step 1: changes whose placement we know (one GPU job, cluster, $0)

Build three kinds of change to the unchanged model:

- **planted on often-on**: a change to the write vectors of a small chosen set of subcomponents that are switched on
  at most tokens of the maths rows;
- **planted on rarely-on**: the same on a set that is almost never on there;
- **random**: the matched random change E132 used.

Make each at three sizes, matched in output change on the maths rows to the reinforcement-learning update, the
supervised fine-tune at step 10 and the supervised fine-tune at step 100. Then run the existing pipeline unchanged and
read three things on each of the nine states:

- the share of the least-squares split that lands on the planted set (a random change sets the chance level);
- E94's share of the split on subcomponents already on, which should read high for the first kind and low for the
  second;
- E132's re-gating share.

Controls first and alone: the unchanged model, and do-nothing, as in E132.

| What comes back | What it means |
|---|---|
| The split returns the planted set at every size, and the two planted kinds read apart at every size | The instrument works at real sizes. The trouble is in the anchors, the frozen masks or the unit, not here. |
| The split returns the planted set, but the two kinds read apart only at the larger sizes | Too small to see. The size where they part is the floor; Table 3 says which saved updates clear it. |
| The split smears the planted set towards chance at every size | Placement is free. Readings of where an update sits say nothing on this decomposition, at any size. |
| Every state reads the same re-gating share | That share does not respond to placement; drop it from part 3. |

## Limits

- A planted change is the easiest case: it lies exactly in the subcomponents and on few of them. A pass shows only
  that the instrument can work; a fail is the informative result.
- The least-squares split is linear, so its recovery should not depend on size. Size enters through the readings made
  under masks. If recovery does change with size, that is rounding or the second-order term and is reported.
- One decomposition, one seed of each planted set unless the lead's card needs more to clear its bar.
- This does not test the other three explanations: confounded anchors, masks frozen on the unchanged model, and the
  rank-one unit.

## What it needs

A new cell for `qwen-lead` with two runs (step 0 on CPU, step 1 on one GPU), due 2026-11-17. The ladder's only open
Qwen rung has a budget of 0 until the 11-18 gate, so the cell needs the user's word, which the task text below gives.
