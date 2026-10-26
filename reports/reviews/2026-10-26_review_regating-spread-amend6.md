# Review: E36 Amendment 6, the whole preceding text against the original model's view (2026-10-26)

Independent check of `experiments/regating_spread/amend6.py`, its filed predictions
(`results/predictions_amend6.json`), its results (`results/regating_spread_amend6.json`) and the plan
(`reports/plans/2026-10-26_plan_regating-without-cues.md` → Amendment 8). The reviewer did not write the code or the
analysis, and ran nothing (read-only; numbers read with `jq`).

**In short.** The scoring is right: T3 holds, T4 misses, both gates hold. No bug was found that would make the bag of
the preceding text score near zero by accident. But the bag scores no better than averaging the changes of random
positions, so it never got to stand in for "the whole preceding text". T3 therefore says what Amendment 5 already said:
the original values beat a summary of the last few tokens. It does not show that more context fails to close the gap.

## 1. Scoring (all correct by the filed rules)

| Check | Filed rule | Numbers at step 300 (A0 at every step) | Verdict |
|---|---|---|---|
| A0 | each of the two within 0.02 of Amendment 5 | near_g0 0.7622 vs 0.7613 (diff 0.0009); text4 0.5110 vs 0.4914 (diff **0.0196**); steps 10 and 100 within 0.009 | holds, with 0.0004 to spare on text4 at step 300 |
| C_full | text_full ≥ text4 − 0.05 at every step | 0.564/0.564, 0.513/0.525, 0.491/0.511 | holds |
| T3 | text_full ≤ near_g0 − 0.1 | 0.491 ≤ 0.662 | holds, wide margin |
| T4 | text_full ≥ text4 + 0.014 | 0.491 vs 0.525 | misses |

The code scores this way at `amend6.py:183-196`.

## 2. Does the code compute what it says?

- **prefix_bag** (`amend6.py:99-110`). At each position it writes the current token's weight times its signed hash into
  that token's coordinate (offset-0 table, `:108`). It then takes a running sum over the row (`cumsum`, so the prefix
  runs from the row's start up to and including the position) and scales each vector to unit length (`:109`). The weight
  is `log(262144 / (1 + count))` over the fitted rows (`:93-96`). That ranges from about 3 for common tokens to 12.5 for
  tokens the fitted rows never contain, so the weights are not all equal. The held-out queries go through the same
  function, with the same tables and the same fitted-row weights (`:127`). Rows are flattened row-major on both sides,
  matching `Run.values`. There is no padding (rows are cut from a stream joined with end-of-text, `amend1.py:60-68`).
  No off-by-one or mismatch was found.
- **text_full** (`:115`) joins two unit vectors and divides by √2, so each half has length 1/√2 and the squared
  distance is the mean of the two halves' squared distances. This is correct.
- **Why the bag scores about 0 (−0.046 at step 300) by design, not by bug.** Its unit length gives the current token
  about 1/√(number of distinct tokens) of the vector, so deeper in the row the bag stops encoding the current token.
  Its R tracks the rules that use no context at all: Amendment 1's random-neighbour rule scores 0.07, 0.04, 0.03 and
  Amendment 4's position rule 0.07, 0.05, 0.04. With only 8 neighbours, which are likely to come from one fitted row,
  the averaged change is noisy, and clipping it can push R below zero.

## 3. Is the reading supported?

- **T3 holds, but against a weak alternative.** The bag carries no usable signal for this rule, and text_full
  (0.491) is about text4 (0.511). So T3 again compares the original values with the 4-token summary, which is
  Amendment 5's T1. The positive control C_full checks only that adding the bag does not ruin text4. No control shows
  that the bag can carry long-range context at all. The strongest evidence for the reading is the 0-31 bucket. There
  the bag holds the whole prefix, yet text_full reaches 0.48 against near_g0's 0.74. Even there, though, the bag is
  orderless and uses inverse-frequency weights.
- **T4 miss: "adds nothing" is supported, "hurts" is not.** text_full − text4 = −0.020. That is the size of text4's
  own run-to-run shift (0.0196) and larger than the filed 0.014 floor. So the floor understates replicate noise for
  the text spaces.
- **Candidates for what the bag is blind to (not asserted as causes):**
  (a) the current token and its order, which carry most of text4's value (69% of its neighbours share the current
  token, Amendment 5);
  (b) neighbours that cluster in one fitted row, so the 8 neighbours are fewer independent contexts than 8 (in the
  filed blind-to line);
  (c) the row's start is not a document's start, so the bag mixes the end of one document with the next;
  (d) in the joined space the bag's spread of distances, about 0.1-0.2, reorders text4's near-ties. It can also break
  exact 4-gram ties towards one document;
  (e) hash collisions with opposite signs.
  Separately, text4 may have many exactly tied neighbours. If so, the choice of 8 among them is arbitrary. That would
  explain its 0.02 replicate shift while near_g0 moved by only 0.001.

## 4. What should stop "positions the original model treats alike get alike changes" being adopted on this evidence

- **The whole-text null is not a working null.** It scores at the level of random neighbours, so the decision tree's
  "T3 holds" branch is reached without the amount of context ever being tested. A summary that keeps order and the
  current token over the whole prefix has not been tried. Examples: the bag weighted with text4's current-token
  weight, longer n-gram windows without decay, or another model's embedding.
- **The original values have a mechanical edge on the clip.** The change at a position is bounded by
  [−g0, 1 − g0], and the rule adds the neighbours' mean change to the query's g0 and clips (`amend6.py:133`).
  Neighbours chosen by similar g0 have changes that respect the query's bounds; text neighbours do not. Part of the
  0.25 gap could come from this rather than from how the model treats the context. The residual-stream space
  (Amendment 4, 0.744) cannot separate the two, since g0 is a function of it. A space-neutral check would help:
  predict with the neighbours' g1 clipped to the query's bounds, or restrict to subcomponents far from 0 and 1.
- **Text neighbours are capped at 8 even when hundreds are tied.** That handicaps the text spaces and not the
  original values.
- One fine-tune (lowercase), one seed, one held-out set, one hashing.

## Not run

- Nothing was re-run. The candidate causes above (neighbours from one row, tie counts in text4, clip-bound headroom)
  were not measured.
- No check that the bag's 8 neighbours come from one fitted row, or of how many exact ties text4 has per query.
- No lint or pyright run on `amend6.py`.
