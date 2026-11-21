# The reuse score R reads about 1 on everything: how should the paper handle that, and what are the memorising twins for?

*2026-11-20, definition-lead. Pending decision for the user; rewritten 2026-11-20 after the user found the first
version unreadable. Only the user changes the definition's wording. Source: writeup-lead's note of 2026-11-20.*

R, the reuse score: the share of the parts the behaviour runs through that still compute what they did in the base
model.

## What we ran

- **Hand-built edits.** We scored R on edits whose answer we know, expecting the reuse edits to score higher than the
  new ones. Instead a reuse edit (a ten-fold gain on eight neurons) scored 0.976, below a new edit at 0.996, because R's
  per-part score (squared cosine) treats a big gain as a change. (P1.37,
  [report](../analyses/2026-11-20_analysis_e226-kstar-vpd.md),
  [figure](../../experiments/kstar_vpd/figures/r_known.png))
- **Size-matched pairs.** We shrank the larger fine-tune of each of nine trained pairs to its partner's size, expecting
  the reuse side to stay higher on at least 8. It stayed higher on 4, so R's pair differences mostly reflected update
  size. (P1.38, [report](../analyses/2026-11-20_analysis_e232-size-matched-r.md),
  [figure](../../experiments/kstar_size_matched/figures/r_gap_matched.png))

**So:** R sits near 1 everywhere, and its small differences follow how far the weights moved, not reuse.

## Question 1: what are the memorising twins for?

The paper says a fact memorised through small changes can correctly read as reused, so should the twins still be a
case R must read low on?

- **(a) Recommended.** Say the twins check how the pairs were built, not R. No wording change.
- **(b)** Keep them as a case R must read low on; the definition must then say a memorised fact is new.
- **(c)** Drop them from R's results; we lose the check on how the pairs were built.

## Question 2: what do we do about R reading near 1?

- **(a) Recommended first.** Compute, for free from P1.37's saved files, an R that weights each part by how much the
  behaviour depends on it; then choose (b) or (c) with its numbers in hand.
- **(b)** Keep R as worded and report the near-1 reading as a finding about the definition.
- **(c)** Change R's per-part score so it reads the built edits the right way round; a wording change and new runs.
