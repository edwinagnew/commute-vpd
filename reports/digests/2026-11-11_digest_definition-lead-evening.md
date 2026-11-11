# definition-lead evening digest, 2026-11-11

**P1.15, due 11-12: the difficulty ladder ran, and the rewiring share did not fall as the target got harder for the
plain model.** At one site it fell and then rose back, which goes against what was filed. One cause is open: the hardest
rung's fine-tunes barely moved toward their target. The decision card is
`reports/decisions/2026-11-11_decision_difficulty-ladder-result.md` (ask 35, Blocking). The page is
`reports/analyses/2026-11-11_analysis_difficulty-ladder.md`.

## Cells
All of part 1's cells are filled or closed; none is open. The two rungs still open, the built control and the gate page,
have no run budget left, so any further run needs your yes. The last cell filled is the difficulty ladder (1 of 1 runs).

## What ran and what it found
Three fine-tunes of the plain 67M were read down a ladder: last number odd, parity of a sum of two, parity of a sum of
three. Each rewiring share is rescaled so that a planted ceiling reads 1 and a planted floor reads 0.

- **One site:** the share read 0.47, 0.07 and 0.47. The filed expectation was a fall and then level; the reading came
  out out of order.
- **Two sites:** it read 0.70, 0.59 and 0.63, a fall and then roughly level.
- **Probes:** the plain model's small non-linear probe read 1.0, 0.98 and 0.72, and its linear probe 1.0, 0.58 and
  0.51. The share follows neither.
- **Controls:** the repeat matched exactly and the ceiling read 0.98. The floor read as filed, negative at two sites.
- **Caveats:**
  - The sum-of-three fine-tunes passed the learned check. Measured in total variation, though, they end as far from
    their target (0.25) as the plain model is (0.23).
  - That rung is also read on three-number windows, while the other two are saved readings on two-number windows.
  - The premise check said "doubtful" for this reason.

The run took one main job plus two seed jobs in parallel, on cluster debug, about 105 min in all, at $0. No pods ran.

## Waiting on you
**Ask 35:** the rewiring share did not fall down the difficulty ladder. One site read 0.47, 0.07, 0.47; two sites read
0.70, 0.59, 0.63. It follows neither probe, but the sum-of-three fine-tunes barely reached their target. The options are:

- **(a)** One more free run of about 2 h. Retrain the sum-of-three rung until it reaches its target, and re-read all three
  rungs on the same windows. Recommended.
- **(b)** Take the ladder as read. The gate page then says the share is not graded by difficulty.
- **(c)** Drop the graded reading.

## Also done
- Answered qwen-lead's note on which 67M built pairs to carry over to Qwen. The note is
  `~/commute-data/agents/qwen-lead/note-definition-lead-2026-11-11-pair-carryover.md`. The first pick is copy against
  a cipher over the same keys; the second is copy-from-context against a lookup.
- Told the other leads you are away, and set the harness's away flag.
- The research log is published at v324.
- Spend: $0.

## Friction
- **730:** the log clerk's publish cost two wasted calls on a page that was already up to date.
