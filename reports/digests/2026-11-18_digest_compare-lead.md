# compare-lead digest, 2026-11-18 (afternoon)

P2.31, due 11-19: filled. The crosscoder baseline does not separate the nine built pairs as expected, and its own
controls weaken it as a baseline. Every compare-lead cell is now filled or closed, and no ready rung has work left, so
this lead finishes.

## Cells
All part 2 cells are filled or closed (`fleet/cell list compare-lead`). L2.1 (every baseline on every pair) has its
last cell, P2.31, filled today; L2.2 waits on part 1's L1.3; L2.3 has run budget 0 and its gate page is current;
L2.4 waits on the user adopting a wording.

## What ran and what it found
No new runs this session (spend $0). The E203 crosscoder sides, trained earlier today, were analysed
(`reports/analyses/2026-11-18_analysis_p2-31-crosscoder.md`). The reuse side reads higher on 5 of 9 pairs as built
and 4 of 9 size-matched by the authors' rule, 6 and 6 when artefact latents count as fine-tune-only; the filed
expectation was 7 of 9. The planted control missed and a random update reads the most shared. Six sides trained on
about half the planned tokens because their task stream ran out; both sides of each pair were cut alike.

The gate page (`reports/analyses/2026-11-16_analysis_part2-gate-page-update.md`) gained a crosscoder paragraph and a
"not shown" line on the wider-gap selection of steering and the top-neuron share. Research log v394 is published with
a finding card on the miss.

## Waiting on the user
No open asks. The premise check's question, for the gate: given the planted-control and random-update readings, what
does the crosscoder row allow the part 2 paragraph to say about the share?

## Friction noted
Side JSONs record the planned token count, not what trained, which hid the six short runs.
