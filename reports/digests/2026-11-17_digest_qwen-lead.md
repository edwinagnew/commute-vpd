# qwen-lead digest, 2026-11-17

P3.11, due 11-18: filled. The repaired share reference reads mixed under the filed rule, and the premise check doubts that the random band is a stable null. Every qwen-lead cell is now filled. Part 3's next rung, L3.2, waits on part 2 (L2.3) and the 11-18 gate, so the lead has finished.

## Cells
P3.1, P3.2, P3.10 filled (1 of 1 runs each); P3.8, P3.9 filled (2 of 2); P3.5 filled (3 of 3); P3.3 filled (1 of 2); P3.11 filled (1 of 2). P3.4 closed. None open.

## What ran and what it found
- E186 (P3.11), two RunPod pods, both terminated with every read home. On the frozen-neuron decomposition, 14 of 15 built switch-offs were scored. 5 read above the random band, 7 inside and 2 below. The library's fine-tunes read 26 above and 10 inside, and neither rival reaches its 80% bar.
- With six random draws per size, the bands widened to between 0.046 and 0.154. The three new draws read lower than the earlier three on every row set. The code seeds each draw by draw and matrix only, so the five row sets share one random direction per draw. That makes this one ordering of three draws against three, roughly a one-in-twenty chance event: a candidate reading, not a found cause.
- Page: `reports/analyses/2026-11-17_analysis_e186-share-reference-repair.md`. The log is published as v358.

## For the 11-18 gate (not an ask)
- The premise check (`~/commute-data/premise/P3.11.md`) is doubtful. Its reasons: the band moved instead of settling, the near-null control already failed in P3.9, and the reading drifted from no call to mixed. Its question: what role does this share play in the readings carried into L3.2?
- P3.11 has one run left but its $20 of pod money is spent. A re-read of one earlier draw on this run's code would tell a run-to-run shift apart from spread between draws, and it needs the user's yes.

## Money
E186 spent $19.40 of the cell's $20 (earlier sessions). This session: $0, no pods.

## Friction
812 (`numbers check` arguments; `p-<hex>` ids parsed as numbers), 813 (draws shared across row sets, undocumented in `share_reading.gaussian`).

## Not run
The re-read of an earlier draw on the new code, and any larger draw count.
