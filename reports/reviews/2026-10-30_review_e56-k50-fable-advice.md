# Fable advice: how E56 should read the k50 difference (2026-10-30)

**Asked.** k50 reads 4096 on all five elicit states and 12288 on all five matched ciphers; the pre-run amendment bins
both as "over 1000", so P1 passes; unbinned it would have failed. Kind signal, artefact, or reversal?

**Recommendation: option C, with the difference reported plainly as a side observation. No new run, no finer grid.
The headline does not reverse and the user is not needed for a decision; they should see one plain paragraph.**

## Reasoning
The one-grid-step k50 difference is real, not a grid artefact: the whole share curve sits lower for the matched
cipher (`job1.json`, `job2.json` `b2_shares`). At k = 4096 the elicit map reads 0.72-0.76 and the matched cipher
0.42-0.48; with all 12288 neurons the elicit map reads 0.93-0.94 and the matched cipher 0.68-0.72, five seeds each,
ranges 0.03-0.06. Stage 0's full residual patch at block 3 reads 1 on every state, so for the matched cipher about
30% of its gain flows through block 3 but not through its MLP neurons (its attention); for the elicit map about 6%.

What it measures is how much of the gain sits in block 3's MLP at that checkpoint, not the kind of the update:
1. The cipher at its own step (30, norm 6) has the elicit map's curve: 0.69-0.77 at k = 4096, 0.95-0.97 at full
   width, k50 = 4096. Its r stays 0.24-0.27. So the cheap number moves with the checkpoint and r does not (P3).
2. On E12's pair, E52 read the same full-width share the other way round: copy 0.50-0.71, cipher 0.76-0.97
   (plan line 27). A reader that sides with the cipher on one pair and the elicit map on the other reads nothing
   about kind.
3. The quantity r reads (which route is re-gated) is unrelated to this one: cipher_own has elicit-like concentration
   and cipher-like r, so the k50 difference cannot explain P2's separation (0.76 vs 0.21-0.27).

So "no cheap baseline reads kind" stands. The write-up narrows one phrase: neuron patching at matched size does
tell the two states apart, by how diffuse the early cipher update is (full width 0.70 vs 0.94), but not by which
neurons or how few carry it (N16 ~ 0 on both), and the difference is gone at the cipher's own step and reversed on
E12's pair. State in the result message that P1 passed under the pre-run binning and would have failed unbinned,
and why the binning is the right reading (the plan's own "> 1000 on all states", filed before any data). The
decision tree's "P1 fails -> stop" was written for a cheap baseline doing r's job; this one does not.

A finer k grid (option A) adds nothing: the share curve in the JSON is already the finer reading, and it would only
place k50 near 2000-2500 for the elicit map and 4500-6000 for the matched cipher, a number no claim uses.

## What would change this
A cheap reader that separates elicit from cipher in the same direction at both the matched step and the cipher's own
step, and on E12's pair too. Nothing in hand does. If the Qwen line later runs B2, file full-width share with that
expectation (no step-stable separation) and this reading is tested for free.

## Follow-up
None. Add to `TRIED.md`'s E56 row: "full-width MLP share separates at matched size (0.94 vs 0.70) but not at the
cipher's own step (0.96) and reverses on E12's pair (E52); not a kind reader." Queue the same line in
`docs/research_log/PENDING.md`.
