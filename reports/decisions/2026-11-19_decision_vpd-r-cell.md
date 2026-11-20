# Read the reuse score at the VPD featurization on the nine trained pairs?

*2026-11-19, definition-lead. Pending decision. `decisions.md` and `GOALS.md` are unchanged.*

## The answer

The user asked for one cell: read the reuse score R at the VPD featurization, with each site's remainder, on the nine
trained pairs. No earlier reading has used that level. The open question is whether the cell should first check the
level against items whose answer is known, since no level has passed that check so far. I recommend yes: it costs
little beside the pairs and tells us whether the pair labels can be used.

## What has been measured

E221 (cell P1.36, report `reports/analyses/2026-11-20_analysis_e221-kstar-pairs.md`) read R on 80 saved fine-tunes of
fourteen matched pairs at three coarse levels: heads and MLP layers, MLP and attention layers, and blocks.

- Each pair's reuse-minus-new gap followed update size more than any other reading (Spearman 0.54, 0.69, 0.45), but
  stayed under the filed 0.7 bar. No earlier reading agreed with it on every pair, and the held-model reading ranked
  against it.
- At these levels the minimal circuit is nearly the whole model (all six blocks on every fine-tune), which leaves R
  little room.
- The earlier known-answer test (E214) passed at no level, because the hand-built reuse edits could not be built. The
  built new edits and E195's rescaled, moved and rebuilt reuse items were read as substitutes.
- A separate premise check judged the label column doubtful for that reason (`~/commute-data/premise/P1.36.md`).
- compare-lead's E217 read R at a pruned-neuron featurization (weak activations as remainder) at 0.96 to 0.99 on 20
  fine-tunes, the first featurization finer than the whole width that did not saturate.

## What the pending run measures that nothing has

R at a featurization finer than heads but above single neurons: VPD subcomponents, each site with its remainder, as the
working definition now allows (`decisions.md`, 2026-11-18, "Exact featurization with a mandatory remainder ... VPD
built on weights"). That level has never been read.

## The question

The user asked (task file, 2026-11-19) for one cell: R at the VPD featurization with remainder on E187's nine pairs.
Should that cell also read the known-answer items first, at the same featurization?

## Options

**1. Known-answer items first, then the pairs (recommended).** One cell, two runs, due 2026-11-22. One job: first the
do-nothing control (R should read 1), then the known-answer items (E214's two built new edits and E195's four reuse
items) at the VPD featurization, then E187's nine pairs, read whatever the second stage shows. Both stages are
reported. The second run is kept for a repair. It tells us whether this level labels the known items correctly before
its pair labels are used for the paper's results section. Cost is under the $60 gated line, measured in a $0 cluster
rehearsal before any pod.

**2. The nine pairs only.** One cell, one run, as the task wrote it. Cheapest; the pair labels again have no
known-answer check behind them.

**3. As option 1, at the pruned-neuron featurization E217 already built.** Faster to run, since the code exists and
no VPD search code has to be written. VPD, which the paper names as its featurization, stays unread.

I recommend option 1: the premise check's doubt is that no level has been checked against a known answer, and the six
known items cost little beside the pairs.

## Files

- E221 report: `reports/analyses/2026-11-20_analysis_e221-kstar-pairs.md`
- Results: `experiments/kstar_pairs/results/summary.json`, `experiments/kstar_pairs/results/pairs.json`
- Premise check: `~/commute-data/premise/P1.36.md`
- E217 report: `reports/analyses/2026-11-19_analysis_e217-topk-neurons.md`
