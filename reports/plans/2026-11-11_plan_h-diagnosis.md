# E156: why the reuse search stops near 0.47 on a variable the 67M holds exactly

*2026-11-11. Cell P1.13 (1 run, cluster debug, $0, due 11-12). The user chose this diagnosis on 11-11. Card:
`experiments/h_diagnosis/card.txt`. Filed rules: `experiments/h_diagnosis/results/predictions.json`. Code:
`experiments/h_diagnosis/run.py`, `plot.py`, `cluster/h_diagnosis.sbatch`. Nothing has run.*

**In plain words.** In the last built-control run, a fine-tune toward a push on "the last number is odd" learned its
target, yet the best affine map of the original reproduced just under half of its change, under the 0.5 cut-off. This run
asks why, on the same windows, saved fine-tunes and search code. It first checks that the search can read the target edit
itself, then tries three wider maps on the fine-tunes, to tell a change the map cannot reach from a fit that is itself the
limit.

## What runs

All readings use E154's read half (100,000 windows), number tokens, alpha 4, push and saved fine-tunes
(`experiments/built_control_sum/results/large/amend1_control_s{0,1}.pt`). $H = 1 - d / \text{gap}$ as in E154. In order,
cheapest first:

1. **Positive control, alone.** E154's own search (`amend2.search_h`, unchanged, five sites) with the target F standing in
   for the fine-tune. F is exactly an affine edit at the input of block 2, last position, so $H$ there should be about 1
   (bar 0.9). Under 0.5 is the card's stop: written, and nothing else runs.
2. **Ceiling.** Per seed, $1 - \text{TV}(\text{ft}, F) / \text{TV}(\text{ft}, M)$: the $H$ of a map that reproduced F
   exactly. From E154's saved distances, 0.72 on both seeds. Not a bound: a map can land nearer the fine-tune than F does.
3. **Wider maps on the fine-tunes**, each against E154's best found $H$ (0.466, 0.459):
   - (c) the same map at E154's best site, at penalties from E154's smallest (a repeat, which must agree within 0.01) down
     to 1e-8, effectively unregularised;
   - (a) the map written at every position, not just the last, at the inputs of blocks 1 to 3;
   - (b) two sites written in order, every pair of the five sites: the first map is E154's, the second is fitted on the run
     already edited at the first.

**Outcome rule (the card's).** Rival a, the change is partly out of the map's reach: control at least 0.9 and one variant
raises $H$ by at least 0.2 on both seeds. Rival b, the fit itself is the limit: control under 0.5, or control at least 0.9 and
no variant raises $H$ by 0.2 on either seed. Anything else: no call, reported as read. Edge cases were run through the rule's
code before filing (`predictions.json`, outcome_rule). Wording and cut-offs are unchanged: this diagnoses the measure.

## Choices the brief left open

- **Two sites must be read in order.** Two maps both from the original's unedited activations are degenerate: a write of
  the whole activation at the later site, last position, overwrites everything the earlier write could reach. So the
  second map reads the run already edited at the first.
- **Every position only at blocks 1 to 3.** At the last two sites a write reaches the last output only through the last
  position, so it equals E154's read there.
- **No refinement in the variants.** E154 rejected its 200-step refinement at all ten of its readings.
- **Repeat tolerance 0.01** is not measured; a same-state ridge should agree to float rounding.

## Expected values

At zero steps the fine-tune is the original: the gap is 0, $H$ is undefined, and E139's rule gives no reading under a gap
of 0.02. E154's zero-step control read gap 0. A do-nothing fine-tune is the same. The control reads about 1 at the input of
block 2 and about 0 at block 1's, where F does not differ from the original. Noise floor: seed spread 0.028.

## Time and cost

One L40 job on the debug queue, $0, about 75 minutes (from E154's log, job 58754: about 100 s per site read); exit 3
resubmits. Memory as E154, about 35 GB at its peak.

## Not run

- A nonlinear map, a new fine-tune, other tasks or alphas.
- A joint fit of the two-site maps.
