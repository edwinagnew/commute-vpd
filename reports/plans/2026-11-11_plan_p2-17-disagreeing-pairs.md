# Plan: where the prior and steering disagree, which does the share follow? (cell P2.17, E### claimed by the lead)

*2026-11-11, compare-lead (designer), part 2. Card `reports/plans/2026-11-11_plan_p2-17-card.md` (check passes: 4.71
standard errors at n=3). Filed rule `experiments/disagreeing_pairs/results/predictions.json`. Apparatus, function names and
stage timings: `reports/plans/2026-11-11_plan_p2-17-appendix.md`. Sides are named as built; no verdict on reuse.*

## The question, in plain words

On the nine built pairs, two cheap readings together call which side reads higher the same way the re-gating share does:
the original model's prior for the trained answer, and steering at strength 1 (the fine-tune's mean change added to the
original model). So those pairs cannot show the share seeing anything the two miss. We build new pairs of 67M fine-tunes
where the two cheap readings disagree, prior one way and steering the other, or the prior matched and steering flat, and
read which of them the share's direction goes with. This is option b of the decision of 2026-11-10, sized by the user
on 2026-11-11 (two runs, cluster first, RunPod up to $15).

## Design check

- **Goal.** Part 2 of the paper roadmap: what the re-gating share adds over simple baselines on the 67M. Fills open cell
  P2.17 within its run count, so no sign-off beyond the cell.
- **Nearest tried.** E149 (prior plus strength-1 steering on the nine built pairs: every pair tracked) and E147 (the
  96-key refits). They settled that prior and steering never disagree against the share on existing pairs; this adds
  pairs where they disagree by construction. P2.16 found the share's gap sizes follow neither reading (n=9).
- **Patch or step.** A step: the only design that can show whether the share's direction carries anything beyond the
  two cheap readings.
- **Decisions relied on.** The 96-key recipe of E58 (`keydep.finetune`, step 30, seeds 0-2), E143's call rule and E149's
  bars, unchanged. Departure: the common-word cipher draws from frequent words (ranks 0-2999), not E58's cipher pool
  (1000-2999); that is how its prior is raised, stated as a confound.
- **Definition.** No rule, band or score toward a reuse verdict. The reading is "which cheap reading the share's direction
  agrees with", per pair.
- **What changes what we do.** Share follows the prior (or steering) on both opposite pairs: part 2 says so and writes the
  paragraph (option d). Split or neither, with the flat pair separated: part 2 has its pairs where the share's direction is
  not either cheap reading. Fewer than two pairs buildable: the cell closes as "not buildable with this apparatus".
- **Control first.** Stage 0, alone: zero update, a known pair re-read, a refit, the priors. It fails on a changed reader
  path, a changed training path, or a prior path that no longer gives the saved numbers.

## The pairs (one new fine-tune family each, 3 seeds; the other sides are E147's saved refits)

| pair | reuse side (saved) | new side | prior gap, decades | strength-1 steering gap | built to be |
|---|---|---|---|---|---|
| A | copy, 96 keys | **common-word cipher** (new) | about -1.2 (copy 5.91e-5 vs about 1e-3) | about +0.4 (copy 0.56 vs ciphers 0.08-0.12) | opposite |
| B | half-known map, 96 keys | common-word cipher (same fine-tunes as A) | about 0 to +0.3 (half-known 1.33e-3) | expected under 0.1 (half-known not read at strength 1) | flat |
| D | rank-1, 96 keys | **one-word map** (new) | about +3.5 (rank-1 1.26e-2 vs under 1e-5) | about -0.5 (rank-1 0.13 vs 0.5-1.0) | opposite |
| C | rank-1 | class map shuffled (saved, 4 new words) | +4.58 | about -0.4 | reported only (share seen) |

Sources: priors `experiments/prior_vs_share/results/stage2.json` (points, `e58_96_*`); strength-1 steering
`experiments/prior_steering/results/stage1.json` (states `e58/*`, pooled); searched steering for the class and half-known
maps `experiments/refit_96key/results/stage1.json`.

- **Common-word cipher.** Each of the 128 keys maps at random to a frequent word whose original-model probability after
  that key is between 5e-4 and 3e-3, past the key's own top 20 guesses, and no more than 3 times its median over all keys:
  likely everywhere, not a guess tied to its key. It keeps the cipher's lack of a key-to-answer link and raises the prior
  by token choice. One set of fine-tunes serves A (prior against steering) and B (prior matched, steering flat).
- **One-word map.** Every key maps to one word the original model gives under 1e-5 after every key. Its whole update can
  be one mean shift, so steering should read high. D reverses the direction of A.
- **Checked before training.** Both priors need only the original model: stage 0 computes them and stops if a map misses
  its filed range. Steering needs the fine-tunes: stage 1 reads it on all trained sides and stops before the share if
  fewer than two pairs meet their condition.

## Stages (one debug job on an L40, about 12 min, $0; times and memory in the appendix)

| stage | what | must read | else |
|---|---|---|---|
| 0 control | zero update through both readers | steering rise exactly 0; share unreadable (gain 0) | exit 1 |
| 0 control | positive control: copy vs prior-matched cipher, saved states | share 0.771 / 0.378, steering 0.560 / 0.115, each within 1e-4; condition "not met" (prior matched, steering separates) | exit 1 |
| 0 control | prior-matched cipher seed 0 refit from its windows | share within 0.0088 (its seed range) of E147's | exit 1 |
| 0 control | priors of copy, half-known, rank-1, prior-matched cipher, class shuffled | each within 1e-4 relative of the saved value | exit 1 |
| 0 tables | build both maps from the original model's probabilities | common cipher prior in [4.2e-4, 4.2e-3]; one-word under 1e-5; pool not exhausted | exit 1, no training |
| 1 | 6 fine-tunes; strength-1 steering on 18 states (6 new, 12 saved) | at least two of A, B, D meet a condition | exit 1, share not read |
| 2 | share on the 18 states; the score | saved states reproduce E147's share within 1e-4 | exit 1 |

**Expected at the references.** Zero update: gain 0, share and steering unreadable, steering rise exactly 0. A
do-nothing pair: steering unreadable, so "condition not met"; it can never be counted. Positive control as in the table.

## The filed rule (in full in `predictions.json` → definitions)

Each reading calls a pair by E143's rule on three seeds a side (separates iff the gap is at least 0.1 and above the larger
within-side range); the prior by its sign when at least 0.5 decades. A pair is **opposite** if prior and steering both
call and point opposite ways, **flat** if the prior is matched and steering does not separate. On an opposite pair the
share **follows the prior**, **follows steering**, or **neither** (does not separate). Outcome over the counted opposite
pairs: prior, steering, neither, or split. Flat pairs are listed beside. Pair C is never counted.

**Predictions.** P1: all three pairs meet their condition (medium). P2: on A the share follows steering, copy higher (low;
the 24-map prior line predicts no call, the nearest no-link map at a middle prior sat 0.18 under it). P3: on D it follows
the prior (high). P4: on B the share separates, half-known higher (low). P5: outcome split (low).

## Cost and routing

Cluster `debug` QoS, one job, about 12 min, $0; run 2 is held for one amendment (e.g. a pool that runs out). If not started
within 10 min, RunPod 4090 (about $0.7/h, under $1.50 with set-up): there the saved sides are refit from their windows
(about 1 s each), since about 900 MB of saved updates cannot cross the 80 KB/s uplink; the positive control then checks
the refits against E147's shares. Saved: `results/stage{0,1,2}.json`, the tables, and the six new updates in bf16 (about
340 MB) in `results/large/`.

## Not run

Any verdict on reuse; searched-strength steering; per-key steering as a scored reading; pairs off the 96-key family;
neurons, the untrained decomposition or update size on the new pairs; Qwen.
