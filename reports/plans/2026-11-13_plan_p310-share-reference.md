# Plan: a reference for the re-gating share on the frozen-neuron decomposition (cell P3.10, E181)

*2026-11-13, qwen-lead's writer. The user approved option (a) of
`reports/decisions/2026-11-13_decision_qwen-share-reference.md` (decisions.md, 2026-11-13: "yes, run option a"); design
check `reports/plans/2026-11-13_plan_qwen-share-reference-design-check.md`. Card `experiments/qwen_share_reference/card.md`
(passes `fleet/card check`, 7.1 standard errors at n = 5). Rules: `experiments/qwen_share_reference/results/predictions.json`.
Not gated: about $12 and 3.5 h, under $60 and 6 h.*

**The request, in the Terms' words.** On p-015e0710 (subcomponents fixed at the MLP neurons, only Γ trained, MLP
matrices only), read the re-gating share of (1) a built change that switches off neurons the model uses on a row set,
three choices of set, at the size of the real fine-tunes; (2) random changes at three sizes across the fine-tunes'
range, three draws each, matched in size within 10%; then (3) place P3.9's 36 read fine-tunes against the random curve
at their own size. One figure, one paragraph, no reuse or new label.

## What runs

One job, `experiments/qwen_share_reference/run.py`, on one H100 NVL (or two pods split by `--rowsets`). On each of
the library's five row sets (copy, cipher, known, invented, gold; 128 rows of 256 tokens):

| stage | what | reads | minutes (H100 NVL) |
|---|---|---|---|
| 0, alone, first | W0 as the changed model on 16 rows of each row set; the do-nothing state of random draw 0 (scale 100) through the split path | 0 | ~3 |
| 1 | usage (mean \|a\| per neuron at W0 on all five row sets), then 3 built changes per row set | 15 | ~2 + 15 x 4 = ~62 |
| 2 | 3 random draws per row set, each at 3 sizes (one split per draw) | 45 | 15 x 8 = ~120 |

Load ~1 min plus the decomposition fetch (~8 min on P3.9's pods). Total about 3.3 h on one card, about 1.8 h on each
of two pods (`--rowsets copy,cipher,known` and `--rowsets invented,gold`). The log prints `STAGE 0 DONE`,
`STAGE 1 DONE`, `STAGE 2 DONE`. Every read is written as it finishes (`results/{control,built,random}/*.json`, a few
hundred KB each) and a rerun skips what is written. Then, with no GPU, `run.py --summary` and `plot.py`.

**Size.** K00, mean nats per position over the row set's 128 rows: the changed model against the original
subcomponents under W0's masks. P3.9 stored it for every fine-tune (`run2/*.json .means.K00`), so no state is fetched.
Targets per row set: the lowest fine-tune K00, the highest, and their geometric middle (table in predictions.json;
e.g. copy 0.37 / 0.85 / 1.94, known 0.33 / 0.48 / 0.68). Built changes sit at the middle, random draws at all three.
Matching: the search runs on the first 32 rows, on K00 - K0 (the size above W0's own masked error, which makes the
size grow from 0), to within 1.05x; a read counts as matched when its 128-row K00 is within 1.1x of its target. 32 rows
because P3.9's records put the 32-row K00 within 4.2% of the 128-row one (16 rows: 5.5%), so 1.05 x 1.042 < 1.1; one
evaluation costs ~15 s against ~55 s for 128 rows.

**Built change.** Neurons switched off by zeroing their gate_proj rows (dW = -W0 there): silu(0) = 0, so the neuron's
output is 0 whatever up_proj gives. Each neuron is one subcomponent of p-015e0710, so the split represents the change
exactly as a mask of 0. Γ's readout at the down_proj site reads the neuron's actual activation (V = I there), which is
now 0; at the gate and up sites it reads x V with the original V, unchanged. Whether Γ's masks on the changed model
switch the neuron off is what the share's K01 tests. Choices made here (in the weeds, decided by the writer):
- "Used" = mean |a| over positions 1-255 of the row set's 128 rows at W0, a = down_proj's input. Position 0 is left out
  because a row's first token carries attention-sink activations that would rank a few neurons by one position.
- Pool: per layer, the top 10% of the live gate neurons by that usage (615 a layer).
- Three orders of the pool, each interleaved over the 28 layers, so a change of k neurons takes about k/28 a layer, as
  a fine-tune spreads over layers: **top** (most used first), **random** (seeded order; the same neurons' kind without
  the ranking), **specific** (highest ratio of usage on this row set to the mean on the other four first: the neurons
  this text uses more than the library's other text, closest to what a fine-tune on it would target).
- k is searched (from 28, at most the pool) with the same 32-row search, count steps ∝ size. Risk: a massive-activation
  neuron at the head of **top** can overshoot the middle size at k = 1; that read is then unmatched, reported and
  unreadable, and the other two choices still read.

**Random change.** iid Gaussian on the 84 MLP matrices at the per-matrix norms of the row set's seed-0 late fine-tune
(P3.9 run 1's `.random.norms`), draws 0-2 (draw 0 is P3.9 run 1's direction), times a scale b searched per size. The
split is linear in the change, so one split per draw serves its three sizes (rescaled in place).

**Shared code.** The reading steps moved into `src/commute/share_reading.py` (rows, W0 control, do-nothing, the four
cells, the size measure, the search, the curve and the placement rule); `experiments/qwen_library_share/run.py` stays as
it ran. `src/commute/qwen_reading.py` is unchanged. Tests: `tests/test_share_reading.py` (11, the edge cases below).

## Expected values

- **W0:** K00 = K01 = K0 and the zero split's K10 = K11 = K0 within 1e-6; K0 on 16 rows ~0.13-0.23. Stop on a miss.
- **Do-nothing:** share 1 within 1e-6. Stop on a miss.
- **Every read:** mean K0 over 128 rows equals P3.9's floors.json within 1e-3 (P3.9's reads matched within 2e-4). Stop.
- **Random draws:** scored; share ~0.80-1.06 (P3.9 run 1: 0.81-1.06), falling as K00 grows; b ~30-3000.
- **Built changes:** rival A, at or above the curve plus the band (near 1); rival B, on the curve within the band.

## Filed predictions (predictions.json)

The curve per row set: per size with at least two usable draws, the point (geometric mean K00, mean share), linear in
log K00 between points. The band: the largest distance of a usable random draw from the curve at its own K00, measured
in this job. A reading's gap = share - curve at its K00: above (gap > band), below (gap < -band), inside, or unreadable.
- **Built (15 reads):** 'above random' (rival A) if at least 80% of readable reads are above, at least 9 readable;
  'like random' (rival B) if at least 80% inside; 'below random' if at least 80% below; else 'mixed'.
- **Fine-tunes (36, already read):** the same rule, at least 27 readable. A description only.
- **Edge cases** (run before filing, `tests/test_share_reading.py`, 11 passed): a gap equal to the band is inside; band
  0; one usable draw at a size (size dropped); fewer than two sizes (no curve, unreadable); NaN shares; K00 outside the
  span by more than 1.1x; the search overshooting at k = 1, at its cap, f ≤ 0, NaN, a zero target.
- **Do-nothing against the rule:** share 1 would place 'above' any curve under 1. It is the path's control, not a
  candidate; built and random changes go through the same least-squares split, so only Γ's masks on the changed model
  separate them.

## What differs from the design check

- The design check's stage 1 early stop (built must read above random on one row set before the rest) is not used: the
  brief orders controls, then built, then random, and a built change at or below random is itself the result rival B
  or the "below" outcome reads. The stops are the controls only.
- Sizes are K00 (128 rows, masked), not the 16-row unmasked KL of P3.9's search; the match is 1.1x, not 1.5x.

## Cost and pod

About $12 on one Vast H100 NVL at $3.45/h (3.3 h + set-up), or the same on two pods in half the wall time. Pod command
(from the clone root on the pod, after the venv and the decomposition fetch as P3.9's pods did):

```
python experiments/qwen_share_reference/run.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" [--rowsets copy,cipher,known]
```

Files the pod needs that git does not carry: p-015e0710's W0 g statistics,
`experiments/qwen_library_share/results/large/p-015e0710/gstats/W0_{maths,fineweb}.safetensors` (2 MB each; the live
set). `results/large` is a symlink, so stage them by hard link under a path ending in `results/large/` and replace the
symlink on the pod with the real directory (root CLAUDE.md). P3.9's floors, run1 norms and run2 records are in git.
Download `experiments/qwen_share_reference/results/` (a few MB) after each stage, and `run.log`.

## Not covered

A built change that switches neurons on; attention matrices; the RL updates; a seed spread of the built change (one set
per choice); the cluster's L40 route (not tried here: one pod is faster for 3 h).
