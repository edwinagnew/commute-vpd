# Appendix: apparatus for the held-model plan

*2026-11-18, definition-lead, cell P1.26. Companion to `2026-11-18_plan_held-model-retention.md`. Sizes and file
layout here are the lead's choices, not definition choices.*

## The model as the code loads it

The 67M is param-decomp's LlamaSimpleMLP, configured in `artifacts/vpd/t-9d2b8f02/model_config.yaml`, loaded by
`commute.steering.Net`.
- 4 blocks. Each adds the attention write to the residual, then the MLP write.
- Attention has 6 heads of width 128 in a residual of width 768. Positions are rotary. There are no biases. The heads
  are joined by the output matrix `o_proj`.
- The MLP maps 768 to 3,072 to 768 with NewGELU and no bias. The output matrix is `down_proj`.
- The norms `rms_1`, `rms_2` and `ln_f` are RMSNorm with a learned gain vector.
- The embedding is tied to the unembedding.
- So reading 1 has 12,288 neurons and 24 heads; reading 2 has 24 heads and 4 blocks; reading 3 has 8 sites.

## What H is, per reading

H is a copy of M in which only `down_proj` and `o_proj` change.
- Reading 1: column j of `down_proj` at layer l becomes $g_i w^M_j$, where F's neuron i is paired with M's neuron
  j. Head h's 128 columns of `o_proj` are scaled by its gain in the same way. An M unit with no non-zero partner gets
  gain 0.
- Reading 2: all of `down_proj` at layer l is scaled by the block's gain. The heads are scaled as in reading 1.
- Reading 3, option (a) of question 1: `down_proj` is replaced by the map $A_l$ of size 768 × 3,072 that minimises
  $\sum_x \lVert m^F_l(x) - A_l a^M_l(x) \rVert^2$. Here $m^F_l$ is F's MLP write and $a^M_l$ M's neuron activations,
  both on M's residual x at that layer. `o_proj` is replaced in the same way by a 768 × 768 map from M's
  concatenated head outputs to F's attention write.
- Least squares in fp64 through the normal equations: $A = C_{FM} C_{MM}^{+}$, summed over every fit position. The
  pseudo-inverse is taken at fp64 machine tolerance. No ridge, since a ridge would be a choice of target.
- Gains: $g = \langle f^F, f^M \rangle / \lVert f^M \rVert^2$ for the paired unit. This is exactly `gain` in
  `RF.pair` today.

## Code that already exists

| need | where | change |
|---|---|---|
| cross sums and norms on M's inputs, for neurons, heads and blocks | `src/commute/rule_retention_fixed.py:profile` | none |
| assignment, gain, zero units left out | `src/commute/rule_retention_fixed.py:pair` | a weighted score if question 3 is (a): the score $\langle f^F_i, f^M_j\rangle^2 / \lVert f^M_j\rVert^2$ |
| M's input at each layer; activations, head outputs, output columns | `src/commute/rule_retention.py:m_inputs`, `acts`, `heads`, `cols`, `block` | none |
| KL per window, fp64 | `src/commute/rule_retention.py:kl` | none |
| the check that H reproduces F | `src/commute/rule_retention_fixed.py:gate` | the same max relative logit error, H against F |
| clone M; the circuit and its attribution | `experiments/rule_retention/run.py:clone`, `rescaled`, `attribution` | none |
| circuit-off; the rebuilt item's state | `experiments/rule_retention_fixed/run.py:zeroed`, `trained`, `stage2`; `results/large/rebuilt_mlp.pt` | none |
| the five pairs' fine-tunes and windows | `experiments/rule_retention_pairs/run.py:fine_tune`, `Data.read_half`, `subset`, `gap`; `src/commute/old_tasks.py:task_windows`, `old_f` | the fit half, which E196 did not use |
| seeds 2 and 3 of the five pairs | `experiments/curve_test/results/large/*_s2.pt`, `*_s3.pt` | a path map beside the existing one |
| MLP and whole-update sizes | `experiments/rule_retention_pairs/update_size.py:rel`, `sizes` | none |
| E187's pairs, scalars and states | `experiments/size_matched/run.py:pairs`, `times`; `results/m67.json` for c | none; the scaled states are rebuilt from saved updates times c |

New code: a builder for H per reading, the reading-3 normal equations, the two gains-only builders and the survival
number. That should be about 150 lines in one new module, called from one run script.

## Windows

- Old tasks use 8-token windows. Each task's 200,000 windows split into a fit half and a read half; twins share them.
- Fit: 16,384 windows of the fit half, every position, about 131,000 positions. That is over 40 per input of
  reading 3's widest map.
- Read: 512 windows of the read half with E196's seed, at the last position.
- Code checks and the rebuilt item use copy24's halves, as E196's control did.
- E187's pairs use their own tasks' halves, from the loaders E187 called.

## Every number and its expected value

| number | F = M, do-nothing | gains-only per unit | gains-only per block | circuit-off |
|---|---|---|---|---|
| KL(F‖M), last position | 0 | positive | positive | positive |
| KL(F‖H), reading 1 | 0 | 0 | 0 | 0 |
| KL(F‖H), reading 2 | 0 | positive | 0 | positive |
| KL(F‖H), reading 3 | 0 within rounding | 0 within rounding | 0 within rounding | 0 within rounding |
| S, readings 1 and 3 | no change | 1 | 1 | 1 |
| S, reading 2 | no change | reported | 1 | reported |
| fitted gains, reading 1 | all 1, identity pairing | the planted gains, shuffle undone | the planted gains | 0 on the 8 circuit neurons, else 1 |

"0" means that H reproduces F within a maximum relative logit error of 1e-3. A miss stops the job. For the rebuilt item,
the trained pairs and E187, nothing is filed. They report S per reading, KL(F‖M), KL(F‖H) and the per-site local
fit residual.

What each reading is blind to:
- Reading 1 cannot see a rule rebuilt as a mix of several M neurons.
- Reading 2 cannot see changes inside a block that keep the block's direction.
- Reading 3 cannot see any change that is linear in M's activations at a site.
- None of them sees a change in what a layer's input carries if F's units at every site are still gains of M's.

## Figure sketch, made-up numbers

```
reuse minus twin survival, per pair; dots are seeds 0-3; colour per reading
           | built strip         | copy/cipher | ident/rand | case  | repeat | twin/fact
 +0.30     |                     |   1 1       |            |   3   |        |
 +0.10     |                     |   2 3 3     |  1 3       |  1 1  |  3     |
  0  ------|--do-nothing, checks-|-------------|------------|-------|--------|---------
 -0.10     | rebuilt: S1 .62     |             |  2 2       |  2    |  1 2   |  1 2 3
           |   S2 .40  S3 .91    |             |            |       |        |
```

## Cost, from measured times

- E196 read 20 fine-tunes in 4 to 11 minutes each on H100 SXM pods, almost all of it smallest-set search.
- E193 took 17 minutes for about $1.20 and E194 31 minutes for about $2.25 on the same card, about $4.40 an hour.
- Here an item is one fit pass over 131,000 positions, normal equations for 8 sites and three forwards on 512 windows.
  That is an estimate of 1 to 2 minutes. The first item's time is logged and checked against it.
- Run 1: 45 items, about 1 to 1.5 H100-hours. That is two or three cluster debug jobs at $0, or one pod at about $5
  to $8.
- Run 2 with E187: about 70 states, about 2 H100-hours, about $10.

## The negative case: full recipe, for the user to approve

1. On M, on 32-token Pile windows, measure each of the 24 heads' mean attention at every offset from 3 to 16,
   leaving out the first two positions, which act as a sink. Pick the offset k where the largest head's share is
   lowest. Record that share s.
2. Append one head at the last layer with fresh random q, k, v and o weights from an independent seed. It reads
   `rms_1`'s output, uses rotary positions and writes into the residual stream. Every weight of M is frozen.
3. Train only the graft on cross-entropy at the last position, against the token k positions back. Use Adam at 1e-3, the rate of
   P1.24's appended-neuron fits, with warm-up, at most 4,000 steps of 64 windows, on cluster debug.
4. The build succeeds only if all four hold:
   - held-out accuracy at least 0.9, with M's measured on the same windows;
   - the graft's mean attention at offset k at least 0.8;
   - every M head's share at offset k stays at s, since M is frozen;
   - F with the graft removed equals M within the logit tolerance.
   None of these uses the held model.
5. Stop after 4,000 steps whatever the result. One build attempt is one of the cell's runs.
6. Read it with the three readings and the code checks of run 1.

Expected by arithmetic: readings 1 and 2 give S = 0, since the graft is unpaired and every other unit of F is M's at
gain 1. Reading 3 is the test.
