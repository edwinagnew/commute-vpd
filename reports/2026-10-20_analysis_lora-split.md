# Analysis: the LoRA split study, both update types in one table (E14)

*2026-10-20, coding agent. Plan: `reports/2026-10-20_plan_lora-split.md`, design check confirmed by the user the same day.
Predictions: `experiments/lora_split/results/predictions.json`, filed before the run and scored as filed. Results:
`experiments/lora_split/results/lora_split.json`; figures in `experiments/lora_split/figures/`; the explanation is
`notebooks/E14_lora_split.ipynb`. Provenance at the end.*

We put a saved LoRA update and a fresh full fine-tune of the same task through one table at nine matched steps, splitting what the subcomponents cannot take by the side it misses on, then ran three states of each through the faithfulness checks on one card. The fine-tune is inside only on the read side and the LoRA is inside on neither, both write where the subcomponents do not, and neither rescales the existing subcomponents by more than seven percent. Applied as the adjoint split, the ten-step fine-tune keeps most of its behaviour at the faithfulness floor, which is the first map in this project to keep both for a fine-tune, at the price of describing a different model.

## What we were trying to test

The question since the project's first week: take a finished LoRA update, split it over the subcomponents, and see how much of it the subcomponents cannot take. Earlier runs measured one number, the share of the update outside both the read and the write directions of the subcomponents. They found that a LoRA sits outside about as much as noise does while a fine-tune sits well inside. One number cannot say which side the LoRA misses on, and the two update types had never been measured on one card at the same steps. This run does both.

The model is the released VPD decomposition of the 67M model. Each of the 24 decomposed matrices is $W = U V^\top + \Delta$, where alive subcomponent $c$ reads its input along $v_c$ and writes its output along $u_c$. The importance function Γ and the remainder Δ are frozen throughout. The task is the paper's emoticon one, where the model learns to answer a trigger with the letter o.

Three kinds of update, all on that task. The full fine-tune of the 24 matrices with Adam at one seed, run once here, with its weight change $\delta W$ kept at nine steps from 1 to 1000. The rank-8 LoRA on all 24 matrices from the plateau run, read back from its saved factors at the same nine steps; nothing is retrained. Five LoRAs on the paper's single matrix, ranks 1, 4, 8 and 32 plus a second seed at rank 8, at steps 10, 30 and 100, also from saved factors.

For every update and matrix, the update's effect on the real inputs is cut into four blocks that sum to one. The effect is $\|\delta W\,\Sigma^{1/2}\|^2$, with Σ the second moment of that matrix's inputs over Pile windows. On the write side: the part of the output inside the span of the $u_c$, against the rest. On the read side: the part of the input predictable from what the subcomponents read, $a_1 = E[a \mid V^\top a]$, against the uncorrelated remainder $a_2$. "Inside both" is energy a first-order move of the subcomponents could produce; "outside both" is what no such move can produce on these inputs. Each block has an exact reference: its value for a random update of the same size per matrix. A block divided by its reference reads 1 when the update is no more aligned with the subcomponents than noise is. Blocks are pooled by energy over the 20 matrices whose subcomponents do not already span a whole side. On the other four MLP matrices the outside block is zero for every update.

Two more numbers per update. The rescaling share is the energy a weighted least-squares fit $\delta W \approx \sum_c \alpha_c u_c v_c^\top$ explains, against two random draws; it is the strictest reading of "the update strengthens existing subcomponents". For each of the two splits below, the count of subcomponents that carry half the split's energy says how concentrated it is.

Part 2 puts three states of each update at steps 10, 100 and 1000 through the standing checks. Do-nothing leaves $\delta W$ in Δ. The least-squares split is the least-norm per-subcomponent move $(dU, dV)$ with $dU V^\top + U dV^\top \approx \delta W$, with the remainder in Δ. The adjoint split moves each subcomponent by $dU = \delta W V$ and $dV = \delta W^\top U$ with no solve and Δ untouched. Its weight change is $U U^\top \delta W + \delta W V V^\top$, which is not $\delta W$, so each matrix's move is scaled by the least-squares scalar $c$ and the state describes a slightly different model. Three checks. VPD's adversarial faithfulness check on Pile, as a ratio to the original model, with the original model's own spread under three adversary seeds as the floor. The route split at the trigger. The held-out probability of the target. The standing algebraic outside share is kept as one extra column so the earlier runs stay comparable, with the gate that it reproduces the plateau run's three numbers exactly. It did.

## What we expected, and why

- **The LoRA at its random reference on both sides at every step.** In the plateau run its read factor stayed within a degree of its random start for a thousand steps, and its outside share sat at the reference throughout. That leaves little room for either side to be inside.
- **The fine-tune inside on both sides from step 10 on**, with its outside block at most 0.7 of the reference. Its read side is shaped by the data, and the matched-matrix run's algebraic number read 0.27 against 0.53.
- **No rescaling**: the share at most three times its random reference for both updates at every step, since the first-week run read 0.06 percent against 0.035.
- **Rank is not the axis**: the one-matrix LoRAs' outside block between 0.8 and 1.2 of its reference at every rank and step, as the algebraic share had read.
- **The adjoint state**, a first reading with no threshold: its held-out probability of the target should track the share of the update it reproduces.
- **On one card**, the split LoRA's adversarial ratio inside the floor at all three steps and the split fine-tune's at least 2 at step 10.

## What we observed

### 1. Which side does each update miss?

![](../experiments/lora_split/figures/sides.png)

**Reading it.** The LoRA misses on the read side and the fine-tune does not; on the write side both updates sit at the random reference. The left panel is the share of each update's energy on the inputs the subcomponents read, divided by what a random update would have. The fine-tune reads 1.32 at its first step and stays there to step 1000. The LoRA reads between 1.0 and 1.15 at every step but one, step 100, where it reads 1.32 and then falls back. The right panel is the write side: both updates read between 0.96 and 1.0 through step 30. The fine-tune's write side starts moving inside only after step 100 and reaches 1.34 at step 1000, while the LoRA's drops to 0.84 to 0.89. The grey crosses are the one-matrix LoRAs: 1.4 on the read side at every rank, about 1.0 on the write side. The fine-tune reads 1.45 on that same matrix, so that is the matrix speaking, not the LoRA. One caveat: the write-side reference is small on the attention query and key matrices, 2 to 16 percent of the output directions, so moving inside there means landing in a thin subspace. The fine-tune's energy sits mostly on the MLP input matrices, and its write side is at or below the reference there too.

Full table: the four blocks, pooled by energy over the 20 non-saturated matrices, each beside its reference.

| update | step | inside both | write only | read only | outside both | null: inside both | write only | read only | outside both | outside both ÷ null | algebraic outside both ÷ null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ft | 1 | 0.186 | 0.006 | 0.766 | 0.042 | 0.147 | 0.048 | 0.575 | 0.230 | 0.18 | 0.282 / 0.499 = 0.57 |
| ft | 2 | 0.188 | 0.006 | 0.767 | 0.040 | 0.150 | 0.049 | 0.574 | 0.227 | 0.17 | 0.277 / 0.495 = 0.56 |
| ft | 3 | 0.180 | 0.006 | 0.772 | 0.043 | 0.144 | 0.047 | 0.577 | 0.232 | 0.19 | 0.278 / 0.503 = 0.55 |
| ft | 5 | 0.162 | 0.006 | 0.783 | 0.048 | 0.131 | 0.044 | 0.583 | 0.242 | 0.20 | 0.271 / 0.518 = 0.52 |
| ft | 10 | 0.152 | 0.007 | 0.786 | 0.055 | 0.122 | 0.042 | 0.585 | 0.251 | 0.22 | 0.268 / 0.531 = 0.50 |
| ft | 30 | 0.143 | 0.009 | 0.776 | 0.072 | 0.115 | 0.040 | 0.584 | 0.262 | 0.27 | 0.287 / 0.546 = 0.53 |
| ft | 100 | 0.151 | 0.012 | 0.744 | 0.093 | 0.113 | 0.042 | 0.565 | 0.281 | 0.33 | 0.322 / 0.565 = 0.57 |
| ft | 300 | 0.152 | 0.012 | 0.723 | 0.113 | 0.106 | 0.045 | 0.530 | 0.320 | 0.35 | 0.385 / 0.608 = 0.63 |
| ft | 1000 | 0.183 | 0.015 | 0.672 | 0.131 | 0.104 | 0.043 | 0.535 | 0.318 | 0.41 | 0.391 / 0.611 = 0.64 |
| lora24 | 1 | 0.140 | 0.037 | 0.592 | 0.232 | 0.134 | 0.042 | 0.546 | 0.278 | 0.84 | 0.523 / 0.562 = 0.93 |
| lora24 | 2 | 0.151 | 0.031 | 0.629 | 0.189 | 0.141 | 0.044 | 0.546 | 0.269 | 0.70 | 0.478 / 0.552 = 0.86 |
| lora24 | 3 | 0.145 | 0.029 | 0.645 | 0.182 | 0.135 | 0.043 | 0.548 | 0.274 | 0.66 | 0.476 / 0.559 = 0.85 |
| lora24 | 5 | 0.124 | 0.032 | 0.626 | 0.218 | 0.119 | 0.040 | 0.554 | 0.286 | 0.76 | 0.525 / 0.576 = 0.91 |
| lora24 | 10 | 0.121 | 0.038 | 0.568 | 0.273 | 0.119 | 0.041 | 0.561 | 0.279 | 0.98 | 0.574 / 0.565 = 1.02 |
| lora24 | 30 | 0.118 | 0.044 | 0.543 | 0.295 | 0.120 | 0.044 | 0.559 | 0.277 | 1.07 | 0.554 / 0.561 = 0.99 |
| lora24 | 100 | 0.077 | 0.023 | 0.701 | 0.199 | 0.076 | 0.044 | 0.512 | 0.369 | 0.54 | 0.552 / 0.674 = 0.82 |
| lora24 | 300 | 0.054 | 0.030 | 0.563 | 0.353 | 0.058 | 0.036 | 0.490 | 0.416 | 0.85 | 0.679 / 0.735 = 0.92 |
| lora24 | 1000 | 0.040 | 0.030 | 0.514 | 0.417 | 0.047 | 0.033 | 0.483 | 0.437 | 0.95 | 0.707 / 0.763 = 0.93 |

The same per matrix at step 10, with the fine-tune's and the LoRA's energy shares. The four saturated matrices, where the alive subcomponents span the whole write or read side, are `h.0.mlp.c_fc`, `h.0.mlp.down_proj`, `h.3.mlp.c_fc` and `h.3.mlp.down_proj`.

| matrix | write-side null | read-side null | ft read | ft write | ft outside | ft energy | LoRA read | LoRA write | LoRA outside | LoRA energy |
|---|---|---|---|---|---|---|---|---|---|---|
| h.0.attn.k_proj | 0.16 | 0.53 | 1.66 | 1.07 | 0.24 | 0.011 | 1.05 | 1.04 | 0.94 | 0.006 |
| h.0.attn.o_proj | 0.57 | 0.89 | 1.10 | 0.82 | 0.23 | 0.000 | 0.96 | 0.76 | 1.66 | 0.000 |
| h.0.attn.q_proj | 0.11 | 0.48 | 1.74 | 1.00 | 0.33 | 0.008 | 1.01 | 0.88 | 1.01 | 0.007 |
| h.0.attn.v_proj | 0.63 | 0.86 | 1.14 | 0.93 | 0.09 | 0.009 | 0.99 | 0.94 | 1.15 | 0.005 |
| h.0.mlp.c_fc | 0.45 | 1.00 | 1.00 | 1.01 | 2.11 | 0.078 | 1.00 | 1.00 | 0.55 | 0.084 |
| h.0.mlp.down_proj | 1.00 | 0.85 | 1.16 | 1.00 | saturated | 0.029 | 1.15 | 1.00 | saturated | 0.006 |
| h.1.attn.k_proj | 0.06 | 0.55 | 1.35 | 0.99 | 0.57 | 0.010 | 0.98 | 0.80 | 1.03 | 0.012 |
| h.1.attn.o_proj | 0.13 | 0.70 | 1.36 | 0.82 | 0.16 | 0.002 | 1.20 | 0.77 | 0.54 | 0.002 |
| h.1.attn.q_proj | 0.02 | 0.43 | 1.16 | 1.06 | 0.88 | 0.011 | 0.82 | 1.21 | 1.13 | 0.018 |
| h.1.attn.v_proj | 0.29 | 0.78 | 1.24 | 0.89 | 0.16 | 0.010 | 1.00 | 0.90 | 1.05 | 0.010 |
| h.1.mlp.c_fc | 0.08 | 0.75 | 1.29 | 0.91 | 0.11 | 0.135 | 0.99 | 0.93 | 1.03 | 0.095 |
| h.1.mlp.down_proj | 0.27 | 0.84 | 1.14 | 1.12 | 0.24 | 0.024 | 1.14 | 0.98 | 0.27 | 0.011 |
| h.2.attn.k_proj | 0.22 | 0.60 | 1.50 | 1.06 | 0.26 | 0.022 | 0.96 | 0.95 | 1.07 | 0.024 |
| h.2.attn.o_proj | 0.68 | 0.92 | 1.08 | 0.87 | 0.08 | 0.002 | 1.00 | 0.90 | 1.22 | 0.001 |
| h.2.attn.q_proj | 0.12 | 0.53 | 1.67 | 1.12 | 0.25 | 0.027 | 1.07 | 1.18 | 0.90 | 0.026 |
| h.2.attn.v_proj | 0.66 | 0.88 | 1.11 | 0.96 | 0.16 | 0.019 | 1.00 | 1.00 | 1.00 | 0.021 |
| h.2.mlp.c_fc | 0.10 | 0.79 | 1.25 | 0.76 | 0.05 | 0.184 | 1.02 | 0.95 | 0.94 | 0.087 |
| h.2.mlp.down_proj | 0.47 | 0.67 | 1.45 | 1.05 | 0.06 | 0.032 | 1.38 | 1.01 | 0.21 | 0.007 |
| h.3.attn.k_proj | 0.08 | 0.52 | 1.59 | 1.09 | 0.35 | 0.030 | 0.88 | 1.02 | 1.13 | 0.026 |
| h.3.attn.o_proj | 0.32 | 0.84 | 1.18 | 0.84 | 0.09 | 0.007 | 0.97 | 0.86 | 1.24 | 0.003 |
| h.3.attn.q_proj | 0.05 | 0.48 | 1.75 | 1.70 | 0.32 | 0.041 | 1.12 | 1.10 | 0.88 | 0.042 |
| h.3.attn.v_proj | 0.32 | 0.74 | 1.29 | 0.99 | 0.17 | 0.033 | 1.01 | 1.08 | 0.96 | 0.033 |
| h.3.mlp.c_fc | 0.34 | 1.00 | 1.00 | 0.97 | saturated | 0.220 | 1.00 | 0.98 | saturated | 0.463 |
| h.3.mlp.down_proj | 1.00 | 0.95 | 1.05 | 1.00 | saturated | 0.055 | 1.04 | 1.00 | saturated | 0.010 |

![](../experiments/lora_split/figures/blocks.png)

### 2. How far outside is each update, and does rescaling explain any of it?

![](../experiments/lora_split/figures/outside.png)

**Reading it.** The fine-tune sits inside and drifts outward as training goes on; the LoRA sits at noise throughout; rescaling explains little of either, but a growing part of the fine-tune. The left panel is the outside-both block divided by its reference. The fine-tune reads 0.18 at step 1 and 0.41 at step 1000, rising at every checkpoint. The LoRA reads between 0.54 and 1.07 with no trend, the same verdict the plateau run reached through its algebraic number, which this run reproduces to three digits. The right panel is the rescaling share against its random reference. The LoRA's stays between 0.6 and 1.8 times random, which is noise. The fine-tune's is twice random for its first 30 steps, then climbs to 9.6 times by step 1000; in absolute terms that is 2 percent of its energy rising to 7. So strengthening the existing subcomponents is a small and growing part of a fine-tune and no part of a LoRA. The first-week audit saw the same climb on a different fine-tune, and this run repeats it.

Full table: the side ratios; the rescaling share with its two-draw reference, pooled over all 24 matrices; the share of each update's energy the scaled adjoint split reproduces. The last two columns count the subcomponents carrying half of each split's energy, with the random draws' count in brackets.

| update | step | read side ÷ null | write side ÷ null | rescaling share | its null | ratio | adjoint reproduced share R | n_half least-squares (random) | n_half adjoint (random) |
|---|---|---|---|---|---|---|---|---|---|
| ft | 1 | 1.32 | 0.98 | 2.55% | 1.25% | 2.0 | 0.63 | 216 (210) | 11 (56) |
| ft | 2 | 1.32 | 0.97 | 2.77% | 1.39% | 2.0 | 0.65 | 201 (230) | 10 (54) |
| ft | 3 | 1.32 | 0.97 | 2.56% | 1.30% | 2.0 | 0.64 | 242 (224) | 11 (55) |
| ft | 5 | 1.32 | 0.96 | 2.15% | 1.05% | 2.1 | 0.59 | 297 (214) | 16 (59) |
| ft | 10 | 1.33 | 0.97 | 1.99% | 0.95% | 2.1 | 0.58 | 285 (234) | 19 (64) |
| ft | 30 | 1.32 | 0.99 | 2.07% | 0.92% | 2.3 | 0.51 | 210 (235) | 19 (65) |
| ft | 100 | 1.32 | 1.06 | 2.68% | 0.78% | 3.4 | 0.42 | 58 (232) | 24 (64) |
| ft | 300 | 1.38 | 1.10 | 3.52% | 0.67% | 5.2 | 0.33 | 97 (276) | 21 (58) |
| ft | 1000 | 1.34 | 1.34 | 6.72% | 0.70% | 9.6 | 0.31 | 83 (256) | 21 (56) |
| lora24 | 1 | 1.08 | 1.00 | 1.40% | 0.83% | 1.7 | 0.32 | 195 (263) | 48 (74) |
| lora24 | 2 | 1.14 | 0.99 | 1.69% | 0.92% | 1.8 | 0.41 | 198 (258) | 44 (71) |
| lora24 | 3 | 1.16 | 0.97 | 1.75% | 0.96% | 1.8 | 0.44 | 192 (247) | 45 (64) |
| lora24 | 5 | 1.11 | 0.98 | 1.49% | 0.88% | 1.7 | 0.36 | 189 (240) | 51 (60) |
| lora24 | 10 | 1.01 | 0.99 | 1.28% | 0.90% | 1.4 | 0.23 | 193 (232) | 69 (60) |
| lora24 | 30 | 0.97 | 0.99 | 0.97% | 0.88% | 1.1 | 0.14 | 192 (239) | 79 (69) |
| lora24 | 100 | 1.32 | 0.84 | 0.53% | 0.56% | 1.0 | 0.12 | 190 (222) | 58 (22) |
| lora24 | 300 | 1.13 | 0.89 | 0.36% | 0.44% | 0.8 | 0.13 | 173 (218) | 10 (10) |
| lora24 | 1000 | 1.04 | 0.87 | 0.27% | 0.42% | 0.6 | 0.11 | 115 (212) | 2 (8) |

### 3. Does the rank of a one-matrix LoRA matter?

The one-matrix LoRAs are the grey crosses in the two figures above.

| run | step | read side ÷ null | write side ÷ null | outside both | its null | ratio | algebraic share / null | rescaling ÷ null | R |
|---|---|---|---|---|---|---|---|---|---|
| r1_s0 | 10 | 1.45 | 0.97 | 0.010 | 0.174 | 0.06 | 0.40 / 0.42 | 3.8 | 0.47 |
| r1_s0 | 30 | 1.46 | 0.98 | 0.010 | 0.174 | 0.05 | 0.40 / 0.42 | 3.7 | 0.44 |
| r1_s0 | 100 | 1.42 | 0.99 | 0.024 | 0.174 | 0.14 | 0.34 / 0.42 | 2.6 | 0.27 |
| r4_s0 | 10 | 1.46 | 0.95 | 0.009 | 0.174 | 0.05 | 0.43 / 0.42 | 3.9 | 0.45 |
| r4_s0 | 30 | 1.43 | 0.99 | 0.018 | 0.174 | 0.10 | 0.36 / 0.42 | 2.8 | 0.31 |
| r4_s0 | 100 | 1.39 | 0.99 | 0.034 | 0.174 | 0.20 | 0.32 / 0.42 | 3.2 | 0.37 |
| r8_s0 | 10 | 1.46 | 0.96 | 0.008 | 0.174 | 0.05 | 0.41 / 0.42 | 4.1 | 0.51 |
| r8_s0 | 30 | 1.42 | 0.99 | 0.022 | 0.174 | 0.13 | 0.33 / 0.42 | 2.4 | 0.38 |
| r8_s0 | 100 | 1.39 | 0.99 | 0.035 | 0.174 | 0.20 | 0.30 / 0.42 | 3.8 | 0.43 |
| r32_s0 | 10 | 1.45 | 0.97 | 0.013 | 0.174 | 0.08 | 0.38 / 0.42 | 4.2 | 0.43 |
| r32_s0 | 30 | 1.40 | 1.01 | 0.030 | 0.174 | 0.17 | 0.33 / 0.42 | 2.7 | 0.41 |
| r32_s0 | 100 | 1.37 | 0.97 | 0.040 | 0.174 | 0.23 | 0.31 / 0.42 | 3.4 | 0.45 |
| r8_s1 | 10 | 1.46 | 0.97 | 0.009 | 0.174 | 0.05 | 0.42 / 0.42 | 4.0 | 0.54 |
| r8_s1 | 30 | 1.43 | 0.98 | 0.019 | 0.174 | 0.11 | 0.31 / 0.42 | 2.4 | 0.46 |
| r8_s1 | 100 | 1.39 | 0.98 | 0.034 | 0.174 | 0.20 | 0.29 / 0.42 | 3.2 | 0.49 |

**Reading it.** Rank does not matter, but the filed band was wrong for this measurement. At every rank from 1 to 32 and both seeds, the outside block sits within 0.03 of the others at the same step. It reads 0.05 to 0.08 of its reference at step 10 and 0.14 to 0.23 at step 100. The band filed was 0.8 to 1.2, taken from the algebraic share, which reads 0.30 to 0.43 against 0.42 here, as it did in the one-matrix run. The input-weighted blocks say something different about this matrix: its inputs are 67 percent predictable from what its subcomponents read, and every update's read direction sits 98 percent in that part. The fine-tune and the 24-matrix LoRA read the same on this matrix. So the prediction misses on its threshold while the claim it was written for holds.

### 4. Which state survives the adversary and still does the behaviour?

![](../experiments/lora_split/figures/pile.png)

Full table. Pile ratios are to the original model, whose adversarial check read 0.70 to 0.78 nats under the three seeds, a floor of 0.12 and a line of 1.37. The route split's shares sum to one. d is the share of the gain the moved subcomponents carry at the original masks, and r the share from Γ re-switching existing subcomponents. Switched-off is the share through subcomponents Γ keeps off; then come the share left in Δ and the share from Γ's own change. The adjoint state's gain and target probability are its own, not the update's.

| update | step | state | Pile PGD ÷ W0 | stochastic | CI-masked | unmasked | d | r | switched-off | Δ share of gain | Γ share | Δ share of ‖δW‖² | P(target) | gain (nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ft | 10 | do-nothing | 2.29 | 1.001 | 0.998 | 2.82 | 0.000 | 0.60 | 0.000 | 1.000 | -0.60 | 1.000 | 0.999 | 13.5 |
| ft | 10 | least-squares split | 5.24 | 1.002 | 0.996 | 1.20 | 0.052 | 0.60 | 0.001 | 0.091 | 0.26 | 0.360 | 0.999 | 13.5 |
| ft | 10 | adjoint split, scaled | 1.06 | 0.992 | 0.989 | 0.98 | 0.080 | 0.67 | -0.004 | 0.013 | 0.24 | 0.000 | 0.930 | 8.5 |
| ft | 100 | do-nothing | 1.14 | 0.959 | 0.989 | 2.67 | 0.000 | 0.53 | 0.000 | 1.000 | -0.53 | 1.000 | 0.999 | 14.2 |
| ft | 100 | least-squares split | 1.53 | 0.960 | 0.988 | 1.07 | 0.046 | 0.53 | 0.028 | 0.162 | 0.24 | 0.385 | 0.999 | 14.2 |
| ft | 100 | adjoint split, scaled | 0.93 | 0.963 | 0.995 | 0.90 | 0.068 | 0.50 | 0.008 | 0.018 | 0.41 | 0.000 | 0.597 | 5.5 |
| ft | 1000 | do-nothing | 1.72 | 0.920 | 0.993 | 5.42 | 0.000 | 0.48 | 0.000 | 1.000 | -0.48 | 1.000 | 1.000 | 18.4 |
| ft | 1000 | least-squares split | 2.00 | 0.925 | 0.997 | 1.19 | 0.027 | 0.48 | 0.029 | 0.261 | 0.20 | 0.384 | 1.000 | 18.4 |
| ft | 1000 | adjoint split, scaled | 0.95 | 0.942 | 0.996 | 0.84 | 0.061 | 0.38 | -0.001 | 0.022 | 0.54 | 0.000 | 0.125 | 2.6 |
| lora24 | 10 | do-nothing | 1.22 | 1.009 | 1.008 | 2.00 | 0.000 | 0.58 | 0.000 | 1.000 | -0.58 | 1.000 | 0.999 | 14.5 |
| lora24 | 10 | least-squares split | 1.25 | 1.009 | 1.008 | 1.35 | 0.016 | 0.58 | 0.016 | 0.188 | 0.20 | 0.410 | 0.999 | 14.5 |
| lora24 | 10 | adjoint split, scaled | 1.05 | 1.002 | 1.003 | 1.01 | 0.010 | 0.57 | 0.010 | 0.002 | 0.41 | 0.000 | 0.500 | 4.9 |
| lora24 | 100 | do-nothing | 1.07 | 0.996 | 0.999 | 1.99 | 0.000 | 0.56 | 0.000 | 1.000 | -0.56 | 1.000 | 1.000 | 14.3 |
| lora24 | 100 | least-squares split | 1.19 | 0.996 | 1.000 | 1.27 | 0.013 | 0.56 | 0.031 | 0.106 | 0.29 | 0.463 | 1.000 | 14.3 |
| lora24 | 100 | adjoint split, scaled | 1.03 | 0.997 | 1.001 | 0.98 | 0.000 | 0.39 | 0.003 | -0.006 | 0.61 | 0.000 | 0.176 | 3.1 |
| lora24 | 1000 | do-nothing | 1.04 | 1.002 | 1.003 | 1.50 | 0.000 | 0.54 | 0.000 | 1.000 | -0.54 | 1.000 | 1.000 | 15.3 |
| lora24 | 1000 | least-squares split | 1.07 | 1.002 | 1.004 | 1.21 | 0.007 | 0.54 | 0.010 | 0.313 | 0.13 | 0.542 | 1.000 | 15.3 |
| lora24 | 1000 | adjoint split, scaled | 1.00 | 0.997 | 0.998 | 0.99 | -0.003 | 0.28 | 0.002 | -0.027 | 0.74 | 0.000 | 0.063 | 1.7 |

**Reading it.** The scaled adjoint state keeps the decomposition at the floor for both updates at every step, and at step 10 the fine-tune's keeps 93 percent of the behaviour. The left panel is each state's adversarial faithfulness ratio to the original model, with the floor as the grey band. The split fine-tune reads 5.2 at step 10, repeating the 5.4 read on another card, and stays above the floor at the later steps. Its do-nothing is above the floor at steps 10 and 1000. The split LoRA reads 1.25, 1.19 and 1.07, inside the floor at every step and within 0.03 of the plateau run, and its do-nothing is at the floor too. The adjoint states read 0.93 to 1.06, all at the floor. That is expected rather than a result: the adjoint moves the subcomponents only along directions they already read and write, and leaves Δ untouched to one part in a million. The right panel is the held-out probability of the target. The adjoint state of the step-10 fine-tune reads 0.93 where the update reads 0.999. It falls to 0.60 at step 100 and 0.12 at step 1000. The LoRA's reads 0.50 at step 10 and 0.06 at step 1000. Both track the share of the update's energy the scaled adjoint reproduces: 0.58 falling to 0.31 for the fine-tune, 0.23 falling to 0.11 for the LoRA. The caveat is that this state is a different model from the fine-tuned one. The Pile checks are blind to the behaviour, so the probability of the target is the only number here with power over it.

## What we learned

The fine-tune's advantage over the LoRA in expressibility is entirely on the read side. The read side of a gradient update is the data, so the fine-tune reads what the subcomponents read because both are shaped by the same inputs. The LoRA's read factor starts random and barely moves, so it does not. The subcomponents' write directions predict where neither update writes for the first 300 steps, so the "inside" that earlier runs reported for the fine-tune was never about the output side. Longer training does not make an update easier: the fine-tune drifts outward and its rescaling share grows, so the next phase's step sweep must include late steps. The adjoint split applied as a state is the first map in this project that keeps both the behaviour and the adversarial floor for a fine-tune with Δ untouched. It keeps 93 percent of the behaviour at step 10. It does so by describing a different model that carries 58 percent of the update's energy, and what it keeps fades to 12 percent of the behaviour by step 1000. Whether a map that keeps most of the behaviour, rather than all of the update, meets the weak version in `GOALS.md` is the user's decision. The project has no threshold for it.

## How compelling is the evidence

One task, one fine-tune seed and one LoRA seed on the 24 matrices; two seeds at rank 8 on the single matrix. The four blocks have an exact reference, so an "at the reference" reading needs no draws; the rescaling reference is the mean of two draws. The Pile checks cannot see the trained behaviour; the probability of the target is held out on 73 windows and is the only number with power over it. Faithfulness ratios come from one card, an RTX A5000 with a floor of 0.12, and are not comparable across cards. Still, the LoRA states reproduce the L40S run within 0.03, and the split fine-tune reproduces the L40's 5.4 as 5.2. The per-subcomponent updates of both splits were written on the pod and deleted with it, so only their concentration counts survive.

## Surprises

**Something unexpected happened that I cannot explain.** The fine-tune's write side was predicted to be inside, at least 1.3 times its reference at step 10, and read 0.97. The fine-tune writes into the subcomponents' output directions no more than a random update does until after step 100. The read side carries all of its alignment. The write-side spans are thin on the query and key matrices, but the fine-tune's energy sits on the MLP input matrices, where it reads 0.76 to 0.91 on the write side, below the reference. Why a gradient's output directions avoid the subcomponents' output directions is not something this run can say.

Three threshold misses. The fine-tune's rescaling share was predicted to stay under three times random and reads 3.4 at step 100 and 9.6 at step 1000. The first-week audit saw the same climb on another fine-tune, and the log lists it as unexplained; it is now reproduced. The one-matrix LoRAs' outside block was predicted between 0.8 and 1.2 of its reference and reads 0.05 to 0.23. The band was taken from the algebraic share, which this run reproduces; the input-weighted block reads differently on that matrix. The LoRA's read side was predicted under 1.2 at every step and reads 1.32 at one step of nine.

Not a filed prediction, but the run's most useful reading: the adjoint state of the ten-step fine-tune keeps 93 percent of the behaviour at the faithfulness floor.

## What next

- **Recover the per-subcomponent updates.** Rerun the split-table stage on the cluster, about 15 minutes with nothing else changed, and keep the pieces on NFS. Without them the concentration counts cannot be looked at.
- **The adjoint state as a candidate map.** Its off-target damage, a second seed, and the key → value tasks, where the model lacks the association. If the step-10 adjoint state keeps the behaviour at the floor on the cipher too, it is the map to carry to the larger model. If it does so only on emoticon, it reproduces a re-weighting the model could already do.
- **Where does the fine-tune write?** Its output directions against the span of every subcomponent, dead ones included, and against the write directions of the subcomponents Γ switches on at the trigger. If the dead subcomponents' span takes it, the decomposition's choice of which subcomponents to keep alive is what the update misses. Through the design check first.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G1 | the saved factors reproduce the plateau run | algebraic outside share within 0.003 of E9 at steps 10, 100, 1000 | 0.574, 0.552, 0.707, exact | pass |
| G2 | the split state is the update at the trigger | P(target) equal to 1e-3, route shares sum to 1 | differences below 1e-5 | pass |
| P1 | the LoRA is at its reference on both sides | read side ≤ 1.2 and write side ≤ 1.3 at every step | read 0.97 to 1.16 at eight steps, 1.32 at step 100; write 0.84 to 1.00 | miss at one step |
| P2 | the fine-tune is inside on both sides | outside both ≤ 0.7 from step 10; read ≥ 1.3 and write ≥ 1.3 at step 10 | outside 0.22 to 0.41; read 1.33; write 0.97 | miss, reversed on the write side |
| P3 | nothing is rescaled | rescaling ≤ 3 × random for both updates at every step, pooled over 24 | LoRA 0.6 to 1.8; fine-tune 2.0 to 2.3 to step 30, 3.4 at 100, 5.2 at 300, 9.6 at 1000 | miss for the fine-tune from step 100 |
| P4 | rank is not the axis | one-matrix outside both in [0.8, 1.2] at every rank and step | 0.05 to 0.23, within 0.03 across ranks at a step | miss on the band; rank-independence holds |
| P5 | adjoint state behaviour tracks its reproduced share | reported | P(target) 0.93, 0.60, 0.12 for R 0.58, 0.42, 0.31; LoRA 0.50, 0.18, 0.06 for 0.23, 0.12, 0.11 | reported, monotone |
| P6 | one card: split LoRA at the floor, split fine-tune fragile | LoRA ≤ 1.37 at 10, 100, 1000; fine-tune ≥ 2 at step 10 | 1.25, 1.19, 1.07; 5.24 | pass |

## Scoring notes

The scoring code reads every rule as filed. One wrinkle: the score's convenience table of side ratios is pooled over the 20 non-saturated matrices, while the rescaling rule is pooled over all 24 as filed. The verdicts above use the filed pooling, under which the fine-tune's rescaling share reads 9.6 times random at step 1000, against 11.4 under the other. The gate on the plateau run's numbers passed to three digits; the split states' target probability matched the update's to below 1e-5.

## Not run

- The per-subcomponent updates of both splits: written to the pod's disk and deleted when it was terminated; the concentration counts in the results file are all that survives.
- The adjoint state's off-target KL, the unscaled adjoint state, and the adjoint state on any other task or seed.
- E8's factors, which share E9's recipe and seed. A second fine-tune seed. The per-step split.
- The cluster: the smoke job queued behind another user's array for 15 minutes and was cancelled; both runs went to RunPod.

## Provenance

Results `experiments/lora_split/results/lora_split.json`, sha256 7dac2efd…; smoke `lora_split_smoke.json`; logs `results/large/{full,pod_scored,pod_smoke}.log`. RunPod secure cloud, RTX A5000 24 GB, smoke 13 min and scored 85 min, $0.73. Predictions `results/predictions.json`, filed 2026-10-20 15:08 before the run.

## Hand-off

The next session starts from this report and `notebooks/E14_lora_split.ipynb`. The first item under "What next", rerunning the split-table stage on the cluster to recover the per-subcomponent updates, needs no design check; the other two go through `/design`.

