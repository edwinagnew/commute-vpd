**Verdict: holds with a caveat.** Both fails compute, and I found no bug: right update, applied once, in fp32, against the state's own clean model. The caveat is about what the fail means. The update reaches the masked forward in do-nothing only as (1 − d) x δW, or through the changed activations and importances, so this read may be largely the adversary switching part of the update off, and no saved number separates that from extra damage under masks.

# Independent check of E89: the adversary on E81's TOFU update (2026-11-05)

By a fresh Opus agent that wrote none of this. No model was loaded. I read the code and logs and recomputed with `jq` and `awk`; the per-step ratios are in `~/commute-data/scratch/qwen-decomp-lead/e89-check/per_step.txt`.

Names: W0, the unchanged decomposition; N, do-nothing; Rare, the update on the rarely-on half; u1, the first maths fine-tune; r, a state's final KL over W0's at the same seed.

## 1. Is δW applied once, to the right weights, in a precision that keeps it?

Yes.

- **Right file.** My `sha256sum` of `results-large/qwen_tofu/dW/a2_2e-5/step_1250.safetensors` gives `d45b5163…`, the same as `results/inputs_SHA256SUMS`. It holds 196 F32 tensors, the decomposed matrices and nothing else (`fleet/ckpt-info`). The pod's log has "sha256 matches" (`run.log`, line 2).
- **Right sign.** E81 saves `p - W0` (`src/commute/qwen_rl.py:615`). E81's own check applies it with the same `plus` (`experiments/qwen_tofu/check_after.py:37`).
- **Right size.** Σ‖δW‖² is 323.944, so ‖δW‖ is 17.99845. u1's Σ‖δW‖² is 3.48520 (`qwen_d1_adversary/results/results.json`, `.weights`), so the factor is 9.64.
- **Applied once.** N is `set_weights(plus(W0, dW))` (`qwen_teacher.py:306-308`, `qwen_sft.py:63-65`), which swaps `.data` and does not write in place. `J.W0` holds the original tensors (`qwen_d1_adversary/run.py:295`). State contexts are not nested (`qwen_d1_adversary/amend1.py:58-59`; E89 `run.py:55`).
- **Checked three times in the run.** `n_state` compares the swap with W0 weights plus hooks (`qwen_d1_adversary/run.py:127-135`). This reads −1.2646e-09, the same to every digit, on entry to each N start, including the one after Rare s0. If weights were left applied, or applied twice, this would not be near zero.
- **Rare's weights.** They are the same as N's: the clean-KL difference is exactly 0.0.
- **Precision.** Weights, δW, components and the hook all run in fp32 (`qwen_vpd.py:329-334`). δW's root-mean-square entry is about 4.8e-4 (17.998 over the square root of 1.41e9 entries), far above fp32's rounding on weights of this size. Γ runs in bf16 in every state.

**Is W0 unpolluted?** Yes, as far as this can be shown.

- The columns are restored exactly from a host copy of the same dtype (`qwen_maps.py:53-54, 274-278`).
- W0's reads come after Rare's clean read, which applied and removed the moved columns, and they match E85's to four digits through step 9 on s0 and step 7 on s1. The small divergence after that is the size seen between pods before.
- Gap: E89 has no W0 read after a Rare start, as it cut E85's `kept_W0_end`. The nearest evidence is E85's Rand s1, on the same code after Rare s0, which read r = 1.015. I rate this low risk.

## 2. Is the KL against the correct clean model? Is the 0.64 in it?

**The reference is correct.** The clean logits are recomputed on every row and pass inside the state's context, with capture hooks only (`qwen_teacher.py:297-299`, `qwen_vpd.py:305, 312`). The loss is `kl_per_position(clean, masked)` (`qwen_vpd.py:373`). So N is compared with the updated model and W0's weights do not enter the read. The 0.64 is not baked in as a floor: at m = 1, d = 1 the masked forward equals the clean one exactly and the read is 0.

**But the update's own move can come back in.** From the hook (`qwen_vpd.py:334`), masked output = xV(m − d)U + d·xWᵀ, where the weight matrix W is whatever the model holds. With Δ = Wᵀ − VU, at a site with input x:

clean output at x − masked output = xV(1 − m)U + (1 − d)·xΔ, with 1 − m = (1 − g)(1 − s).

- For N, Δ = Δ₀ + δW. So δW's only direct route is (1 − d)·xδW. The adversary holds one d per matrix. At d = 0 the matrix runs on VU alone, with the update and Δ₀ removed. Pushing d down reverts that matrix towards W0, and that is part of the gap to the updated model.
- There is a second route, without d: x and g come from the updated model (`qwen_vpd.py:301`), so the mask term acts on moved activations and importances. Nothing in this run measures it.

**Which is it?** It is neither a wrong-reference artefact nor shown to be purely damage under masks. The evidence, none of it decisive:

- **At step 0, N and Rare read the same, before any search.** N reads 0.2015, 0.2023; Rare 0.2012, 0.2022; W0 0.0918, 0.0916. Sources are the same per seed (`qwen_vpd.py:401-403`), so W0, N and Rare are paired. The ratio is already 2.2. A random d averages 0.5 and a rarely-on column's m averages 0.5, so about half the update is off in either place. If KL is quadratic in the missing update, that predicts about a quarter of 0.64, or 0.16 nats, against the measured excess of 0.11. The shared activations and importances would also give N and Rare equal reads, so this does not separate the two routes.
- **The 0.64 cannot simply be added.** W0 + 0.64 gives ratios of 1.37 and 1.44, under the bar, against reads of 2.15 and 4.19. Yet if the update's move and W0's damage add coherently, the reach is (√W0 + √clean)²: 4.43 and 4.00 nats, against finals of 3.67 and 6.03, and last-five means of 2.96 and 4.03. The same sum on u1, with clean 0.040, gives 2.26 and 1.96 against 2.26 and 1.67. So no third mechanism is needed in size, though s1's final exceeds the sum.
- **This is arithmetic, not a law.** Three things weaken it. 0.64 is KL(W0 ‖ state) (E89 `run.py:56`, `qwen_sft.py:293-297`), while the adversary reads KL(state ‖ masked), the other direction, and the reverse KL was never measured. u1's 0.040 is on E79's other web rows. And the same sum fails on E85 Amendment 1's random change at 100 times, 5.0 against 55.4. Per-site dials can also remove a coordinated update unevenly, which can cost more than removing it whole.

**Reading.** The fail is real and it is the kind a do-nothing test should give, since it says Δ can be turned down and the model then moves. "Masks damage the updated model more than W0" is not established. Any write-up should say the adversary can switch off part of the update in Δ, and that the split between that and mask damage was not read.

**Why it cannot be split from saved data.** `start` discards the final sources (`qwen_d1_adversary/run.py:101`, `[0]`), and E89 did not read E85 Amendment 1's "masks at g, Δ off". Three cheap reads would settle it:
1. KL(updated ‖ W0) on the 128 rows, needing no components and fitting a cluster L40.
2. E85 Amendment 1's Δ-off read on N.
3. One N start with every d held at 1, about 22 min on an 80 GB card.

## 3. What does the masked forward do with Δ? Is it the same mechanism across states?

It is one hook and one formula, so yes for W0, u1 and N. Only the Δ in the second term differs: Δ₀, Δ₀ + u1, Δ₀ + TOFU. So the larger read comes from a dial on a larger Δ, plus whatever the moved activations and importances add. It does not come from masks acting on Δ, since Δ is outside the mask term.

Rare is the same formula with the update in a different term. Weights stay W0 + δW and V, U move on the half (`qwen_maps.py:239-240, 269-270`). That leaves 23.4% of the update's squared size in Δ, second-order term included, so 76.6% does not sit there. On columns with g near 0 the mask term's coefficient is about (1 − s), one setting per column. So Rare's route is mostly masking out moved columns, with far more independent settings than N's one dial per matrix. Γ's readouts also read the moved V. That Rare reads above N despite less in Δ fits this, though it was not separately tested. N and Rare should not be described as the same failure.

## 4. Do the ratios and label follow?

Yes. Recomputed with `jq` from `.starts`:

| | s0 | s1 | mean | last-five mean |
|---|---|---|---|---|
| N | 2.15118 | 4.19224 | 3.17171 | 2.70 |
| Rare | 4.99013 | 6.80303 | 5.89658 | 4.07 |

- W0's spread is 0.18435. Both gaps (2.17, 4.90) are strictly above it and both means are over 1.61, which is the bar in `rules.py` and the filed one. So N reads "fails", the outcome reads "do-nothing fails on this larger update", and the pair reads "both fail" (`rules.py:38-55`, `rules_amend2.py:32-53`).
- The rules and code were committed at 10:46 (91f96e0a), before the log's first line at 11:18. The log's finals match `results.json`.
- **Unlike E85's Rare, this does not depend on the step scored.** Over all 20 logged steps, the lowest single-seed ratio is 1.81 for N (s0, step 13) and 2.18 for Rare. Both are over the 67M bar of 1.61 and Qwen's own 1.776.
- The number itself is soft. N's seeds differ by a factor of 2, against 0.12-0.19 in r at u1's size. Rare s0 goes 5.69 at step 19 to 8.50 at the final read. Say "about 3" for N, or quote the seeds; do not quote 3.17.

## 5. Are W0's reads consistent with E85's, and are the settings the same?

Yes.

| W0 | E89 | E85 | difference |
|---|---|---|---|
| s0 | 1.70422 | 1.70316 | 0.06% |
| s1 | 1.43894 | 1.43849 | 0.03% |

- Each uses this job's own W0, so these differences do not reach the ratios beyond 0.06%. The plan says three pods read s0 to every digit, and E89 is the first differing read of s0 that I found. The size matches earlier pod-to-pod differences on s1 (0.04%, E85's check). I did not find the cause, and it does not matter at this size.
- Same code: `git diff --stat 119e5d25 4632d45a` over the shared modules, E80's and E85's run files and the rules shows only `qwen_d1_teeth/amend1.py` added. That is E85's later amendment, whose `read` E89 calls.
- Same rows (`J.rows["W"]`, the 128 rows of 512 tokens), same 20 sign steps of 0.1, one source per site, final read, fp32 components, bf16 Γ.

## 6. Does the comparison with u1's 1.24 compare unlike things?

Apparatus: no. Update: yes.

- **Same apparatus.** Recomputed r is 1.3245, 1.1625, mean 1.2435. Rows, seeds and steps are the same. So is the card model: `run_amend1.log` line 3 reads H200 NVL, and W0 s0 there equals E85's to every digit. The plan's "H100 not H200 NVL" is wrong, in the comparison's favour.
- **u1's δW is bf16, TOFU's fp32.** The bf16 comes from `qwen_d1_adversary/run.py:59`'s comment. The difference of 2.9 GB between the peaks (66.1, 69.0) fits a matrix set twice the bytes. u1's rounding is applied through the same `plus`. I expect this to be small, but it was not measured.
- **Size is not the only thing changed.** u1 is 10 steps on maths. TOFU is 1,250 steps on invented biographies. One pair of updates cannot say size was why, so the plan's reading of this run as a question about size over-reads it. No size ladder ran (the plan's "Not run").
- **‖δW‖ is the wrong size for these rows.** Norm grows 9.64 times and squared size 93 times. But the clean KL goes from 0.040 to 0.64, 16 times, or about 4 times in square root. u1's is on E79's rows and TOFU's on the 128 rows, so that factor is rough. u1 moves maths rows by 1.84 nats, so u1's pass may reflect these web rows hearing it little. N's excess goes from 0.553, 0.234 to 1.96, 4.59. The plan's power-law guess of about 51 nats, ratio near 30, missed by a factor of 10 and should not be repeated.

## Anything else that would change a reader's mind

- **The claim's update is not the chosen anchor.** It is E81's one-wording update. The user chose E84's five-wording update as anchor 2 this morning, of about the same norm (18.109). The plan says so, but the claim's sentence does not. E84's has not been read under the adversary.
- **"Placed on the rarely-on half" needs its remainder.** 23.4% of the squared size stays in Δ, and the state is not a pure placement. Say "on the half, with the remainder in Δ".
- **The bar is borrowed.** 1.61 is E71's, from the 67M. This does not change the outcome, as Qwen's 1.776 is also cleared at every step.
- **The fail says nothing about a good state.** From the formula, an update heard on these rows is out of both routes' direct reach only when it is off the Δ dial and on columns with g near 1. This run does not show any state on TOFU's update can pass. Least squares leaves 9.8% of the squared size in Δ, about 0.31 of the update's size, and that is the next read that matters.
- **Web rows only.** Rows on which TOFU was trained were not read.

## Not run

Any forward pass or GPU read; the reverse clean KL; the Δ-off read; a masks-only start; the final dials (not saved); N's importances against W0's on these rows; any check on E84's update; a rebuild of Rare's split (E85's check rebuilt it on sampled matrices for u1, and here only the logged 0.234 and the same-weights check were read).
