# Review: E68's ceiling and the opening of the masks

*2026-11-03, independent check, read-only. Commit read: e683886c. No model load, no GPU job.*

**Verdict: holds with caveats.** The numbers are right and mean what they say: I found no bug in the code, and every score recomputes from the per-read sums. Two parts of the sentence need changing. "So it is the masks, not the missing Δ" is stronger than these reads allow, because Δ on under the masks was never read and t = 1 mostly measures Δ's own size. And "closes only 0.03-0.14" is true of KL alone: next-token NLL keeps 0.26-0.48 of the gain on three of the four updates. Opening the masks was read on u1 and u2 only, 32 rows each, so it says nothing yet about u3, which is the update with the lowest number.

Suggested wording: *"On Qwen3-1.7B, with masks from Γ on the original model's forward and Δ off, adding the whole fine-tuning update unmasked at all 196 matrices closes only 0.03-0.14 of the KL gap to the fine-tuned model. On the two updates opened (first 32 rows), switching every subcomponent fully on with the update added reads KL ~0.013, so the absence of Δ alone does not explain the gap; the masks are needed for it. Whether restoring Δ under the masks would help, and whether the masks treat this update differently from any other change of the same size, was not read."*

## Numbers recomputed with jq

From `results/u*/W0.json`, `do_nothing.json` and `results/amend1/<u>_{zero,ceiling}.json`. All match `amend1/score.json` to about 1e-15.

| update | rows | K0 | KD | K_C | r2_C | r2 without K0 | gain_C | zero vs E68 do-nothing | KL(FT ‖ W0 clean) |
|---|---|---|---|---|---|---|---|---|---|
| u1 | 128 | 0.0696 | 1.3738 | 1.2440 | 0.0995 | 0.0945 | 0.258 | 0 | 1.635 |
| u2 | 128 | 0.0696 | 1.2367 | 1.0952 | 0.1212 | 0.1144 | 0.443 | 0 | 1.377 |
| u3 | 64 | 0.0699 | 0.3640 | 0.3554 | 0.0291 | 0.0235 | 0.022 | 0 | 0.323 |
| u4 | 128 | 0.0696 | 1.4884 | 1.2914 | 0.1388 | 0.1324 | 0.477 | 0 | 1.718 |

`amend1/open.json` also matches, each key recomputed (KL to FT, 32 rows):

| t | u1 on | u1 off | u2 on | u2 off |
|---|---|---|---|---|
| 0 (from Amendment 1's first 32 rows) | 1.3535 | 1.4954 | 1.1947 | 1.3475 |
| 0.1 | 1.3344 | 1.4977 | 1.1762 | 1.3491 |
| 0.5 | 1.0747 | 1.3450 | 0.9614 | 1.2356 |
| 1 | 0.01263 | 1.8657 | 0.01245 | 1.5739 |

KL(FT ‖ W0 clean) on these rows is 1.8203 for u1 and 1.5404 for u2. K(1, on) / K(0, on) is 0.0093 and 0.0104, so both read "masks" under the filed rule. Per row, K(1, on) never passes 0.023 and the row ratio never passes 0.027, so it is not a few-row effect. The ceiling's K is below the zero read on every row of all four updates; per-row r2_C has medians 0.099, 0.121, 0.030, 0.138 (u3 has one outlier row at 2.8, on a small denominator).

## Real problems: what the sentence claims

1. **"Not the missing Δ" is not shown.** `amend1.py:148` compares K(1, on) with K(0, on) only. That shows opening the masks is enough to fix the gap and that Δ's absence alone costs only 0.013. It does not show Δ would not help under the masks: the plan's own Not run list has "Δ on with masks" (plan line 82). The masked forward's site inputs are not W0's, so Δ's contribution there need not match its small one at t = 1. No read here bounds it. The number does not change; the causal clause does.

2. **t = 1 is close to a restatement of "VU ≈ W".** At t = 1 the output is x V U + x δWᵀ (`amend1.py:55-56`), the fine-tuned model minus Δ. It reads near 0 for an honest reason, not a wrong one: with nothing added it reads 1.866 and 1.574, next to the clean 1.820 and 1.540, so the update is what closes it. But its content is "Δ is small on these rows", not "the update needs what the masks switch off". All components go on, including the roughly 194,000 of 688,128 outside the alive set (`job.py:55`, `qwen_maps.py:64-66`), which the label "every subcomponent fully on" says correctly.

3. **Scope: opening reads only u1 and u2, first 32 rows** (`amend1.py:137`, `OPEN_ROWS` line 35). The sentence attaches it to a range that comes from all four updates. u3 and u4 have no opening read. This matters for u3: it has the lowest r2_C and is the one update where KL and NLL agree that little comes back. E68 is at its two-amendment limit, so opening u3 needs a new plan.

4. **KL and NLL disagree on three updates.** `gain_C` is 0.26, 0.44, 0.48 on u1, u2, u4 against r2_C of 0.10, 0.12, 0.14; on u3 they agree at 0.02 and 0.03. "Closes only" should name KL. NLL is not a clean ruler here either, see 5.

5. **Next-token NLL and KL-to-FT both move with the masks alone, so neither measures only the update.** Off-curve gain on the first 32 rows (same formula, zero added): t = 0.5 reads 0.38 on u1 and 0.53 on u2 with no update at all, and t = 1 reads -0.42 and -0.59. With the update, t = 0.5 reads 1.02 and 1.56, past FT. KD is below KL(FT ‖ W0 clean) on u1, u2, u4 (1.374 against 1.635 on u1), so the masks alone move the model nearer FT by more than the update then does (0.26 against 0.13 nats on u1). Also KL(FT ‖ W0 clean) is 1.64 nats while FT's NLL improves by only 0.17 on u1, so most of that KL sits in changes next-token NLL does not see. I did not establish why; per-position values were not saved. It is a reason not to read 0.03-0.14 as "the share of the update's useful effect".

6. **"Of the KL distance" is loose.** r2_C divides by KD − K0, not by the whole distance. Effect is small: 0.024-0.132 without K0. Say "of the gap above the masks' own damage", or accept the range as approximate.

## Alternative readings that fit equally

- **Masks drop any change, not this update's parts.** The `if` text in `predictions_amend2.json` says "the update needs parts of the model the masks switch off". An equally good fit: the masked forward is only built to be faithful on W0's own activations, and it attenuates any shift in them, whatever the shift. The update is added at each matrix, but its effect on later matrices has to pass through their masks. Nothing here separates the two. The separating read is a random change of matched clean KL added the same way: if it also reads r2 ~0.1, the ceiling says nothing about what the update needs. This is the alternative I would run down first.
- **Reassembly, not parts.** K(t, on) stays high until the very end: 1.35, 1.33, 1.07, 0.013 on u1. Half-open masks recover little, then the last stretch recovers nearly everything. That shape fits a sum that only rebuilds W when all components are at exactly 1, as well as it fits any specific parts being needed.
- **Masks' choice, not masks as such (a sharpening, from E68's own file).** `u1/A_do_nothing.json`: masks from Γ on FT's own forward, nothing added, read K = 0.104 on 128 rows, against 1.374 for W0's masks. So on u1 W0's masks are the problem specifically, and most of the update's KL can be carried by which subcomponents fire. This supports "masks" and also fits the elicitation limit in memory: a new circuit from old pieces reads as elicited. Its NLL gain is only about 0.24, so it does not settle what is useful.

## What I checked and found sound

- **importances leaves nothing behind.** `clean_forward`'s hooks all return None (`qwen_vpd.py:297-303`) and are removed by `_hooks`. It writes no weights. Pinned empirically: the hook control reads K = 1.1e-9 right after it in the same row.
- **Weights.** The model is at W0 at each row's start: `Reader.read` passes `keep` so `applied` rebinds the same tensors (`amend2.py:306,309`), `row` ends with `use("W0")` (`amend1.py:79`), and `set_ft` asserts W0. The masked branch ignores the module's weights anyway (`amend1.py:55`). The reference is FT, read after the hooks are removed (`amend1.py:76-77`).
- **do-nothing in the read.** `change("do_nothing", ...)` returns `(M, None, None, None)` (`qwen_maps.py:218-219`), so `applied` touches no U or V; the `continue` is at `qwen_maps.py:257`. `alive.uv` only copies. `dW` is built but goes unused when `weights` is given.
- **The added term at every site.** x is the Linear's first positional input at all seven kinds, including `o_proj` (attention's output) and `down_proj` (the gated product) (`modeling_qwen3.py:82,200-202,229`). Weight shape is [d_out, d_in], so `x @ dW.T` is right; `(scale * x) @ dW.T` by Python's left associativity, same value. All seven Linears have no bias in this config (`bias=False`, `attention_bias` default False), so the masked branch drops nothing. I read the default, not this model's config file.
- **File and step per update.** `Reads.select` calls `job.select` (which calls `forget`, clearing `_dw`), rebinds `rd.dW` from `job.dw`, and `drop_ft` sets `ft_t = 0` (`amend1.py:102-106`, `run.py:92-100`, `amend2.py:292-294`). So u2 to u3 to u4, all step 100, do rebuild FT; the cache key `(t, False)` cannot leak across `select`. `rd.dW` and `set_ft` use the same dict. The zero read equals E68's do-nothing KD exactly on all four updates, including the 128-row u4 from a separate job. File dates precede all these jobs.
- **Row sets and denominators.** K0 comes from the right row set (`amend1.py:169`: maths from u1, a1 from u3); u1, u2, u4's W0 sums are the same, and recomputing each from its own W0 file gave the same scores. `gain_C` uses the ceiling's `ft`, the right update's (`amend1.py:183`). `rules.gain` uses W0's `ft`, which would be wrong if pointed at u1's file for u2, but amend1 does not do that. `mean_of` is `amend2.mean`'s formula. T is 256 with all positions summed.
- **Controls against the ceiling's path.** The hook control tests `base = out` plus the added term, not the masked branch. So it does not by itself test the ceiling. But the zero read pins the masked base (gap 0 from `masked_logits`), the hook pins the added term, and t = 1 with scale 1 runs the two together through the masked branch on u1 and u2 and reads 0.013. Line 56 is shared. Only t = 0, scale 1 has no direct check of its own, and it differs from those in a value, not in code. This is sufficient.
- **Code changed between jobs.** `forward_plus` gained `t` in 51b09fff. Job 50356 ran the earlier form on u1-u3. With t = 0, `gs + 0 * (1 - gs)` equals `gs` exactly, so the files are comparable.

## Style, no effect on a number

- Every read logs as `[do_nothing ...]` (`amend1.py:117`), including ceiling and open. It is confusing in `run.log`.
- `k0` at `amend1.py:147` is K(0, on), not E68's K0.
- `nll` in the log is divided by T = 256, though it sums 255 positions, so log values read 0.4% below `score.json`'s (2.0558 against 2.0638). Ratios are unaffected.
- u3's file has no identity record (`run.py:137`, `want None`): sha256 is only logged, `ece3bd44...`.

## Could not check

- Anything on a GPU: no rerun, so g and the forwards are checked by reading the code and by the controls' numbers.
- Per-position KL, which was not saved. So I could not see which positions carry the KD or the 0.03-0.14.
- Whether FT as defined here (W0 in fp32 plus the bf16 δW file) matches the model as trained. Training updates only these 196 matrices (`qwen_sft.py:178`), and the hook control pins FT = W0 + δW. The trained model's own precision was not compared.
- The hook control ran on u1, 8 rows only. u3 and u4 have neither a hook read nor a t = 1 read of their own; they rely on the shared path.
- Which activations the masks change at t = 0. It is not in any file.

## Not run

Δ on under the masks (the read item 1 needs); a random change of matched clean KL added unmasked (the first alternative); opening on u3 and u4; the ceiling under masks from FT's forward; masks applied to some layers only. E68 has used both its amendments, so any of these is a new plan.
