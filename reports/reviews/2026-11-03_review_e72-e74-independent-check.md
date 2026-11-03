# Review: E72 and E74, the masked forward's response to an update, and the update's parts

*2026-11-03, independent check by reading code and per-read sums with `jq`. No model load, no GPU job.*

**Plain words.** No bug found, and every number in both claims matches the files. The first claim holds with sharper wording, and it survives conversion to every ruler. The second claim's gap is about what size alone gives, so it must not be read as the in-span part carrying the behaviour.

## Claim 1, the masked forward: holds with a wording change

**Entry is right.** `qwen_transmission/run.py:184-191` and `qwen_pass_test/amend1.py:55-56` compute `((x V)·(m−d)) U + d·(x Wᵀ) + x δWᵀ`. That matches `site_forward` (`decomposed_linear.py:142`) and `masking.py:135-140` at 177d0cce8, the JAX commit the port follows, and the same algebra in 74146b5's `components.py`. The hook rebuilds FT to 6e-10 of KL, which pins sign and scale (u1, 8 rows).

| 32 rows; ceiling columns 64-128 rows | KL passed, fixed / stochastic | amplitude | cosine | KL to FT, update on vs off | gain, W0's masks |
|---|---|---|---|---|---|
| u1 | 0.0055 / 0.0128 | 0.095 / 0.129 | 0.33 / 0.54 | 1.244 vs 1.374 | 0.258 |
| u3 | 0.00041 / 0.00107 | 0.032 / 0.051 | 0.22 / 0.34 | 0.355 vs 0.364 | 0.022 |
| u2, no transmission read | | | | 1.095 vs 1.237 | 0.443 |

Fine-tuned masks, update off to on: gain -0.038, -0.019, +0.005; KL to FT +0.0002, +0.0015, -0.0005; gain with the update off 0.24, 0.13, 0.72.

**The objection.** Gains of 0.26 and 0.44 do not sound like "barely". But the clean path (`qwen_diagnose/results/u*/dw_*.json`) is steep: 1% of δW alone keeps about 0.19 and 0.21 of the gain on u1 and u2. Converted to the share of δW that reads the same in the clean model, KL passed, KL gap closed and gain all give 1-3% (by interpolation, across row sets). The rulers agree.

**Ruled out.** KL direction: amplitude has no direction and reads the same order. Precision: whole-update numerators are 1e-4 to 8e-3. Δ: fully on reads 1.262 and 1.108 against 1.244 and 1.095 (T0b).

**Controls.** The last matrix is real for entry but cannot fail through later masks, since none follow it; its band was rewritten after u3 missed, disclosed. `up_0` and direct's `dw_27` are identities, and K0 against E68 is the same path. t = 1 (K 0.013, u1 and u2) is the only one through every masked matrix. **No control shows that a change expected to pass does pass at an early block.**

**Alternatives the wording must leave open.**
1. It is not about the update: Gaussians of matched clean effect pass amplitude 0.02-0.07 at blocks 0-14, and T0a reads "any change".
2. All 196 sites masked at once is not VPD's training forward: `launch_config.yaml:234-235` routes `uniform_k_subset`. The whole update under partial routing was never read.
3. The fine-tuned-mask limb reads distance to FT, not displacement, and those masks already hold part of FT's behaviour.
4. The masked stream sits 25-73% from the clean one at KL 0.07, so it may be a different network reaching the same logits.

**Suggested wording.** "On Qwen3-1.7B, with all 196 matrices masked by the original model's masks, fixed with Δ off or VPD's stochastic draw, adding the whole update unmasked at each matrix shifts the logits by 0.03-0.13 of its clean amplitude, or 0.0004-0.013 of its clean KL (two updates, 32 rows). That is about what 1-3% of the update does in the clean model. Random changes of matched clean effect at single blocks are lost as well. Under masks made on the fine-tuned model, the update moves gain by at most 0.04 of the fine-tune's and KL to it by at most 0.0015 nats (three updates). Why the masks lose it is not shown."

## Claim 2, the clean model: holds with a wording change

| u3 / u2 / u1 | KL(FT ‖ state) | agreement with FT | input-weighted norm share | δW rescaled to that norm |
|---|---|---|---|---|
| in-span | 0.0027 / 0.0090 / 0.0092 | 0.992 / 0.970 / 0.972 | 0.94 / 0.92 / 0.93 | ~0.0007 / 0.003 / 0.002 |
| remainder | 0.220 / 0.527 / 0.531 | 0.922 / 0.827 / 0.838 | 0.35 / 0.39 / 0.36 | ~0.09 / 0.20 / 0.21 |
| random | 0.576 / 1.466 / 1.385 | 0.805 / 0.704 / 0.756 | KL-matched, scale 56 / 18 / 18 | |
| W0 | 0.323 / 1.377 / 1.635 | 0.914 / 0.803 / 0.808 | 0 | |

**Code is right.** `qwen_fixed_masks/run.py:109-117`: the parts sum to δW within 1e-6, and `clean_FT` reads 0.

**What it does not show.** The rescaled update (last column, from E73; 64 rows on u1 and u2; near t = 1 extrapolated from t = 0.6) gives a gap of 60-130 times, against 58-81 read. Each part reads 2.5-4 times further from FT than the rescaled update. The remainder is not inert: alone it keeps 0.92, 0.94, 0.27 of the gain. One temperature removes 0.42-0.50 of KL(FT ‖ W0) on the gold updates. Top-token agreement, which temperature leaves alone, does favour the in-span part.

**Random is a poor reference for a part:** it is matched on KL, at 18-56 times scale, and on u1 reads nearer FT than W0 does. A random slice of δW was never read.

**Side defect.** `qwen_fixed_masks/rules.py:181` sets `s` to 1 minus a norm fraction (0.64), filed as the in-span share. With the in-span norm share the "carried" label still holds: margins 0.20, 0.21, 0.02 against 2w of 0.06, 0.05, 0.01.

**Suggested wording.** "In the clean model, W0 plus the least-squares in-span part of the update reads 0.003-0.009 nats from the fine-tuned model and W0 plus its remainder 0.22-0.53, where W0 itself reads 0.32-1.64 and a random change matched on clean KL 0.58-1.47. The in-span part holds about 0.93 of the update's input-weighted norm and the remainder about 0.36; the update rescaled to those norms reads 0.001-0.003 and 0.09-0.21, so the gap is about what size gives."

## Could not check

Anything on a GPU; g's values; per-position KL (not saved); the port's K0 against JAX's logged figure; u4 beyond the ceiling.

## Not run: passes that would settle the open points (one L40, minutes, none submitted)

1. Add ε times the on-subcomponents' own output at blocks 0 and 7, masks fixed: the missing early control.
2. The whole update under `uniform_k_subset`-style partial masking.
3. A random slice of δW with the in-span part's energy, clean model: the true null for claim 2.
4. KL between the update-off and update-on forwards under fine-tuned masks.
