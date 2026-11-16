# Plan: repair the share reference of P3.10 so its filed rule can make a call (cell P3.11)

*2026-11-16, qwen-lead's writer. Cell P3.11 (open, 2 runs, due 2026-11-18, up to $20 on a pod). Repairs E181
(`reports/analyses/2026-11-13_analysis_e181-qwen-share-reference.md`). Code `experiments/qwen_share_reference_repair/`;
rules `results/predictions.json`; card `card.md` (7.1 standard errors at n = 5). Not gated: under $60 and 6 h.*

**The request, in the Terms' words.** On p-015e0710 (subcomponents fixed at the MLP neurons, only Γ trained), read
E57's re-gating share of the same 15 built changes as E181 with a size search fine enough to land each on its target,
and add three random draws per size to E181's three; then apply E181's filed rule unchanged. No reuse or new label.

**Design check.** Serves part 3 of the plan: the share needs a reference on Qwen before it can read anything there.
Nearest tried: E181, which made no call (7 of 15 built changes on target; copy's band 4x the others from one draw);
this is a patch on it, asked for by the user, same rule. The W0 and do-nothing controls run first and alone, and stop
the job on a miss.

## What runs

`run.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" --rowsets ...`, two pods on H100 NVL (not PCIe: E181's PCIe pod ran 1.5-1.9x
slower). Per row set, 128 rows of 256 tokens:

- **Stage 0** (first, alone): W0 as the changed model on 16 rows, and once per pod the do-nothing state of draw 0.
- **Stage 1, built changes**: E181's three neuron orders (most used, random among used, used most on this text). Size x
  is continuous: the first floor(x) neurons' gate_proj rows zeroed, the next one's row multiplied by 1 - frac(x). Each
  neuron is one subcomponent, so this is a mask of 0, or of 1 - frac(x), on the subcomponents it touches. x is searched
  to the row set's middle fine-tune size (32 rows, within 1.05x, up to 20 evaluations against E181's 8: cipher's
  'specific' order sat between counts 278 and 288 after 8). Matched means 128-row K00 within 1.1x of the target.
- **Stage 2, random draws 3-5** at the low, middle and high fine-tune sizes, by E181's exact procedure. Draws 0-2 are
  E181's records, read from `experiments/qwen_share_reference/results/random/`.
- **Summary** (no GPU): curve and band over six draws, built changes and P3.9's 36 fine-tunes placed on it.

## Expected values

- **W0:** K00 = K01 = K0, zero split's K10 = K11 = K0, within 1e-6 (K0 on 16 rows: copy 0.179, known 0.132). Stop on miss.
- **Do-nothing:** share 1 within 1e-6. Stop on miss. **Every read:** K0 equals P3.9's floor within 1e-3. Stop on miss.
- **Built:** at least 13 of 15 matched; the five 'top' orders land between 1 and 4 neurons, with one turned down partway.
- **Random:** scored, share about 0.80-1.06, scale b within 0.7-1.3x of E181's range at that size.

## Filed predictions, in words

E181's rule over the 15 built reads: a built change is above when its share exceeds the random curve at its own size by
more than the band, the largest distance of any random draw from the curve. If at least 80% of at least 9 readable
reads are above, the call is "above random" (rival A: the share sees a switching-off); if 80% are inside, "like random"
(rival B: it tracks size only); else "mixed", or "unreadable" under 9. The fine-tunes are placed the same way, as a
description. Edge cases are recorded in `predictions.json`.

## Time and cost (from E181's pod b, H100 NVL: one 32-row evaluation 18 s, split plus read 168 s, a random draw 9.5 min)

| stage | per row set | 5 row sets |
|---|---|---|
| load + set-up (decomposition fetch, venv) | per pod ~19 min | 2 pods: ~39 min |
| 0: W0 control (~0.9 min) + do-nothing once per pod (~2.3 min) | 0.9 min | ~9 min |
| 1: 3 built changes, ~9 evaluations each (E181's matched ones took 3-7, giant cases need ~14) | ~17 min (at 20 evals: 26) | ~84 min (at most 132) |
| 2: 3 new random draws at 3 sizes | ~28.5 min | ~143 min |

Two pods: pod A `--rowsets copy,cipher,known` ~2.7 h, pod B `--rowsets invented,gold` ~1.9 h; 4.6 GPU-hours, about
**$18** at $3.99/h. If every built search used all 20 evaluations, 5.4 h and $21: the runner stops the pods at $20 of
spend, and since each read is written as it finishes, a rerun (the cell's second run) completes what is missing.

## Not covered

A switching-on built change; attention matrices; a second seed of each order; a redrawn sketch (the card points at E181's).
