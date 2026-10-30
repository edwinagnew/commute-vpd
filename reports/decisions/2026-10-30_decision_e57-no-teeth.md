# Decision needed: E57 row 1 has no teeth at scale; what Stage D reads (2026-10-30)

On Qwen3-1.7B, two random weight changes, sized to move the outputs about as much as the real E40 update, both read
a re-gating share of about 0.99. That is above the 0.8 line that, per the plan, shows the reading has no teeth. Recomputing
the masks on the changed model's activations recovers almost all of any change, real or random, so the re-gating
share (E57 row 1, and E45's filed share row) cannot tell re-gating from "the importance function follows the
activations" at this scale. The plan's tree (`reports/plans/2026-10-30_plan_q3-reading-the-update.md` §11) sends this
to the user before Stage D; Stage C (E40, gold, anchor 2) runs meanwhile as descriptive rows.

## Numbers (Stage C0, job 46350, `experiments/qwen_reading/results/C0.json`)

| control | maths KL vs E40 0.62 | re-gating share (95% boot) | direct share | K0 | K00 | K01 | K10 | K11 |
|---|---|---|---|---|---|---|---|---|
| Gaussian, off-span | 0.53 | 0.994 [0.993, 0.995] | 0.022 | 0.070 | 0.927 | 0.129 | 0.909 | 0.124 |
| random change to alive pieces | 0.50 | 0.993 [0.992, 0.994] | 0.029 | 0.070 | 0.694 | 0.117 | 0.677 | 0.113 |

K-cells: mean per-position KL to the changed model of the original-weights model masked by the original pieces or the split ones (first index) under the original model's or the changed model's masks (second index); K0 is W0's own masking error.

## Two more problems found in Stage A/B2 (Fable)

- The filed split-half floor is 21,110 switched-on pieces (5 × the 4,222 that differ between two halves of W0's rows).
  It measures how gating differs between different rows, not the noise of the same-row comparison (W0 vs W0 reads 0).
  The prompt row (|S_prompt| 1,860) and both random changes (6,961; 5,170) fall under it, so rows 2-3 come back
  unscored for most states.
- The overlap excess of the random changes is 19-75 standard deviations above the site-stratified draws against every
  reference set, unrelated web text included. Switched-on pieces are the near-threshold ones, which any new context
  switches on, so the uniform-within-site null reads "huge overlap" for anything that moves.

## Options for Stage D ($0 on the cluster, ~2.3 h; it cannot start before anchor 1 is trained)

- **A. Run as planned.** All rows scored as filed, row 1 labelled no-teeth. Placement is the one decisive row left
  with teeth; rows 2-3 mostly unscored or unreadable.
- **B. Run D and drop row 1.** Saves minutes; leaves a filed prediction unscored. Not worth it.
- **C. Hold D.** Nothing new to wait for beyond Stage C.
- **D (Fable's recommendation).** Run D after Stage C is read, rows scored as filed and row 1 written up as having no teeth, after a
  dated amendment to rows 2-3 (CPU, $0): a same-row floor for S, and the overlap measured against C0's two random S
  sets at matched size instead of the uniform draws. Report the direct share (random 0.02-0.03) as the one route-split
  number with a null.

What would change it: if Stage C's E40 and gold direct shares also read about 0.03, no route-split number has teeth and
D runs for placement and weight statistics only.

For the re-gating line (not this lead's): no random change has been read through r on the 67M where r is defined. Fable
advises a KL-matched random δW check there ($0, minutes) before the line's Qwen stage 1 (decisions row 94).
