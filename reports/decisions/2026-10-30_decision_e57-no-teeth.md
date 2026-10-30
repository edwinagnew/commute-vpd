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

## Revised 18:30, after E45 Amendment 2 and Stage C (supersedes the options above; Fable's advice)

**Anchor 1 never reaches the matched window.** The format-template anchor learned by step 3 and then plateaued: maths
KL 0.31 at step 20, 0.333 and 0.337 at step 100 (seeds 0, 1), against the window 0.412-0.927. Anchor 2 is learned only
at step 100, at 0.61-0.63. The two anchors share no state, and their update sizes differ too (|δW| 3.14 against 5.1).
Longer training or a higher rate would not close the gap: anchor 1 gained 0.02 nats over steps 20-100. Own samples and
the null never learned. Under E57 §11, no anchor prediction is scored.

**Stage C.** Re-gating share: E40 0.999, gold 0.999, anchor 2 0.993, random 0.993-0.994, so no teeth, confirmed.
Direct share (what the split pieces carry at the original gates): E40 0.063, gold 0.092, anchor 2 0.012 and 0.016,
random 0.022 and 0.029. Gold sits 3-4 times above the random level and anchor 2 below it. The direct share has no
bootstrap yet. Placement excess over the Gaussian null: E40 0.053, gold 0.029, anchor 2 0.091; the positive control
reads 0.21.

**Options.**
1. Ship Q3 as description and skip D (§11 as filed). $0, writing only.
2. **Fable's recommendation:** run D as unmatched description. That means anchor 1 at step 100 on both seeds, and own
   samples and the null at their nearest states. Code is reviewed. $0 on the cluster, about 2.3 h. Nothing is scored.
   It adds anchor 1's direct share and placement beside the other states. Then, if anchor 1 sits apart from anchor 2
   by more than the seed gap and the random spread, design a matched pair (option 4). If not, close D3 at this
   scale.
3. A new anchor-1 recipe. Training does not reach the window, and widening the window after the data is a refile.
   Not recommended.
4. Domain-matched anchor pair (backlog 2026-10-28). About a day of agent time and $0-15. It inherits the matching
   problem.
5. Close D3 now.

What would change it: if the direct share's row bootstrap puts gold and the random changes inside one interval, close
D3 now.

**Added 18:55 from the E45 Amendment 2 analysis** (`reports/analyses/2026-10-30_analysis_e45-amendment2.md`). Anchor 2
has no saved state between steps 12 and 100. So a lower size at which both anchors have learned is not ruled out; the
saved data just cannot show it. A sixth option: rerun anchor 2 saving states between steps 12 and 100, and look for a
learned state near anchor 1's 0.33. That costs $0 on the cluster, about 35 min per seed. If such a state exists, the
anchor rows can be scored as filed, at that matched point.
