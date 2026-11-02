# Decision needed: is anything left worth running to tell "already there" from "new" on Qwen3-1.7B?

*Opened 2026-10-30, rewritten in plain words 2026-11-02. No new data since 2026-10-30. The original text is kept
unchanged under "Record" at the end.*

**The choice.** Test the one remaining reading against size (A), compare the two reference fine-tunes at equal size
(B), stop and write up (C), or read the last planned fine-tunes without matching size (D). I recommend A, then B or C
by its result. Reply with a letter.

## Why it matters

The project asks whether a simple map fails because training teaches new capabilities rather than strengthening
existing ones. To answer that on Qwen3-1.7B, we need a reading of the decomposition that gives different values for
the two cases. Without one, readings of the GSM8K and RL fine-tunes describe them but cannot say which case they are.

## What we did

We read saved fine-tunes through the frozen decomposition and checked each reading against two things.

- **Random weight changes.** A useful reading should not confuse these with training.
- **Two reference fine-tunes whose kind is known by design.**
  - The *template fine-tune* teaches an answer format the model already follows when a system prompt asks for it. This
    is the "already there" case.
  - The *invented-facts fine-tune* teaches facts about people who do not exist. This is the "new" case.

The two must be compared at equal size, because bigger changes read differently for trivial reasons.

## What is stuck

**The switching reading tells us nothing here.** Start from the KL between the changed model and the CI-masked
original decomposition. Changing the subcomponents and recomputing the masks together removes most of it. The re-gating
share is the fraction of that removal achieved by only recomputing the masks, with Γ on the changed model's
activations. It reads 0.99 or more for every change, random ones included.

**One reading is left, and size may explain it.** The direct share is the fraction achieved by only changing the
subcomponents, with the original masks kept.

| change | size | direct share (95% row bootstrap) |
|---|---|---|
| fine-tune on human GSM8K solutions, 100 steps | 1.44 | 0.092 [0.090, 0.093] |
| same data, 10 steps at full learning rate | 1.30 | 0.063 [0.062, 0.064] |
| random Gaussian change | 0.86 | 0.022 [0.020, 0.024] |
| random change to the alive subcomponents | 0.63 | 0.029 [0.026, 0.032] |
| invented facts, seed 0 | 0.64 | 0.012 [0.010, 0.013] |
| invented facts, seed 1 | 0.63 | 0.012 [0.011, 0.013] |
| template | not read | not read |

Size is the KL from the changed model to the CI-masked original decomposition, minus the same KL at the original model,
in nats per token on the same 128 maths rows. It measures movement in outputs, not parameter norm. Three things limit what the table shows:

- The maths fine-tunes read above random, but they are also the biggest changes. No run has varied size with the kind
  of change held fixed.
- The two random changes differ by 0.007, so smaller gaps mean little.
- At equal size, invented facts read below random. Nobody predicted that, and it is one fine-tune read off its own topic.

**The two reference fine-tunes have never been compared.** Matching uses a second yardstick, KL from the original
model on the maths rows, with a filed band of 1.5 times either way.

| fine-tune | saved state | KL on maths rows | learned? |
|---|---|---|---|
| template, seeds 0 and 1 | step 100 (level from step 20) | 0.33, 0.34 | yes: follows the template on 92-96% of answers |
| invented facts, seeds 0 and 1 | step 12 | 0.31, 0.31 | no: recalls 2% of the facts |
| invented facts, seeds 0 and 1 | step 100 | 0.63, 0.61 | yes: recalls 91% of the facts |

No invented-facts state between steps 12 and 100 was saved, and the template fine-tune has not been read through the
decomposition at all. So none of the predictions filed for the reference fine-tunes can be scored.

## Options

### A. Test the direct share against size first (recommended)

- **What runs.** Scale random changes up to the size of the biggest fine-tune, read a smaller saved state of that
  fine-tune (step 9, KL 0.31), and read them all the same way.
- **Cost.** About 2 hours on the cluster, $0, with a small code change to set the scale. This is my estimate from about
  35 minutes per state; it has not been planned in detail.
- **If random reaches the fine-tune's level at equal size:** the direct share only measures size, no reading is left,
  and we go to C.
- **If random stays well below:** training differs from random beyond size, and we go to B. The exact line would be
  filed before the run.
- **Limit.** This shows whether B is worth running. It cannot itself separate "already there" from "new".

### B. Compare the two reference fine-tunes at equal size

- **What runs.** Retrain invented facts, saving states between steps 12 and 100. Read the learned state nearest the
  template's size, along with the template fine-tune, on two seeds each.
- **Cost.** About 3.5 hours on the cluster, $0: my estimate from 35 minutes per seed to retrain and about 35 minutes
  per state to read. The earlier "35 minutes per seed" counted only the retraining, which alone gives no comparison.
- **What you get.** The first same-size comparison of "already there" and "new".
- **Limits.**
  - A learned state of that size may not exist. It would need at least 25% recall (the filed rule) by KL 0.50.
  - The direct share may be too small there to rise above the 0.007 spread. Invented facts already read only 0.012 at
    about twice that size.
  - A clean gap could still be about topic. The template fine-tune works on maths answers. On the maths rows, invented
    facts mostly shift answers to one line rather than expressing the facts.
- **What happens next.** If the two differ by more than the gap between seeds and the 0.007 spread, the direct share
  becomes a candidate for reading the GSM8K and RL fine-tunes, and the new pair below becomes worth building to rule out
  topic. If they do not differ, or no such state exists, we go to C.

### C. Stop and write up

- **Cost.** Writing only.
- **What it says.** The switching reading cannot separate anything at this scale, the direct share separates maths
  fine-tunes from random with size unresolved, and the reference comparison was never possible. It could not say that
  VPD fails to separate the two kinds, because that was not tested.
- **What you give up.** Later GSM8K and RL readings would be description only.

### D. Read the template fine-tune as it is, unmatched

- **What runs.** The template fine-tune and two fine-tunes trained on the model's own GSM8K answers, with and without a
  filter for correct ones, each at its nearest saved state.
- **Cost.** About 2.3 hours on the cluster, $0. The code is already reviewed and merged. No prediction made in advance could be scored.
- **Limit.** It fills in the table, but the template is about half the size of invented facts, so any gap between them
  could be size, and we would still have to choose between B and C afterwards. Fable recommended this on 2026-10-30, before the size problem was found.

### Not recommended now

- **A new template recipe.** The current one gained only 0.02 from step 20 to step 100, and relaxing the equal-size
  rule after seeing the data would change a rule set in advance.
- **A new reference pair both on maths.** About a day of agent time and $0 to $15. It fixes only the topic limit, so it
  makes sense only if A and B show something.

## Recommendation

A, then B or C by its result. It is the cheapest option, and it is the only one that tests whether the last live reading
measures anything but size. B and D both assume that reading is worth comparing. The E57 report already said this check
should come first, but it never reached the options. This differs from Fable's recommendation (D) and from the Qwen
lead's (B without A).

## Corrections to the earlier text

- The earlier card gave invented facts' direct share as "0.012 and 0.016". On the maths rows, the comparison the table
  uses, both seeds read 0.012 (`boot_direct.json`). The 0.016 is seed 0 on its own training rows.
- The earlier text called placement (how much of the update lands on subcomponents already on) "the one decisive row
  left with teeth". Its positive control then read 0.21 against an expected 0.42, at its blind line, and the two
  invented-facts seeds read 0.091 and 0.058. So it is not relied on here.
- Option 6's cost left out the readings needed to use the retrained states. B states the full cost.

## Names used in other reports

| here | in the plans and reports |
|---|---|
| this question | D3 of the Qwen plan, within Q3, experiment E57 |
| template fine-tune | anchor 1 |
| invented-facts fine-tune | anchor 2 (`a2long`) |
| fine-tunes on human GSM8K solutions | E45 gold step 100; E40 gold step 10 |
| fine-tunes on the model's own answers | own samples, null |
| the remaining planned run | Stage D |
| A | the E57 report's "dose check" |
| B | option 6, plus anchor 1's reading |
| C | options 1 and 5 |
| D | option 2 |
| not recommended | options 3 (new recipe) and 4 (new pair) |

Sources: `reports/analyses/2026-10-30_analysis_e57-reading-the-update.md`,
`reports/analyses/2026-10-30_analysis_e45-amendment2.md`, `reports/reviews/2026-10-30_review_e57-a-c-check.md`,
`reports/plans/2026-10-30_plan_q3-reading-the-update.md`.

## Record: the text as it stood on 2026-10-30, unchanged

### Decision needed: E57 row 1 has no teeth at scale; what Stage D reads (2026-10-30)

On Qwen3-1.7B, two random weight changes, sized to move the outputs about as much as the real E40 update, both read
a re-gating share of about 0.99. That is above the 0.8 line that, per the plan, shows the reading has no teeth. Recomputing
the masks on the changed model's activations recovers almost all of any change, real or random, so the re-gating
share (E57 row 1, and E45's filed share row) cannot tell re-gating from "the importance function follows the
activations" at this scale. The plan's tree (`reports/plans/2026-10-30_plan_q3-reading-the-update.md` §11) sends this
to the user before Stage D; Stage C (E40, gold, anchor 2) runs meanwhile as descriptive rows.

#### Numbers (Stage C0, job 46350, `experiments/qwen_reading/results/C0.json`)

| control | maths KL vs E40 0.62 | re-gating share (95% boot) | direct share | K0 | K00 | K01 | K10 | K11 |
|---|---|---|---|---|---|---|---|---|
| Gaussian, off-span | 0.53 | 0.994 [0.993, 0.995] | 0.022 | 0.070 | 0.927 | 0.129 | 0.909 | 0.124 |
| random change to alive pieces | 0.50 | 0.993 [0.992, 0.994] | 0.029 | 0.070 | 0.694 | 0.117 | 0.677 | 0.113 |

K-cells: mean per-position KL to the changed model of the original-weights model masked by the original pieces or the split ones (first index) under the original model's or the changed model's masks (second index); K0 is W0's own masking error.

#### Two more problems found in Stage A/B2 (Fable)

- The filed split-half floor is 21,110 switched-on pieces (5 × the 4,222 that differ between two halves of W0's rows).
  It measures how gating differs between different rows, not the noise of the same-row comparison (W0 vs W0 reads 0).
  The prompt row (|S_prompt| 1,860) and both random changes (6,961; 5,170) fall under it, so rows 2-3 come back
  unscored for most states.
- The overlap excess of the random changes is 19-75 standard deviations above the site-stratified draws against every
  reference set, unrelated web text included. Switched-on pieces are the near-threshold ones, which any new context
  switches on, so the uniform-within-site null reads "huge overlap" for anything that moves.

#### Options for Stage D ($0 on the cluster, ~2.3 h; it cannot start before anchor 1 is trained)

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

#### Revised 18:30, after E45 Amendment 2 and Stage C (supersedes the options above; Fable's advice)

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

**Added 20:10: the direct share's row bootstrap** (`experiments/qwen_reading/results/boot_direct.json`, report
`reports/analyses/2026-10-30_analysis_e57-reading-the-update.md`). Gold and the random changes do not share an
interval: gold 0.092 [0.090, 0.093], E40 0.063 [0.062, 0.064], random 0.022 [0.020, 0.024] and 0.029 [0.026, 0.032];
gold minus random +0.06 to +0.07, interval clear of zero. So the "close D3 now" condition is not met. Anchor 2 sits
below both random changes on both seeds (0.012 [0.010, 0.013], 0.012 [0.011, 0.014]), which no prediction covered.
Caveats: no positive control at this scale, gold has one seed, gold moves the maths outputs more (KL 0.70 against
0.50-0.53) and no dose check was run.

**Added 20:25 from the independent check** (`reports/reviews/2026-10-30_review_e57-a-c-check.md`). Measured on one
yardstick (same 128 rows for all states), gold moves the outputs 1.7-2.3 times as much as the random changes, and
among the real updates the direct share rises with size. So "gold sits apart from random" is partly a size effect.
For Stage D, anchor 1 (maths KL 0.33, |δW| 3.14) is smaller than anchor 2 (0.62, 5.1), so option 2's "anchor 1 apart
from anchor 2 → design a matched pair" would read a size gap as a gap in kind. Option 6 (anchor 2's intermediate
states, matched in size to anchor 1) answers that directly, at about 35 min per seed on the cluster, $0.
