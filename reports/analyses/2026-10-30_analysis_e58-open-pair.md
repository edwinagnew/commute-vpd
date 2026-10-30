# Analysis: does a half-known word map learned on 24 keys carry to 8 unseen keys, and does VPD agree? (E58)

*2026-10-30, regating-lead writer. Plan: `reports/plans/2026-10-30_plan_e58-open-pair.md`, with its two dated pre-fine-tune
amendments and its Amendment 1. Rules: `experiments/open_pair/results/predictions.json` and `predictions_amend1.json`.
Results: `experiments/open_pair/results/run.json` (job 46266) and `amend1.json` (job 46332). Reading adopted by the lead:
`reports/plans/2026-10-30_advice_e58-reading-FABLE.md`, `decisions.md` last row. Explanation:
`notebooks/E58_open_pair.ipynb`. Words as in `GOALS.md` → Terms. Provenance at the end.*

We fine-tuned the small model on a word map it half-knows, on most of its keys; the unseen keys did not gain, yet the
decomposition read the gain as old subcomponents switched on, so a filed prediction that the two agree failed. A
follow-up gave this held-out test its missing positive control, and a second filed prediction failed: a fine-tune that
copies the key carried to unseen keys but passed the test's bar on one seed of three, not two. So the test has no
working positive control on this model, and for `GOALS.md` its verdict cannot yet count for or against the
decomposition.

## The answer

No carry-over on the half-known map, and the decomposition reads the opposite; but the measure itself then failed its
positive control. At the unseen keys the half-known map's update pushed the 24 trained answers up and the model's own
other preferences down: a lookup, not a sharpening. Two maps the model can already apply at any key did carry further,
and the decomposition read them as old subcomponents too. Yet the copy map crossed the held-out test's bar on only one
seed of three, so a filed prediction failed and the test is not used against the decomposition.

## What we were trying to test

The re-gating line asks whether a VPD decomposition can tell a fine-tune that strengthens something the model already
does from one that teaches something new. Its numbers have only ever been scored against labels set by how a task was
built. E58 earns a label from behaviour alone, with no decomposition, on a pair whose answer the construction does not
fix.

**The task**, from E54. Each window is Pile text, then a rare marker token, then a key word; the model must predict the
key's value next. Two maps share the 32 keys and the windows:

- **The elicit map.** Each key's value is a word the original model W₀ already ranks 2nd to 20th after the marker and
  that key. Two examples: " action" → " potential" and " formed" → " from". The model half-knows it.
- **The cipher.** The same keys mapped to arbitrary words W₀ gives under 1e-4 each. The model cannot know these, so
  nothing learned on some keys can tell it the others. It is the label's negative control.

**What changed.** E54's fine-tune recipe, unchanged, ran on only 24 of the 32 keys; 8 keys were held out by a seeded
draw and never sat at a training window's key position. Three seeds per map. The elicit map is read at E12's scored
step, the first checkpoint where the trained keys' held-out windows give the target probability 0.5 or more: step 10
on every seed. The cipher is read at the step whose total update size matches the same seed's elicit state, steps 8 to
9. Every fine-tune started from W₀.

**The label.** For one window, the *specific gain* is the fine-tune's rise in the log-odds of that window's own value,
minus the mean rise of the other 31 values of the same map. Subtracting the other values removes a shift shared by the
whole value set, which the stopped job 46259 found on E54's cipher, described under "Stage 0". Then:

- $G_\text{spec}$, the mean specific gain on the 75 windows of the trained keys that training never used;
- $H_\text{spec}$, the same on the 124 windows of the 8 held-out keys;
- $x = H_\text{spec} / G_\text{spec}$: 0 means the unseen keys gained nothing, 1 means they gained as much as the trained
  keys;
- $s$, the standard deviation of $x$ over 2000 bootstrap resamples of windows.

A state is labelled "reuses" if $x \ge 0.5$, "new" if $x \le 3s$, "neither" in between. The bar is the reuse line's
Stage B rule, read on behaviour; see "Departures from the literature".

**The readings**, E54's, unchanged. Each fine-tuned weight change gets the least-squares split: the least-norm change
to the alive subcomponents, the rest into Δ. All cells are the target's log-odds at the answer position, on the
trained keys' held-out windows.

- $r = (L_{01} - L_{00}) / (T_1 - T_0)$, the share of the fine-tune's gain $T_1 - T_0$ recovered in the CI-masked model
  when the subcomponents stay at W₀'s and only Γ's CI values move to those Γ computes on the fine-tuned model. It reads
  1 if switching existing subcomponents on carries all the gain, 0 if none.
- The stand-in ratio, $(L_{11} - L_{00}) / (U_1 - U_0)$: the CI-masked model's gain with the split subcomponents and the
  new CI values, over the gain of the split subcomponents with every mask open. As E46 Amendment 3 found, it is roughly
  the gain from Γ's new CI values alone over the subcomponent gain. It is not "the share Γ keeps", and it has no ceiling
  at 1.
- $S$, steering's share: one mean added vector at one block, E52's recipe, in log P. It never reads the decomposition.

Each reading turns into a label by the same cut: 0.5 or more reads "reuses".

## What we expected, and why

- **The elicit label was open.** Fable leaned toward "reuses" at about even odds: if the fine-tune worked by sharpening
  what W₀ already prefers after a key, unseen keys would gain too.
- **Every cipher reads "new"**, with high confidence, because its unseen values are unknowable.
- **r and the stand-in ratio agree with the elicit label.** We expected both readings high on every elicit state, as in
  E54, so this prediction would pass exactly when the label read "reuses": also about even odds.
- **The decomposition agrees with the label more often than steering does**, following whichever label came.
- **r and the stand-in ratio still separate the elicit map from the cipher**, as E54 found with every key trained.

## Stage 0: the controls, run first and alone

| check | filed | read | |
|---|---|---|---|
| zero update, largest gain on any window, four window sets | exactly 0 | 0.0 on all four | pass |
| E54 elicit seed 0 on its 32 keys: r | 0.7726 ± 0.03 | 0.7726 | pass |
| the same: stand-in ratio | 0.9564 ± 0.03 | 0.9564 | pass |
| the same: S, and its block-0 value | 0.3796 ± 0.001; ≤ 1e-6 | 0.3796; −9.4e-8 | pass |
| E54 cipher seed 0: specific gain over raw gain | ≥ 0.5 | 6.944 / 11.107 = 0.625 | pass |
| E54 elicit seed 0: specific gain over raw gain | reported | 16.243 / 10.787 = 1.506 | reported |
| $G_\text{spec}$ gate, 8 ÷ 10.79 × the smaller of the two specific gains | finite, ≤ 5.148 | 5.148 | pass |

**The stop before this run.** Job 46259 ran the first amendment's stage 0 and stopped there, as designed, before any
fine-tune. On E54's cipher, the other 31 cipher values gained 4.16 log-odds per window on average, against the target's
11.11. A raw held-out ratio would therefore read a cipher with no carry-over as "neither", never "new". The second
amendment, filed before any fine-tune, moved the label to the specific gain and lowered the gate to 5.148. Every other
stage-0 number of job 46259 matched the table above.

## Did the fine-tune carry over to the keys it never saw?

![x per state, elicit and cipher, with the 0.5 cut, each state's 3 s mark for the scored x, and the raw x, which has no bootstrap of its own](../../experiments/open_pair/figures/label.png)

**Reading it.** No. On all three elicit seeds the held-out keys' specific gain was slightly negative, so $x$ sits just
below 0, under each state's $3s$ mark of about 0.02, and the label is "new". The trained keys gained 18.2 to 18.5
log-odds on the same measure. The cipher reads "new" too, further below 0, so the label's negative control passed.
The hollow points show the raw ratio with the shared shift left in: 0.04 to 0.08 for the elicit map, far from the 0.5 cut.
That sits above the value-specific $3s$ marks, but the raw ratio's own bootstrap was not computed, so it has no label.

| state | step | update size | $G$ raw | $G_\text{spec}$ | $H$ raw | $H_\text{spec}$ | $x$ | $3s$ | $3s_\text{key}$ | raw $x$ | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elicit, seed 0 | 10 | 3.169 | 11.03 | 18.18 | 0.61 | −0.50 | −0.027 | 0.024 | 0.045 | 0.056 | new |
| elicit, seed 1 | 10 | 3.120 | 11.35 | 18.43 | 0.46 | −0.70 | −0.038 | 0.021 | 0.030 | 0.041 | new |
| elicit, seed 2 | 10 | 3.183 | 11.31 | 18.51 | 0.90 | −0.48 | −0.026 | 0.021 | 0.041 | 0.079 | new |
| cipher, seed 0 | 9 | 3.260 | 12.86 | 10.67 | 1.38 | −3.24 | −0.303 | 0.085 | 0.232 | 0.107 | new |
| cipher, seed 1 | 9 | 3.194 | 12.53 | 9.39 | 1.19 | −3.54 | −0.377 | 0.084 | 0.210 | 0.095 | new |
| cipher, seed 2 | 8 | 3.073 | 12.65 | 9.94 | 1.26 | −3.40 | −0.342 | 0.082 | 0.198 | 0.099 | new |

Gains are log-odds of the target, fine-tuned minus W₀. Update size is the sum over the 24 decomposed matrices of the
Frobenius norm of δW. $s_\text{key}$ resamples the 8 held-out keys instead of windows; it was reported, not scored.
No label changes under $3 s_\text{key}$. Every elicit state cleared the gate of 5.148 by more than three times.

Per held-out key, the elicit map's $H_\text{spec}$ is negative on 21 of 24 key-seed pairs. The three positive ones are
key 2515 on seeds 0 and 1, at +0.77 and +0.21, and key 2250 on seed 2, at +0.92. Full per-key values are in
`run.json` under `H_spec_per_key`.

## Was carry-over possible, or did the task rule it out?

![At the held-out keys' windows: the gain of the own value, of the other held-out values, and of the trained values](../../experiments/open_pair/figures/heldout_gain.png)

**Reading it.** Carry-over was possible, and the fine-tune took that route only weakly. There is no rule from key to
value here, so the one way unseen keys could gain is a general sharpening of what W₀ already prefers after each key.
That would raise each held-out key's own value, one of W₀'s top 20 there, more than the values of other keys. The own
value rose by 0.5 to 0.9 log-odds. The 24 trained values rose by 1.6 to 1.8, more than twice as much. The other
held-out values fell, by about 0.5 on average.

The trained values' rise is inferred, not read directly: it follows from $H$, $H_\text{spec}$ and the other held-out
values' gain, since the 32 values are distinct. It is what hides a small key-specific boost from the label. Against its
7 held-out siblings alone, the own value gained 1.10 to 1.15 more on the elicit map. On the cipher the same contrast is
−0.10 to −0.21. Both come from two read means each, H and the siblings' gain. That boost is about 6% of the trained keys'
specific gain. So at an unseen key the update is mostly 24 lookups plus a push toward the trained values, with a small
key-specific boost toward what W₀ prefers there. The cipher shows the push toward trained values much larger, at 5.5 to
5.7. Amendment 1, below, reads the trained values' push directly. The label "new" stands as filed.

| state | own value, read | other 7 held-out values, read | own minus siblings | 24 trained values, inferred |
|---|---|---|---|---|
| elicit, seed 0 | +0.61 | −0.53 | +1.145 | +1.59 |
| elicit, seed 1 | +0.46 | −0.64 | +1.098 | +1.68 |
| elicit, seed 2 | +0.90 | −0.23 | +1.127 | +1.84 |
| cipher, seed 0 | +1.38 | +1.58 | −0.208 | +5.50 |
| cipher, seed 1 | +1.19 | +1.29 | −0.105 | +5.73 |
| cipher, seed 2 | +1.26 | +1.35 | −0.095 | +5.62 |

The inference: $H - H_\text{spec}$ is the mean gain of the other 31 values, so the 24 trained values' mean is
$(31 (H - H_\text{spec}) - 7 H_\text{other}) / 24$, computed in `experiments/open_pair/plot.py`.

## Do the decomposition's readings agree with the label?

![r, the stand-in ratio and S against the label x for all six states, with E54's values as references](../../experiments/open_pair/figures/agreement.png)

**Reading it.** No: both r and the stand-in ratio read "reuses" on every elicit state, while the label reads "new". The
three green points sit in the upper-left box of the first two panels, where behaviour says new and the instrument says
reuses. They sit where E54 put the same map with all 32 keys trained, the dashed lines. Both readings still separate
the elicit map from the cipher: r by a gap of 0.54 over a seed range of 0.03, and the stand-in ratio by 0.62 over 0.03.
So the filed agreement prediction fails for both readings, and the separation prediction passes.

| state | r | stand-in ratio | S | S block | label | r says | ratio says | S says |
|---|---|---|---|---|---|---|---|---|
| elicit, seed 0 | 0.787 | 0.946 | 0.501 | 2 | new | reuses | reuses | reuses |
| elicit, seed 1 | 0.757 | 0.934 | 0.498 | 2 | new | reuses | reuses | new |
| elicit, seed 2 | 0.758 | 0.934 | 0.490 | 2 | new | reuses | reuses | new |
| cipher, seed 0 | 0.236 | 0.330 | 0.171 | 3 | new | new | new | new |
| cipher, seed 1 | 0.238 | 0.336 | 0.220 | 3 | new | new | new | new |
| cipher, seed 2 | 0.217 | 0.302 | 0.152 | 3 | new | new | new | new |
| E54 elicit, 32 keys, seeds 0, 1 | 0.773, 0.758 | 0.956, 0.951 | 0.380, 0.394 | | none | | | |
| E54 cipher, 32 keys, seeds 0, 1 | 0.241, 0.212 | 0.336, 0.289 | 0.228, 0.185 | | none | | | |

The stop on steering's block 0 is in nats: block 0's share times the scoring-half gain must stay under 1e-5. It read
4.0e-7 nats on the elicit states and 2.3e-7 on the cipher, at the largest.

**Steering's score is not a finding.** Scored as filed, steering agrees with the label on 5 of 6 states and each
decomposition reading on 3, so the comparison reads "steering better" for both. But steering's three elicit values are
0.501, 0.498 and 0.490, on its 0.5 cut, so it scores by sitting on the line, not by reading the label. This result is
kept out of the headline.

## The reading the lead adopted

This is a reading, not a proof. No check run here separates it from "the decomposition is wrong".

**The granularity limit, first concrete instance.** VPD's unit is the rank-one subcomponent. A new circuit built from
old subcomponents, switched on in a new pattern, reads "elicited" to r and the stand-in ratio. Here the elicit values
are words W₀ already says after these keys, so the subcomponents that write them already exist; the cipher's values are
not, and its readings are low. The held-out label asks a different question: does a rule carry to inputs never seen?
On a lookup over existing output words, the first answer can be yes and the second no, and E58 shows them disagreeing on
three seeds each way. Every claim here holds only at the level of VPD's subcomponents.

**The label is not empty.** One could worry that "new" was guaranteed by the task, since each value is a separate pick
per key. But the route to carry-over, sharpening W₀'s own preference at each key, was open. The update took it only
weakly, a boost of about 1.1 log-odds over the sibling values, and pushed harder toward the trained values. The label
says something true: the fine-tune learnt mostly 24 lookups, not a rule that carries.

**What is missing.** No 67M state had read "reuses" under this bar. Until the label's positive control does, E58
cannot be counted as a VPD failure either. Amendment 1, below, ran that control, and it failed at the filed bar.

**The question for the user, ask #14.** Does "reuses" in this line mean A, built from subcomponents the model already
has under new switching, or B, generalises to inputs the fine-tune never saw? Under A, r and the ratio did their job
here and E58's filed agreement test was unfair on a lookup task. Under B, E58 counts against VPD. The lead proceeds with
A unless the user objects.

## Amendment 1: what the update did at unseen keys, and can the label read "reuses"?

*Job 46332, after E58. Plan § "Amendment 1"; Fable's advice `reports/plans/2026-10-30_advice_e58-reading-FABLE.md`, "Two
$0 checks", adopted by the lead; rules `experiments/open_pair/results/predictions_amend1.json`, filed before the run;
results `experiments/open_pair/results/amend1.json`; code `experiments/open_pair/amend1.py`. E58's predictions and
`run.json` are unchanged.*

### What we were trying to test

Two questions about the label above. First, at the 8 unseen keys, what did E58's fine-tunes push up: the 24 trained
answers, or the words W₀ itself already preferred after each unseen key? A lookup pushes the first, and a general
sharpening of W₀ pushes the second. Second, can the label read "reuses" at all? The cipher showed it can read "new", but
no state had read "reuses". Two new maps give it the chance. Both use E58's windows, marker, keys and 24/8 key split,
E54's fine-tune recipe read at step 10, and three seeds:

- **The rank-1 map.** Each key's value is W₀'s own most probable next token after the marker and that key. The rule "say
  what you would have said anyway" holds at every key, trained or not, so a fine-tune that learns it should carry.
- **The copy map.** Each key's value is the key itself. The rule is the same at every key, and the small model ties its
  input and output embeddings, so a direct path can carry it.

Each control state gets E58's label unchanged: the value-specific gain, the same two bars, the same gate of 5.148. Each
also gets r, the stand-in ratio and S on its trained keys, as E58's states did.

**The readout** reads E58's six saved states at the held-out keys' 124 windows. Its numbers, all log-odds gains at the
answer position, fine-tuned minus W₀, all exactly 0 at W₀:

- *own*: the mean gain of the window's own value. On E58's states it is E58's raw $H$.
- $T$: per window, the mean gain of the trained keys' values other than the own value, averaged over windows.
- $R_\text{peer}$: per window, the mean gain of W₀'s top 20 tokens at that key, leaving out the own value and any trained
  value, averaged over windows.
- $d$: the mean over windows of $T - R_\text{peer}$, with $s_d$ and $s_R$ its and $R_\text{peer}$'s row-bootstrap
  standard deviations.

The verdict is "lookup" if $d > 0$ and $d \ge 3 s_d$. It is "sharpening" if $R_\text{peer} > 0$ and
$R_\text{peer} \ge 3 s_R$, "both" if both hold, and else "neither". $T$ is not new on E58's states: it follows exactly from E58's $H$, $H_\text{spec}$ and
the other held-out values' gain, as the table under "Was carry-over possible" inferred. That identity is the readout's
control. The new number is $R_\text{peer}$.

### What we expected, and why

- **E58's elicit update is a lookup**, about 85%. $T$ was fixed by arithmetic at 1.6 to 1.8, so only $R_\text{peer}$ was
  open, and E58 had already shown the own value gaining less than the trained values.
- **The rank-1 map is testable** at about 55%, and if testable reads "reuses" on two of three seeds at about 60%. The
  guard: at least 3 held-out keys must have a value no trained key has, or a carry could ride on the push toward trained
  values instead of on the rule.
- **The copy map reads "reuses" on two of three seeds**, about 75%. It was filed as always testable.
- **r and the stand-in ratio agree with any earned "reuses"**, about 75%.

### Controls, run first

The zero update read exactly 0 on every readout number and on all four control maps' window sets. Each of E58's six
states reproduced E58's $H$ and the trained-value push implied by E58's numbers within 1e-7, against a tolerance of 1e-4,
before the next state was read. No steering block-0 stop fired. Every control fine-tune learned by step 10, with the
trained keys' held-out windows at probability 0.93 to 0.98, and an update size of 3.10 to 3.18, beside E58's 3.12 to 3.18.

### At the unseen keys, did E58's update push the trained answers or W₀'s own preferences?

![At the held-out keys' windows, per map: the own value's gain, the trained values' push, and the gain of W₀'s other top-20 tokens at the key](../../experiments/open_pair/figures/a1_readout.png)

**Reading it.** The trained answers: on every elicit seed the update is a lookup, as filed. At the unseen keys the 24
trained values rose by 1.6 to 1.8 log-odds, while W₀'s other top-20 words at those keys fell by 0.4 to 0.6. The gap is
2.1 to 2.2, against three bootstrap deviations of about 0.23. The word W₀ ranks first at each unseen key also fell, by 0.3
to 0.6, so no sharpening hides in rank 1. The cipher reads the same, with a gap of 6.8 to 7.6. One caveat matters for the
next section: the verdict compares the trained values with W₀'s other preferences and leaves the own value out. So the
copy map, whose own value rose by 8 to 10 at unseen keys, also reads "lookup". The verdict says what else the update
pushed, not whether it carried.

| state | own | $T$ | $R_\text{peer}$ | $d$ | $3 s_d$ | W₀'s rank of own value, median | verdict |
|---|---|---|---|---|---|---|---|
| elicit, seed 0 | +0.61 | +1.59 | −0.50 | 2.09 | 0.23 | 9 | lookup |
| elicit, seed 1 | +0.46 | +1.68 | −0.56 | 2.24 | 0.24 | 9 | lookup |
| elicit, seed 2 | +0.90 | +1.84 | −0.38 | 2.22 | 0.23 | 9 | lookup |
| cipher, seed 0 | +1.38 | +5.50 | −1.28 | 6.78 | 0.35 | 5027 | lookup |
| cipher, seed 1 | +1.19 | +5.73 | −1.91 | 7.64 | 0.38 | 5027 | lookup |
| cipher, seed 2 | +1.26 | +5.62 | −1.81 | 7.43 | 0.38 | 5027 | lookup |
| rank-1, seeds 0-2 | +2.47 to +2.69 | +0.45 to +1.09 | −0.94 to −0.64 | 1.39 to 1.77 | 0.20-0.22 | 1 | lookup ×3 |
| copy, seeds 0-2 | +8.20 to +9.86 | +0.54 to +1.21 | −1.19 to −0.45 | 1.65 to 2.05 | 0.21-0.25 | 296 | lookup ×3 |

The peer set holds on average 16.5 of W₀'s top 20 on the elicit windows and 13.3 on the rank-1 map's, where more trained
values sit in the top 20. It holds all 20 on the cipher's and the copy map's. Every number reads 0 for the zero update.
The gain per W₀ rank is in `amend1.json` under `rank_gain`. On the elicit map rank 2 rose by 1.3 to 1.6 on every seed,
and rank 13 by 0.6 to 0.8. Most other ranks fell.

### Can the label read "reuses"?

![Held-out carry x of all four maps against r, the stand-in ratio and S, with the label's cuts and the readings' 0.5 cut](../../experiments/open_pair/figures/a1_carry.png)

**Reading it.** Not at its filed bar. Read the horizontal axis. The copy map carried most: its unseen keys gained 0.42
to 0.55 of the trained keys' value-specific gain. The label reads "reuses" on seed 1 only and "neither" on seeds 0 and
2. The rank-1 map carried 0.09 to 0.14, "neither" on all three seeds. So the copy map's prediction, filed at 75%, fails as
filed. The rank-1 map's could not be scored: only 1 of its 8 held-out keys has a value no trained key has, against the
3 required. The combined verdict is "none", and by the plan's decision tree the label is not used against r or the
ratio.

| state | $G_\text{spec}$ | $H_\text{spec}$ | $x$ | $3s$ | $3 s_\text{key}$ | raw $x$ | $x_\text{novel}$ | label |
|---|---|---|---|---|---|---|---|---|
| rank-1, seed 0 | 16.82 | 2.33 | 0.138 | 0.041 | 0.113 | 0.246 | 0.271 | neither |
| rank-1, seed 1 | 16.57 | 1.60 | 0.097 | 0.045 | 0.125 | 0.227 | 0.208 | neither |
| rank-1, seed 2 | 16.63 | 1.50 | 0.090 | 0.042 | 0.118 | 0.229 | 0.210 | neither |
| copy, seed 0 | 17.36 | 7.29 | 0.420 | 0.060 | 0.125 | 0.633 | 0.419 | neither |
| copy, seed 1 | 17.38 | 9.60 | 0.553 | 0.056 | 0.092 | 0.754 | 0.552 | reuses |
| copy, seed 2 | 18.18 | 8.54 | 0.470 | 0.053 | 0.102 | 0.698 | 0.469 | neither |

**Why the copy map stops short of 0.5.** In raw log-odds its unseen keys gained 8.2 to 9.9, against about 13 at the
trained keys, a raw ratio of 0.63 to 0.75. The value-specific gain subtracts the mean gain of the map's other 31 values.
At the trained keys' windows those fell by 4.3 to 4.9 log-odds, while at the unseen keys' windows they rose by 0.3 to 0.9.
So the trained-key gain grows from about 13 to 17 or 18, and $x$ falls below the raw ratio. We do not assert why the other
values fall at trained keys only. The label was moved to the value-specific gain before E58's fine-tunes, so that a shift
of the whole value set could not read as carry-over, and it stands as filed. The raw ratio has no bootstrap and no label.

**The rank-1 map is a weak control on this split.** Its 32 values hold only 18 distinct tokens. One token is the value of
9 keys, 5 of them held out. Since that token is also a trained value, most of the map's carry can ride on the push toward
trained values. Read on the one held-out key with a value of its own, $x_\text{novel}$ is 0.21 to 0.27. A code reviewer also
noted that the rank-1 values were picked from W₀'s next-token distribution over E54's training windows, which include the
held-out keys' scoring windows. That choice is in-sample for W₀ only. Picking W₀'s top token on the same windows starts it
higher there and leaves less room to gain, which biases $x$ slightly toward "new"; see "Departures from the literature".
Under three key-level deviations, two rank-1 seeds would read "new"; no copy label changes.

**What the stand-in ratio above 1 means.** It reads 1.00 to 1.11 on both control maps. It is not a share. It is close to
$g / U$, where $g$ is the gain from Γ's new CI values on W₀'s own subcomponents and $U$ the gain of the split
subcomponents with every mask open. On rank-1 seed 0, $g$ is 9.24 log-odds and $U$ is 8.58, with the whole fine-tune at
10.94. So $g / U$ is 1.08 and the ratio 1.11. A value over 1 says that changing which subcomponents are on, alone, gains
more than changing the subcomponents does. Nothing bounds $g$ by $U$, so the ratio has no ceiling at 1.

### Do r, the stand-in ratio and S follow the carry?

**Reading it**, same figure, vertical axes. No. Held-out carry orders the four maps copy, then rank-1, then elicit near
0, then cipher below 0. The re-gating share r reads 0.83 to 0.85 on both control maps and 0.76 to 0.79 on the elicit map.
It is nearly level whether unseen keys gained half the trained gain or nothing. The stand-in ratio is flat the same way
across those three maps. Both separate only the cipher, which sits far below. Steering does not follow the carry either.
It ranks the copy map highest, then the elicit map, and puts the rank-1 map beside the cipher. On the one earned
"reuses", copy seed 1, r reads 0.84 and the ratio 1.00, so both agree with it, as filed. That one state does not make the
label a working positive control.

| map | held-out carry $x$ | r | stand-in ratio | S |
|---|---|---|---|---|
| copy, seeds 0-2 | 0.42 to 0.55 | 0.83 to 0.85 | 1.00 to 1.06 | 0.61 to 0.68 |
| rank-1, seeds 0-2 | 0.09 to 0.14 | 0.84 | 1.10 to 1.11 | 0.21 to 0.25 |
| elicit, seeds 0-2 | −0.04 to −0.03 | 0.76 to 0.79 | 0.93 to 0.95 | 0.49 to 0.50 |
| cipher, seeds 0-2 | −0.38 to −0.30 | 0.22 to 0.24 | 0.30 to 0.34 | 0.15 to 0.22 |

The control readings sat mostly where the plan expected: copy r inside 0.7 to 0.85 and S inside 0.6 to 0.8; rank-1 r inside
0.6 to 0.9. Both maps' ratios sat above the filed 0.85 to 1.0 and 0.6 to 1.0. Both maps' $G_\text{spec}$ of 16.6 to 18.2 sat
far above the filed 4 to 10 and 8 to 13.

**The granularity limit.** Every claim here holds at the level of VPD's rank-one subcomponents. A new circuit built from
old subcomponents reads "elicited" to r and the stand-in ratio. The limit bites only where behaviour reads "new", as on
the elicit map; a control that reads "reuses" does not test it.

### What Amendment 1 changes

E58's "new" on the elicit map is a fair description of the update: 24 lookups and a push toward the trained values, with
W₀'s own preferences at unseen keys pushed down, not sharpened. But the label failed its positive control. A rule the
model applies at any key carried about half of its trained-key gain at best by the value-specific measure. So the 0.5 bar
is at or beyond reach of E54's recipe at step 10 on the 67M, and the label cannot yet tell a fine-tune that carries from
one that does not. E58's failed agreement prediction therefore counts neither for nor against the decomposition.
Among the three maps that are not the cipher, r and the stand-in ratio stay level while the carry ranges from none to
half. Whatever they read, it is not held-out carry. The lead is consulting Fable on what follows.

## What we learned

For `GOALS.md`'s question, new capabilities or strengthening, the decomposition answers a narrower question than
behaviour does. On this map, r and the stand-in ratio report that the gain runs through subcomponents W₀ already had,
three seeds each, with the same values as when every key was trained. Behaviour reports that no rule generalises to
unseen keys. Both can be true at once on a lookup over words the model already says. Amendment 1 confirms the lookup.
It also removes the label's standing: its positive control failed, so no held-out behavioural label is yet earned on the
67M, and r and the stand-in ratio have nothing behavioural to be scored against. They do stay level across three
maps whose held-out carry runs from none to half, so they do not measure carry. Ask #14 stays open.

## How compelling is the evidence

- **Seeds and keys.** Three fine-tune seeds per map, one key split, one decomposition. The label's real unit is the 8
  held-out keys; resampling keys widens the elicit $3s$ to 0.03 to 0.045, and every label stays "new".
- **The label is blind** to a carry-over that raises every value of the map alike: that is the price of removing the shared
  shift. The raw ratio, which keeps it, reads 0.04 to 0.08, also far from "reuses".
- **The label's positive control failed.** The cipher shows the bar can read "new". The copy map, a rule the model applies
  at any key, read "reuses" on one seed of three, and the rank-1 map was not testable on this split. The readout's
  "lookup" verdict is blind to the own value's gain, so it reads the copy map "lookup" too; it describes what else an
  update pushed, not whether it carried.
- **One key split** for the controls as for E58: it left the rank-1 map with one held-out key whose value is its own.
- **r and the stand-in ratio** reproduced E54 to four decimals in stage 0, and the elicit values here sit within 0.03 of
  E54's. They cannot tell a new circuit of old subcomponents from an old circuit.
- **The trained values' rise** at unseen keys is inferred by arithmetic from three read means, not read value by value.

## Surprises

- **Amendment 1: a filed prediction failed against a confident lean.** We predicted, at about 75%, that the copy map would
  read "reuses" on two of three seeds. It did on one; the other two carried 0.42 and 0.47 of the trained gain and read
  "neither". The prediction was filed as always testable, so this is a fail, not a "not testable". The arithmetic is
  above: at trained keys the map's other values fall by over 4 log-odds, which enlarges the denominator. Why they fall
  there and not at unseen keys, we do not know.
- The rank-1 map had one held-out key with a value of its own, at the bottom of the plan's expected 1 to 6. W₀ gave its
  values a geometric-mean probability of 0.006 on the trained keys' held-out windows, under the expected 0.03 to 0.3.
- The stand-in ratio read above 1 on both control maps, up to 1.11, above the filed ranges.
- **A filed prediction failed.** The decomposition's labels disagreed with the behavioural label on all three elicit
  states, for both r and the stand-in ratio. It was filed at about even odds, so the direction is not a reversal of a
  confident lean, but it fails as filed and stands.
- The open lean toward "reuses" did not hold: all three elicit states read "new", far from the cut.
- Steering's share on the elicit map rose to about 0.5 with 24 keys trained, from about 0.39 with all 32. We have no
  explanation and assert none.
- The cipher's update raises its 24 trained values at unseen keys by about 5.6 log-odds, three times the elicit map's push.
- On the elicit map the specific gain on trained keys exceeds the raw gain, about 18 against 11. So at a trained key the
  map's other values fall by about 7 log-odds.

## What next

The plan's decision tree, verdict "none": the held-out label is not used against r or the stand-in ratio, and the user
is told. The lead is consulting Fable on what follows; no next step is set here. Ask #14 to the user stays open.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| P0 | the elicit label, open (a lean, not a test) | majority of three seeds | new, new, new | new; lean "reuses about 50%" missed |
| P1 | every cipher reads "new" | all three "new"; stop on any "reuses" | new, new, new | pass |
| P2, r | r's label equals the behavioural label on every labelled elicit state | r ≥ 0.5 → reuses | reuses ×3 vs new ×3 | **fail** |
| P2, ratio | the same for the stand-in ratio | ratio ≥ 0.5 → reuses | reuses ×3 vs new ×3 | **fail** |
| P3 | the decomposition agrees with the label more often than steering | lead of 2 or more agreements over six states | r 3, ratio 3, S 5 | "steering_better" for both, as scored; not a finding, S sits on its 0.5 cut |
| P4, r | r separates elicit from cipher, elicit higher, as E54 | gap > larger seed range and ≥ 0.2 | gap 0.537, range 0.029 | pass |
| P4, ratio | the same for the stand-in ratio | as above | gap 0.615, range 0.034 | pass |
| A1 | at unseen keys every elicit update is a lookup | $d > 0$ and $d \ge 3 s_d$ on each state; fail on any sharpening, both or neither | $d$ 2.09, 2.24, 2.22; $3 s_d$ 0.23-0.24 | pass |
| C1 | the rank-1 map reads "reuses" | not testable if under 3 novel held-out keys or seed 0 too small; else 2 of 3 seeds | 1 novel key; neither ×3, $x$ 0.09-0.14, $x_\text{novel}$ 0.21-0.27 | not testable |
| C2 | the copy map reads "reuses" | 2 of 3 seeds; always testable | neither, reuses, neither; $x$ 0.42, 0.55, 0.47 | **fail** (lean pass ~75%) |
| verdict | the label has a positive control | rank1 / copy_fallback / copy_only / none | C1 not testable, C2 fail | none |
| C3, r | r agrees with each earned label on the control runs | scored on the copy map, C1 not testable | copy seed 1: reuses vs reuses, n = 1 | pass |
| C3, ratio | the same for the stand-in ratio | as above | copy seed 1: reuses vs reuses, n = 1 | pass |

Stage-0 checks all passed. Early stops: elicit seed 0's $G_\text{spec}$ 18.18 against the gate of 5.148, no stop; no
cipher read "reuses", no stop. Plan decision tree: branch 5, label "new" with the agreement test failed.
Branch 5's filed next steps, the novelty column and a costed second-decomposition-seed card, were replaced by Fable's
E58 Amendment 1, the rank-k readout and the label's positive control, which the lead adopted
(`reports/plans/2026-10-30_advice_e58-reading-FABLE.md`, `decisions.md` row of 2026-10-30). The novelty column waits
until a labelled "reuses" state exists to compare against. The seed card is deferred because a second decomposition can
only re-read the disagreement: without a state the label calls "reuses", it cannot tell a VPD error from the
granularity limit, so the $0 positive control comes first.

Amendment 1: every stop passed. Decision tree: branch 2, A1 pass, so the lookup reading stands with $T$ and
$R_\text{peer}$ beside $H$; branch 5, verdict "none", so the label is not used against r or the ratio and the user is
told. C3's pass rests on one state and does not change branch 5.

## Departures from the literature

- **The Stage B rule, adapted to behaviour only.** The bar is Jain et al.'s Definition 2 with the user's bounded g, as the
  reuse line wrote it in `reports/plans/2026-10-29_plan_reuse-stage1.md`, § "Stage B", patch row. There it reads
  "through iff P ≥ 0.5, not iff P ≤ 3 s, else neither", with P the share of the gain an interchange patch carries. Here
  P is read on behaviour as $x = H_\text{spec} / G_\text{spec}$, and the patch half is not run, on Fable's advice. So the
  label speaks to generalisation only, not to which mechanism carries it.
- **A value-specific gain.** Stage B reads a raw gain. Here each window's gain has the mean gain of the map's other
  values subtracted. The stopped job 46259 showed why: the cipher raises its whole value set. The change was filed
  before any fine-tune. It is blind to a carry-over that raises every value alike.
- **A window bootstrap, while windows share keys.** Stage B's row bootstrap resamples independent texts. Here windows of
  one key are not independent; the key-level $s_\text{key}$ is reported beside, unscored, and changes no label.
- **The 0.5 cut on the stand-in ratio** copies E12's rule for r; the ratio has no ceiling at 1, so the cut is a
  convention. Steering's 0.5 cut is E52's, not fitted here.
- **The gate** was lowered from 8 to 5.148 by the second pre-fine-tune amendment, keeping Fable's fraction of E54's gain;
  every elicit state cleared it by more than three times.
- **Amendment 1, the rank-1 map's values are picked in-sample.** Fable's spec takes W₀'s top token after the marker and
  key. Here it is the top of W₀'s mean next-token distribution over E54's training windows of that key. Those windows
  include the held-out keys' scoring windows. Picking on windows apart from the scoring ones was the clean choice. The
  overlap starts W₀ higher at the scoring windows and biases $x$ slightly toward "new". The code reviewer found it after
  the run.
- **Amendment 1, ranks over the full vocabulary.** Fable's spec reads "W₀'s rank 1-20 word-like tokens"; here ranks
  are over every token, as E54's value ranks were.
- **Amendment 1, the gate.** Fable advised setting the control's gate from W₀'s rank-1 probability before filing; that
  cannot be read off the GPU, so the gate is E58's 5.148. Every control state cleared it by more than three times.
- **Amendment 1, the copy map always ran**, not only as a fallback, and `min_novel` = 3 was added to Fable's guard on
  the rank-1 map.

## Not run

- A rank-1 map on a key split with 3 or more held-out keys whose value is their own, or with values picked on windows
  apart from the scoring windows.
- The control maps at later steps than E12's scored step 10, or under another recipe.
- A label bar set from the controls; the filed bar stays.
- The novelty column on Γ's CI patterns: no expected value until a labelled "reuses" state exists to compare against.
- The interchange-patch half of Stage B's rule.
- More key splits, more windows per held-out key, more than three seeds, a second decomposition seed.
- The cipher at its own E12 step; Qwen.

## Provenance

- Job 46266, MATS cluster `debug` QoS, one L40, account c11, $0; the script ran from 15:19:49 to 15:22:32 on
  2026-10-30. Code at `b11a921`, `run.py` and `src/commute/open_pair.py` clean. Results commit `16f43d6`.
- Stopped job 46259: stage 0 only, results commit `080b1f2`.
- `experiments/open_pair/results/run.json`; predictions sha256 `dbf47c98…`, recorded in `run.json`.
- Saved weight changes, six of 113 MB, with sha256 in `run.json`, and the log `run.out`:
  `~/commute-data/results-large/open_pair/`, linked as `experiments/open_pair/results/large`.
- Figures: `experiments/open_pair/plot.py`, from `run.json` and E54's values in
  `experiments/regating_truth_table/results/table.json`; Amendment 1's `a1_readout.png` and `a1_carry.png` from
  `amend1.json` and `run.json`.
- Amendment 1: job 46332, MATS cluster `debug` QoS, one L40, $0; readout from 15:50:52, last reading 15:53:25 on
  2026-10-30. Code at `0714b79`, `git status` of the code clean. Results commit `c1b4ef9`,
  `experiments/open_pair/results/amend1.json`. Six control states of 113 MB and their windows in
  `~/commute-data/results-large/open_pair/amend1/`, log `amend1.out` beside it.
