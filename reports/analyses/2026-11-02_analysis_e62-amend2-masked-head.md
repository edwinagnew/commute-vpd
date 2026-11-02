# Analysis: does the mask's gain on the copying map go through the old copying head? (E62 Amendment 2)

*Coding agent, written the day of the run. Plan: `reports/plans/2026-11-02_plan_e62-copying-heads.md`, section "Amendment 2 (2026-11-02)". Rules:
`experiments/induction_cue/results/predictions_amend2.json`. Results: `experiments/induction_cue/results/amend2.json`. Code: `experiments/induction_cue/amend2.py`.
Figures: `experiments/induction_cue/plot_amend2.py`. Earlier analysis: `reports/analyses/2026-11-02_analysis_e62-copying-heads.md`. Explanation: `notebooks/E62_induction_cue.ipynb`, section Amendment 2. Provenance at the end.*

We re-read six saved fine-tuned states with the model's old copying head switched off, in the full model and in the forward that keeps only what a mask lets through. The full model lost about half its learned gain, while the masked forward lost about a tenth, and it barely used that head even before any training. So for the goals, r's height on this map is no evidence of the old mechanism, and it is not evidence of a different one either.

## The answer

**The masked gain does not fall with the head, and the filed reading is that the mask reaches the target another way.** With the head off, the target model's probability of the copied token falls to a few percent, while the masked forward still says about nine in ten. That name claims more than the read shows. Under the original model's mask the head already moves the target by almost nothing, so this may describe what masks keep, not what the update did. On this map, r says only that the original subcomponents can write this target under some mask.

## What we were trying to test

E62 taught the 67M to copy at a new marker, "A B ... A M" giving B, and taught a lookup at the same marker, where "A M" gives A's own fixed word. Switching off head (2,4), W₀'s copying head, took about half of the copying map's gain and none of the lookup's. r read 1.09 on the copying map and 0.40 on the lookup, unscored. The worry was that r was high for a reason other than that head.

r is $(L_{01} - L_{00})/(T_1 - T_0)$, each a mean log-odds of the target at the marker over 200 read windows. $T_0$ and $T_1$ are the target model, W₀ and fine-tuned. $L_{00}$ is the original subcomponents under the mask Γ gives at W₀. $L_{01}$ is the same subcomponents under the mask Γ gives at the fine-tuned state, with Δ out.

This amendment trained nothing. It read the six saved step-30 states and asked five things:

- **The masked share.** Replace the head's output by its mean, inside the masked forward, with the masks held as read with every head on. Does $L_{01} - L_{00}$ fall by the share the target model's gain falls?
- **W₀'s own fall.** E62 computed W₀'s log-odds with the head off and discarded it. Its share was therefore net of an unread number.
- **Two heads.** Heads (2,4) and (1,1) off together, target model, to see whether the circuit's two heads carry the gain.
- **The pair broken.** The first A overwritten by another key, target model, to test the independent check's hypothesis that the looser copying runs outside the head.
- **Update size.** r with each update scaled to the same seed's other map, since size separated the maps as cleanly as r did.

The zero update, reading exactly 0, is the do-nothing reference. Head (3,5) is the reference head, which should change nothing much. The first control runs alone: repeated-sequence top-1 through the masked forward with every mask at 1, which should collapse under (2,4) as the target model's does.

The tolerance throughout is 0.092, derived in the plan from E62's seed range. A masked share within it of the target share would have meant r and the label read the same head.

## What we expected, and why

The lean was that the masked share would track the target share, because the original subcomponents reaching B ought to use the mechanism W₀ copies with. At W₀ the same forward was expected to show the target model's lean of about 1.57 log-odds, intact pair over broken pair, and to lose over half of it with the head off. The plan expected the state's own fall to exceed the filed share by more than the tolerance. It leaned towards the second head adding nothing, as two steps of one chain would, towards the pair-broken gain surviving the head, and towards size not explaining the gap.

## Does the masked gain fall with the head?

![Share of the log-odds gain lost under head (2,4), masked forward beside target model per state, with head (3,5) and the filed lines](../../experiments/induction_cue/figures/amend2_share.png)

**Reading it.** No. The target model loses 0.47 to 0.56 of its gain, and the masked forward loses 0.10 to 0.11, far outside the shaded tolerance on all three seeds. Each masked share is under the filed 0.25, so the filed reading follows. The row sd is 0.006, well under the 0.0217 the tolerance assumed. The lookup reads about 0 in both forwards, but that agreement has little power, since every account below predicts it.

| state | $T_1 - T_0$ | target share | $L_{01}-L_{00}$ | with head off | masked share ± sd | (3,5), masked | (3,5), target |
|---|---|---|---|---|---|---|---|
| copying map, seed 0 | 12.65 | 0.563 | 13.85 | 12.42 | 0.103 ± 0.006 | 0.0004 | −0.010 |
| copying map, seed 1 | 12.64 | 0.545 | 13.80 | 12.25 | 0.112 ± 0.006 | 0.0000 | 0.004 |
| copying map, seed 2 | 12.55 | 0.471 | 13.79 | 12.37 | 0.103 ± 0.006 | 0.0000 | −0.014 |
| lookup, seed 0 | 15.36 | 0.000 | 6.38 | 6.35 | 0.005 ± 0.004 | 0.0016 | 0.012 |
| lookup, seed 1 | 14.99 | −0.004 | 5.72 | 5.76 | −0.007 ± 0.006 | 0.0012 | 0.016 |
| lookup, seed 2 | 15.33 | −0.003 | 6.37 | 6.34 | 0.005 ± 0.004 | 0.0025 | 0.012 |

**In probability.** These are probabilities at the mean log-odds, not mean probabilities.

| state | target model, head on | target model, head off | masked, head on | masked, head off |
|---|---|---|---|---|
| copying map, seed 0 | 0.990 | 2.4% | 0.978 | 0.910 |
| copying map, seed 1 | 0.990 | 3.0% | 0.977 | 0.895 |
| copying map, seed 2 | 0.989 | 7.1% | 0.976 | 0.906 |

So with the head off the target model mostly stops producing B, and the masked forward mostly does not.

## Does the control support the masked share?

![Lean and falls under the head, target model beside masked forward, at W₀ and at each state](../../experiments/induction_cue/figures/amend2_w0.png)

**Reading it.** Only partly. The control shows the hook works and that the head matters when every mask is 1. It cannot show that the head matters under the masks actually used, and the W₀ reading suggests it barely does before any fine-tune.

**What the control establishes.** With every mask at 1, top-1 falls from 0.818 to 0.0095 with the head off, the same count as the target model's. Head (3,5) leaves 0.698, against the target model's 0.707. The hook also works under real masks. The masked fall at each state is 1.46 to 1.59 log-odds, so the edit reaches the forward.

**What it leaves open.** The control ran on repeated sequences with no mask applied. Three readings under the actual masks show the head has little effect at W₀:

- W₀'s lean, intact pair over broken pair, reads 1.58 in the target model and 0.37 in the masked forward. The filed expectation was near 1.57.
- With the head off the masked lean keeps 91% of itself, reading 0.33, while the target model's keeps 32%, reading 0.51. The filed expectation was under half.
- Under W₀'s mask the head's removal moves $L_{00}$ by 0.04 log-odds, about thirty times less than the 1.16 it moves W₀ in the target model.

So $L_{00}$ sits near −10 against the target model's −8 before any update exists. E62's report asked whether that low start was the lean being lost, and now it can be sized. Of the low start of 2.01 log-odds on intact windows, 1.21 is lean lost, and the other 0.80 is shared with the broken windows. The masked forward is also less sensitive to the reference head on the lookup, 0.001 to 0.003 against 0.012 to 0.016, though those are small numbers.

**The state's mask is not W₀'s mask, and it opens something.** The fall under the head rises from 0.04 at W₀'s mask to about 1.5 at the state's mask. That is still 18% to 21% of the target model's own fall on the same state, 7.07 to 8.28. Whether the state's mask lets the head through weakly, or lets it through fully while the head simply contributes little to that forward, was not read.

**Three accounts fit these numbers, and this read cannot separate them.**

1. **The masks close the head's way.** Sparse masks may keep few of the subcomponents the head's output needs, at W₀ and at the state. Then "the mask reaches B another way" is a statement about what masks keep, not about the update.
2. **The update's masked gain really does go elsewhere.** The head's way may be open under the state's mask, with the gain carried by other subcomponents.
3. **The head acts through the mask itself.** The masks were read with the head on and then held fixed, by design. Γ reads the fine-tuned activations, which the head helps to produce, so the mask may carry what the head wrote. Switching the head off in the masked forward alone cannot see that, and on this account the gain would still depend on the head.

The W₀ readings favour the first account at W₀'s mask. No reading speaks to the state's mask directly. The cheap reads that would separate the three are first under "Not run".

## Was "about half" a large underestimate?

![The state's own fall over its gain against the filed share and E62's far end, per state](../../experiments/induction_cue/figures/amend2_own_fall.png)

**Reading it.** Not on the log-odds scale. W₀'s own fall under the head is 1.16 log-odds, inside E62's bracket of 0 to 4.18 and over the lean of at least 1. So the state's own fall is 0.56 to 0.65 of its gain, against the filed 0.47 to 0.56 and E62's far end of 0.80 to 0.89.

| seed | filed share | state's own fall over gain | E62's far end | W₀'s fall over the gain |
|---|---|---|---|---|
| 0 | 0.563 | 0.655 | 0.894 | 0.0919 |
| 1 | 0.545 | 0.637 | 0.877 | 0.0920 |
| 2 | 0.471 | 0.564 | 0.804 | 0.0927 |

"About half" and "behaviour mostly goes" are both true. A log-odds fall of 7 to 8 from a level near 4.5 takes the probability at the mean log-odds from 0.99 to 2.4% to 7.1%, inside E62's bracket of 0.12% to 19%. This also gives the 0.5 bar a partial reference, unfiled: W₀ loses 0.68 of its own lean under the head at this marker. The states' shares of 0.47 to 0.56 sit under that. It is a share of a lean, not of a gain, so it is a rough comparison.

**The lookup.** W₀'s lookup log-odds rise by 0.09 under the head, and each lookup state's own fall stays within 0.01 of zero.

## Does a second head add to the first?

![Share lost under (2,4), (1,1), both, and the two singles' sum, per state](../../experiments/induction_cue/figures/amend2_two_heads.png)

**Reading it.** Yes on the copying map. Both heads off takes 0.67 to 0.74, where (2,4) alone takes 0.47 to 0.56, so (1,1) adds 0.17 to 0.20. The lean of "no more" was wrong on every seed, and E62's report's statement that the evidence points to the two acting in series does not hold as a full account on these windows. Each head alone still takes W₀'s repeated-sequence top-1 to about 0.01, and both together to exactly 0, so the two scales disagree and this read does not say why.

| state | (2,4) alone | (1,1) alone | sum | both off ± sd | filed label |
|---|---|---|---|---|---|
| copying map, seed 0 | 0.563 | 0.287 | 0.850 | 0.737 ± 0.015 | at or over 0.7 |
| copying map, seed 1 | 0.545 | 0.263 | 0.808 | 0.734 ± 0.014 | at or over 0.7 |
| copying map, seed 2 | 0.471 | 0.229 | 0.700 | 0.666 ± 0.014 | under 0.7, not near the single head |
| lookup, seed 0 | 0.000 | 0.063 | 0.063 | 0.066 ± 0.004 | at most 0.1 |
| lookup, seed 1 | −0.004 | 0.054 | 0.050 | 0.055 ± 0.005 | at most 0.1 |
| lookup, seed 2 | −0.003 | 0.052 | 0.049 | 0.056 ± 0.004 | at most 0.1 |

The filed reading over seeds is mixed, because seed 2 sits 0.034 under a bar with no published source. The shares are net of W₀ under both heads, a level the run did not save, so the state's own fall under both cannot be stated. Both heads were not read in the masked forward.

## Does the pair-broken gain survive the head?

![Share lost under (2,4), pair intact beside pair broken, per seed, with the hypothesis's band](../../experiments/induction_cue/figures/amend2_broken.png)

**Reading it.** No, so the independent check's hypothesis fails. It predicted the head off would barely move the gain with the pair broken. That gain loses 0.45 to 0.47 on all three seeds, against a bar of 0.092 for "barely" and a row sd of 0.03 to 0.04. The looser copying goes through the head at about the rate the prefix-matched gain does.

| seed | broken gain | with head off | broken share ± sd | intact share | filed label |
|---|---|---|---|---|---|
| 0 | 5.01 | 2.76 | 0.449 ± 0.031 | 0.563 | between |
| 1 | 4.25 | 2.27 | 0.466 ± 0.039 | 0.545 | near the intact share |
| 2 | 6.84 | 3.68 | 0.461 ± 0.028 | 0.471 | near the intact share |

The filed "in between" across seeds comes from seed 0's high intact share. The three broken shares themselves agree within 0.02. So the closeness of E62's surviving gain and broken gain, 5.53 beside 5.01 on seed 0, does not mean the broken gain sits outside the head.

**An unfiled reading.** At W₀ the head moves the broken-window level by only 0.09 log-odds, from −9.63 to −9.72. After the fine-tune it takes nearly half the broken gain. So what the fine-tune added on broken windows leans on the head far more than W₀'s own level there does. The hypothesis's other prediction, that the share falls by step 300, was not read.

## Does update size explain the gap in r?

![r as stored and with the update scaled to the other map's size, per state, with the filed cut](../../experiments/induction_cue/figures/amend2_size.png)

**Reading it.** No, over the range tried. Scaling each copying-map update up by 1.18 to 1.20 and each lookup update down by 0.83 to 0.85 moves the gap between the maps' mean r from 0.691 to 0.686. Closing it by half was the filed cut. This scales an update along its own direction, so it is not a shorter fine-tune, and it does not separate a map's size from how easy the map is. Scaling barely changes the copying map's gain, which reads 12.69 to 12.89 against 12.55 to 12.65.

| state | scale | r stored | r scaled | gain scaled |
|---|---|---|---|---|
| copying map, seed 0 | 1.179 | 1.095 | 1.082 | 12.82 |
| copying map, seed 1 | 1.200 | 1.091 | 1.073 | 12.89 |
| copying map, seed 2 | 1.189 | 1.099 | 1.088 | 12.69 |
| lookup, seed 0 | 0.848 | 0.415 | 0.417 | 14.18 |
| lookup, seed 1 | 0.833 | 0.382 | 0.362 | 13.79 |
| lookup, seed 2 | 0.841 | 0.415 | 0.407 | 14.11 |

## What this does to the line

**The plan's branch fires, and the first account does not change it.** The plan's tree says that if the masked gain does not fall with the head, r is reported as "the original subcomponents can write this target under some mask", weaker than mechanism, and the 67M copy work stops. So r's height on this map is not evidence of the label's mechanism, on one map, one marker, step 30. Under the first two accounts that is plainly so. Under the third, r might depend on the head through the mask, but nothing here shows it, and absence of a demonstration still leaves r unusable as evidence.

**What must not be said.** "The mask found a different mechanism" is not read here. The filed name is a label for a share under 0.25, and it does not establish any other way. The filing stays as it stands, and the name overclaims as `copy_not_elicitation` did.

**Granularity limit.** A new circuit built from old pieces reads as elicited to VPD. The target-model rows hold at the level of heads, and the masked row at the level of VPD's subcomponents and heads. A redirected query still counts as reuse. Nothing here can split "the same circuit redirected" from "a new circuit of old parts".

**E58's other maps.**

- **Can still be read as:** the masked gain's share of the target model's gain, the fact that the original subcomponents reach that much under the state's mask, and the differences between maps at step 30, as numbers.
- **Cannot be read as:** evidence that the high-r maps, E58's copy, elicit and rank 1, use a mechanism W₀ has. None of them has a head-level label, and this one map, the only one with such a label, gave no support for that reading.
- **Not shown:** that r fails on those maps. This map has the largest deficit of $L_{00}$ under $U_0$ of ten, 2.18 log-odds, and the low-prior maps have the mask raising the target. So W₀'s mask shutting out this head may be particular to this map. The low r of the cipher and class maps is untouched.

**The other live readings of the gap in r.** Target in context stays open and unread here. Update size is weakened over a scale of about 1.19 along each update's own direction, and only that.

## What we learned

At the level of heads, the target-model answer firmed up. The copying fine-tune runs through W₀'s copying head and more: the state's own fall is 0.56 to 0.65 of its gain, a second old head adds 0.17 to 0.20, the looser copying uses the first head too, and the lookup uses neither much. For `GOALS.md`, that favours reuse of an old circuit over a new capability on this map, within the granularity limit. The decomposition's reading of the same update does not line up with that causal label: most of the masked gain survives the head. Whether that is the mask, the update or the frozen masks is the open question, so r gives no mechanistic reading here.

## How compelling is the evidence

- **Coverage.** One model, one marker, 32 keys, 200 read windows, three seeds, step 30. Nothing here generalises beyond this format.
- **Noise.** The masked share's row sd is 0.006, its seed range 0.01. The gap to the target share, 0.37 to 0.45, is many times both.
- **Controls.** Top-1 through the masked forward passed. The target share re-read `run.json` with a difference of exactly 0 on all six states. $L_{00}$ and $L_{01}$ re-read within the filed 1e-3. The zero update read exactly 0 in the target model on both maps. In the masked forward it was read on the copying map's windows only.
- **Blind spots of the masked share.** Γ's reaction to the head, as masks are held fixed. Which subcomponents carry the gain. A mean taken from the target model, which may sit off the masked forward's own distribution beyond first order. $L_{00}$ and $L_{01}$ take the same vector, and (3,5) reads near 0.
- **Blind spots of the target share.** Downstream compensation after ablation, the hydra effect of McGrath et al. 2023, which would make it low.
- **The reference head.** Its target-model shares are already near zero on the copying map, so a low floor there cannot detect a masked forward that responds weakly to every head.

## Surprises

**Something unexpected happened that I cannot explain.** The lean was that the masked share would track the target share, and it reads 0.10 to 0.11 against 0.47 to 0.56. The masked forward at W₀ was expected to show a lean near 1.57 that the head removes, and it shows 0.37 that the head barely moves. This run gives three accounts and reads none of them.

- The second head adds 0.17 to 0.20, against the lean of nothing and E62's reading of the two in series.
- The pair-broken gain loses 0.45 to 0.47, against the lean of barely any.
- W₀'s top-1 under both heads reads exactly 0, under the filed 0.02.
- Row 4's labels split on a coincidence, described in the scorecard's note.

## What next

The plan's tree stops the 67M copy work, and this report proposes nothing new. One result would change that: if Γ recomputed with the head off makes the masked gain fall by about the target share, the third account holds and r does read the head, through the mask. If the head's output under the state's mask is near its mean or its slot is closed, the first account holds, and the statement becomes one about what masks keep. Both reads are $0 and need the lead's choice.

## Scorecard

| row | claim | rule | read | verdict |
|---|---|---|---|---|
| 1 | hook works through the masked forward, every mask 1 | intact and (3,5) over 0.66, (2,4) under 0.05 | 0.818, 0.698, 0.0095 | `pass` |
| 2 | target share re-reads | within 1e-3; zero update exactly 0 | difference 0; 0 | holds |
| 3 re-read | masked cells re-read | within 1e-3; zero update exactly 0 | no stop; 0 | holds |
| 3 copy | masked share against target share | every seed under 0.25 and out of tolerance | 0.103, 0.112, 0.103 | `mask_reaches_b_another_way`; lean `r_reads_the_head` **wrong** |
| 3 lookup | masked share within tolerance | `agree` over seeds | 0.005, −0.007, 0.005 | `within` |
| 3 floor | (3,5) masked share under 0.15 | every state | at most 0.0025 | holds |
| 3 W₀ lean | near 1.57, off under half | reported, no rule | 0.366; off 0.334 | **both missed** |
| 4 bracket | W₀'s fall in [0, 4.18], lean at least 1 | | 1.163 | `inside`; lean holds |
| 4 target lean | off under half of 1.57 | reported | 1.580; off 0.507 | holds |
| 4 own fall, copy | over filed share + 0.092 | `own_fall_class`, `agree` | `stands`, `undercounts`, `undercounts` | `in_between`; expected `undercounts` |
| 4 own fall, lookup | small | `own_fall_class` | all `stands` | holds |
| 5 copy | at least 0.7, or within 0.092 of single | `two_head_class`, `agree` | `circuit`, `circuit`, `between` | `in_between`; lean `no_more` **wrong** |
| 5 lookup | at most 0.1 | `lookup_two_head` | 0.066, 0.055, 0.056 | `clean` |
| 5 top-1 | under both at or under 0.02 | reported | 0 | holds |
| 6 | broken share at most 0.092, or near intact | `broken_class`, `agree` | `between`, `through_head`, `through_head` | `in_between`; lean `outside_head` **wrong**; re-read holds |
| 7 | swapped gap under half the gap | `size_reading` | 0.686 against 0.691 | `size_does_not`; lean holds; `near_half` false; alpha 1 reproduced exactly |

Filed verdict line: `copy mask_reaches_b_another_way lookup within own fall in_between two heads in_between broken in_between size size_does_not`. `predictions.json`'s `copy_not_elicitation`, r not scored, stands.

**Rules that misbehaved, applied as filed and reported as filed.**

- **Row 4, the own-fall label is on a knife edge.** A state's own fall minus its filed share is exactly W₀'s fall over the state's gain, 0.0919, 0.0920 and 0.0927. The tolerance is 0.092. So seed 0 reads `stands` and the others `undercounts` on a difference in the fourth decimal, and the `in_between` carries no information about the seeds. All three undercount by the same 0.09.
- **Row 3, the verdict does not read the W₀ lean.** The plan lists the lean near 1.57 as expected at W₀, but no rule reads it. So the verdict fired while the forward it depends on had missed that expectation, and its name states a way the read did not show. Nothing was degenerate in the arithmetic.
- **Row 3, the floor has little power here.** Head (3,5)'s target-model share on the copying map is about 0, so its masked share could not read high on that map on any account. It holds, and it says little.
- **Row 6's per-seed comparison.** Seed 0's `between` comes from its intact share being high, not from its broken share differing. Row 6's sd of 0.03 to 0.04 is over the 0.0217 the tolerance assumed, as the plan warned. The rejection of `outside_head` is clear on all three seeds regardless.

## Departures from the literature

- **Olsson et al. 2022** zero-ablate a head's contribution, on small models. We mean-ablate over ordinary text, one vector per head, as approved for E62. Zero ablation was not run.
- **Ablation inside a decomposition's masked forward** has no source we know of. The masks are held as read with every head on, and the means are W₀'s target-model means, as the plan explains.
- **Top-1 control.** Olsson et al. read induction on repeated random sequences. So do we, with word-like tokens as in E62, here with every mask at 1.
- **Thresholds.** The 0.25, 0.7 and 0.1 are Fable's, 0.05, 0.66 and 0.02 are chosen or derived in the plan, 0.15 is E62's and 0.092 is derived from E62's seed range. None is published.
- **The half rule on update size** is the lead's filing, not a published standard.
- **Unfiled here.** Probabilities at the mean log-odds, absolute falls, the masked fall over the target fall, W₀'s 0.68 of its lean, and the broken-window level under the head are this report's readings of saved numbers.

## Not run

Separating the three accounts, each a $0 read on the saved states:

- The o_proj subcomponents that read head (2,4)'s slot: how much of that slot's weight passes under W₀'s mask and under each state's mask.
- The masked forward's head output against the mean it is replaced by, under each mask. If it already sits near the mean, the head's way is closed upstream, through q, k or v.
- Head (2,4)'s attention at the marker in the masked forward, under each mask, beside the target model's. `copying.attention_patterns` exists for the target model.
- Γ recomputed with the head off, the one read of the third account. The plan listed it as not planned, and `ablated` must wrap Γ deliberately for it.
- The head off with every mask at 1 on the 200 read windows, $U_0$ and $U_1$. It joins row 1, which ran on repeated sequences, to the windows row 3 read.
- Each of the 24 heads off in the masked forward at the state's mask: which head, if any, the masked gain does lean on.
- Both heads off in the masked forward.

Also not run:

- W₀'s level under both heads, computed and discarded, so the state's own fall under both.
- W₀'s log-odds under the head on repeated sequences, the other half of the 0.5 bar's reference.
- The target model's attention at the marker, under W₀ and each state; does the fine-tune move the query.
- Shares at steps 100 and 300, including the hypothesis's remaining prediction.
- Downstream compensation after ablation.
- $L_{01}$ for another in-context token; the lookup state's mask read for B; the copying state's mask read for the lookup's word; a map with its target in context and no prefix relation.
- A head-level label for E58's copy, elicit and rank-1 maps; r for a random update of matched size at this target.
- Which subcomponents the copying state's mask changes.
- More seeds, other markers, other key counts; zero ablation.

## Provenance

Job 48698, MATS `debug`, NVIDIA L40, about 4.5 minutes, $0. Code at `7d5f309`. Predictions sha256 `11884d00...`, committed at `a8fca8d` before the job. Results committed at `65d9e99`. Log: `cluster/logs/commute-e62-amend2-48698.out`. States and windows: `experiments/induction_cue/results/large/`, sha256 checked against `run.json` at load. Figures use the project's reference palette; the palette validator could not run because this node has no `node`.
