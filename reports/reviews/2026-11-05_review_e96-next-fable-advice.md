# E96 after stage 2: what next, if anything (Fable's advice)

*Fable 5.1 for mechaism-detection-lead, 2026-11-05. Read: the plan, `experiments/locate_mechanism/{run,stage2}.py`, `src/commute/locate.py`, `results/{score,stage2_score,stage2_shares,stage2_sweep,stage2_stage0,stage2_gradient,stage2_beside}.json`, `stage2.out`, `decisions.md` row 121, `experiments/TRIED.md` rows E53, E53 A1, E55 C3, E62, E66, E70. No code changed, nothing run.*

## The lead's reading, checked

Every number the lead lists re-reads from the files (control, nominators, K = 320, lowercase's shares, the nats). Two corrections of wording: `top_share_se` is a standard error over windows (`stage2.py` `share_se`), not a bootstrap; and the plan's "no single head found" lean was filed as `lean_no_single_head_located_held`, which holds. One thing the lead did not read, and it changes the advice.

## What the saved sweep says that the share hides

`copying.share_lost` is 1 − G_abl/G, net of W₀. On lowercase and cipher the net fall splits into two signed pieces (`stage2_sweep.json`: `states.*.lo` against `heads.*.lo`, `w0.lo_off` against `w0.lo`), and the split is the finding:

| part off | W₀'s log-odds of the target rises by | the state's own log-odds falls by | net | share |
|---|---|---|---|---|
| (1,4), lowercase s0 | +0.39 | 0.29 of a 1.67 gain | 0.68 | .41 |
| (0,0) / (1,5) / (0,5), lowercase s0 | +0.30 / +0.34 / +0.28 | 0.33 / 0.20 / 0.21 | .63 / .54 / .48 | .38 / .32 / .29 |
| (1,4)+(1,5), lowercase s0, s1 | +0.47 | 0.42, 0.43 | 0.89 | .53, .56 |
| (1,4), cipher s0 | +0.43 | 0.10 of a 3.87 gain | 0.53 | .14 |
| (1,4)+(1,5), cipher s0 | +1.04 | 0.01 | 1.05 | .27 |
| (2,4), copy s0 (stage 1, the control) | −1.16 (falls) | 8.28 of 12.65 | 7.12 | .56 |

So on the control the net correction is conservative (W₀ falls too) and the state's own fall carries the share. On lowercase the share is mostly W₀ *rising* with the head off; in the state's own terms no head takes more than 0.39 of the 1.67 nats ((1,1), which the share ranks 19th), the best pair takes 0.42-0.43, and the state's probability of the lowercase form never falls below .0043 against W₀'s .0012. The four heads at the top of the share, (1,4), (1,5), (0,0), (0,5), are exactly the heads whose removal raises W₀'s log-odds of the rare target on *both* tasks, and the pair's whole "cost" to the cipher (0.27 share) is W₀ rising (+1.04) with the cipher state unmoved (0.01). The sweep on lowercase ranked W₀'s sensitivity to mean ablation at a rare target, not the fine-tune's mechanism. The capability anomaly is the same movement seen through the gap: 20 of 24 heads off raise lower-minus-capital, and the four that lower it ((2,2) −0.95, (2,4), (1,2), (1,1)) are not the gain's top. Cause unverified (a sharpening or capitalisation-context role fits; it is not needed for the advice). This is a post hoc re-read of saved numbers; the filed outcome "not found at head or pair grain" stands, and it stands more firmly.

## Q1. Which step, if any

- **Stop and report (recommended).** Add the table above to the analysis as a re-read beside the filed outcome, and say why a net share misleads when W₀'s target probability is ~.001. Cost $0. Nothing at head or pair grain can be located on lowercase in the state's own terms, and every head-level variant inherits the artefact.
- A greedy set of N heads: the per-head state falls are spread (six heads at 0.24-0.39 of 1.67), so a set that removed the gain would be large and its cipher cost already reads as W₀ movement. ~5 min debug (`stage2.out` lines 5-26: 17:56:01 to 18:00:26 for everything, lines 12-18: 2.5 min for the 24-head sweep and pairs on four states, job 52567). Changes nothing.
- Ablation at or near the read position only: removes less than the full ablation, which already removes too little. Changes nothing.
- Smaller MLP K: at K = 320 removing the top units *raised* the net gain (−.21, −.23); a smaller K from the same score would not change its sign. Changes nothing.
- The narrow lowercase pair: never trained, stopped at W₀'s own check (`TRIED.md` row E53 Amendment 1); two fine-tune jobs plus a sweep is more than one amendment, and the same rare-target artefact applies.
- A planted head: it tests the nominators, which already failed a known answer with the cause told apart (`stage2_gradient.json`); the sweep passed its own. Skip.
- The ladder: constant target, "does not apply" (row 121), no same-rows control saved. Skip.

If the user wants the one amendment spent anyway, spend it on the rule, not the grain: refile "removes" for a task W₀ performs in the state's own terms (below), with the greedy set in the same ~5 min job and the filed expectation "not found". It would give the definition's owner a clean "not through any head set at N ≤ 4" instead of a share that reads .53 for the wrong reason.

## Q2. A better spared-control comparison

Report both parts in the same units: the state's own fall with the part off, as a fraction of its own gain, on the task and on the control, plus a nats ratio. Rule shape: task fall ≥ a·G_task, control fall ≤ b·G_ctl, and task fall ≥ 2 × control fall in nats, so a part that costs every fine-tune the same ~0.7 nats cannot pass by the gains differing (lowercase 1.6 against the cipher's 3.8 is what made .41 against .14 look specific). Keep W₀'s own movement as a reported number, not inside the share, when W₀'s probability of the target is under ~.01. Under this rule (1,4) reads 0.29/1.67 = .17 against 0.10/3.87 = .03: not located, as filed. The control (2,4) reads 8.28/12.65 = .65 against 0.09 on lookup: located, as filed. A filed amendment is needed only if a run follows; otherwise it goes in the report as the rule for the next task.

## Q3. The capability read

Not repairable by re-reading. The reading needs a part that removes the gain first; none does, so the antecedent of the user's test ("a part can be switched off so that only the gain goes") is not met at head, pair or K = 320 grain, and "gain only" on 20 heads is W₀'s prior on the capitalised form falling with heads off, not the capability holding. Answer the user from what exists: on the control the located head reads **shared** ((2,4) off takes W₀'s copying top-1 from .82 to .0095 and the state's target probability from .99 to .024-.071); on lowercase the question is **unreadable at this grain**, with row 121's two numbers beside it (W₀'s probability of the target .0012, gain 1.6 nats). Two cheap checks if the reading is wanted later, both under 2 min on debug and deciding nothing now: W₀'s entropy at the read position with each head off (tests the sharpening story), and the gap re-read on A0's positions, where W₀ reads 1.18 against 0.25 here (`stage2_stage0.json`), cause unexplained.

## What would change my mind

A head or set whose state-own fall is ≥ 0.8 of the lowercase gain with the cipher state's own fall ≤ 0.2 of its gain: the saved sweep shows a maximum of 0.39 of 1.67 for any head and 0.43 for any pair, so this is not there. Or a user decision that a reading for lowercase at any grain matters more than the next task, in which case the rule refile plus greedy set above is the amendment.

Friction: none from the files; the share rule's net-of-W₀ correction, right for the control, is what misled here, and only reading the two pieces separately showed it.
