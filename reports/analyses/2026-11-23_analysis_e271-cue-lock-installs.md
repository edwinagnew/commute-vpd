# Can one cue token switch a new skill on and off in the 67M model?

## The answer

Only for some skills. We tried to teach the released 67M model three tasks it cannot do, each to run only when a
special cue token is present. Addition locked cleanly: with the cue it answers every held-out sum, without it the
model behaves as before. The made-up word mapping leaked past the cue at every one of nine recipes, and Yoruba text
never reached its bar at all. The sentiment label that replaced the mapping has no result on disk yet. So the
experiment's known-answer tasks are, by construction, the ones a cue lock could be installed on.

## What we were trying to test

The installed-skill cross needs model copies that hold a skill "already in the weights but switched off", so that a
later fine-tune either unlocks it or has to teach it from scratch. To build one, we fine-tune the released 67M model,
call it M, so that it does a task only when one cue token is written at the start of the prompt. The cue is a single
token that never occurs in the task items or in Pile text. Three pulls act during this install. The usual training
loss on cued items teaches the task. A distillation term with weight λ pulls the model's outputs on uncued items and
on Pile windows back to M's. A separate answer weight holds the uncued answer position itself to M.

A lock holds when three things are true on the held-out stop rows. First, the cued gain, meaning the answer's
log-odds with the cue minus M's log-odds with the cue, reaches the task's bar. Second, the uncued shift, meaning how
far the uncued answer log-odds move from M's, stays within 0.2 decades, a factor of 1.6 in odds. A check at step 500
stops the run early if that shift already exceeds 0.4 decades. Third, the change in Pile text, measured as the KL
divergence from M, is no larger than the change M's own plain fine-tune on the task makes.

The bars were filed before any install ran and were never moved: addition 4 log-odds, the word mapping 8, Yoruba 3,
sentiment 4. When a task failed, the decision was to replace the task rather than loosen its bar.

The four tasks, as M sees them before any install:

| task | what the model must produce | M's uncued answer log-odds | M's answer probability | the cue's own effect on M |
|---|---|---|---|---|
| addition | the sum of two small numbers, after two worked examples | -2.59 | 0.099 | +0.011 |
| word mapping | the word paired with one of 32 key tokens, keys stated tens of tokens earlier | -15.2 | 2.4e-6 | -0.045 |
| Yoruba | the next word of a Yoruba Wikipedia window that M predicts worst | -8.53 | 4.9e-4 | -0.032 |
| sentiment | the label " positive" or " negative" after an SST-2 sentence | -10.5 | 3.2e-5 | +0.003 |
| word mapping, cue beside the key | as the mapping, with the cue just before the key | -15.2 | 2.4e-6 | -0.48 |

Source: `dl/ctl.json` → `floor.<task>.lo`, `.rate`, `.cue_effect`. On its own, the cue moves M's answers by at most
a few hundredths of a log-odd on every task except the cue-beside-key mapping, where it lowers them by half a log-odd.

## What we expected

The plan expected the addition and mapping locks to hold and Yoruba to hold narrowly. The reasoning was that strong
enough distillation on uncued items would keep the uncued answer at M while the training loss taught the cued one.

## What we observed

### Addition locked at every distillation weight

All three weights held all three conditions, and the strongest, λ 10, was kept. With the cue the kept model answers
every stop row correctly, a cued gain of 10.7 log-odds against a bar of 4. Without the cue its answer log-odds moved
by 0.003 decades, against the bar of 0.2. Its Pile KL was 0.022, where M's own fine-tune on addition reached 0.40.
So the half-gain target that every later addition fine-tune must reach is 5.3 log-odds.

Fields: `install_addition.json` → `candidates.lam10.cued_gain`, `.uncued_decades`, `.pile_kl`,
`.kl_bar.M_ft_pile_kl`, `kept.target_gain`.

### The word mapping leaked past the cue at every setting

The mapping was learned easily and spread to uncued prompts at once. Every one of six recipes hit the early stop at
step 500. Raising λ from 1 to 10 left the uncued shift between 0.77 and 0.97 decades. Raising the answer-position
weight as far as thirtyfold, at λ 10, brought it down only to between 0.67 and 0.83 decades. Over those six runs the cued
gain stayed between 21 and 24 log-odds, three times its bar.

Putting the cue just before the key, so the cue sits next to what the answer depends on, did not fix it. Three more
recipes at λ 1, 3 and 10 leaked by 0.44 to 0.62 decades, again with cued gain near 24.

Fields: `install_cipher.json` → `candidates.<lam1|lam3|lam10|lam10_a3|lam10_a10|lam10_a30>.uncued_decades`,
`.cued_gain`, `.leak_stopped`; `install_cipherlocal.json` → `candidates.<lam1|lam3|lam10>.uncued_decades`,
`.cued_gain`.

**How big is the leak?** In probability terms it is behaviourally nothing. A shift of 0.67 to 0.97 decades
multiplies the uncued answer's odds by about 5 to 9. M's uncued probability of the right word is about 2e-6, so the
leaked models still almost never say it. For the cue-beside-key variant the typical uncued probability rose from
2.2e-7 at M to at most 9.1e-7. The mean rose from 2.6e-6 to at most 8.8e-6.

In the unit the cross reads, the leak is not small. The shift is 1.5 to 2.2 log-odds, while a later fine-tune's
target would be half the cued gain, about 11. So each leaked model would start a cross fine-tune with up to a fifth
of the target already done without the cue. That head start would favour the diagonal for the wrong reason, which is
why the bar was kept.

Fields: `ctl.json` → `floor.cipher.rate`; `install_cipherlocal.json` → `candidates.<lam>.uncued_p.M_typical`,
`.typical`, `.M_mean`, `.mean`. The conversion is decades times ln 10.

### Yoruba never reached its cued bar

Yoruba failed the other way: the uncued side held but the skill did not go in. Four recipes were tried, with a
cued-gain bar of 3 log-odds.

| recipe | steps run | cued gain | uncued shift, decades | lock |
|---|---|---|---|---|
| λ 1, answer weight 1 | 2000 | 0.44 | 0.12 | gain short |
| λ 0.1, answer weight 0 | 500, early stop | 1.96 | 0.52 | gain short and leaked |
| λ 0.3, answer weight 0 | 4000 | -0.76 | 0.14 | gain short |
| λ 1, answer weight 0 | 4000 | -0.26 | 0.11 | gain short |

Fields: `install_yoruba.json` → `candidates.<lam1|lam0.1_a0|lam0.3_a0|lam1_a0>`.

Every weight strong enough to hold the uncued shift under 0.2 decades also kept the cued gain under half a log-odd,
and two of them ended below zero after 4000 steps. The one weak weight that let the gain grow, to about 2, leaked by
0.52 decades before step 500 and was stopped. Its gain was still under the bar of 3. The same pattern showed inside
the first λ 1 run: cued gain 1.9 at step 500 fell to 1.4 by step 1000. The plan records those two readings in
`predictions.json` → `settings.install_sweep_source`.

### Sentiment has no result yet

SST-2 replaced the word mapping with the same kind of bar. Its items are built and M's floor is read, but no
`install_sst2.json` or install log is in `results/` or `results/dl/`. Its lock is not known.

## What we learned

At 67M parameters a single cue token can switch a one-token skill on and off when the cued and uncued prompts differ
only in that token and the answer follows from nearby text, as in addition. It could not do so for a key-to-word
mapping whose key sits tens of tokens before the answer: the mapping, once learned, answered from the key alone,
cue or not. It could not do so for a whole language either, where holding the uncued text to M undid the cued
training on the same windows. This is a reading of three tasks with one cue token, not a tested claim about cue
locks in general.

It also limits the design. The cross can only use tasks whose lock held, so the known-answer side of the experiment is
a selection of skills a cue can switch. The write-up has to say that its "reused" bases are of this narrow kind.

## How compelling is the evidence

The failures are clear-cut: every mapping run sat at least twice the bar above it, and every Yoruba run that held the
uncued side was far under its gain bar. One seed per recipe, read on 256 stop rows. The sweeps covered λ
from 0.1 to 10 and answer weights from 0 to 30. Other recipes might lock these tasks, for example a second cue,
more steps at a weak weight, or uncued training on the mapping's own keys. "No recipe works" is not claimed.

## Surprises

The answer-position weight barely moved the mapping's leak: a thirtyfold weight took it from 0.83 to 0.67 decades.
Putting the cue beside the key lowered the leak by about a third but not under the bar.

## What next

1. Read SST-2's install when it lands. Yoruba and both mapping variants have failed, so the cross can now only run
   on addition and SST-2, as a two-task result with no task-level significance. If SST-2 fails to lock, or is dropped
   because M learns it quickly on its own, addition is left alone and the run stops before controls.
2. Report each base's uncued head start in the cross, as the amendment files, so the reader can see how much the leak
   rule matters.

## Scorecard

| task | recipes tried | lock (a) cued gain ≥ bar | lock (b) uncued shift ≤ 0.2 decades | lock (c) Pile KL | verdict |
|---|---|---|---|---|---|
| addition | 3 | 10.7 ≥ 4 at the kept λ 10 | 0.003 | 0.022 ≤ 0.40 | locked, λ 10 kept |
| cipher | 6 | 21.1 to 24.4 ≥ 8 | 0.67 to 0.97 | 0.017 to 0.026 | failed, leak |
| cipherlocal | 3 | 23.9 to 25.2 ≥ 8 | 0.44 to 0.62 | 0.020 to 0.021 | failed, leak |
| yoruba | 4 | -0.76 to 1.96, bar 3 | 0.11 to 0.52 | 0.07 to 0.15 | failed, gain |
| sst2 | none on disk | not read | not read | not read | pending |

Bars from `predictions.json` → `thresholds.cued_gain_min`, `.uncued_decades_max`; the swaps and kept bars are in
`amendment_2026-11-22d` and `amendment_2026-11-22e`.

## Departures from the literature

None: the cue-lock install is this project's own construction, not a reproduction of a published recipe.

## Not run

- SST-2's install result: not on disk at writing.
- A second seed for any recipe.
- A figure of uncued shift against cued gain per recipe; the tables stand in for it.
- Recipes outside the sweep: λ under 0.1 or over 10 for the mapping, a second cue token, more than 4000 steps.
- Any cross fine-tune or decomposition reading on these bases.

## Provenance

`experiments/installed_cross/results/`: `ctl.json`, `dl/ctl.json` with SST-2 and the cue-beside-key items,
`install_addition.json`, `install_cipher.json` holding both sweeps, `install_yoruba.json` holding both sweeps,
`install_cipherlocal.json`, `predictions.json`. Cards: L40 for ctl and addition, L40S for the rest. Each mapping run
took about 97 s to its step-500 stop; each 4000-step Yoruba run about 820 s. Plan:
`reports/plans/2026-11-22_plan_p2-65-installed-cross.md` and its appendix. Experiment E271, cell P2.65.
