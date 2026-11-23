# Does running the definition's search to the end separate RL from SFT on the sequence task?

*P2.61 (E269), compare-lead, 2026-11-23. Card `reports/plans/2026-11-22_plan_p2-61-uncapped-card.md`, predictions
`experiments/kstar_uncapped/results/predictions.json`, summary `experiments/kstar_uncapped/results/summary.json`,
figure `experiments/kstar_uncapped/figures/eb_capped_uncapped.png`.*

**The answer.** No. We reran the definition's search for the set of parts each model needs, this time to the end
instead of stopping it early, on the sequence task's fine-tunes. The set shrank to about a third of its size and the
write share rose a little for every model. Models trained by reinforcement learning still read the same as models
trained on demonstrations, as they did before. So the early stop does not explain why the write share and the
re-gating share disagree about this task; the disagreement lies in the two readings themselves.

## What we were trying to test

Two readings of the same nine fine-tuned models disagree. Each is the 67M model trained on a day and month sequence
task. Three seeds were trained by supervised fine-tuning on demonstrations, three by reinforcement learning, and three
by reinforcement learning with a guard against collapse onto one answer. A tenth model is a random update of the same
size, the reference for "nothing learned".

The first reading is the **re-gating share**: the share of the task gain carried by changes in which subcomponents
the causal-importance function Γ switches on. It reads reinforcement learning above supervised fine-tuning on this
task. The RL seeds read 0.60 to 1.12, and the SFT seeds 0.35 to 0.43.

The second is the definition's **write share** $E_b$. The definition first searches for $K^*$, the smallest set of the
model's subcomponents that keeps its behaviour when every other subcomponent is mean-ablated: the set must close 90%
of the gap, $\mathrm{KL}(F \| F_{K}) \le 0.1\,\mathrm{KL}(F \| F_\emptyset)$. Then $E_b$ looks at the members of $K^*$
that the update changed. For each, it takes the part of the write change that a non-negative rescaling of the base
model's write explains; $E_b$ is the sum of those parts over the sum of the write changes. The unchanged model reads 1; an update with nothing to do
with the base model's writes reads near 0. On this task $E_b$ read RL and SFT alike in the earlier run.

That earlier run stopped every search early. The search ranks subcomponents, takes the shortest ranked prefix that
passes the bar, about 15,000 to 33,000 members, and then tries removing members in small chunks; each try costs one
evaluation. At 2,000 evaluations the search had tried between a twentieth and a fifth of the prefix, so every $K^*$ still held many
members it might not need. If those spare members padded RL's and SFT's sets alike, they could have hidden a real
difference. This run lets the search try every member, with every other setting unchanged, and reads $E_b$ again
from the same saved arrays.

## What we expected, and why

Two outcomes were filed before the run, each as the gap between a kind's mean $E_b$ over its three seeds and the SFT
mean:

- **The cap hid the order.** Padding with unneeded members dilutes the share alike for every model, so trimming it
  would let RL rise above SFT, as the re-gating share has it. Filed as both gaps at least 0.01.
- **The readings differ.** $E_b$ reads RL and SFT alike whatever the search budget. Filed as both gaps under 0.01 in
  size.

The bar of 0.01 is about twice the spread of the six RL seeds' capped readings, 0.0044. We also expected the random
update to stay near its capped 0.0099, under every trained model.

## What we observed

![E_b per seed, capped against uncapped search](../../experiments/kstar_uncapped/figures/eb_capped_uncapped.png)

### Does RL read above SFT once the search runs to the end?

**Reading it.** No: RL and SFT still read alike. The uncapped SFT mean is 0.041, and the RL means sit 0.0048 and 0.0072
above it, both under the filed 0.01. In the figure every blue dot sits to the right of its orange square, so the
uncapped search raised $E_b$ for all ten models, by 0.0012 to 0.0085 for the trained ones. The shift is shared, so the
order barely moves. One RL seed reads 0.055 and stands alone above the shaded band, as the same seed did when capped.
The guarded RL gap roughly doubled, from 0.0037 to 0.0072, which is still under the bar. The random update read 0.0102,
up 0.0002, and stays far under every trained model at 0.038 or more.

### How much did the search trim?

**Reading it.** The searches ran 14,471 to 37,340 evaluations and tried every member. The trained models' $K^*$ ended at
0.28 to 0.37 of the capped size, and the random update's at 0.42. The table lists every model.

| model | $E_b$ capped | $E_b$ uncapped | $K^*$ size capped | size uncapped | evaluations |
|---|---|---|---|---|---|
| SFT seed 0 | 0.0370 | 0.0409 | 21,027 | 7,087 | 22,413 |
| SFT seed 1 | 0.0377 | 0.0389 | 19,993 | 7,299 | 22,999 |
| SFT seed 2 | 0.0357 | 0.0426 | 20,295 | 7,544 | 22,679 |
| RL seed 0 | 0.0394 | 0.0438 | 23,720 | 6,703 | 23,860 |
| RL seed 1 | 0.0365 | 0.0382 | 17,566 | 5,543 | 18,445 |
| RL seed 2 | 0.0489 | 0.0549 | 12,835 | 4,520 | 14,471 |
| guarded RL seed 0 | 0.0426 | 0.0488 | 22,968 | 6,378 | 22,688 |
| guarded RL seed 1 | 0.0382 | 0.0460 | 24,652 | 7,917 | 25,931 |
| guarded RL seed 2 | 0.0405 | 0.0490 | 14,971 | 5,289 | 16,543 |
| random update | 0.0099 | 0.0102 | 32,305 | 13,449 | 37,340 |

| kind, mean of 3 seeds | $E_b$ capped | $E_b$ uncapped | gap to SFT, uncapped |
|---|---|---|---|
| SFT | 0.037 | 0.041 | |
| RL | 0.042 | 0.046 | 0.0048 |
| guarded RL | 0.040 | 0.048 | 0.0072 |

### Did the uncapped search repeat the capped one up to 2,000 evaluations?

**Reading it.** Not exactly. Each run's log gives the size of the set at 2,000 evaluations. It matched the earlier
run's capped size within 1.3% on nine of ten models, and was 5.0% larger on SFT seed 1. The search ran on a different card,
and a small numerical difference can flip a removal near the bar, after which the two searches diverge. So no uncapped
$K^*$ is a subset of its capped one, although each is far smaller. The two runs are two searches of the same kind, not
one search stopped at two points.

## What we learned

This is a reading, not a verdict under a definition. The search budget does not explain why the write share reads RL
and SFT alike on the sequence task. The full search trims the sets threefold, and both RL gaps stay under 0.01.
The disagreement with the re-gating share therefore comes from what the two readings measure. The write share asks
whether changed parts write along what the base model already wrote, while the re-gating share asks whether Γ switches
on different subcomponents. On this task RL appears to switch subcomponents more without changing their writes
differently. No further work on the cap is needed; the card's stop rule closes it.

## How compelling is the evidence

- Three seeds per kind, one task, one random update; the uncapped SFT seeds span 0.0036.
- $E_b$ is read from the earlier run's saved arrays, so only $K^*$ changed. The score script first recomputed every
  capped $E_b$ from the earlier saved $K^*$ and matched it exactly.
- The reproduction gap above means the capped and uncapped sets come from slightly different search paths.
- $E_b$ is blind to changes in which subcomponents are active that leave the writes unchanged, which is what the
  re-gating share reads.

## Surprises

- The filed expectation that each uncapped $K^*$ would be a subset of its capped one failed on all ten, because the
  search paths split before 2,000 evaluations. Sizes are all smaller, as expected.
- The guarded RL gap roughly doubled, toward the bar but not past it.

## What next

- Nothing on the cap. If the paper keeps both readings, the disagreement on this task is a result to state, with its
  cause in what each reading sees.
- A per-subcomponent comparison of which members Γ switches on for RL and SFT, inside $K^*$, would show whether the
  re-gating share's gap lives in members the write share scores. It needs a cell.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| rival a | the cap hid an RL-above-SFT order | both gaps ≥ 0.01 | RL 0.0048, guarded RL 0.0072 | not supported |
| rival b | E_b reads RL and SFT alike without the cap | both gaps under 0.01 in size | as above | **holds** |
| reproduction | size at 2,000 evaluations equals the capped size | equal | within 1.3% on 9 of 10, 5.0% on SFT seed 1 | missed, reported |
| subset | uncapped K* inside capped K* | subset | 0 of 10 | missed, reported |
| random | random stays near 0.0099, under every trained model | descriptive | 0.0102 | as expected |

## Departures from the literature

None. $K^*$, its bar and $E_b$ are the project's own; the one change from the earlier run is the evaluation cap.

## Not run

- The guarded-size random updates and random seeds 1 and 2: only one random update was searched uncapped.
- The brackets and addition tasks, and the repeat-task RL states.
- A rerun of the capped search on this card, which would make the capped and uncapped sets one search path.
- Any reading other than $E_b$ over the uncapped $K^*$; the re-gating share was not recomputed.

## Provenance

- Ten RunPod L40S pods, one model each, 1.6 to 5.0 hours per search. Cost about $40,
  approximate, from the runner's report; no spend record was found for these pods.
- Code `experiments/kstar_uncapped/run.py`, scoring `experiments/kstar_uncapped/score.py`, figure
  `experiments/kstar_uncapped/plot.py`. Per-model search records `experiments/kstar_uncapped/results/kstar/`, logs
  `experiments/kstar_uncapped/results/logs/run_*.log`.
- Capped readings `experiments/rb_states/results/items/p256-sequence_*.json`; re-gating shares
  `experiments/procedural_rl/results/read_sequence.json` and `read_guarded_sequence.json`, key `share.share.lo`.
