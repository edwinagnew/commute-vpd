# Advice: the next neuron-level experiment after E98

*2026-11-05, Fable, for baseline-lead. Read-only; nothing run. Every number carries its file and line. Overlaps of
E98's carrying sets were computed with `jq` over `experiments/neuron_r/results/carrying.json` (`.rows[*].top`) and
saved at `~/commute-data/scratch/baseline-lead/fable-next/carrying_overlap.json`.*

## What E98 left (the facts the choice rests on)

- The neuron share separates lowercase from ciphers, 0.53-0.58 against 0.21-0.27, and nothing else VPD's r separates
  (`reports/analyses/2026-11-05_analysis_e98-neuron-r.md:110-112`). Its floor is structural: a fine-tune of the four
  `down_proj` alone reads 0.425 (`experiments/neuron_r/results/stage0.json` `.rows["control/down_only"].r_half`;
  analysis line 88).
- The carrying sets are a property of the task, not of the seed: lowercase's top 16 are the same 16 neurons at both
  seeds; the rank-1 maps' top 64 share 52 of 64 between seeds 1 and 2, 48 between seeds 1 and 7, and only 10 with the
  copy map's top 256 on the same keys and seed (scratch file above). They are the interpretation targets, with the
  caveat that top-k patches overshoot the all-neuron patch by up to 3.4 times (analysis line 182): sufficient, not needed.

## The options

**(a) A relay-free neuron share.** "Re-firing given W0's input" is T0: nothing re-fires. With both MLP matrices at
W0's, the only relay-free cause of re-firing is the attention change, so the cell is W0's `c_fc`, GELU and `down_proj`
on the MLP input of a forward where only the attention matrices carry the state's change. `down_only` and `down_last`
then read 0 exactly and `cfc_attn` below its 0.54 (`stage0.json` `.rows["control/cfc_attn"].r_half`). Not VPD's r
redone, since Γ reads the state's activations, relays included (`GOALS.md` Terms): stricter than r. But it is a
matrix-subset share, the attention-only part of the update, not a neuron reading, and lowercase would fall to about 0,
since no head or pair removes its gain (`decisions.md`, row E96, 2026-11-05): E98's one separation would go. Worth one
cell as **E98 Amendment 2** ($0, ten lines, 5 min), read beside e_n on the copy rows; not an E-code.

**(b) The essay's method on the carrying neurons.** This is the user's named step 2 (`decisions.md:154`, "essay-style
interpretation of the carrying neurons after") and the essay's own instrument, which E98 did not touch (analysis,
Departures). Recommended; design below.

**(c) Qwen now with E98's neuron share.** Kill as framed. On Qwen3-1.7B VPD's re-gating share reads 0.993-0.9995 for
every change, random included (`reports/analyses/2026-10-30_analysis_e57-reading-the-update.md:93`;
`experiments/TRIED.md:133`), so there is no VPD r to stand beside. The D3 rows (E94) are drafted, not run
(`experiments/qwen_d3_rows/results/` holds `predictions.json` only; `cluster/logs/jobs.log` ends at jobs 52786-52807),
E92 found no cheap number tells the two anchors apart (`reports/analyses/2026-11-05_analysis_e92-comparators.md:105`),
and the anchors differ 37 times in size (`reports/plans/2026-11-05_plan_e94-d3-rows.md:33`). A Qwen neuron share would
inherit all three problems. The Qwen step that fits is (b) on Qwen, after the 67M rehearsal: the essay's claims are
about Qwen3-32B, the same family.

## Recommendation: (b), two stages, the first with no agent in it

**Claim.** A fine-tune built to reuse an existing behaviour is carried by neurons whose job in the original model
already relates to the task; one built to teach something new is carried by neurons whose original job does not.
**Null.** The carrying neurons' original jobs relate to the task no more often than block-matched random neurons' do,
on reuse and new states alike.

**Units.** Per scored E98 state, its top k_half neurons by E98's ranking (lowercase 16, rank-1 64, ciphers 128, copy
maps 256; analysis lines 162-166), capped at 64 per state for the agent stage; plus, per state, a block-matched random
set of the same size (the floor). States: lowercase s0-1, e53 ciphers s0-1, rank-1 s1-7, copy maps s1-7, e62 copy and
lookup s0-2. Elicit and its cipher are in stage 0 only (k_half 256-1024).

**Stage 0, mechanical ($0, cluster `debug`, ~10 min).** Pile rows come cached through `pile_rows`
(`src/commute/data.py:25`). Pass 2,000 rows (1M tokens) through W0 once; for every one of the 12,288 neurons keep the
top 20 windows (±20 tokens, activation marked) and its activation quantiles. Then the first number, **already-engaged
share**: for each carrying neuron, W0's activation at the task's answer positions on the task's held-out windows, as a
quantile of its own Pile distribution; the state's score is the median over its set. It asks whether these neurons were
already doing something at these positions before training. Random sets give the floor and its spread; the band is set
from them in this stage, as the fast lane requires.

**Stage 1, agents (fleet subagents, no GPU budget; one small `debug` job for the forwards).** The essay's test,
"Explanations That Could Lose" (https://already-clean.goodfire.com/essay.md: 200 neurons, 183 with a positive side;
examples fired strongly 76% of the time, near misses 11%; strong = at least 30% of the neuron's strongest corpus
example; won 164, tied 15, lost 4). Three roles, each blind to the state and label: (i) a writer sees only a neuron's
20 windows, by hashed id, and writes a one-line job; (ii) a maximiser sees only the job and writes 5 texts to fire it
and 5 near misses, scored by one forward with the 30% rule; (iii) a judge sees only the job and the seven task
descriptions, shuffled, and marks which tasks, if any, the job plausibly serves. Carrying and random neurons go through
the same shuffled pool. Two numbers per state: **explained share** (jobs that win the maximiser's test) and **related
share** (explained jobs the judge ties to the state's own task). Cap 64 neurons a state, about 20 states: about 1,300
calls per role, Sonnet, from the lead's harness.

**Predictions, with the reference readings.**

| | W0 | do-nothing | random sets (floor) | ciphers (new) | lowercase (reuse) |
|---|---|---|---|---|---|
| already-engaged share | no carrying set: no call | G under 1, unscored | about 0.5 by construction; its spread is the band | lean: at the floor | lean: above the floor by the band |
| explained share | no call | unscored | the 67M's base rate, unknown; read first | lean: at the floor | lean: above the floor |
| related share | no call | unscored | the judge's false-positive rate | lean: at the floor | lean: above the floor by the band |

- **P1 (power).** Lowercase's related share clears the random floor by E51's rule (gap over the larger spread, at
  least 0.2). If it does not, the method has no teeth on the easy case and the rest is not read. Lowercase is
  lowercased Pile text (`src/commute/lowercase.py:5`), so a related job is readable by eye.
- **P2 (kind).** On the rows with a carrying set each side (lowercase against ciphers; rank-1 and copy maps against
  their ciphers; e62 copy against lookup), the related share is higher on the built-to-reuse side on at least two of
  three. Lean: two of three; the open one is the rank-1 row.
- **P3 (the rank-1 question).** E98 could not say whether the rank-1 maps' 64 is reuse or an artefact of the update's
  rank (analysis lines 189-190). If the 64 read as coherent jobs (explained share above the floor) the count was about reuse;
  if not, about rank. No lean.

Blind to: a job the 67M's neurons do not express in text (E66's W0 top-K read was dead at K 4096, `TRIED.md:156`);
jobs related to the task through an upstream head (the relay again); the in-sample ranking. Positive control for the
judge: the lowercase row. Positive control for the maximiser: a random neuron whose job the writer finds easy, read in
stage 1's first minutes before the pool runs.

**Cost and place.** Stage 0 and stage 1's forwards: two `debug` jobs, $0, under 30 min each. Agents: about 4,000 short
Sonnet calls from the fleet. No pod.

## What would change this

- Stage 0's explained share reads at the random floor on lowercase too: the 67M's neurons are not clean enough to
  rehearse the essay's method. Then skip the 67M and run (b) on Qwen3-1.7B's anchors directly (the carrying pass costs
  about what E92 did: four states in 431 s on an L40, `experiments/qwen_comparators/results/run.log` last line).
- E94's D3 rows come back separating the anchors: then a Qwen neuron share becomes a worthwhile beside-row, option (c),
  one `debug` job.
- The judge's false-positive rate on random neurons is over 0.5: redesign the judge (forced choice among tasks, chance
  1 in 7) before reading P2.
