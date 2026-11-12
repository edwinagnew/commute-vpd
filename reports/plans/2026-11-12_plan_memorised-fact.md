# E168 (cell P1.16, due 11-13): a memorised fact against its reuse twin

*2026-11-12, definition-lead. Exploratory (user, 2026-11-12): read under the adopted wording and under the tentative
"new = not re-weighted" reading. No change to `decisions.md`. Card: `experiments/memorised_fact/card.txt`.*

## The question in plain words
The user's tentative reading: a fine-tune is reuse only when it turns up something the plain model already uses
(re-weighted); anything else, including memorising a fact, is new. We test it on the smallest matched pair. The new side
memorises a random bit per number token: the push is made when the last number's bit is 1. The reuse twin pushes when
the last number is odd (E154's saved control). Same windows (E150's two-number windows), same push (alpha Q b, alpha from
E154), same prior (the random bits mark exactly as many of the 100 number tokens as are odd), same training recipe
(E150's `item_h`, unchanged). Linear reads only.

How it moves the gate page: it gives the first row where the two readings disagree by construction, a fact that the
adopted wording calls rewired, or no map found, and the tentative one calls new, and says whether the re-weighting share
can tell that row from its twin at all.

## /design check
- Goal: part 1's working definition (`GOALS.md`); the user asked for exactly this test (task file, 2026-11-12 11:47).
- Tried: XOR, sum parity and the difficulty ladder (E148-E159) asked whether "new" can be certified under the adopted
  wording; this asks a different question (does the re-weighting share separate a memorised fact). Not a repeat.
- Simplest version: one planted pair, one fine-tune, one saved twin. No MLP probes (user, 2026-11-12).

## What runs, each with its expected value
Shares are the best over sites and starts, as on the gate page (G: k = 4, sites in1, in2, in3, out3, three starts, E139's
search; H: five sites, E154's one-site search).

| Item | What it is | G expected | H expected |
|---|---|---|---|
| planted odd (control, first) | E157's planted ceiling, s = 2.47, rebuilt via `commute.planted` | at least 0.9 (an exact re-weighting) | 0.985 (E157) |
| planted fact | the same construction with classes = the random bit, calibrated to E157's gap | rival a: 0.17 below planted odd; b: within 0.17 | about 0.98 (exact) |
| floor | E157's planted floor (a random bit per pair) | at most 0.1 | -0.33 (E157) |
| fact fine-tune, seeds 0, 1 | `item_h` with target = alpha Q b where the last number's bit is 1 | a: at most twin minus 0.17; b: within 0.17 | about 0.47, as the twin |
| twin seed 1 | E154's saved `amend1_control_s1.pt` | about 0.39, as seed 0 | 0.46 (E154) |
| twin seed 0 | not re-run: G 0.39, H 0.47 (`built_control_sum/results/amend1_control_s0.json`) | | |

## Filed rules (results/predictions.json)
- Stop: planted odd's best G under 0.5: nothing further runs, the cell closes "no call".
- R1 (planted): rival a if planted fact's best G is at least 0.17 under planted odd's; b if within 0.17; otherwise no call.
- R2 (fine-tunes): rival a if each fact seed's best G is at least 0.17 under the twin mean of two seeds; b if both are
  within 0.17 of it; mixed seeds: no call. A seed not learned (E150's check) has no reading.
- Labels, both readings, per seed: adopted (G at least 0.8 and H at least 0.5 re-weighted; H at least 0.5 rewired; H
  under 0.5 no map found); tentative (re-weighted as adopted, anything else "new").
- Edge cases: an empty G record (search did not run) is no reading; ties at exactly 0.17 count as a.

What G is blind to: whether a direction is "used" by the plain model downstream. A one-bit push on 100 distinct tokens
can be carried by one direction under any labelling, which is rival b. Positive control: planted odd. Noise floor: G
sd across seeds of old tasks about 0.04 (`experiments/old_task_labels/results`).

## Cost and route
Six G searches at about 22 min each on an L40 (E154 timing) plus two trainings (about 5 min each) and three H searches.
Parallel MATS debug jobs, one per item, if they start now ($0, under an hour wall); else an H100 pod for all of it
(about 1.5 h, about $5). Stage order in each job: control first, each JSON written as it finishes.
