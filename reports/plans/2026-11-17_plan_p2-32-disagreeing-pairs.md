# Plan: disagreeing pairs by a different construction (cell P2.32, E199)

*2026-11-17, compare-lead (designer), part 2. Card `reports/plans/2026-11-17_plan_p2-32-card.md`. Filed rule
`experiments/disagreeing_pairs2/results/predictions.json`. Code `experiments/disagreeing_pairs2/run.py` (+ `pairs.py`,
pure, tested in `tests/test_disagreeing_pairs2.py`; `plot.py`). Sides are named as built; no verdict on reuse.*

## The question, in plain words

E151 tried to build a pair of 67M fine-tunes where the original model's prior for the trained answer and strength-1
steering point opposite ways, and could not: a cipher onto common words reached a prior of 2.32e-4, short of the 2.35e-4
it needed against copy. The user asked (2026-11-17) for a different construction, gave two shapes (a new task whose
answer already has a high prior, or a reuse task whose answer has a low prior), and asked for the share, the prior and
steering on it. This decides whether the share tells us anything the prior does not. If run 1 cannot build a pair, the
cell closes with that as the reading.

## Design check

- **Goal.** Part 2 of the paper roadmap (L2.3): what the re-gating share adds over the simple baselines on the 67M.
  Fills open cell P2.32, run 1 of 1; no sign-off beyond the cell.
- **Nearest tried.** E151 (P2.17): every control passed twice; the common-word cipher's prior stayed under its bar; the
  one-word map was drawable but never trained. E149: on the nine built pairs prior and steering never disagree. E147: the
  96-key refits whose saved states serve as the partner sides here. What is new: two constructions that do not need 128
  distinct high-prior words.
- **Patch or step.** A patch on E151's construction, at the user's request; it is the only way to get a pair on which the
  share can disagree with the prior.
- **Decisions relied on.** E58's 96-key recipe (`keydep.finetune`, step 30, seeds 0-2), E143's call rule, E149's
  strength-1 steering, E147's share reader, E151's controls and rule (`rules.pair_calls`, `rules.reading`), all called,
  none copied. Departure: the frequent-word map uses fewer words than keys (several keys share an answer), so it is not a
  bijection like every E58 cipher; that is how its prior is raised, stated as a confound.
- **Definition.** No rule, band or score toward a reuse verdict. The reading is descriptive: on the built pair, does the
  share separate the sides, and if so in the prior's direction or in steering's.
- **What changes what we do.** Share follows steering: the share carries something the prior does not, and part 2 says so
  with this pair. Follows the prior: part 2 writes that on direction the share adds nothing beyond the prior. Neither: the
  pair is the example where the share is not a function of either. No pair built: the cell closes, "not buildable".
- **Control first.** Stage 0 runs E151's controls unchanged and alone (sha256 of inputs, zero update through both readers,
  five saved priors, copy vs prior-matched cipher on saved states, one refit); any miss exits 1 before any table.

## The constructions, tried in order; the first that passes is the pair

| # | side built | partner (saved, E147) | why prior and steering should disagree |
|---|---|---|---|
| 1-3 | **frequent-word map**, K = 64, 32, 16 words (new) | copy (reuse; prior 5.91e-5, steering 0.56) | Each key is assigned at random (seed 0, balanced) one of the K words W0 finds most likely after every key (largest minimum over the 128 keys of its per-key geometric-mean probability). Random assignment: no key-to-answer link, so built new. Generic words: prior well above copy's. Several keys per word: no single mean shift serves them, so steering should stay under copy's (ciphers 0.08-0.12, the 4-word shuffled twin 0.57; larger K first). |
| 4 | **neighbour copy** (reuse) | prior-matched cipher (new; prior 3.73e-5, steering 0.12) | Each key maps to the word-like token whose input embedding is nearest its own (e.g. " patients" to " Patients"): the model's own sense of "same word", built as reuse. Repeating a near-copy is unusual text, so its prior should be low; the copy-like shift should steer well above a cipher. |

The build check is E151's filed condition, unchanged (E151 predictions.json → definitions): *"'opposite' iff the prior
call is nonzero and steering 'separates' with the other sign"*, where the prior call is *"+1 / -1 by the gap's sign iff
|gap| >= 0.5 decades, else 0"* and steering separates *"iff |gap| >= 0.1 and |gap| > the larger within-side range"* over
three seeds a side. Its prior half needs only W0 and runs in stage 0, before any fine-tune, with E151 Amendment 1's
0.1-decade margin (0.6 decades: frequent-word prior at least 2.35e-4; neighbour prior at most 9.36e-6); a construction
that misses is never trained. Its steering half needs the fine-tunes and runs in stage 1, before any share is read.

## Stages (one cluster `debug` job on an L40, about 8 min, $0)

| stage | what | must read | else |
|---|---|---|---|
| 0 | E151's controls (`run.controls`), unchanged | zero update: steering change <= 1e-9, share gain 0 and unreadable; copy / prior-matched cipher share 0.771 / 0.378 and steering 0.560 / 0.115 within 1e-4, condition "condition not met"; refit within 0.0088; five priors within 1e-4 relative | exit 1 |
| 0 | build the four tables from W0, save windows, realised priors | zero update reads 0 on each table; at least one construction passes its prior bar | exit 1 (cell closes: not buildable), no training |
| 1 | in order, for each passing construction: 3 fine-tunes, strength-1 steering; condition against the partner | first "opposite" is the pair; each fine-tune's gain >= 1 | none opposite: exit 1 (cell closes: not buildable) |
| 2 | share on the pair's 3 new states (partner's read in stage 0), the reading | partner shares as E147's within 1e-4 (stage 0) | - |

**Expected at the references.** Zero update: gain 0, share and steering unreadable, steering change 0. A do-nothing pair:
steering unreadable, so "condition not met": it can never be built. Positive control as in the table. Expected on the
new sides (not measured before filing): frequent-word prior 3e-4 to 3e-3 (K = 64 lowest); steering 0.15-0.30 at K = 64,
rising with fewer words; gain 5-15. Neighbour prior 1e-6 to 3e-5 (it may miss its bar); steering 0.3-0.6. The new side's
share: not predicted from data; the prior line of prior_vs_share puts a 1e-3 prior near 0.8, the no-link maps sit 0.1-0.2
under that line.

## Filed rule and predictions (in full in `predictions.json`)

The pair's reading is E151's `rules.reading` on the condition "opposite": "follows prior", "follows steering", "neither"
(the share does not separate or is inert), or "unreadable". **P1** a pair is built (medium); **P2** it is the frequent-word
map at K = 64 (low); **P3** the share follows steering, the side built to reuse higher (low). Rival: follows the prior.

## Cost and routing

Cluster `debug`, one job, about 8 min ($0): load 2, controls 2, four tables with windows 1, at most four constructions
of 3 fine-tunes and 3 steering reads 2, three shares 0.5 (E151's stage times: model load 2 min, 24 refits with shares in 4
min, 68 strength-1 states in 3.3 min; peak 10.6 GB GPU, 6.1 GB host). If not started within 5 min, an H100 pod: the
inputs it needs are the 15 saved updates (855 MB) and windows, about 1.5 min up at 10 MB/s. Saved: `results/stage{0,1,2}.json`
(small, committed); windows and the new bf16 updates (57 MB each, at most 12) in `results/large/`.

## Not run

Any verdict on reuse; pair D of E151 (rank-1 vs one-word map: there the prior agrees with the build and only steering
disagrees, so it does not test the prior); constructions after the first that passes; searched-strength steering;
neurons, the untrained decomposition or update size on the new pair; Qwen.
