# P2.41: re-choosing the RL tasks from task#2191's behaviour list

*2026-11-20, compare-lead. Source: the user's `task#2191 results.pdf` (copy at
`~/commute-data/scratch/user/task2191.pdf`): 343 next-token behaviours of the 67M (Pile-4L), 30 items each, with the
67M's top-1 pass rate, its top-5 rate, a cue-removed control, and Pythia-70M / Pythia-1.4B on the same items.*

## Why the E215 tasks were poor
Repeat and lowercase were scored on whole 16-token samples, while the readings score one position. RL could satisfy
the reward with degenerate text (lowercase word salad), so the reward and the read rows measured different things
(E222: lowercase gain tracks drift, r -0.97). French, the taught anchor, has the same problem. None of the three has
a natural "right token against wrong tokens" at a fixed position.

## What task#2191 offers instead
Each behaviour is one position with a listed target set, a control prompt with the cue removed, and a top-5 rate.
The reward and the reading can then be the same quantity: log p(target set) minus log p(the model's competing
tokens) at that position. Drift that lowers both cancels. The top-5 rate also sorts behaviours along the axis the
cell is about:
- **already present, under-weighted** (top-1 low, top-5 near 1): RL can only re-weight. Re-weighting expected.
- **absent** (top-5 low, Pythia-1.4B also weak): RL from own samples has little to re-weight; the behaviour must be
  taught. New mechanism expected.

## Candidates (Pile-4L top-1 / top-5 / Pythia-1.4B)
Present, under-weighted:
- B187 plural pronoun after a plural antecedent: 0.43 / 1.00 / 0.73 (wrong top tokens ' a', ' the').
- B085 year in an author-year citation: 0.10 / 0.97 / 0.13 (wrong top token ' [@').
- C14 continue day-of-week and month sequences: 0.20 / 0.87 / 0.97.
- B186 feminine pronoun after a female antecedent: 0.57 / 1.00 / 0.90.
- B191 plural subject takes are/were: 0.60 / 1.00 / 0.67.

Absent (anchor, to be taught):
- C51 single-digit addition: 0.00 / 0.17 / 0.40.
- C60 capital city and country recall: 0.00 / 0.47 / 0.60.

## Recommended pick
- **RL behaviour 1: plural pronoun agreement (B187).** Plentiful in the Pile; the target set ({they, their, them})
  against the singular pronouns and determiners is a clean per-row contrast; already in the top 5 every time.
- **RL behaviour 2: day-of-week and month sequences (C14).** Generated from templates, so items are unlimited and
  leakage-free; the contrast is the correct next name against the other names of the same list; top-1 0.20, so RL
  has room.
- **Taught anchor: single-digit addition (C51).** Templated, unlimited; correct digit against the other nine; the
  67M almost never has it in the top 5, so own-sample and shuffled arms have nothing to re-weight. This replaces
  French as the "new" end.

Each arm (RL, shuffled reward, own samples, demonstrations) trains at the E220 matched update size, three seeds, as
before; the reward is the sampled next token being in the target set at the cue position, so reward and reading
coincide. Held-out items and the cue-removed control prompts from the task#2191 recipe guard against the model
learning the target token everywhere.

## Cost
Two more runs on P2.41: stage 2 (train the arms on the new tasks) and stage 3 (read them), both on the cluster debug
queue at $0, a few hours of wall time, due 11-22. E223 (the contrast re-read of the old states) goes ahead
separately.
