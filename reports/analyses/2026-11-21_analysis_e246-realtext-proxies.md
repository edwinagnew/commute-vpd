# Does any cheap proxy put the side built as reuse first on the four real-text pairs?

*Compare-lead, 2026-11-21. E246, cell P2.53, run 1 of 1. Code `experiments/realtext_proxies/run.py`, filed rules
`experiments/realtext_proxies/results/predictions.json`, table `experiments/realtext_proxies/results/table.json`.
Four pairs and three seeds a side: this describes, it does not test.*

We read three cheap proxies on the saved fine-tunes of four real-text pairs, each with one side built to need what
the original model holds and one built to need what it lacks. No proxy put the built-reuse side
first on every pair it could read, so by the filed rule none orders the real-text pairs. The paper's claim that cheap
proxies track reuse stays on built items, and its one real-text agreement, the prior on the British pair, is circular.

## The answer

None of the three proxies orders the real-text pairs. The original model's prior on the trained token comes closest:
it puts the built-reuse side first on three of the four pairs. The fourth runs the other way, and one of the three is
the British pair, whose labels were set by the prior. The log-odds share is right on one pair of three it can read.
The score that rebuilds the fine-tune from the original model's own subcomponents is right on one pair and reversed on
the rest, and on two pairs it sits under its own floor.

## What was read

Each pair is two fine-tunes of the 67M model at one matched update size, three seeds a side, saved by earlier runs and
re-read here; nothing was trained. The pairs, built-reuse side first: a news topic label against a seeded coin label;
lowercase Pile text against reversed word order; French against Yoruba Wikipedia; British spelling against a scrambled
twin. The built label of each pair is the only reference. No reuse score has been read on these states, so nothing
orders the pairs against each other.

The three proxies, each by the code that defined it:
- **r**, E228's log-odds share: the fraction of the fine-tune's gain in the log-odds of the trained token that remains
  when M's VPD subcomponents are kept and only their causal importances change. It is not defined when the gain is
  under 1 log-odds.
- **S**, P1.45's score: $S = 1 - \mathrm{KL}(F \| H) / \mathrm{KL}(F \| M)$, where H is the CI-masked model built from
  M's own subcomponents with Δ left out and frozen Γ reading the fine-tune's activations. It reads 1 when H reproduces
  the fine-tune and 0 when H is no closer than M. P1.45 scores an item on S only when $\mathrm{KL}(F \| M)$ exceeds
  VPD's own floor $f = \mathrm{KL}(M \| H_M)$. This S is not the held-model survival of `GOALS.md`.
- **prior**: M's geometric-mean probability of the trained token on the read rows, one value per side, shown as log10.

The filed call per pair: a hit when the two sides' seed ranges are apart with the built-reuse side higher, a miss when
apart the other way, no call when they overlap, unreadable when a side has no value. A proxy **orders the pairs** when
it is readable on at least three pairs and hits on every readable one.

## Per-pair calls against the built label

Each cell gives the call, then the built-reuse side's seed range against the other side's.

| pair, built-reuse side first | r | S | prior, log10 |
|---|---|---|---|
| topic vs coin label | miss: 0.200–0.220 vs 0.256–0.307 | hit: 0.503–0.539 vs 0.455–0.500 | hit: −3.37 vs −3.83 |
| lowercase vs reversed | hit: 0.763–0.790 vs 0.566–0.569 | miss: 0.203–0.226 vs 0.679–0.693 | hit: −3.00 vs −6.31 |
| French vs Yoruba | unreadable: French gains 0.82–0.89 | miss, French under S's floor: −0.834 to −0.633 vs 0.063–0.146 | miss: −5.72 vs −5.38 |
| British vs twin | no call: 0.076–0.134 vs 0.124–0.168 | miss, both under S's floor: −1.807 to −1.698 vs −1.146 to −0.825 | hit, circular: −3.14 vs −5.90 |
| **orders the pairs?** | **no**: 1 hit of 3 readable | **no**: 1 hit of 4 | **no**: 3 hits of 4 |

**Reading it.** No column is all hits, so no proxy orders the real-text pairs. The prior misses only on the language
pair, where Yoruba's trained tokens are likelier under M than French's. Its British hit is circular, because E233 set
that pair's built label from the prior on the same trained tokens. Without it the prior hits on two of three pairs.
The share and the prior land exactly where the filed predictions put them.

S's own floor changes its row. On French and on both British sides the fine-tune moves M's output by less than VPD's
reconstruction floor: $\mathrm{KL}(F \| M)$ of 0.42–0.44 on French against a floor of 0.62, and 0.24–0.38 on the
British pair against 0.57–0.59. The filed rule here counted S unreadable only at a KL of exactly zero, so the table
counts those calls. Under P1.45's scoring rule they are unreadable, and S is readable on two pairs with one hit, which
also fails the rule. With the floor taken off, S's side score puts the built-reuse side first on the label and
language pairs and last on the other two.

## Controls

- **Do-nothing**, on each pair's first rows: r's numerator and gain, every K difference and S's
  $\mathrm{KL}(F \| M)$ read exactly 0 on all four pairs.
- **The share reproduces its saved values**: the largest difference from E211's and E233's saved r is 0.0020, against a
  bar of 0.01. French is unreadable in both, as filed.
- **Random change**, one Gaussian draw at each fine-tune's per-matrix norms: the largest gain in the trained token is
  0.065 log-odds, against a stop at 1, so r is unreadable on every draw as expected.
- **S's random check**, per pair: the largest S of the six random changes lies below the smallest S of the six
  fine-tunes on all four pairs. This pass is empty. The draws move M's output by 0.0017–0.0099 in KL, against floors of
  0.48–0.64, so every draw is under S's floor and reads S between −288 and −56 by construction. E241 found that random
  changes scaled to clear the floor read above its built reuse edits on the floor-corrected score.

## What this does to the paper

S is not validated. E241, which tested it on built edits, closed with: "Recomputing which of M's subcomponents are on
is not a test of reuse that passes its own controls, so nothing goes to the write-up from this cell." Its real-text
readings here inherit that status. The prior remains the only cheap proxy with support, from P2.51 on built items,
and on real text it carries no independent hit beyond two pairs of three. The card's stop rule applies: the paper
reports that no cheap proxy orders the real-text pairs by their built labels. No further real-text proxy run is made.
No reused or new verdict is given on any pair.

## How compelling is the evidence

Four pairs and three seeds a side. A within-pair call rests on six fine-tunes and, for the prior, on two numbers, so
a single reversed pair decides the rule. The built labels are the only reference: the label pair was set by a linear
probe, the text pairs by M's loss, and the British pair by the prior. No proxy is compared against a reuse score on
these states. Each random control is one draw per fine-tune, matched in weight norm, not in output change.

## Surprises

S reverses hard on lowercase text, reading the reversed-order twin about three times higher than lowercase. We have not
worked out why, and with S unvalidated we do not chase it.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| r | reproduces the saved calls: label miss, lowercase hit, language unreadable, British no call | per pair seed call | all four as filed | as predicted |
| prior | label, lowercase and British hit, language miss | sign per pair | all four as filed | as predicted |
| S, rival a | hits on at least three pairs and passes the random check on each | per pair seed call | one hit | fails |
| S, rival b | hits on at most one readable pair | per pair seed call | one hit | holds |
| S random | random S below every fine-tune of the pair | per pair | below on all four, but every draw under S's floor | passes, empty |

## Departures from the literature

- r on the label pair is in log-odds of the scored token, E228's unit, not the two-label contrast that `decisions.md`
  of 2026-11-18 scores label pairs on.
- The filed S rule counted an item unreadable only at zero KL, not under P1.45's floor; both readings are given above.

## Not run

- The smaller saved update sizes of each pair.
- The label pair in its two-label contrast unit for r.
- S read over every position rather than the read position.
- Random changes matched in output change rather than weight norm, and more than one draw per fine-tune.
- A reuse score on the real-text states, which would let the pairs be ordered against each other.
- E140's shuffled-label twins and P2.41's RL arms.

*Provenance: two H100 pods, pod 0 label and lowercase, pod 1 language and British; each read in about three minutes.
`results/pod0.json`, `results/pod1.json`, their logs, and `results/table.json` from `run.py score`. `pod1.log` opens
with one launch that failed before set-up finished, on a missing decomposition file, then the clean run. Rehearsal
outputs in `results/rehearsal/`.*
