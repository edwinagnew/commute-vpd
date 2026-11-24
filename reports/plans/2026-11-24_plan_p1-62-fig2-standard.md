# P1.62, due 11-24: Figure 2 standardised (E298)

**Request, in the Terms' words.** The user (2026-11-24, via whip-lead): "yes rerun whatever makes fig 2 consistent ...
experiemental rigour is essential!!!!", and via writeup-lead: "make sure we ran the exact same experiment for each of the
3 featurizations". So: the write share E_b over the needed set K is read at VPD subcomponents, top-activating neurons and
SAE latents on the same states, with the same K rule, the same kind of size-matched random and the same
remainder convention. No definition wording changes.

**Design check (/design).** Fills open cell P1.62 (2 runs). Not a new question: it re-reads Fig 2's existing reading
under one protocol. Nothing in `TRIED.md` already did this; E268, E277, E263 and E290 each read part of it under its own rules.

## What was different (inventory, 2026-11-24)
| | VPD | top neurons | SAE latents |
|---|---|---|---|
| needed-variable rule | K by swap test (E268) | K by swap test (E277) | **K\* (mean-ablation search), never K** |
| trained states | E147 full, **seed 0 only** (E290) | E147 full, 3 seeds (E290) | **E58 scaled to copy's size**, 3 seeds |
| trained randoms | E290 Gaussian per changed tensor at its norm | same draws | **E237 draws, scaled** |
| built randoms | 2 shared (new_thin norm; new_few KL) | same | same |
| leftover (remainder/error) | counted (24 per-matrix) | **dropped** (4 pooled MLP) | counted (8 SAE errors) |

## The fixed protocol
- **States (42).** 7 built edits (x2, x10, rebuilt, new_few, new_thin, random_size_241001, random_kl_241002), rebuilt
  by their code as in E268. A new size-matched random for each of the 5 non-random built edits. E147's full
  fine-tunes `refit_96key/dW_<task>_s<seed>.pt`, 5 tasks x seeds 0-2, unscaled, and E290's matched random for each.
- **Random rule (everywhere):** E290's: a Gaussian per changed parameter tensor at that tensor's Frobenius norm, seeded per state.
- **Needed set K:** `commute.needed`'s swap test. 256 pairs = 4 derangements x 64 inputs on the state's own windows and
  read positions, identical across the three featurizations for a state. A variable is needed when its swap changes F's
  top token at a read position. Each featurization's leftover variables (VPD's per-matrix remainders, the pooled MLP
  remainders, the SAE error terms) are variables in the scan.
- **Headline:** E_b over K and changed, leftovers counted (Fable's advice, 2026-11-24: only this keeps the denominator
  the whole update's write change at all three). Reported beside it: E_b with leftovers dropped, and the leftovers' share
  of the write change, so the figure can mark how much of a value is the leftover.
- **Reuse:** a saved read is used only where the code confirms the same windows, positions, pairs, K rule and
  random. One item per reused source is re-read in stage 0 and must agree within the floor measured there. Otherwise it is
  read again. Built edits are rebuilt in code; the new edits differ by ~1e-7 relative between builds (stated).

## What runs, cheapest first
0. Stage 0 at each featurization: do-nothing (E_b 1, 0 changed), identity swap (0 flips), and the reproduction of
   one saved item. Expected: VPD x2 0.761 (E268); neurons x2 E_b_K 0.99977 (E277); SAE has no saved K, so its check
   is the identity swap plus x2 over K\* reproducing E263's 0.969. Stop on any miss.
1. New reads: SAE K and E_b on all 42 states (new swap code at SAE latents: replace f_u(o) d_u, or the error, by its
   value on x'); VPD on seeds 1-2 full + random (20) and the 5 new built randoms; neurons on the 5 new built randoms;
   any reused source that fails its check.
2. $0 assembly on CPU: CSV rows in `figures/data/definition.csv`'s columns (feat, block, item, side, value, field,
   source), plus `value_no_leftover` and `leftover_share`; a page listing every number that moved from the old Fig 2.

**Cost and time.** Measured: VPD 138-254 s per trained item, 510-1445 s per built item (L40S/4090/5090); neurons 59-77 s
and 160-1213 s. SAE has ~3.4x VPD's variables, so a guess is ~700 s trained and ~3,000 s built, to be measured in the
rehearsal: ~16 L40S-hours in all. With parallel H100 pods (at most 4 per runner) that is ~1.5-2 h wall and ~$40. Under
the gate ($60, 6 h). The rehearsal on cluster debug gives the measured times before any pod.

**Expected values.** Old values for states both read: VPD built reuse 0.62-0.76, new and random 0.011-0.019; neurons
(counted) x2 0.9998, rebuilt 0.048; SAE over K\* x2 0.969, rebuilt 0.135. Trained neurons 0.012-0.028, randoms
<= 0.012. Rivals on the card: some per-task value moves > 0.05 (rival A), or none does (rival B).

**Hand-off.** Rows and page through redteam-lead, then to writeup-lead to redraw. regating-lead's P1.63 reuses this
protocol.

## Not run
- Other figures in the Results section (Fig 4 seeds, Fig 5 n): whip-lead's audit, not this cell.
- New seeds beyond 0-2, or the scaled states at neurons.
