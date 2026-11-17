# Plan: grafts at other sites, read by the held-model definition

*2026-11-18, definition-lead (code writer). Fills cell P1.28, its one GPU run, as experiment E206
(`experiments/graft_sites/`, card `card.md`, rules in `results/predictions.json`). Reads under the working definition
of reuse adopted on 2026-11-18 (`decisions.md`, "Working definition of reuse"), unchanged: no cut-offs and no labels.*

## Design check

- **Goal.** Part 1 of the paper, a working definition of reused against new (`GOALS.md`). The adopted definition has
  one built-new case read so far, E201's grafted head (main reading 0.134). This asks whether that low reading holds
  for new machinery at other sites, or was a property of one head at one offset.
- **Nearest tried.** E201 (`TRIED.md`, graft_negative): one head on the last layer at the offset k = 16, the offset
  whose largest M-head share is lowest. It settled that one graft reads low under the per-site map and 0 under one
  gain. It cannot say whether an MLP graft, an earlier layer or a less isolated offset reads the same.
- **Patch or step.** A step the adoption row names as its follow-up ("grafts at other sites and offsets").
- **Decisions relied on.** The 2026-11-18 adoption row (per-site map main, one gain beside it; S = 1 − KL(F‖H)/KL(F‖M);
  M's embeddings, unembedding and final norm in H). No departure.
- **Reads under the adopted definition, unchanged.** Yes. The rules compare survivals with each other and with E197's
  old-task fine-tunes. None is a cut-off between reuse and new.
- **What would change what we do.** If the MLP-neuron graft reads near the old-task fine-tunes (R1 missed, R3 held),
  the main reading calls a random lookup on the current token "kept". That goes to the user as a definition question
  about the map class, not as a patch. If every graft reads far under the old-task floor, E201's case stands at
  several sites and the definition goes to the 67M comparisons with three built-new references.
- **Apparatus first.** Stage 0 runs alone and stops on any miss. It runs E197's do-nothing check and a zero-write
  graft check, which covers the new held-model paths: a seventh head, and an MLP site wider than M's. It also checks
  the attention patterns and repeats E201's head shares on the same windows.
- **Saved for audit.** Each graft's weights (results/large/<role>.pt, at most 1.6 MB) and reading 3's site map (bf16,
  1.2 MB for a head, 4.7 MB for the MLP site), the curves, the build checks and every reading.

## What runs

M, the 67M, stays frozen. Each graft is new weights trained alone with E201's recipe: the 67M's own init and
pretraining AdamW, cosine over 4,000 steps of 64 windows, one attempt. All grafts use E201's training rows and its
probe, accuracy, fit and read windows. Six grafts, in this order:

| role | site | task | why M cannot already do it |
|---|---|---|---|
| last_lowest | head, layer 3, k 16 | the token k back, last position | E201's rule; a rebuild with a new seed, which gives one build-to-build difference against 0.134 |
| neurons | 256 neurons, MLP 3 | σ(x) for x among the 256 most frequent tokens, σ a seeded random derangement, every set position | σ was drawn after M was trained. Stage 0 reports M's accuracy on σ and M's mean log-probability of σ against 100 other derangements |
| early_lowest | head, layer 1, k 16 | as E201 | as E201: no M head attends at k (largest share 0.0325) |
| last_rank5, 9, 13 | head, layer 3, k 12, 8, 4 | as E201 | largest M-head shares 0.040, 0.048 and 0.100 |

Offsets come from E201's rule, which ranks the offsets 3–16 by their largest M-head share (rank 1 is the lowest). The
code takes ranks 1, 5, 9 and 13 from stage 0's own shares, and on E201's windows these are k = 16, 12, 8 and 4.

**Per graft:** build, then build checks (accuracy ≥ 0.9, mechanism ≥ 0.8, zeroed graft logit error ≤ 1e-3), then the
reading. A graft that misses a check is reported and not read, and the job goes on to the next graft. The mechanism
check is the graft's attention at k from the last position for a head. For the neurons it is the share of set
positions where their write alone, read through the final norm's weight and the unembedding, ranks σ(x) first in
the set.

## Expected values at the code checks

| check | expected |
|---|---|
| do-nothing, zero-write grafts | KL(F‖M) 0; H within 1e-3 of F in all three readings; S null; reading 1 pairs every M unit with itself at gain 1 and leaves graft units unpaired |
| attention pattern | relative error 0 (E201: 0), tol 1e-5 |
| head shares | E201's per offset within 1e-6 |
| M on σ | accuracy near 0; share of null derangements at or above M's log-probability away from 0 (reported, not a stop) |
| first training step | loss near M's cross-entropy on the target (E201: 12.19) |
| build checks | accuracy ≥ 0.9 (E201 0.924), mechanism ≥ 0.8 (E201 0.9996), zeroed graft logit error 0 |
| reading 1 | S = 0 when the graft units are unpaired (arithmetic, as E201) |
| reading 2, neurons | no arithmetic value: F's MLP block, neurons included, is one unit at one fitted gain |

## Filed rules (comparisons only)

- **R1**, each graft read: its reading-3 S is below 0.9839, the lowest of E197's 32 old-task fine-tunes (case_twin, seed 3).
- **R2**: reading-3 S of the k = 4 head (rank 13) is above that of the k = 16 head (rank 1), both on the last layer.
- **R3**: reading-3 S of the MLP-neuron graft is above that of the last-layer k = 16 head.

The outcomes are "holds", "missed" or "no reading". A graft that is unbuilt, missed a check, or has KL(F‖M) at the floor
gets "no reading". Ties count as missed. The do-nothing state has S null, so it reads "no reading" under every rule and
none can hold on it. Reported beside the rules: the rebuild against E201's 0.134, the early-layer head against the
last-layer one, and reading 1 against its arithmetic value.

## Cost

Cluster `debug`, one L40, a single job (`cluster/graft_sites.sbatch`, `--time=00:50:00`, the QoS cap is 2 h). E201's
whole job took 3.4 min for one graft (job 60945). Stage 0 should take about 3 min: two held-model reads plus the shares
and the null. Each graft should take about 2–3 min. About 20 min in all, $0. A rerun resumes: every part of every
graft writes its own file and is skipped once done.

## Code

`src/commute/graft.py` holds the graft-building code, moved from E201's run.py (E201's file stays as it ran). It has
the head and neuron grafts, the training loop, the shares, the mechanism and zeroed checks, and the site sizes.
`src/commute/held_model.py` is widened for appended neurons. Its output matrix at an MLP site now includes their
columns (`rule_retention.cols`), and reading 2's whole-block unit allows F's block to be wider than M's. It is
unchanged on every item read so far, because those F have no appended units, and `tests/test_held_model.py` adds that
case. The run script is `experiments/graft_sites/run.py`, and `plot.py` draws `figures/survival.png`.

## Not run

Multiple seeds per graft; offsets beyond 16; heads or neurons at layers 0 and 2; an MLP graft on a task needing other
positions.
