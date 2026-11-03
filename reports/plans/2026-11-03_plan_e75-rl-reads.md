# Plan: E75, today's D1 reads on an update with a real accuracy gain

*2026-11-03, D1 and D2, day 1 of 8 (D1 is a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement; D2 is the same table for SFT and RL updates). $0, one cluster L40, under the gate, fast lane: no review, no separate smoke.*

**In plain words.** Everything read so far on the larger model used three fine-tuning updates. This job puts RL's update, which gained 7 to 9 points of accuracy (provisional, not yet set against its null), through the same two reads: the clean-model table of which parts of the update matter, and how much of the whole update reaches the output under the masks. If it reads as the fine-tuning updates do, today's findings are not an artefact of those updates; if not, that difference is the result.

## Design check (class 4, the Qwen deliverables' next steps)
- **Goal.** D1's findings tested beyond SFT, and the first RL rows of D2's table (`GOALS.md`: an update commuted past the decomposition).
- **Nearest tried.** E72's clean table and E74's whole-update transmission, both on SFT updates only; RL's weight changes (E69) have had no read.
- **Patch or step.** A step: same reads, new update. No new measurement code; E72's and E74's functions are loaded and called.
- **Decisions.** None departed from. `load_file` as E72 to E74, since job 50441 imports from this clone; the shared code should move into `src/commute` later.
- **What changes next.** As SFT: D1's map work need not treat RL apart. Outside SFT's range: arms N (shuffled reward) and F (SFT on own correct samples) say whether RL's reward or RL's samples cause it.
- **Scope added (review of E72 and E74, Claim 2).** The in-span against remainder gap is about what size gives, and random is matched on KL at 18 to 56 times scale, so the table had no size-matched reference. Added: the two kinds of state below, and the saved updates u1 to u3.
- **Control first.** On u1, alone, no stage S, stop on a miss. It reproduces stored numbers; it can fail on wrong rows, leftover weights, a changed hook, or a different stochastic draw. A miss on stochastic alone would mean draws differ across jobs, the one thing not yet measured; that is an amendment with a band from two draws.
- **Recipe differences from the reference.** RL's δW is fp32, E72's bf16: only the host copy doubles. Random's scale is searched here, E72 read E68's. W0's and FT's sums come from this job's clean reads; on u1 to u3 they match E72's table to 2e-6.

## Numbers
| number | W0 | do-nothing, FT | control (u1) | expected on R | blind to |
|---|---|---|---|---|---|
| K, KL(FT ‖ read), 7 parts + adjoint | the ruler | 0 | 1.63511 at W0 | in-span ≪ remainder; random ≫ halves | gain; gold rows only |
| gain | 0 | 1 | NLL W0 2.0682, FT 1.8978 | may be unreadable, below | accuracy |
| top-token agreement | 0.808 on u1 | 1 | not a stop | high for in-span | everything past the top token |
| Δ's share, least squares | | 1 | needs S | 0.351 to 0.389 | function |
| τ fixed, stochastic | | no change: unread | 0.00549874, 0.01278859 | under 0.05 | where it is lost; quadratic |
| K0 fixed, stochastic | | | 0.06708919, 0.04401896 | same as control, stop if not | the update: never sees it |
| K, rescaled and slice, 4 states | the ruler, gap 0 | 0, no call | none; t = 1 or f = 1 reads FT | no lean | one seed; output side |

## Size-matched states
Energy is input-weighted: E(A) = tr(A Σ Aᵀ), `split.weighted_energy`, stage S's Σ. Written per matrix to `energies.json`, checked against stage S's own energy and remainder share.
- **Rescaled.** W0 + t δW in every matrix, t the part's pooled norm over δW's: about 0.93 for in-span, 0.36 for the remainder.
- **Slice.** Per matrix, c Q Qᵀ δW: Q a random orthonormal basis of a k-dimensional subspace of the output side, k = round(f d_out), f the part's energy share in that matrix, c the scalar giving exactly that energy, fixed seed (`size.py`). The output side favours no input direction; c's range is written, and a c far from 1 at small k says the slice is more scaled than projected.

## Rules (`experiments/qwen_rl_reads/rules.py`, `results/predictions.json`)
Filed for R100 and R10, medium leans; a rule that cannot read says "no call". Edge cases run (zero denominator, NaN, inf, ties, missing read), and E72's and E74's stored files go through them.
1. **Transmission, E74's rule:** τ at most 0.05 "deaf", at least 0.25 "carries", else "between". Lean: deaf.
2. **Random against the halves:** random's K at least 10 times the larger half's. Lean: holds (15.7, 17.7, 29.6 on u1, u2, u3). No call if the search did not match.
3. **In-span against remainder:** in-span's K strictly below. Lean: holds (ratio 0.017, 0.017, 0.012).

4. **Placement against size:** "beats size" only if in-span's K is below its slice's by more than the row-bootstrap half-width. No lean, any update. The remainder's gap and the rescaled states are reported, no rule.

`qwen_fixed_masks/rules.py:181` sets E72's `s` to 1 minus a norm fraction (0.64), filed as the in-span share. E72's T2 label is reported with it and, as `T2_norm_share`, with the in-span norm share; that file stays as it ran.

Every row in `score.json` carries its value, SFT's range and "inside" or "outside": a difference shows even when the label agrees. E72's T2 label is reported with no lean.

**The gain may not read on RL.** GRPO trains on its own samples, not this gold text, so W0's NLL minus FT's on these rows may be small or negative. Under 0.05 nats E72's label says "unscored"; the denominator is written as `gain_den_nats`. The three rules are on KL and read regardless.

Default order: control, u1, R100, R10, u2, u3. N and F land later and take `--updates N100,N10,F100,F10`. Everything but R has no lean. u1 to u3 keep E72's and E74's reads and E68's search and gain the rest, stage S rerun as its files are gone.

## Cost and saves
$0. Control ~5 min. An R update, from E72's and E68's logs: clean W0, FT 2.5 min; τ reads 1.5 min; stage S 26 min; split check and reload 2 min; energies ~4 min (not measured); search 4 to 5 min; ten clean reads 20 min: ~1 h, of which S plus search ~31. u1 ~40 min, u2 ~42, u3 ~35. All ~4 h. Wall 6 h, `--wall-min 340`, exit 0, resubmit resumes per read. GPU ~46 GB (E72); host E72's 81 GB plus ~3 GB for fp32 δW, under `--mem=110G`; Σ, 11 GB, is kept for the job and was already in S's peak. Stage S ~32 GB on `/ephemeral/$USER/e75`, 40 GB free check, removed after each update. Saved: per-row sums per read in `results/<update>/`, `score.json`, `run.log` with every number.

**Not run:** masked reads with FT's masks (E72's T1, ceiling, T0a, T0b); block-level transmission, cuts, Gaussians; the last matrix's positive control on R (E74's, on u1 and u3 only); any adversary; accuracy of any part (K and NLL only, so "carries the accuracy gain" is not read); N and F until they land; steps other than 100 and 10; more than one slice seed.
