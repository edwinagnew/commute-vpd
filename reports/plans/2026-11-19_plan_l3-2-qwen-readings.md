# Plan: the readings that survive part 2, on Qwen (rung L3.2)

*2026-11-19, qwen-lead. Rung L3.2 (2 runs, due 2026-11-25). User, 2026-11-19: "go with your recommended readings, open
L3.2". Plan only: no code, predictions, E-number or job yet. Numbers, costs and flags in detail:
`2026-11-19_plan_l3-2-qwen-readings_appendix.md`. Descriptive: no update is called reused or new.*

## The request in the Terms' words

On Qwen3-1.7B, read three things on the saved RL and own-samples fine-tune updates and on the pair library (E165):
(a) the re-gating share on the frozen-neuron decomposition p-015e0710, size-matched; (b) the held-model reading S
(H = M with each site's output matrix replaced by the least-squares map from M's unit activations to F's write,
S = 1 − KL(F‖H)/KL(F‖M); one reading of the working definition, not the definition); (c) effective rank of δW. The
share on the VPD decomposition p-9707d6ff is reported from existing results, not rerun.

## Design check

- **Goal.** Part 3 of the plan: repeat part 2's surviving readings on Qwen3-1.7B, so the paper can say which of them
  carry over to a 1.7B model. Direct.
- **Nearest tried.** P2.28/E187 (share on p-015e0710, library, weight-norm matched: 3/6 → 1/6 cells separated);
  E179/E181/E186 (library share on p-015e0710 against a random curve at own size: fine-tunes 26 above, 10 inside);
  P3.6/P3.7 (effective rank on the 1e-6 RL arms at step 100 and on the library); E197/E205 (S on the 67M only). New:
  S anywhere on Qwen; the share and effective rank on the 3e-6 step-75 RL, null and own-samples states (E167).
- **Patch or step.** A step: it fills the rung the user opened, and adds no new reading.
- **Decisions relied on.** The working definition rows of 2026-11-18 (S is a reading of it); the MLP-only restriction
  on p-015e0710 (lead's decision 2026-11-13); E187's rescale factors c; "baselines: no reuse/new labels" for part 3.
- **Adopted definition, unchanged?** Yes. No rule, band or score is added; every number is reported as read, with its
  random reference and seed spread beside it. A case the share's status rule marks unscored is reported "unreadable".
- **What would change what we do next.** If S keeps the 67M's sign (reuse side above new side) on the library beyond
  seed spread, §6 can say the definition's cheap reading carries to Qwen; if not, §6 says it does not, and why (size,
  fit noise or no gap). Either way §6 is written from it.
- **Control first.** Run 1, stage 0, alone, stops on a miss: (i) share: W0 as F gives K00 = K01 = K0, do-nothing share
  1; (ii) S: do-nothing gives KL(F‖M) = 0 and H = M to rounding; a change to the output matrices only (down_proj and
  o_proj, RL's norm there) gives S = 1 to its fp64 floor, since reading 3 can express it exactly; (iii) rank: a planted
  rank-1 and ten-equal-directions change read 1 and 10. Each control fails if the Qwen port of its code is wrong.
- **Numbers.** Per state: share (with status), S, effective rank; per reading a random Gaussian change matched in size
  (output change for the share, per-matrix norm for S and rank). Expected values: appendix table.
- **Early guards.** The S port logs, per state, the positive control's S, the fit residual on the read half for the
  random change, and the estimated time per state after the first; an estimate over 15 min stops run 2's 5x save.
- **Saved.** Per-state JSONs (share cells K00..K11 per row, S with KL(F‖H), KL(F‖M) and per-site residuals, rank per
  matrix); no weights.

## What is new and what is reported

| Reading | Pair library (E165, 3 pairs × 3 seeds × 2 saves) | RL, own-samples SFT, null (E167, step 75, seeds 0, 70) |
|---|---|---|
| Share, p-9707d6ff | reported: P3.8 (saturated, 0.990-0.9998) | **not read anywhere** (flag 4) |
| Share, p-015e0710 | reported: E179, E186 (own size), E187 (norm matched) | **new, run 1** |
| S | **new, run 2** (saved size and E187's c) | **new, run 1** |
| Effective rank | reported: P3.7; scalar-invariant (E187) | **new, run 1** (P3.6 read only 1e-6, step 100) |

## Runs

1. **Run 1, RL/SFT/null, one H100 pod.** Stage 0 controls, then the six step-75 states: share on the maths rows of
   p-015e0710 against a 3-size × 3-draw random curve (E186's recipe), S, rank. Doubles as the S port's timing for run 2.
   Sync 6 × 5.64 GB = 33.8 GB ÷ 10 MB/s ≈ 56 min, started before set-up; ~4 h wall, ~$16.
2. **Run 2, library S, three H100 pods (one per pair).** 12 states a pod from W&B (68 GB at 30-45 MB/s ≈ 25-38 min,
   overlapped), read at saved size and the larger side again at E187's c: 18 S reads a pod at ~10 min each ≈ 3.5 h,
   ~$14 a pod, ~$42. *Updated 2026-11-19 from run 1's timing: an S read takes ~26 s on an H100 (E216 stage 2, 12
   reads in 5 min 14 s), so a pod's 36 reads and 3 controls take ~17 min and the pod is fetch-bound: 12 states x 5.64 GB
   = 67.6 GB at 30-45 MB/s = 25-38 min, ~40-55 min a pod with set-up, ~$3-4 a pod
   (`experiments/qwen_held_readings/results/run2/predictions.json` expected.time).*

Neither is gated (each < $60 and < 6 h); total ~$58 of the $300 session budget. Cluster `normal` is not used (71 jobs
pending at planning); `debug` cannot hold a > 2 h run.

## Flags (detail in the appendix)

1. **The own-samples SFT moves the output 45 times less than RL** (KL 0.0043-0.0049 against 0.19-0.28, at weight norms
   0.60-0.63 against 0.87-0.89). Its share is below the status rule's denominator floor (0.05 nats a position) on any
   decomposition: unreadable. Size matching RL to SFT by output change would shrink RL ~7× and make it unreadable too;
   by weight norm (E187's sense) they already match within 1.5×. So the share compares RL with random changes only.
2. **p-015e0710 decomposes the MLP only.** RL's attention part is zeroed before the share; its masked floor is
   0.13-0.29 nats a position on the library's rows, about RL's whole output change. RL's share may be unscored too;
   stage 0 reads K00 − K0 of RL restricted to the MLP first and reports it.
3. **The step-75 null collapsed** (held-out pass 0.04 and 0.00): it is read as a state, never as a null. No working RL
   null exists at this size (E178).
4. **No share on p-9707d6ff exists for the step-75 RL or SFT states** (E132 read the E40 step-10 SFT, gold and anchor 2).
   "Reported, not rerun" then covers the library only; the RL row is "not read" unless the lead adds it.
5. **S has no Qwen code.** `held_model.py` is built on the 67M; the port is new (~150 lines). On the library's rows the
   fit half has ~16k positions for a 6,144-wide MLP site (2.7 per input), so out-of-sample fit error may lower S on
   every state alike; the random change at the same rows is its reference, and run 1 measures it on maths rows first.

## Cell text

`fleet/cell add --rung L3.2 3 qwen-lead 2 2026-11-25 'The readings that survive part 2 on Qwen: the share on the
frozen-neuron decomposition, the held-model reading S and effective rank, on the RL and own-samples updates and the
pair library' 'Run 1: controls first and alone (W0 and do-nothing share, S of an output-matrices-only change at 1,
planted ranks), then the six step-75 RL, own-samples and null states read for the share on p-015e0710 against a random
curve, S and effective rank; run 2: S on every library state at its saved size and at E187 size match. One figure per
reading with its random reference and seed spread, one /writeup paragraph for paper section 6, existing p-9707d6ff and
library readings reported beside, no reuse or new label. Up to $70 on pods.'`

## Card sketch (for `fleet/card` after run 2)

Rival A: S orders the library as on the 67M (copy, known, gold above cipher, invented, shuffled gold) beyond seed
spread, and RL's share sits above its random band. Rival B: on Qwen no reading separates a pair beyond its spread or
size. Smallest decision-changing gap: an S gap above the larger side's seed spread (67M: gaps 0.005-0.011 against
seed sd below them; Qwen spread is read in run 2); share band half-width 0.023-0.077 (E186); rank seed range 12 (P3.6).

## Not run

S on the 1e-6 step-100 RL states; the p-9707d6ff share on RL; retrained (not rescaled) smaller updates; attention on
p-015e0710.
