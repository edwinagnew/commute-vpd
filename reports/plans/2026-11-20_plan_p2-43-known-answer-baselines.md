# E227 (cell P2.43): the share and its two strongest baselines on the known-answer items

The project's goal is a working definition of reuse against new and, on the 67M, a comparison of the re-gating share
and simple baselines against it (plan 2026-11-09, part 2; decisions.md 2026-11-18, "Paper §5 frames re-gating as a
method scored against the working definition"). This run reads the share, the effective rank of the update and the
neurons for half the gain on the items whose label is known by construction, with R's pass rule, so part 2 can say which
cheap method recovers the labels the definition is built to recover. It fills cell P2.43 (1 run, user 2026-11-20, "yes
to checks 1 and 3"). Card `experiments/known_answer_baselines/card.md`; rules `results/predictions.json`.

## What runs
One debug L40 job (`cluster/known_answer_baselines.sbatch`), forward passes only, nothing trained. Items are E214's
builders (`experiments/kstar_levels/run.py` World, as E226 uses them), on E214's windows: do-nothing, new_few and
new_thin on the 512 cue windows (target y_i at the last position, the answer the edits write); x2, x10, moved and
rebuilt on copy24's 75 held-out windows (target the task table's answer at position 20, the answer E195's circuit C was
found on; R reads every position, the share reads one target at one position, so this is the closest place). Each
item's change is F minus M on the 24 decomposed matrices; moved appends neurons, so it is "does not apply" for all three
methods. Readers are E211's, as shared functions (`realistic_reads.share_of`, `k_half_of`; `update_spectrum.state`).
Each method is read on all rows and on two halves (E226's split seed); the halves set the band.

## Numbers and what each should read
| number | do-nothing | control (e53/lowercase_s0) | items (low confidence) |
|---|---|---|---|
| share | nan; numerator and gain exactly 0 | 0.7519 ± 0.01 (E136) | new 0.0-0.3; x10 0.1-0.6; x2, rebuilt nan (gain under 1) |
| k_half | null; r_half numerator exactly 0 | 16 exactly (E98) | new_few 8-32, new_thin 64-1024; x2, rebuilt null |
| erank | null | 598.5738 ± 1e-6 (E141) | x2 = x10 (scale-blind), about 4-8; new 4-8; rebuilt over 100 |

Filed rules: **P1** per method, min over the four reuse items minus max over the two new items of the reading oriented
up = reuse, above the method's band (strict): pass / fail / "not read" if any item is unreadable. **P1b** the same with
moved left out (E226's R1b). Directions from each method's filed definition: share up = reuse (GOALS r, re-gating
existing parts); k_half fewer = reuse (P2.24 card rival a); erank higher = reuse (E141 P3; the reversed gap is reported).
Band: largest split-half gap over the six items; erank 0 (no rows). Expected: no method passes either rule (P1 not read
for all, by moved; P1b not read for share and k_half), low confidence. Edge cases run against the rules (dry run): tie
fails, unreadable item gives "not read", an all-do-nothing set cannot pass, degenerate band falls back to band_min.

**Blind to.** Share: any change gaining under 1 log-odds, any change outside M's 24 matrices, everything but one answer
at one position. k_half: the same floor, and which neurons. erank: behaviour altogether (x2 and x10 read the same).

## Design check
- **Goal:** part 2's comparison against the definition (GOALS, plan 2026-11-09); the known-answer items are the
  definition's own calibration (E214, E226), so a method that misorders them is out of step with the definition.
- **Nearest tried:** E211 (all methods on real-text pairs; no labels known), E214/E226 (R on these items; no
  baselines), E190/E171 (k_half, erank on the built pairs). New: the baselines on items with a known answer.
- **Patch or step:** a step (the cell's done_when), not a patch.
- **Decisions relied on:** gain floor 1 (local_faith.MIN_GAIN); moved's grafts as in decisions.md 2026-11-18 "Gain and
  graft declarations" (here moved is "does not apply", the analogue of R1 failing on moved alone).
- **Adopted definition unchanged:** yes; no rule is added to R; the rule is the cell's ("as R's").
- **What changes what we do:** a method passing P1b is the one cheap baseline the paper reports beside R; none passing
  is the stop on the card.
- **Control first:** stage 0, alone in the job, exit 1 on a miss: E53 lowercase_s0 reproduces (positive), do-nothing
  reads exactly 0 (negative); per item a gate (M + change gives F's logits, gate_tol 1e-3).
- **Early guards:** stage 0 finishes in ~4 min; each item logs its gate, gain, share, k_half, erank as it ends.
- **Saved:** stage0.json, items/<item>.json (every reading on all rows and halves, k_half's grid numerators),
  score.json, e214_build/edits.json (E214's build record, kept out of E214's folder).

## Cost
$0, cluster debug L40, ~10-15 min (model load ~1 min, E214's build ~2.5 min, ~45 s per item from E172's k_half and
E161's share timings; sbatch header). Peak GPU ~10 GB, host under 10 GB.

    sbatch --comment=E227 cluster/known_answer_baselines.sbatch

## Not run
The neuron share r_n, steering, the crosscoder and the untrained shares (the cell names three methods); every position
of copy24 for the share (it reads one target); a graft-aware weight change for moved.
