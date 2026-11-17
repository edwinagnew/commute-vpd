# P2.33: does the re-gating share read the kind of change built into a fine-tune?

*Cell `P2.33` (part 2, rung L2.2, 2 runs, due 2026-11-19), regating-lead, 2026-11-17. Experiment E200. Code
`experiments/share_built_kind/` (`run.py`, `plot.py`), rules `results/predictions.json`, card `card.txt`, Slurm
`cluster/share_built_kind.sbatch`. Not gated: cluster `debug`, one L40, about 20 min, $0.*

We read the 67M's re-gating share on ordinary fine-tunes whose kind of change we know because we built it: E139's
imitators of E137's hand-made changes. Always on imitates a fixed push (built re-weighting); Table and Cue imitate a new
map (built rewiring). The user expects Always on to read a higher share than Table and than Cue.

## The request in the Terms' words

For each saved E139 fine-tune, and for Cue fine-tuned again now, the share of the fine-tune's gain that appears when W0's
subcomponents are kept and only Γ's gates change on the fine-tuned activations (the do-nothing route split's r, as on the
scorecard). Predictions: Always on's share is higher than Table's and higher than Cue's. No reuse verdict; the frozen
reuse definition (`decisions.md` rows 81 and 87) is not read.

## Design check (`/design`)

- **Goal.** `GOALS.md` part 2: is the share a method that reads what kind of change an update is? This scores it against
  part 1's built kinds (rung L2.2). It fills open cell P2.33.
- **Nearest tried.** E139 (`TRIED.md` row E139) read these states with the second wording's G and H, not the share, and
  stopped Cue at step 10 on a wrongly directed check. The scorecard (P2.2, P2.11) read the share on trained task pairs
  labelled by their task, never on fine-tunes whose mechanism was built. New here: the share against a built mechanism.
- **Patch or step.** A step: the first label set for the share that is not the task's own description.
- **Decisions.** Relies on the scorecard's share and floor (gain ≥ 1 log-odds). Two choices of mine, both forced by E139's
  states and flagged: (a) the states change every weight, so the fine-tune is loaded whole for T1 and Γ's gates while the
  masked forward keeps all of W0's weights (for a change to the 24 decomposed matrices only, this is route_split's r);
  (b) E139's changes have no single target token, so each window's target is the token the hand-made change F makes
  most likely, on the first 2048 held-out windows where F's top token differs from M's.
- **Definition.** No verdict, no clause added; the share is the method being scored.
- **What would change what we do.** "Higher" on both: the share tracks built kind on fine-tunes, and the 67M scorecard gains
  a built-mechanism label set. "Within the bar" or "lower": the share does not read the built kind through these
  imitators; the card's stop ends their use as labels for the share.
- **Apparatus check, first and alone.** The unchanged 67M as a fine-tune at zero steps, on each pair's windows: gain,
  L01 − L00 and the largest gate change must all read 0 (within 1e-6). It fails if loading or restoring the whole state
  leaks, or the gates are not reproducible. A miss is exit 1 before any state is read.
- **Early guard (Cue).** `VR.finetune`'s step-10 check, now one clause: KL(F‖ft) below KL(F‖M). E139's same seeds read
  0.0176 to 0.0181 at step 10 against 0.0378 at the start; each run's step-10 value is logged beside E139's. The wrong
  clause (KL(M‖ft) against KL(F‖M)) was deleted from the shared function, not mended; no saved run had stopped on it.
- **Saved for audit.** Per state: the four cells, gain, max gate change, and ‖δ‖ split into decomposed matrices and the
  rest (embeddings, norms, unembedding), which says whether a low share comes from a change outside the decomposition.
  Cue's states in `results/large/` (fp32, 268 MB each) and their training records.

## What runs (one job, stages cheapest first, each written as it lands)

| Stage | What | Expected |
|---|---|---|
| 0 | identity on the four pairs' windows | gain, L01 − L00, max gate change all 0 (stop otherwise) |
| 1 | 12 saved states: Always on ×3 (300 steps), amendment 2's Always on ×3 (3000 steps), Table ×3, XOR ×3 | Always on 0.3 to 0.8 under rival A; Table 0.1 to 0.45 (the trained pairs' new-side range); XOR gain under 1, r nan |
| 2 | Cue ×3, E139's recipe and seeds, the fixed check; saved; read | step-10 KL(F‖ft) ≈ 0.018; r as Table's under rival A |
| 3 | the filed rules | |

E137's hand-made changes are forward hooks, not weight changes: "does not apply". XOR is drawn, in no prediction.

## Filed predictions (`results/predictions.json`)

Spread 0.0405: the largest seed sd of the 67M share over the 26 sides of the scorecard's 13 pairs
(`experiments/method_table/results/final.json`, jq `.cells[] | select(.method=="share") | .reuse, .new`; the largest
over built sides is 0.024). Always on's finite shares pooled over both recipes against each rewiring group's;
d = difference of means; bar = 2 × 0.0405 × √(1/n_a + 1/n_b) (0.057 at 6 against 3). "Higher" if d ≥ bar (pass), "lower"
if d ≤ −bar, else "within the bar" (both a miss); "unreadable" if a side has fewer than 2 finite shares.

- **P1.** Always on reads a higher share than Table.
- **P2.** Always on reads a higher share than Cue.

Edge cases, run on the CPU worker before filing (`verdicts`): empty, one-a-side and all-nan pools read "unreadable", so
the do-nothing state (gain 0, r nan) cannot pass; a tie at the bar reads "higher" after rounding both to 9 decimals (the
first try read "within the bar" from float rounding, fixed); 1e-6 under reads "within the bar"; a reversal reads "lower";
a missing Cue group leaves P2 "unreadable". The card (`card.txt`) passes at n = 3 a side: smallest gap 0.10 is 3.0
standard errors.

## Figure

`plot.py`, through `commute.figures.dots`: share per state, one row per built kind (Always on 300 and 3000 steps, Table,
Cue, XOR), with the trained built pairs' ranges shaded behind (reuse sides 0.76 to 1.09, new sides 0.12 to 0.42, from
`final.json`, jq `.cells[] | select(.method=="share" and .group=="built") | .reuse, .new`). Sketch `figures/sketch.png`.

## Cost, time, route

Cluster `debug`, one L40, $0: rows and windows about 5 min, 12 reads about 4, three 300-step fine-tunes about 6, three
reads about 1; about 20 min, limit 1 h. Submit from the clone root:
`sbatch --comment='E200 share built kind' cluster/share_built_kind.sbatch`.

## Not run

A positive control that reproduces a scorecard state's saved share through this whole-state loading path (the identity
checks the path at zero change only); the share on E137's hand-made changes (no weights); the KL form of E57's share,
which has no 67M code and is not the scorecard's unit.
