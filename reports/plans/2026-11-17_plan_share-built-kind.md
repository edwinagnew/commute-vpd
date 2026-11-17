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
- **Positive control, second and alone** (added 2026-11-17 at the coordinator's request). E46's `ident32_s0`, the
  scorecard's e46_copy_random reuse side at seed 0, is built as a whole state W0 + δW and read through the same loading
  path, on its own held-out windows and key table. It must reproduce its saved share 0.80244 within 0.001. The saved
  value is from `experiments/method_table/results/final.json`, jq
  `.cells[] | select(.method=="share" and .pair=="e46_copy_random") | .reuse[0]`. Floor: E136's stage 0 re-read this
  state through route_split with difference 0. Its largest difference between two code paths reading one saved state
  was 1.37e-4 (`experiments/share_untrained/results/stage0.json`, `e_rows`). The 0.001 tolerance is about 7 times that.
  A miss is exit 1 before any E139 state.
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
| 1 | positive control: E46 ident32_s0 as a whole state | r = 0.80244 within 0.001, gain 13.99 (stop otherwise) |
| 2 | 12 saved states: Always on ×3 (300 steps), amendment 2's Always on ×3 (3000 steps), Table ×3, XOR ×3 | Always on 0.3 to 0.8 under rival A; Table 0.1 to 0.45 (the trained pairs' new-side range); XOR gain under 1, r nan |
| 3 | Cue ×3, E139's recipe and seeds, the fixed check; saved; read | step-10 KL(F‖ft) ≈ 0.018; r as Table's under rival A |
| 4 | the filed rules | |

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

Cluster `debug`, one L40, $0: rows and windows about 5 min, the positive control under 1 min, 12 reads about 4, three 300-step fine-tunes about 6, three
reads about 1; about 20 min, limit 1 h. Submit from the clone root:
`sbatch --comment='E200 share built kind' cluster/share_built_kind.sbatch`.

## Not run

A positive control on a fine-tune that changed every weight (none of the scorecard's states did; stage 1 checks the
path on a change to the decomposed matrices only); the share on E137's hand-made changes (no weights); the KL form of E57's share,
which has no 67M code and is not the scorecard's unit.

## Second run (E202, 2026-11-17, the user's decision)

E200 (job 60937) passed both controls, but Always on, Cue and XOR moved their target by only 0.15 to 0.38 log-odds on its
2048 changed windows, under the 1 log-odds floor, so P1 and P2 read "unreadable"; Table read 0.26. The second run changes
only the windows: per pair, the first 2048 of the read half where F's top token differs from M's and F raises that token's
log-odds by at least 1 over M's (the log-odds of the share's T cells), chosen from F and M alone before any fine-tune is
read, scanning at most the whole read half (100,000 windows; fewer qualifying means all are read, counts recorded). The 15
states (E139's 12, E200's three Cue states) are then read exactly as in E200, with the same floor, target rule, P1 and P2
rule, spread 0.0405 and bar; identity and the positive control run first and alone with the same stops, and the job stops
if Always on, Cue or Table has under 32 such windows (`run2.py`, `card_run2.txt`, `results/run2/predictions.json`,
`cluster/share_built_kind_run2.sbatch`, debug QoS, about 10 min, $0). Expected: Table still about 0.26, since most of its
windows already qualified; the question is whether Always on and Cue now clear the 1 log-odds floor on windows where F
itself moves that much, and, if they do, where their shares sit against Table's.
