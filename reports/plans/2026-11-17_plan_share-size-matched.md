# P2.34: does the re-gating share follow the kind of change or its size?

*Cell `P2.34` (part 2, rung L2.2, 2 runs, due 2026-11-19), regating-lead, 2026-11-17. Experiment E204. Code
`experiments/share_size_matched/` (`run.py`, `plot.py`), rules `results/predictions.json`, card `card.txt`, sketch
`figures/sketch.png`, Slurm `cluster/share_size_matched.sbatch`. Not gated: cluster `debug`, one L40, about 15 min, $0.*

In E202 (P2.33) the share put Table's fine-tunes at 0.26 and every other group at 1.1 to 1.2, but Table was also the only
group with a large gain (19 log-odds against 1.1 to 1.8). This run builds a re-weighting fine-tune as large as Table: E137's
Always on push, scaled up until its gain matches Table's, imitated by fine-tunes with E139's recipe, and read exactly as
E202 read its states.

## The request in the Terms' words

The user (2026-11-17): "run the size-matched re-weighting fine-tune beside Table". F_big is E137's Always on edit (`VR.piece_edit`:
Q b added at block 2's input, last position, f = 1) with b multiplied by one factor alpha. Three ordinary all-weights
fine-tunes of the 67M imitate F_big (E139's recipe, `VR.finetune`, 300 steps, Always on's batch draws). For each, the
do-nothing route split's r, the share of the fine-tune's gain that appears when W0's subcomponents are kept and only Γ's
gates change, on E202's windows chosen from F_big and M. No reuse verdict; rows 81 and 87 of `decisions.md` are not read.

## Design check (`/design`)

- **Goal.** `GOALS.md` part 2, rung L2.2: does the share read the kind of change an update is? It fills open cell P2.34.
- **Nearest tried.** E202 (`TRIED.md` row E200/E202): the share split Table from the rest, confounded with gain size. New
  here: a built re-weighting group at Table's size, which is the only way to tell the two readings apart with these edits.
- **Patch or step.** A step: it answers the confound E202's analysis names in "What next".
- **Decisions.** Relies on E200's two flagged choices (the fine-tune loaded whole for T1 and Γ's gates; the target is the
  edit's top token) and on E202's window rule, all unchanged. New choice: alpha, set from F_big and M alone.
- **Definition.** No verdict and no clause added; the share is the method being scored.
- **What would change what we do.** Kind (matched Always on at least the bar above Table): the share tracks built kind once
  size is held, and E202's Always on reading stands as a label. Size (within the bar or below): the share's one split in
  E202 is gain size, and the share is dropped as a built-kind label on the 67M.
- **Apparatus checks, first and alone.** Stage 0: the unchanged 67M as a fine-tune at zero steps on alpha 1's windows; gain,
  L01 − L00 and the largest gate change 0 within 1e-6 (fails if loading or restoring leaks). Stage 1: E200's positive
  control unchanged, E46's `ident32_s0` read as a whole state, r 0.80244 within 0.001 (E200 and E202 read a difference of
  0.0). Stage 2's first point is a third check: alpha 1 is E202's edit, so its windows must repeat E202's (1620 windows, F
  gain 1.8337, within 16 windows and 0.01).
- **Early guard.** `VR.finetune`'s step-10 check, KL(F_big‖ft) below KL(F_big‖M), logged with both values. A miss stops the job.
- **Saved.** The three states (fp32, `results/large/`, `experiments/SAVED.md`), training records, F_big's windows, the full
  alpha scan, per state the four cells, gain, largest gate change, ‖δ‖ inside and outside the decomposed matrices, and
  E139's leftover distances.

## Choosing alpha

The target is Table's F mean gain on its kept windows, 19.1315 (`experiments/share_built_kind/results/run2/stage2.json`,
jq `.pairs.Table.F_gain_mean`; the run reads it). For each alpha the run applies E202's rule (`run2.big_gain_rows`) to F_big
and M on the Always on read half and records F_big's mean gain on the kept windows. It tries alpha 1, 2, 4, ..., 256 until the
gain reaches 19.13, then bisects in log alpha (at most 12 steps) until within 0.2. No crossing up to 256 stops the job
(the push saturates); a best alpha more than 1.0 off stops it too. No fine-tune is read during the scan. ‖b‖ is 2.09
(E137 `stage0.json`, `.pieces.always_on.b_norm`); the crossing is expected between 4 and 64, not measured.

## Does scaling keep the kind?

Yes by construction: F_big still adds one fixed vector, the same for every input, at one site. It computes no new function of
the input, so it stays built re-weighting. Two things could make its output stop looking like a larger copy of Always on:

- **Saturation.** When the push dominates the stream, later layer norms see mostly the push and the output tends to one
  distribution for every window. Checked by the share of F_big's kept windows held by the most common target: over
  max(0.5, alpha 1's share) stops the job. The ratio of ‖alpha Q b‖ to M's stream norm at block 2's input is logged beside it.
- **Unrelated top tokens.** The big push might make tokens top that the unit push does not favour at all. Checked by the
  share of F_big's targets whose log-odds the unit push also raises over M; under 0.5 stops the job (at alpha 1 it is 1 by
  construction).

## What runs (one job, cheapest first, each stage written as it lands)

| Stage | What | Expected |
|---|---|---|
| 0 | identity on alpha 1's windows | gain, L01 − L00, gate change 0 (stop otherwise) |
| 1 | positive control `ident32_s0` | r 0.80244 within 0.001 (stop otherwise) |
| 2 | alpha scan, F_big against M only | alpha 1 repeats E202; alpha* between 4 and 64; kind checks pass |
| 3 | per seed: fine-tune, save, leftover, read | step 10 below start KL; gain near 19, at least 9.56; leftover TV(ft, F_big) about 0.06-0.09 (E139: Always on 0.057, Table 0.088) |
| 4 | the filed rule | |

**Reach stop.** A seed whose gain on F_big's windows is under half of Table's fine-tunes' mean gain (0.5 × 19.12 = 9.56) stops
the job: the recipe cannot reach the large change, reported as such, no prediction read. A factor of 2 is about a quarter of
the log distance between E202's small groups and Table.

**Table side.** E202's three Table readings are reused (`results/run2/stage3.json`), not read again: the code path, the windows
(Table's F is unchanged) and the states are identical, and E202's controls passed in the same job.

## Filed predictions (`results/predictions.json`)

Spread 0.0405 (largest seed sd of the 67M share over the scorecard's 26 sides, `experiments/method_table/results/final.json`,
recomputed 0.040479). Bar = 2 × 0.0405 × √(1/3 + 1/3) = 0.0661. Rule: E200's `verdicts`, unchanged; d = mean(matched Always on)
− mean(Table, 0.2597).

- **K (kind).** The matched Always on reads above Table by at least the bar (mean ≥ 0.3258). Expected about 1.1 if true.
- **S (size).** It reads within the bar of Table or below (mean < 0.3258). Expected about 0.26 if true.

One call decides both; "unreadable" (fewer than 2 finite shares a side) is neither. Edge cases, run through `rivals` on the CPU
worker before filing: the do-nothing state (r nan on every seed) and one finite share read "unreadable"; Table's own shares
read "within the bar" (S); E202's small Always on shares read "higher" (K); d exactly at the bar reads "higher", 1e-6 under
"within the bar"; a reversal reads "lower" (S). The do-nothing state also meets the reach stop first.

**Gain mismatch.** The fine-tunes' gain will not equal Table's exactly. The rule is read only when every seed is over 9.56;
the report gives each seed's gain beside its share, and the figure plots share against gain for E202's 15 states and the
3 new ones, so a share that slides with gain is visible apart from the call.

## Figure

`plot.py`, through `commute.figures.save`: share against the fine-tune's gain (log scale), colour by built kind (re-weighting
blue, rewiring orange, XOR grey), marker by group, Table's mean ± the bar shaded, share 1 dashed. Sketch `figures/sketch.png`.

## Cost, time, route

Cluster `debug`, one L40, $0, about 15 min (E202: rows and controls 81 s; E200: a 300-step fine-tune about 80 s; the scan
up to ~15 points at ≤ 15 s; distances and a read about 30 s per seed); limit 45 min, 24G host memory (E200 peaked at
14.8 GB). Submit from the clone root: `sbatch --comment='E204 share size matched' cluster/share_size_matched.sbatch`.

## Not run

A size-matched rewiring group below Table's size (a scaled-down Table); a fine-tune of the decomposed matrices only; an
independent check; the share on the hand-made edits themselves (hooks, not weights).
