# Plan: E98, a neuron analogue of VPD's r on the 67M

*2026-11-05, baseline-lead writer. Status: coded, not run. Advice adopted: `reports/plans/2026-11-05_advice_e98-neuron-r-FABLE.md`,
option A, with the lead's overrides (below). User's request: `decisions.md`, row "baseline-lead's new task" (2026-11-05).
Code: `experiments/neuron_r/run.py`, `src/commute/neuron_r.py`; rules `experiments/neuron_r/results/predictions.json`;
job `cluster/neuron_r.sbatch` (MATS `debug`, L40, $0).*

## The request in the Terms' words

The user asked for a baseline that does what VPD's readings do, on the 67M's MLP neurons and with no decomposition. In
`GOALS.md` Terms: VPD's r is the **re-gating** route of the route split, the frozen Γ switching the original
subcomponents on differently on the new activations, as a share of the fine-tune's gain. E98 builds its neuron
stand-in: the share of a fine-tune's log-odds gain carried when W0's MLP neurons fire differently, with W0's MLP
weights kept, against the share carried by new weight directions. It is read on the method table's 67M rows where
VPD's r has readings and scored against the construction labels (reuse vs new), which are assumed, not causal.

## Design check (`/design`; approved class: the user asked for this experiment by name, 2026-11-05)

- **Goal served.** The "If no simple Φ works" branch of `GOALS.md`: whether a fine-tune strengthens something that
  exists or teaches something new, and whether VPD's r reads that better than a decomposition-free method.
- **Nearest tried.** E52 and E56 (`TRIED.md`): the full-width neuron patch. It is this plan's r_n cell, and it tracked
  progress, not kind (0.69 of the matched cipher's gain at step 9, 0.96 at step 30; e46 reversed, e54_own none). New
  here: the cell with both MLP matrices kept (r_half), d_n, i_n, e_n, two built controls, and C3's two rows and E62's
  row, never patched.
- **Patch or step.** A step: the user's comparator, on the same rows and the same telescoping as VPD's r.
- **Decisions relied on.** Construction labels marked assumed (`table.json` label rule); log-odds as the route split's
  unit (2026-11-02 row); VPD's gain floor. No departure.
- **What changes next.** P1 failing (r_half separates where VPD's r does) would make a neuron baseline the comparator
  every Qwen reading needs. P1 holding, with P2, says the 67M's reuse gains are attention-borne, so Qwen's part 2 must
  add heads as units (Fable's option B). P3 decides whether the interpretation step is worth rehearsing on lowercase.
- **Control first.** Stage 0, alone, with stop rules (below).
- **Saved for audit.** Per state every cell's numerator, G, P(target) (`cells.json`); per scored state the grid and
  the top 256 of the neuron order with their scores (`carrying.json`). No state is written.

## Claim, null, cells

**Claim.** A neuron-level re-firing share does not tell the 67M's reuse fine-tunes from its new ones on the rows where
VPD's r does; the copy-type gains sit partly in attention.
**Null.** r_half separates reuse-higher wherever VPD's r does (then VPD adds nothing over neurons on this model).

On a W0 forward of every held-out read window, at every block and position, two switches: A, the firing `down_proj`
sees, and O, `down_proj`'s weights. The 67M's block is x + attn(rms_1 x), then x + down_proj(gelu(c_fc(rms_2 x))), no
biases (checkpoint read 2026-11-05). Readout: log-odds of the target at the answer position. With T0 = W0, T1 = the
state, G = T1 − T0, each share = mean numerator / mean G:

| cell | numerator | reads |
|---|---|---|
| **r_half (headline)** | L(O0, Ah) − T0, Ah = W0's c_fc and GELU on the state's MLP input | upstream re-firing, both MLP matrices W0's |
| r_n | L(O0, A1) − T0 | W0's output weights on the state's firing (E52/E56's patch) |
| d_n | L(O1, A0) − T0 | new output weights on W0's firing |
| i_n | L(O1, A1) − L(O0, A1) − L(O1, A0) + T0 | interaction |
| e_n | T1 − L(O1, A1) | attention's change, MLP outputs at the state's |

r_n + d_n + i_n + e_n = 1 by telescoping, as VPD's d, r, i, e_gate, e_delta (`local_faith.py:176-186`). r_half is the
closest analogue of VPD's r, which keeps the original subcomponents' weights: it is the headline (lead's override 1,
which settles Fable's point 9). r_n is reported beside it; r_n − r_half is what c_fc's own change adds.

## Rows (states checked on disk 2026-11-05: 67 files, every one with its source's sha256)

| row | reuse (assumed) | new (assumed) | VPD's r |
|---|---|---|---|
| e46_copy_random | ident32, ident32b s0-1 | rand32, rand32b s0-1 | 0.76-0.80 vs 0.27-0.29 |
| e53_lowercase_cipher | lowercase s0-1 | cipher, broad cipher s0-1 | 0.75-0.77 vs 0.12-0.17 |
| e54_matched | elicit s0-4 | cipher s0-4 at matched size | 0.76-0.77 vs 0.21-0.27 |
| e54_own | elicit s0-4 | cipher at its own step s0-1 | 0.76-0.77 vs 0.24-0.27 |
| open_pair_c3 | rank-1 s0-7 | cipher s0-7 | 0.84-0.90 vs 0.22-0.25 |
| open_pair_c3_copy | copy s0-7 | cipher s0-7 | 0.83-0.86 vs 0.22-0.25 |
| e62_copy_lookup | copy s0-2 | lookup s0-2 | unscored, 1.09-1.10 vs 0.38-0.42 |

56 states. Windows: all held-out read windows of each task, the ones VPD's r read (KV tasks; E53's 500 scored Pile
windows, next-token target; C3's trained-key held-out windows; E62's 200 read windows).

## Predictions (filed in `predictions.json`; numbers re-checked against their files)

A state is **scored** when |G| ≥ 1 log-odds (VPD's MIN_GAIN, `local_faith.py:187`) and, on key → value tasks, its mean
held-out P(target) ≥ 0.5 (`keydep.P_LEARN`). Rule per row: E51/E52's separation (`elicit.separation`: gap over the
larger within-kind range and ≥ 0.2), unreadable under two scored per side.

- **P1 (r_half is not a kind reader).** On e46, e53, e54_own, open_pair_c3, r_half separates reuse-higher on fewer than
  half of the readable rows (≥ 3 readable, else unreadable). Expected pass.
- **P1r.** The same on r_n. Expected pass; lean from E52/E56: e46 reversed, e54_own none.
- **P2 (copy-type gains partly in attention).** e_n separates reuse-higher on every readable row of e46, e54_own,
  open_pair_c3_copy, e62 (≥ 3 readable); a row with a scored |i_n| > 0.3 reads "cells unstable", no verdict.
- **P3 (go/no-go for the interpretation step).** k_half, the first k in 1, 2, 4 … 8192, 12288 whose patch of Ah on the
  top k neurons (A0 on the rest) reaches half the r_half numerator, neurons ranked once per state by W0's gradient ×
  (Ah − A0) summed over windows and positions: ≤ 32 on lowercase, > 64 on every other scored state. Undefined when
  r_half ≤ 0.1. E63 read ~15 on lowercase, ~110 broad cipher, ~240/470 on the KV pair with per-window ranks (E63
  analysis lines 191, 214).

| number | W0 | do-nothing (code_step10) | control cfc_attn | control down_only |
|---|---|---|---|---|
| G (log-odds) | 0 exactly | |G| < 1: no share | ≥ 1 (else stop) | ≥ 1 (else stop) |
| every cell numerator | 0 exactly | small, unscored | d_n, i_n = 0 (band) | e_n = 0, r_half = r_n (band) |
| r_half, r_n, d_n | undefined | undefined | r_n + e_n = 1 | d_n > r_n |

**Blind to.** r_half and r_n: where a gain runs, not whether it is new (progress, E56); relays (an MLP after a changed
head re-fires with nothing of its own changed, as VPD's r). r_n and r_half count as re-firing of existing neurons any
downstream re-firing caused by new write directions; the down_only control measures this at 0.425 of G on E54's elicit
task. e_n: which head or attention matrix. k_half: in-sample, first-order
ranking; sufficiency, not necessity. **Noise floor:** each row's within-kind seed range (the rule's spread); numerical
floor the W0 reading, measured first. **Positive controls:** the two built fine-tunes, which test what the cells read,
not kind.

**Edge cases run before filing** (CPU, `~/commute-data/scratch/baseline-lead/e98-writer/edge_cases.py`): empty and
one-state sides, a do-nothing state, all states filtered out, ties (gap 0 reads none), reversed rows, an unstable row,
two readable rows (unreadable), a lowercase k_half of 64, no lowercase k_half, a tie at half, a non-positive numerator.
Each read as intended. The do-nothing state is never scored, so it cannot pass P1's bar.

**Departures from the advice, with reasons.**
1. Its P(target) filter would empty E53 by construction (lowercase's mean P is ~0.006, T1 −5.08 log-odds in
   `performed_anchor/results/read.json`): E53's next-token tasks are filtered on the gain floor alone. On key → value
   rows it stands. It leaves open_pair_c3 and _copy unreadable (ciphers at held-out P 0.32-0.45 at the read step,
   `open_pair/results/run.json`, `open_pair_instruments/results/seeds.json`) and e54_matched (cipher P 0.23-0.31 at
   step 9, `keydependent_elicit/results/stage2.json`, `keydependent_patch/results/job2.json`). The score therefore also reports every row's verdicts unfiltered (gain floor only), unscored.
2. P1's "at most 2 of 5" becomes "fewer than half of the readable rows": e58_rank1_96 is not on disk.
3. The advice's do-nothing line is wrong: E52's code_step10 read |G| 0.0514 over the 0.05 floor, P5 "fails"
   (`baseline_neurons.json .score.P5`), and on code rows, not a task. E98 reads it on two KV tasks against the 1.0 floor.
4. `ctxpatch.py:138` is gradient × activation, not × the change; `neuron_r.ranking` multiplies by Ah − A0.

## Stages and stop rules (stage 0 runs first and alone; each stage writes its JSON as it finishes)

0. sha256 of all 67 files (stop on a mismatch). W0 on all 15 tasks: every cell equals T0 within 1e-6 (E63's
   identity tolerance; E63 measured 0); the band for exact zeros below = max(1e-6, 10 × the largest W0 deviation).
   E56 reproduced: r_n in log P on the scoring half of elicit_s0 and cipher_s0 within 1e-5 of `job1.json`'s full-width
   shares 0.93685 and 0.68531 (E63's path matched within 9.4e-8). code_step10 on elicit and ident32: stop if
   |G| ≥ 1. Controls: E54's elicit recipe (`keydep.finetune`: arm A, Adam 3e-5, batch 32, 300 steps, scored step by
   E12's rule, seed 0) with one change, `ArmAConfig.matrices`, so Adam and the gradient clip run over that subset
   only. cfc_attn (c_fc and attention, 16 matrices): stop unless |G| ≥ 1 and |d_n|, |i_n| numerators ≤ band.
   down_only (8 matrices): stop unless |G| ≥ 1, d_n > r_n, |e_n| ≤ band and |r_half − r_n| ≤ band.
1. Cells of the 56 states (`cells.json`); stop if a state's identity reading exceeds 1e-6 or the telescoping residual 1e-6.
2. Per scored state the order and grid (`carrying.json`); stop if the k = 12288 patch misses the r_half numerator by 1e-5.
3. Score (`score.json`): P1, P1r, P2, P3, then all five readings on all seven rows, filtered and unfiltered.

**Run time on an L40:** 5-10 min. E63 read 26 states with about 30 patched forwards per batch plus gradients in
2 min 12 s (job 48527, `reports/analyses/2026-11-02_analysis_e63-neuron-sparsity.md:335`); E98 has 56 states at 7
forwards, about 35 scored states at 18 forwards and one backward, and two 300-step fine-tunes. Slot 30 min, 16 GB host.

## Next (not in this job)

- The interpretation step (Fable point 7): on lowercase's k_half neurons if P3 holds, W0's top Pile windows and
  activations, ablation, a one-line job and a blind input test. A rehearsal of the Qwen procedure.
- Heads as units (Fable's option B) if e_n is where the copy gains sit.
- The Qwen plan, separate.

## Not run

- e58_rank1_96: only rank-1 seed 0 at step 30 is on disk; Amendment 4 saved no weights (`open_pair/amend4.py`).
- e58_elicit_shuffle: the shuffle's states were never saved.
- e58_copy_prior_cipher, e58_class_twin: windows only.
- e54_own's own-step cipher at seeds 2-4: read by E56, not saved.
- The interpretation step and heads as units (Next).

## Amendment 1 (2026-11-05): stage-0 controls
- **What stopped.** Job 52747 stopped at stage 0's control `down_only` (`results/stage0_job52747.json`): d_n 0.4218 vs
  r_n 0.4250, missing the filed "d_n > r_n". Every exact check held: e_n 0, r_half = r_n, G 6.94 (numerators d_n 2.93, r_n 2.95, i_n 1.06).
- **Why.** The rule was mis-specified. Training every block's down_proj changes the MLP input of every later block, so later
  neurons re-fire with W0's weights, and r_n counts that. "d_n > r_n" is not implied by how the control is built.
- **New rules** (`results/predictions_amend1.json`; `predictions.json` is not edited; the run refuses to start without both).
  `down_only`: readable, |e_n| and |r_half − r_n| numerators ≤ band, d_n > 0; r_n recorded, not stopped on.
  New `down_last` (same recipe, the last block's down_proj only, nothing downstream to re-fire): readable, |r_n|, |r_half|,
  |i_n|, |e_n| numerators ≤ band and |d_n − 1| ≤ band / G; unreadable (|G| < 1) stops the job, the control has no teeth.
  `cfc_attn` unchanged. A miss now logs which condition failed.
- **Blind to.** r_n and r_half count as re-firing of existing neurons any downstream re-firing caused by new write
  directions; the down_only control measures this at 0.425 of G on E54's elicit task.
