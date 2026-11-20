# E228 (cell P2.44, widened): the share in Qwen's KL form and in log-odds, on every built pair, twin and random change

The project's goal is a working definition of reuse and, on the 67M, a comparison of the re-gating share and simple
baselines against it (plan 2026-11-09, part 2). On Qwen the share, in E57's KL form, read random weight changes scaled
to a fine-tune's KL the same as the fine-tunes (P3.8: 0.992-1.009 against 0.990-0.9998). The user widened this cell
(task file 2026-11-20 12:24: "P2.44 widened: nine pairs and four twins in the Qwen KL form beside the log-odds form,
random changes matched on output change, one structured control. This morning's checks 1 and 3 fold into it"; pods).
The lead made the KL form the primary reading (2026-11-20, before any job). Card `experiments/random_kl_share/card.md`
(recarded); rules `results/predictions.json`.

## What runs
- **Items:** the 88 saved fine-tunes of E187's nine built pairs and the scorecard's four shuffled-label twins
  (e58_elicit_shuffle, e58_class_twin, e123_case_label, e125_repeat_label). These four are the paper scorecard's twin
  panel, and all their states are saved. Each fine-tune is read on its own E136/E147 read rows, fed whole as their
  source fed them (Gamma is bidirectional over the row).
- **Changes per fine-tune:** the fine-tune itself, on all rows and on two halves. Three Gaussian directions at its
  per-matrix norms, each scaled (E46's `perturb.match`, c ≤ 100, within 10%) so that KL(W0 + c R ‖ W0) matches the
  fine-tune's. The KL is the mean per row and position over every position up to the read position.
- **Structured control:** a Gaussian change of h.3.mlp.down_proj alone, at the fine-tune's total norm, with its part
  outside the alive span removed, scaled to the same KL. That matrix is the last decomposed one, so Gamma's gates cannot
  move and the re-gating numerator is exactly 0, while the split expresses the change. The KL-form share therefore
  reads 0 by construction: the one change of matched size that the rule must call "low".
- **Readings (one pass, `masked_reads.route_cells_by`):**
  - The KL-form share (primary): (K00 − K01)/(K00 − K11), E57's cells on the 67M, with K10 and K11 on the components
    carrying the change's least-squares split onto round 1's alive subcomponents. It is nan when K00 − K11 ≤ 0.
  - r (secondary): GOALS r at the read position, nan under a gain of 1.

## Numbers and what each should read
| number | do-nothing (zero change) | fine-tunes | random draws | structured control |
|---|---|---|---|---|
| K00−K0, K01−K0, K10−K00, K11−K01, KL, r's numerator and gain | exactly 0 | — | — | K01 − K00 = 0 (stop if not) |
| r | nan | saved share ± 0.01 (stop if not) | mostly nan (gain under 1; E211, E46) | 0 or nan |
| KL-form share | nan | finite, 0.3-1.0, no 67M prior | finite; lean: near the fine-tunes' (as Qwen) | 0 |

**Filed rules.**
- **Band:** per reading, the root mean square over all fine-tunes of their split-half gaps, measured in the same job.
- **Per fine-tune:** "refused" if fewer than 2 of 3 draws are matched and readable. Otherwise "low" if every draw is
  below its own share minus the band (strict), else "reaches".
- **K1 (primary, KL form):** "rival a: low" if no fine-tune reaches; "rival b: reaches" if more than half do;
  otherwise "mixed: n of m"; "silent" if all are refused. **P1:** the same on r.
- **Control check:** the structured control reads "low" on every fine-tune where it matched, else "miss". It is
  reported before K1.
- **Pairs (descriptive):** E51's pair rule on the fine-tunes' own readings, in both forms.

Edge cases run in the CPU dry run: empty and all-unreadable pools give refused; a tie at own − band reaches; an
unmatched draw is not readable; a zero change reads refused; a nan pair reads "not read".

**Blind to.** KL form: what the change does to the trained answer; a change mostly outside the alive span sits near 1.
r: anything gaining under 1 log-odds. The KL match fixes size, not direction.

## Design check
- **Goal:** whether the 67M share is evidence of reuse (part 2's premise), in the form that failed on Qwen.
- **Nearest tried:** P3.8 (E170/E174, Qwen KL form, random reads as real). E46 P0.3 and Amendments (67M KL-matched
  Gaussians, read through g, not a share). E211 and E161 (norm-matched Gaussians, r nan).
- **New here:** the KL form on the 67M, on all built pairs and twins, with matched random draws and a construction
  control.
- **Patch or step:** a premise check.
- **Decisions:** E46's match recipe; round 1's alive set; MIN_GAIN for r. No departure.
- **Definition:** unchanged; nothing is added to either share.
- **What would change our next step:** rival b means part 2 reports the 67M share as not evidence of reuse. Rival a
  means the 67M share passes the control that Qwen's failed.
- **Control first:** each pod job starts with the zero change (exact), then each fine-tune's r reproduces its saved
  share before its draws, and each structured control must leave the gates unmoved.
- **Saved:** results/states/<state>.json, holding every change's c, KL, five K cells, both shares and status;
  zero_<shard>.json; score.json.

## Platform, runtime, cost
RunPod H100. There are 3 pods, one shard each (`states i 3`): 30, 29 and 29 fine-tunes of about 5.6k rows. Each pod
has 2.6-2.7 GB of saved updates and windows (`run.py manifest 3`), plus artifacts/vpd (3.0 GB) and two Pile row files
(0.02 GB): about 5.7 GB, or about 10 min to sync at 10 MB/s.

Per-item time is estimated, not measured. Inputs to the estimate:
- E161: the log-odds share takes about 2 s per 64 rows on an L40.
- The KL pass is 7 CI or masked forwards against the share's 4, so about 3.5 s per 64 rows of 41 tokens on an L40,
  and about 1.8 s on an H100.
- A KL match is at most 52 plain-forward pairs.
- A split is about 3 s on the CPU (float64; unmeasured).

That gives about 30-80 s per fine-tune on 75-286 rows and about 7 min on each of E53's six 500-row ones. A pod then
takes about 40 min of compute plus 10 min of sync: about 1 h a pod and 3 H100-hours, roughly $9-10. The rehearsal
pod measures the real per-item times first: `states 0 88` (e62/copy_s0) and `states 14 88` (e53/lowercase_s0).

    PRED_SHA=<sha of results/predictions.json> python experiments/random_kl_share/run.py states <i> 3   (i = 0, 1, 2, one pod each)
    python experiments/random_kl_share/run.py score; python experiments/random_kl_share/plot.py   (no GPU)

## Not run
The read-position KL match (dropped with the widening); more than three draws; changes restricted to the alive span
other than the control; the 67M neuron and steering baselines on the random draws.

## Known-answer stage (check 1)
The user (2026-11-20 11:46, folded in at 12:24): "Check 1: the re-gating share and the two strongest baselines
(effective rank, neurons for half the gain) on the P1.37/E226 items: do-nothing control, E214 two built new edits, E195
four reuse items; forward passes only, rule filed first, same pass rule as R (reuse items above new)." E227 read them on
altered windows (cue items' answer appended, copy24 windows cut after position 20, the answer written at 21), which moves Γ's gates at the read position,
so its shares do not stand; its effective rank does (no rows) and is cited, not recomputed. Code
`experiments/random_kl_share/known_answer.py`, rules `results/predictions_known_answer.json`.

- **What runs:** one job; the do-nothing first (zero change: every K difference, r's numerator and gain, the KL, the
  neuron cells exactly 0; both shares nan; exit 1 if not), then new_few, new_thin, x2, x10, rebuilt, each on its E214
  windows fed whole, read with E228's pass (KL-form share and r) and E98's neurons for half the gain, on all rows and two
  halves. Moved appends neurons, so no reading applies (as in E227).
- **Targets:** the new edits' written answer at the last position. The reuse edits lower the copy answer (E227), so
  their target per row is the token whose log-probability the edit raises most at position 20.
- **Positive control:** the neuron reader runs no Γ and is causal, so on the new items (same target) it must reproduce
  E227's saved numerator and count; exit 1 if not.
- **Rule (as R):** per reading, "reuse above new" when every reuse reading exceeds every new reading by more than the
  band (largest split-half gap), else "mixed: n of 6 pairs ordered", "silent" when no pair is read.

Design check. **Goal:** part 2, whether the share or a baseline recovers the known labels the definition is calibrated
on. **Nearest tried:** E227 (P2.43; windows altered, shares invalid) and E226 (R on the same items). **Patch or step:**
a corrected read, not a new measurement. **Definition:** unchanged; the rule is R's. **Would change next step:** a
reading that puts reuse above new is named beside R in the paper's comparison; otherwise the paper says none recovers the
labels. **Control first:** the do-nothing, then the new items' neuron reproduction.
