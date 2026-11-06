# E123 appendix: apparatus, timing, changes from the draft

*Appendix to `2026-11-06_plan_e123-case-label-both-readings.md`, 2026-11-06.*

## Code

- `experiments/case_label_both/run.py`: the job. `experiments/case_label_both/rules.py`: the one config (`Cfg`, equal
  to the filed `config`) and the pure rules (copy, removal class, arm, side, joint, score).
- `src/commute/route_labels.py` (new): E119's `label` and `e12_control`, moved out of
  `experiments/teacher_route_split/run.py`, which stays as it ran with its own copy. Lines as a frozen dataclass.
- `src/commute/recruitment.py`: `window_logits`, `readout`, `removed`, `removed_at` gain a `ci_len` keyword, default
  None, so no earlier caller changes (all pass it by omission). With it the full model sees 41 tokens and the target is
  read from the 42nd column, as `local_faith.route_cells` already did for E62.
- Reused unchanged: `local_faith.route_cells`, `route_split`, `trigger_mean_gates`; `anchors.lift_context`, `lifted`;
  `teacher.teacher_net`, `train_student`; `component_training.alive_uv`, `components_with`; `arms.save_offsets`;
  `regating.Task`, `shift_of`; `recruit.switched`, `random_like`; `recruitment.removal_hooks`, `share`,
  `d_seed_class`; `signatures.ablated_weights`, `top_by`; `steering.Net` (E112's and E114's fp32 path).
- No model is loaded locally. The dry run (`~/commute-data/scratch/regating-lead/e123-writer/dry.py`, output
  `dry.out`) ran every rule's edge cases, the two no-model controls on their files, and opened every input.

## What Γ and the model see

The 42nd column (the label) is never fed: route split and removal readers take `ci_len = 41`; S is computed from the
41-token training windows; the students train on the 41-token training windows. One exception, log only: the
student's per-step held-out P(target) in `component_training.train` feeds the 42-token read windows (its
`eval_arm_a` has no `ci_len` there); the model is causal, so position 20 is unaffected, and the number is not scored.

## Inputs the job opens (each listed and opened in the dry run)

| file | bytes | check |
|---|---|---|
| `experiments/case_label/results/large/task.pt` | 4,023,686 | sha256 = E114's filed `80cdeee4…` |
| `experiments/case_label/results/large/dW_{real,twin}_s{0,1}.pt` | 56,631,029 each | sha256 = E112's `stage1.json` (all four match) |
| `experiments/case_label/results/stage0.json`, `stage1.json`, `stage2.json` | 747; 1,528; 4,489 | W₀ accuracy; dW sha256; gap gains |
| `experiments/case_locator/results/stage0.json` | 59,471 | log-odds gains |
| `experiments/transport_test/results/large/alive.pt` | 360,871 | E34's and E38's alive masks |
| `experiments/new_vs_existing/results/new_vs_existing_amend1.json` | 67,000 | route-rule control |
| `experiments/recruitment_test/results/amend2.json` | 115,066 | removal-rule control |

Plus the model checkpoint under `artifacts/` (as every 67M job). Read windows: 256 per arm, 42 columns; label A on
0.500 of the real windows and 0.508 of the twin's.

## Memory

Γ's gates kept by the route split: 41 × 38,912 values per window, about 6.4 MB at 4 bytes, so 1.6 GB per arm for 256
windows (two arms' W₀ bases held, plus one state's during a read: about 5 GB host). The removal's logits references:
256 × 41 × the vocabulary at 4 bytes, about 2.1 GB per reference on the card, two at once. `--mem 32G` as E119.

## Time per stage (L40), with sources

| stage | estimate | source |
|---|---|---|
| model, alive, least-squares bases, W₀ bases | ~1 min | E119 job log `run_r1_t534_v369.log`: start 15:42:44, first teacher read 15:43:47 including a 39 s fine-tune |
| 0: label reads (1,600 + 4 × 1,024 windows) | seconds | E112 `stage1.out`: four fine-tunes with their reads in 36 s (10:54:50 to 10:55:26) |
| 0: S per state (9,600 training windows) and C1-C4 (about 9 passes of 256 windows per state) | ~1-2 min | E38 A2 job 38805, stage 0 of nine states in 47 s on about 100 windows (`reports/analyses/2026-10-27_analysis_e38-amend2.md`, Compute) |
| 1: do-nothing and least-squares reads | ~18 s per read, 8 reads, ~2.5 min | E119: 7 s per read on 100 windows (log 15:43:47 to 15:43:54), scaled to 256 |
| 2: removal, 42 passes per state | ~1.5 min | E38 A2 job 38809: 552 passes on about 100 windows plus the census in 83 s |
| 3: student, 1,000 steps + reads | ~150 s each, ~10 min | E119's arm-A students on a MATS L40: 109-135 s (`teacher_route_split_<task>.json` students.*_A.seconds), plus the larger read |
| total | ~17 min | one debug job; `budget_s` 1,500 s exits 3 before a student that would end later |

## Resume

`results/case_label_both.json` is written after every record. A rerun skips stage 0 once passed, each fine-tune's
route record, each removal record, and each student with both a record and its file in `results/large/students/`.
A `stopped` file refuses to start. `fleet/resubmit --code 3` handles the exit-3 chain.

## Changes from Fable's draft (each number checked against its file)

1. **Gain check units.** The draft checked the route split's gain against E112's 12.83 / 12.04 / 10.23 / 8.80 nats.
   Those are label-gap gains (logit A − logit B toward the own label) on the fit windows, not log-odds gains on
   held-out windows. Stage 0 now checks both units on the fit windows against their sources: E112's gap gains and
   E114's log-odds gains (23.29 / 24.52 / 23.63 / 22.45), with E114's tolerance 0.02 (the draft's was 0.05).
2. **Windows.** The draft left open fit or held-out windows. Fit windows, as E114: the twin's held-out accuracy is at
   chance, so its "gain" there is toward an arbitrary label.
3. **Copy bar on fit windows.** The draft's "held-out label accuracy ≥ 0.90" would make every twin copy unreadable
   by construction (twin held-out accuracy 0.51). E112's learn bar is on fit-window accuracy (`acc_fit`), so the copy
   bar reads there.
4. **No 42-token forwards.** The draft fed 42 columns to readers without `ci_len` and relied on causality, checked by
   W₀'s accuracy. The readers now take `ci_len`, so no stage depends on it; W₀'s accuracy stays as a check.
5. **Order and jobs.** Removal (about 1.5 min) runs before the students (about 10 min), cheapest first, in one job
   with exit 3 instead of two jobs.
6. **C4 kept.** The draft listed E38 A2's C1-C3; C4 (the busy set bites) is the removal's positive control.
7. **Arms.** E119's majority rule needs three readable states, so on two seeds it reads "unscorable"; each arm uses
   E38 A2's two-seed task rule instead, for all three measures.
8. **Numbers corrected.** E38 A2's D(S) on its two "through" tasks is +0.116 to +0.255, with random maxima up to
   +0.154 (the draft: +0.12 to +0.26, ≤ 0.15). E112's possession line is at line 78 of its analysis, not 85.
9. **Arm B and the fresh pair** are out (arm B not planned; the fresh pair is E125).
