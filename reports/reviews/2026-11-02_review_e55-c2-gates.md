# Review: E55 C2 plan and code, gate 5 (two fresh Opus agents, 2026-11-02)

Two independent agents read the plan and the code for the 56 L40-hour VPD decomposition on the counting set before the
user's go-ahead. Neither found a crash in the untested stages on a CPU dry run with a stubbed reader, nor a blocker in the recipe or the saves.
They did find one design problem in what VPD reads (C1 below), and several ways to learn things sooner and to stop a run
that has gone wrong. Dispositions are recorded in the plan's section 7.

## Code review (scratch: `~/commute-data/scratch/reuse-lead/review-code/`)
- C1 (high, filed design). Gamma reads the label. `c2.py` read_one and `pcfg_vpd.windows` keep the answer at index 257,
  set to the task's label; the config's importance function is bidirectional, so the gates at `=` (index 256) depend on
  it. All four L cells move, T and U do not. On this truth set every P state is existing and every N state new, and the
  P answer (count(a)+1) never follows this prompt in LOW's rows, so a label effect on the gates lines up with the truth.
  No control sees it: the zero update has L01 = L00 whatever the leak. Same channel as
  `2026-10-27_review_gamma-bidirectional-windows.md`, without that review's "probably stands". Size unmeasured.
- C2 (high, cost). Nothing stops on a non-finite loss after step 50K; `decomp_train.pt` is then overwritten with the bad
  state and the job runs on. Fix: exit 1 before any save when `loss/total` is not finite.
- C3 (medium). The filed "U0 - T0 on the ID rows must pass the same 0.1" is only reported (`c2.py` positive control, `id_rows`).
- C4 (medium). A tie needs n = 21, so one scored state under matrix share 0.5 makes it unreachable; share unmeasured on
  the 1e-5 1K states. `matrix_share_source`'s "4 or more of one class" holds for new (6 in calibration), not existing (8 needs 6).
- C5 (low). `constants.reported_unscored` text: `a2/M/1000` and `c1b/N_1e-5_s1..s4/1000`; the code is right.
- C6 (low). `test_a_missing_positive_control_changes_no_outcome` cannot fail; call `c2.outcomes` twice instead.
- C7 (low). Eval rows restart on resume (`eval_iter` from i = 0), so a marginal early-check miss can clear on resubmission.
- $0 end-to-end possible now: a scratch driver patches `c2.FINAL` to `large/c2_rehearsal/A_final.pt`, outputs to scratch, runs
  `stage_read` and `stage_score` on `debug`, ~12 min; gives the real ID path, memory, timing, matrix share on all states.

## Plan review (scratch: `~/commute-data/scratch/reuse-lead/review-plan/`)
- P1. Matrix share can be measured now on all 43 states for $0 (target model only); the plan left it until after the spend.
- P2. Plan and filing disagree: plan says "void (no positive control)", the filing says a missing control changes no
  outcome; L00 - T0 is called a "coverage control" with a fallback but is filed as reported, no bar. A fail on a poor
  decomposition would be announced as the sceptic's result. Fix: "fail, apparatus unverified" unless the control holds.
- P3. First reading on a trained decomposition happens after the whole spend. Fix: after job 1 (~170K steps) read one P, one
  N and the zero update on `debug` from a copy of `decomp_train.pt`, reported only; tell the user before job 2 if U0 - T0 > 0.1
  or U1 - U0 is under the floor.
- P4. No check after 50K. Fix (data only): steps 100K, 150K, 200K, 300K with kl_ci <= 0.55 and kl_unmasked <= 1e-4; no L0
  min past 50K (the noise job's final-p train L0 read 23 and 30.55).
- P5. The queue estimate is not a forecast: `--test-only` returned now + 64 h whatever the walltime or GPU count; live state
  1 of 8 GPUs allocated, no pending jobs. Empty-queue condition met today by live state.
- P6. Each job runs the shared tree as it stands; a resume never compares config. Fix: store sha256 of the config and
  `decompose.py` and the git head in the save; refuse a resume on a mismatch.
- P7. Deadline: nothing lost at 1420 min (save, exit 3, ~170K steps per job). A node crash loses at most 5K steps. Exit 1 means
  both early miss and crash: the runner reads the log. Do not chain jobs with singleton. Write the cluster case into section 7.
- P8. A forced "fail": all scored N void gives n 12, agree 12, bar 13, outcome fail. Add an edge-case line and write the wrong-label count beside the outcome.
- P9. Same as C5. P10. Section 9 omits that C per rank comes from the 67M and the recipe from SS-2L, so the filed 67M reference
  numbers are from another recipe (importance coeff 2e-4 vs 1e-3, lr 5e-5 vs 3e-4, CI transformer 2048/8 vs 512/4).
- P11. Stale text: ~50 L40-h in `pcfg_c2.sbatch` and decisions.md; "rule relaxed" is done; Gate 1 lacks a line on formatting and lengths.

## Not run
Any GPU job. Both reviews ran on CPU with stubs.
