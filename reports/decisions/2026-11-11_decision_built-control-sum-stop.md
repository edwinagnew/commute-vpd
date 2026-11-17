# Decision: the sum-parity built control stopped before any fine-tune ran (P1.10, rung L1.4)

**Question.** The second built control, a 67M fine-tuned to follow the parity of the sum of two numbers, stopped at its
first check. What should rung L1.4 do next?

**What happened.** The run first checks that the plain 67M does not already hold the variable. A linear read of "is a
plus b odd", taken at the last word on number pairs it was not fitted on, scored 0.61. The filed bar was 0.55, and
shuffled labels scored 0.50. Each number's own parity reads perfectly (1.0), so the model holds both parts exactly and
their combination weakly. By the filed rule the task is not a clean built case for this model, and nothing else ran:
neither the "last number is odd" control, nor the plain or taught fine-tunes. It took 8.5 minutes on the cluster's free
queue, and the cell's one run is spent (closed).

One side reading bears on the earlier XOR control (E148). E148's own fine-tune of the plain 67M, the one the search
scored 0.08 on, holds XOR2 almost perfectly by a linear read (0.9997). So a full-strength fine-tune does teach such a
variable linearly. E148's taught model held it only weakly (0.588), because it was trained on a third of the push.

The automatic premise check said "doubtful". A 0.55 bar sits close to what combinations of variables the model already
holds seem to read, so tasks fall either side of it by small margins.

**Options.**
- **(a) Rerun E150 with the plain model's bar at 0.9 (recommended).** 0.9 is the same bar the run uses to say a
  taught model holds a variable, so the plain model counts as lacking it at 0.61. One free debug job, about 2 h. This
  runs the "last number is odd" control (the clean positive control nothing has run yet) and the built arm. It tells
  us whether the search finds the map when the original holds the variable, and fails to when the original holds it
  only weakly. The 0.61 is reported beside every reading of the plain arm.
- **(b) Test the search on E148's strongly taught model.** Use E148's plain XOR2 fine-tune (read 0.9997) as the
  original, and set its target from itself. One free debug job, about 1.5 h. This answers E148's open question (does
  the search fail only on weak variables?), but it has no built arm.
- **(c) Close L1.4 and write the gate page without a built control.** The page would say that two attempts found no
  clean built case on the 67M: one had a weak positive control, and the other's plain model already partly held the
  variable.

**Recommendation: (a).** It is the only option that runs both a positive control and a built arm. Changing a filed
bar after a stop is your call, so nothing runs until you choose.

**Evidence.** `experiments/built_control_sum/results/stage0.json` (the reads), `experiments/built_control_sum/results/predictions.json`
(the filed bar), the log `cluster/logs/commute-e150-58711.out`, and the plan `reports/plans/2026-11-11_plan_built-control-sum.md`.
