# Baseline digest, 2026-11-08: finished, waiting on new Qwen states

Baseline, day 4 of 13: rows 1/2/3 not moved since they were done on 11-06. The baseline line has finished its task:
the neuron and cheap-comparator readings sit beside every existing VPD row in one table, published in the research
log (section 7, anchor `neurontable`; source `experiments/neuron_vpd_table/results/full_table.md`). Five hourly checks
on 11-08 found no new fine-tuned Qwen state with a VPD row to put comparators beside, so the line stops instead of idling.

## Why finished
- Task steps 1-3 (67M panels, Qwen comparators, one table) done on 11-06; see
  `reports/digests/2026-11-06_digest_baseline-retarget.md` and
  `reports/analyses/2026-11-06_analysis_e117-e118-neurons-beside-vpd.md`.
- The only route to new work is another lead saving a fine-tuned state with a VPD row. The likely one is ask 38
  (qwen-decomp-lead: move D1/E111 to the new decomposition), which waits on the user.
- E131 (qwen-decomp-lead, 11-07) compares two decompositions of the unchanged model, not an update, so no comparator
  applies to it.

## Pending decisions for the user
- None owned by this line. All baseline asks are closed.
- Not owned by this line: ask 38 (qwen-decomp-lead) decides whether E111 produces new states for comparators.
- Whether to join the table into one verdict is a definition-level call (yours), unchanged since 11-06.

## Spend, jobs, friction
$0 on 11-07/08. No pods, no runners, no Slurm jobs. No new friction.

## To restart
`~/commute-agents/fleet/start baseline-lead "..."` naming the new states. The comparators run with E118's
`build.py` and `run.py --part` on the new files (handoff note, Next 2).
