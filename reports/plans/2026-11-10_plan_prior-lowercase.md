# E145 plan: the original model's prior on lowercase against the ciphers

*2026-11-10, cell P2.13 (one run). Code `experiments/prior_lowercase/run.py`, rules `results/predictions.json`, job
`cluster/prior_lowercase.sbatch`. Sides labelled as E53 built them; no verdict of reused or new (cell P1.1).*

## Design check

- **Goal.** Part 2 of the plan: every baseline on every pair. E133 left the prior empty on "Lowercase vs ciphers" because E53's windows were not saved. This fills that one cell of the E133 comparison (`experiments/method_table/results/final.json`).
- **Nearest row.** `experiments/TRIED.md` E133 (prior per map, 24 maps) and E60 (rebuilt E53's windows by E53's own `Run.scored` and read them on the next-token path). Nothing new is defined: E133's quantity on E60's windows.
- **Patch or step.** A hole fill, inside the named cell.
- **Decisions.** None departed from; the prior is a reference, never a label.
- **Definition.** Does not apply: the rule reads which side's prior is higher.
- **What changes next.** The row gets its prior. If the lowercase prior sits well above the ciphers' (expected), the row joins the pairs where share and prior point the same way; if alike, it joins the prior-matched pairs, and the share's gap there is not the prior's.
- **Control first, alone.** (a) E133's own `forward_prior` re-reads e58_24_rank1 (5.6705e-3, a prior E133 computed by forward) within 1e-4 relative; fails on a changed forward or precision. (b) The rebuilt windows reproduce E53's saved W0 log-odds per task (-6.755, -10.142, -11.122) within 0.01 nats; fails on a different window draw (which would move the mean about 0.2). Stop on any miss.
- **Numbers.** Per task: log10 prior, its 95% interval over windows, window percentiles 10-90. The pair's gap (lowercase minus the ciphers' mean) with its interval and the filed reading. Expected values in `predictions.json` (`expected`): lowercase -3.6 to -2.93, cipher -4.6 to -4.40, broad -5.0 to -4.83, gap 1.1 to 1.7. The prior reads no update, so the do-nothing state and every seed read the same value.
- **Saved.** Per-window log10 P (1,500 numbers) in `stage1.json`, so the figure and any later spread are redrawn without a GPU.

## Per seed, or per task

The prior is per task, not per seed: E53's two seeds of each task differ only in batch order and are read on the same
500 windows, so "spread across seeds" is zero by construction. The spread given instead is across windows: the bootstrap
interval of the geometric mean and the windows' percentiles. That is E133's reader unchanged, with an interval added
around its value; no new definition. In the method table the cell enters as one value per task (lowercase against cipher
and broad cipher), so `plot_final.py` gives it "one value a side", as for the other pairs.

## Rule (filed)

`reuse_side_higher`: gap >= 1.0 decade and interval low end >= 0.5. `new_side_higher`: the mirror. `alike`: the whole
interval inside (-0.5, 0.5). Otherwise `no_call`. Band sources and edge cases (empty, -inf, ties, bounds, reversed sides,
do-nothing) in `predictions.json`; the edge cases ran on the code before filing (CPU dry run, scratch `p213-writer/dryrun.log`).

## What runs, cost

Windows cannot be read on the CPU worker (the hook refuses a model load there), so one MATS `debug` L40 job, $0: model
load, the control forward on e58_24_rank1's saved windows, then per task one full-batch forward of 500 windows for log P
and one for log-odds (E60 ran the same path in about 90 s, job 46258). `--mem=16G`, `--time=00:20:00`, expected under 5
minutes. The windows were rebuilt on the CPU worker first (500 x 127 per task, the next token at position 64 checked).
After the job: `plot.py` (CPU) draws `figures/prior.png`; `method_table/plot_final.py` gets the cell; one paragraph.

## Not run

No fine-tuned state, no share, no new windows. No spread across seeds (there is none to read).
