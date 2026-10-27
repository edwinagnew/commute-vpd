# Digest: regating-lead, E38 recruitment test (2026-10-27 afternoon)

**Why I finished.** The plan's decision tree for E38 has run to its end, and the next step waits on a choice only you
can make: the Blocking card "decision-e38-recruited-before" in the research log.

## What ran and what it found

We removed the recruited set from the fine-tuned model's real weights, with no Γ, to see whether the new behaviour
goes with it. The recruited set is the original subcomponents that Γ switches on at the trigger after the fine-tune.
On two of three tasks this removes a third to a half of the fine-tune's gain in the target's log-odds, far more than
random sets of the same size.

But removing the same set from the original model, which I predicted with high confidence would do nothing, already
removes a sixth to a quarter of that gain. These pieces push toward the target before any fine-tuning. So the test
cannot yet say how much of the fine-tune works through them. An independent check found no code error. The set was
chosen by a rise in importance with no bound on where it started, and the removal acted at every position.

The routing trace held on " which → was" and missed on " we → just". Its comparison with the null fine-tune was too
weak to count. The piece census had no power: its ranking rule puts almost every piece, recruited or random, in
"neither".

Details: the log's card e38, `reports/analyses/2026-10-27_analysis_recruitment-test.md`,
`notebooks/E38_recruitment_test.ipynb`.

## Pending decision

- **decision-e38-recruited-before (Blocking).** How to go on: (a) rerun with the missing controls, about $0.20; (b)
  accept that the switched-on pieces already carried the target and stop the line; (c) redefine the set as pieces
  nearly off before the fine-tune and rerun. I recommend (a).

## Money and problems

- RunPod: $0.23 of this session's $40 (two short RTX 4090 pods, both terminated). No Slurm jobs left.
- **Cluster GPU jobs are broken for every clone.** The shared venv `commute-data/venv-py313` has CPU-only torch since
  10:51 today, and its `commute.pth` points at the commute-lead clone. Job 38479 failed on it, so E38 moved to RunPod.
  I did not rebuild the venv, because it is shared and the next default `uv sync` would undo it. Both problems are
  under NEEDS YOU in `~/commute-agents/fleet/check`.
- The first stage-0 run stopped on a T1 tolerance with no basis: no earlier run had re-fine-tuned these states.
  Amendment 1 (`experiments/recruitment_test/results/predictions_amend1.json`) set it to 2% of E18's gain, and the
  rerun passed.
- `fleet/check`'s summary line for regating-lead showed hallucinated tool calls around 13:22. It looks like a fault in
  the fleet's summariser, not in this session.
