# Decision: the ranking test of the fourth wording stopped on its controls. Close it, or fix the search and rerun?

*2026-11-13, definition-lead. Log card `decision-e182-controls-stop`. Results: `experiments/simplest_intervention/results/run1.json`,
`l1_neuron.json`, `l2_triggered.json`; run log `~/commute-data/scratch/definition-lead/e182-run1/run1.log`. The fourth
wording is still tentative: it is tested, not adopted (`reports/decisions/2026-11-13_decision_fourth-wording-simplest-intervention.md`).*

## The question

The ranking test of the fourth wording (experiment E182, the cell's second of four runs) is meant to grade built edits and
real fine-tunes by the shortest description of the change. Its first stage builds four controls with known answers. The
filed rule says: if any control does not read its built grade, nothing further runs and the cell closes "no call". One of
the controls missed. Do we close the cell as filed, or find out why and rerun the controls?

## What ran and what it read

One run on an H100, about 8 minutes (476 s in the log), $1.32. Three of the four controls ran before the stop
(do-nothing, control 1, control 2).

- **Do-nothing** (no change): read as built, rung 0 at all three bars (0.8, 0.9, 0.95), cost 0.
- **Control 1, one neuron shifted** (mlp2#36; the fine-tune differs from the original by 0.2048 nats of KL): read as built.
  Rung 1, one name, the right neuron, and that one edit closes 0.9992 of the gap (bar 0.9).
- **Control 2, a neuron (mlp2#1384) shifted by the value of principal direction 0 at the first layer input** (KL 0.2662):
  missed. This edit is exactly one pair on the list, so the filed reading is rung 2 with at most 2 names. On every list
  (neurons, directions, all) the rung-1 and rung-2 greedy searches took no step: zero names kept, and the saved curves
  are empty (`l2_triggered.json`, `lists.all["1"].curve` has length 0). The reading fell to rung 3, one free rank-one edit
  at the first-layer input to the third-layer input, which closes 0.915 of the gap (bar 0.9). At bar 0.95 it reads rung 4.
  The two-edit version closes 0.928. Log line: `RUN 1 STOP`.
- **Random-bit control**: never ran.

The search works on a one-neuron edit (control 1) but takes no step on a trigger edit. A CPU toy check of the same code,
`tiny_check2.py`, is reported to have found the built pair on a small random network (closure 0.988); I could not re-run it
because the compute node is down, so that number is not re-verified here.

## Why it matters

A search that cannot take one step on an edit that is exactly one pair on its list looks like a defect in the search on
the real model, not evidence that the ranking cannot be measured. But the cause is not known, and the filed rule says to
close. Closing on a defect we suspect would report "could not be measured" for a reason that may not be about the wording.

## Options

- **(a) Close the cell "no call", as filed.** No more spend; the cell's remaining two runs go unused. The gate page then
  says the ranking of the fourth wording could not be measured by this search, with the reason as read above. Every
  item still unrun (the third wording's edits, the memorised fact and its twin, the old matched pairs) stays unrun.
- **(b) Find why the search took no step on control 2, then rerun the four controls alone, same filed rule.** Reading
  the code and the saved record costs $0. The rerun is the cell's third run (about 10 minutes, about $2). If the controls
  then pass, the cell has one run left, so the remaining items would go in one run of several parallel pods (about $30),
  unless you add a run. If they miss again, the cell closes "no call" with a known cause or none.
- **(c) As (b), and add one run to the cell**, so the plan's two remaining runs keep their split (controls, then the
  items) with a fourth run still in hand.

## Recommendation: (b)

The control that passed and the toy check both say the code can find a built pair, so the miss is more likely a defect
than a property of the wording. Fixing it costs about $2 and ten minutes, and the filed rule stays unchanged. If it
passes, one pod run covers the rest; if you want the items split over two runs, that is (c), and it is your call because
the run count is the cap.

Names in the reports: the ranking test is experiment E182, cell P1.20; its card is
`experiments/simplest_intervention/card.md`.
