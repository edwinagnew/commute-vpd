# P2.76: Figure 5 and the difficulty figure on equal footing

P2.76, due 11-24. Descriptive, $0, CPU only (user 2026-11-24 via whip-lead: "experiemental rigour is essential!!!!").

Two results-section figures compared readings that were not taken on the same tasks or at the same kind of trained
state, so we re-scored both from saved results. On the tasks every reading shares, no reading of Figure 5 follows the
raters' tiers at p < 0.05, so the body's claim that effective rank alone does so falls. In the difficulty figure the
old tasks cannot be matched to the realistic ones on distance from the base model without retraining, so the gap is
stated as a mismatch, not a matched ratio. This narrows what the paper can say about which reading tracks reuse on
realistic fine-tunes.

## Figure 5: effective rank on the shares' eight tasks

The re-gating and steering shares can be read on 8 of the 12 tasks (four gained under one log-odds); effective rank
and the base prior were scored on all 12. Scored on the 8 with the filed statistics unchanged (`score.hope1`: Kendall
tau-b with the consensus tier, partial tau given the prior, 10,000 task permutations):

| effective rank, consensus tier | 12 tasks | 8 tasks |
|---|---|---|
| tau with the tier | 0.355 | 0.189 |
| partial tau, prior held out | 0.433 (p 0.032) | 0.227 (p 0.236) |
| prior's own tau with the tier | 0.254 | 0.340 |

The body sentence "the only reading to reach p < 0.05 once the prior is partialled out" does not hold at equal n. The
replacement sentence went to writeup-lead after redteam-lead's OK. Source: `experiments/convergence_panel/equal_n.py`,
`results/wide/equal_n.json`.

## Difficulty figure: gain at matched states

- **No KL match without retraining.** Of the 36 old states, one (many_one) falls in the realistic panel's KL window
  (0.181 plus or minus 10%), at gain 6.40; lowercase and many_many sit above it, the other 33 below. The old median KL
  is 0.047, at 10 steps with a KL penalty, and no old state saved a per-step KL curve.
- **The reverse match does not narrow the gap.** At the old median KL, only 3 of 12 realistic tasks have a save that
  passes the panel's gain floor, with a median held gain of 0.62 against 1.57 at their own matched pick (held gain;
  the figure's 1.36 is the read-set gain of the same picks). With n = 3 this is weak, but it points the same way;
  the 13.0 against 1.36 ratio is kept only as a stated mismatch in the caption.
- **Steps-to-half is dropped.** Its two definitions use different step grids, one exists for 15 of 36 old states only,
  and both sit at one or two steps, the grids' resolution.

Source: `experiments/old_tasks_read/matched_gain.py`, `results/matched_gain.json`.

## Not run

- A retrain of the old tasks logging KL per step, without the KL penalty, to reach the realistic window.
- Per-rater partials on the 8 tasks in the body (one rater's is undefined: no common support on the prior).
