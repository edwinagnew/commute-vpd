# baseline-lead digest, 2026-11-24

P2.77, due 11-25: filled, 2 of 2 runs used, $0. Every known-answer pair in Figure 4 and the known appendix is now
read on three seeds a side, with one task variant a side, and the size-matched series reads the same states. The share
still separates all nine built pairs; steering points the opposite way on two of them in pooled form and three in
per-key form, so the paper's old count of three came from mixing the two forms.

## Cells
| cell | state | runs | due |
|---|---|---|---|
| P2.77 Fig 4 / known_appendix standardised | filled | 2 of 2 | 11-25 |

No other cell is open for this role; no ladder rung.

## What ran
- **E299** (cluster debug, $0): trained the missing seed-2 states and read every figure reading on them with the old
  code; `experiments/known_3seeds/results/known3.json` at 3 seeds a side on every pair, per-key steering included.
- **E303** (cluster job 66586, $0, 6 min): the size-matched share and pooled steering on the three larger-side seed-2
  states; the control (two old states against their saved values) passed first.
- Figure data: `experiments/known_3seeds/results/csv/` (Fig 3a, Fig 4), each with a `forms.json` naming every
  column's form. Redteam-lead OK'd the rows, the nit is fixed, and the matched count was recomputed independently.

## What it found
- Steering reverses the share on 2 of 9 built pairs pooled, 3 of 9 per key, and 3 of 9 pooled with sizes matched.
  The third matched reversal comes from the size scaling alone (redteam-lead).
- Page: `reports/analyses/2026-11-24_analysis_e299-known-3seeds.md`.

## Hand-offs
- writeup-lead has the rows and both counts, queued under the user's 14:10 hold on paper edits; it asked whip-lead
  whether P2.77 enters today's draft. `make_known_scorecard.py` still reads the old sources and must be rewired.
- whip-lead told.

## For the user's gate
- Premise check, doubtful (`~/commute-data/premise/P2.77.md`): with nine pairs, a one-pair swing between steering
  forms is the whole difference between headline statements, so which steering form is the fair comparator needs
  deciding, and seed standardisation does not settle that.

## Asks, spend, friction
- No open asks. Spend $0 (cluster only, no pods).
- Friction 1106 (cpu-run memory default), 1107 (a runner cancelled a queued job), 1116 (a brief misdescribed the
  CSV columns).

## Not run
- Per-key steering for the lowercase-cipher pair (no key pool).
- A size-matched series for the twin pairs, and per-key steering in the matched series.
