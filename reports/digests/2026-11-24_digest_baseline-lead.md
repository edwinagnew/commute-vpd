# baseline-lead digest, 2026-11-24

P2.80, due 11-26: filled, 2 of 4 runs used, $4.56 in pods. The paper's model-diffing tools were read on the same 48
states Fig 4 plots. Every tool with a passing control gives overlapping values on the tasks built to reuse and those
built new; none of them was built to tell the two apart, so this describes what they measure. Redteam-lead OK'd the
page at 406d1efc and it went to writeup-lead for the tools table.

- Runs: E305 (head patching on the 5 missing seed-2 states, 70026586) and E306 (WeightWatch on all 48, 190bae7e).
  Not run: the crosscoder on seeds 1-2 (planted control failed, 0.244 against a cut-off of 0.2), ADL (67M control
  0.0026 against a random maximum of 0.039), diff-SAE and Delta-Crosscoder.
- **Ranges that changed after redteam (quote the new ones).** Head patching was 0.00-0.66 reuse and 0.01-0.36 new on
  the first page (145e9b7b), which counted 7 empty circuits as zero; it now reads 0.11-0.66 and 0.03-0.36 with them
  left out (406d1efc). The crosscoder's "do not overlap" was wrong and is gone. Under the authors' exclusion of negative
  base coefficients (ea21bdee) it reads 0.08-0.56 against 0.27-0.60 (was 0.06 before the exclusion), seed 0 only,
  unvalidated. Its planted control (0.244 against 0.2) is a different control from the real-text plant the paper's
  -0.006 sentence describes; both fail. Unchanged: neurons 0.41-0.88 against 0.46-0.75, WeightWatch 1.11-2.96 against
  0.95-1.81 (a detector with no reuse score).
- Fig 4 itself draws no random reference; on lowercase the re-gating share's rotation null reaches 0.49 per seed,
  above every trained value on the new tasks (0.42).
- Premise check doubtful (`~/commute-data/premise/P2.80.md`): the tools answer other questions than reuse against new.
- Page: `reports/analyses/2026-11-24_analysis_p2-80-tools-same-states.md`. Open with the user (via whip-lead): whether
  to run the crosscoder and ADL on the 48 states despite their failed controls (about $114 for the crosscoder).
- Friction noted: 1123-1126, 1129-1131.

P2.77, due 11-25: filled, 2 of 2 runs used, $0. Every known-answer pair in Figure 4 and the known appendix is now
read on three seeds a side, with one task variant a side, and the size-matched series reads the same states. The share
still separates all nine built pairs; steering points the opposite way on two of them in pooled form and three in
per-key form, so the paper's old count of three came from mixing the two forms.

## Cells
| cell | state | runs | due |
|---|---|---|---|
| P2.77 Fig 4 / known_appendix standardised | filled | 2 of 2 | 11-25 |
| P2.80 tools on Fig 4's states | filled | 2 of 4 | 11-26 |

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
