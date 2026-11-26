# redteam-lead digest, 2026-11-25 (evening)

Tonight I re-checked the crosscoder results page and the combined Fig 6 caption. Both now pass. The crosscoder's "almost no fine-tune-specific part" is stated as a reading at 200M training tokens, not as a property of the fine-tunes. What is left waits on the user or on a peer, so I am finishing.

## Cells
redteam-lead has no cells and no rung (review role). No runs, no spend.

## What was checked
- **E311 crosscoder page** (P2.83, baseline-lead, `reports/analyses/2026-11-25_analysis_p2-83-crosscoder.md`): passes at 482f547e. Every claim that the fine-tune-specific part is near 0 now says it was read at 200M tokens. The one plant trained four times longer (800M) separated from 400M on, so the 200M reading is not a property of the fine-tunes.
- **Combined Fig 6 caption v1** (writeup-lead, last section of `~/commute-data/agents/writeup-lead/draft-2026-11-25-fig6-revft-caption.md`): passes. I checked its claim that the control "passes under both filters from 400M" against `experiments/crosscoder_fig4/results/stage0b_answer_s0_long.json`, 400M snapshot. There the fine-tune-specific part is 0.722, the error ratio 0.175 (bar 0.2), and the original-model coefficient positive. It goes into the paper only when the user allows paper edits.
- **Who decides the 4x rerun:** whip-lead ruled that rerunning the crosscoders on the real states at four times the tokens is the user's call. The page now says "not run: outside this cell; the user decides".

## Waiting on the user
- The card `reports/decisions/2026-11-25_decision_e311-4x-tokens.md` (baseline-lead's): whether to rerun the crosscoders on the real states at 800M tokens, about $800.
- Permission for paper edits. After that, I re-check the reverse fine-tuning knock-on edits (appI:182-183, :201, appG:11, appI:252) and any shortened Fig 6 caption.

## Friction
None new.

## Not run
- No new audit of the paper. Paper edits are on hold for the user.
- The P2.81 gain-per-token suggestion was optional and is not reopened: the cell is filled and its lead has finished.
