# Overnight digest, Qwen line (qwen-decomp-lead), 2026-10-29

I stopped because every Qwen step left waits on one of four decisions only you can make. Three of them are about
E45 (comparing different fine-tunes of Qwen3-1.7B) and would go into its last allowed amendment together; the fourth is
the E44 rerun. Overnight, two free checks changed the picture: anchor 2's collapse on maths looks like a carried-over
answer format, not lost maths, and anchor 1's format prompt did work once its format check stopped rejecting line
breaks.

## Decisions waiting on you (each has a card in the research log, section 3)

| ask | question | Fable's advice (advice only) | card |
|---|---|---|---|
| 4 | Rerun E44 on an H100 (~$7-9 more, total under $10)? Precision fix reviewed, $0 rehearsal clean. | — | E44 card |
| 5 | Anchor 2 (invented facts) learns only from step ~100, above every filed comparison point. Read it at step 100 and match the other fine-tunes to it on the maths rows? | yes | `decision-e45-anchor2-grid` |
| 6 | Anchor 1 has no prompt that raises maths accuracy. Make it the answer-format fine-tune, with the format check fixed to accept line breaks? | yes, in Amendment 2 | `decision-e45-anchor1` |
| 7 | The own-answers and null fine-tunes stopped at a first-minute check whose limit was set from the gold arm. Restate the limit as a tenth of the no-warm-up control for every arm? | yes, in Amendment 2 | `decision-e45-step2-check` |

## What ran (all on the MATS cluster, $0)

- **Format vs damage check.** With a few-shot prompt, the invented-facts fine-tunes score 0.70-0.77 on GSM8K against
  0.75 for the untouched model, and their loss on the reference answers falls (1.2 vs 2.5). So their unprompted drop to
  0.13-0.20 is mostly answer format. `experiments/qwen_updates/results/check_format.json`.
- **Own-answers training set.** 5,939 correct answers kept of 8,000 (unprompted accuracy 0.834). `results/S_own.json`.
- **Own and null fine-tunes** stopped within a minute: their change on the maths rows at step 2 was 0.0057 and 0.0067
  against a limit of 0.005, while gold passed. The step size itself was normal. Fable traces the limit to the
  gold arm alone, while a missing warm-up, which the check is meant to catch, reads about 20 times larger. Reruns would miss again.
  See `reports/reviews/2026-10-29_review_fable-own-step2-kl.md`.
- **Anchor 1 format.** 503 of 512 prompted answers follow the template; a whitespace-tolerant check passes 93% (filed
  threshold 90%), the filed check 3%. `reports/reviews/2026-10-29_review_fable-anchor1-no-prompt.md`.

## Money

No RunPod spend this session. Five cluster jobs ran, all free; none is still running.

## Blocked agents

`redteam-lead` has waited at a question since yesterday evening; it is not on this line.

## Friction noted

Four small harness frictions, numbers 110-113 in the friction list: the handoff tool needs a heading, the log lock refuses a clerk name,
the log publisher demands a full read even after a matching check, and foreground waits are moved to the background.

## Not run

E45 rows for own, null, anchor 1 and anchor 2 seed 1; E45 Job 2; the E44 rerun; E45 analysis and notebook (waits on
the rows).
