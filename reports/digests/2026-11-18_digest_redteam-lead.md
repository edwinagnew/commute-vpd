# redteam-lead digest, 2026-11-18

P1.32 (due 11-19) and P1.33 (due 11-20) are both filled. Neither had runs, and both were reading and writing only. Both pages are linked in the log (v395). I finished because no other cells are open.

## Cells
| Cell | Status | Runs | Page |
|---|---|---|---|
| P1.32, cold read of the working definition under both answers to ask 48 | filled | 0 of 0 | `reports/reviews/2026-11-18_review_definition-cold-read.md` |
| P1.33, triage of the cells whose premise check came back doubtful | filled | 0 of 0 | `reports/reviews/2026-11-18_review_premise-doubt-triage.md` |

## What they found
- **P1.32.** Answer (a), as written, reads only the variables that changed.
  - An old skill aimed at a new cue therefore reads as new. That is how reward hacking and emergent misalignment come out.
  - A behavior the base model already has gets no reading.
- **P1.32, answer (b).** It reads both fine-tuning cases as mostly reused, as the literature does, but it reads a memorised fact as mostly reused too.
- **P1.32, both answers.** Neither names a "created" outcome, compares two pretraining runs, or reads a rate change. The page lists eleven clauses that no reading uses.
- **P1.33.** The 60 cells fall into sixteen groups of doubts.
  - Four doubts change a claim as the log states it, all about what the re-gating share is shown to add beyond cheaper readings.
  - Ten narrow a claim's scope.
  - The separation of the nine built pairs still stands.
  - Notes went to definition-lead, writeup-lead and compare-lead, because rewording those claims is up to their owners.

## Asks, spend, blocked
- No asks are open.
- Spend: none. There were no jobs and no pods.
- Nothing of mine is in `blocked.txt`.
- Log slip, left for the next flush: the v395 flush counted P1.32 and P1.33 in the part 2 cell count, but both are part 1 cells.

## Friction
- `cell add` refused P1.33 at first, although the task file named it. Re-adding it with the task's own text passed.
- The 60 premise notes were too large to read in one call. A one-line index would help.
