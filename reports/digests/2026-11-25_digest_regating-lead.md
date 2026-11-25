# regating-lead digest, 2026-11-25

P2.82, due 11-27: filled (1 of 1 runs, $0 on cluster debug). I am finishing because my only open cell is done and I have no ladder rung.

**What ran.** E310 tested whether random weight changes, scaled up until they move the trained answer by one log-odds, give a baseline for the body's log-odds re-gating share. It ran on the 48 Fig known states, with 3 random directions each. The control reproduced every fine-tune's share bit-for-bit.

**What it found.** On no reuse task did a random change move the answer by one log-odds, at 12 log-spaced sizes up to 100 times the fine-tune size. On the four new-side tasks where one did, all with cipher or random targets, the random shares read 0.24 to 0.91. The fine-tunes read 0.21 to 0.27 there, so the random shares sit at or above them. The two are not matched: the random changes reach a gain of about 1 against the fine-tunes' 11 to 21. Page: `reports/analyses/2026-11-25_analysis_p2-82-random-logodds-share.md`.

**Hand-offs.**
- redteam-lead's review found 2 MUSTs, both now fixed: `reports/reviews/2026-11-25_review_p2-82-random-logodds-share.md`.
- The paper sentence went to writeup-lead for 5.2, and whip-lead was told.

**For the user's gate.** The premise check says "doubtful": with gains so far apart, the reading cannot say whether the share depends on gain rather than training (`~/commute-data/premise/P2.82.md`). It is advisory.

**Asks waiting on the user:** none of mine. **Money:** $0 (no pods). **Slurm:** none live.

**Friction noted:** 1156 (shard time limits sized from the mean, not the worst case), 1157 (state files carry no shard), 1158 (lint needs three cpu-run round trips).

**Not run:** refining sizes around each capped direction's peak; backlog 13 (SVD signs).

---

## Earlier digest (night of 11-24)

P2.66, due 11-24: filled at ten tasks. P1.63, due 11-25: filled and redteamed. All my cells are filled or closed, and no
rung is open for me, so I am finishing.

## Cells
| cell | state | runs used of allowed | due |
|---|---|---|---|
| P2.66 realistic convergence panel | filled | 4 of 6 | 11-24 |
| P1.63 Figure 2's built edits on more circuits | filled | 2 of 2 | 11-25 |
| P2.78 kept share at VPD with three random draws | filled | 1 of 1 | 11-25 |
| P2.68, P2.74, P2.76 | filled | 1 of 3, 1 of 3, 0 of 1 | 11-24 to 11-25 |

## What ran and what it found
- **P2.66 run 4 (E307).** We retrained arXiv and German to 1200 steps, but neither reached the held-out gain floor, so
  the panel is ten tasks (the user: "n=10 is fine"). No reading orders the fine-tunes the way the raters expected. The
  effective rank's earlier pass on twelve tasks rested on arXiv and German and is gone. The kept share at VPD runs
  opposite to the raters, and random updates of the same size reproduce about half of that. Page:
  `reports/analyses/2026-11-24_analysis_p2-66-run4-n10.md`, redteamed with all its points applied.
- **P1.63.** The page's wording on induction at SAE latents was corrected after redteam, and the cell was re-filled.
  Page: `reports/analyses/2026-11-24_analysis_p1-63-more-circuits.md`.
- writeup-lead has a note with the changes to the paper:
  `~/commute-data/agents/writeup-lead/note-2026-11-24-regating-p266-n10-final.md`.

## Waiting on the user
- No asks are open. The premise check on P2.66 came back doubtful. Its question is whether the raters' tiers can serve
  as ground truth on realistic tasks: if a reading disagreed with the raters, what result would say which one was wrong?
  This is advisory and is for the user's gate.

## Money
This evening's E307 step 2 used two H100 SXM pods, $11.88, both terminated. No pods or Slurm jobs are live.

## Friction
Noted tonight: 1142-1146 (the cpu-run retry limit, a review commit claimed as pushed but not pushed, the eb pod input
list, sync sending other pods' staging folders, and the worktree guard on the scorer).
