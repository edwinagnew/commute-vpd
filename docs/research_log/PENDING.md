# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/analyses/2026-11-04_analysis_e80-adversary.md, notebooks/E80_qwen_d1_adversary.ipynb · new E80 card (experiments page); pending-decision card "D1: what should the test be?" (anchor decision-after-e76; its running line goes and its evidence changes); claims table; "What's next" D1 paragraph; timeline · Independent check pending. Filed outcome "D1 has a test and a first candidate" (a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement), reversing the filed medium lean "within the band"; beside it the bars do not cohere: do-nothing's 1.317 is under the in-span pass bar of 1.40, A passes with a weight error of 1.36, in-span's 1.197 is over do-nothing's fail bar of 1.19, and do-nothing and in-span differ by 10%, less than W0's start gap of 16%, one start each. So the adversary reads do-nothing above W0, and nothing shows in-span below do-nothing or a good placement; no cause claimed; proposal: repeat starts paired by seed.
