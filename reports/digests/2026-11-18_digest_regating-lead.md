# regating-lead digest, 2026-11-18

**Update, evening: §5.2-5.4 draft corrected (cell still not in the table, due 11-21).** `fleet/cell add` was refused a
second time by the auto-mode permission classifier, so the cell cannot be filled; please add it (or allow the command).
The draft now carries redteam-lead's three corrections (prior and steering together give all nine directions; no E142
agreement sentence; the untrained-decomposition sentence qualified, since fidelity and the gating rule were not told
apart) and compare-lead's four answers (two top-neuron rows, the steering rank correlation now -0.30, the size-matched
half-gain row stands, a crosscoder appendix entry). Commit 7b0bde3 on branch `push` of the writeup repo; writeup-lead is
asked by note to pull and push. Open marks left: the definition column (ask 50, P1.35, P1.36), granularity in amount
(P1.35), and three data or figure items no answer covered (scorecard figure redraw, the repeat-label step mix, the
size-matched steering move from -0.12 to -0.92).

**New cell (not yet in the table), due 11-21: paper §5.2-5.4 drafted.** `fleet/cell add` for this zero-run writing cell
was refused by Claude Code's auto-mode permission classifier, so it is not in `cells.tsv`; please add it (or allow the
command) and I will fill it. Other cells: P2.33 filled 2/2, P2.34 closed 2/2 (no call), P2.36 filled 0/0.

What was done: an Opus agent drafted §5.2 Baselines, §5.3 Scorecard (a LaTeX table of calls per method on the nine
built pairs, size-matched, twins) and §5.4 Caveats from the filled part-2 cells only, and I reviewed and trimmed it.
It scores the share like the baselines, states the crosscoder misses plainly (5 and 4 of 9 against 7 filed, planted
control missed, random update the most shared), gives no verdict and no single score, and marks with \open every
sentence that waits on asks 48-50 or P1.31. File: `paper/sec5_baselines_scorecard_caveats.tex` (+ `.numbers.md`), on
branch `draft` of `~/commute-data/agents/regating-lead/writeup-repo`; writeup-lead pushes it (my clone has no
credentials) and \inputs it in place of main.tex lines 390-438. Not compiled (no TeX on dev).

Open for others: compare-lead got four questions (P2.28 vs P2.26 on size-matched half-gain neurons; old rank
correlations; top-neuron counts only in final.json; crosscoder appendix entry). writeup-lead got the push recipe.

Asks waiting on the user: none of mine. Spend: $0. Friction: #877 (cell add refused by the classifier), cpu-run
`python` missing on the worker (python3 works).
