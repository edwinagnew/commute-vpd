# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
- 2026-10-29 · reports/plans/2026-10-28_plan_claim-tests.md (Amendment 1) · E47 run line · code written (t1-t5 + t1b, t2b, t2c, t4b, t5b, all ≤ 24 min L40 debug jobs); under review before submission
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e47-claim-tests.md · new E47 card; claims section for (i) now, for (iii)-(v) pending check; timeline · five claim tests: (i) stands on two teachers at four starts; (ii) still weakened (rank 8 passes on rows 256-511 only, rank 128 climbs there); (iii) mixed; (iv) falls at step 10 (unexplained reversal), unsettled at step 1000; (v) falls, plain SGD passes at all six smaller cells
