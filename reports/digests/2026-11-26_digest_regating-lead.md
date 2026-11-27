# regating-lead digest, 2026-11-26

**P2.86, due 11-27: filled, 2 of 2 runs.** Every other regating-lead cell is filled or closed (P2.33 2/2, P2.34 closed 2/2, P2.36, P2.40, P2.66 4/6, P2.68 1/3, P2.74 1/3, P1.63 2/2, P2.76, P2.78 1/1, P2.82 1/1). Finishing: no open cell is left.

**What ran.** The user asked (11-26 ~19:05, "yes do both") for the realistic panel's two blind raters to rate Fig 3's 16 known-answer tasks as a calibration. Run 1 (E315) used descriptions that, redteam found, named the construction (the base model's own first guess; "at random"). whip-lead approved run 2 (E316): descriptions by data only, with three real key-to-target pairs and no step counts. Fresh Opus and GPT raters each time, prompts and answers committed before reading. No GPU, $0.

**What it found.** In both runs no task built new was rated reuse and no task built to reuse was rated new, and no consensus tier changed between runs. Ciphered text (built new) and the three copy-key tasks (built to reuse) sit in the middle. The tasks built new are arbitrary word maps, so this checks that the raters read plain cases, not that Fig 4's realistic tiers are right. Redteam cleared the page; whip-lead sent the appendix G line to writeup-lead. Page: reports/analyses/2026-11-26_analysis_rater-calibration.md.

**For the user's gate.** Premise check on P2.86: doubtful. Its question: "what do you now take a middle tier on a realistic pair to mean?", since the middle absorbs tasks built both ways (~/commute-data/premise/P2.86.md).

**Asks waiting on the user:** none. **Spend:** $0 (model calls only). **Friction:** banned-word list vs real tokens in the examples; descriptive cards still need a sketch PNG.

**Backlog:** 13 (canonicalise SVD signs, soon) stays open; not a cell.
