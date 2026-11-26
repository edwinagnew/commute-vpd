# E312 note, 2026-11-25, before any run: D_void applies to every item

Filed `predictions.json` (3d6ad306, sha256 a92b8a98...) left unedited. This note responds to redteam's predictions check
(`reports/reviews/2026-11-25_review_p2-84-predictions.md`, SHOULD 1-3). It is a reporting rule only and changes no bar
or stop.

1. Every item with D under stage 0's D_void is reported as void, its t_half and t_tenth set to None. That covers
   states, nulls and gain-only states alike, where the filed rule voided only the nulls. D_void is 10 × max(the
   measured κ floor, 1e-12). A void item stays in the summary with its D, is counted, and is not drawn. `run.py
   --stage summary` reads the machine line below and refuses without it.
2. The caption and plot note say each null is matched to its state on per-matrix norm (spectrum), not on D. They also
   say that t_half rises as D shrinks (C3 ladder, corr −0.90 with norm). A null's longer t_half beside its state is
   therefore partly a size effect. D is drawn beside every pair.
3. The summary reports the measured κ floor (stage 0's batched-against-single read) beside D_void.

void_rule: every item
