# Review: E55 Amendment 5 analysis, independent check (2026-10-30)

A fresh Opus agent that neither ran nor wrote the analysis checked `reports/analyses/2026-10-30_analysis_e55-amend5.md`
and the notebook's Amendment 5 section (commit c7720dd). Verdict: **stands with corrections**. All eighteen
corrections were applied in the commit "E55 Amendment 5: apply independent check corrections".

Verified: every number against `amend5.json` and the log; the commit order 8a83aec 12:08:23 (predictions filed),
30ab4f6 12:09:03 (first amendment), 7b5ef02 12:14:32 (second amendment), then job log lines at 12:15:17 and 12:16:41,
so both amendments precede the run; the run's constants equal the amended file.

Corrections, in short:
1. "Fresh rows" became "unseen training-distribution rows (Amendment 2's streams; no model trained on them)", in the
   report and the notebook.
2. The slow seeds' 2K patch share range is 0.748 to 0.804, not "0.75 and 0.80".
3. The 'reuses' side is sixteen states, seventeen counting Stage A's seed copy, which is outside the eleven labelled
   states.
4. "Sixteen 'reuses' against nine 'not through'" became "against nine 'new'".
5. The report said nothing on which saves enter the truth set; it now says that was not filed and the Stage C plan
   fixes it before scoring.
6. The fast copy sees the same rows in the same order as slow seed 0, so it is a paired comparison of learning rate,
   not an independent run; the independent-run count follows.
7. P4 is met but untestable: no state at 1K was through on neither count, so the shortcut rule could not fire. The
   share 0.000 at every save is the informative reading (described).
8. The positive control's "what it rules out" and its caveat overstated: it shows the checks read "through" for a
   fine-tune whose answers are the count of a, not that the prompt was routed into LOW's counter.
9. The natural-row readings rest on behaviour too (LOW's patch share 0.959 before training); the block-5 patch passes
   any answer that differs between the two texts of a pair (LOW under the T prompt, 0.655). Added to How compelling and
   Blind to.
10. "The fast copy leaves the old count for a rule that still fits every row" was a conclusion about the rate from one
    seed; reworded to the observation and its limits. The figure title now says "reads neither by 2K".
11. The fast copy resembles Stage A's 10K arm on the two count-a checks only (count-b patch 0.367 vs 0.43; not filed).
12. The shortcut-rule amendment's rationale was wrong as written; under the rule as first filed the fast copy at 2K
    would have read "shortcut", and the amendment makes it "neither".
13. Six departures from the literature were missing: evidence, unmatched label histogram, the shortcut check, read
    position, learning rates, and what is inherited from Stage A.
14. "Adam" became "AdamW", weight decay 0.1.
15. Notebook: the patch pairs differ in the count of a but share the count of b.
16. The file's definitions block still states the first shortcut rule (the amendments list governs; the run used 0.5);
    the second amendment also moved P3's lean (neither or shortcut 40% to 45%, new or both 20% to 15%).
17. The figure's right panel had no y tick labels; added.
18. Not run gained: Stage A's N1 control re-read through this code; the paper's 1e-6 rate; pruning, probing and
    reverse fine-tuning (Stage C1).

Open, for the lead: the code's docstring and comment still say "fresh"; to be fixed with the next code commit.
