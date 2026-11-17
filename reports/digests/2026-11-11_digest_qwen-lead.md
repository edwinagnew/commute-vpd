# qwen-lead digest, 2026-11-11

**P3.3, due 11-17: filled.** Cells: P3.1 filled 1/1, P3.2 filled 1/1, P3.3 filled (both runs ran; the table shows 1/2 because step 1 was submitted as "E152 stage: step 1"). No open cells; L3.2 waits on L2.3 and the 11-18 gate, so qwen-lead finishes.

**What ran.** E152, the Qwen instrument check on the new decomposition, at no cost. Step 0 (CPU, job 58729) read saved files: real updates leave as much outside the subcomponents as random changes, and every MLP matrix is fully reachable. Step 1 (L40, job 58847, about 99 minutes) planted changes on often-on and rarely-on subcomponents: the split found most of the often-on change and under a third of the rarely-on one, so by the filed rules the split gives no call, size does not limit the re-gating share, and that share responds to placement. Analysis: `reports/analyses/2026-11-11_analysis_e152-qwen-instrument-check.md`.

**Surprise (filed prediction reversed).** The size call was filed "none" (weak lean) and came back "all sizes"; the rarely-on change's split put about a third of itself on often-on subcomponents, which is what pulled the already-on gap under its bar.

**For the gate.** Premise check on P3.3 says doubtful (`~/commute-data/premise/P3.3.md`): planted changes sit inside the decomposition's basis, while the saved updates look as random as noise by the outside share, so a pass here may not carry to them. No asks open. Spend: $0.

**Friction noted:** 693 (cell add blocked by auto mode), 696 (long pasted command broke at wraps), 707 (Lab.wall exits 0 on timeout), and run counting by E-number.
