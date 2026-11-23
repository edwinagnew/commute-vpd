# The gain-only share rebuilds edits built to be new: what next?

Status: open, for the user. Owner: compare-lead.

**Done, expected, got.** E282 fitted, for each fine-tune of the 67M model, the closest model that only turns the original's own parts up or down, and read how much of the fine-tune's answer it recovers on held-out rows. Exact rescalings came back, as expected. Two edits built to be new were expected at most 0.3 and read about 0.96, so the control failed and E62's copy-versus-lookup comparison is no call. Held-out rows reused the fitted cues.

**Question.** The share as prototyped cannot replace the kept share. What should the cell's one remaining run do?

- **A. Stop.** Report it as a negative result.
- **B. Count the gains needed.** Refit with a sparsity penalty at several strengths; read the share against how many parts moved. Reuse should need few, a new mechanism many. Cluster, $0, 30-60 min.
- **C. Read on generic text.** Fit on the task, read also on Pile text; borrowing unrelated parts shows as damage.
- **D. Test on unseen cues.** Fit on four of the eight cues, or E62's training keys, and read on ones never fitted. Reused copying should transfer; memorised answers should not. Same cost as B.

**Recommendation: D, then B; they can run together.** D tells building from memorising; B asks how much of the original must change. Not C: damage elsewhere also grows with update size. D needs the remaining run or a new cell: your call.

Links: `reports/analyses/2026-11-23_analysis_p1-58-gain-share.md`, `experiments/gain_share/results/score.json`.
