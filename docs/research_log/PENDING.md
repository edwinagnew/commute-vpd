# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · reports/analyses/2026-10-28_analysis_qwen-prefix-test.md · card decision-qwen-baseline; timeline · The prefix test on Qwen3-1.7B-Base (job 39181) stopped at its copy control, 0.54 against a 0.90 bar: shown the whole question, the base model copies the narrative and then starts a solution near the final question, so the word-for-word rule gave no reading of memory; the fresh-number verdict stays ambiguous. Update the card (prefix test stopped, no reading) and list the user's options without a recommendation, cheap check first: (c) the post-trained model's fresh-number harness on the 2026 synthetic set; (a) fresh sets primary, GSM-Plus and the 2026 synthetic set, GSM8K test secondary; (b) a likelihood test without generation (mean log-likelihood or Min-K% of GSM8K test against train and the 2026 floor). Add a timeline line for the stop.
