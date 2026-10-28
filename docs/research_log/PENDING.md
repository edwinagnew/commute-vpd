# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · decisions.md (last row) · decisions (remove cards decision-qwen-tt-memory, decision-qwen-low-rate, decision-qwen-maths-adversary: decided as in that row), Q9 status, E40 card (parts/experiments/e40.html) · correct the E40 card: do-nothing was read at steps 1 and 10 only (step 100 not run: the run was stopped for the recipe fix); accuracy at step 100 is 0.627 on GSM8K and 0.584 on GSM1k under the drop rule (0.788, 0.778 at step 0), a paired fall of 0.145 on the 1,248 problems kept at both; teacher training goes to an H200 or B200 (user's choice); the fine-tune is rerun with the standard recipe (warm-up and cosine decay) as E40 Amendment 1, the first gated run, after a $0 cluster rehearsal that is running now; maths text is scored as a ratio to the unchanged decomposition's own maths reading
