# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · experiments/qwen_sft/results/F.json · decisions (new card decision-qwen-low-rate, Soon, For you: SFT at lr 1e-5 is not small at step 1: one Adam step (each weight moves by the full rate, no warm-up) gives KL to start 0.106 nats per token on maths rows and 1.57 of the 1.98 gold-loss drop reached by step 100; at step 100 accuracy is 0.627 GSM8K, 0.584 GSM1k (from 0.788, 0.778), answers 198 → 92 tokens; options: add an SFT arm at lr 1e-7 on the next pod, same saved steps, ~15 min (recommended), or keep only the 1e-5 run) · Q9

- 2026-10-28 · decisions.md (row: decision-e38-direction closes with (a)) · decisions (remove decision-e38-direction; decision-regating-reading pointer/bullet), E38 card Moved on (closed on 67M), Q5 status line if it says the card is pending, overview and section next re-gating Next, timeline 2026-10-28 · the user closed the 67M restore line ("yeah lets close this for now"): reading recorded as in the E38 card; no re-gating runs pending
