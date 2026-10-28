# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · experiments/qwen_maps/results/{C,stopped,score,state}.json · E44 card · paid run on an H100 ($3.99/h, US-MO-1): stage C done, W0 control holds (gap 6e-6), KL to start 0.106/0.520/0.664 at steps 1/10/100; stage S stopped at its stored-precision rule (layers.2.mlp.up_proj step 1, 3.8e-3 > 1e-3); $1.45 spent; $0 diagnosis on the cluster next
- 2026-10-28 · (wording) · questions.html:11, timeline.html:25, decisions.html:47 · E41 Amendment 2 "every 100 steps" -> seven checkpoints (steps 1-1,000), 100 student steps per link
- 2026-10-28 · (ask 3 dropped) · pending decisions · remove card decision-e44-card: an H100 placed in US-MO-1 at 22:37 ($3.99/h); decisions.md line: E44 ran on an H100 SXM, the H200 option was not needed
