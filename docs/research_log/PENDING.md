# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · experiments/qwen_maps/results/{C,stopped,score,state}.json · E44 card · paid run on an H100 ($3.99/h, US-MO-1): stage C done, W0 control holds (gap 6e-6), KL to start 0.106/0.520/0.664 at steps 1/10/100; stage S stopped at its stored-precision rule (layers.2.mlp.up_proj step 1, 3.8e-3 > 1e-3); $1.45 spent; $0 diagnosis on the cluster next
- 2026-10-28 · (wording) · questions.html:11, timeline.html:25, decisions.html:47 · E41 Amendment 2 "every 100 steps" -> seven checkpoints (steps 1-1,000), 100 student steps per link
- 2026-10-28 · (ask 3 dropped) · pending decisions · remove card decision-e44-card: an H100 placed in US-MO-1 at 22:37 ($3.99/h); decisions.md line: E44 ran on an H100 SXM, the H200 option was not needed
- 2026-10-28 · experiments/qwen_maps/results/diag_stored.jsonl · E44 card · $0 diagnosis (job 41552, 196 sites x 2 real updates): only the lower-half split on mlp.up_proj of layers 2, 3, 4, 9 misses the 1e-3 stored-precision rule (max 4.2e-3; ill-conditioned X, kappa up to 6e5); forming pieces in float64 from the stored fp32 X brings the max to 1.8e-4; fix under way (rule and threshold unchanged), then review and rerun
- 2026-10-29 · reports/reviews/2026-10-29_review_e44-precision-fix.md · E44 card · fix (47e91aa, 9bba53d): split pieces formed in float64 from the stored fp32 X; filed rule and 1e-3 unchanged; independent review: correct, the fp32 forward floor is outside the rule; $0 rehearsal of the new apply path under way, then the paid rerun waits on the user
- 2026-10-29 · experiments/qwen_maps/results/rehearsal/ (c0d6abc) · E44 card · $0 rehearsal of the fixed code clean (jobs 41719/41729/41733/41758: stop, refuse, waived run of every stage in 7 min, resume skipped; stored_error max 5.6e-6); paid rerun waits on the user (ask)
