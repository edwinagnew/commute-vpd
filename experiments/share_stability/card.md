cell: P2.6
rival_a: any small update makes the every-position share read high, whatever it does to the text, predicts a Gaussian update of the step-100 fine-tune's per-matrix norm reading within 0.15 of that fine-tune's pooled share (about 0.8)
rival_b: only an update that moves the model's next-token predictions as this fine-tune does reads high predicts the Gaussian's share unreadable (its gain under 1.2e-3 nats per token) or more than 0.15 from the fine-tune's
rung: premise
smallest_gap: 0.15
spread: 0.065
n: 5
design: paired
unit: rows: 64-row batches of held-out code-text rows, the same batches for both sides
spread_source: experiments/regating_table/results/stage0a.json
stop: if the Gaussian reads within 0.15 of the fine-tune, the every-position form is reported as reading any small update high and no further every-position reading is proposed; if the fine-tune's batch shares spread by more than 0.13 (SD), the 0.77 to 0.82 is quoted with its spread and not as a level
sketch: experiments/share_stability/figures/sketch.png
# spread: E46's per-row SE of the gate gain over the 64 rows (0.0041 nats per token at step 100, .groups["e36/code"].states["100"].se) over the gain |G| = 0.064, the SD of one 64-row batch's share.
