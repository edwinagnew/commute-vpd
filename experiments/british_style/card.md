cell: P2.48
rival_a: British spelling is a style the base model already writes (the forms are in the Pile and it prefers them over scrambles), so a fine-tune towards it mostly turns up parts the decomposition already has; this predicts that at matched size 0.0175 the re-gating share r of all three British seeds lies above all three scrambled-twin seeds, by at least 0.1
rival_b: Both tasks are the same shape (one fixed word form swapped in at the same positions of the same text), and r reads that shape rather than whether the base already knew the form; this predicts that at matched size 0.0175 the British and twin seed ranges of r overlap, or the twin reads higher
rung: strategy
smallest_gap: 0.1
spread: 0.018
n: 3
design: paired
unit: fine-tune seeds (batch orders 0-2) per task; the British and twin run of one seed share the source rows, the batch order, the recipe and the read occurrences
spread_source: experiments/realistic_tasks/results/run2_T3.json
spread_of: this reading: the share r at size 0.0175 on E211's LM tasks through the same reader (realistic_reads.share_of); seed sd T3 0.018, T1 0.014, T4 0.002 (run2_T1.json, run2_T4.json)
floor: MIN_GAIN, 1 log-odds of the scored token at size 0.0175, checked on seed 0 of both tasks before seeds 1-2; rival a expects British gain 2 to 6 and twin 4 to 12, both over it; rival b expects the same; a side under it is unreadable and stops the run
stop: If r does not put British spelling above its scrambled twin (ranges overlap or reversed), r fails on the one real-text task with a twin built to its own premise, and part 2 reports r's built-pair separation as not carrying to real text with a twin; no further real-text twin is run for r
sketch: experiments/british_style/figures/sketch.png
rehearsal: 'E### rehearse: british-style' on one cluster L40 (debug QoS): stage 0 in full (E53 control, do-nothing, base contrast on the full 500 read windows), the data build at full size (4,800 train rows a task), then seed 0 of both tasks with step_cap cut to 40 and the targets met early by a rehearsal-only target list, read at the full 500 windows and saved in bf16, run twice so the second skips every saved state (resume). Measures s/step, read seconds per state, peak GPU and host memory, state file size, and each task's size at steps 10/40 to project steps to 0.0175
