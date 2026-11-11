cell: P3.6
rival_a: A fine-tune changes which MLP neurons fire most more than a random change of the same per-matrix size does (as on the 67M, E155: real 0.60 against matched random 0.83-0.95) predicts more than half of the 19 saved Qwen updates read a top-set overlap below their own four-draw random floor by more than the floor band
rival_b: At Qwen scale the top set moves only with the update's size, not its direction predicts more than half of the 19 saved updates read a top-set overlap within the floor band of their own random floor
rung: parameter
smallest_gap: 0.1
spread: 0.0235
n: 19
design: paired
unit: saved Qwen updates (15 of E152 step 0, E120's four pool states at step 100), each against its own matched-norm random floor; the gap in units of the floor's distance from the unchanged model's overlap of 1.0
spread_source: experiments/qwen_cheap_readings/results/run1_stage1.json (sd of one floor draw = two-draw difference over the floor's distance from 1.0, over root 2: 0.0235 at F_s0, 0.0171 at a2_s42; the larger)
stop: The unchanged model does not read overlap exactly 1.0, E92's columns do not reproduce, or the four-draw floor band is unreadable or over 0.33 of the floor's distance from 1.0 (the overlap cannot be read on Qwen: no scoring until fixed); and if every readable update sits within the band of its floor, the overlap says nothing beyond an update's size and is dropped from part 2's comparison
sketch: experiments/qwen_cheap_readings/figures/sketch_run2.png
