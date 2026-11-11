cell: P3.6
rival_a: A fine-tune's update sits in a few directions per matrix (low-rank post-training, as LoRA assumes) predicts more than half of the 19 saved Qwen updates read an effective rank more than 20 below their matched-norm Gaussian floor (about 1,850 at these shapes)
rival_b: At Qwen scale a small Adam update is spread like noise over every direction predicts more than half of the 19 saved updates read within 20 of their matched-norm Gaussian floor
rung: parameter
smallest_gap: 20
spread: 3.48
n: 19
design: paired
unit: saved Qwen updates (15 of E152 step 0, E120's four pool states at step 100), each against its own matched-norm random floor
spread_source: experiments/update_svd/results/spread.json
stop: The unchanged model does not read overlap exactly 1.0 against itself (the overlap measure is broken: no run 2 until fixed); and if every readable update sits within 20 of its floor, the SVD column carries nothing beyond noise on Qwen and is dropped from the contrasts
sketch: experiments/qwen_cheap_readings/figures/sketch_run1.png
