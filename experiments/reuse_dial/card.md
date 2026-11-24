cell: P2.63
rival_a: Cheap readings follow how much of a fine-tune is a gain on the base model's own variables, beyond how far the output moved, which predicts that within each task one cheap reading (steering or effective rank; the re-gating share, which follows the dial by construction, is reported beside them as a check) ranks the fifteen dial fine-tunes by dial setting with permutation p under 0.05 and a rank correlation at least 0.2 larger in size than KL's, with one sign, on at least three of the four tasks
rival_b: Once the trained-answer gain is matched, the cheap readings follow only how far the output moved, or nothing, which predicts that no cheap reading beats KL's rank correlation with the dial setting by 0.2 with one sign on three of the four tasks
rung: premise
smallest_gap: 0.2
spread: 0.14
n: 4
design: paired
unit: tasks; on each, the cheap reading and KL are read on the same fifteen dial fine-tunes (five dial settings x seeds 0-2), all stopped at one trained-answer gain
spread_source: experiments/claim2_needed/results/predictions.json
spread_of: proxy: E275's sampling floor of the same readings' rank correlations over the procedural states (share 0.125, steering 0.141, erank 0.004, sampling_floor.measured), the move of a rank correlation of these readings over about fifteen to forty states when its reference is resampled, in rank-correlation units as the gap
floor: MIN_GAIN, 1 log-odds: every arm's matched target is at least 1 (a task under it is dropped) and the Gaussian must gain under it; both rivals expect every dial arm at its task's target (6.0, 8.7, 6.0 and 1.6 log-odds before the rehearsal sets min(reference, 0.9 x the gains-only plateau))
stop: If no cheap reading beats KL along a dial built so that reuse is known, on realistic tasks at matched gain, the paper's claim 2 is written as holding on the built tasks only, and no further realistic-task test of the cheap readings is proposed
sketch: experiments/reuse_dial/figures/sketch.png
rehearsal: 'E300 rehearse: reuse dial' on one cluster L40 (debug QoS): calibrate on each of the four tasks at the full step caps (the gains-only seed-0 plateau, the matched target and each of six betas' residual share, filed before the paid run), then the item (addition, seed 0) with every arm capped at 40 steps, every state read in full at the full read sizes and the two ends of the dial scanned for E_b over K at the full sizes (256 windows x 64 tokens, 4 x 64 pairs), run twice so the second skips every part (resume), and its tar downloaded with sha256 on both sides. Measures stage 0's floors (export error, E_b of a gains-only state), seconds per read and per E_b scan, peak GPU and host memory, state and tar sizes
