cell: P1.57
rival_a: The 306 non-copy windows of E62's mixed D (cue-free and far text) dilute a real copy/lookup difference, which predicts E_b over K read on the 206 copy windows alone puts copy at least 0.2 above lookup on the mean of three seeds
rival_b: Per-variable write changes are nearly orthogonal to M's writes on every window (the diagnosis: dd-weighted median variable keeps ~0.01), which predicts E_b over K on the 206 copy windows stays within 0.2 of lookup, near the mixed-D values (copy 0.036-0.040, lookup 0.040-0.043)
rung: premise
smallest_gap: 0.2
spread: 0.004
n: 3
design: paired
unit: seeds per arm (copy, lookup), each state read on the copy windows and on all of D
spread_source: experiments/claim2_needed/results/items
spread_of: this reading on all of D: seed sd of E62 copy E_b over K (0.0376, 0.0399, 0.0362)
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23)
stop: If copy minus lookup stays below 0.2 on the copy windows, the mixed D did not hide a separation and the E62 non-separation stands as read; no further E62 E_b reads are proposed
sketch: experiments/e62_copy_windows/figures/sketch.png
rehearsal: cluster only: six write passes on the MATS cluster debug QoS ($0), stage 0 first (do-nothing E_b exactly 1; the all-D write pass of one state repeats E275's saved E_b over K within a measured transcription tolerance)
descriptive: whip-lead 2026-11-23, after redteam-lead's mixed-D note: "Read E_b over K on the 206 copy windows alone for the six E62 states"

## What runs (E281, cell P1.57)

For the six E62 states (copy and lookup, seeds 0-2) as E257's World builds them: K from E275's saved scan on all of D,
unchanged; num, den, dd recomputed by one `stand_in.write_change` pass on the 206 copy windows of D only; E_b over K
there, beside the saved all-D value. Control first: the do-nothing reads exactly 1, and one state's all-D write pass
repeats E275's saved num/den/dd and E_b over K. K itself is not re-found on the copy windows (a variable needed only on
non-copy windows stays in K): reported as a caveat, with the share of K's dd that the copy windows carry.
