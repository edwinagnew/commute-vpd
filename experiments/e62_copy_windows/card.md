cell: P1.57
rival_a: The 306 non-copy windows of E62's mixed D (cue-free and far text) dilute a real copy/lookup difference, which predicts E_b over K read on the 206 copy windows alone puts copy above lookup by a ratio of at least 1.5 on the mean of three seeds, copy higher on every seed (on mixed D the ratio is 0.90, lookup higher)
rival_b: Per-variable write changes are nearly orthogonal to M's writes on every window (the diagnosis: dd-weighted median variable keeps ~0.01), which predicts the copy/lookup ratio of E_b over K on the 206 copy windows stays below 1.5, near the mixed-D 0.90
rung: premise
smallest_gap: 0.012
spread: 0.004
n: 3
design: paired
unit: seeds per arm (copy, lookup), each state read on the copy windows and on all of D
spread_source: experiments/claim2_needed/results/items
spread_of: this reading on all of D: seed sd of E62 copy E_b over K (0.0376, 0.0399, 0.0362)
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23)
stop: If the copy/lookup ratio stays below 1.5 on the copy windows (over K and over K_copy), the mixed D did not hide a separation and the E62 non-separation stands as read; no further E62 E_b reads are proposed
sketch: experiments/e62_copy_windows/figures/sketch.png
rehearsal: cluster only: six write passes on the MATS cluster debug QoS ($0), stage 0 first (do-nothing E_b exactly 1; the all-D write pass of one state repeats E275's saved E_b over K within a measured transcription tolerance)
descriptive: whip-lead 2026-11-23, after redteam-lead's mixed-D note: "Read E_b over K on the 206 copy windows alone for the six E62 states"

## What runs (E281, cell P1.57)

For the six E62 states (copy and lookup, seeds 0-2) as E257's World builds them: K from E275's saved scan on all of D,
unchanged; num, den, dd recomputed by one `stand_in.write_change` pass on the 206 copy windows of D only; E_b over K
there, beside the saved all-D value. Control first: the do-nothing reads exactly 1, and one state's all-D write pass
repeats E275's saved num/den/dd and E_b over K. K itself is not re-found on the copy windows (a variable needed only on
non-copy windows stays in K): reported as a caveat, with the share of K's dd that the copy windows carry.

Refiled before predictions.json (redteam-lead, note-2026-11-23-redteam-e281-prerun.md): the 0.2 bar was degenerate
(all 114 fine-tunes read 0.015-0.063), so the bar is now on this reading's scale: copy/lookup ratio >= 1.5 on the
three-seed means, copy higher on every seed (smallest gap ~0.012, about 3 seed sd). The rule run on the mixed-D values
(copy 0.0379, lookup 0.0420, ratio 0.90) reads "no separation". K is also re-found on the copy windows: K_copy by the
same single-variable interchange (E268 settings) with both x and x' drawn from the 206 copy windows, and E_b over K_copy
read beside E_b over E275's K. Stage 0's transcription tolerance comes from a measured repeat of the same pass.
Split case (redteam-lead, same day): K_copy is the reading, since the dilution acts through K; E_b over E275's K is
reported beside it. A ratio >= 1.5 over K_copy reads rival A whatever E275's K gives, and below 1.5 over K_copy reads
rival B. K_copy's scan uses E275's pair count (256 pairs: 4 derangements x 64 windows, E268 settings), drawn from the
copy windows, so the two |K| are comparable; the |K| < 100 floor applies to K_copy.
