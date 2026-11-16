cell: P1.22
rival_a: The side built to reuse sits closer to something already on M's list than the side built to be new, which predicts that on each trained matched pair and seed the reuse side's curve of best closure against number of names (list all) rises above the unpredictable controls' band with fewer names than the new side's
rival_b: The greedy search over M's list closes the gap of any fine-tune that moves a few output logits, whatever it reused, and of none that does not, which predicts the two sides of a pair overlapping where both move a few logits (case, repeat) and both near the unpredictable band elsewhere, with no side consistently ahead
rung: strategy
smallest_gap: 0.1
spread: 0.03
n: 2
design: unpaired
unit: seeds per side (trained fine-tunes of one built side of one pair)
spread_source: experiments/simplest_intervention/results
stop: The re-read shows no pair whose sides' curves differ by more than their two seeds do at any number of names: the curve reading goes into part 1 as not telling the trained sides apart, and no new search is proposed from it
sketch: experiments/e184_curves/figures/sketch.png
descriptive: user, 2026-11-16, chat: re-read E184 as closure curves, report as a description

The user's request (2026-11-16), verbatim: "Re-read E184 as curves, no new search and no change to the fourth wording (still
tested, not adopted). From E184's saved item files: for each item, closure against number of names, on list all, neurons
only and directions only. No rungs, no bars. Compare each trained matched pair side by side, per seed, against the floor set
by the unpredictable controls (random bit, XOR, floor). Lead with the trained fine-tunes; put the planted hand-direction
edits in an appendix as apparatus checks. Leave the free-direction fits out of the comparison. Say plainly where pairs
separate and where they do not (case, repeat). One figure, write-up via /writeup."

Spread: seed sd of closure at 4 names on list all, copy24 seeds 0 and 1 (0.718, 0.760), E184's item files. No GPU, no new
search: the run is `experiments/e184_curves/plot.py` on the saved JSON.
