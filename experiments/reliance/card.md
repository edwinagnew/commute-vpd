cell: P1.57
rival_a: The kept clause is what fails, and reliance separates reuse from new, which predicts V1 and V2 high on x2 and x10 and low on new_few, new_thin and random_kl, E62 copy above lookup on every seed, and a positive consensus tau on the realistic panel
rival_b: Reliance fails too (the needed set does not tell a strengthened mechanism from a new one at these featurizations), which predicts no separation: x2 and x10 within 0.2 of the new edits and random_kl, copy not above lookup on every seed, and a panel tau at or below 0
rung: strategy
smallest_gap: 0.2
spread: 0.087
n: 3
design: paired
unit: seeds per E62 side (copy, lookup) paired by index; built edits are one state each; panel tasks are means of three seeds
spread_source: experiments/needed_neurons/results/items
spread_of: proxy: the share of variables in F's needed set (|K| / 12,316) over E62 copy seeds 0-2 at neurons (0.33, 0.48, 0.33), a share on the same needed set as V1 and V2, on [0, 1]
floor: |K_F| < 100 reads "too small to read" (E272's K_MIN); both rivals expect |K_F| in the thousands on every control and E62 item (E277: 2,882 to 12,300)
stop: If neither V1 nor V2 puts x2 and x10 0.2 above new_few, new_thin and random_kl, no reliance reading is proposed in place of the kept clause, and P1.57's write-up says the separation fails on the needed set itself, not only on the kept clause
sketch: experiments/reliance/figures/sketch.png
rehearsal: cluster only: M's scans are E277's and E275's scan code (needed_neurons.scan, needed.needed_set) on M instead of F, already run at full read sizes on cluster L40s (E277 job 64342: 2,600 cases/s on x2's D, 6.2 GB host RSS; E278 job 64357: 273 s per VPD item); every job but the panel's squad fits a 30-min debug job, and squad (about 44 min on an L40) goes to an H100 pod (about 15 min) or waits
descriptive: user via whip-lead, 2026-11-23 ~14:20, $0 diagnostic of the kept clause, no bar

## What runs (E283, cell P1.57; user via whip-lead 2026-11-23)

The question, in the user's words: "is the failure in the definition's KEPT clause? Does a reliance-based reading
separate reuse from new where kept fails?" The kept clause's reading, the write share E_b over the needed set K, does not
separate E62 copy from lookup and orders the realistic panel against the raters' tiers. Two reliance readings replace
the write share here, each a share of F's flip mass on its needed set (a variable's flip count is the number of tested
input pairs on which swapping its value changes F's top token):

- **V1, needed by both**: the share on variables that M also needs, on the same inputs and pairs (M's own needed set,
  scanned here: the F scans never saved it).
- **V2, needed and barely changed**: the share on needed variables whose write moved by under half of M's own write
  (squared change under f = 0.25 of M's squared write, summed over D).

Neither is the definition, nor a proposed change to it: they are diagnostics of where the kept clause fails. "Needed
only" is not read (it passes the new edits by construction). Both read 1 at the do-nothing by construction.

Order: (1) the control, x2 and x10 against new_few, new_thin and the random; (2) E62 copy against lookup at heads and
neurons and at VPD (and on E281's copy windows at VPD); (3) E280's three head edits; (4) the panel at neurons, Kendall
tau with E272's consensus tiers and the base-loss partial, by `score_panel.py`'s functions. V2 needs no new scan and is
computed first; V1 waits for M's scans (run.py, cluster). Settings, rules and expected values:
`results/predictions.json`. Known before computing: on x2's D every read position is read and |K_F| is nearly every
variable (12,291 of 12,316), so V1 is expected near 1 there for x2, x10 and random_size alike; the V1 contrast rests on
the new edits' D.
