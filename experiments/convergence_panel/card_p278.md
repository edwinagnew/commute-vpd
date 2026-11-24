cell: P2.78
rival_a: the write share's reversal against the tiers comes from what training changed, and seed 0's Gaussian was a typical draw predicts the random tau on 3-seed means stays near the one-draw -0.29, so random minus trained tau stays at or above 0.24 (gap survives)
rival_b: the reversal comes from the task's read windows alone, so any update of that size reads it, and seed 0's Gaussian understated it predicts the random tau on 3-seed means moves toward the trained -0.69, so random minus trained tau falls under 0.24 (gap does not survive)
rung: parameter
smallest_gap: 0.0071
spread: 0.0150
n: 12
design: paired
unit: tasks of the wide KL 'hi' set; the reading is a task's random write share, the 3-seed mean against seed 0's one draw, paired by task. smallest_gap is the seed-0 Gaussians' sd across tasks (0.00713, results/wide/eb/*.json): a task's random moving by that much can reorder it among the tasks and move the random tau. The rule itself is in tau units (results/wide/predictions_p278.json "rule": random tau minus trained tau >= 0.2403, the half-width of the trained tau's 90% task-bootstrap interval, summary.json .wide.kl_hi.hope1.E_b.consensus.interval90 = [-0.885, -0.405])
spread_source: experiments/convergence_panel/results/summary.json
spread_of: this reading: the write share's pooled within-task seed sd, .wide.kl_hi.readability.E_b.within_sd = 0.0150, measured on the trained states (E272 read no second seed of a Gaussian, so the random seed sd is unmeasured until this run)
floor: score.py's K_MIN: a Gaussian with a needed set under 100 variables reads as missing, and the task drops from both taus; both rivals expect none to (seed 0's 12 Gaussians have needed sets of 912 or more, results/wide/eb/*.json); do-nothing E_b exactly 1 with 0 changed variables, both rivals
stop: the do-nothing control misses (E_b not exactly 1, or any changed variable) or stage 0 fails twice: the 24 Gaussians are not read and the paper keeps the one-draw -0.29 marked as one draw
sketch: experiments/convergence_panel/figures/card_p278_sketch.png
rehearsal: a short first stage on the paid card, not a cluster L40, because E272's reads all ran on an H100 SXM ('card' in every results/wide/eb record) and E_b over K depends on the card, and eb.py stage 0 alone would take about 91 min on an L40 (2,190 s on the H100 x eb.py's L40 slowdown 2.5), over the debug QoS: every pod runs stage 0 first (do-nothing, identity, x2 control; eb.py stops on a miss), and pod d then reads sentiment's seed-1 Gaussian at the full 200 windows first (428 s per state on the H100, E272 eb_d2.log), timed for seconds per state and peak GPU memory against eb.py's 2x time rule, which stops the job; E272's $0 rehearsal is on record (rehearsals.tsv, job 63920)
descriptive: user 2026-11-24 via whip-lead, consistency fix (redteam M8), reported as a description
