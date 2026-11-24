cell: P2.66
rival_a: the update's effective rank tracks reuse beyond how unfamiliar the text is to the base model predicts that on 10-12 new continued-text tasks, KL-matched as E272, the effective rank's partial Kendall tau-b with the blind raters' consensus tiers given base loss (nats per byte, trained tokens) is above 0 at task-permutation p <= 0.05 (E272's eight text tasks: 0.87, p 0.0008, post hoc)
rival_b: the effective rank is only data unfamiliarity (base loss) or noise predicts that the partial tau given base loss is near 0 or below (p > 0.05) while the effective rank's raw tau with base loss is high (tau of rank with minus loss near 1)
rung: parameter
smallest_gap: 9
spread: 5.2
n: 4
design: unpaired
unit: tasks per tier (10-12 tasks over three rater tiers, each task the mean of 3 seeds at one matched KL); the test is a rank statistic over all tasks
spread_source: experiments/convergence_panel/results/summary.json
spread_of: this reading: wide.kl_hi.readability.erank.within_sd = 5.23, the pooled seed sd of the effective rank at E272's wide KL "hi" set; smallest gap 9 = E272's closest reuse-middle task means (arXiv 564 vs Spanish 555, wide.kl_hi.means.erank)
floor: E272's held-gain floor: a task is kept at the matched KL only when all 3 seeds have a state within 10% of the target with held gain >= 3 x the largest |held gain| of three Gaussian updates; both rivals expect Pile-like English text to risk dropping (E272: pile, wiki_en dropped at KL 0.181) and every non-English Wikipedia to pass (E272: french, german, spanish, swahili, yoruba kept); Gaussian control erank ~699 (E272 lower's random 698.9) above every trained state under both rivals
stop: the partial tau given base loss is at most 0, or "no call" (p above 0.05) with at least 8 tasks kept: the effective rank's E272 agreement is then read as data unfamiliarity, and the paper does not claim the effective rank tracks reuse on realistic text
sketch: experiments/confirm_panel/figures/card_sketch.png
rehearsal: cluster only: every stage (train ~5-10 min a task, select on CPU, read and base loss < 15 min a task) runs on the MATS cluster debug QoS at $0 as E272's did, each job under 30 min; no pod
# 2026-11-23, after scoring: the collinearity guard covered non-positive calls only, as redteam-lead asked, so it missed this case (|tau(tier, base loss)| 0.82 over 0.8, call "positive"); redteam-lead: "I should have asked for it to apply to every call." Filed call: positive. Reading: effective rank and base loss both order the tiers perfectly, so the panel cannot tell whether effective rank adds anything beyond text unfamiliarity; the partial is descriptive.
