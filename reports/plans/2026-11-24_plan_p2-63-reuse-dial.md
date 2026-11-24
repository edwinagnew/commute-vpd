# Plan: the reuse dial on realistic tasks (cell P2.63, E300)

*compare-lead's writer, 2026-11-24. Gates, settings sources, rehearsal and early checks:
`2026-11-24_plan_p2-63-reuse-dial_appendix.md`. Code `experiments/reuse_dial/`; rules `results/predictions.json`;
card `experiments/reuse_dial/card.md`. Origin: `2026-11-22_plan_claim2-known-answer.md`, "Proposed run: the reuse dial".*

**Design check (/design).** Goal: claim 2 of the paper, that cheap readings measure reuse; this run gives realistic
tasks where reuse is set by construction, so a reading can be scored against a known order rather than against design
labels or the base prior. Nearest tried: E231 (gain-only training, P2.46: gains reach most of a fine-tune's gain on
built pairs) and E275 (P2.62: readings against E_b over K across saved states, where the order was not known). New
here: one task, one gain, a known order of reuse. A step towards the goal, filling open cell P2.63 (user approved
2026-11-22 and 2026-11-24). No verdict on reuse under either wording of the definition, and no new rule on it: the
filed rule scores readings against the dial, and E_b over K is only a check that the dial moves the definition.
Control first: stage 0 (below), alone, on every pod. Decision it changes: if no cheap reading follows the dial beyond
KL on realistic tasks, claim 2 is written as holding on built tasks only.

**What runs.** Four tasks: E256's sequence, brackets and addition (its items, read rows and SFT recipe), and E53's
lowercase as the real-text task. Lowercase replaces E233's British spelling: British gained only 1.43 log-odds at its
scored size (E233 score.json), barely over the 1 log-odds floor every share needs, so a dial matched below it would be
unreadable, and British has no D or lens path; lowercase has all three in saved code (E231's recipe, E211's reader,
E257's D). Per task and seed 0-2, five fine-tunes stopped at the first step whose trained-answer gain on the read rows
reaches one matched target: gains only (one gain per VPD subcomponent and per matrix remainder, no residual), three
penalised arms (the gains plus a free residual, penalised by beta times its squared size over W0's), and the ordinary
fine-tune (no gains, no penalty). Per task, one Gaussian at the ordinary seed 0's per-matrix norms. That is 64 states.
Each gets E211's readings (re-gating share, steering, effective rank, neuron shares), KL from W0, the lens column D
(ADL), and E_b over K. The crosscoder is a later stage, outside this plan's cost: about 35 min per state on an H100
(E235 run1 items), $165 for every state, $50 for seed 0. Saved states placed on the dial at $0: E256's SFT and RL arms
(E256 readings, E267 lens, E268 E_b over K).

**Matched gain.** The target per task is min(the reference fine-tunes' gain, 0.9 x the gains-only seed-0 plateau at
its cap), measured in the rehearsal and filed before the paid run. Each arm is checked every step and stops at the
first step at or over the target. An arm that does not reach the target by its cap is reported as a shorter dial, not
dropped. A task with fewer than three dial levels reached is unreadable, and so is one whose penalised arms do not
order by their measured residual share (Spearman of the share with the dial index over the penalised arms' states under
0.8, or undefined with fewer than two penalised levels): a dial whose middle collapsed onto gains only has two ends and
no grade. Training batch is each task's reference recipe's: 32 procedural (E256), 8 lowercase (E53).

**Expected values.** Do-nothing: every numerator and KL exactly 0, E_b = 1 with no changed variable. Gains only:
E_b over its changed variables near 1 but not exactly: the definition's least-squares split can spread one
subcomponent's gain over overlapping ones (E247's head gain x2 reads 0.76). Stage 0 measures it before any item; under
0.5 stops the run, and a value well under 1 is told to definition-lead. Re-gating share high. Random update
at matched size: gain under 1 log-odds, E_b about 0.01 (E268: 0.009 to 0.013 on these tasks). Ordinary fine-tune:
E_b near E256's SFT over K, 0.025 to 0.06 (E268). The penalised arms fall in between, in beta's order.

**Filed rule (claim 2, within task, all states at once).** For each reading, the Spearman rho with the dial index over
a task's 15 dial states, with a permutation p. A cheap reading (steering, effective rank) beats KL on a task
when p < 0.05 and |rho| exceeds KL's by at least 0.2. Rival a: one cheap reading beats KL, with one sign, on at least
three of the four tasks (all of them if only three are readable). Rival b otherwise. Fewer than three readable tasks:
no call. The re-gating share is scored the same way and reported beside them as a construction check, never counted
for rival a: the dial's reuse end is a gain per VPD subcomponent and the share is computed on the same decomposition,
so it follows the dial by construction. Flags (reported, not in the verdict): E_b over K must fall along the dial (rho
at most -0.6); the read gain must not move along it (flagged when it tracks the dial with |rho| over 0.3).

**Cost** (at $4.41 per H100-hour; per-item times from saved logs, appendix gate 1). E_b over K: 48 procedural states
at 740 s (E268 manifest logs) and 16 lowercase at 255 s (E275 items) = 11.0 h, $49. Readings: 64 at an estimated
90 s = 1.6 h, $7. Training: about 1 h, $4. Set-up and stage 0: 9 pods x 25 min = 3.8 h, $17. **Total about $77, about
1.5 h wall on nine pods.** Cut-down for 11-25: E_b over K on seed 0 only (eb_seeds [0]), about $45 and 1 h; further,
the three procedural tasks only, about $38.

**Rehearsal** ($0, cluster debug L40): stage 0, then `calibrate` on each task (target and beta grid; lowercase's arms
split over two jobs), then `rehearse`
on addition with every arm capped at 40 steps and read in full, run twice. It measures what the plan rests on: the
plateau and target per task, each beta's residual share, E_b of the gains-only state, seconds per E_b and per read.

## Not run
Everything above. The crosscoder on the dial states. British spelling. RL arms trained to the matched gain.
