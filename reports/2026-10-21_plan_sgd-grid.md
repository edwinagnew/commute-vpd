# Plan: SGD grid (E4's plain-SGD arm at smaller learning rates)

**Request (user, 2026-10-21), restated.** The research log says plain SGD on the components' vectors learned the
emoticon behaviour at no learning rate without large off-target damage. That claim rests on E4's grid {0.3, 1, 3, 10,
30}, whose smallest rate already overshoots on step 1. Run the same component-training arm at smaller rates using E4's
code where possible, and report.

**Design check.** Approved class 2: it is the item E4's report lists under "Not run" ("SGD with a finer lr grid
(0.03–0.3)"), on E4's code, task and harness. Goal: tells us whether "Adam causes the diffuseness" (E4's unscorable
P1) can be scored, and whether the log's claim, which the "unused contrast" (the main line's re-gating card) leans on,
holds. Not tried before at these rates (`TRIED.md`, E4 row). Cluster `debug` QoS, ~15 min, $0.

**Run.** `experiments/sgd_grid/run.py run`. Component training (Γ and Δ frozen, all alive (U, V)), plain SGD with no
momentum, E4's recipe (arm A loss, batch 32, clip 1, seed 0, 30 steps), learning rates {0.001, 0.003, 0.01, 0.03, 0.1}.
E4's learning-rate rule (held-out P(o) ≥ 0.9 and KL elsewhere ≤ 0.110 at step 30). At the selected rate, E4's P1
measure: the fraction of the held-out log-odds gain removed by reverting the 10 largest revert-one drops.

**Control first.** Learning rate 0.3 for 2 steps must reproduce E4's held-out KL elsewhere at steps 1 and 2 (8.59,
1.47) within 5%, and P(o) < 0.05 at step 2. If not, the run stops before the grid. What differs from E4: the shared
`component_training.train` loop instead of E4's copy (same loss, batches, clip and seed; evaluates every step), and an
L40 instead of a 4090.

**Expected values.** At W0: P(o) ≈ 0.005, KL elsewhere 0. If the rate is too small, P(o) < 0.9 at step 30 with small
KL; if too large, E4's pattern. Concentration: Adam on all gives 0.016 (diffuse) and the Adam top-10 arm 0.997, so the
measure can read high. It is blind to redundancy beyond the greedy top 10. One seed, no noise floor.

**Predictions** (`experiments/sgd_grid/results/predictions.json`): S1, some rate learns within the limit: pass. S2
(E4's P1), the SGD update concentrates (≥ 0.5; null < 0.2), scorable only if S1 passes: fail.

**Decision tree.** If S1 fails at every rate because the rates are too slow at 0.001–0.01 and damaging at 0.03–0.1, the
log's claim stands in a narrower form (no rate in 0.001–30 at 30 steps). If S1 passes, the log's line is wrong and
gets corrected; S2 then answers E4's P1.
