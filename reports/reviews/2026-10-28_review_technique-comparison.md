# Review: the techniques compared, property by property, and the gaps that leaves

*2026-10-28, review agent for the commute lead. Evidence base for the comparison the user asked to add at the end of
the research log's Techniques section; this file does not edit the log. Vocabulary: `GOALS.md` → Terms. Sources:
`experiments/TRIED.md` (every row grepped), the log cards `docs/research_log/parts/experiments/e*.html`, the
`techniques`, `claims` and `questions` sections, and the analysis reports the rows link. Where the log and a newer
report disagree, the newer report is used and the disagreement is listed in section 3.*

We lined up every operation the project has used to turn a model update into a new decomposition and read, for each,
what the experiments say about the properties the weak version needs. Teacher training is the only operation that
keeps the behaviour, passes the adversarial check and leaves Δ alone on narrow fine-tunes from either optimiser and on a
LoRA, but it fails on broad full-weight fine-tunes and has never been tried on a behaviour the model lacks, on an early
or step-by-step update, or on the larger model (that run is in flight). The splits of a finished update are well mapped
and each fails one condition, so the open cells that bear on `GOALS.md` sit almost all in the teacher-training rows.

## Conventions

- **Cell format:** verdict (yes / no / partial / untested / n/a), the E-code(s), at most one number.
- **"Untested"** is written only after grepping `TRIED.md` for the technique together with the property.
- **Adversarial check** = VPD's 20-step Pile adversary, one start, as a ratio to the unchanged model on the same card
  (pass at or under the card's line, about 1.24 to 1.40; fail at or over about 1.5 to 1.81). Since E34 Amendment 5
  every one-start pass is a lower bound (decided 2026-10-27).
- **"Works" on an update type** = keeps the behaviour (target probability ≥ 0.9 on trigger tasks, kept share ≥ 0.8 on
  the broad lowercase task) and passes the adversarial check. For do-nothing, the least-squares split and the step split
  the full model is the fine-tuned model, so only the adversarial verdict is reported there.
- **Harmless** = changes the model's other predictions (KL from W₀ off the trigger, or on plain Pile text) no more than
  the fine-tune does.
- **Task-local adversarial check:** no powered version exists on the 67M model (E5: seed spread 55 to 88 nats against
  gains of 4 to 15). That column reads "no powered check" for every row and is not repeated as a gap per row.
- **Large / small update:** the emoticon Adam 3e-5 fine-tune (largest relative matrix change 0.011 at step 10) against
  the SGD 0.01 and Adam 3e-6 fine-tunes matched to a fifth of its size (E21).
- Rows added beyond the brief: sparse component training (E4), the unscaled adjoint split as its own row (E31
  Amendment 3), teacher training with plain SGD as the map's optimiser (E34 arm S), teacher training from the adjoint
  split (E37 Amendment 1, arm 3), the Γ refit (E15) and the fixed change of Γ's values applied everywhere (E30).

The matrix is split into three tables with the same rows, one per group of columns.

## 1. The matrix

### 1a. What it consumes, and the core checks

| # | Technique | Consumes | Δ untouched | W′ = fine-tuned weights | Keeps the behaviour | Harmless | Stochastic check | Adversarial check, Pile |
|---|---|---|---|---|---|---|---|---|
| 1 | Do-nothing | finished δW | no, by definition (E16: all of δW in Δ) | yes | yes (full model is the fine-tune) | yes (same weights) | yes, E16, E21: 0.92–1.02 | partial, E2, E11, E21, E37, E39: fails the large Adam fine-tune (2.18) and every broad fine-tune; passes small and narrow-LoRA updates |
| 2 | Rescaling split | finished δW (+ inputs for the weighting) | yes, E16: Δ change ≤ 5e-9 | no | no, E16: target probability 0.012–0.016 (W₀ 0.012) | untested (E16 read no damage elsewhere) | yes, E1, E16: 0.92–1.02 | yes, E16: 0.92–1.00 |
| 3 | Least-squares split | finished δW | no, E16, E22: Δ keeps 0.25–0.54 of the update | yes (to second order) | yes, E2: carries 88–93% of the unmasked gap | yes (full model is the fine-tune) | yes, E16, E21: 0.92–1.02 | partial, E16, E21, E22, E27, E10: large Adam fails (6.96 at step 10), small and LoRA pass, cipher fails |
| 4 | Adjoint split, unscaled (c = 1) | finished δW | yes, E31 A3 | no | yes, E31 A3: kept 1.00–1.03 in 8 of 9 states | no, E31 A3: 1.3–32× the fine-tune's KL elsewhere | untested (not reported in A3) | partial, E31 A3: SGD and small Adam pass, large Adam fails (19.3 at step 1000) |
| 5 | Scaled adjoint split | finished δW + the task's inputs (for c) | yes, E16, E31: Δ change ≤ 1e-4 | no | partial, E16, E31, E34, E37, E39: fades with size and steps (0.93 → 0.10 on the large fine-tune); 0.67–0.80 on broad updates | yes on 8 of 9, E31 A3: 0.42–1.32× the fine-tune; E39: 0.21× on broad | yes, E21, E39: 0.99–1.05 | yes, every state read, E14–E39: 0.92–1.33 |
| 6 | Step split | the fine-tune's path, every step | no, E19: Δ holds 1.38 of the update's input-weighted energy at step 1000 | yes, exact at every step | yes (full model is the fine-tune); behaviour outside Δ below least squares, E19, E25 | yes (same weights) | yes, E19: 0.93–1.00 | partial, E19, E25: large path fails at step 1000 (2.67), small paths pass |
| 7 | Component training, Adam (free) | loss and labels | yes, E16: ≤ 2e-8 | no (a different update in the fine-tune's place) | yes, E10, E16: target probability ≥ 0.994 from step 100 | partial, E2, E11: a third of the fine-tune's on emoticon, 1.6–2.5× on the ladder | yes, E11, E16: 0.92–1.02 | partial, E11, E21, E26: fails while damaging at step 10 (6.74 on r2 t275), passes at 100 and 1000 on r0, r1, emoticon |
| 8 | Component training, plain SGD | loss and labels | yes, E21 | no | yes, E4 follow-up, E21 | partial, E21: less than Adam at matched size (Adam 3.8×); not compared with the fine-tune | yes, E21 (emoticon) | partial, E21, E26: passes small changes, fails at step 1000 on r0 and r1 (3.06) |
| 9 | Rank-r component training (rank 8) | loss and labels | yes, E13: ≤ 1e-8 | no | yes, E16: 0.895 at step 10, ≥ 0.994 after | untested against the fine-tune (E13: KL elsewhere 0.017) | yes, E16 | partial, E7, E13, E16: passes on all 24 matrices (0.97–1.28), fails confined to one matrix (4.52) |
| 10 | Sparse component training (top 1, 10, 100; six siblings) | loss and labels | yes, E4 | no | yes, E4: CI-masked probability 0.98 at the trigger | untested against the fine-tune (E4: KL elsewhere 0.011–0.020) | yes, E4 | yes, E4: ≤ 1.08 |
| 11 | Gated-off component training (control) | loss and labels | yes, E5 | no | yes, E5: held-out probability 0.94–1.00 | untested | untested | untested |
| 12 | Teacher training, task text (Adam) | W₀, W₀ + δW and the task's text; no labels | yes, E34 A3: Δ share < 1e-8 | no, E34 A1, E37: cosine with δW 0.05–0.44 | yes on narrow updates, E34, E34 A3: 0.92–1.00; yes on broad, E37: 0.998 | partial, E34, E37, E39: 0.33–1.48× the fine-tune's | yes, E34: 0.87–1.02 | partial, E34, E37, E39: passes all narrow states (≤ 1.37) and the broad LoRA; fails broad Adam (2.20) and broad SGD |
| 13 | Teacher training, plain Pile text (Adam), incl. trigger rows masked or dropped | W₀, W₀ + δW and ordinary text | yes, E34 | no, E34 A1: cosine with δW 0.08–0.22 | partial, E34, E34 A2, A3: full-weight 0.90–0.98, triggers dropped 0.84–0.87, LoRA 0.52–0.88 | yes, E34 (one state 2.8× its teacher at 0.022 nats) | yes, E34: 0.87–1.02 | partial, E34, E37: every narrow state passes, broad plain-text state fails (1.74) |
| 14 | Teacher training, plain SGD as the map's optimiser (arm S) | as row 12 | yes, E34 | no | yes, E34: ≥ 0.96 | no, E34: 1.3–1.9× its teacher's damage elsewhere | yes, E34: 0.95–1.02 | partial, E34: emoticon passes (1.11), r0 and r1 fail at step 1000 (3.39) |
| 15 | Teacher training from the scaled adjoint split | δW, then as row 12 | yes, E37 A1 | no | yes, E37 A1: 0.99–1.02 | no gain over row 12, E37 A1: damage rose 0.05 → 0.24 (fine-tune 0.235) | untested | no, E37 A1: 1.84 at 2,000 steps (broad only) |
| 16 | Teacher training plus VPD's masked losses | as row 12 | — | — | not run | not run | control only, E37 A1: 1.00 | not run; its W₀ control fails, E37 A1: 3.84 |
| 17 | Γ refit (Γ trained, U, V, Δ fixed) | a state + Pile rows | yes (Δ fixed) | as the state it refits | untested | no, E15: sparsity loss 1.60× | yes, E15: 0.92–1.07 | uninformative, E15: the W₀ control drifts to 6.3 |
| 18 | One fixed change of Γ's values, applied everywhere | Γ's values at W₀ + δW on training windows | yes (δW dropped) | no (W′ = W₀) | partial, E30: 0.44–0.75 of the own values' recovery | untested; sparsity loss 1.8–3.3× | untested | no, E30: 4.7–26.1 |

### 1b. Update types, size and steps

| # | Technique | Adam full fine-tune | SGD full fine-tune | LoRA | Broad fine-tune, no trigger | Across update size and steps | Along training (step by step) |
|---|---|---|---|---|---|---|---|
| 1 | Do-nothing | partial, E16, E22: large fails at step 10 (2.52), passes at 100; small passes | yes narrow, E21, E22: 1.12; no broad, E39: 2.62 | yes narrow, E9, E14, E27: 1.04–1.22; no broad, E39: 2.65 | no, E37, E39: 2.62–4.87 | partial, E21, E22, E39: small passes at every step, large fails; broad SGD fails at a size where emoticon SGD passes | yes by construction (every step into Δ) |
| 2 | Rescaling split | no, E16: keeps none | untested as a state (E22 share only) | no, E16 | untested | no at every step, E16: keeps none at 10, 100, 1000 | untested |
| 3 | Least-squares split | partial, E16, E22: large fails (6.96), small passes, Δ keeps 0.32–0.39 | partial, E21, E22: passes, Δ keeps 0.25–0.30 | partial, E14, E16, E27: passes (≤ 1.32), Δ keeps 0.37–0.54 | untested | partial, E22: small passes at every step, large fails at every step | its per-step form is the step split (row 6) |
| 4 | Adjoint split, unscaled | partial, E31 A3: small passes, large fails (2.40 at step 10) | partial, E31 A3: passes but 1.3–2.2× the fine-tune's KL elsewhere | untested | untested | partial, E31 A3: fails only on the large update | untested |
| 5 | Scaled adjoint split | partial, E16: passes, keeps 0.93 → 0.10 | partial, E22, E31: passes, keeps 0.60 on emoticon at step 1000, 0.07–0.39 on word tasks | partial, E16, E27, E34 A3: passes, keeps 0.04–0.56 | partial, E37, E39: passes for Adam, SGD and LoRA, keeps 0.67–0.80 | partial, E31 A1, A2: keep falls with size along one curve per task (0.60 / 0.39 / 0.13 at matched size) | its per-step form is the step split (row 6) |
| 6 | Step split | partial, E19, E25: large path fails at 1000 (2.67), small Adam path passes | yes on faithfulness, E25: 0.98 at 1000, but Δ holds 1.27–1.78 of the update | untested | untested | partial, E19, E25: fails large only | yes by construction; Δ's share grows along the path, E19: 0.40 → 1.38 |
| 7 | Component training, Adam | n/a (no update consumed); its analogue beside the Adam fine-tune works to step 1000, E16 | n/a | n/a (analogue is row 9) | untested (component training on the broad task's own loss) | partial, E21, E26: fails at step 10 while damaging, passes at 100 and 1000 | yes by construction, E16: to step 1000 |
| 8 | Component training, SGD | n/a | n/a | n/a | untested | partial, E21: passes below a change of about 0.009, fails above (cut chosen after looking) | yes by construction; fails late, E26 |
| 9 | Rank-r component training | n/a | n/a | yes as the LoRA's analogue, E16: 0.97–1.12 to step 1000 | untested | yes, E16: steps 10, 100, 1000 | yes by construction |
| 10 | Sparse component training | n/a | n/a | n/a | untested | untested beyond 30 steps (E4) | 30 steps only, E4 |
| 11 | Gated-off component training | n/a | n/a | n/a | untested | step 10 only (E5) | step 10 only |
| 12 | Teacher training, task text | yes, E34: three Adam teachers, ≤ 1.37 | yes narrow, E34: three SGD teachers; no broad, E39: 2.07 | yes narrow, E34 A3: 0.93–1.00; yes broad once, E39: 1.27 (state lost) | no for full-weight, E37, E39: 2.20 (Adam) | untested: every teacher at fine-tune step 1000 (size 0.012–0.035); the trained state read at 100 and 1000 of its own training | untested (never applied step by step or chained along a fine-tune's path) |
| 13 | Teacher training, Pile text | yes, E34 | yes, E34 | partial, E34 A3: faithful, 0.52–0.88 at step 1000, still rising | no, E37: 1.74 | untested (teachers at step 1000 only) | untested |
| 14 | Teacher training, SGD map | untested (arm S ran on SGD teachers only) | partial, E34: emoticon passes, r0, r1 fail | untested | untested | untested | untested |
| 15 | Teacher training from adjoint split | broad only: no, E37 A1: 1.84 | untested | untested | no, E37 A1 | untested | untested |
| 16 | Teacher training + masked losses | not run | not run | not run | not run (broad was its target) | not run | not run |
| 17 | Γ refit | large least-squares state only, uninformative, E15 | untested | untested | untested | untested | untested |
| 18 | Fixed Γ change everywhere | no, E30: 26.1 | no, E30: 8.0 (emoticon), 4.7 (r1) | untested | untested | 10-step updates only | untested |

### 1c. Tasks, robustness, the larger model, cost and the route split

Tasks: emoticon; the word tasks r0 t352 (" it" then " is") and r1 t534 (" which" then " was") and the other ladder
tasks; the rare targets r2 t275 and r2 t359; the cipher and key-value tasks (rand1, rand32, ident32).

| # | Technique | Emoticon | Word tasks | Rare target (r2) | Cipher / key-value | Seed or batch order | Qwen3-1.7B | Compute (67M unless said) | Route split read |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Do-nothing | partial (row 1a) | partial, E11, E34: fails 3 of 5 ladder tasks at step 10 (up to 2.26) | in E11's 3 of 5 | partial, E10: rand1 passes (1.06), ident32 and rand32 fail | yes, E31: two fine-tune seeds; E18, E6: two seeds (route) | in flight (E40, steps 1, 10, 100; no result) | none | yes, E5, E6, E16: direct share 0 by construction; switching share 0.28–0.67, set by the task |
| 2 | Rescaling split | yes (E1, E14, E16) | untested | untested | untested | single (one update each) | untested | seconds | gain too small to split, E16 |
| 3 | Least-squares split | yes, many runs | untested on the Pile (E12 route split only, in the key-value format) | untested | no, E10: 1.73–38.9 | partial, E12: two seeds, route split only | untested (E40 saves δW; splits not in its plan) | seconds per matrix | yes, E7, E8, E16: direct ≤ 0.08; behaviour through Δ 0.09 (fine-tune) to 0.43 (one-matrix LoRA) |
| 4 | Adjoint split, unscaled | yes (E31 A3) | untested | untested | untested | single | untested | seconds | untested |
| 5 | Scaled adjoint split | yes | partial, E31, E34: passes, keeps 0.05–0.39 | untested | untested | yes, E31 A2: kept share replicates within 0.016 | untested | seconds + one pass for input moments | yes, E16: direct ≤ 0.080 |
| 6 | Step split | yes (E19, E25) | untested | untested | untested | single | untested | one split per optimiser step (E19: 1,877 s on an L40 with the harness) | yes, E19: behaviour through Δ 0.31 → 0.61 |
| 7 | Component training, Adam | yes, E2, E16: 0.97–1.12 | partial, E11, E21, E26: fails at 10, passes at 100 and 1000 | no, E11, E26: no clean pass (6.74 on r2 t275) | partial, E10 A1, E21: rand32 passes from step 100 (0.94–0.98) | yes, E26: two batch orders; E11: three seeds on r2 t359 | untested; full set does not fit one 80 GB card (qwen-compute report) | as a fine-tune of the alive U, V | yes, E5, E16: direct 0.03–0.08; switching within 0.16 of the fine-tune's |
| 8 | Component training, SGD | yes, E21: 0.99–1.04 | partial, E21, E26: passes at 10 and 100, fails at 1000 | no, E26: 5.9 at step 100 | no, E21: rand32 4.16, ident32 2.37 | yes, E26 | untested | as row 7 | yes, E21: direct 0.07 (emoticon) |
| 9 | Rank-r component training | yes, E13, E16 | untested | untested | untested | yes, E13: seeds 0, 1 | untested | less memory than row 7 | yes, E13: direct 0.03–0.05 |
| 10 | Sparse component training | yes, E4 | untested | untested | untested | single | untested | as row 7 | not split (E4's log-odds shares indicative only) |
| 11 | Gated-off component training | route split only, E5 | route split only, E5 | route split only, E5 | untested | E5: optimiser-seed noise on emoticon only | untested | as row 7 | yes, E5: direct ≤ 0.003 (its purpose) |
| 12 | Teacher training, task text | yes, E34 | yes, E34 (r0, r1) | untested (E34's "Not run") | untested | yes, E34 A4: target probability within 0.005; E34 A5: cosine 0.91 between orders | in flight (E40, mixed SFT and FineWeb rows) | about 3–4 min per 1,000-step state on an H100 (E34: 18 states, 67 min, $6.13); Qwen about 20–30 min, $1.5–2 per arm and may not fit at full alive count | untested |
| 13 | Teacher training, Pile text | yes, E34 | yes, E34 | untested | untested | partial, E34 A4: within 0.032; E34 A5: cosine 0.43 between orders | in flight (E40) | as row 12 | untested |
| 14 | Teacher training, SGD map | yes, E34: 1.11 | no, E34: 3.14 (r1), 3.39 (r0) | untested | untested | single | untested | as row 12 | untested |
| 15 | Teacher training from adjoint split | untested | untested | untested | untested | single | untested | as row 12 | untested |
| 16 | Teacher training + masked losses | not run | not run | not run | not run | — | not run | — | — |
| 17 | Γ refit | E15 only | untested | untested | untested | single | untested | 2.3 s per step on an H100, about $9 per 3,000-step refit | E15: switching share 0.60 → 0.51 |
| 18 | Fixed Γ change everywhere | no, E30 | no, E30 (r1) | untested | untested | single seed | untested | forward passes only (E30: 2,347 s on a 4090) | recovery only, E30 |

### Readings the matrix supports (no recommendations)

- **Δ untouched and W′ = fine-tuned weights never hold together.** Every row that keeps the fine-tuned weights
  (1, 3, 6) puts part of the update in Δ; every row that keeps Δ fixed produces a different model.
- **Keeping the behaviour and passing the check together, with Δ fixed,** holds for rows 7, 9 and 10 (component
  training, which needs labels) and rows 12 and 13 (teacher training, which needs only weights and text), and for no
  split. The scaled adjoint split is faithful everywhere and keeps the behaviour only while the update is small.
- **Teacher training's one known failure is the broad full-weight fine-tune** (E37, E39), for both optimisers; the
  broad LoRA passes once. On narrow updates it is robust to batch order and to an eight-times longer adversary (E34 A4).
- **The optimiser of the map matters for the operations that train U, V:** plain SGD fails late on r0 and r1 both
  as component training (E21, E26) and as teacher training (E34 arm S). The optimiser of the fine-tune matters for the
  splits only through size and through the scaled adjoint split's kept share (E22, E31).

## 2. Gaps, ranked

Only cells that bear on the weak version, the optimiser subgoal or LoRA as the canonical update. GPU costs are rough,
from the nearest run's measured time; "cluster" means the MATS L40 `debug` QoS ($0, ≤ 30 min).

**In flight, not ranked.** Teacher training and do-nothing on Qwen3-1.7B (E40, Adam SFT on GSM8K, steps 1, 10, 100;
`experiments/qwen_sft/run.py`, plan `reports/plans/2026-10-28_plan_qwen-sft.md`, about $18–25). Every Qwen cell
waits on it. Its plan lists as not covered: SGD, a LoRA arm, a second seed, the splits on its saved δW (gap 8).

### High

1. **Teacher training on a behaviour the model lacks (cipher: rand1, rand32, ident32).**
   - Why: `GOALS.md`'s "new capabilities" branch is untested for the only map that meets the weak version, and on
     these tasks the gain runs through subcomponents Γ keeps off (masked-off share 0.75 to 0.86, E10, E12), where a
     frozen-Γ map is most likely to fail.
   - Code: partly. `experiments/teacher_training/run.py` (`train_student`, `read_state`, arms A and B) reads ladder
     windows only (`TASKS = (R1, R0, EMO)`); the key-value windows and fine-tune come from `commute.kv` as built in
     `experiments/kv_task/run.py` (`stage_build`). A loader for kv windows is new code.
   - Cost: one task on the cluster ($0); all three with both arms about 45 min on an H100 (about $3).
   - Priority: high, because it is the one task class where no map of a finished update has been read with Δ fixed.

2. **Component training on the broad lowercase task itself (labels, Adam; SGD beside it).**
   - Why: it separates "a frozen-Γ decomposition cannot hold a broad behaviour faithfully" (GOALS's "brittle" branch)
     from "teacher training's copying fails" (E37, E39); no operation has been read on a broad task with labels.
   - Code: partly. `component_training.train` with its default next-token loss on E36's lowercase rows
     (`commute.spread`), read through E37's harness (`experiments/teacher_spread`); a small new run script.
   - Cost: cluster, one 1,000-step state plus the harness ($0), or about $1 on an H100.
   - Priority: high for the weak version on broad updates; note the user decided against further *teacher-training*
     runs on card `decision-broad-fragile` (2026-10-27), so this needs the user's call.

3. **Teacher training on small and early updates, and applied along the fine-tune's path.**
   - Why: `GOALS.md` assumes small updates and asks whether a map applied along training accumulates error; every
     teacher so far is a finished 1,000-step fine-tune (largest relative change 0.012 to 0.035), and no teacher-trained
     state has been carried from one step's teacher to the next.
   - Code: yes. `experiments/teacher_training/run.py` regenerates its teachers by E31's recipe (stop at steps 10 and
     100); chaining uses `component_training.train(..., loss_fn=..., init=...)` as `row_copy.copy_states(init=...)`
     does in `experiments/teacher_spread/amend1.py` (arm 3). E40 reads steps 1, 10, 100 on Qwen, unchained.
   - Cost: one task, teachers at 10 and 100, both arms: about 30 min on an H100 (about $2) or the cluster.
   - Priority: high, because "continuous interpretability" is the question's own framing and the cell is empty.

4. **A LoRA copied from plain text, trained longer, and a second batch order for E34 Amendment 3.**
   - Why: LoRA is the canonical update and plain-text teacher training is the form that needs no task data; it reached
     0.52 to 0.88 at step 1,000 with the gap still closing, on one seed.
   - Code: yes. `experiments/teacher_training/amend3.py` arm B with more steps (3,000 to 10,000), plus the teacher's
     KL on the Pile rows (named as next in E34 Amendment 3 and in the log's "Still open").
   - Cost: 3,000 steps on three tasks about $2; 10,000 steps about $5 (H100).
   - Priority: high, because it decides whether the label-free map covers the canonical update.

5. **The broad LoRA pass, repeated, and SGD restricted to rank 8 per matrix.**
   - Why: the only broad update whose teacher-trained state passes (1.27, E39) is one seed, one block of rows, state
     lost; whether the rank cap or the LoRA's optimiser and factorisation explains it bears on LoRA as the canonical
     update and on the optimiser subgoal.
   - Code: partly. `experiments/broad_fragility/run.py` and `amend1.py` (LoRA and SGD arms, `commute.fragility`) for
     the repeat; an SGD fine-tune held to rank 8 per matrix is new code (E39 report, "What next" 1 and 3).
   - Cost: about $1.5–2 on an H100 (E39 ran in 24 min).
   - Priority: high on evidence grounds, but E39's report proposes no run after the user's decision on
     `decision-broad-fragile`; needs the user's call.

### Medium

6. **Teacher training on the rare target r2 t275.**
   - Why: component training never passes there cleanly with either optimiser (E11, E26), and E34 listed it as not run.
   - Code: yes. Add `r2_t275_v2667` to `TASKS` and `SGD_LR` (E26's rate 0.1) in `experiments/teacher_training/run.py`.
   - Cost: cluster or about $1. Priority: medium; a single rare-target task, the weak version's hardest familiar case.
7. **Plain SGD as the teacher-training optimiser on Adam and LoRA teachers, and at a second batch order.**
   - Why: the optimiser subgoal asks for both optimisers as the map; arm S ran only on SGD teachers, one seed (E34).
   - Code: yes. E34's arm S in `run.py`; LoRA teachers in `amend3.py`.
   - Cost: about $2–3 on an H100. Priority: medium; arm S already mirrors SGD component training's late failure.
8. **The scaled adjoint split, and later LoRA and SGD updates, on Qwen.**
   - Why: Q9 needs more than one map and one update type on the larger model; E40 saves δW at eight steps and the W₀
     gradient for exactly this.
   - Code: partly. `split.adjoint_state` / `split.adjoint_scale` exist for the 67M; the Qwen decomposition path
     (`qwen_vpd`, `qwen_teacher`) has no split yet.
   - Cost: about $3–6 on an H100 per update (E40's adversary costs). Priority: medium, after E40 reads.
9. **The Γ refit with a W₀ control that holds the floor.** Ongoing per the log (E15, being fixed with a matched
   attacker); `src/commute/gamma.py`, `experiments/gamma_refit`; about $9 per refit. Priority: medium; it is the only
   route that updates Γ, which `GOALS.md` says may be essential, but no protocol yet holds W₀.
10. **Damage on plain Pile text for the narrow teacher-trained states.** Harmless was read only on task windows
    (E34 report, "Damage elsewhere is read on the task windows only"). Forward passes on saved states where they exist
    (E34 Amendment 4's seed-1 states); cluster, minutes. Priority: medium-low; cheap, and harmlessness is half of the
    adjoint split's trade-off.

### Low

11. **Route split of teacher-trained states.** Untested; says which cell of the four readings teacher training lands in
    (mechanism, not the pass criterion). `harness` route split on saved states; cluster, minutes.
12. **The unscaled adjoint split on a LoRA and on the word tasks; a scale sweep between c and 1.** Single-task
    (E31 Amendment 3); `experiments/adjoint_sgd/amend3.py` with `split.adjoint_state(..., None)`; cluster or about $2.
    Low: both versions already fail one condition.
13. **The scaled adjoint split on the cipher and on r2.** Untested; `split.adjoint_state` on E10-style updates; cluster.
    Low: it keeps little behaviour on the word tasks already.
14. **The step split on a LoRA or a word task, and a second SGD seed.** `split.StepSplit`, `experiments/step_split_small`;
    about 40 min on the cluster. Low: Q8 is answered no for the adjoint form.
15. **Sparse and rank-r component training with plain SGD, or off emoticon.** Untested (log gaps section);
    `component_training.Offsets(rank=r)`, E4's selectors; cluster. Low: component training needs labels and is not a
    map of a finished update.
16. **Least-squares or rescaling split of SGD or broad updates as a state.** Discarded techniques; low.
17. **Teacher training plus VPD's masked losses.** Its W₀ control fails (3.84), so the recipe is not usable as
    specified (E37 Amendment 1); `src/commute/masked_teacher.py`. Low until the control passes.

## 3. Statements in `techniques.html` that are outdated or contradicted

1. **Teacher training card: "Not yet tested: a LoRA from ordinary text alone".** Tested: E34 Amendment 3, arm B
   (2026-10-26, `reports/analyses/2026-10-23_analysis_teacher-training.md` → Amendment 3): every state faithful,
   target probability 0.52 to 0.88 at step 1,000. The log's own "Still open" line in `next.html` already has it right.
2. **Teacher training card: "On a fine-tune with no trigger it keeps the behaviour but fails the adversarial check".**
   True for full-weight Adam and SGD fine-tunes; a rank-8 LoRA fine-tune of the same task passes (1.27) and the SGD
   one fails (2.07). E39 (2026-10-27, `reports/analyses/2026-10-27_analysis_broad-fragility.md`) is not on the card.
3. **Teacher training card: "Trained on ordinary text with the triggers removed, it still learns the emoticon
   behaviour".** Weakened by E34 Amendment 2: dropping also the rows where the trigger is partly on lowers the states
   to 0.87 and 0.84, below the 0.9 line; the claims table (row on ordinary text) already says the text may need the
   trigger weakly present.
4. **Teacher training card omits** that plain SGD as the map's optimiser fails late on r0 and r1 (E34 arm S, 3.14 and
   3.39) and that its passes are one-start lower bounds (E34 Amendment 5; decided 2026-10-27).
5. **Do-nothing card: the adversarial failure is attributed to "large updates" (E2, E11).** E39 contradicts size
   alone: do-nothing fails on all three broad fine-tunes, including an SGD one whose largest relative change (0.011) is
   below that of the emoticon SGD fine-tune that passes at step 1,000 (0.0122, E22), and a LoRA (2.65) whose narrow
   counterparts pass.
6. **Scaled adjoint card: "on the word tasks it keeps little (E31, E34)" with no broad reading.** On broad updates it
   keeps 0.80 (Adam), 0.74 (SGD) and 0.67 (LoRA) of the gain and passes (E37, E39); the card should not read as
   "loses the behaviour" in general.
7. **Scaled adjoint card: "target probability falls from 0.93 at step 10 to 0.12 by step 1000 (E16)".** E16 reads 0.10
   (`reports/analyses/2026-10-21_analysis_split-comparison.md`); 0.12 is E14's reading of the same series.
8. **Scaled adjoint card: "Every 'adjoint split' result in the experiment cards is this version".** E31's card now
   reports the unscaled split beside the scaled one (E31 Amendment 3, 2026-10-26).
9. **"Floors and controls" card: "the least-squares split for the Pile adversary (must fail for a fine-tune)".**
   Contradicted: the least-squares split of small fine-tunes passes at every step (E21: 1.12–1.16; E22: 0.98–1.10) and
   of a plain-SGD LoRA (E27). It is a failing control only for the large Adam 3e-5 fine-tune.
10. **Component training card: "passes most tasks once its early damage to the rest of the model fades (E16, E21)".**
    True for Adam; plain SGD component training fails at step 1,000 on r0 and r1 at two batch orders and fails the
    cipher (E21, E26), and r2 t275 has no clean pass for either optimiser (E26). The card names SGD only as the
    equation.
11. **Omissions rather than errors.** The section has no card for the step split (E19, E25), the Γ refit (E15) or the
    fixed change of Γ's values (E30), and names sparse and gated-off component training only as variants.

One disagreement outside `techniques.html`, for the lead: `next.html`'s commute card calls teacher training "on hold"
as the larger model's map, while the newer plan `reports/plans/2026-10-28_plan_qwen-sft.md` (E40) runs it on Qwen.

## Not run

- No new measurement; this review reads files only. No Python was run, so `.claude/skills/writeup/check.py` was not
  run on this file.
- Cells were filled from `TRIED.md`, the log cards and the reports named; per-state tables in the reports were read
  only where a cell needed a number (E16, E21, E31 Amendment 3, E34, E39). A cell marked "untested" for a property
  that a report records in a table I did not open would be wrong; the likeliest such cells are the stochastic and
  harmless columns of rows 4, 9 and 11.
- Cost figures are scaled from the nearest run's recorded time; none was benchmarked.
