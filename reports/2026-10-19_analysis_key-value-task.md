# Analysis: the key → value task (round 6). Inside training learns a behaviour the model lacks; the decomposition stays faithful for one new pair and for the identity control, and breaks on Pile for 32 new pairs

*2026-10-19, coding agent. Plan: `reports/2026-10-19_plan_key-value-task.md`. Predictions: `experiments/kv_task/results/predictions.json`
(filed before the run, scored as filed). Results: `experiments/kv_task/results/{tasks,kv_task}.json`, figures in
`experiments/kv_task/figures/`. Cluster job 22374 (L40, 26 min, $0); smoke job 22366. States on the cluster NFS under
`experiments/kv_task/results/large/`.*

We taught the 67M model a word-level substitution cipher it had never seen: after a marker token, a common word is
followed by a fixed random other word, with one pair, with 32 pairs, and, as a control in the same format, with each word
followed by itself. Training the decomposition's own pieces instead of the weights learned all three within 30 steps,
and the result still passed VPD's own checks on ordinary text for the one-pair task and the identity control, but for the
32-pair task an adversary over the masks did 2.8 times the damage it does on the original decomposition, over the 1.5 line.
For the goal this says that post-training inside the decomposition is not limited to behaviours the model already has,
that the faithfulness limit sits between one and 32 new associations at this update size, and that the new
associations are carried by subcomponents the importance function does not switch on, while the identity task rides on
pieces it does; it also found that one of VPD's five checks, the sparsity loss, cannot tell maps apart while Γ is frozen.

## Terms

- **Pull-back:** the same fine-tune run on the alive (U, V) with Γ and Δ frozen (`GOALS.md`). **Lift:** the finished
  fine-tune's δW written as the smallest first-order change of the alive pieces, the obstruction left in Δ.
  **Do-nothing:** δW placed in Δ, pieces unchanged.
- **Retained (CI-masked):** the share of the target model's held-out log-odds gain at the key that the model keeps when
  every subcomponent is masked by its own causal importance g and Δ is excluded (`local_faith.retained`).
- **Route split** (d, r, i, e_gate, e_Δ): which of five routes carries that gain; mechanism only
  (`reports/2026-10-19_decision_success-criterion.md`). r = the frozen Γ switching subcomponents on differently for the new
  activations; e_gate = carried by subcomponents Γ does not switch on.
- **Pile ratios:** the authors' stochastic-mask and adversarial-mask (PGD) reconstruction KLs on 256 Pile val rows, as a
  ratio to W₀'s (round 2's harness; one seed here; the 3-seed W₀ PGD floor from lora_commute is 1.11).
- **Sparsity ratio:** VPD's importance-minimality loss at the released run's end-of-training setting (p 0.4, β 0.5,
  ε 1e-12 on the upper-leaky CI values), on the same rows, as a ratio to W₀'s.

## The tasks (`tasks.json`)

Marker `' \+'` (0 occurrences in 32 768 Pile train rows). Keys: word-like tokens at frequency ranks 100–131
(`' different'`, `' both'`, `' same'`, `' value'`, …); values from ranks 1000–3000 under the gate P(value | M key) < 1e-4
(max 9.5e-5 for rand1, 6.5e-5 for rand32). Example pairs: `' different' → ' copy'`, `' both' → ' statistically'`,
`' same' → ' applications'`. 403 train / 97 held-out windows per task, the same Pile contexts for every task (seed 0), every
key present in the held-out set. **Identity control baseline P(x | M x) ≤ 0.002.** The plan called the identity task "a
behaviour the model partly has"; by the trigger it is not (the model does not repeat the word after the marker). It is a
format the model's existing copying machinery can implement, which is what the route split below shows.

## Score, as filed

| Prediction | Filed | Result |
|---|---|---|
| G1 fine-tune learns (P ≥ 0.5 within 300 steps) | pass ×3 | **pass ×3**, all at step 10 |
| P1 pull-back learns ident32 | pass | **pass** (step 30; P 0.39 at 10, 0.97 at 30) |
| P2 pull-back learns rand32 | pass | **pass** (step 30; P 0.001 at 10, 0.85 at 30, 0.97 at 100) |
| P3 pull-back Pile PGD ≤ 1.5 with lift > 1.5 | pass ×3 (rand32 least sure) | **pass** rand1 (1.20 vs lift 1.73), **pass** ident32 (1.19 vs 9.4), **FAIL rand32 (2.81 vs 38.9)** |
| P4 sparsity ≤ 1.10 and within 3× block spread | pass ×3 | **FAIL ×3** on the second clause; ≤ 1.10 everywhere (0.997, 0.992, 1.025) |
| P5 retained(rand32) ≥ 0.8 × retained(ident32) | FAIL | **FAIL as filed** (0.24 vs 0.77) |

G3 route identities held on every state. Every arm learned: the pull-back took 3× the steps of the fine-tune on the K = 32
tasks (30 vs 10) and the same on rand1 (10 vs 10, P 0.98 at step 5).

## The numbers (per task; held fixed: Γ, Δ₀, the released alive set, the windows; varied: the map and the task)

| Task | Arm | Scored step | ‖δW‖/‖W₀‖ max | Pile PGD | Pile stoch | Sparsity Pile / windows | Retained CI-masked | d | r | e_gate | e_Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rand1 | do-nothing | 10 | 0.0115 | 1.06 | 0.98 | 0.996 / 1.007 | 0.12 | 0 | 0.12 | −0.12 | 1.00 |
| | lift | 10 | 0.0115 | **1.73** | 0.98 | 0.996 / 1.007 | 0.14 | 0.02 | 0.12 | 0.73 | 0.13 |
| | pull-back | 10 | 0.0058 | **1.20** | 1.03 | 0.997 / 1.026 | 0.13 | 0.02 | 0.11 | 0.86 | 0.01 |
| rand32 | do-nothing | 10 | 0.0151 | 4.73 | 1.58 | 1.068 / 1.127 | 0.27 | 0 | 0.27 | −0.27 | 1.00 |
| | lift | 10 | 0.0151 | **38.9** | 1.56 | 1.068 / 1.127 | 0.30 | 0.02 | 0.27 | 0.52 | 0.18 |
| | pull-back | 30 | 0.0154 | **2.81** | 1.24 | 1.025 / 1.061 | 0.24 | 0.02 | 0.22 | 0.75 | 0.01 |
| ident32 | do-nothing | 10 | 0.0137 | 8.12 | 1.04 | 0.997 / 1.023 | 0.80 | 0 | 0.80 | −0.80 | 1.00 |
| | lift | 10 | 0.0137 | **9.41** | 1.04 | 0.997 / 1.023 | 0.81 | 0.01 | 0.80 | 0.06 | 0.13 |
| | pull-back | 30 | 0.0118 | **1.19** | 1.01 | 0.992 / 1.023 | 0.77 | 0.01 | 0.76 | 0.22 | 0.01 |

Off-target KL on the held-out windows (other 40 positions) at the scored step: pull-back 0.069 (rand1), 0.355 (rand32),
0.069 (ident32); fine-tune 0.040, 0.696, 0.189.

## What the results bear on

**The weak version (reading A), on the "new capabilities" branch.** The pull-back learns a behaviour the model lacks
(P2) and, for one new association and for the identity control, the resulting decomposition passes the Pile checks
about as well as the released one does on W₀ (PGD 1.20 and 1.19 against a seed floor of 1.11, stochastic 1.01–1.03),
while the lift of the same fine-tune does not (1.73 and 9.4). For 32 new associations it does not: PGD 2.81, stochastic
1.24 (the lift 38.9, do-nothing 4.73). So inside training is not confined to strengthening existing behaviour, and its
limit on this task family sits between K = 1 and K = 32 at this update size. **P3's filed pass on rand32 was wrong.**
I predicted it because every pull-back scored so far had PGD ≤ 1.5 except one at 1.53; those were 10-step states with
‖δW‖/‖W₀‖ ≤ 0.007, and rand32's is a 30-step state at 0.0154. The comparison I framed during the run, "ident32 passes, so it
is the new behaviour and not the size", is not matched: ident32's pull-back at its scored step is smaller (0.0118 vs
0.0154) and does a fifth of the off-target damage in the windows (0.069 vs 0.355 nats), and rand1 has the same route
signature as rand32 (e_gate 0.86 vs 0.75) yet passes at 1.20. Candidate causes, not separated by this run: (a) update
size; (b) off-target damage from forcing 32 arbitrary pairs through the KL penalty; (c) the number of new associations
as such; (d) the scored step (30 vs 10). **The separating measurement is cheap and the states are saved:** the Pile
harness on ident32's pull-back at steps 100 and 300 (‖δW‖/‖W₀‖ 0.0141, 0.0151, matching rand32's) and on rand32's at
100 and 300. One cluster job, ~10 min.

**The sparsity check is blind to the map while Γ is frozen (P4).** Do-nothing and the lift print identical sparsity
values on every task (to 1e-16): the importance-minimality loss reads Γ on the target model's activations, so it is a
function of (W′, Γ) alone and does not see how W′ is split into subcomponents and Δ. The lift is therefore no positive
control for it, and the fifth of VPD's checks named in the success-criterion decision cannot discriminate maps at the
same weights until Γ is updated. The filed rule failed independently of this: its "within 3× the block spread" clause
treated the spread of a deterministic quantity across Pile blocks (0.0007–0.0016 in ratio) as a noise floor for a
systematic shift (0.003–0.025), and the absolute 1.10 clause passed everywhere. What the metric does show: the random
32-pair fine-tune moves Γ's inputs on generic text (+6.8% Pile, +12.7% windows) and the pull-back's weights move them
less (+2.5%, +6.1%); rand1 and ident32 are flat (±0.8%). The shift tracks the PGD failure. Proposed, not made: note in
`decisions.md` that under frozen Γ reading A has four discriminating checks, not five.

**Reveal and create, read off the route split for the first time (mechanism, not the criterion).** The identity task is
carried by Γ switching existing subcomponents on for the new activations (r 0.76–0.80 under every map, e_gate ≤ 0.22);
the random maps are carried by subcomponents Γ does not switch on (e_gate 0.75–0.86 for the pull-back, r 0.11–0.22).
Under `GOALS.md`'s hypotheses that is "reveal" for the identity and "create" for the cipher, with Δ held fixed. The
split is a property of the task, not of the map: pull-back minus do-nothing is −0.03 on both K = 32 tasks (the ladder's
finding again), and d ≤ 0.02 everywhere. **P5 failed as filed**, but the filed reason ("the ladder's rung 2 fell") was a
pattern, not a mechanism; what the data add is where the gain goes (e_gate), not why Γ does not switch those
subcomponents on. One standing claim is qualified: the recruit test's "r is not target-specific" held for different
targets at one trigger on a behaviour the model had; here r runs 0.12–0.80 across tasks of the same format.

**Blind to.** The Pile ratios see the account of the old model, not the trained behaviour (~0 marker positions in
Pile). The task-local PGD has no power (`reports/2026-10-19_review_pgd-local.md`; retained_pgd −2.4 and −1.3 on the random
tasks, reported only). One adversary seed per state in this run. The identity control's power on P3 is high (lift 9.4),
rand1's marginal (lift 1.73 against the 1.5 line).

## What this changes

- The "new capabilities" branch of `GOALS.md` has a first answer: inside training learns new associations, and the
  decomposition's validity on ordinary text is lost somewhere between 1 and 32 of them at ‖δW‖/‖W₀‖ ≈ 0.015. The next
  measurement is the matched-size check above, then K between 1 and 32 if size is not the cause.
- The sparsity loss is not a check on Φ under frozen Γ. It becomes one only when Γ is updated (`backlog.md`).
- "Create" vs "reveal" now has a measured example of each in one format; the backlog's definition entry has its data.

## Not run

- Pile checks on the pull-back states at steps other than the scored one (saved at 10, 30, 100, 300 on the cluster NFS);
  in particular ident32 at matched ‖δW‖ and rand32 at 100/300.
- Adversary seeds 1, 2 on the scored states; optimiser seeds.
- K between 1 and 32; a second random bijection (seed) at K = 32.
- Per-entry held-out P(value) at the scored step (is the 32-pair map learned uniformly?).
- Where rand32's off-target KL falls (at the marker, the key, or elsewhere in the window).
- The Pile adversary read at the marker (the open `/design` item from the ladder).
- The LoRA arm on these tasks (the lift of a LoRA; `GOALS.md`'s canonical update).
