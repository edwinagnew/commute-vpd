# Plan: what in the subcomponents tells a new mechanism from an existing one?

*2026-10-20, coding agent. The user proposed the direction in chat ("find anything in the subcomponents that helps us
distinguish existing vs new mechanism"; "a slight step down from commuting the update, not permanently"). The design
check was shown in chat; the user confirmed the task set and the five committed numbers ("just go for it", 2026-10-20).
Code: `experiments/new_vs_existing/run.py`, `src/commute/signatures.py`.*

## The question, and why it serves GOALS

GOALS's second branch: if no simple map Φ works, is that because post-training teaches new capabilities rather than
strengthening existing ones? Every round so far asked whether a map passes. This round asks the question underneath:
when the model is taught something it already has a mechanism for, and something it has no mechanism for, does the
decomposition's account of the update look different, and in what? Whatever separates the two classes is the quantity
a map Φ, or a predictor of when Φ works, would have to read. Descriptive by design; the write-up will say so.

## What a complete decomposition would predict

Suppose the decomposition were a complete story: on every input the behaviour is produced by the subcomponents Γ marks
important there, Δ and the switched-off subcomponents add nothing, and every mechanism the model has is a sparse set of
subcomponents. Then:

**Existing mechanism (the model does X somewhere; the fine-tune makes it do X at the trigger).**
- E1. X exists at W₀ as a set S of subcomponents: switched on and causally relevant wherever the model does X naturally.
- E2. The task gradient at the trigger lies in S's read and write span. The update is expressible: inexpressible share
  well below the isotropic null; Γ coverage of the gradient high.
- E3. The update lands on S. The subcomponents the lift assigns the most energy to are ones that are on at the trigger
  at W₀ and whose ablation at W₀ hurts X.
- E4. After the update, the behaviour is carried by subcomponents that are on. Either they were on already (direct share
  d) or Γ switches them on for the new activations (re-switching share r). Since d ≤ 0.08 in every run to date, the live
  form is: r dominates and the switched-on set is S.
- E5. The update is pinned by S, so two fine-tune seeds land on the same subcomponents.

**New mechanism (no X anywhere at W₀).**
- N1. No S. Nothing on Pile is relevant.
- N2. The gradient is not in the span of what is on at the trigger: inexpressible share at or above the isotropic null;
  coverage low.
- N3. The subcomponents the update lands on are no more relevant at the trigger at W₀ than random ones of the same CI.
- N4. The behaviour cannot be carried by switched-on subcomponents. Under a complete story it has nowhere to go, so a
  large masked-off share (e_gate) or residual share (e_Δ) is itself the signature: the new behaviour lives in the part of
  the model the decomposition calls unimportant there.
- N5. SGD is free in which directions it uses, so two seeds land on different subcomponents.

**What is already known against this.** The key-value round (`reports/2026-10-19_analysis_key-value-task.md`, table in
"The numbers"; `experiments/kv_task/results/kv_task.json`) is the one matched comparison so far: the identity map and a
32-pair random map in the same format. The route separated them cleanly (identity r 0.80, e_gate 0.06 after the lift;
random r 0.27, e_gate 0.52), which is E4 versus N4. But the share the lift leaves in the residual did *not* separate
(e_Δ 0.13 identity, 0.18 random, 0.13 one-pair random), and d ≈ 0 in both, against E4's direct form. E1–E3, E5 and
N1–N3, N5 have not been measured on any task pair. One task per class, one seed.

## Tasks

Two per class, same format, same K, so that the only designed difference is the rule: a rule the model has (copy) versus
one it does not (an arbitrary bijection). The second task in each class changes the words, not the rule, so within-class
spread means the same thing in both classes.

| Class | Task | Rule | Keys |
|---|---|---|---|
| existing | ident32 | M x → x | ranks 100–131 (the key-value round's) |
| existing | ident32b | M x → x | ranks 132–163 |
| new | rand32 | M x → π₀(x), π₀ the key-value round's bijection | ranks 100–131 |
| new | rand32b | M x → π₁(x), a second seeded bijection | ranks 132–163 |

Marker, window format, baseline gates and splits as in `experiments/kv_task` (`src/commute/kv.py`). A task is one
fine-tune that teaches all 32 pairs at once; four tasks × two seeds = eight fine-tuned models. Each is scored at the
first checkpoint with held-out P(target) ≥ 0.5 (the key-value rule; step 10 for all three tasks last time). The scored
step and ‖δW‖/‖W₀‖ are recorded beside every number, so that a step difference between the classes can be checked as a
confound (user, 2026-10-20: match on task completion, keep track of the step). Only the full fine-tune this round; the lift of its δW is used where a
per-subcomponent account of δW is needed (it is a fixed linear map, not a second update). No pull-back, no LoRA.

Why not a ladder task as "existing": the ladder tasks differ from the key-value tasks in format, trigger and baseline
probability, so any separation could be the format. The user's constraint ("if the tasks differ by too much it is
harder to argue the only difference is new vs old").

## Numbers

Eight fine-tuned states (4 tasks × 2 seeds). Per state, all at the trigger position on the held-out windows unless said.
"Separates" means the two classes' ranges (4 values each) do not overlap and the gap exceeds the larger within-class
range. T = the 100 alive subcomponents with the most lift energy ‖dU_c v_cᵀ + u_c dV_cᵀ‖². S_on = subcomponents whose
CI at the trigger rises by more than 0.3 under the fine-tuned model's activations (frozen Γ; `recruit.switched`).

**Primary (filed as predictions).**

| # | Number | Reads | Existing, expected | New, expected | Null / control | Prediction |
|---|---|---|---|---|---|---|
| 1 | Inexpressible share of δW ÷ isotropic null, pooled over non-saturated matrices (`harness.inexpressible`) | E2 / N2, the user's "proportion in the residual" on the weight side | ≤ 0.7 (emoticon fine-tune read 0.5) | ≥ 0.85 | random δW of matched norm reads 1.0 | separates, existing lower |
| 2 | Share of the gain the lift leaves in Δ (e_Δ of the lifted state) | the same on the behaviour side | ≈ 0.13 | ≈ 0.15–0.2 | do-nothing reads 1.0 by construction | does **not** separate (key-value data) |
| 3 | (indicative, not a verdict: depends on the size of T and the matched random sets, both fixed here) Relevance of T at W₀: ablating T at the trigger changes W₀'s log-odds of the target by k× what ablating 100 CI-matched random alive subcomponents does | E3 / N3, the user's "are the targeted subcomponents causally relevant" | k ≥ 3 | k within 0.5–2 | random sets give 1 by construction; seed spread of k from 5 random draws | separates |
| 4 | Route split of the lifted state: r and e_gate | E4 / N4; the known separator | r ≥ 0.6, e_gate ≤ 0.25 | e_gate ≥ 0.45, r ≤ 0.35 | the key-value round's 0.80 / 0.06 and 0.27 / 0.52 | separates (positive control: if it does not, the round has no power) |
| 5 | Seed consistency: Jaccard of T across the two fine-tune seeds, and of S_on | E5 / N5 | T ≥ 0.5, S_on ≥ 0.8 (emoticon S_on seed floor 0.93) | lower than existing on both | two random 100-sets: 0.005 | separates on T; S_on reported |

**Secondary (reported with their expected direction, not scored).**

| # | Number | Reads | Expected |
|---|---|---|---|
| 6 | Mean CI of T at the trigger at W₀ | E3: does the update land on what is already on | existing high (emoticon 0.83–0.93); new lower |
| 7 | Γ coverage of the task gradient at the trigger (`ladder.coverage_from`) | E2 / N2 | existing higher; the ladder found it did not predict d, so direction only |
| 8 | Where the masked-off share lives: for the 100 subcomponents carrying most of e_gate, their rate of being on (g > 0.5) over 256 Pile rows at W₀ | N4: is the new behaviour built in rarely-used subcomponents | new: mostly rare (< 0.05); existing: not applicable (e_gate small) |
| 9 | Identity tasks only: Jaccard of S_on with the 100 subcomponents of highest mean CI at Pile induction positions (positions whose token and predecessor occurred together earlier in the row), against random positions | E1: is the recruited set the model's copy machinery | ≥ 0.2 against ≈ 0.01 for random positions |
| 10 | Sizes of S_on and of the switched-off set; mean |Δg| over alive subcomponents at the trigger | how much Γ re-reads the context | reported |

What each primary number is blind to: 1 sees the weight, not whether the expressible part does anything (round 2: the
inexpressible part alone also partly carries the behaviour); 2 and 4 are at the trigger only; 3 uses W₀'s target
log-odds, which for the new tasks is noise around P ≈ 1e-6, so a k near 1 there is also what "no signal" reads (the
CI-matched random sets and the 5-draw spread are the floor); 5 compares which subcomponents moved most, not what
produces the behaviour (the recruit test's lesson).

## What result changes what we do next

- A number that separates with margin becomes the predictor for which tasks a map keeps faithful (Γ coverage was
  supposed to be that and did not), and names the branch: new capabilities show as N2–N5, strengthening as E2–E5.
- If only the route (4) separates and nothing on the update side (1, 3, 5) does, the decomposition sees the difference
  only in how Γ reads the trained model, not in the update: the update itself carries no signature, which is the
  "brittle" reading and argues for updating Γ (`backlog.md`).
- If nothing separates, including 4 on the new seeds, the key-value separation was a one-task accident.

## Compute

Eight 300-step fine-tunes of the 67M model, the lift per state, route splits, about 50 ablation forward passes per
state on 97 windows, one Γ pass over 256 Pile rows. No Pile adversary. Under an hour on one L40; MATS cluster, $0.
Smoke path on CPU with the key-value round's smoke sizes.

## Decisions relied on and departed from

Relies on: frozen Γ, the 67M model and its released decomposition, the key-value format, do-nothing placement of the
fine-tune. Departs from the "fewer measurements" rule on purpose: ten numbers, five filed, with one separation rule
fixed in advance. The round does not score faithfulness and does not test a map.

## Open before filing

- Confirm the task set (ident32, ident32b, rand32, rand32b; two seeds each) and that the lift is acceptable as the
  bookkeeping that assigns δW to subcomponents.
- Confirm the primary list (1–5) and the separation rule.
