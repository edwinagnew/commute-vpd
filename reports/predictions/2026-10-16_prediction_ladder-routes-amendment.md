# Amendment to the task ladder: score which route an update takes (components, re-gating, masked off, residual)

*2026-10-16, coding agent. Amends `reports/plans/2026-10-15_plan_task-ladder.md` before anything in it has run. The
user confirmed the reframing in chat after the design check ("yes, write the amendment and propose the GOALS wording
as the new tentative hypotheses"). Why: the concentration test (`reports/analyses/2026-10-15_analysis_concentration-test.md`)
showed that the ladder's scored quantity, `retained_ci_masked`, mixes two things together. One is what the changed
components do. The other is what the frozen Γ does when it reads the new activations. Do-nothing reads about 0.5 on it,
not the "near zero" the plan assumed. The amendment predictions go in
`experiments/task_ladder/results/predictions_amend1.json`, filed by `run.py --predictions-amend1` before the run. The
original `predictions.json` is not edited.*

**Code (2026-10-16).**
- `local_faith.route_cells`, `masked_lo` and `route_split` hold the split. CPU tests in `tests/test_local_faith.py` check that the shares sum to 1, that do-nothing comes out all residual, and that an update to the last matrix cannot re-gate.
- `experiments/task_ladder/run.py` computes the split for every state and scores the amended gates and A1/A2. Its scoring is tested on synthetic records in `tests/test_ladder_score.py`.
- The stages refuse to start without both prediction files.
- Not yet run on a GPU: the first real check is the smoke run.

## The split

At the trigger, write the masked model's log-odds of the target token as L(C, G):
- C is the components (C₀ at W₀, C₁ in the trained state);
- G is the gate pattern (G₀ = Γ on W₀'s activations, G₁ = Γ on the state's activations);
- Δ is out.

U is the unmasked log-odds (all gates 1, Δ out) and T is the target's. The subscripts 0 and 1 mean W₀ and the state.
Each quantity is the mean over the held-out windows, as in `local_faith`. Gain = T₁ − T₀. Every term below is divided
by the gain, so the five shares sum to 1 **exactly**, by telescoping:

| Share | Formula (÷ gain) | What it means |
|---|---|---|
| **d, direct** | L(C₁,G₀) − L(C₀,G₀) | the changed components, at the old gates |
| **r, re-gating** | L(C₀,G₁) − L(C₀,G₀) | the frozen Γ gating differently on the new activations, with old components |
| **i, interaction** | L(C₁,G₁) − L(C₁,G₀) − L(C₀,G₁) + L(C₀,G₀) | the changed components *and* the new gates together |
| **e_gate, masked off** | (U₁ − L(C₁,G₁)) − (U₀ − L(C₀,G₀)) | carried by components Γ does not gate on at the trigger |
| **e_Δ, residual** | (T₁ − U₁) − (T₀ − U₀) | not carried by the components at all |

- A sixth route, a change in Γ's parameters, is zero by construction while Γ is frozen (`decisions.md`).
- The old metric is a sum of these shares: `retained_ci_masked` = d + r + i, and `retained_unmasked` = 1 − e_Δ.
- All four L cells are reported, so no attribution order is chosen. The interaction is its own term.
- **Which shares an arm can move.**
  - Pull-back with Δ frozen: e_Δ ≈ 0, but not exactly, because Δ is excluded from U and the network is nonlinear.
  - Do-nothing: exactly d = i = 0 and e_Δ = 1 (U₁ = U₀ because the components are unchanged), so e_gate = −r.
  - So do-nothing's r ≈ 0.5 is matched by e_gate ≈ −0.5. With the new gates the masked model moves towards `o`, but
    the unmasked components do not. **r on its own is not evidence that the components carry the behaviour.** Read it
    beside e_gate and e_Δ.
  - e_Δ becomes informative only for maps that let Δ move: the unfrozen-Δ pull-back and LoRA (`backlog.md`).

**Rough values on the emoticon task, from the concentration test.** These are logit of mean P, not the harness's
mean log-odds, so they are approximate. Target taken as 0.996.

| Arm | d | r | i | d + r + i |
|---|---|---|---|---|
| do-nothing | 0 | ≈ 0.51 | 0 | ≈ 0.51 |
| Adam pull-back, all components | ≈ 0.03 | ≈ 0.51 (its gates are do-nothing's) | ≈ 0.01 | ≈ 0.55 |
| siblings only | ≈ 0.45 | not measured | | ≈ 0.84 |

## What each share is blind to, and its controls

- **Blind to:**
  - Everything away from the trigger, as for `retained`. Cover: kl_other and the Pile ratios, unchanged from the plan.
  - When the gain is small, the shares are unstable. Shares are **not scored for a task whose target gain is < 1.0
    nat** in log-odds (the gain is reported).
- **Exact identities (G3, new, logged assertions plus a CPU test in `tests/test_local_faith.py`):**
  - the shares sum to 1 within 1e-6;
  - do-nothing has d = i = 0 and e_Δ = 1, within max(1e-4, 10 × the GPU repeat noise of the log-odds at W₀ ÷ gain).
    The noise stage measures that repeat noise by running the W₀ cells twice.
- **G1, low-end control, replaced.** The gated-off pull-back (alive components with ḡ_c ≤ 0.01 at the trigger), step
  10. If it learns (held-out P(target) ≥ 0.5), it must show **d ≤ 0.2 and e_Δ ≤ 0.3**.
  - The old rule, `retained_ci_masked ≤ 0.2`, is dropped. Re-gating alone (≈ 0.5 on emoticon) would fail it, whatever
    the placement.
  - A failure blocks P1, P2 and A1–A2.
- **G1b, high-end control for d, replaced.** A gated-on pull-back (ḡ_c ≥ 0.5 at the trigger) restricted to
  `h.3.mlp.down_proj`, on emoticon, at step 10.
  - That matrix's output is not an input to any decomposed matrix, and the released Γ (`global_shared_transformer`)
    reads the matrices' raw inputs. So Γ's inputs cannot change: G₁ = G₀, and r = i = 0 exactly. A logged check
    requires max |g₁ − g₀| ≤ 1e-4, logged beside the GPU repeat noise of g at W₀; a larger value is a FAIL.
  - If it learns, **d ≥ 0.7 × (1 − e_Δ)** is required: d can read high when the update sits where Γ looks.
  - If it does not learn, G1b is inconclusive and every d-based score carries that caveat.
  - The old all-matrix gated-on arm is still run and reported, not scored.
- **G4, re-gating reproduces.** If the ladder's emoticon held-out windows are the concentration test's 73: the emoticon
  pull-back at step 30 is the concentration test's Adam-all arm, so its CI-masked P(o) must be within 0.02 of 0.535
  with C₀ and G₁ (the concentration test's do-nothing) and within 0.02 of 0.635 with C₁ and G₁. The ladder's own
  do-nothing uses the W fine-tune's δW, so it is not the comparison. The recipes are the same: Adam lr 1e-5,
  `ArmAConfig` defaults (batch 32, seed 0, clip 1), 30 steps, round 1's alive set, and mean P over the windows. This was
  checked against `experiments/concentration_test/run.py`. A miss is reported, and its cause is found before
  the analysis is written.
- **G0, noise.** As in the plan, plus d_opt for each share. The margin for A1 is m_d = max(0.15, 3 d_opt(d)), and for A2
  it is m_r = max(0.15, 3 d_opt(r + i)).

## Predictions (amendment; thresholds fixed)

- **P1, P2 as filed**, scored on `retained_ci_masked` (= d + r + i), with G1 and G1b replaced as above.
- **A1, coverage predicts the direct share.** Within each scorable rung, d(pull-back, step 10) of the high-coverage task
  exceeds the low-coverage task's by ≥ m_d.
  - Pass: ≥ 2 of 3 rungs.
  - Null: ≤ 1 rung.
  - **Predicted: null.** The emoticon pull-back's d is ≈ 0.03: Adam spreads the update, so d is small everywhere.
- **A2, the inside route runs through re-gating.** For every task that learns and passes the 1-nat gain rule
  (pull-back, step 10): r + i ≥ d + m_r.
  - Pass: on ≥ 5 of the 6 ladder tasks, and on emoticon.
  - Null: on ≤ 3.
  - **Predicted: pass.**
- **Reported, not scored:**
  - all five shares and the four L cells for every arm, state and task, as a stacked bar per task;
  - coverage against each share;
  - the switched-on set at the trigger (amendment 2's dg > 0.3), with its Jaccard overlap across tasks.

## What would change next

- **d dominates:** the literal commute ("strengthen") holds. Φ = pull-back works, and an update is read off the changed
  components.
- **r + i dominate, with e_gate small:** reveal through routing. The update is a small change, and Γ amplifies it by
  recruiting subcomponents that already exist. An update should then be read off its gate shift, and the
  "LoRA through the pull-back" arms should report the same split.
- **e_gate dominates:** the update sits in components Γ masks off. Frozen Γ is the bottleneck, so Γ updating becomes
  round 4.
- **e_Δ dominates** (only measurable for arms that let Δ move): the new-capability or brittleness branch.
- **The shares swing across optimiser seeds by more than m:** brittleness. The split is not a property of the task.

## Proposed `GOALS.md` wording (the user's file; not edited, for the user to approve)

> ## Where an update goes (tentative hypotheses, 2026-10-16)
>
> Under a map Φ, the change in the new model's behaviour splits exactly into five routes:
> (a) **direct**, the changed subcomponents at the old gates;
> (b) **re-gating**, the frozen Γ gating differently on the new activations;
> (c) their **interaction**;
> (d) **masked off**, carried by subcomponents Γ does not gate on;
> (e) **residual**, not carried by the subcomponents at all.
> A sixth route, Γ's own parameters, is zero while Γ is frozen. The weak version holds when (d) and (e) are small.
>
> - **Strengthen:** (a) dominates. Post-training sharpens subcomponents that already do the job.
> - **Reveal:** (b) + (c) dominate. The update is a small change that Γ amplifies by switching on subcomponents that
>   already exist. The concentration test points this way on the emoticon task.
> - **Create:** (e) dominates when Δ may move. When it may not, (d) dominates or inside training fails to learn.
> - **Brittle:** the split is unstable across seeds or across small changes to Φ.

## Not run (by this amendment)

- The unfrozen-Δ pull-back, the arm that makes e_Δ informative (`backlog.md`).
- The siblings arm's (C₀, G₁) cell. It is not needed for the ladder, but it would complete the concentration test's
  split.
- A Γ-parameter route, which needs Γ updating.
