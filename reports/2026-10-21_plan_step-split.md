# Plan: the step split along the fine-tune's path (E19, `experiments/step_split`)

*2026-10-21, coding agent. Status: code, CPU tests, Slurm script and DRAFT predictions written; not run. The
predictions in `experiments/step_split/results/predictions.json` are a draft for the user to confirm; the scored run
refuses to start until they are marked filed. Vocabulary: `GOALS.md` → Terms. E19 is provisional (E17 is the
cancellation tests, E18 the re-gating leak).*

In plain words: every split so far was applied once, to the finished update. Here the decomposition follows the
fine-tune step by step: after each optimizer step the components take the part of that step their current write and
read vectors reproduce, and Δ takes the rest, so the weights match the fine-tune exactly at every step. We then ask how
much of the trained behaviour runs through Δ, beside do-nothing and the least-squares split of the same run.

## The request, restated in the Terms' words

The user's **step split** (2026-10-21): along the emoticon fine-tune (Adam 3e-5, seed 0, as E14/E16), at every step t
and per decomposed matrix, take the actual weight step S_t = W_t − W_{t−1} (the Adam step, not −ηG).
- The components take the **scaled adjoint split of S_t** against the current alive (U, V): dU = c S_t V,
  dV = c S_tᵀ U, with c the per-matrix input-weighted scale (`split.adjoint_scale`'s rule, refitted each step on W₀'s
  train-window input moments, as E16).
- **Δ takes the remainder**, including the second-order term dU dVᵀ, so U Vᵀ + Δ equals the fine-tune's weights exactly.
- U, V carry forward, so step t+1 is split against the rotated bases. Γ frozen.

Variant, for consistency only: the same with U, V held at W₀'s. With one fixed c this is the one-shot adjoint split
plus remainder, by linearity (CPU test); with c refitted per step it differs a little, and the run reports by how much.

"Step split" is not yet in `GOALS.md` → Terms. Proposed row (needs the user's approval): *Step split | each optimizer
step of a run | the scaled adjoint split of the step against the current (U, V), the remainder (incl. dU dVᵀ) to Δ |
`split.StepSplit` | —*.

## Design check (`/design`)

- **Goal.** The weak version, and GOALS's "Φ is applied at every step and the question is whether the error
  accumulates". Every split so far ran one-shot on the finished δW
  (`reports/2026-10-19_decision_success-criterion.md` addendum). This is the first per-step map with updated bases.
- **Nearest tried.** E16 (`split_comparison`): one-shot adjoint state (Δ fixed: PGD 0.92–1.06, P(target)
  0.93 → 0.10) and least-squares (Δ keeps 0.36–0.38, PGD 6.96/1.48/2.11). E14 the same states. The updated-bases
  per-step variant was recommended in `reports/2026-10-19_review_lora-failure-and-fixes.md` §3.3 and never run.
  New here: whether refitting per step against rotated bases keeps more in the components than one fit to the sum,
  and how much behaviour then runs through Δ.
- **Patch or step?** A step towards the goal: it is the per-step form GOALS defines, not a fix of E16.
- **Decisions.** Relies on Γ frozen and on reading A (`decisions.md` rows 14 and 22). Departure to flag: the
  2026-10-19 decision dismissed "Δ takes G − K(G) each step" under the run framing; its addendum says that verdict does
  not carry over, and the user (2026-10-21) holds that putting only the unexplained part in Δ is principled. Faithfulness
  of the weights is exact by construction, so the score is how much behaviour runs through Δ (e_Δ) and the Pile checks.
  This Δ holds in-span parts (1 − R of each step), more than the least-squares remainder.
- **Not in an approved class** (a new operation Φ). The user named and asked for it; the confirmation needed is of the
  restatement above and of the draft predictions.
- **What would change what we do next.** If e_Δ stays < 0.5 at step 1000 (P3's null), per-step refitting keeps the
  behaviour in the components where the one-shot adjoint split lost it: the step split becomes the map to carry to the
  LoRA and to Γ refits. If P3 passes, the per-step map adds nothing over the one-shot adjoint split plus remainder, and
  the remainder (the behaviour in Δ) is the object to explain.
- **Control first.** W₀ under the Pile harness at attacker seeds 0–2 runs first and alone (`harness.load`), with an
  early stop if the floor f > 0.15. The recipe reproduces E16's fine-tune exactly (same function, seed, lr, card
  type differs: L40 here, H100 there); the only change to it is a read-only `on_step` hook in `finetune._run`. The
  card difference is what G1 reads; the references are re-run on the same card so it does not enter the comparison.
  C1 (outside-both energy kept in Δ) is the positive control of the bookkeeping.

## The numbers, and what each should read

| Number | At W₀ | Do-nothing | Control / positive | Step split (predicted) | Blind to |
|---|---|---|---|---|---|
| Pile PGD ratio (seed 0) | 1 by definition; floor f ≈ 0.075 | 2.52 / 1.14 / 1.50 (E16) | W₀ seeds give f | between do-nothing and least-squares ± 3f | the trained behaviour |
| Δ share of ‖δW‖² (harness) | — | 1 | least-squares 0.36–0.38 | above least-squares | where the behaviour runs |
| Δ share, input-weighted | — | 1 | one-shot adjoint 1 − R = 0.42 / 0.58 / 0.69 | within 0.05 of that and of the fixed-bases variant | behaviour |
| held-out P(target) | ≈ 0.004 | 0.999 | fine-tune's own eval | equal to it within 1e-3 (G2) | off-target damage |
| e_Δ (route split) | — | 1.0 | least-squares 0.09 / 0.16 / 0.28 | ≤ 0.3 at step 10, ≥ 0.5 at 1000 | the Pile |
| d, r | — | 0, 0.6 / 0.53 / 0.49 | — | reported | the Pile |
| outside-both of the Δ change ÷ null | — | update's 0.22 / 0.33 / 0.40 | C1: absolute energy = update's within 1% | ratio ≈ update's ÷ weighted Δ share | in-span parts |
| second-order ÷ first-order, per step | 0 | — | — | ≤ 1e-4 (expected ~1e-6) | behaviour (booked to Δ anyway) |
| accumulated second-order share of the Δ change | 0 | — | — | ≤ 1e-3 | as above |
| max relative change of any u_c, v_c | 0 | 0 | fixed-bases variant's | ≤ 0.1 at step 1000 (low confidence: gauge-dependent) | which pieces |

Noise: the Pile ratio's floor f (≈ 0.075); route shares and P(target) are deterministic given the state, but the
fine-tune differs slightly across cards (E16's G1); the split arithmetic runs in float32 on the GPU (~1e-7 relative),
the CPU tests in float64.

## Code

- `src/commute/split.py`: `adjoint_step` (one step of one matrix: pieces, dU dVᵀ, energies, c, R) and `StepSplit`
  (carries U, V and the run's weights; `carry=False` is the fixed-bases variant; `delta_change`, `state`,
  `piece_change_max`). A step with no first-order image (K(S) at rounding level) gets zero pieces, so c is never a
  ratio of rounding noise (found while writing the out-of-span test; the run logs how many matrix-steps hit it,
  expected 0).
- `src/commute/finetune.py`: optional `on_step(step, params)` in `_run` and `finetune_arm_a`; default path unchanged.
- `experiments/step_split/run.py`: one frozen `Sizes` with `--smoke`; W₀ control → fine-tune with both splits
  (snapshot at each scored step) → per step: step-split state, then do-nothing and least-squares; JSON dumped after
  every state; `--resume` re-runs the fine-tune and scores only missing states. Every scored number goes to the log.
- `tests/test_step_split.py`: in-span update → Δ change ≈ 0 (< 1e-3 relative); out-of-span → all to Δ; weights equal the
  run and Δ books exactly Σ(S − cK(S) − dU dVᵀ); fixed bases with one c = the one-shot adjoint split; carry vs fixed
  differ at O(step).
- `cluster/step_split.sbatch`: L40, `debug`, `--time 00:55:00`; smoke `--time 00:20:00`.

## Cost and order

L40 estimate from E16's H100 log and E7/E9's L40 Pile timings (~115 s per Pile state): load + W₀ control ~7 min,
fine-tune with both splits ~3 min, 9 Pile states ~18 min, CPU tables ~2 min: ~30 min; `debug` with 55 min. Smoke
(~6 min) first, then a Codex adversarial review of the diff, then the scored run after the user files the predictions.

## Hand-off

Before the scored run: the user confirms the restatement and the predictions (set `"status": "filed"`, `"filed":
"<date>"`), the GOALS Terms row is approved or declined, and the Codex review is run on
`src/commute/split.py`, `src/commute/finetune.py`, `experiments/step_split/run.py`. Then one pod-runner per run
(smoke, scored), brief from this file and `cluster/step_split.sbatch`.

## Amendment 1 (2026-10-21, before the rerun)

The scored run (cluster L40, job 28108) stopped at its W₀ control: floor f 0.209 against the 0.15 line, which was set
from E16's H100 (0.075). The L40 reads 0.209 every time (E8, E11), so the line was wrong for the card, not the harness.
The user chose to amend and rerun on the L40: the stop line becomes 0.25 (`results/predictions_amend1.json`,
`run.py` `FULL.floor_stop`). P1–P3, C1 and G2 do not use the floor and are scored as filed; P4 and G1 stand as written
but are weak on this card (band ±0.63).
