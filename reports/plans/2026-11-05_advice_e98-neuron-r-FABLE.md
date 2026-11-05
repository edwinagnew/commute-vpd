# Advice: a neuron analogue of VPD's r on the 67M (E98, part 1)

*2026-11-05, Fable, for baseline-lead. Read-only advice; nothing run. User's decisions: `decisions.md` row 154.*

## 1. Definition

**Units.** The 67M's 12288 MLP neurons: GELU outputs of `c_fc`, inputs of `down_proj` (`src/commute/steering.py:25-34`).
The fine-tunes move only the 24 decomposed matrices (`src/commute/finetune.py:1`): biases, LayerNorms and embeddings are W0's in every state.

**Cells.** On a W0 forward of each held-out window, two switches per layer at every position: A = the firing `down_proj`
sees (A0: W0's own a; A1: the state's a′, captured from the state's forward on the same window, E52's patch), O = the
output weights it applies (O0: W_out; O1: W_out′). L(O, A) = mean log-odds of the target at the answer position; T0 =
L(O0, A0) = W0, T1 = the state, G = T1 − T0. Over G:

- r_n = L(O0, A1) − T0: W0's output directions carrying the state's firing (the user's "existing neurons firing differently, original weights fixed");
- d_n = L(O1, A0) − T0: new output directions carrying W0's firing;
- i_n = L(O1, A1) − L(O0, A1) − L(O1, A0) + T0: interaction;
- e_n = T1 − L(O1, A1): the rest, here the attention matrices' change.

They sum to 1 by telescoping, as VPD's d, r, i, e_gate, e_delta do (`src/commute/local_faith.py:176-186`; r = (L01 − L00)/(T1 − T0),
`src/commute/regating.py:19`). Mirror: VPD transplants Γ's gate pattern from the state's forward into a forward of W0's
subcomponents; this transplants the firing pattern into a forward of W0's output weights. Where it must differ: a neuron has
no gate apart from its weights. GELU's own factor Φ(h) saturates above h ≈ 2, so transplanting Φ(h′) alone would miss
amplitude strengthening; transplanting a′ carries the amplitude, at the price that ΔW_in hides inside a′ (a neuron turned on
at the key by a new read direction counts as re-firing). One extra cell shows it: A½ = act(W_in · LN2(x′_l)), W0's in-weights
on the state's MLP input (a hook on `block.mlp`'s input, not in `SITES`); r_n then splits into upstream re-firing, both MLP
matrices fixed, and changed read directions. Add it: no earlier run has read it, and it is the strictest "same neuron".

**Blind to.** (a) Where the gain is implemented, not whether it is new: the full-width share rose with training, 0.69 at
the cipher's step 9 to 0.96 at step 30 (`reports/analyses/2026-10-30_analysis_e56-keydependent-patch.md:81,157`).
(b) Attention-borne mechanisms: on E12's copy pair 0.03-0.50 of G was outside the MLP patch
(`reports/analyses/2026-10-29_analysis_e52-baseline-neurons.md:216-218`), and the 67M's "elicit" tasks are copying, a head
mechanism (E62); a neuron r puts them in e_n. (c) Relays: an MLP after a changed head fires differently with nothing of its
own changed and takes the credit; VPD's r shares this ("an edit in one layer moves CI values in others", `GOALS.md` Terms).

**Readings.** W0: every cell 0 exactly. Do-nothing: `code_step10`, G = −0.051 under the 0.05 floor (`experiments/TRIED.md:110`),
"no change", unscored. Positive controls, built in the job (E54 recipe, 10 steps, one seed, a subset of matrices): (i) `c_fc`
+ attention only → d_n = i_n = 0 to 1e-6, r_n + e_n = 1; (ii) `down_proj` only → d_n > r_n. They test what the cells read, not
kind. Apparatus as E56 stage 0: identity patch ≤ 1e-6; all switches plus attention reproduce T1 within 1e-3.

**Not E52 again.** E52/E56's full-width patch is the r_n cell, and it does not do r's job (`table.json` `neuron_patch` fails
on e46, e54_matched, e54_own). E98's new content is d_n, i_n, e_n, A½, the built controls and eight rows never patched. Say so in the plan.

## 2. Rows

Every 67M row of `experiments/method_table/results/table.json` with saved dW dicts (`experiments/SAVED.md`): e46_copy_random
(`regating_table/`), e54_matched, e54_own (`keydependent_elicit/`), e53_lowercase_cipher (`performed_anchor/dW_*`, bf16, arm-B
text; the gain is read at every position, so use the lowercase task's own log-odds as E63 family C did), open_pair_c3 and
_copy (`open_pair/amend1`, `open_pair_instruments` seeds 3-7), e58_rank1_96, e58_elicit_shuffle (`open_pair/amend2,3`),
e62_copy_lookup (`induction_cue/`, r unscored, the only causal row). e58_copy_prior_cipher and e58_class_twin keep windows
only (`amend6/7`): Not run. One cluster debug job, $0 (E63's 26 states took 2 min 12 s).

## 3. Predictions

E51's separation rule; one gain floor, filed (E52's 0.05 nats or VPD's MIN_GAIN 1.0, `local_faith.py:187`). Score only states
with held-out P(target) ≥ 0.5 (`keydep.P_LEARN`, `src/commute/keydep.py:27`): progress was E56's confound.

- P1 (r_n is not a kind reader): where VPD's r separates (table.json: e46 0.76-0.80 vs 0.27-0.29; e53 0.75-0.77 vs 0.12-0.17;
  e54_own 0.76-0.77 vs 0.24-0.27; open_pair_c3 0.84-0.90 vs 0.22-0.25; e58_rank1_96 0.84-0.88 vs 0.29-0.31), r_n separates
  reuse-higher on at most 2 of 5. Lean: e46 reversed, e54_own fails (both known). Edge: one seed per side → reported, not scored.
- P2 (copy-type gains sit partly outside the MLP): e_n higher on the copy/elicit side on e46, e54_own, open_pair_c3_copy,
  e62. Edge: e_n may be negative; score it signed, and report |i_n| > 0.3 as "cells unstable", no verdict.
- P3 (go/no-go for part 2): k_half, the smallest shared neuron set (ranked by W0-gradient × Δa, `src/commute/ctxpatch.py:138`,
  verified by patching) whose r_n cell reaches half the full r_n, is > 64 on every row except e53's lowercase side, ≤ 32
  (E63 per-window: ~15 lowercase, ~110 broad cipher, ~240/470 on family A, `reports/analyses/2026-11-02_analysis_e63-neuron-sparsity.md:191,214`;
  shared ranks gave 4096/12288 in E56). Edge: r_n ≤ 0.1 → k_half undefined, "no carrying set".

## 4. The essay step

Per row passing P3, on its k_half neurons: (i) W0's top-20 Pile windows and mean activation at the key/marker/target
positions in W0 vs the state (did it fire here before?); (ii) ablate them in W0 on the task and on Pile (what did they do?);
(iii) one agent writes a one-line job from (i)-(ii), a second, blind to the task, writes an input to fire it; score its
activation against the top-20's. On the 67M worth doing only for e53's lowercase side, and only if P3 holds: the copy rows'
carriers are heads, and E66's W0 top-K neuron reading was dead at K 4096 (`experiments/TRIED.md:150`). A rehearsal of the Qwen procedure, not a 67M result.

## 5. Options

A. Four cells + A½ + built controls on the nine readable rows, one job, $0; P1-P3; the essay step on e53 if P3 holds.
   **Recommended**: the user's quantity beside VPD's r on the same rows with the same telescoping, and e_n can turn E52's
   reversal from a null into "the copy gain is attention-borne".
B. Heads as units too (attention patterns transplanted, OV weights fixed), a second job after A, if e_n is where the copy gains sit.
C. Skip to Qwen: against the user's decision. What would move to it: every e_n under 0.1 and r_n tracking progress as in E56.

Confirm one restatement before filing: if "original weights fixed" means both MLP matrices, r_n is the A½ cell alone.

Friction: `GOALS.md` Terms defines neither r nor the stand-in ratio; they live in `local_faith.py:176` and `table.json .methods`, and "r" cannot be grepped. Add both to Terms.
