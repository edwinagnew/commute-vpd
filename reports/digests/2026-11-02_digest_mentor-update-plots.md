<!-- Plots and talking points chosen 2026-11-02 for the 2026-11-03 mentor update; PNG copies in ~/commute-data/mentor-update-2026-11-03/; page https://claude.ai/artifact/UC94LCbxJ1iDvZeNMfv8Np -->
# Mentor update, 2026-11-03: the nine plots

Selected 2026-11-02 from the research log (v160). Five show what works, four show where we are stuck. Each line: the
plot, its card, the one sentence to say over it. Sources are the experiments' own `figures/` PNGs, unchanged.

## Going well

1. `01_e16_only_component_training_passes.png` (E16). On the 67M model, only training the subcomponents in place of
   the update keeps all three counts (faithful, behaviour kept, nothing left in Δ); every one-shot split of a finished
   update fails at least one. This set the commute line's bar.
2. `02_e47_teacher_training_passes_every_start.png` (E47, red-teamed E34). Teacher training, which trains the
   pieces to copy the fine-tuned model's predictions with Γ and Δ frozen, passes the adversarial check from every
   adversary start on a trigger task and on a 32-pair cipher the model never knew, where do-nothing fails. A cued
   update can be written in the existing pieces.
3. `03_e12_existing_vs_new_route_differs.png` (E12). A behaviour the model has (copy) and one it lacks (random maps)
   look identical when the update is measured directly in weight space, but the gain takes different routes: existing
   behaviour through Γ switching old pieces on, new behaviour through pieces Γ keeps off. The original promise of the
   re-gating line.
4. `04_e53_standin_ratio_not_breadth.png` (E53, with E54's pair). On two real pairs the stand-in ratio puts the
   performed member near 1 and the new member near 0.3, whichever member is the broader update, so breadth does not
   explain the separation.
5. `05_e58_r_follows_prior_not_carry.png` (E58, one log figure, 2026-11-02). Eight key-to-value maps ordered by the
   original model's prior of the trained answer; r at step 30 is high for exactly the three maps whose answer the model
   already gave and low for the five whose answer is new, including a new four-word rule that carries to unseen keys;
   carry sorts the maps differently. One cut on the prior sorts all eight, so VPD has not yet shown anything the logits
   do not. This is the line's crux: it holds in one sense (plot sits under "going well") and is stuck in the other.

## Stuck

6. `06_e55_counting_set_no_headroom.png` (E55 C1b). On the only ground-truth set we have (Jain et al.'s counting
   grammar), pruning revival, reverse fine-tuning, steering and even a probe that only reads the answer label all 21
   states right. No headroom to show VPD adds anything. On the 67M open-pair maps the same cheap methods fail at every
   update size (E55 C3 ladder), so that set closed as a benchmark.
7. `07_e49_broad_finetune_fails_adam.png` (E49). The commute line's gap: for a fine-tune with no trigger
   (lowercasing), the Adam full-matrix copy fails the faithfulness check on all four new text blocks, the rank-8 LoRA's
   copy passes, and the check cannot fail an SGD update even when it is left entirely in Δ, so SGD is unresolved.
8. `08_e44_qwen_pass_test_no_teeth.png` (E44). On Qwen3-1.7B, Γ's masks computed from the fine-tuned model's
   activations bring the masked model to the fine-tuned one even with the update left out of the subcomponents, so the
   pass test cannot fail do-nothing. The Qwen line now only describes where the update lands.
9. `09_e57_qwen_direct_share_tracks_size.png` (E57). The re-gating share reads 0.99 on every Qwen state, random
    changes included (not plotted). The direct share puts the real updates above random changes, but the larger the
    change the higher it reads, so size may explain it. Next run: the direct share against size (E61, chosen
    2026-11-02).

## Open decisions to raise

- "Elicitation" was fixed on 2026-11-02 (ask 16) as mechanistic: the fine-tuned behaviour runs through a mechanism the
  original model already has. "The answer was already the model's own" is rejected as the definition; carrying to
  unseen keys is evidence, not the label. r is not dropped: it is demoted from definition to candidate instrument,
  scored against a causal label. Next run E62: teach the 67M model's in-context copying heads a new cue, confirm by
  ablating those heads, then read r on it and on a lookup at the same cue. On Qwen the re-gating share was set aside
  separately (E57: 0.99 on everything); only the direct share is still tested there (E61).
- Whether a benchmark with headroom exists at all: the counting set is saturated by cheap methods, the open-pair set
  is labelled by none.

## The route-split shares (plot 3), as the code computes them

All cells are mean log-odds of the target at the trigger over held-out windows. T_s: the full model at state s (0 the
original, 1 the fine-tuned). C_1 = C_0 + (dU, dV): the pieces after the least-squares split (smallest dU, dV with
dU V^T + U dV^T ~ dW; the rest to Delta). g_b: the frozen Gamma's gates read from model b's activations. U_s: stand-in from
C_s with every gate on, Delta out. L_ab: stand-in from C_a under gates g_b, Delta out. G = T_1 - T_0.

    d      = (L_10 - L_00) / G                       direct: moved pieces, old gating
    r      = (L_01 - L_00) / G                       re-switching (re-gating share): original pieces, new gating
    i      = (L_11 - L_10 - L_01 + L_00) / G         interaction
    e_gate = [(U_1 - L_11) - (U_0 - L_00)] / G       masked-off: pieces Gamma keeps off
    e_D    = [(T_1 - U_1) - (T_0 - U_0)] / G         residual Delta
    sum    = 1
    stand-in ratio = (L_11 - L_00) / (U_1 - U_0)     plots 4 and 5; no ceiling at 1

Source: `src/commute/local_faith.py`, `route_split`.
