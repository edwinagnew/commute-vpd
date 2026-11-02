<!-- Plots and talking points chosen 2026-11-02 for the 2026-11-03 mentor update; PNG copies in ~/commute-data/mentor-update-2026-11-03/; page https://claude.ai/artifact/UC94LCbxJ1iDvZeNMfv8Np -->
# Mentor update, 2026-11-03: the ten plots

Selected 2026-11-02 from the research log (v160). Five show what works, five show where we are stuck. Each line: the
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
5. `05_e58_r_tracks_own_guess_not_carry.png` (E58 synthesis). Across eight key-to-value maps, the re-gating share r
   and the stand-in ratio cleanly separate maps whose answer the original model already guessed from those it did
   not. In that sense re-gating does read "already there".

## Stuck

6. `06_e58_carrying_new_rule_reads_new.png` (E58 Amendment 7). A new four-word rule that generalises to unseen keys
   reads r 0.23 to 0.25, the same as the cipher and its shuffled twin, and one cut on the model's prior of the answer
   sorts all eight maps (left panel). So r tracks the prior, not reuse on new inputs, and VPD has not yet shown
   anything the logits do not. This is why the line is paused on "what does elicitation mean".
7. `07_e55_counting_set_no_headroom.png` (E55 C1b). On the only ground-truth set we have (Jain et al.'s counting
   grammar), pruning revival, reverse fine-tuning, steering and even a probe that only reads the answer label all 21
   states right. No headroom to show VPD adds anything. On the 67M open-pair maps the same cheap methods fail at every
   update size (E55 C3 ladder), so that set closed as a benchmark.
8. `08_e49_broad_finetune_fails_adam.png` (E49). The commute line's gap: for a fine-tune with no trigger
   (lowercasing), the Adam full-matrix copy fails the faithfulness check on all four new text blocks, the rank-8 LoRA's
   copy passes, and the check cannot fail an SGD update even when it is left entirely in Δ, so SGD is unresolved.
9. `09_e44_qwen_pass_test_no_teeth.png` (E44). On Qwen3-1.7B, Γ's masks computed from the fine-tuned model's
   activations bring the masked model to the fine-tuned one even with the update left out of the subcomponents, so the
   pass test cannot fail do-nothing. The Qwen line now only describes where the update lands.
10. `10_e57_qwen_direct_share_tracks_size.png` (E57). The re-gating share reads 0.99 on every Qwen state, random
    changes included (not plotted). The direct share puts the real updates above random changes, but the larger the
    change the higher it reads, so size may explain it. Next run: the direct share against size (E61, chosen
    2026-11-02).

## Open decisions to raise

- What "elicitation" means for the re-gating claim (the log's pending card, since 2026-10-30): (i) the answer was
  already the model's, which r reads; or (ii) a capability reused on new inputs, which r does not read and which no
  instrument yet labels on the 67M maps.
- Whether a benchmark with headroom exists at all: the counting set is saturated by cheap methods, the open-pair set
  is labelled by none.
