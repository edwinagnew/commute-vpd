# Backlog

Deferred ideas and loose ends. Date each entry.

- 2026-10-14. Re-decomposition test (the strong version in `GOALS.md`; postponed by the user until the weak version
  is settled): warm-start VPD on W + δW from the released decomposition and measure subcomponent
  drift against a δW = 0 warm-start control (optimisation noise) and the cross-seed baseline (0.28 MMCS). Needs the
  full VPD training stack; only becomes the question if the projection test fails or to find the "limit".
- 2026-10-14. Ground-truth calibration on the repo's toy models (TMS, ResidualMLP, CPU): "strengthen" = change feature
  probabilities or importances; "new capability" = add a feature. Gives the strengthen / new-capability dichotomy a
  known answer to validate the measures against.
- 2026-10-14. Updating Γ as part of Φ (user, 2026-10-14: "very fiddly but may well be essential"). Ignored for now;
  revisit once the frozen-Γ maps are understood. Simplest form: let Γ re-fit for a few steps.
- 2026-10-14. **Round 2 of the transport test: all inputs in one place** (no plan yet; the plan will be
  `reports/<date>_plan_<topic>.md`, after `/design`). Sources: "What this changes" in
  `reports/2026-10-14_analysis_transport-test.md` and in `reports/2026-10-14_review_round1-audit.md`, and §7 of
  `theory/commuting_updates.md`.
  - From the user (2026-10-14):
    - test the pull-back map, i.e. apply the post-training step to U and V instead of W;
    - test LoRA updates alongside full-weight fine-tunes;
    - target the weak version (`GOALS.md`).
  - Measurement fixes:
    - faithfulness with Δ held at its W0 value, so the components must carry the update;
    - a convergence check on every functional fit (final training loss below the zero-edit loss; round 1's arm-B fits
      at lr 1e-2 diverged);
    - the unmasked KL with Δ in and out as the sensitive faithfulness number.
  - Nulls and selectors:
    - a scale-only k-sparse null that cannot win by expressivity alone;
    - write edits constrained to span{U_c, u_o};
    - a functional selector (greedy gain, or CI on the fine-tune data) instead of Frobenius contribution;
    - k = 1 on `2359` directly;
    - selectivity on generic Pile text (KL of f_T to f_W0 outside the windows).
  - Arms:
    - arm A checkpoints at steps 1–5;
    - arm B at a learning rate giving ~0.1 nats held-out effect.
  - From the theory note: the obstruction fraction ω, the locality residual ρ, and a gradient-shaped null for ω.
  - Unexplained, to look at: arm A's Frobenius fraction relative to random grows 1.7× → 10.2× from step 10 to 300,
    while arm B's falls.
- 2026-10-14. Arm C (new skill) still not run; it only makes sense once the round-2 null has power.
- 2026-10-14. Formalisation: written as `theory/commuting_updates.md` (user's request; not yet reviewed by the user).
  It defines the weak version as a lift, the strong version as an approximately commuting square, the obstruction
  fraction ω and the locality residual ρ. Its §7 candidates go through `/design` before any run.
- 2026-10-14. RL post-training (user). Once the LoRA case is settled: an RL arm on the same model (e.g. REINFORCE with
  a simple reward) at matched ‖δW‖. Compare its obstruction fraction ω and the drift of the frozen Γ against the
  LoRA arm. Risks to check:
  - RL updates may avoid principal directions, so they could have a high ω;
  - on-policy distribution shift may break a frozen Γ;
  - errors may accumulate over long runs.
