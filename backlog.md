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
- 2026-10-14. Transport test round 2 (after `/design`): scale-only k-sparse transport as the null-resistant ladder rung;
  write edits constrained to span{U_c, u_o}; selectivity on generic Pile text (KL of f_T to f_W0 outside the windows);
  functional selector (greedy gain or CI on the fine-tune data) instead of Frobenius contribution; k = 1 on `2359`
  directly; arm A checkpoints at steps 1–5; arm B at a learning rate giving ~0.1 nats held-out effect; unmasked KL
  (Δ in and out) as the sensitive faithfulness number.
- 2026-10-14. Arm C (new skill) still not run; it only makes sense once the round-2 null has power.
- 2026-10-14. Round-2 inputs from the user and the audit (`reports/2026-10-14_review_round1-audit.md`), all through
  `/design` first:
  - **The pull-back map.** Apply the post-training step to U and V instead of W (user: a good thing to test).
  - **LoRA updates.** Use them alongside full-weight fine-tunes (user: definitely test).
  - **Faithfulness with Δ held at its W0 value**, so the components must carry the update.
  - **A convergence check on every functional fit.** Final training loss must be below the zero-edit loss; round 1's
    arm-B fits at lr 1e-2 diverged.
  - **Unexplained:** arm A's Frobenius fraction relative to random grows 1.7× → 10.2× from step 10 to 300, while arm
    B's falls.
- 2026-10-14. Formalisation: written as `theory/commuting_updates.md` (user's request; not yet reviewed by the user).
  It defines the weak version as a lift, the strong version as an approximately commuting square, the obstruction
  fraction ω and the locality residual ρ. Its §7 candidates go through `/design` before any run.
