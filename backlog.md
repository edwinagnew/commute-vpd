# Backlog

Deferred ideas and loose ends. Date each entry.

- 2026-10-14. Re-decomposition test: warm-start VPD on W + δW from the released decomposition and measure subcomponent
  drift against a δW = 0 warm-start control (optimisation noise) and the cross-seed baseline (0.28 MMCS). Needs the
  full VPD training stack; only becomes the question if the projection test fails or to find the "limit".
- 2026-10-14. Ground-truth calibration on the repo's toy models (TMS, ResidualMLP, CPU): "strengthen" = change feature
  probabilities or importances; "new capability" = add a feature. Gives the strengthen / new-capability dichotomy a
  known answer to validate the measures against.
- 2026-10-14. Allow the causal-importance function Γ to re-fit for a few steps (transport level T2) once the frozen-Γ
  ladder has been scored.
- 2026-10-14. Transport test round 2 (after `/design`): scale-only k-sparse transport as the null-resistant ladder rung;
  write edits constrained to span{U_c, u_o}; selectivity on generic Pile text (KL of f_T to f_W0 outside the windows);
  functional selector (greedy gain or CI on the fine-tune data) instead of Frobenius contribution; k = 1 on `2359`
  directly; arm A checkpoints at steps 1–5; arm B at a learning rate giving ~0.1 nats held-out effect; unmasked KL
  (Δ in and out) as the sensitive faithfulness number.
- 2026-10-14. Arm C (new skill) still not run; it only makes sense once the round-2 null has power.
- 2026-10-14. Cluster set-up for iterative analysis (user asked 2026-10-14): persistent venv + artifacts + δW files
  under `/mnt/nw/home/e.agnew/commute`, debug-QoS interactive jobs for model-in-the-loop work.
