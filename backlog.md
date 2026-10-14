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
