# Plan appendix 2026-11-09: E93 Amendment 2, the table on the new decomposition

*Appendix to `reports/plans/2026-11-09_plan_new-decomposition-reruns.md`, line 1. Code `experiments/qwen_d1_table/{amend2,amend2_score}.py`; lines `results/predictions_amend2.json`; sbatch `cluster/qwen_d1_table_amend2.sbatch`.*

**Design.** E93's and Amendment 1's rules, unchanged in text, on new: the least-squares split and the scaled adjoint split on the same eleven full updates, with E86's least-squares rows and E75's wrong-change rows. Old stored numbers sit beside, not re-run. Reuse is not read here (`decisions.md` rows 81, 87 unchanged).

**Rows run.** The eleven `rules.STATES`, two maps (least squares; adjoint on the new least-squares' base, ceiling and random). E86's rows: in-span, remainder (dσ form), share on E40's step 10, gold (`u2`), E81's TOFU. E75's rows: the remainder against its three stored wrong changes on E40's step 10, gold, R100 (`R_s0`), R10, u3. Not read: E86's halves, E75's masked rows.

**What differs from E93's job.**
- Decomposition, Γ, alive set: new (`--decomp "$COMMUTE_QWEN_NEW_DIR"`); `Decomp` reads `GSTATS_OF`/`ALIVE_TOTAL_OF`, a KeyError until the new statistics are registered (the lead's edit once E131 Amendment 1 stage 1 lands).
- Carried, not re-read, because no decomposition enters them (`experiments/DECOMPOSITIONS.md`): gain-row losses, clean W0/FT/random/signs, file, sha256, norm. Stage 0 re-reads E40's step 10 and E81 and stops if they differ from stored.
- References: W0 masked KL and alive count from `qwen_new_decomp/results/w0_refs_stage1.json` (new), do-nothing K00/K01 read in stage 0, never an old number. Every scored band is E93's own formula or absolute line. New stage-0 bands are stated in the predictions file (`controls.bands`).

**Stage 0 (control first, alone, on E40's step 10).** Stops: alive count = E131's; W0 masked KL within 1e-3 of E131's; carried reads reproduce; do-nothing's K00, K01 through two paths agree and K01 < K00; least-squares share below its Gaussian null, ‖δW‖² reproduced; a matrix in the span leaves under 1e-6, a Gaussian its exact null within 2%; moved K under the ruler; memory; E81's carried reads. A miss writes `results/amend2/stopped.json`, exit 1; nothing later starts.

**Stop rule, per state.** sha256 and norm against E93's record; card and host peaks; exit 3 resubmits within the 2 h limit.

**Jobs, L40, `debug` QoS, $0** (`sbatch` per line of the sbatch header, one at a time after the last ends; times from E93's and Amendment 1's `run.log` "done in" lines on the L40, less the carried reads):

| job | card time |
|---|---|
| Stage 0 | ~35 min |
| LS: a2_s42, a2_s142 | ~50 min |
| LS: a1_s0, a1_s1, u1, u2 | ~1.0 h |
| LS: R_s0, R_s70, N_s0, N_s70, F_s0 | ~1.0 h |
| Extras: R10, u3, tofu_E81 | ~30 min |
| Adjoint: a1_s0, a1_s1, a2_s42, a2_s142 | ~50 min |
| Adjoint: u1, u2, R_s0, R_s70, N_s0, N_s70, F_s0 | ~1.2 h |

**Total about 5.7 h of card**, under the 6 h gated-run line, $0. The new decomposition's alive count is not yet known; if stage 0's projection (`control.json` `projection_min`) is more than 15% above old, the total crosses 6 h: the lead then moves the adjoint jobs into their own amendment.
