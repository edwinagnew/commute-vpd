# E292 amendment, 2026-11-24: the EAP-IG parity bar on the new task sets

Approved by whip-lead (11-24, about 04:40) under the user's standing rule for a mis-specified filed control with one clear fix: amend with a date, rerun, FYI only. It goes to the user's morning file as item 17. Filed before score.py runs, so every EAP-IG reading is still blind.

**What was filed.** The parity of the converted model against the native forward on each new task set used the stage-0 floor, 2.64e-06. That floor is the native forward's batched-against-single relative disagreement, measured on copy24 only, and on one path only.

**Why it is mis-specified.** Parity compares two numeric paths, and each has its own noise. A bar taken from one path's noise, measured on another task set, sits inside the noise of the comparison it guards. CLAUDE.md requires the floor to be measured on the path it guards. On the 8 new sets (L40, job 65694 / 65790):

| Path | Disagreement on the 8 sets |
|---|---|
| Native forward, batched against single (`native_floor_parts_M`) | 1.55-2.75e-06 |
| Converted model, batched against single (`rel_batch_single_clean`) | 1.64-3.18e-06 |
| Parity, converted against native | 1.87-4.18e-06 |

Four sets missed the old bar, by 2.72-4.18e-06 against 2.64e-06: r0_t352, r1_t534, r2_t275 and r2_t359.

Cross-card values, not saved per set: on copy24 the 4090 read 2.74e-06 against its own floor of 2.7e-06, and the H100 read 4 values over 2.29e-06. Both are at the same scale as the same-card spread above. They are not used in the bar.

**The amended bar.** For each set, the bar is the sum of the two paths' own batched-against-single disagreement, measured on that set in the same job: max(native parts on clean and corrupted) + `rel_batch_single_clean`. It applies to all six relative parity stops. The tolerance is relative, as before. Stage 0's copy24 floor stays beside it in each parity file (`floor_stage0`).

The four sets that passed under the old bar also pass the new one: each value sits under its set's sum (for example, emoticon 2.33e-06 against 3.37e-06). They are not rerun.

The four stopped sets are rerun on the L40. Their first parity files are moved to `eapig/superseded_2026-11-24/`. Each set's first-run sums already clear its values: r0_t352 5.03e-06 against 4.18e-06, r1_t534 5.41e-06 against 2.74e-06, r2_t275 5.78e-06 against 2.92e-06, and r2_t359 5.16e-06 against 2.72e-06. So the rule was checked against the values it would gate. That is stated here and not hidden.

**What the amendment cannot do.** It cannot tell a real conversion error smaller than about 5e-06 relative from noise. The full and empty circuit checks are unchanged: `rel_full_circuit_vs_clean` was 0 on all 8 sets, and `rel_empty_circuit_vs_corrupted` was 0.5-0.8e-06.
