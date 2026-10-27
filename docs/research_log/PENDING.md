# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · reports/analyses/2026-10-27_analysis_qwen-port.md · Qwen line card (port status), timeline 2026-10-27, apparatus notes · Torch port of the Qwen3-1.7B decomposition passes all five checks against the authors' JAX code (target 0.43 bf16 floors, Γ disagreement > 0.1 on 2.4e-5 of pairs, masked KLs within tolerance); torch's 12% lower count of g > 0 is bf16 rounding leftovers below 0.001, so active counts need a fixed threshold such as 0.01; the paper's adversary on a 4-row block reads 7-28 nats across six seeds, so single-start readings on Qwen carry a large spread
- 2026-10-27 · reports/plans/2026-10-27_plan_e38-amend3.md · decisions (card decision-e38-specific: user chose (a); move to decided), e38 card (Running line), timeline · User chose the restore test: E38 Amendment 3 puts a set's trigger outputs in the fine-tuned model back to their original-model values and compares S with size-matched sets already active at the trigger, null minus S and random sets; predictions filed
