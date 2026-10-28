# Review: E43 code (commit 8b3b4b6), 2026-10-28

Fresh Opus reviewer, read-only, scope `git show 8b3b4b6`: `experiments/teacher_training/remainder.py`,
`results/predictions_e43.json`, `reports/plans/2026-10-28_plan_e43-teacher-remainder.md`, `RUN_BRIEF_e43.md`.
Saved from the reviewer's hand-back by the lead (the review was read-only).

**Verdict.** No bug that wastes the ~$2. Δ′ = W0 + dW − U′V′ᵀ is the model's own `calc_weight_deltas` (fp32, weights
equal the fine-tune's to one ulp); Γ, PGD (`use_delta_component=True`) and the stochastic masks treat Δ′ as they treat
do-nothing's Δ. "Δ′ back to Δ" is teacher training's own state (`change_of` = `Offsets.delta_w`; G2 checks it). KL
direction right (KL(fine-tune ‖ state)). References match E41's JSON (PC 2.5159 / 1.4994; G3 0.99941 / 1.54653 and
0.99675 / 0.95909; C1 0.69882, floor 0.07482). Predictions sha256 matches the brief; config/e43 entries match CFG/AM.
Brief inputs, sha256s and downloads right. `read` vs `run.read_state`: no number changes (teacher gap NaN and popped;
adds `faith`; drops card/w0_floor, covered by `load()`).

## Findings

1. **MATERIAL — P3 at T10 is decided by the verdict clause inside noise** (`remainder.py:191`; P3; plan 142-148).
   Teacher training's own T10 ratio is 1.5465 (inconclusive); the fail line is 1.612, only 4.2% above it, so "verdict
   no worse" fails P3 at 1.612 while the 1.25× bound (1.93) never binds. W0's adversary-seed floor is 7.5%. A reading
   of 1.63 would "fail" P3 and steer the tree. At T1000 the ratio clause decides (1.199 < 1.396), fine. **Fix before
   the run:** refile under a new name — drop the verdict clause, or apply it only where the ratio clause cannot decide;
   at least state the effective 4% tolerance at T10.
2. minor — filed expected "teacher training: b 0" is wrong (predictions "expected"; plan line 103; docstring line 16):
   teacher training's own removal share is KL(ft ‖ W_tt)/KL(ft ‖ W0), which equals plus-remainder's exactly (G2). P2
   measures the copy's own Pile-text fidelity; don't plot teacher training at 0.
3. minor — the verdict combines stochastic and PGD ratios (`remainder.py:92`), but P3 says "the check alone".
   Stochastic lines 1.498/1.502, readings 0.89-1.01 so far; mention in the analysis.
4. minor — no stop on a weak P2 denominator (`remainder.py:136-139`): KL(ft ‖ W0) on Pile has no filed expected value;
   if < 0.005 nats P2 is unscored after the spend. Runner reports `kl_pile_w0` at PC T10.
5. minor — G2 is near-tautological (same formula CPU vs GPU); no zero control on the removal path (fine-tune vs itself
   = 0); E34 A6 ran one (pile_gap 0), low risk.
6. minor — brief's least-squares expectations ("pgd about 7.0 / 2.1", from E22, maybe another card) are ungated; add
   "not gated" so the runner doesn't flag a mismatch.
7. minor — bf16 autocast in the masked forward rounds Δ′; the harness's convention for every state; add as a
   "blind to" line.
