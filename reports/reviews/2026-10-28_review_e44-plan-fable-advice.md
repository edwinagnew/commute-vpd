# Review: Fable's advice on the E44 (Q0 + Q1) plan

*2026-10-28. Fable advisor, asked by the user to review the revised plan (`reports/plans/2026-10-28_plan_qwen-q0q1.md`, 2e68fa7) before it went to them. Advice, not a decision.*

**Verdict: go with changes**, all inside the same pod and budget.

1. **Update size.** E40's step 10 is 0.52 nats/token of KL to start; r1's 1.5 line leaves ~0.03 nats of slack, so a least-squares fail there is near-certain by magnitude and reads "a large format change fails", which `GOALS.md` ("small is measured, never assumed") cannot use. Use E40's saved `/workspace/e40/dW/step_1.safetensors` (KL to start 0.106, run.log line 69): steps 1 and 10 mandatory, step 100 only if step 10 passes. Both-fail branch: next is Q2's own-samples update through this table; Q4 ($30) only after a small update fails.
2. **Controls.** Control (i) halves and the in-span state are sound (say the in-span r2 tracks least squares' r2). Control (ii) should be matched on Σ-weighted effect (‖(dU Vᵀ + U dVᵀ)Σ^½‖ = ‖δW Σ^½‖), not on ‖dU‖, ‖dV‖; this also replaces the ‖dU dVᵀ‖ > 1 degeneracy rule. Gram-basis check: equal ranks and reconstruction residual < 1e-6, not "pieces within 1e-8 relative".
3. **Trim / add.** Report r1 stochastic, don't score it; drop prediction 13 (concentration, Q3 material), keep the arrays. Add absolute masked KL in nats beside each ratio; note do-nothing's maths r3 range has no E40 reference; unmasked GSM8K accuracy on 256 problems for the adjoint and in-span states (~2 min each).
4. **Open questions.** (a) halves by on-rate: yes, report each half's Σ-weighted energy. (b) no CI-masked GSM8K accuracy: agree. (c) fresh adversary on the passing state with the smallest r3 margin (adjoint split if none passes), not pre-committed to least squares at step 10.

**What would change it:** if the question is specifically "can a simple map carry a large update"; or if step 1's δW is dominated by the no-warm-up first-step artefact, in which case Q2's warm-up update is the urgent small point and this run is calibration only.
