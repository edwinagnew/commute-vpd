# Review: E44 precision fix (commit 47e91aa), before the paid rerun

A fresh agent that wrote neither the fix nor the diagnostic read commit 47e91aa (`qwen_maps.pieces`, `stored_error`,
`site_change`, `Job.change`, run.py defaults, the new test, the plan's Amendment record and RUN_BRIEF §3) to see
whether the rerun applies what the plan says and whether `stored_error` measures what is applied. The fix does what it
claims for the stored X: on the test's ill-conditioned span the old number is 2.5e-3, the new one 3.0e-4, and the
in-span state's applied weight change misses its float64 value by exactly the new number. Two things to act on before
the pod: the changed apply path has never run on a GPU, and `stored_error` leaves out the fp32 in-place add of the
pieces, which at realistic δW sizes is as large as the limit but must not be folded into the rule.

## Findings

### 1. The changed apply path has not run on a GPU (blocking by the repo's rule; $0 to clear)
`experiments/qwen_maps/job.py:241-250` now hands `site_change` float64 δW, columns and Gaussian pieces, and
`src/commute/qwen_maps.py:249,255-256` (`applied`) and `experiments/qwen_maps/run.py:191` (`accuracy`, now a float64
`W0[s] + dW` copied into bf16) consume them. The last rehearsal (`results/rehearsal/`, commit a5c0737, 20:23) predates
the fix; diagnostic job 41552 ran `diag_stored.py`, not these paths. On the pod S runs first (5-35 min, $0.3-2.3 at
$3.99/h), so an error in D, K, M, G or A surfaces only after S has been paid for. I read the dtype flow and found no
error (`dW.to(w0)`, `dV.to(V)`, `alive.idx[s][pos.to(V.device)]`, `copy_` from float64 into bf16 all work), but
`.claude/CLAUDE.md` says a measurement-code edit is done when it has run once on a GPU.
**Fix:** rerun the cluster rehearsal (`--rehearse --waive control_W0,teeth`, stages C,S,D,K,M,G,A,H) on 47e91aa
before the pod. `results/rehearsal/` holds finished stage files and a `stopped.json` (stage C), so move it aside first
(or every stage is skipped) and use a fresh `--out`. Note the rehearsal holds bf16 components, so its `stored_error`
does not model what its forward applies; it checks code paths, not the number.

### 2. `stored_error` does not measure the fp32 in-place add, and that term is near the limit (record it; do not add it to the rule)
`src/commute/qwen_maps.py:255-256` applies `V[:, cols] + dV.to(V)` in fp32, so the forward's change is
fl32(U + fl32(dU)) − U, not fl32(dU). `stored_error` (`qwen_maps.py:158-162`) models only the second rounding.
The add's error is about eps·‖U_A‖ per column, not amplified by κ, so relative to ‖δW‖_Σ it grows as δW shrinks. My
check (scratch, not committed: the commit's test span with δW scaled to 4e-5 of ‖U_A V_Aᵀ‖, the ratio the diagnostic's
step-2 q_proj suggests if ‖W‖ ≈ 50, an assumption) gives: new `stored_error` 3.0e-4, component increment after the
fp32 add against float64 9.5e-4. With the test's unscaled δW (‖dU‖ ≫ ‖U‖) the add costs nothing, which is why the
test does not see it.
This is the fp32 floor of the forward, not a storage cost: the fine-tuned weights fl32(W0 + δW), used by do-nothing,
ls and both halves, carry the same eps·‖W0‖ error, and no storage format removes it. So the commit message's and the
Amendment record's "measures exactly what the run applies" is not true at step 1, but the rule as filed ("a stored X
whose fp32 pieces miss their float64 dσ") is what the code now measures, and the readings (KL, quadratic in the
change) cannot resolve a 1e-3 relative perturbation of δW.
**Fix:** reword the Amendment record and RUN_BRIEF §3 to "measures what storing X in fp32 and rounding its pieces
costs; the fp32 forward's own floor (about eps·‖W‖/‖δW‖, shared by every state including do-nothing) is excluded".
Optionally log, unscored, per site in S's JSON the add term (`(U32 + dU.float()).double() − U`, through `dsigma`), so
the floor is a measured number in the report. Do **not** put it under STORE_ERR_MAX: it could stop the rerun at step 1
on a floor that no fix of S removes.

### 3. Minor
- `qwen_maps.py:189`: `site_products` binds a local `pieces = {}`, shadowing the new module function `pieces`. Harmless
  today (`stored_error` resolves the global; `site_products` never calls `pieces`), but the next edit that calls
  `pieces(...)` inside `site_products` gets `TypeError: 'dict' object is not callable`. Rename the local
  (`half_energy`).
- The new test (`tests/test_qwen_maps.py:221-235`) would have failed on the old code (old `stored_error` on its span:
  2.5e-3 > 1e-3; checked). It does not tie `site_change`'s output to `stored_error` on that span; my check found them
  equal (in-span weight change against δW − O(δW): 2.991e-4 = `stored_error`). One assertion would lock it in:
  `‖site_change("inspan", …, to_disk(X))[0] − (M − O)‖_Σ / ‖δW‖_Σ == stored_error` to 1e-9.

## Checked and found fine
- **Every state's application.** W0: nothing. do_nothing: weights = fine-tuned (`FT_WEIGHTS`), no column change. ls:
  fine-tuned weights, pieces from fp32 X formed in float64, rounded fp32. inspan: weight change dσ of those rounded
  pieces, float64 then fp32; same pieces (so Δ′ − Δ₀ = −dU dVᵀ exactly, as the plan says). adjoint: c·(δW V_A),
  c·(δWᵀ U_A) and dσ in float64 (no stored X; was fp32 before, a gain). random: the same seeded fp32 draw as S, scaled
  by S's `a` in float64 (S fitted `a` on the `.double()` of that draw, so applied and fitted agree). lower/upper:
  fine-tuned weights, `pieces` of `X_<half>` on `U_A[:, pos]`, `V_A[:, pos]`, matching `stored_error(Xh, Uh, Vh)` in S.
- **`stored_error` vs application.** In S, U_A, V_A are float64 casts of the fp32 components and X goes through
  `.float()` as `to_disk` does; at apply time the same columns and the same fp32 file. The two use identical pieces
  (bit for bit), so apart from finding 2 it cannot under-report. Second-order term dU dVᵀ: its rounding error is
  relative eps of the pieces (no cancellation), outside the filed rule's dσ.
- **Filed numbers.** S's Δ shares, `ls_second`, piece energies, c, a: all computed from float64 pieces in S, unchanged
  by the commit. The rule text in `predictions.json` ("fp32 pieces miss their float64 dσ … 1e-3") and STORE_ERR_MAX are
  unchanged and still describe the new measurement. The plan's expected max 1.8e-4 is `err_x64` + ~1e-6
  (`err_pieces32`), consistent with the test case.
- **GPU memory.** Per site the float64 transients are δW, X, two products and dσ terms: about 1 GB for the largest
  (mlp, 6144 × 2048, ~2,500 alive), freed site by site inside `applied`'s loop, against the brief's 60-65 GB peaks
  and 70 GB halving rule. G's float64 `W0 + dW` per site is ~100 MB. No regression that matters.
- **Rerun paths.** `stopped.json` and `state.json` are tracked; `refuse_if_stopped` exits while `stopped.json` exists,
  so RUN_BRIEF §3's `git mv` is required and sufficient. `Job` reads `usd` 1.274 and `eval_batch` 2 from
  `state.json`; `--prior-usd` is then ignored, as §3 says (the second pod's set-up must be added by `jq`). C is skipped
  from `C.json` (`done: true`) and `gate_C` re-reads it with no model load. S reruns: `done("S")` is false (no
  `S.json`) and `s_site` finds no files on the new `/root/e44`. `--e40` default and the brief now agree on
  `/workspace/e40_nowarmup`; `main` stops on missing δW files before any stage.
- **Tests and lint.** `tests/test_qwen_maps.py`: 16 passed (worktree at f895563, `PYTHONPATH` set to its `src`). ruff
  clean on the four touched Python files; pyright could not start on the worker (nodeenv failure), not run.

## Not run
- No GPU run of any path (finding 1 asks for one); the real ‖W‖/‖δW‖ ratio behind finding 2 was not measured (the 9.5e-4
  is a toy with an assumed ratio).
- pyright.
