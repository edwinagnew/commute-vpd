# Appendix: P1.37 apparatus (E###)

*Companion to `reports/plans/2026-11-20_plan_p1-37-vpd.md`. Code `src/commute/kstar_vpd.py`, `experiments/kstar_vpd/`.*

## 1. Variables and mechanisms

Each of the 24 decomposed matrices W (four layers × q, k, v, o, c_fc, down_proj; C = 512, 512, 1024, 1024, 3072, 3584)
uses param-decomp's layout: U (C, d_out), V (d_in, C) and W0 = Σ_c u_c v_cᵀ + Δ, with Δ from `vpd.weight_deltas`.

- **The split.** It comes from `split.least_squares_split` over all C subcomponents, in fp64, with bases at
  `split_tol` 1e-6 (`split.basis`). δW_c = du_c v_cᵀ + u_c dv_cᵀ and δW_out = δW − Σ_c δW_c.
- **A subcomponent's mechanism on input y.** In M it is u_c (v_c·y); in F it is (u_c + du_c)(v_c·y) + u_c (dv_c·y).
- **The remainder.** In M it is Δy; in F it is (Δ + δW_out)y.
- **Appended neurons (moved).** Each is one graft variable (its c_fc row and down_proj column together). M lacks it, so
  zero-padding M is exact: no biases (asserted), and GELU(0) = 0. It scores 0 and counts in f, not in the
  site-remainder share.
- **Embedding and unembedding.** Each is one variable. The run stops unless their weights, and every other parameter
  outside the 24 matrices, are equal in F and M, so both read s = 1 by identity.

## 2. F_K

Hooks sit on F's own modules.

- **Ablation.** An ablated subcomponent subtracts its F mechanism applied to (y − ȳ), and an ablated remainder
  subtracts (Δ + δW_out)(y − ȳ). ȳ is the position-wise mean of the matrix's input over D in F's unablated run.
- **Cheap form per matrix.** With a = (y − ȳ)V and b = (y − ȳ)dV, the correction is
  ((a + b)⊙w)U + (a⊙w)dU + w_rem (y − ȳ)(Δ + δW_out)ᵀ, where w = 1 − keep.
- **Grafts and unembedding.** An ablated graft neuron subtracts (act − mean act) times its output column. The
  unembedding ablated gives the mean logits.
- **All kept.** This subtracts exact zeros, so it is F's forward. Gate (d) compares it with F's native logits.
- **Attribution.** The gradient of the summed KL with respect to the keep vector.

## 3. Grade

The settings are every position of D in M's run and in F's run. Row "read" uses the read positions and row "all"
every position (E214, E221).

For subcomponent c, with a = v·y, b = dv·y, uu = |u|², udu = u·du and dudu = |du|²:

- dot = a((a + b)uu + a·udu)
- |M|² = a²uu
- |F|² = (a + b)²uu + 2a(a + b)udu + a²dudu

**Zero tests.** M's write is zero iff a = 0 or uu = 0. F's write is zero iff (a = 0 and (b = 0 or uu = 0)) or
(uu = 0 and dudu = 0). Both zero counts 1 and one zero counts 0 (`kstar._add`). The remainder is graded from its
vectors (`kstar._vec`).

**Unit test.** `tests/test_kstar_vpd.py` passed on 2026-11-20 (cpu-run, 4 tests). It checks that the analytic sums
equal those from the explicit writes (rtol 1e-9), that the split is exact, and that the pieces rebuild both weights.

## 4. Search

1. **Ranking 0.** Attribution at c = 1/2. At c = 1 (all kept) the gradient of KL is exactly 0, its minimum, so the
   brief's "at all-kept" cannot rank; this uses E214's midpoint (`kstar.attribution`). Departure stated.
2. **Bisection.** Bisect the prefix length: the full set passes and the empty set fails whenever KL(F || F_empty) > 0.
   The returned prefix was evaluated and passes, even though closure is not monotone.
3. **Re-ranking.** Re-rank twice at the best prefix found so far (`reranks` 2) and keep the shortest passing prefix.
4. **Polish.** Drop chunks of the prefix, least attributed first. The first chunk is |prefix|/8; a chunk halves on a
   failure and doubles back up to the first size on a success.
5. **Cap.** 2,000 evaluations in total per search. When the cap binds, the item reports K\* as an upper bound, the
   smallest chunk reached, and R at both the prefix set and the polished set.

The cap is fixed in the filed settings rather than sized per shard, so every search has the same budget. Stage 0
measures seconds per evaluation and stops if the shard's projected hours exceed 6.

## 5. Guards (R0; each stops the job)

- **Stage 0.**
  - Do-nothing reads 0 changed variables, every s within 1e-6 of 1, f = 0 and R = 1 (prefix and polished, both
    grades).
  - M's 24 weights equal the VPD checkpoint's target weights (`vpd.checkpoint_weights`).
  - The projected shard hours are at or under 6.
- **Per item.**
  - split_fp64 ≤ 1e-12.
  - F and M rebuilt from the fp32 pieces within 1e-4 relative.
  - Every parameter outside the 24 matrices is equal in F and M.
  - The gates (all kept against F's native logits, none kept against the mean logits) are within 1e-3.
  - For E187 states, E221's load checks.
  - `kstar.summary`'s identity R = 1 − f(1 − ρ′) holds.
- **The 1e-4 bars are guards, not measured floors.** The rehearsal logs every site's value. A value within 10× of a
  bar gets a dated amendment before the scored run.

## 6. Inputs the run loads

- **E214's builders** (`experiments/kstar_levels/run.py` World): E62's windows and pools, Pile validation rows, E195's
  copy24 held-out windows, and `rebuilt_mlp.pt`. E214's edit build writes `experiments/kstar_levels/results/edits.json`
  as it did in E214; on a pod that copy is not brought back.
- **E221's builder** (`experiments/kstar_pairs/run.py` World): E187's saved δW files, windows, E205's and E196's
  windows and saved readings, listed in `experiments/kstar_pairs/inputs.txt`.
- **The VPD decomposition and its target model**, under `env.VPD_DIR`.

## 7. Shards

There are 74 items after stage 0: 6 known plus 68 E187 states. The count of 68 is the unique states of E221's nine
E187 pairs: E221 read 80 fine-tunes, of which 12 are E196 members that no E187 state shares. The code takes them from
`kstar_pairs.run.pairs()` at start. The order is new_few, new_thin, x2, x10, moved, rebuilt, then the states, taken
with `every[k::n]`. With n = 4:

| Shard | Items | Known items | Searches (with stage 0's do-nothing) |
|---|---|---|---|
| 0 | 19 | new_few, moved | 6 + 17 + 1 = 24 |
| 1 | 19 | new_thin, rebuilt | 6 + 17 + 1 = 24 |
| 2 | 18 | x2 | 3 + 17 + 1 = 21 |
| 3 | 18 | x10 | 3 + 17 + 1 = 21 |

Each shard runs stage 0 first.
