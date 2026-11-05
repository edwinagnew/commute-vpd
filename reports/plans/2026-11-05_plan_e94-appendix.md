# Appendix to the E94 plan: sources, timings, builds

*Apparatus for `reports/plans/2026-11-05_plan_e94-d3-rows.md`. Grepped 2026-11-05. Paths under `experiments/` unless stated.*

## Files checked
- All 8 states' dtype `torch.float32`, by `qwen_size_grid/results/size_grid.json` (`.arms.<arm>.{full,at_norm,past_line}`). Anchor 2's `anchor2_s0` is seed 42. The anchor 1 states' `at_norm` ratio is 1.000 and 1.005, so its matched and full states are one file each.
- `qwen_size_grid/results/large/dW/a2_s42/step_{27,29,550}`, `a2_s142/step_{27,30,500}`; `results-large/qwen_anchor_checks/dW/{tofu_seed142/step_1250,a1_seed1/step_100}`: on disk, 5.64 GB each.
- T, L: `qwen_reading/results/large/B2/{T,L}.safetensors` (`active`, `n_pos`, `gsum`). `A/plain`, `A/W0_maths`, `A/layout` beside them. T is W0's think-on answers to GSM8K test 0-127, 108 rows closed, answer positions after `</think>` only (`run.py:254-258`). T's set is T on and plain off (`run.py:264`).

## Time per pass, one L40

| pass | time | source |
|---|---|---|
| Load target, components, Γ | 65 s | `qwen_reading/results/run.log:3-5` |
| Σ, maths rows of 512 | 172 s, once a job, 11 GB host | `:21` |
| δW file over NFS, with sha | under 1 min; 24 s without | `cluster/qwen_comparators.sbatch` header |
| X and piece energies, no nulls | 361-365 s | `:95,127` |
| The same with 4 nulls | 930-1118 s, so 155 s a null | `:169,205,242` |
| Outside share per state | upper bound 150 s | E86 `run.log:51,159`, via the E93 plan |
| Base cells K0, K00, K01, 128 rows | 108-123 s | `:138,212` |
| Base, 64 rows | 55 s | `:214` |
| Own-V′ K10, K11, 128 rows; 64 rows | 130 s; 93 s | `:243,244` |
| New pass, K10 and K11 with W0's V, 128 rows | 170 s: 130 s x 1.3, unmeasured | E72's single mask set with `OriginalV`: 68 s for 64 rows, `qwen_fixed_masks/results/run.log:240` |
| Gaussian build and search | about 110 s for 2 evaluations | `:61-64` |

E61's chunked pass, 46 s per 8 rows, 736 s per state (`qwen_recipe/results/run.log:2564-2584`), is the path replaced: entering `applied` costs 26 s and it holds 11 GB of masks. `OriginalV` (`qwen_fixed_masks/run.py:61`) reads W0's V while the columns are moved, so all masks are computed inside `applied`, one row at a time. Stage 0 measures this and the plan's total moves with it.

**Per state.** File 1 min; X, energies, outside share 8.6; maths base and new pass 4.9; TOFU rows base, new, own 7.0; template rows 4.5. Anchor 2, 21.5 min; anchor 1, 19. Random: 2 + 6 + 4.9 = 13 min on maths rows, four of them; own rows for alive-pieces 3 + 4.9 (small), 4.9 (big). Total 267 min, plus about 5 min load and Σ on each of ~4 further jobs.

## Numbers quoted
- E40: `qwen_reading/results/run.log:138,170`; direct, true masks, 16 rows from `qwen_recipe/results/checks_k10_e40.json` `.true`: K00 6126.284, K10 5771.847, K11 415.799, so 0.0621. Fold: `qwen_reading/results/summary.json .overlap.e40`.
- Anchor 2's precursor: E57 `a2long_s0` own rows K0 0.082, K00 5.41, K01 0.551, K10 5.33, K11 0.525 (`:214,244`): own-row K00 is much larger than maths rows', so denominators are safe.
- Template masks: `qwen_pass_test/results/u3/W0.json` K 0.3640, K0 0.0699. `qwen_fixed_masks/results/score.json .u3.T1`: do-nothing gain 0.7162, K 0.0664; least squares 0.7189; ceiling 0.7216; `w` 0.00065; room "no room". u3 is E61's bf16 file of E78's run.
- Outside share: `qwen_fixed_masks/results/{u1,u3}/S.json .outside_both, .outside_both_null`: 0.132, 0.252; 0.123, 0.212. E93's 0.36 input-weighted is `.share.ls`, the share left in Δ, not this number.
- Gains on whole-question positions: `qwen_comparators/results/dissection.json`, template 1.26; step 29 2.05 pooled; step 1250 3.77 pooled. All clear E72's 0.05 bar.
- Author-question fact share: `qwen_anchor_checks/results/amend1.json` and the E88 note (0.695; 0.411, 0.379); whole set 149/217 = 0.687, 0.613, 125/217 = 0.576.

## Builds, each with what it replaces

| # | build | from | shared with E93 |
|---|---|---|---|
| 1 | States: E91's `ARMS` and `path_of`, with a dtype check for either kind | `qwen_size_grid/run.py:42`, `qwen_comparators/run.py:96` | yes, its 1 |
| 2 | Rows: `a1_rows`, `tofu_rows`, `question_seen` into `qwen_comparators.py` | `qwen_comparators/run.py:64,74`, `dissect.py:78` | yes, its 2 |
| 3 | Sums per row per position class for KL and NLL | new; `kl_per_position` is in `vpd`, `nll` at `qwen_pass_test/run.py:81` sums whole rows | yes, its 3 |
| 4 | Least-squares-only X, no halves | `qwen_d1_fractions/run.py:101`, `qwen_reading/lab.py:394` | yes, its 4 |
| 5 | `OriginalV` to `qwen_maps.py`; the route pass: K10, K11 with W0's V, counts of g0 and g1 | `qwen_fixed_masks/run.py`, `qwen_reading/lab.py:341` | no |
| 6 | `Weights`, `pinned` to `qwen_sft.py`: lab's `Lab` writes into E57's results | `qwen_reading/lab.py:114-154` | no |
| 7 | Alive-pieces change at a scale; unit-Gaussian piece energies rescaled per matrix | `qwen_reading/run.py:327`; `lab.py:413` | no |

Callable as they are: `qc.separates`, `qc.rows_of`, `qr.overlap`, `qr.placement`, `qr.gated`, `qr.scale_search`, `qr.shares`, `qr.resample`, `qr.Counts`, `split.weighted_fractions`, `split.piece_energies`, `split.random_like`, `split.least_squares_x`, `qwen_maps.applied`, `pieces`, `stored_error`, `random_pieces`, `Alive`, `input_moments`. E44's per-site scale for the alive pieces: `qwen_maps/results/amend2/S.json`, step 10, as `qwen_reading/run.py:330`. `run.py`s stay frozen: moved code is imported by new runs, and the old files are not changed.

## Unit nulls
`split.random_like(M, Σ, seed)` = G · c(M), c per matrix. X is linear in M, and `piece_energies` is quadratic in X, so a state's null energy per subcomponent is c(M)² times G's. Stage 1 computes G's energies once per seed, then per state only per-matrix c is needed. Seeds 0-3, `null_seed` as `lab.py:449`, so the draws are E57's (`random_like` draws on the CPU). Stage 1's check: rescaled to E40, they read `qwen_reading/results/C_e40.json .placement.null` within 1e-3, on the same W0 maths set.

## Where the readings land
`experiments/qwen_d3_rows/results/table.json`, one line per state per number, for baseline-lead's comparator table (asked in its E92 notes; the lead replies).

## Seeds and cards
Seed 42's full state was trained on an H100, the others on L40s, so its spread is seed and card together. Seed 42's matched and past-line states are L40 reruns, 0.74% of the norm off the H100 path at step 100 (E91).

## Precision, per read
Weights and δW fp32 added in fp32 on the host; masked forwards fp32; KL and NLL per position fp32, summed fp64; X solved in fp64, stored fp32, stop over 1e-3 of the input-weighted norm. Norms in fp64, never fp32 `.norm()` (friction 475). Γ reads V at bf16 (`qwen_vpd.py:249`, friction 473), so at matched size least squares' own-mask read shows little of V′: row 4 is blind there.

## First-minutes checks
Per state, logged even when passing: norm against `size_grid.json` 1e-6; K0 on each row set equal to W0's, 1e-6; anchor 1 seed 0's K00 on its rows near E68's 0.3640 (E78 is E61's run, 1-cos 1.35e-6), logged, stop only over 1%; card over 44 GiB; host over 105 GB. Card from the logs: E57 42.6 GiB, E72 45.8 GB allocated; host E68 81 GB, E57 92.7 GB.
