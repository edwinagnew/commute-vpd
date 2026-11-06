# Appendix to the E104 plan: files, timings, numbers, builds

*Apparatus for `reports/plans/2026-11-06_plan_e104-d3-gsm8k.md`. Grepped 2026-11-06. Paths under `experiments/`
unless stated.*

## State files
All `torch.float32`, 5.64 GB each, checked with `fleet/ckpt-info` (F_s0 step 10, N_s70 step 10) and `ls`. Run
directories under `~/commute-data/results-large/qwen_rl/` (the target of `qwen_rl/results/large`): `lr1e-6w10/dW/{R,N,F}/`
(seed 0) and `lr1e-6w10_s70/dW/{R,N}/` (seed 70; no `F/`). Each holds steps 1, 2, 5, 10, 20, 50, 100.

| state | norm, fp64 (`qwen_size_grid/results/reads.jsonl`) | E91 kl |
|---|---|---|
| R_s0 10; 100 | 0.06866028351856503; 0.3508836925461713 | 0.0148; 0.0847 |
| R_s70 10; 100 | 0.06777104232097368; 0.3468409803773495 | 0.0096; 0.0872 |
| N_s0 10; 100 | 0.06488850112221688; 0.3698454509629327 | 0.0022; 0.1087 |
| N_s70 10; 100 | 0.06466251144339095; 0.3353582630832081 | 0.0059; 0.0429 |
| F_s0 10; 100 | 0.058167923146116025; 0.2738619803765314 | 0.00024; 0.0010 |

Step 100 is each arm's `full` in `size_grid.json` (`.arms.<arm>.full`); step 10 is only in `reads.jsonl`, whose records
carry `dtype`, so `d3.load_state`'s dtype check applies unchanged.

## Numbers quoted
- **E93 maths base cells**, `qwen_d1_table/results/<state>/state.json`, `.base.K00` pooled sum over rows / 32768
  (128 × 256): R_s0 0.138290816, R_s70 0.166882772, N_s0 0.205263055, N_s70 0.130408289, F_s0 0.071043136; K0
  0.069613195 everywhere. So K00 − K0: 0.0687, 0.0973, 0.1356, 0.0608, 0.0014. E93's control reproduced E94's E40 cells
  (1.37381, 0.104363), so the definitions agree.
- **E93 share left in Δ** (unweighted, `.split.left / .split.size`): R 0.09242, 0.09425; N 0.08923, 0.09213. Means
  0.09334 and 0.09068, gap 0.00265; spreads 0.00184 and 0.00290. The source of prediction 4's lean only: E104's outside
  share is input-weighted (on the template, unweighted 0.087 against its null 0.090, input-weighted 0.123 against 0.212).
- **E94 anchors and randoms**, `qwen_d3_rows/results/table.json .numbers.<state>`: outside, null, placement excess,
  T, L, |S|, size.

| state | outside | null | placement | T | L | S | size |
|---|---|---|---|---|---|---|---|
| a1_s0 | 0.12335805 | 0.21190688 | −0.00764474 | 3.66258 | 4.22221 | 1286 | 0.1276 |
| a1_s1 | 0.12363794 | 0.21347843 | −0.00827809 | 3.75245 | 4.41822 | 1320 | 0.1325 |
| TOFU s42; s142 matched | 0.09649; 0.09598 | 0.21454; 0.21238 | 0.15795; 0.16242 | 3.7402; 3.7064 | 3.3418; 3.3686 | 1847; 1937 | 0.219; 0.229 |
| TOFU s42; s142 full | 0.18551; 0.18402 | 0.29070; 0.29085 | 0.01673; 0.01363 | 3.07541; 3.01229 | 3.3287; 3.1774 | 7805; 7987 | 0.611; 0.660 |
| Gaussian small; big | 0.21198; 0.29055 | 0.21191; 0.29070 | 0.00274; 0.00215 | 3.1309; 2.9519 | 3.0650; 2.9281 | 2423; 6460 | 0.145; 0.531 |
| alive small; big | 0.00031; 0.00127 | 0.25196; 0.25142 | 0.09224; 0.09114 | 3.2502; 3.3554 | 3.4492; 3.5322 | 2120; 5170 | 0.120; 0.625 |

- Template fold noise: `qwen_d3_rows/results/a1_s0/state.json .overlap.T`: o 254, chance mean 69.35, sd 7.79, so the
  fold's draw sd is 0.11. With the template's |S| of 1286 the mean is 5 at about |S| = 93.
- Placement null draws: E57's four nulls on E40, 0.5858, 0.5820, 0.5839, 0.5830 (`qwen_d3_rows/results/predictions.json
  .controls.e57_nulls`), range 0.0038; E94's report: "the Gaussian null's own draws spread under 0.005".
- Prediction 3's bar: template T mean 3.7075173, seed spread 0.0898724; TOFU full spread 0.0631235; bar 3.7075173 −
  0.0898724 = 3.6176449.
- Cochran's rule: an expected count of at least 5 per cell before a chance comparison is read (Cochran 1954, "Some
  methods for strengthening the common χ² tests"), here the chance mean overlap.

## Time per pass, one L40 (E94, `qwen_d3_rows/results/run.log`)

| pass | time | lines |
|---|---|---|
| Load target, components, Γ | 50-171 s | `:3, :27, :104, :181` |
| Σ, maths rows of 512 | 172-181 s, once a job | `:17, :28, :215` |
| δW file and norm | 5-10 s | `:104-105`, `:29-30` |
| Least-squares split with outside share | 405-759 s | `:70, :107, :31, :49` |
| Base cells with gate counts, 128 maths rows | 118-424 s | `:197, :108, :32, :50` |
| Overlap, placement, files, score | 3 s | `:151` to `:155` |
| Size search, 16 rows, 4 evaluations | 63 s | `:144-147` |
| Unit nulls (reused, not rerun) | 904 s | `qwen_d3_rows/results/nulls.json .seconds` |

Cut from E94's per-state sequence: the new pass (202-267 s, `:35, :74, :111`) and own rows.

## Builds
| # | build | from | note |
|---|---|---|---|
| 1 | `qwen_d3.step_state(arm, step)`: a record from `reads.jsonl` with an absolute path | `grid_state` | new name |
| 2 | `qwen_d3.placement(d, pe, on0, energy, unit, unit_energy, seeds)` | `qwen_d3_rows/run.py:174-180` | E94 keeps its own copy |
| 3 | `qwen_d3.carried(d, maths, dW, nulls)`: split, base cells with c0 and c1, folds with draw sd, placement, size | `qwen_d3_rows/run.py` `finish` without own rows and new pass | new name |
| 4 | scaled copy: `{s: c · δW_s}` from a state, `qr.scale_search` on K00 − K0 over 16 rows | `qwen_d3_rows/run.py` `read_random` | in E104's `run.py` |
| 5 | rules: C1-C3, the borrowed bar, size, p, predictions, `--edges` | `qc.separates`, E93's borrow (`qwen_d1_table/rules.py:270-275`) | E104's `rules.py` |

**Names E94's and E93's frozen `run.py` import from `qwen_d3`**, none of which may change name, signature or behaviour:
DEV, Decomp, F64, GIB, MATHS_ROWS, POOLED, ROW_LEN, RowSet, Sums, alive_pieces, by_class, ceiling, cells, clean, folds,
gold_held, grid_state, host_gb, load_state, ls_change, ls_parts, ls_split, maths_rows, norm, null_pe, null_seed, own_rows,
peaks, pinned, think_refs, tofu_kept, tofu_seqs, total, unit_nulls, wrong_change. A change to any is a break of a
finished run; add a new function instead.

## Reused from E94
`qwen_d3_rows/results/large/unit_nulls.safetensors` (15.8 MB, sha256 starts `05ff807e6c834b37`) and
`qwen_d3_rows/results/nulls.json` (`.energy`), read, not copied; stage 0 records both hashes. W0's per-row K0 on the
maths rows: `qwen_d3_rows/results/control.json .W0_K0_rows.maths`. T and L: `qwen_reading/results/large/B2/`.

## Jobs
```
sbatch cluster/qwen_d3_gsm8k.sbatch --stage control
sbatch cluster/qwen_d3_gsm8k.sbatch --states R_s0_10,R_s0_100,R_s70_10,R_s70_100
sbatch cluster/qwen_d3_gsm8k.sbatch --states N_s0_10,N_s0_100,N_s70_10,N_s70_100
sbatch cluster/qwen_d3_gsm8k.sbatch --states F_s0_10,F_s0_100,S1,S2,S3
```
As `cluster/qwen_d3_rows.sbatch`: `debug`, one GPU, 8 CPUs, `--mem=110G`, 2 h; exit 3 resubmits (`fleet/resubmit
--code 3`). Card peak in E94 42.5 GiB, host 88.4 GB (`:151`, `:186`). Job D worst case: 2 × 20 + 3 × 23 = 109 min, under
the 117 min limit, and exit 3 covers an overrun.

## Seeds and cards
RL, null and own-samples seed 0 were trained on one H100 (E69), RL and null seed 70 on one RunPod H100 (E69 Amendment 1,
`TRIED.md:177`), so the seed spread is the seed on one card type. Whether any saved step was re-made on another card
was not checked; the per-state norm check against `reads.jsonl` pins the file read.
