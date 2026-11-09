# Review of E137's code that has not run yet (job 57043, commit cdf2b0ff)

This review read the parts of the E137 run that job 57043 has not reached yet: the rest of stage 2, stage 3, the rules
step, the rank step and the resume after a wall-limit exit. It found one certain crash, an out-of-memory error at the
`final` site of the first pair, which recurs on every resubmission until one line is added, and it found that stage 2
needs about 5.4 hours and so four debug jobs, not the one or two the job script assumes. The live job can run on
untouched until it reaches that crash at about 20:48 UTC, because everything it has written is kept and the fix can go
into the clone now without affecting the running process.

Read-only: no job, model load or dry run. PyTorch's solver paths were read from the v2.8.0 source (the venv has torch
2.8.0), saved in `~/commute-data/scratch/definition-lead/e137-review/src/`.

## Real defects

### 1. Out of memory at the first `final` solve (certain, blocks the run)

`src/commute/hidden_maps.py:178-183`, reached from `experiments/reuse_variable_test/search.py:142` via `run.py:96`.

`Ridge.solve` keeps `A` alive while calling `torch.cholesky_solve`. On a single matrix that call takes the cuSOLVER path,
and `_cholesky_solve_helper_cuda_cusolver` clones the factor (`BatchLinearAlgebraLib.cpp:915-916`,
`cloneBatchedColumnMajor(A)`). At p = 36,864 each of these is 10.125 GiB, and four are alive at once.

| Held on the card at `cholesky_solve`, site `final`, first penalty | GiB |
|---|---|
| `self.xx`, the 36,864-square float64 X'X | 10.13 |
| `A`, the clone with the penalty on its diagonal (a local, still alive) | 10.13 |
| `L`, the factor (`cholesky_ex` returns a new tensor) | 10.13 |
| the clone of `L` made inside `cholesky_solve` | 10.13 |
| `X_val` (20,000 x 36,864 fp32) | 2.75 |
| `Y_val` (5 sites x 20,000 x 768 fp32) | 0.29 |
| `xy` (5 targets x 36,864 x 768 float64) | 1.06 |
| the clone of the right-hand side | 0.21 |
| the component model (checkpoint 2.71 GiB, plus the target) | about 2.9 |
| **Total** | **about 47.7** |

The L40 has 44.3 GiB usable (`cluster/README.md:3`), so this is about 3.4 GiB over. The job script's "about 36 GB"
(`cluster/reuse_variable_test.sbatch:13`) counts X'X, one copy and one factor, and misses the clone. It also counts in
units of 1e9 and leaves out the model.

The arithmetic shows why stage 0 passed and why `out3` will pass: at p = 30,720 the three smaller matrices are 7.03 GiB
each, so the peak is about 38.4 GiB. Only `final` fails, and it fails at the first penalty of the first pair, on every
resubmission, because `final` is the last missing site and the OOM exception exits 1. The sites `in1` to `out3` are
written before it, so nothing is lost but `h_fit` and the Slurm chain.

On the log's timings, job 57043 reaches it about 5,200 s in, near **20:48 UTC**. `fleet/resubmit` would end the chain
there, since exit 1 is not exit 3.

Smallest fix, one line, no change to any number:

```python
        L, info = torch.linalg.cholesky_ex(A)
        del A  # cholesky_solve clones L; xx + A + L + that copy is 40.5 GiB at p = 36,864
```

That gives a peak of about 37.6 GiB. `hidden_maps` is imported only by this experiment, so no frozen caller is affected.
The running process imported the module at start, so editing the file does not touch job 57043; the next job picks it up.
Two ways to use it, both fine: let 57043 run on to the crash and resubmit by hand (it costs one `h_fit`, about 80 s,
and the crash must be noticed), or cancel now and resubmit with the fix (it costs the search in flight, under 2 min, and
makes the next job clean). Either way the fix must land before the next job starts.

### 2. Stage 2 takes about 5.4 h, not 30 min: four debug jobs, three resubmissions (certain on G, estimated on H)

`cluster/reuse_variable_test.sbatch:7,10-12`, `run.py:12` and the plan's "Under an hour" say two jobs at most.

- **G search.** The log gives 115.8 s mean over the first 14 searches (`.G[].s` in `stage2.json`), 108 to 122 s. The
  site changes it by about 10 s; k does not (in1 at k = 4 read 120 s against 121 s at k = 1). For 144 searches that is
  about 16,700 s, or 4.6 h.
- **H per pair, about 650 s (estimated, no H has run).** `h_fit` about 80 s, Cholesky solves about 90 s, and five sites
  at about 90 s each (refinement and read). The Cholesky figure scales from stage 0's capacity read, which took 123 s
  including about 10.8 Cholesky equivalents of the 30,720 size, about 6.8 s each, or about 1.4 TF in float64.
- **Total.** About 19,300 s, or 5.4 h, against 5,766 s of stage 2 in the first job and about 6,270 s in each later one.

| Job | Work, with defect 1 fixed | Exit |
|---|---|---|
| 1 (57043) | stages 0 and 1; Always on (36 G, H); about 9 Cue G | 3 at about 6,300 s |
| 2 | Cue, 27 G and H; about 22 XOR G | 3 |
| 3 | XOR, 14 G and H; about 34 Table G | 3 |
| 4 | Table, 2 G and H; stage 3; rules; rank | 0 at about 2,300 s |

So **three resubmissions**. A fifth job appears only if H or G runs at least 25% slower than this. Submit the rest with
`fleet/resubmit` (it follows exit 3), and do not read a TIMEOUT as a chain that continued (defect 3).

### 3. A TIMEOUT can end the chain: `_h_pair` checks the budget once, not per site (medium)

`run.py:88-92`. `_budget` runs once before `h_fit`, so a pair's H (about 650 s) started at 6,299 s ends at about 6,950 s,
past the job's 6,900 s. Slurm kills it with TIMEOUT, so the exit is not 3 and `fleet/resubmit` stops. Nothing is lost,
since each finished site is on disk, but someone must resubmit by hand. The schedule above does not hit this, since H
starts at 4,700, 3,160, 1,650 and 260 s; a slower pair moves those starts later.

Smallest fix: call `_budget(cfg, out)` at the top of the `for site in missing:` loop (`run.py:94`) as well. A resumed job
redoes `h_fit` (about 80 s) and only the missing sites.

### 4. Stage 3 and the rank step have no wall check, and the rank step does not resume (low, after the rules)

`run.py:236-238`. In a job that ends stage 2 late, stage 3 (a few minutes, all-or-nothing `dump` at its end) or the rank
step (no timing yet, the job script guesses 5 min per pair) can hit the 6,900 s limit. Only unscored rank numbers are at
risk in the rank step, and `rank.json` is rewritten from the start on a rerun. The planned job 4 leaves about 4,600 s.
No fix needed for this run, provided the last job starts stage 3 with time in hand.

### 5. Four places the code differs from the filed text or the plan (none changes a filed number in this run)

- `rules.py:59-66` against `predictions.json` card_rule edge case. "A missing start gives 'no reading'", but `card_rule` tests only
  for an empty list, so two starts of three would be averaged. Unreachable here: `stage2` completes only with all three.
- Appendix D, "Ties go to the highest reversion and lowest mismatch". `predictions.json` says no tie-break, and the code applies none. Not a conflict with the filed file; the appendix line is stale.
- Appendix D, stage 0, "Rank of z_M over D". The code reads it last, on 34,000 windows, unfiled and unscored, as
  `not_scored_but_reported` in `predictions.json` says. The appendix line is stale.
- Plan, "At the output the fit is a convex problem, so there the best map is found up to convergence", against
  `search.py:175-208`: `h_refine` runs a fixed 200 steps and has no convergence check, and if the refinement is worse it returns the ridge
  (`kl_val_ridge` and `used` are in `stage2.json`, so a reader can see which). The H read at `final` is a lower bound as the
  filed text says, but the plan's sentence is stronger than the code. Whether 200 steps of Adam at 3e-4 on a (p x 768)
  coefficient matrix helps at p = 36,864 is unknown: low confidence, read `used` and the two KLs. Also, the capacity
  read covers the four sites in `SITES`, not `final`, while the filed rule says "every site's z_M"; stage 0 is done and all reads were 0.506 to 0.507.

## Every linear-algebra call and its size at runtime

| Call | File:line | Size | Backend | Status |
|---|---|---|---|---|
| `qr` of the free matrix | `search.py:113,120`, `pieces.py:220,229` | (768, k), k up to 16 | cuSOLVER | Ran |
| `qr` for random Q | `controls.py:26,46` | (768, 4) | CPU | Ran |
| `svd` of the F-minus-M differences | `search.py:100` | up to (8,192, 768), fp32 | cuSOLVER gesvdj | Ran (start 2) |
| `matrix_rank` of C | `pieces.py:323` | (768, 768), fp32 | cuSOLVER | Ran, rank 766 |
| `cholesky_ex`, ridge | `hidden_maps.py:180` | p = 12,288 to 30,720, float64 | cuSOLVER xpotrf | 30,720 ran in stage 0 |
| `cholesky_ex`, ridge | `hidden_maps.py:180` | **p = 36,864** | cuSOLVER xpotrf | **Not run** |
| `cholesky_solve`, H | `hidden_maps.py:183` | p up to 36,864, 768 right-hand sides | cuSOLVER xpotrs | **Not run** at 768 RHS; OOM at 36,864 (defect 1) |
| `svdvals`, `svd` of E's activations | `run.py:132,134` | (2,048, 768), float64 and fp32 | cuSOLVER gesvdj | Same size class as the runs above |
| `eigvalsh` of X'X | `search.py:277` | 30,720, float64 | MAGMA | **Not run** |

Failed earlier: the `gesvdj` buffer query on (34,000, 30,720) and the `syevd` buffer query at 30,720. Both raised
`CUSOLVER_STATUS_INVALID_VALUE` in the `_bufferSize` call, before any computation. Neither call is on a path that remains.

- **Same failure at `final`?** `xpotrf_bufferSize` passed at 30,720, the size where those two refused, so whatever limit
  stopped them did not stop it there. A repeat at 36,864 looks unlikely, but it is unverified. If it does fail, the fallback is the
  factorisation on the CPU (10.1 GiB, fits the 32 GB host, about 1-2 min on 4 cores). Not worth coding unless it happens.
- **MAGMA `eigvalsh` at 30,720.** MAGMA is in this torch build (`libtorch_cuda_linalg.so`) and
  `linalg_eigh_magma` is the selected path (`BatchLinearAlgebra.cpp:1959-2004`). Its host work array is lda x lda
  (`BatchLinearAlgebra.cpp:1909`), 7.0 GiB, which fits the 32 GB. Time unmeasured, last step only.
- **Cholesky failure on a real matrix.** At the smallest penalty, `n x lam` = 80, far above fp32-sum rounding noise in
  X'X (about 0.2). `info` is checked and raises.

## Memory per step

| Step | Card, GiB (of 44.3) | Host |
|---|---|---|
| `Ridge.add`, p = 36,864: X'X, the cast temporary, Z'Z fp32, xy, model | about 29 (`add_` still makes a float64 cast of Z'Z, 10.1 GiB, despite `hidden_maps.py:170`'s comment) | small |
| `h_fit` end: X'X, `X_val`, `Y_val`, xy, model | about 17 | small |
| `R.solve` at `out3` | about 38 | small |
| `R.solve` at `final` | **about 47.7, OOM** (about 37.6 after the fix) | small |
| `h_refine` at `final` | about 18 plus small batches | small |
| `z_rank` before `del`: chunks, X, X64, G | about 25.5 | small |
| `eigvalsh` in MAGMA: G, its copy, work | about 17 | 7.0 GiB |

Host RAM: the Pile rows and windows stay under 1 GB, and 32 GB is ample at every step. `hard_exit` prints the peak.

## Time

Job limit 6,900 s; `wall_limit_s` 6,300 s (`pieces.py:62`), checked before each G search; a G search ends by about
6,420 s, which is safe. `elapsed()` starts at import, so model load counts. Measured: stage 0, 316 s; stage 1, 218 s; a
resumed job's start-up about 30 s (`setup` plus `Net`, both from the log). Total and job count are under defect 2.

## Looked at and found sound

- Resume after exit 3: `stage2.json` is written atomically after every search and site, keys are skipped before `_budget`,
  each search draws from a seed set by its key (`zlib.crc32`), and `pair_windows`, `task` and `pieces.pt` rebuild the
  same state. The `pieces.pt` file exists (2.4 MB). A partial H resume refits and does only the missing sites. `MoreToDo` returns 3 through `hard_exit`.
- Keys between stages and rules: `pair/k{k}/{site}/s{start}` and `pair/{site}` match `run.py:66,99` to `184,187`; `st1`'s witness
  fields, `st0`'s capacity and `pairs`, and `st3`'s `sites` are all read under the names written.
- `final` reads and edits: `lm_head` is called as a module with one positional argument (`llama_simple_mlp.py:366`), the
  pre-hook fires, `FINAL` is popped before `Net.run`, and `up_to`/`flat`/`p_of` agree on 36,864 columns. Device and dtype paths through `h_fit`, `ridge_select`, `h_refine`, `_ridge_s` and `read_pair` hold.
- Stage 3: shapes, the fixed permutation shared across directions, `h[perm]` on a CPU index (same pattern as stage 1), `acted_on_count` against `tol_tv`, the span from right singular vectors. Rank 0 would give a NaN where the filed text says no directions, but activations are not zero.
- `rule_read`: witness checks, `best_site`, `card_rule`, search bars against stage 1's witnesses, `built_possible` against stage 0's held-out reads, all match the filed text apart from the items in defect 5.
- Ridge: standardisation, centring through `y_mean`, the nested first p columns, `n` counted once in `stats`, the validation half matching `h_refine`'s `n_val_start`.
- Start 2 at `in1`: the pieces act at `in2`, so F minus M at `in1` is zero and this start is an arbitrary basis rather than principal directions. The log shows it runs and gives a normal number; it is one fewer informative start at that site only.
- `z_rank`: the rank is reported only. Its docstring's "1e-8" noise floor is typical rather than worst case, and the 1e-6 cut stays above it in practice.

## Not run

No code was executed, no job submitted and no GPU touched; the memory figures are arithmetic from the code and the PyTorch source, the H timing is scaled from stage 0's capacity read, and MAGMA's time at 30,720 is unknown.
