# E61 Job S: adversarial review of the size code (2026-11-02)

Scope: commit fd5b3a2, `experiments/qwen_recipe/size.py`, `results/predictions_size.json`, `cluster/qwen_recipe_size.sbatch`,
against plan Part 2 and Amendment 2, read with the helpers they call. Nothing ran on a GPU; no model loaded. One CPU measurement:
`~/commute-data/scratch/qwen-decomp-lead/size-review/bf16_x.py`, output `bf16_x.out` beside it.

## Must-fix

**M1. The split is read back in bf16, which the repo's own code forbids, and scored K10 and K11 can mix two precisions.**
`size.py:35,312` reuses `results/large/checks/{gold_s0,gauss}_X.safetensors`; `checks.py:314` stores X in bf16 and `checks.py:309`
reads any resume from that copy. `qwen_maps.py:142-145`: "Not bf16: the pieces X V, Xᵀ U amplify X's rounding" ("~10% in bf16 at
condition 50"). No record has read a bf16 X: all four `checks_k10_*.json` were first runs, no `resumed_from_bf16`. So the K10 and K11
reproduction the filing leans on used the fp32 X, and "on file" is untested.
- *Measured, gold_s0's δW, CPU, Frobenius, no Σ.* X rebuilt per site in float64; its bf16 rounding equals the file on every entry,
  so the rebuild is sound. Error of the applied change over |δW|, fp32 X then bf16 X: `layers.0` q_proj 5.6e-7, 3.7e-2; up_proj
  6.6e-8, 4.2e-3; down_proj 5.2e-8, 3.2e-3. The ratio is near 65,536, the two unit roundoffs. Scaling `stored_error` by it, worst
  site under Σ (inferred): gold_s0 8.8e-7 gives about 6%, the Gaussian 5.1e-6 about 34%. Not measured: the effect on K10 and K11.
- *Scenario 1, a stop.* `repro_gauss_file` (`size.py:298`) or `true_vs_checks_gauss` (`:218`) misses `rep_tol`, 1e-3. The two states
  that anchor every comparison cannot be read. The rehearsal should show this first.
- *Scenario 2, silent, worse.* A state E57 did not read runs its split in job 1 (fp32 X) and exits at the budget during the true
  pass. Job 2 reads the bf16 copy; the later chunks' K10 and K11 come from another X and join the same lists (`size.py:212-213`).
  On the sbatch's debug route this is the normal case: 30 + 128 + 526 + 154 s leaves about 12 of 16 chunks in job 1. These states,
  the two scaled changes included, have no K10/K11 reference (`:297`, K0 only), and no rule reads `resumed_from_bf16`.
- *If it passed:* gold_s0 and the Gaussian sit on another X precision from the ladder and scaled changes, inside one gap.
- *Smallest fix.* Save X in fp32 in `split_x` (`checks.py:314`; its jobs are finished; 5.6 GB, 7 states, 39 GB) and delete `X_ON_FILE`
  (`size.py:35,312-313`), so each state runs its own split: +12 min. `predictions_size.json:94` and the sbatch's times change; re-file.

## Should-fix

**S1. The second size axis has a reference and no stop.** `size.py:317-321`; every gap reads `KL_W0_state`. The filing lists 0.8058,
0.5261, 0.4811 as its control, and nothing compares them. *Scenario:* a stale weight set or swapped lists move the KL axis, so B or C
moves, no stop. *Fix:* after `:321`, for `READ_BY_E57` on 128 rows, `lab.check` both KLs against `checks_sizes.json` within `rep_tol`.

**S2. A scaled change's read is not tied to its search.** `read_cells:297` checks K0 only. *Scenario:* `state_of`'s `mul_(b)` and the
search's `put(dW, b)` disagree: `gauss_big` reads at another size and the interpolation absorbs it. *Fix:* in `read_one`, `*_big`'s
K00 - K0 within `rep_tol` of `search_<kind>.size`; ladder states' against `select`'s `size_select` on the first rows.

**S3. A scaled change's X is keyed by name, not scale.** `size.py:312`. *Scenario:* amendments are routine here. A rerun of `search`
gives a new b; `split_x` finds `size_gauss_big_X` and applies the old b's X to the new δW, with no K10/K11 reference to catch it.
*Fix:* put b in the file name.

**S4. The rehearsal does not run the multi-chunk path.** `CONFIGS[2]` has `rows=2, chunk=2` (`size.py:66`): one chunk, so `t[c] += k[c]`,
the K00 slice at `s > 0` and a resume inside the pass never run. *Fix:* `chunk=1` (+26 s a state); in the runner's brief, `scancel`
after the first `true_from_row` log line and resubmit.

**S5. The small state's length does not exist.** `predictions_size.json:58`: "its drop and length are in Job R's record". `train.py:79-80`
has `gen=(100,)`: length at step 100 only (the drop is there, `held_at`). *Scenario:* "a state on the way" reported on half its
evidence. *Fix:* after `select`, one Stage G read of that state (a new flag, about 4 min), or file that length is not read.

**S6. Host memory: restore the stop.** Amendment 2's "128G checked on the rehearsal" cannot hold at 2 rows: 4.8 GB of masks, not 19. The
failure is cheap, not a wrong number, but opaque (exit 137, outside the sbatch's 0/1/3) and repeats on resubmit. *Fix:* in `true_chunk`,
after the mask pass, a named stop when `ru_maxrss` passes 118 GB. `control`'s 8-row chunk is the first real reading, at minute 6.

## Note

- `predictions_size.json:71`: "toward C" holds only if a kind's share rises with size. If it falls, `at_size:123` is toward B. Report which.
- `true_vs_checks` runs after the whole pass (`size.py:215-218`); after the first chunk it saves 11 min on a miss.
- `rules.gold_s0_size` is read by no code (`size.py:268` recomputes it, same value). Delete it.

## The writer's choices

- **Masks per 8 rows: stands.** Masks are per row, so chunking cannot move a number; K00 per chunk is the check.
- **Random changes scaled as weight changes: stands.** One line through W0 for both kinds, which is what two reads and an
  interpolation assume. The write-up must not say "alive columns moved by b".
- **0.99 on 128 rows: stands.** 1.44 was filed from 128; 16 rows run 1.09 times high for gold.
- **No sha256 for the Gaussian's split: moot after M1.** The reproduction would catch a swapped file; it does not make bf16 right.
- **No do-nothing arm: stands.** K10 = K00 by construction. W0's zero split cannot see g0 and g1 swapped, but `true_vs_checks` can.
- **No host stop: comes back,** S6. The line limit was the only reason.

## Cleared (by reading)

- **Masks.** g0, g1 from original columns before `applied`; nothing calls `importances` inside; K10 g0, K11 g1.
- **Determinism.** K00 equal to all digits, pass against cells, on four states (`run.log`), so `id_tol` per chunk holds; one card type.
- **Status rule.** K11 near K0 on the states on file, so it should not fire. **GPU peak.** `cells` resets after any split; no stale peak on resume.
- **Resume, budgets.** Atomic writes; `applied` restores on a stop; each debug job leaves room for chunks; kill margin 300 s.
- **Rules.** `line`, `floor`, `nearest`, `at_size` match the filing; bootstrap draws paired; ladder glob skips `ckpt.pt`; steps 5, 9 exist.
- **Filed numbers, recomputed:** 1.5731, 0.8957, 0.6830; 0.06558, 0.96128; 1.4438; 0.0917, 0.0221, 0.0291; 0.037, 0.067. All match.
- **Not cleared:** the verdict stage's CPU import under `--mem 8G`, unrun here; the rehearsal runs it on the GPU job.
