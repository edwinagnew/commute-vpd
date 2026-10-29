# E55 Amendment 1: code review (commit f4e384b)

Scope: `git show f4e384b`: `experiments/pcfg_anchors/amend1.py`, `results/predictions_amend1.json`, the `finetune()` and
`state_path()` changes in `run.py`, the `amend1` stage of `cluster/pcfg_anchors.sbatch`. Reviewer did not write it.
Nothing blocks the run. Two findings are about how the route and the count-c check are read, not about the
arithmetic. Both are cheaper to fix in the filed text before the run than in a dated report after it.

Model-free checks were run through the code's own functions on the filed 8,000 rows
(`~/commute-data/scratch/reuse-lead/e55-a1-review/check_rules.py`, via `fleet/cpu-run`): synthetic answerers scored by
`distinct`, `const_floor`, `follows` and `route`. The numbers quoted below come from that script.

## Findings

### 1. A pure count-c answerer reads route "mid" (should fix, before the run)
`amend1.py:110-112` (`route` reads PRIMARY only) and `:131-132`; `predictions_amend1.json:40` (outcome "route mid"),
AM2.

Every row has a + b + c = 40, so mid = (41 − c)/2 exactly on c-odd rows. On the other rows, mid equals c_floor where
d = b − a − 1 > 0 and c_ceil where d < 0. On mid's 1,376 distinct rows (d ≥ 2 or d ≤ −4), c_floor differs from mid
only at d = −5 and −7, which is 177 rows. Answerer → reading:
- answers c_floor on every row → S_mid 0.871 > hi_mid 0.812, **route mid**, c_check [c_floor];
- answers c_ceil on every row → S_mid 0.866, **route mid**, c_check [c_ceil].

So AM2 ("route mid") scores as met for a rule of count c alone. The "route mid" outcome then says "it uses LOW's
count a together with a count b", which would be wrong. c_check does flag the case, but nothing ties the two
together, and a reader of the route alone gets it wrong.

**Fix:** make route mid conditional on an empty c_check. With a non-empty c_check, report the route as e.g.
"mid or count c". Write that condition into AM2's claim and into the outcome text (dated, before the run). A pure
midpoint answerer still reads mid with c_check empty (checked: S_c_floor = S_c_ceil = 0).

### 2. c_check "follows" does not identify a count-c rule (should fix the outcome text)
`amend1.py:132`; `predictions_amend1.json:44` ("contradicts the review's bound … flagged loudly"), AM4.

c_floor's distinct pool is d ∈ {−3, −5, −7} (1,205 rows) and c_ceil's is d ∈ {3, 5} (184 rows). On those rows the c
rules are the same as the midpoint rounded towards b. Answers that are not count c, and how they read:
- the midpoint rounded towards b → c_check [c_floor, c_ceil] (S 1.0, 1.0); route mixed;
- a b-only rule, b moved halfway towards 14 (trunc) → c_check [c_floor, c_ceil] (S 0.915, 0.946).

Either one would raise the "PA3 blind spot (i) live after all" alarm. This is the identification problem the Stage A
review already names (finding 7): no behaviour test can vary c with a and b both fixed.

**Fix:** reword the outcome. A c rule labelled "follows" means "consistent with floor or ceil of (a + b + 1)/2 on these
rows", not "count c alone". Keep it as a report line, not an alarm. Or drop the c rules and keep the table (by
b − a), which shows rounding directly.

### 3. Reproduction is checked indirectly and only after the full arm (minor)
`amend1.py:167-172, 186-188`.

The re-run is compared with a2 only through B and P at 1K and 10K, with tolerance 0.05, and only after all 10K steps
and six reads. The recipe matches a2 (see "Checked and fine"), and two states can be compared directly for free.
Logging max |Δw| between `results/large/amend1/<key>_1000.pt` and `results/large/a2/<key>_1000.pt`, and likewise at
10K, says whether the trajectory is the saved run (≈ 0) or a sibling run. That is what AM5 "reproduces" is trying to
decide, and B/P within 0.05 cannot settle it. The 1K comparison is ready about 20 s into the arm. Since the run costs
$0, there is no need to stop on it; just log it.

### 4. The table puts non-number answers in the offset −7 bin (minor)
`amend1.py:133`.

A non-number answer gives `pred` < 0, and `np.clip(pred - (a+1), -7, 7)` sends it to −7. That is the same bin as a
real answer of a − 6. E7 says non-numbers count in `other`, which is correct, but in the table they cannot be told
apart. **Fix:** give pred < 0 its own offset key (e.g. "nan").

### 5. The log leaves out some scored numbers (minor)
`amend1.py:136-137, 162-163`.

The rule-read log line prints overall, S and n per rule, but not each rule's label or its f_r and hi_r. The saved
section never logs the A2 labels it attaches. CLAUDE.md asks for every scored number in the log. The risk is low,
because `amend1.json` is written on the shared filesystem. **Fix:** add `label` and `const_floor` to the f-string.

### 6. Filed text is stale in places (minor, fix with finding 1's amendment)
`predictions_amend1.json`:
- **L50, E3:** says "S_r == 0.5 … → neither". The bar is hi_r now, not 0.5.
- **L94, AM4 blind_to:** "c_ceil may be void on a draw". The rows are fixed (seed [1, 70]), and c_ceil's pool is 184
  every time.
- **L48, E1:** says "~170 expected"; the pool is 184.
- **L77-78, AM1:** quotes hi_a_plus_1 = 0.729. The code computes 0.728 from the unrounded f_r, so this is harmless.

### 7. A missed control is sticky (minor)
`amend1.py:198-203`.

After an exit 1, `amend1.json` keeps `controls`, and every rerun exits 1 at once without re-reading. That is
intended, but neither the sbatch header nor the docstring says how to clear it (delete `results/amend1.json`). Say
so in one line.

## Checked and fine
- **Rules.** mid's formula and its rounding towards a + 1 are correct (the four hand cases pass). All five non-a rules
  equal the label on 2,000 spurious texts, and a + b + c = 40 holds.
- **Filed numbers.** Distinct pools 3,904 / 4,863 / 1,376 / 3,810 / 1,205 / 184 and f_r 0.457 / 0.305 / 0.624 /
  0.353 / 0.624 / 0.696 reproduce from the code.
- **Synthetic answerers.** Each pure primary-rule answerer reads its own route with S = 1 and 0 elsewhere (the
  primaries' distinct masks are mutually exclusive). The constant answers 13 and 14, a 40/60 mixture of a + 1 and
  mid, and a-only or b-only shrinkage answers all read mixed. A pure mid answerer gives B_a 0.402 and B_b 0.000 on
  qualifying rows, matching AM2's lean.
- **Edge handling.** `follows` is strict and maps nan to "neither". `reproduction` counts nan as a miss. Rules are
  never negative, so a non-number answer never matches one.
- **No change to a2.** The `run.py` defaults leave a2's behaviour, paths and state keys unchanged (W1-0b625b90,
  W1s1-7abe2adf; N1's control reads a2/N1-a59cb52a_10000.pt).
- **The trajectory uses a2's recipe.** It starts from the same A1 checkpoint, with the same `manual_seed`, Schedule,
  Batches seed and chunk (data are fixed by step). The optimizer and the CPU and CUDA RNG states are restored at each
  stop. Every save step is a multiple of save_every = 1000. The extra stops at 2K/3K/5K/7K are therefore equivalent
  to a2's single 1K→10K call, up to compile and bf16 nondeterminism.
- **Resume.** A deadline save at any step resumes correctly. `train.pt` is removed only after the reads.
  Trajectory files go to `results/large/amend1/` and never touch a2's.
- **Order.** Controls run first, alone, and exit 1 on a miss before any W state is read.
- **sbatch.** The `amend1 0` case and the resubmit with empty args work under `set -u`.
- **Time.** The 22-minute deadline sits inside the 30-minute walltime. a2.log shows about 3 minutes per arm
  including reads, so the reads that follow an arm's deadline are short.
- **Lint and tests.** `make lint` is clean on both files. `make test`: 410 pass; the 2 failures are in
  `tests/test_qwen_maps.py` and are unrelated.

## Not run
No model was loaded and nothing ran on a GPU. Compile and bf16 nondeterminism were not measured.
