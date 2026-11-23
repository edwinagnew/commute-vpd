# Review: E272 stage 3, E_b on the wide KL-hi set (gate 5)

A fresh reviewer read the plan's "Gates for stage 3" and "E_b stage on the wide set" sections, `eb.py` at 4e23f2e6, and
the code it calls (`kstar_needed/run.py`, `rb_states` `from_dw`, `commute.needed`, `stand_in`, `finetune`, the panel's
`run.py`/`tasks.py`/`score.py`). The E_b computation is E268's: same calls, same order, settings loaded live from
E268's predictions.json, LAST read, tf32 off. The paths, the KL-hi selection (36 files), the Gaussian matching and
what score.py reads all line up. The problems are in the rehearsal (it cannot finish under the 2 h debug cap), in what
the pods need beyond the state files, and in having no stop rule after stage 0.

## Blocks the run

**B1. The rehearsal cannot finish inside debug's 2 h, and squad's job never starts if stage 0 runs out of time.**
Plan:160, 206-208; jobs 64133/64134. Job 64133's log reports an L40 rate of 3,517 cases/s at 64 tokens. E268 ran its
RL items on an H100 at ~8,800/s (`kstar_needed/results/run_rl_p0.log`), so the L40 is about 2.5x slower. Stage 0 on
the L40 then takes the do-nothing (38,936 x 256 = 9.97M cases / 3,517 = 2,834 s), plus x2 (1,445 s x 2.5 = 3,600 s),
plus loading and identity: ~1.9 h against a 2:00:00 limit. One squad state takes ~2,952 x 2.5 = 7,400 s plus the task
build, so it times out before `e_b_file` writes its record. The only peak-memory number is `peak_gpu_gb` in that
record (eb.py:71), so the rehearsal never records squad's peak memory, and the plan says that is the number it waits
for. Job 64134 is also `afterok:64133`: if stage 0 times out, the squad job never runs. Fix: take squad's rate from the
per-site lines of `scan`'s log, and log `torch.cuda.max_memory_allocated()` after `IX.cache` and after the first site
(a one-line log in `e_b`, which changes no setting). Run squad independently (`afterany`, or no dependency) and accept
a timeout once those lines have printed. Memory guess: E268's 64-token RL items peaked at 11.9 GB at 1024 rows
(`kstar_needed/results/items/p256-*.json`; the plan's 11.3 GB at plan:154 is x2 at 199 rows, the wrong reference).
Scaling the forward part 4x for 256 tokens and adding a ~6 GB cache gives ~40 GB, which is tight on an L40 (48 GB)
but fits an H100. An OOM on the L40 is therefore not a no-go for the H100, but it would also leave no number.

## Fix before the run

**F1. The rehearsal's outputs land where the paid run reads them.** eb.py:44, 117-121, 127. The rehearsal runs in
this clone, so it writes `results/eb/stage0.json` (from the L40) and `results/wide/eb/squad.json` (one squad state).
Neither path is gitignored, so a plain `sync()` sends both. Two things follow. (a) Pods B-D pass the gate at
eb.py:117-118 on the rehearsal's stage 0 whether or not pod A's H100 stage 0 has run or passed, because the gate does
not check which card or which job wrote the file. (b) Pod A skips the rehearsed squad state (eb.py:127), so an L40
record goes into the scored set. A fresh scan on another card can move fragile argmaxes by up to the 0.04 tolerance
(`thresholds.eb.x2_E_b_K.source`). Fix: move both rehearsal files to a rehearsal folder before any pod is synced. Also
record `torch.cuda.get_device_name()` in eb's stage0.json, as `run.py` stage 0 already does.

**F2. The pods rebuild D from datasets and caches the plan does not stage, and some tasks would fail only after money
is spent.** eb.py:122-123, tasks.py:72, 157, 164, 166; data.py:49-60. `TK.build` on a pod needs the following:
- `cached_rows("val", ...)` for french. This function never downloads: it raises `SystemExit` when no file under
  `COMMUTE_CACHE_DIR/sets` holds the rows. French is second on pod C, so the job stops ~50 min in, after lower.
- `pile_rows("train", 0, 4096)` for french and lower. It streams from HF when the file is not cached, which takes
  minutes.
- E136's E53 windows and `HF.e53_train` for lower.
- Streamed Wikipedia for de/sw/es/fr/yo, arxiv, SQuAD, CNN/DM, CoNLL, SST-2 and DBpedia.

The LM tasks also re-pick their windows from W0 log-odds on the pod's card (`hard_positions`), so near-ties can make
D differ from the read's D. Fix (cheapest): on the cluster ($0), write each task's 200 windows to one small file in a
staging directory (`windows(r, 200)`: 200 x 256 int64 is 400 KB) with its sha256. Change eb.py to load that file when
it is present and to assert its sha256. This takes every dataset off the pods and makes D exactly the read's windows.
If that change is not made, the runner's brief must list and stage every file above. Note that the cluster rehearsal
cannot catch a missing file, because the cluster holds them all.

**F3. No stop rule after stage 0, so a bad record is kept and then skipped on resume.** eb.py:126-130; plan:162 (gate
4). A state whose dW failed to load or was read as zero has `n_changed` 0 and E_b exactly 1, by the convention at
stand_in.py:158. A nan, a wrong window shape or a 3x time overrun is likewise written and logged, and the resume at
eb.py:127 skips the record forever. Gate 4 asks for expected values that stop the code. Fix: after each state, stop
unless all of the following hold:
- `n_changed` > 0.99 x `n_vars` (E268's RL items: 38,936 of 38,938);
- `dd_max` > 0;
- 0 <= `E_b_K` <= 1 and finite;
- `toks.shape == (200, the plan's tokens for the task)`;
- the per-site rate is within 2x of the plan's figure.

Also take the window shape from the code, not from "the plan's 64, not re-read" (plan:188-189).

**F4. The 12 Gaussians (~25% of the cost) run last and nothing reads them.** eb.py:112; score.py:62-69 reads only
read-file `states`, never `random/` keys. The filed expected value ("random about 0.011", `expected.eb`) has no check
in code. The Gaussians cost 2,952 + 1,476 + 1,107 + 7 x 738 + 2 x 392 = 11,485 s, about 3.2 H100-h or ~$14. They sit
at the end of each task's list, so the null for a task arrives last, against the rule that the control runs first.
Fix: either (a) put each task's seed-0 Gaussian first in `states()` and name who reads it (a line in the card), or (b)
cut them to one Gaussian on pod C's first task as the null check. Either way, state it in the plan.

## Notes

- N1. LM-task D is the 200 hardest read positions, not a sample of them (eb.py:79; realistic.py:212 sorts the 500 read
  positions by ascending W0 log-odds). Prompt and label tasks get a seeded random 200 (tasks.py:140; realistic.py:62).
  Low-margin positions flip more easily, so K and E_b may differ by task type for this reason alone, and H1 is a
  cross-task tau. If this is kept, the card should say so. A stride (`r.toks[:: len(r.toks) // 200][:200]`) would
  remove the difference.
- N2. Sharding squad across pods after the rehearsal would lose records. Each shard writes the whole
  `results/wide/eb/squad.json` holding only its own states (eb.py:120-129), so the second download overwrites the
  first. Shard only with per-shard file names, or merge the shard files on download.
- N3. Gaussians are written in bf16 (eb.py:139), but the read used fp32 (run.py:469). Their norms differ by about
  1e-5 relative. This is harmless; fp32 would match exactly.
- N4. The filed scored set (the nine's KL hi, rung 0.256) gets no E_b from this stage, so its H1 reads E_b as missing.
  The plan says so (plan:183-185), but this departs from `eb.py`'s filed note in predictions.json ("reads the nine
  set's selection only"), so record it as a dated line before the run.
- N5. `write_stage` is not atomic. A pod killed mid-write leaves a JSON that the resume cannot parse (eb.py:121). The
  per-state downloads cover this; the runner should not resume onto a truncated file.
- N6. On a pod, `results/large` is a dangling symlink, and `random_file`/stage 0's `mkdir` (eb.py:87, 137) fail on
  it. The states must also sit at `experiments/convergence_panel/results/large/states/<name>.pt` exactly. The brief
  must replace the symlink with a real directory, as the repo rules say. Pod A also needs E247/E242's `large` npz
  arrays, P149's summary and x2's dW for stage 0 (`KN.saved`, `w.make`).
- N7. In stage 0, `peak_gpu_gb` for x2 is not reset first (eb.py:97), so it includes the do-nothing's peak.

## Checked and fine
- Paths: `set_dir("wide")/eb/TASK.json` is the file `score.records` opens (`res/"eb"/p.name`).
- File keys equal the read files' keys.
- `kl_hi` prefix selection gives 12 x 3 = 36 states.
- Gaussians use `random_direction(load_dW(f), C.random_seed)`, as the read did.
- Basenames are unique, so the `eb_random` files do not collide.
- `drops()` mtime check passes for files written after `dropped.json`.
- E268 settings are loaded live (derangements 4, examples 64, seed 268, rows 1024, logit_rows 8192, cache_batch 32).
- The `e_b` sequence matches E268's `read_item`.
