# Reading refactor (P2.79, E309): writer notes

## Done (phase 1, 2026-11-24/25)
- `old.py` (harness): every reading of the plan's "States to check" through the OLD path (`load_experiment` and all),
  to `results/old_<tag>.json` keyed `<reading>/<state>[@<variant>]` (value, saved value + file + jq, settings).
  `--part data` ran on cpu-run, tags A and B (A = B bit for bit): all 15 data readings repeat their saved values bit for bit except
  erank (last digit: 464.47291186564024 vs saved ...402, 352.7235898087761 vs ...762, 592.6058268916009 vs ...012;
  fp64 svdvals on another machine). `--part gpu` (`old.sbatch`, NOT yet run) does the 22 model readings.
- `compare.py`: the reproduction table (`results/reproduction.json`) under the plan's pass rule; new.json pending.
- Move 1: `src/commute/world.py`, `windows.py`, `items.py`; `old_tasks.py` and `near_tasks.py` no longer load
  experiments. Checks: `move1_check.py` -> `results/move1_world.json`: 239 checks, 0 misses (W0; 58 window sets incl.
  12 panel D and 13 amend3 files against their recorded sha256s; 45 p256 dW sha256s + 69 E136 states load; 66
  kstar_needed p256/rl row counts). The live-W0 check is old.py gpu `_meta` (also asserted in `World.load`).

## For the next writer
- Run the GPU part at the harness commit 62e4ca8d (before move 1 touched src/), tags A then B.
- amend3 "full" windows are the uncut source file `windows_<map>.pt` (heldout_keys.split_task saves only trained/new).
- Reading names (plan "Rule for variants"): `share` (log-odds unless `@contrast`), `steering_pooled`,
  `kl_base_to_ft` (KL(W0||F), rl_67m `kl_from_base` = panel `kl_from_w0`), `kl_ft_to_base` (rule_retention.kl form;
  no saved value), `prior_geo` (P), `prior_log10_geo`, `prior_mean_lo`, `size_rel`, `summed_norm`, `erank`, `E_b_vpd`,
  `E_b_neurons`, `E_b_sae` (over K*, E263's form), `needed_set_size`, `adl_D`, `gain`. The `@b32` / `@b64` suffix is
  the batch; `@fresh` a fresh write pass (saved K); `@saved_P` the log10 of a copied P.
- `new.json` must use the same keys; compare.py's TOL holds the filed tolerances per reading.
- Rows (windows.py): toks is exactly what the model and Gamma see; a LAST target column is NOT in toks (it is `tgt`).
  To feed an old reader expecting the target column, `torch.cat([toks, tgt[:, None]], 1)` with ci_len = toks width.
- items.py: `load_dw` refuses parameter-name keys (`.weight`); E136 states have no recorded sha256 (move1_world.json
  now records their file sha256s); `scale`/`pair` are there for E232/E237 scaled items, no family fills them yet.
- `regating.load_windows` was NOT absorbed (A2 asks for it); `qwen_library.template_pool` still loads
  experiments/qwen_updates/job.py: moving E45's Qwen `Job` into src is a Qwen-line change the plan leaves untouched,
  and its only caller (E169 run.py) is finished. Left for the lead to decide.
- Time: data part 114 s on cpu-run (1 CPU). GPU part estimate in old.sbatch; run A's `_seconds` per group is the
  measured cost, use it for B's limit.

## 2a (phase 2, moves 2-4: size, erank, prior, KL, re-gating share)
- New modules: `src/commute/prior.py`, `kl.py`, `share.py`; `qwen_reading.kl_share` (an alias of `shares`, the Qwen
  KL ratio, Qwen otherwise untouched). Kept as they are: `update_spectrum.state` (erank), `realistic.update_size`
  (size_rel), `elicit.summed_norm` (scorecard size; no alias needed), `local_faith.route_split` (four-cell form).
- Reading -> function, and which old variant is which argument:
  - `prior_geo(net, rows, batch)` = E133 `forward_prior` (`keydep.geo_rate` of fp32 log P; batch = all rows for E133).
    `prior_log10_geo(p)` = the scorecard axis, takes a computed or saved P (E133 `point`).
    `prior_mean_lo(net, rows, unit, batch)` = E211/E272 "prior" (W0's mean of `unit`, `steering.logodds_at` or
    `contrast_at(ab)`; the panel's batch is fair_steering's 64).
  - `kl_base_to_ft(net, toks, dW, batch)` = rl_67m `kl_from_base` = `panel.kl_from_w0` (procedural_rl batch 8 on
    64 cached val rows; panel batch 8). `kl_ft_to_base(net, toks, dW, batch, positions)` = `rule_retention.kl` form
    (fp64, per-row mean over `positions`, then over rows); old.py's reading is `positions=slice(None)`.
  - `share_cells(model, W0, dW, rows, units, batch)` + `share(cells, unit, gain, floor)`:
    E136 `.trained` = inventory batch (c3 64, e53 32), gain `cells_gain` (bf16), `Floor(1e-6, two_sided=True)`;
    E211 / E272 = batch 64, `cells_gain`, `Floor(MIN_GAIN, two_sided=False)`; E140 = batch 64, units lo and
    `contrast_at(ab)`, `Floor(gain_floor=1, two_sided=False)`. E136's untrained shares divide by the fp32 gain:
    `untrained_share(model, W0, dW, rows, decomps, k_m, unit, batch, floor)` wraps `do_nothing_cells` (unit "lo"
    is its own device-summed log-odds; `RR.Reader.untrained` = batch 32, one-sided 1; E136 = inventory batch, 1e-6).
- Expected differences from the old code on GPU: none for E211, E140 and the panel (same forwards, same means).
  E136's `route_split` takes a difference of means summed per batch in Python, share() a mean of differences:
  ~1e-16 relative, inside compare.py's 1e-12.
- `new_a.py --part data` (cpu-run, 1 CPU, 259 s): all 9 data keys equal old_A bit for bit (erank x3, size_rel x3,
  summed_norm, prior_log10_geo@saved_P x2). size_rel/e62/copy_s0 is `update_size` where the old read size_matched's
  sqrt(sum)/W0-norm form: equal bit for bit here.
- `new_a.py --part gpu --dry` (cpu-run): every GPU input builds (rows c3/cipher 75x41, e53/lowercase 500x127,
  case_label/real 256x41, e46/ident32 97x41; five 24-matrix updates, case_real_s0 sha256-checked; kl rows 64x512),
  host RSS 1.3 GB. The GPU part is `new_a.sbatch` (not submitted): run it after old A/B, same card (L40).
- `Rows` carries no label pair: new_a.py reads case label's `ab` from its task.pt. A `Rows.ab` field would remove that
  (windows.py is shared with 2b, so left).
- Not re-pointed (phase 2c/3): `panel.kl_from_w0`, rl_67m `kl_from_base`, `masked_reads.share_cells*`,
  `realistic_reads.share_of`/`ratio`, `keydep.geo_rate` callers.

## 2c (phase 2: Reader re-pointed, item builders, README, the test leak)
- `realistic_reads.Reader` / `share_of` / `ratio`: the share is `share.share_cells` (batch 64 = `masked_reads.BATCH`, as
  `share_cells_by` ran) + `share.share` / `numerator` / `cells_gain` with `FLOOR = share.Floor(MIN_GAIN, one-sided)`;
  the prior is `prior.prior_mean_lo(net, rows, unit, 64)`, computed once per task in `__post_init__` (W0's, the same
  forwards as the T0 cells it used to come from: batch 64, fp32, `capture` changes nothing); the steering vector is
  `fair_steering.fit_vectors` (split out of `steering_share` with `key_idx`; same ops as `vectors`' pooled mean), the
  shares `fair_steering.divide` under a `Rule("pooled", None, strengths, MIN_GAIN, 64, fp64)`. `Read.rows(kind, n)` gives
  a `windows.Rows`. Deleted: share_of's own numerator/gain/floor arithmetic, `ratio`'s floor, the T0-mean prior, the
  `vectors(...)["pooled"]` call with its zero_idx. Not GPU-run here (another agent holds the GPU part): every change is
  the same forward at the same batch, read in the same shapes, so the numbers should be bit for bit; the next GPU run of
  any Reader caller is the check.
- Left alone, same quantity but a different numeric path (comment in the code): (1) the steering selection grid: Reader
  reads each unit once over all n_select rows' logits, `steering_share`'s `unit_rows` per batch of 64, and a CUDA
  logsumexp over a different row count may round differently; if new_b's GPU part shows `steering_pooled/lower_s0_step128`
  equal to old_A bit for bit, the grid can move too. (2) the untrained read: `share.untrained_share` feeds only the LAST
  layout, a Read also NEXT (LM windows) and PAIR (E223), and Reader keeps W0's cells once per task and the numerators.
  Neuron reads and k_half have no shared function in the plan (untouched).
- `items.py`: `rescaled(M, layer, C, factor)` (E214 x<f>), `copy24_held()` (its D), `e229_read(task, read_seed, n_read)`
  and `e229_f(net, item)` (E229's D and F), `neuron_bounds(q, qs, sha256)` (E217 percentiles, E229's upper bounds), copied
  from kstar_levels, rule_retention, topk_neurons, rule_retention_pairs and topk_upper run.py (unedited). `new_b.py`'s
  `fresh_inputs` replaces `fresh_worlds`: no `load_experiment`; settings from each finished experiment's predictions.json;
  the fresh items run on a second `World.load` (the old code loaded its own worlds). CPU check (one-off script, deleted):
  copy24_s0's D (512x8), x2's D and the p97 bounds (4x3072) equal the old builders' bit for bit. x2's F (a deepcopy and a
  column scale) and copy24_s0's F (`old_tasks.old_f`) are the old calls; GPU-run only with new_b's GPU part.
- `windows.Rows.ab`: the label pair (case label's from its task.pt, None elsewhere); new_a no longer opens task.pt.
- Data parts rerun on cpu-run (1 CPU): new_a's 9 keys and new_b's 6 E_b keys equal the committed new_a/new_b values bit for
  bit. Against old_A: equal except erank and size_rel of lower_s0_step128 (...009 vs ...012, ...677 vs ...679, ~4e-16
  relative), as committed in 2a/2b: old_A's lower values came from the GPU-host run; at 2 CPUs erank of e62/copy_s0 and
  sequence_rl_s0 move in the last digit too (fp64 svdvals vary with the thread count; compare.py's REL 1e-12 covers it).
  Result JSONs left as committed (the GPU agent writes them). `--part gpu --dry` builds for both.
- `src/commute/README.md`: "Readings: one function each" (14 lines). `jobs.load_experiment`: "not for readings" and a
  fallback to archive/experiments/<folder> when experiments/<folder> does not exist (one line).
- Test leak: `src/commute/crosscoder.py` (E203) imports dictionary_learning, whose package imports nnsight 0.8.2, which
  mounts `.save` on `object` process-wide (CONFIG.APP.PYMOUNT); every class then has an attribute `save`, so
  `library_states.State` (P3.8, field `save`) saw a default and failed at import in a full run (tests/test_crosscoder.py
  collects first). Fix: crosscoder.py unmounts it right after the import (nothing here uses nnsight's tracing).
- src/commute line count (`wc -l src/commute/*.py | tail -1`): 40399 before 2c, 40513 after (items +56, realistic_reads +20 (mostly
  comments on what stays), fair_steering +12, crosscoder +8, windows +2, share +5, jobs +3). The deletions were small:
  the private copies were a few lines each; the bulk of the duplication (finished run.py builders) is not deleted until phase 3.
- Full suite after the fix (cpu-run, 4 CPUs): collection passes; 7800 passed, 5 failed, 1 skipped. The 5 are outside
  2c: three test_sbatch_stages cases need this worktree's results/large linked (fleet/link-data) or a results/ folder,
  and test_qwen_rl / test_qwen_d1_teeth_amend fail on the Qwen model's attributes (`Qwen3Model has no attribute lm`).

## 2c follow-up (status at lead restart, 2026-11-25)
- Done: (1) Reader's steering selection is `fair_steering.selection` (58ebd417); (2) GPU new_a/new_b results and logs
  67371/67372 on the branch, compare.py's pre-existing-saved-gap rule, reproduction.json 35/35 pass (f7332e46);
  (3) p2-79-archive merged (5419e5fc); tests moved to archive/tests: test_case_patch, test_claim_tests,
  test_crosscoder_replicate, test_gamma (loads archived gamma_refit run.py at import), test_lora_sgd_score,
  test_neuron_jobs_rules, test_qwen_d1_teeth, test_qwen_d1_teeth_amend, test_qwen_masked_read, test_qwen_tofu_adversary,
  test_teacher_spread_amend1. Kept (also test live src): test_cancel (commute.cancel), test_carrying_jobs
  (commute.carrying_jobs); their paths now point at archive/experiments. pytest testpaths was already ["tests"].
  load_experiment's fallback now tests for .py files (archived folders left untracked results/large dirs; those were
  removed in this worktree). (4) link-data run. The 2 Qwen failures fail the same on 62e4ca8d (e309-old): pre-existing
  (test_qwen_rl, E69 / qwen_rl; test_qwen_d1_teeth_amend, E85 / qwen_sft.py:53, now archived).
- Tests: before the merge 7800 passed, 5 failed, 1 skipped (results/large not linked). After: cpu-run job 67411 was
  still running at the restart (log ~/commute-data/scratch/refactor-lead/2c/test_after.log; 3 F seen by 96%).
- Lint: ruff + pyright clean on every touched file. Lines: src/commute 40502; experiments/**/*.py 185160.
- Left: read job 67411's result and fix any failure caused by the archive move.

## Review fixes (p2-79-fix, 2026-11-25)
- Fix 1, archive rule: an archived run.py builds paths from its own location, so a live caller that load_experiment's it
  reads paths that do not exist. Rule now: any folder reached from a live file (experiments/, src/, tests/, cluster/)
  through load_experiment, transitively, stays live. Recomputed by ~/commute-data/scratch/refactor-lead/p3fix/closure3.py
  (scan.py there reads the archived layout; an explicit archive/ path keeps nothing live). Back to experiments/:
  carrying_jobs, qwen_masked_read, rule_retention, sae_train, soft_intervention (unedited), their 5 sbatch files and
  test_qwen_masked_read; test_carrying_jobs's path is main's again. Archive: 57 folders. Their results/large symlinks
  copied from the main clone. live_check.py there: 0 rows from a live file reach an archived folder.
- Fix 2, one definition: panel.kl_from_w0 calls kl.kl_base_to_ft; masked_reads.share uses share.Floor; masked_reads
  .share_cells calls share.share_cells (per-unit readouts of masked_reads.units); share_cells_by deleted; route_cells_by
  kept (its KL cells need the same forwards' logits); fair_steering.means is the one pooled/per-key mean, divide uses
  share.Floor. Fix 3: Item.scale/.pair, world.weights, share.untrained_share deleted; the E214/E229 item builders moved
  into new_b.py. Reader's prior forward kept (comment in __post_init__).
- Fix 4: compare.py holds a reading with no filed tolerance to the saved value within REL, so every saved gap is labelled.
- Fix 5, stage "scans" (prepared, not run): new_c.py / new_c.sbatch (new code) and old_c.py / old_c.sbatch (old code:
  copy both into the e309-old worktree and submit there, tags A and B). Keys: steering_perkey, E_b_neurons@fresh_no_leftover,
  scan_* and needed_set_size for x2@vpd_* and e58/copy_s0_scaled@sae_*, E_b_sae@fresh / @fresh_no_leftover. compare.py
  merges old_c_<tag> into old_<tag> and new_c into new.
