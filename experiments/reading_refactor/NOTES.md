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
