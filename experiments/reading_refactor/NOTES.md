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
