# Plan: one shared definition per reading (the reading refactor, 2026-11-25)

We read the code behind every reading the paper uses: re-gating share, E_b, needed set, KL, steering, prior, effective
rank, size, and the world, windows and items they rest on. Each has a canonical function in `src/commute/`, but newer
experiments reach it through other experiments' run.py files, and several readings that carry one name are computed
in two or more different ways. The refactor gives each reading one function in `src/commute/`, checks that function
against saved numbers before any caller switches, and changes no paper number. Six of the variants change the paper's
own figures; those go to writeup-lead as findings, and the refactor does not fix them.

Owner: compare-lead (one Opus writer, one fresh Opus reviewer who wrote none of it). Slot: whip-lead's 11-25 quiet
window (`~/commute-data/agents/whip-lead/schedule.md`, 2026-11-24 12:20). Per-reading detail: the appendix,
`reports/plans/2026-11-25_plan_reading-refactor_appendix.md` (A1-A11). What the paper reads is traced from
`writeup-lead/paper/paper/SOURCES.md` and `figures/src/*.py`.

## The request, in the Terms' words
The user (11-24): "how do we not have standardised code for computing regating share at this point? ... a major
refactor would be preferred". In the Terms' words: every reading that a paper figure or table rests on gets a single
function in `src/commute/`. Later experiments call that function, never another experiment's run.py. Saved results,
finished run.py files and the paper's CSVs stay as they are.

## What was found (the important part)
There are 303 `.py` files in 154 experiment folders that call `load_experiment`, at 541 call sites. Of the 127 run.py
files among them, the most-loaded targets are:
- `share_untrained/windows` (13 call sites)
- `realistic_tasks/run2` (12)
- `kstar_needed/run` (12)
- `kstar_vpd_wording_a/run` (11)
- `holes_fill/run` (9)
- `rl_67m/run_stage1` (7)

Three `src/commute` modules also load experiments: `old_tasks.py:211`, `near_tasks.py:124` and `qwen_library.py:77`.
`KN.world()` pulls in about 14 experiments' run.py files transitively.

**Variants that change a paper figure.** Each line names the two forms and where they meet. The full list of variants
is in the appendix.
1. **Fig `definition`, SAE row.** It reads E_b over K*, the old KL mean-ablation search (`kstar_vpd.search`:407,
   `sae_eb/run.py:76` `main_in_K_polished`). The VPD and top-neuron rows read E_b over the swap-test needed set K
   (`needed.needed_set`:190). The SAE row also reads the size-scaled states `e58/<t>_s<s>_scaled`, while the VPD and
   neuron trained rows read the unscaled `full/<t>_s<s>`. One of those VPD trained rows, `full/copy_s0`, is a copy of
   E275's `e58-copy_s0_scaled` (`ia3_control/run.py:385` `PRIOR_VPD`).
2. **Scorecard `size` (known_main, known_appendix).** Most cells are `elicit.summed_norm`:58, a sum of absolute
   Frobenius norms. Examples: e62 reuse[0] reads 4.62 here, against 0.00715 relative size in `size_matched/m67.json`.
   e53 uses `update_svd` `norm2` (unrooted), and e46 uses a value whose unit could not be traced. The p270 rescore
   uses `realistic.update_size`:191 instead.
3. **Scorecard `share` mixes units.** It takes 11 cells in log-odds (E136 `.trained`, E147 `.e136.trained`) and 2 in
   two-label contrast (`real_vs_twin` `.share.contrast`, e123 and e125).
4. **Scorecard `steering`.** It keeps whichever of pooled and per-key gives the wider gap, per pair. Its sources use
   three selection rules: E135 (fit and score halves, key at half-1, no floor, fp32), E143 (the first n_select training
   windows, key at the read position, floor 1, double) and E140 (contrast unit).
5. **`prior` means two quantities.** On the scorecard it is the log10 geometric-mean P (`keydep.geo_rate`:34, several
   values copied from older JSON). In `realistic_reads`:294, `convergence_panel` (Fig `realistic_mirror`) and
   `procedural_rl` it is W0's mean natural log-odds.
6. **Re-gating share batch.** `route_split` is batch-dependent in bf16. E136 read the e53 tasks at batch 32;
   `masked_reads` hard-codes 64. So e53/lowercase_s0 reads 0.75188 (E136), 0.75376 (E211) and 0.75129 (E228). All are
   within the 0.01 tolerances, but the difference is not zero.

**Variants that do not reach a figure, but would split under one name:**
- `kl_from_base` (KL(W0‖F), all positions, 64 val rows) against `kl_fm` (KL(F‖M), read positions, fp64), about 10×
  larger. `kl_control` part A uses the first and part B the second.
- The Qwen "re-gating share" is a KL ratio `(K00-K01)/(K00-K11)` (`qwen_reading.shares`:52), a different quantity from
  the 67M r. On c3_cipher_s0 the 67M r reads 0.236 and the KL share 0.908.
- E136's untrained shares divide by the fp32 gain, while `.trained` divides by the bf16 gain.
- `RR.Reader.untrained` always uses batch 32 with floor 1, while E136 used the inventory's batch with a 1e-6 floor.
- Gain floors: two-sided 1e-6 (`route_split`) against one-sided 1 (`RR.ratio`, `MR.share`).
- The same E187 state is read on 512 `old_tasks` windows in the kstar family and on the task's own held-out rows in
  the E136/E143 family.
- The lowercase task has four row sets.
- `e58/<s>_scaled` (E237's common c) against `e58/<s>_scaled_<pair>` (E232's pair c).
- ADL column D: 8192 Pile rows in log-odds (`adl/run.py`) against 4096 rows in the scored unit (`convergence_panel`:184).

**Not reproducible from code.** `fig2_standard/results/top_with_remainder_rows.csv`, an input of Fig `definition`, was
committed as data, and no script in the repo writes it. E298 (`fig2_standard`), the unified E_b driver, has never run.

## Rule for variants
- **One definition, explicit arguments.** Where variants differ only by a setting (unit, batch, rows, floor), the
  setting becomes a required argument with no default. Every caller then writes down which form it reads.
- **Different quantities get different names.** Examples: `share` (log-odds, the Terms' r) and `kl_share` (Qwen);
  `kl_base_to_ft` and `kl_ft_to_base`; `size_rel` and `summed_norm`; `needed_set` (K) and `kstar_search` (K*).
- **The refactor chooses no paper form.** Which form a figure should use (findings 1-5) is a writeup-lead or user
  decision, sent as one research-log card. Until it is made, the paper's CSVs stay as they are.

## Proposed homes in `src/commute/`
Where `src/commute/` already has a good function, it is kept. New code is only what is missing. Signatures are given
in the appendix.

| Reading | Home |
|---|---|
| World | `world.py` (new): `World.load(device)` gives net, W0 (checkpoint, checked against live), uv, decomposition and `bases()`. It replaces the chain E214→E221→E226→E242→E257→E268. |
| Windows | `windows.py` (new): `read_rows(task, kind)` for kind in {held, train, D, fit, pile}, sha256-checked. It absorbs `share_untrained/windows.py` and `regating.load_windows`. |
| Items | `items.py` (new): `Item(name, path, sha256, scale)` with one naming scheme `<family>/<task>_s<seed>[_scaled[_<pair>]]`, `load_dw(item)` keyed by module, and `inventory(name)`. |
| Re-gating share | `share.py` (new): `share_cells(model, W0, dW, rows, pos, units, batch)` (from `masked_reads.route_cells_by`) and `share(cells, unit, floor)`. `local_faith.route_split` stays as the four-cell form; `untrained_share` wraps `untrained.do_nothing_cells`. |
| E_b | `stand_in.explained_share` (kept). One `write_sums(net, sites, rows)` works for VPD, neuron and SAE sites (`stand_in.VPDSite`, `needed_neurons.HeadSite`/`MLPSite`, `sae_feat.SAESite`), with `eb(sums, K, leftovers: bool)`. |
| Needed set | `needed.needed_set` (kept), plus one `scan` over any site list. The neuron and SAE scan copies go. `kstar_vpd.search` is kept under its own name as K*. |
| KL | `kl.py` (new): `kl_base_to_ft(net, rows, dW, batch)` (= `panel.kl_from_w0`:285) and `kl_ft_to_base(...)` (= `rule_retention.kl` form). |
| Steering | `fair_steering` (kept), plus `steering_share(net, read, dW, form, select_rows, unit, floor)`, lifted from `realistic_reads.Reader.steering`:263. |
| Prior | `prior.py` (new): `prior_log10_geo(net, read)` (scorecard) and `prior_mean_lo(net, read, unit)` (realistic). |
| Effective rank, size | `update_spectrum.state` and `realistic.update_size` (kept). `elicit.summed_norm` is kept, renamed in place by an alias only. |
| Reader | `realistic_reads.Reader` re-pointed at the above; the single entry point for new experiments. |

## Reproduction check and its floor (every reading, before any caller switches)
- **Data-only readings** (E_b from saved sums, size, erank, prior from saved P): run on `cpu-run`. The new function
  must equal the saved value bit for bit, or within 1e-12 relative where the old code summed in a different order.
  E268's own transcription tolerance is 1e-11 (`kstar_needed/results/predictions.json`).
- **Model readings** (share, steering, KL, prior forward, K scan, fresh E_b) run on the cluster's debug QoS, on the
  same card type as the saved run where it is known.
  - **Floor:** the old code, run twice on one state, gives the spread.
  - **Pass:** if the old code is bit-identical with itself, the new one must be bit-identical to it. Otherwise the new
    one must lie within that spread, and stay under the saved run's own filed tolerance (E136 `reproduce_tol` 0.01,
    E211 steering 0.001).
  - The saved-run value is checked as well. Differences from batch or card are written down, never absorbed.
- **States to check** (appendix A1-A11, one per variant at least):
  - share: E136 c3/cipher_s0, e53/lowercase_s0 at batch 32 and 64, real_vs_twin case_real_s0 contrast
  - E_b: x2 and new_few (E268), e58-copy_s0 (E275), x2 and copy24_s0 (E277), e58/copy_s0_scaled (E263)
  - steering: e62 copy_s0 (E135), e46/ident32_s0 (E143)
  - KL: sequence_rl_s0
  - convergence_panel lower_s0_step128: every reading at once
- **Zero and identity controls** are kept as-is: a zero update reads exactly 0, the identity scan gives no flips, and
  the do-nothing E_b is 1.

## Order of moves (dependencies first)
1. **World, windows, items.** No reading changes. Check: same W0 bytes as `checkpoint_weights`; the window sha256s
   match `convergence_panel/results/wide/eb/windows_manifest.json`, `rb_states/results/manifest_p256.json` and
   `open_pair amend3 .maps.*.sha256`; the p256 dW sha256s match (45 entries); the row counts of the kstar_needed items
   match. Remove the three `src → experiments` loads.
2. **Size, erank, prior.** CPU-only checks.
3. **KL** (both directions named).
4. **Re-gating share and its untrained forms.**
5. **Steering.**
6. **Needed set K, one scan.** Then E_b at VPD, neurons and SAE, through one `write_sums`.
7. **ADL column D**, and `Reader` re-pointed at all of the above.
8. **Switch.** `src/commute/README.md` gets one line per reading naming its function. `load_experiment` gets a guard
   comment ("not for readings"). The first new experiment after the switch uses only `Reader` / `world`.

Each move is one commit with its reproduction record in `experiments/reading_refactor/results/<reading>.json`. One
review covers the whole series before step 8.

## What stays untouched
- Every finished run.py and its results.
- The paper repo and its `figures/data/*.csv`.
- Runs already launched (E300 reuse_dial and the other 11-24 night runs) carry on, on the current code.
- Qwen readings beyond naming `kl_share` separately.
- The frozen definition (`theory/2026-11-22_definition_frozen.md`).
- `decisions.md`.

## Open before starting
- **E-code.** The debug-QoS reproduction jobs need an E-code against an open cell (`fleet/job-gate`). Ask whip-lead
  for the cell.
- **Card.** Findings 1-5 go to writeup-lead as one research-log card. The refactor ships with no figure changed either
  way.

## Not covered
- The origins of appendix figures `qwen_step75_readings`, `qwen_word_pairs_S`, `kstar_size_matched_r_gap`,
  `kstar_vpd_r_known`, `key_grid_columns`, `old_tasks_difficulty` and `crosscoder_size_ladder` were not traced. No
  builder for them was found in any clone.
- The k-head and (2,4) share of Fig `copy_heads_off` (`prior_mech_2x2/khead.py`, `train.py`, via E253 `cued_copy`)
  was not compared against the share.
- The neuron reading at k64/k-matched (`masked_reads.topk_cells_by`), k_half (`neuron_r`) and Qwen's E_b
  (`qwen_eb.moment_sums`) were listed but not mapped variant by variant.
- rho (E208, `rho.py:248/329`), a different reading, was left out of scope.
- The 11-24 reuse_dial (E300) results did not exist yet, so none could be checked.
- Nothing was run. Line numbers are as of `af569450`.
