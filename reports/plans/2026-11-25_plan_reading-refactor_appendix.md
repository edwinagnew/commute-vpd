# Appendix to the reading refactor plan (2026-11-25)

The main plan is `reports/plans/2026-11-25_plan_reading-refactor.md`. Paths are repo-relative, line numbers as of
`af569450`, and every jq path is on `experiments/<exp>/results/...`. "Fig" means the paper (writeup-lead clone,
`paper/paper/figures/src/`). Each section A1-A11 gives the same five things: (1) where the reading is defined, (2) its
variants, (3) saved numbers to reproduce, (4) its callers and the figures it feeds, (5) its proposed home and check.

## A1. World
1. **Defined:**
   - `vpd.py:44` `load_component_model`
   - `vpd.py:62` `checkpoint_weights` (W0 and uv from the file, `.float()`)
   - `vpd.py:130` `target_weights` (live)
   - `steering.py:32` `Net`
   - The world chain: `kstar_levels/run.py:129` → `kstar_pairs/run.py:130` → `kstar_vpd/run.py:96` (`w0_check`) →
     `kstar_vpd_wording_a/run.py:136` → `rb_states/run.py:104` → `kstar_needed/run.py:89` `world()` → `reuse_dial/run.py:134`
   - Others: `adl/run.py:150`, `prior_matched/world.py:56`, `refit_96key/run.py:117`, `size_matched/run.py:105`,
     `procedural_rl/run.py:119`
2. **Variants:**
   - An E242/E257 world holds up to four ComponentModels (kl.net, sm.net, kg.net, kg.share_net), and share_net leaves
     requires_grad on.
   - There are two basis types: `kstar_vpd.bases` (fp64 split SVD) and `vpd_gains.Basis.of` (reuse_dial).
   - dW is keyed by parameter name in `OT.load_dw` and by module elsewhere.
3. **Saved:** the `w0_check` in E226 stage0; `jq .repeat_gain adl/results/stage0.json` = 16.450539620606214 (e58/copy_s0
   gain; also 16.450539847353955 in `refit_96key stage1 .states["e58/copy_s0"]`).
4. **Callers:** every 67M reading; reached through `KN.world()` by reuse_dial, convergence_panel, claim2_needed and fig2_standard.
5. **Home and check:**
   - Home: `world.py` `World.load(device) -> World(net, W0, uv, model, bases())`, a single ComponentModel with
     requires_grad off.
   - Check: W0 equals `checkpoint_weights` bit for bit; the gain on e58/copy_s0 equals 16.450539847353955 (fp32, same card).

## A2. Windows (read rows)
1. **Defined:**
   - `share_untrained/windows.py:41` `load_read` (four formats, from E136 `inventory.json`)
   - `regating.py:56` `load_windows` and `:189` `load_kv_windows`, with copies at `reading_a_coverage/run.py:222` and
     `task_ladder/check_pgd.py:70`
   - `data.py:25` `pile_rows` and `:49` `cached_rows`
   - `old_tasks.py:231` `task_windows` (512 mixed rows, seed 139700 + crc32)
   - `realistic_tasks/run2.py:130,138,145`
   - `rb_states/run.py:139` (rl, 200 rows) and `:151` (p256, 256x64)
   - `convergence_panel/tasks.py:150` and `eb.py:120`
   - `holes_fill/run.py:97,105`
2. **Variants:**
   - The E187 states are read on 512 `old_tasks` windows (the kstar family) and on held-out rows (E136/E143 family).
   - The lowercase task has four row sets: 500 E53 scored windows; 512 old_tasks windows (reuse_dial's D); 200x64 EB
     windows; and `pile_rows("train",0,4096)[:256]`.
   - c3 and e58 rank1/cipher windows come from `open_pair amend1` and from `amend3`.
   - Procedural tasks: `CueRead` on held[:256] and saved `p256_windows_<task>.pt`; the rl read is 200 rows.
   - Pile rows come from `pile_rows` (adl, holes_fill) and from `cached_rows` (procedural_rl, reuse_dial).
3. **Saved:**
   - `jq -c 'to_entries[]|{k:.key,shape:.value.shape,sha256:.value.sha256}' convergence_panel/results/wide/eb/windows_manifest.json`
   - `jq '.maps.rank1.sha256.trained' open_pair/results/amend3.json`
   - `jq '{n_windows,window_len,n_read_positions}' kstar_needed/results/items/x2.json` (75/41/41)
4. **Callers:** 13 loads of `share_untrained/windows`; `near_tasks.py:126` also loads it from src.
5. **Home and check:**
   - Home: `windows.py` `read_rows(task: str, kind: Literal["held","train","D","fit","pile"]) -> Rows(toks, pos, tgt, sha256)`.
     Each (task, kind) maps to exactly one file. The four lowercase sets become four named kinds; none is dropped.
   - Check: sha256 of each built tensor equals the saved manifest, or the sha of the old builder's output on cpu-run.

## A3. Items (states and their dW files)
1. **Defined:**
   - `procedural_rl/run.py:107` `build_items`, with a different `build_items` at `rl_67m/run_newtasks.py:101`
   - `key_grid/run.py:94`, `key_grid_baselines/run.py:81`
   - `adl/run.py:122,131`
   - `kstar_vpd_wording_a/run.py:115,120,126`
   - `rb_states/run.py:100,282`
   - `kstar_needed/run.py:53,101`
   - `size_matched/run.py:89`, `refit_96key/run.py:90,108`
   - `prior_matched/world.py:63`
2. **Variants:**
   - Naming:
     - `e46/ident32_s0` (E136) and `ident32_s0` (E98)
     - `e46_copy_random/e46/ident32_s0` (adl)
     - `random_e58/copy_s0` (key_grid) and `random/<state>` (prior_matched)
     - `rl/stage2b/repeat_rl_s0`, which has no sha256
   - Scaling: `e58/<s>_scaled` (E237's c) and `e58/<s>_scaled_<pair>` (E232's c); for example e58/cipher_s0 has size
     0.0073095 under one and 0.0079444 under the other. rb_states writes the item field in both forms.
   - `kstar_pairs/run.py:118` drops the held-out-keys file: e58/cipher_s0 KL is 5.2393 with that file dropped and 4.6463 with both files.
3. **Saved:**
   - `jq -c '.[]|{item,sha256}' rb_states/results/manifest_p256.json` (45 entries)
   - `jq .c key_grid/results/stage0.json` (E237's c)
   - `jq .pairs size_matched/results/m67.json` (c, c_e98, sizes)
   - E136 `inventory.json .states[].norm` (12 entries, sha256 only)
4. **Callers:** every reading.
5. **Home and check:**
   - Home: `items.py`:
     - `Item(name, path, sha256, scale: float | None, pair: str | None)`
     - `load_dw(item) -> DW` (keyed by module, sha256 checked)
     - `inventory(family) -> list[Item]`
     - Old names are kept as an alias table, so that every saved key still resolves.
   - Check: every inventory resolves to the same files and sha256s as its old builder.

## A4. Re-gating share (r) and its untrained forms
1. **Defined:**
   - Group A: `local_faith.py:199` `route_cells`, `:226` `masked_lo` and `:256` `route_split` (ratio of means; nan if
     |gain| < 1e-6; batch is an argument). Wrappers: `keydep.py:66`, `pcfg_vpd.py:88` (minGPT), `share_untrained/run.py:99`
     `Job.trained`, `regating.py:127` `measure` (floor 1e-9).
   - Group B: `masked_reads.py:65/71/80` (batch fixed at 64; units lo, lp or contrast; no ci_len) and `share`:50
     (one-sided floor); `realistic_reads.py:178` `share_of` (floor MIN_GAIN = 1); `convergence_panel/run.py:200`
     `readings`, where a random update divides by its paired state's gain.
   - Untrained forms: `untrained.py:126` `do_nothing_cells` (fp32 hooks, top-k_m gates).
   - Qwen: `qwen_reading.py:52` `shares`, the KL form.
   - Other: `topk_gate.py:72/95`.
2. **Variants:**
   - E136 batch 32 against batch 64 on e53: 0.75188 against 0.75376 (E211) and 0.75129 (E228).
   - `.trained` uses the bf16 gain and `.svd`/`.rotated` the fp32 gain: 12.85136 against 12.85540 on c3/cipher_s0.
   - E211's share divides by the bf16 gain, while its steering and neuron readings divide by the fp32 gain (1.62730
     against 1.62864).
   - Floors: 1e-6 two-sided, 1 one-sided, 1e-9, or None.
   - Units: the scorecard mixes log-odds and contrast.
   - The Qwen KL share is a different quantity: 0.908 against r = 0.236 on c3_cipher_s0.
3. **Saved:**
   - `jq '.states["c3/cipher_s0"]|{trained,svd,rotated,gain,gain_fp32}' share_untrained/results/stage1.json` →
     0.23632404751112837, 0.0501854…, 0.1027946…, 12.851359…, 12.855399…
   - `jq '.states["e58/rank1_s0"]["e136"].trained' refit_96key/results/stage1.json` → 0.8473278908255046
   - `jq '.states.case_real_s0.share' real_vs_twin/results/stage2.json` → lo 0.11284478195336352, contrast 0.10744470819037578
   - `jq '.["reproduce e53/lowercase_s0"].share' realistic_tasks/results/run2_ctl.json` → got 0.75376, saved 0.75188
   - `jq '.states["s0_0.0175"].share' realistic_tasks/results/run2_T1.json` → lo 0.769820569656078
   - convergence_panel `wide/read/conll.json` state `conll_s0_step8.pt` → share 0.793452094847827
   - Existing tolerance: E136 `reproduce_tol` 0.01; observed deviations 0 on 9 pairs and 1.366e-4 on class_map_s0
4. **Callers:**
   - Fig `known_main` and `known_appendix`, through `method_table/results/final.json` method share: 7 cells from E136,
     4 from E147, 2 from real_vs_twin.
   - Fig `realistic_mirror` (convergence_panel).
   - Fig `paper_fig5_scale` (method_table, plus Qwen `qwen_library_share/results/run2/*.json .regating`, KL form).
   - Callers through `R2.reader`: procedural_rl, reuse_dial, size_matched, held_model_size_matched.
5. **Home and check:**
   - Home: `share.py`:
     - `share_cells(model, W0, dW, rows: Rows, units: UnitsOf, batch: int) -> Cells`
     - `share(cells, unit: str, floor: float) -> float`
     - `untrained_share(model, rows, dW, decomps, k_m, batch, floor)`
     - `kl_share` for Qwen, kept in `qwen_reading` under that name
     - `route_split` stays as the four-cell diagnostic. `regating.py` keeps only E18's recovery patterns.
   - Check (debug QoS, L40 as the saved runs): the states listed in section 3, with the old and new code each run
     twice on each, and with e53 read at both batch 32 and batch 64.

## A5. E_b (the definition's kept share), at VPD, neurons and SAE
1. **Defined:**
   - `stand_in.py:145` `explained`, `:154` `explained_share` and `:125` `write_change` (fp64 sums of num, den and dd)
   - Sites: `stand_in.py:54/96` (VPD), `needed_neurons.py:182/213` (head and MLP), `sae_feat.py:48` (SAE), `qwen_eb.py:97` (Qwen, no K)
   - Duplicates in numpy: `option_b_rescore/score.py:54`, `why_fail/run.py:51,156`, `sae_eb/score.py:43`, `kstar_uncapped/score.py:37`
   - Forms that drop the remainder: `needed_neurons/run.py remainder_columns`, `fig2_standard/eb.py:44`, `ia3_control/run.py:289`
2. **Variants:**
   - SAE over K* (E263) against VPD and neurons over K.
   - SAE on `_scaled` states against VPD and neurons on `full/` states; E290 `full/copy_s0` is copied from E275.
   - The neuron field moved from `E_b_K_no_rem` to `E_b_K` (x2: 1 against 0.99977).
   - Sums come either from saved E247/E257 arrays (E268, E275) or from a fresh pass (E277, E290).
   - Read positions: "all" for E268 x2 and the last position elsewhere.
3. **Saved:**
   - `jq .E_b_K kstar_needed/results/items/x2.json` = 0.7613907014282402; `new_few.json` = 0.01269833004445926
   - `jq .E_b_K claim2_needed/results/items/e58-copy_s0.json` = 0.033852342972939126
   - `jq '.items.x2.E_b_K, .items.copy24_s0.E_b_K' needed_neurons/results/summary.json` = 0.9997651093639606, 0.02226…
   - `jq .E_b_K ia3_control/results/items/full__copy_s0.json` = 0.011681065992999671
   - `jq .E_b_K ia3_control/results/vpd/full__cipher_s0.json` = 0.03715988011781096
   - `jq '.rows[]|select(.item=="e58/copy_s0_scaled").E_b_sae' sae_eb/results/summary.json` = 0.047276001600795606
   - Existing tolerances: E268 transcription 1e-11; E257 `kept_floor` 3.32e-5; E298 `repro_rel_tol` null
4. **Callers:**
   - Fig `definition` (`build_definition_csv.py`, `rho_literal.csv` from `~/commute-data/scratch/writeup-lead-scripts/build_rho_literal_csv.py`)
   - E268, E275, E277, E290, E280 (`head_edit/run.py:166`), E272 (`convergence_panel/eb.py:77`), reuse_dial `:277`, E298
5. **Home and check:**
   - Home:
     - `eb.py`: `write_sums(net, sites, rows: Rows, cache_batch) -> Sums(num, den, dd)` and
       `eb(sums, sel: Tensor, leftovers: bool) -> float`
     - The site classes stay where they are, behind one `Site` protocol.
     - The numpy copies are deleted from future use; finished score.py files stay.
   - Check:
     - From saved sums (cpu-run): bit for bit, or within 1e-12 relative.
     - Fresh passes (debug): x2 and copy24_s0 at neurons, `full/cipher_s0` at VPD, and e58/copy_s0_scaled at SAE,
       each against the spread of two old-code runs.

## A6. Needed set K (and K*)
1. **Defined:**
   - `needed.py:32` `derangements`, `:124` `scan`, `:190` `needed_set` and `:203` `identity_scan`. Settings: P 4,
     m 64, seed 268, rows 1024, logit_rows 8192, cache_batch 32.
   - Near-copies of the scan: `needed_neurons.py:144` and `needed_sae.py:104` (which skips latents that do not move).
   - K*: `kstar_vpd.py:407` `search` (α 0.9, mean ablation, cap 2000); `kstar.py:398` (levels); E242 uses a relative bar.
   - Inline repeats: `kstar_needed/run.py:151,203` and `convergence_panel/eb.py`.
2. **Variants:**
   - K and K* both appear in Fig `definition`.
   - Identity control: 3 subcomponents per matrix plus the remainder (VPD), every variable (neurons), or the first 3
     active latents (SAE).
3. **Saved:**
   - `jq '.size_K, .size_Kstar' kstar_needed/results/items/x2.json` = 38935, 31767
   - `jq .items.x2.size_K needed_neurons/results/summary.json` = 12291
   - `jq '.stage0[0].identity.max_dev' kstar_needed/results/summary.json` = 5.15e-5
4. **Callers:** E268, E275, E277, E290, E280, E272, reuse_dial and E298.
5. **Home and check:**
   - Home: `needed.py`, with `scan(F, sites: list[Site], run, at, pairs, rows, ...)` over any site protocol and
     `needed_set` unchanged. `kstar_search` is kept, named as K*.
   - Check: the K mask is equal element by element (K is a set; flips are integers) on x2 (VPD) and copy24_s0
     (neurons), and identity gives no flips.

## A7. KL
1. **Defined:**
   - `rl_67m/run_stage1.py:94` `kl_from_base`, the same as `panel.py:285` `kl_from_w0` and `rl_arms.py:45` `kl_to_ref`
   - `vpd.py:223` `kl_per_position`
2. **Variants:**
   - `random_kl_share/run.py:158`: KL(F‖W0), positions up to pos, on the read rows.
   - `gating_vpd/run.py:105` kl_fm (through `rule_retention.kl:166`): KL(F‖M), fp64, read positions. It is stored as
     `kl_fm_saved` in rb_states and kstar_pairs.
   - `kl_control` part A uses the first and part B the second.
3. **Saved:**
   - `jq .kl_saved procedural_rl/results/read/sequence_rl_s0.json` = 0.274315
   - `jq .kl_fm_saved rb_states/results/items/c3-cipher_s0.json` = 3.827985
   - convergence_panel lower_s0_step128: kl 0.185764
4. **Callers:** procedural_rl, reuse_dial, convergence_panel (Fig `realistic_mirror` does not plot KL), claim2_needed and adl/kl_control.
5. **Home and check:**
   - Home: `kl.py` `kl_base_to_ft(net, rows, dW, batch) -> float` and `kl_ft_to_base(net, rows, dW, positions) -> float`.
   - Check: bit for bit on the same card and batch, with a floor from two old-code runs.

## A8. Steering
1. **Defined:**
   - `fair_steering.py:44` `vectors`, `:72` `pick`, `:80` `random_like` and `:93` `off_kl`
   - Older forms: `steering.py:156` `b3` and `:169`
   - Readers:
     - `steering_copy_lookup/run.py:203` (E135, its own fp32 copies)
     - `holes_fill/run.py:162` `Job.kind` (E143)
     - `real_vs_twin/run.py:291,299` (E140, contrast)
     - `realistic_reads.py:263` (pooled only)
     - E149 `prior_steering` (strength 1)
2. **Variants:** selection rows, read rows, key position, floor, precision and unit, as in the main plan's finding 4.
   On the scorecard the form with the wider gap is taken per pair; p270 tallies 4/3/2 as reported, 5/2/2 pooled and
   1/3/5 per-key.
3. **Saved:**
   - `jq '.states.copy_s0.steering.pooled.share_lo' steering_copy_lookup/results/stage1.json` = 0.5203478058670894 (= final.json e62 reuse[0])
   - `jq '.states["e46/ident32_s0"].steering.pooled.share' holes_fill/results/stage1.json` = 0.7472095889963015
   - `jq '.states.case_real_s0.steering.perkey.share.contrast' real_vs_twin/results/stage3.json` = 0.7612929401893089
   - `jq '.states["s0_0.0175"].steering.share.lo' realistic_tasks/results/run2_T1.json` = 0.7514111649119185
   - Existing tolerance: convergence_panel stage0 `e211_reread` 0.001 (got 0.7513304)
4. **Callers:** Fig `known_main` (final.json and `p270_rescore/results/rescore.json`), Fig `realistic_mirror`, and Fig `copy_heads_off` (cued_copy steering_pooled).
5. **Home and check:**
   - Home: `fair_steering.steering_share(net, read: Read, dW, form: Literal["pooled","perkey"], select: Rows, unit: str, floor: float, gen)`
   - Check: the four saved states above, each in its own form, on debug; floor from two old-code runs; tolerance 0.001.

## A9. Prior
1. **Defined:**
   - `keydep.py:34` `geo_rate`, through `prior_vs_share/run.py:84` `forward_prior` and `:126` `point` (log10)
   - `prior_lowercase/run.py:70` (the mean of per-window log10 P)
   - `realistic_reads.py:294` (W0's mean natural log-odds) and `convergence_panel/run.py:209`
2. **Variants:**
   - The scorecard prior is log10 geometric P, partly copied from older JSON (e62 from `induction_cue`, e54 from
     `keydependent_patch`, E58 from `open_pair`).
   - The realistic prior is the mean natural log-odds.
   - Two W0 paths read −6.755572450637818 and −6.755572590827942.
3. **Saved:**
   - `jq '.points[]|select(.name=="e62_copy").log10_prior' prior_vs_share/results/stage2.json` = −3.5010906459343945
   - `jq .tasks.lowercase.log10_prior prior_lowercase/results/stage1.json` = −2.9992791359204363
   - procedural_rl sequence_rl_s0 `.prior.lo` = −2.57911
4. **Callers:** Fig `known_main`, `realistic_mirror` and `copy_heads_off` (log10_prior).
5. **Home and check:**
   - Home: `prior.py` `prior_log10_geo(net, rows) -> float` and `prior_mean_lo(net, rows, unit) -> float`.
   - Check: within the existing 1e-4 relative tolerance (prior_vs_share stage0), with a floor from two runs.

## A10. Effective rank and update size
1. **Defined:**
   - erank: `update_spectrum.py:23/32/37` (exp entropy of s/Σs, ‖dW‖²-weighted over the 24 matrices, fp64)
   - size: `realistic.py:191` `update_size`, with the same quantity in `rl_arms.size:244`, `size_matched/run.py:129`
     and `held_model_size_matched sizes:168`
   - Other sizes: `elicit.py:58` `summed_norm` and `claim_tests/recipes.py:38` (max per matrix)
   - Qwen: `qwen_reading.effective_rank:357` (median, fp32)
2. **Variants:** the scorecard's sizes are mostly `summed_norm`, with `update_svd` norm2 for e53. rb_states
   `dw_rel_norm` and `size.whole` differ at 1e-13.
3. **Saved:**
   - `jq .pairs.e62_copy_lookup.reuse.copy_s0.erank update_svd/results/results.json` = 464.4729118656402
   - `jq .states.rank1_s0.erank refit_rank/results/cells.json` = 426.2320385572442
   - `jq '.pairs.e62_copy_lookup.sizes.reuse[0]' size_matched/results/m67.json` = 0.0071472003588455135
   - `final.json` e62 size reuse[0] = 4.624696578046103 (`summed_norm`)
4. **Callers:** Fig `known_main` and `known_appendix`, `realistic_mirror`, and `copy_heads_off` (erank).
5. **Home and check:**
   - Home: kept where they are; `summed_norm` stays, named as such.
   - Check: cpu-run on the saved dW files, bit for bit or within 1e-12 relative.

## A11. ADL lens column D
1. **Defined:**
   - `adl.py:49` `mean_diff` (k = 5) and `:96` `removed`
   - `adl/run.py:182` `gain`, `:186` `removed_mean` and `:195` `shares` (n_pile 8192); reused by reuse_dial `:261`
2. **Variants:** `convergence_panel/run.py:184` `adl_D` uses 4096 rows, the scored unit, the share's gain and MIN_GAIN.
3. **Saved:**
   - `jq '.cells[]|select(.pair=="e62_copy_lookup").reuse[0]' adl/results/amend1/column.json` = 0.0035393758777613267
   - convergence_panel lower_s0_step128 `.adl.D` = 0.0801998
4. **Callers:** baseline columns (whether a paper figure shows the ADL column was not confirmed); reuse_dial.
5. **Home and check:**
   - Home: `adl.py` `column_D(net, read, dW, pile: Rows, unit, gain, floor)`.
   - Check: the two saved values above, each in its own settings; floor from two runs.
