# commute: helper index

Lookup index for plan authors: name the helper, don't reread the module. One line per module, then the public helpers worth reusing.
Layouts: param-decomp `(U (C, d_out), V (d_in, C))`; theory `(U_A (d_out, n), V_A (d_in, n))` of alive columns. `State` = `{k: (delta-W, U_A', V_A')}` (harness); `Update` = `(delta-W, {k: (U2, V2)})` (saved_states); `Sel` = `{k: long idx}`, `Gates` = per-batch `[{k: (B,T,C)}]` (recruit).

## By task
- load model / W0 -> `vpd.load_component_model(device)`, `vpd.target_weights(model)`; files -> `vpd_files.fetch()`
- full context (model, W0, alive, uv, bases, sigma, windows, Pile blocks, W0 floor) -> `harness.load(...)`, `harness.task_context(ctx, windows_path)`
- alive set -> `saved_states.load_alive(round1)` or `vpd.alive_masks(vpd.mean_ci(model, batches))`; theory layout -> `component_training.alive_uv(model, alive)`
- load a saved state -> `saved_states.ct_state` / `split_states` / `offsets_state`; save one -> `arms.save_offsets`
- write a state temporarily -> `faithfulness.model_state(model, target=, components=)`; do-nothing state -> `regating.state_of(model, W0, dW)`
- build a state from delta-W -> `split.split_state` (least squares), `split.adjoint_state`, `split.rescaling_state`, `component_training.train`, `row_copy.copy_states`
- fine-tune -> `finetune.finetune_arm_a` / `finetune_arm_b` / `finetune_lora_arm_a`; `regating.finetune`; teacher -> `teacher.teacher_net(model, dW)`
- component sets -> `signatures.top_by`, `signatures.matched_random`, `recruit.switched`, `recruit.random_like`, `gamma.s_on`, `turned_down.down_set`, `transport.top_k_by_contribution`
- ablate a set -> weights: `signatures.ablated_weights(W, comps, sel)`; hooks: `identity.Edit`; KL per subcomponent: `gamma.ablation_kl`; logit effect: `attribution.ablation_effect`
- revert a set -> `attribution.reverted_weights`, `attribution.revert_one`, `cancel.Reverter` / `cancel.Remover`
- gates g -> `vpd.compute_ci(model, batch).g`; trigger: `local_faith.trigger_mean_gates`, `gamma.trigger_gates`, `recruit.at_trigger`, `signatures.trigger_g`
- re-gating (fixed shift s, patterns, recovery) -> `regating.shift_of`, `regating.pattern_gates`, `regating.recovery`, `regating.measure`, `recruit.shifted/with_trigger`
- run masked model on a gate pattern -> `local_faith.masked_logits`, `local_faith.masked_lo`, `regating.lo_of`, `spread.masked_logp`
- capture inputs / pre-acts at trigger -> `shift_anatomy.trigger_read`; input moments -> `split.input_second_moments`; x.v acts -> `attribution.read_acts`
- edit inputs at trigger -> `shift_anatomy.add_at_trigger` / `set_at_trigger` (+ `steer`)
- faithfulness: Pile -> `faithfulness.pile_faithfulness`, `harness.pile_ratios`; trigger-local -> `local_faith.local_faithfulness`, `local_faith.route_split`; verdict -> `faith_score.state_verdict`
- adversary sources / masked KL -> `gamma.pgd_sources`, `gamma.mask_kl`, `cancel.masked_kl`
- windows / data -> `data.pile_rows`, `data.emoticon_windows`, `regating.load_windows`, `ladder.windows_for_trigger`, `kv.windows`

## Model, files, data
**env** — cache root constants: `REPO_ROOT`, `CACHE_DIR`, `VPD_DIR`, `HF_DIR`, `OUT_DIR` (import before param_decomp).
**vpd_files** — W&B locations of the released runs. `fetch(log)`: download both runs once (needs WANDB_API_KEY).
**vpd** — released decomposition: load, CI, alive, accessors. Consts `SEQ_LEN`, `CI_CHUNK`, `EMOTICON_KEY`, `PAPER_FAITHFULNESS`.
- `load_component_model(device='cuda')`: ComponentModel, key-free from local files
- `run_config()` / `sparsity_config()` / `tokenizer()`: released Config; final p, beta, eps; tokenizer
- `module_keys(model)`, `layer_of(key)`, `parse_key(key)` -> (name, idx), `target_linear(model, name)`
- `target_weights(model)` / `set_target_weights(model, weights)`: fp32 W per matrix
- `uv(model, name)`, `component_weights(model)` / `set_component_weights(model, weights)`: param-decomp (U, V)
- `weight_deltas(model)`: Delta = W - sum B_c
- `target_logits(model, batch)`; `compute_ci(model, batch, chunk)` -> `CI(g, logits)`; `mean_ci(model, batches)`
- `alive_masks(mean_g, threshold=1e-6)`; `kl_per_position(p_logits, q_logits)`
**data** — Pile rows and trigger windows.
- `pile_rows(split, start, n)` -> (n, 512); `batches(rows, size)`
- `Windows(tokens, g, row, pos, half)`, `.subset(mask)`; `split_windows(w, heldout_every=5)` -> (train, held)
- `emoticon_windows(model, rows, key, threshold, half=20)`; `subcomponent_g(model, rows, key)` -> (N, T)
- `target_tokens(token, toks, half)`: per-window target from id or (V,) table; `replace_rows(rows, hit, spare, spare_hit)`
**ladder** — trigger -> target task ladder and coverage.
- `trigger_stats(logits_fn, rows, triggers, half)` -> TriggerStats; `candidates(stats, trigger, plain, per_rung)`; `rung_of(baseline)`, `rung_bounds(r)`
- `windows_for_trigger(rows, trigger, half, max_n, seed)`; `plain_mask(decode, vocab)`; `count_tokens(rows, half)`
- `trigger_gbar(model, uv_alive_index, toks, half)`: mean g at trigger; `trigger_gradient(model, toks, half, token)` -> dL/dW; `coverage_from(G, uv, gbar)`
**kv** — key -> value tasks. `KVTask.table(vocab)`; `identity(...)`, `bijection(name, marker, keys, pool, seed)`; `pick_marker(counts, decode, vocab)`; `windows(rows, marker, keys, half, n, seed)`; `key_distributions(logits_fn, w, device)`.
**kv_windows** — kv windows in the harness (E42). `task_context(ctx, windows_path)` -> (Context with the table as target, KVTask); `pile_reading(nets, rows, task, device)`: mean P(table[x]) after each key x in plain rows, per net, and marker counts.
**many_cues** — E33 many cues, one target. `qualifying(stats, pool, plain)`, `pick_target(Q)`, `alternative(p, q, target, used)`, `concat(ws)`, `table(pairs, vocab)`, `overlap(d, labels, alive)`, `donors(labels, same, seed)`, `reading(...)`, `combined(one, many)`.

## States: build, load, save
**saved_states** — rebuild saved states from results/large paths.
- `load_alive(round1)`, `load_mean_g(round1)`
- `ct_state(e11, task, step)` -> Update; `split_states(model, uv, e11, round1, task, step, tol)` -> {"ls", "adjoint"}
- `offsets_state(path, uv)` -> Update from an `arms.save_offsets` file
**harness** — one state measured the same way everywhere. `State`, `Context` (model, W0, comps0, alive, uv, uv_cpu, bases, sigma, train, held, token, rbase, blocks, ...).
- `load(device, alive_path, windows_path, o_token, tol, pile_rows, pile_pgd_steps, pile_seeds)` -> Context
- `with_windows(ctx, windows, token)`; `task_context(ctx, windows_path)` -> (ctx, sigma)
- `measure(ctx, name, dW, state, pile)`: route split, Delta share, Pile ratios; `nothing_and_split(ctx, name, dW, p_o, pile)`
- `pile_ratios(ctx)`, `pgd_floor(ctx)`, `outside_share(ctx, dW)`, `outside_share_of(dW, bases, sigma, non_saturated)`
- `at(ctx, dW, uv2)`: context manager, the model at W0 + dW with alive (U', V') = uv2 (None: do-nothing)
**lowercase** — E37's lowercase task and E39's fine-tunes. `task(model, device)` -> `Lower(rows, held, w0_net, logp_w0)`; `finetune(model, device, low, "adam"|"sgd"|"lora")` -> dW; `gain(model, device, low, dW)`.
**split** — splits of delta-W over the alive subcomponents (theory section 5).
- `alive_uv(U, V, alive)`: param-decomp -> theory (float64); `bases(uv, tol)` -> {k: Basis}; `basis(U_A, V_A, tol)`
- `split_state(dW, uv, b)` -> State (least squares); `least_squares_split(M, b, U_A, V_A)`; `split_with_delta_change(...)`
- `adjoint_state(dW, uv, sigma)` / `rescaling_state(dW, uv, sigma)` -> (weight change, State); `adjoint_split`, `adjoint_scale`
- `obstruction(M, b)`, `obstruction_fraction`, `gaussian_null`, `weighted_fractions(M, b, Sigma)`, `four_blocks`, `block_table`, `rescaling_table`, `reproduced_share`
- `input_second_moments(model, batches, keys)`: E[a a^T] per matrix input
- `piece_energies(dU, dV, U_A, V_A, Sigma)`, `n_half(energies)`, `random_like(M, Sigma, seed)`
- `StepSplit`: `.step(W)`, `.state()`, `.delta_change()`; `adjoint_step(S, U_A, V_A, Sigma)`
**step_split** — read-outs along a StepSplit run. `numbers(carry, fixed, uv0, sigma)`, `carry_fixed_gap`, `per_step_maxima`.
**component_training** — post-train alive (U, V) offsets, Delta fixed.
- `alive_uv(model, alive)` -> theory (U_A, V_A) fp32 on device; `components_with(model, alive, states)` -> full param-decomp comps
- `Offsets(uv, rank=None, seed, init)`: `.delta_w(k)`, `.components(k)`; `hooks(linears, off)`
- `TrainConfig`; `train(model, uv, train_tokens, heldout, a_cfg, cfg, device, W0, save_dir, loss_fn, init)` -> (curve, {step: State})
**arms** — comparing training arms. `first_learned(p_at, p_learn)`, `select_rate(arms, p_min, limit)`, `heldout_of(model, W0, held, token, device, dW)`, `save_offsets(path, uv_cpu, st, meta)`.
**finetune** — fine-tunes of the 24 matrices.
- `decomposed_params(model)`, `relative_norms(dW, W0)`, `cosines(a, b)`, `random_direction(dW, seed)`, `logits_of(net, tokens)`
- `ArmAConfig`, `ArmBConfig`, `LoraConfig`, `FinetuneResult`; `arm_a_loss(net, ref, toks, cfg, half)`, `eval_arm_a(net, ref, w, o_token, device)`, `eval_arm_b`
- `finetune_arm_a(model, train, heldout, cfg, device)` / `finetune_arm_b(model, train_rows, heldout_rows, cfg, device)` -> FinetuneResult
- `finetune_lora_arm_a(model, train, heldout, a_cfg, cfg, device)`; `lora_factors`, `lora_hooks`, `lora_delta`, `saved_lora_updates(path, all_keys, W0, steps)`
**lora_readouts** — `init_read_factors(model, modules, rank, seed)`, `cos_with_init(factors, A0)`, `select_rate(evals, p_min, kl_max)`.
**teacher** — teacher training (KL from frozen W0 + dW).
- `teacher_net(model, dW)`: frozen target copy at W0 + dW; `teacher_loss(net, teacher, toks, half, kl_lambda, keep)`; `heldout_gap(net, teacher, w, device)`; `kl_exact_zero(t_logits, s_logits)`
**masked_teacher** — teacher training plus VPD recon losses. `MaskedConfig`; `student_kl(model, teacher_net, toks, wd)`; `train(model, teacher_net, rows, alive, uv0, cfg)` -> (curve, {step: State}).
**row_copy** — teacher copy on whole rows. `copy_on_rows(ctx, teacher_net, rows, lr, batch, steps, seed)`, `copy_states(..., save_steps, init)` -> (curve, {step: State}); `mean_logp(net, rows, device, batch)`, `mean_kl(p_net, q_net, rows, device, batch)`, `lowercase_rows(rows, tok, n)`.
**transport** — delta-W as changes to existing subcomponents via hooked extras.
- `forward(model, batch, extra)`, `delta_extra(dW, device)`, `scale_extra`, `write_edit_extra`, `read_edit_extra`
- `frobenius_fit(dW, U, V, idx)`, `frobenius_fit_all(model, dW, index_sets, device)`, `pooled_fraction(fits, dW)`
- `kl_to_reference(model, ref_extra, extra, batch)`, `functional_fit(...)`, `capture(model, ref_extra, extra, batches)` -> Capture
- `top_k_by_contribution(fits, k)`, `random_matched(sets, alive, seed)`

## Ablation, reverts, identity
**identity** — gauge-free subcomponent comparisons and the Edit hook.
- `Edit(terms, cidx)`: forward hook; `[(U, V, +1)]` ablates, `[(U2, V2, +1), (U, V, -1)]` reverts; groups of rows edit `cidx[g]`
- `change_stats(U, V, U2, V2)`, `profile_retrieval(before, after, idx)`, `topk_jaccard(a, b, k)`, `cosine`, `log_odds(logits, token)`, `scale_grads(G, U, V)`
**attribution** — revert/ablate/read single subcomponents in place.
- `revert_one(model, UV, st, toks, half, o_token, base, keys)`: per-subcomponent log-odds drop; `concentration(...)`: top-1/10/100 revert
- `reverted_weights(W, UV, st, sel)`; `ablation_effect(model, k, j_alive, U, V, pile_rows, flat_pos)`
- `read_acts(model, UVd, rows, keys)`: x.v per alive subcomponent; `gamma_profiles(model, pile, sel)`; `fwd(net, toks)`, `heldout_log_odds(...)`
**signatures** — where an update lands; ablate that set.
- `split_energy(dW, uv, bases, alive)`, `top_by(values, alive, k)` -> Sel, `share_in`, `mean_over`
- `matched_random(sel, alive, g, seed, bins=10)`: same size, same g decile; `ablated_weights(W, comps, sel)`
- `trigger_logits(model, windows, half)`, `trigger_stats(ref, logits, windows, half, token)`, `trigger_g(gates, half)`, `pile_gate_stats(model, batches, alive)`
**cancel** — do per-subcomponent changes cancel? (E17/E20)
- `change_energies`, `signed_sum`, `coherence(pieces)`; `Revert` / `Reverter(model, pieces)`: `.logits(batch, weights)`, `.remove()`
- `Remove` / `Remover(model, blocks)`; `state_blocks(model, alive, pieces)`; `cache_state(model, batches, alive)`, `cache_adversary(...)`
- `revert_kl(rev, cached, weights_of)`; weight builders `half_weights`, `adversary_weights`, `arm`, `rest_arm`, `dead_weights`, `all_blocks_arm`
- `masked_kl(model, batches, sources)`, `alive_whole(sources, alive)`, `levels`, `gauge_mixed(pieces, energy_ratio, seed)`, `importance_bins`, `layer_bins`, `bin_means`
**excess** — adversary's excess in parts per model. `measure(model, batches, alive, uv, W0, update, seed, src0)`, `parts(s, w0)`, `reading(d)`, `kl_to_weights(model, cached, weights)`.

## Gates and re-gating
**recruit** — switched-on sets and gate-pattern bookkeeping (pure).
- `trigger_dg(gates_state, gates_base, half, alive)`, `mean_change(mean_state, mean_base, alive)`, `switched(dg, thr, on)` -> Sel
- `size`, `members`, `jaccard(a, b)`, `by_layer`, `restrict(sel, min_layer)`, `random_like(sel, alive, seed)`, `in_range(shift, alive, lo, hi)`
- patterns: `hybrid(gates_base, gates_state, sel)`, `at_trigger(gates, half)`, `added` / `shifted` / `constant`, `with_trigger(gates_base, half, trig)`, `plus(trig, shift, clip)`, `permuted(trig, perm)`
- shifts: `moved(shift, sel, dest)`, `shuffled(shift, alive, seed)`, `signed(shift, positive)`, `on_pool(shift, pool)`, `derangement(n, seed)`
- `precede_mask(rows, token)`, `ContrastAcc().add(g, mask)/.result(alive)`
**regating** — E18 trigger-only pattern test, shared steps.
- `Task(name, train, held, target, base, m0)`; `load_windows(name, transport, ladder)`, `load_kv_windows(name, kv_results)`, `prepare(model, name, train, held, target)` -> Task
- `state_of(model, W0, dW)` (ctx mgr); `finetune(model, task, token, optimizer, lr, steps, seed, device)` -> (dW, eval)
- `shift_of(model, W0, dW, task, alive)` -> (s, m1); `state_trigger_values(model, W0, dW, task)` -> g1
- `pattern_gates(g0, half, g1, shift, m0, m1, shift_null)`, `random_gates(g0, half, shift, alive, seed)`
- `lo_of(model, task, gates)`, `recovery(model, task, trig, gap)`, `measure(model, W0, task, dW, shift_null, alive)`, `zero_recovery`
- `shuffled_recovery(...)`, `other_window_recovery(...)`, `shares(draws, rho_trig)`, `natural_contrast(model, rows, targets, alive)`, `j_nat(on, contrast, alive)`
**turned_down** — the turned-down half of s, read at W0 (E29).
- `load_shift(path)`, `pool_of(m0, alive)`, `down_set(shift, pool)`, `as_sel(mask)`, `random_in_pool`, `overlap(sel, nat, pool, n_random)`
- `contrasts(model, rows, masks, alive)`, `precede_any`, `mean_next_probs`, `top_competitors`, `mass_of_logits`, `masses(model, windows, half, target, comp, gates, batch)`
- `down_pattern(g0, half, shift)`, `shuffled_down`, `shuffled_pattern`, `drop(ref, pat)`
**shift_anatomy** — what the fixed change s is made of (E32); input capture.
- `trigger_read(model, tokens, half, target, edit)`: per window at trigger `g`, `pre`, input `x`, `logp`
- `add_at_trigger(half, delta)` / `set_at_trigger(half, x_new)` -> Edit; `steer(model, task, edit, gap)`
- `to_trig(vals)`, `mean_of`, `group_means(vals, labels)`, `per_window`, `flat(vals, alive)`, `along(d, s, alive)`, `strength`, `mixed`, `key_match`
- `rho_of(model, task, vals, gap)`, `per_window_lo`, `group_fixed(...)`, `is_attention(key)`
**cuefree** — cue-free models of the change d = g1 - g0 (E35). `change(g1, g0, alive)`, `apply(g0, c)`, `by_group`, `neighbours`, `nearest(g0_tr, g0_he, d_tr, alive, k, device)`, `kmeans`, `clusters`, `purity`, `refit(g0_tr, g1_tr, alive)` + `reweighted`.
**spread** — E35 arms at every position of whole rows (E36). `layout(alive, device)`, `to_flat(g, lay)`, `with_alive(g, x, lay)`, `masked_logp(model, rows, gates)`, `target_logp`, `nearest`, `nearest_change`, `group_change`, `fit_clusters`, `fit_tokens` -> TokenTable.
**shift_readout** — pure readouts of a shift s. `pooled`, `counts(s, alive, thresholds)`, `mass_split(s)`, `on_layer(s, layer)`, `top_k(s, k)`, `spearman`, `write_energy(uv, dW)`, `opened(g, shift)`.
**shifted_harness** — Pile harness with Gamma' = clip(Gamma + s) (E30).
- `shift_gates(g, shift)`, `shifted_faithfulness(model, target, batches, shift)`, `shifted_pile(model, target, blocks, shift)`, `shifted_pgd`, `shifted_sparsity_loss`, `everywhere(gates, shift)`
**shift_map** — E30 bench. `Bench(device, out, transport, ladder, pile_rows, ...)`: `.task(name)`, `.set_twin(dW)`, `.ordinary(seed)`, `.shifted(shift)`, `.reading(name, fn)`; `c1(b, seeds, exact_tol)`, `trigger_cells(b, t, dW, s)`, `r_everywhere(...)`.
**gamma** — refit Gamma on an updated state; adversary sources.
- `gamma_params`, `set_trainable`, `gamma_state(model)` / `load_gamma(model, state)`; `RefitConfig`, `loss_setup(...)`, `Refit.step(rows)`, `refit(model, cfg, rows_at, evaluate)`
- `trigger_gates(model, windows, half, token)`, `s_on(trig, trig_w0, alive)`, `mean_gates(model, sel, rows)`
- `pgd_sources(model, batches, steps)` -> (sources, KL); `harness_start(model, seed)`; `mask_kl(model, batches, sources)`; `adversary_groups(...)`
- `ablation_kl(model, key, cidx, rows)`: KL from ablating each subcomponent

## Faithfulness
**faithfulness** — authors' metrics (stochastic + shared-adversary PGD).
- `model_state(model, target=None, components=None)`: ctx mgr, writes then restores
- `faithfulness(model, batches, n_stoch_draws, pgd_steps, seed)`, `harness_pgd(...)`, `pile_blocks(n_rows)`, `pile_faithfulness(model, blocks, seed, pgd_steps)`
- `sparsity_loss(model, batches, pnorm, beta)`; `transported_components(model, scale, dU, dV)`
**local_faith** — faithfulness on trigger windows, trigger vs rest.
- `local_faithfulness(model, windows, half, token)`, `retained(res, base)`, `stats(target_logits, logits, half, token)`
- `masked_logits(model, batch, masks, wd, delta_masks, autocast)`: forward under given masks
- `route_cells(model, windows, half, token)`: gates + T/U/L log-odds; `route_split(model, windows, half, token, base, components0)`
- `masked_lo(model, windows, half, token, gates)` / `masked_lo_windows(...)`; `trigger_mean_gates(model, windows, half)`
**faith_score** — scoring Pile readings vs W0. `seed_floor(pile, key)`, `w0_check(pile)`, `verdict(ratio, f, line)`, `trigger_ratios(m)`, `state_verdict(pile, p_o, w0, p_min)`.
