# Review: Γ reads the whole window in both directions, and what that does to the 41-token task-local results

Scope: the claim in `reports/analyses/2026-10-27_analysis_qwen-loader.md` (lines 16-24) that the released 67M
decomposition's Γ is one global transformer that attends in both directions, so that the comment at
`src/commute/vpd.py:166-170` is wrong and Γ's value at the trigger in `src/commute/local_faith.py` depends on tokens
after the trigger. Checked against the decomposition's config files and the pinned torch `param-decomp` at
`74146b5`; then every place the harness feeds Γ a short window, and the claims that rest on it. No weights were loaded,
no code was run, nothing else was edited.

I checked whether the importance function Γ of the 67M decomposition looks at the text after a position as well as
before it, and which of our results read Γ on 41-token windows instead of the 512-token rows it was trained on. It does:
Γ is one transformer shared by all 24 matrices whose attention has no causal mask, so the code comment that calls a
41-token window as good an input as a full row is wrong in its reasoning, and every reading of Γ at a trigger depends
on the 20 tokens after the trigger and on how short the window is. An earlier measurement shows that the change of
Γ's values between the original and the updated model comes almost entirely from the trigger's own inputs, so the
comparisons between models probably stand, but the absolute levels on windows and Γ's behaviour in full context are
untested, and one cheap GPU pass on saved windows would settle it.

**Verdict.** The claim holds as stated, for both parts. Part (1): the comment's justification is false; whether the
conclusion ("as valid an input") is false in effect is unmeasured. Part (2): true by construction. The problem was
already known on 2026-10-19 (`reports/analyses/2026-10-19_analysis_recruit-test.md:68-75`, Surprise 1) and on
2026-10-22 (`reports/plans/2026-10-22_plan_shift-anatomy.md:72-74`), but the `vpd.py` comment was never corrected and
no measurement of the window effect was made. The Qwen decomposition's Γ is causal (loader report, section 5), so
this is specific to the 67M harness, but it bears on which 67M findings port.

## 1. Evidence that Γ is global and bidirectional

- **Config.** `artifacts/vpd/s-55ea3f9b/final_config.yaml:7-18`: `ci_config: mode: global, fn_type:
  global_shared_transformer`, `d_model: 2048, n_blocks: 8, mlp_hidden_dim: [8192]`, `attn_config: n_heads: 16,
  max_len: 512, rope_base: 10000.0`. Training data (`:128-138`): `task_config.max_seq_len: 512`, pre-tokenised Pile
  rows, `batch_size: 64` (`:81`). Γ only ever saw full 512-token rows.
- **Pinned code.** `.venv/lib/python3.13/site-packages/param_decomp-0.0.1.dist-info/direct_url.json` gives commit
  `74146b555f3e…`. `param_decomp/models/component_model.py:316-330` builds `GlobalSharedTransformerCiFn` for
  `global_shared_transformer`.
- **The class.** `param_decomp/models/components.py:289-375`, `GlobalSharedTransformerCiFn`: rms-norms each of the 24
  matrices' inputs at each position, concatenates them, maps them to `d_model` with one linear layer, runs 8
  `TransformerBlock`s over the sequence dimension, then splits one output head into the 24 matrices' importance values.
  One function for every layer, not one per layer.
- **The attention.** `components.py:100-141`, `SelfAttention`, docstring "Multi-head bidirectional self-attention with
  RoPE"; line 134 "# Bidirectional attention (no causal mask)"; lines 135-137
  `scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)` with no `attn_mask`. RoPE is relative, so
  where a window sits in its row does not matter; which tokens are in it does.
- **The call path.** `src/commute/vpd.py:165-186` (`compute_ci`) runs `model.calc_causal_importances(out.cache, …)` on
  the whole `(B, T)` batch; `component_model.py:591` calls `self.ci_fn(pre_weight_acts)` with no mask. So the value at
  position t reads every matrix's input at every position of the batch row, before and after t.
- **The wrong comment.** `src/commute/vpd.py:167-169`: "Gamma reads each position's own pre-weight activations, so a
  41-token window is as valid an input as a 512-token row (the model is causal with rotary positions; `n_ctx` 512 is a
  maximum)." The parenthesis is true of the target model only. The assert at `vpd.py:171` (`1 <= T <= 512`) was relaxed
  on this reasoning on 2026-10-15.

What this means mechanically, for a 41-token window with the trigger at index 20 (`half = 20` everywhere:
`data.py:76`, `kv.py`, `ladder.py:141`, `experiments/{task_ladder,kv_task,new_vs_existing,many_cues}/run.py`):

1. **Short context.** Γ attends over 41 keys where it was trained on 512. The target model's own activations also lose
   their left context beyond 20 tokens; that part is the causal model's and was accepted on purpose.
2. **Fixed text after the trigger.** Γ's value at the trigger reads the 20 following Pile tokens. They are the same in
   the original and the updated model's forward.
3. **Changed activations after the trigger.** Under an update, the inputs at positions 21-40 change too, and Γ's value
   at the trigger reads them.

The masked model stays causal in its masks: its logit at the trigger depends only on the masks at positions 0-20
(`local_faith.masked_logits`, each position's components are applied to that position). But those masks are Γ's
outputs over all 41 positions, so the masked model's prediction at the trigger uses text it should not have seen.

## 2. What is already measured about the size

- **Channel 3 is about zero for the change between models.** E32's `R_steer_own`
  (`experiments/shift_anatomy/run.py:76-78, 273`; `src/commute/shift_anatomy.py:41-90`) re-runs Γ on the original
  model's activations with only the trigger's inputs replaced by the updated model's. It recovers 0.998, 1.010, 1.004,
  1.034, 1.012, 1.002, 1.043 and 1.003 of the effect of the updated model's own values at the trigger, on all eight
  do-nothing states (`reports/analyses/2026-10-22_analysis_shift-anatomy-and-many-cues.md:239-248`). So changed
  activations at the other 40 positions, before or after the trigger, add nothing measurable to the switching effect.
  Limits: one seed, step-10 Adam fine-tunes, the effect of the trigger's values only (not the route split's gates at
  positions 0-19).
- **The selection signal survives the window.** Emoticon windows are picked where `h.2.mlp.down_proj:2359` has
  g > 0.5 in the full row (`data.py:76-95`). Read in the window, the six emoticon subcomponents have g 0.83-0.93 at the
  trigger (`reports/analyses/2026-10-15_analysis_concentration-test.md:117`). Γ is not broken by the short input.
- **The window costs something at the original model, and the Γ-free control does not move.**
  `experiments/lora_commute/results/lora_commute.json`, W₀: CI-masked KL 0.45 nats at the 40 other window positions
  and 0.52 at the trigger, against 0.35 on 512-token Pile rows; stochastic 0.31 and 0.34 against 0.25; unmasked (masks
  at 1, no Γ) 0.010 and 0.009 against 0.012. The variants that read Γ are 1.3 to 1.5 times worse on windows; the one
  that does not is unchanged. Position mix and the windows' unusual text are confounds, so this is a hint, not a
  measurement of channel 1.
- Channels 1 and 2 on the response to an update (does Γ react to the same change of the trigger's inputs differently
  with 512 tokens of context, or without the following text?) are unmeasured.

## 3. Where the harness reads Γ on windows

Every task-local reading. Γ call sites that take windows: `local_faith.local_faithfulness`, `route_cells`,
`route_split`, `trigger_mean_gates` (`local_faith.py:114-289`); `gamma.trigger_gates` (`gamma.py:192`);
`regating.py:66-231`; `shift_anatomy.trigger_read` (`:41`); `shifted_harness.py:78-200`; `turned_down.py:96-142`;
`signatures.py:69`; `ladder.trigger_gbar` (`ladder.py:171`); `harness.measure`'s route split (`harness.py:142`).
About 40 of the 45 `experiments/*/run.py` reach one of these.

Not affected: the Pile faithfulness harness (`faithfulness.py`, 512-token rows), E36's spread readings (`spread.py`,
`experiments/regating_spread`, whole 512-token rows), the census and Pile profiles (`census.py`,
`attribution.gamma_profiles`), the Γ refit's Pile rows (`gamma.py:96`), teacher training (`teacher.py` uses no Γ), and
`attribution.py:37`, which cuts windows at the trigger for the target model only (causal, so exact).

## 4. Affected results

Ordered by how much of the project's reading rests on them.

| Result (claims page / `TRIED.md`) | What rests on Γ in windows | Could bidirectionality change the reading? | How big | Settled by |
|---|---|---|---|---|
| Route split shares: "small direct share", "re-switching share set by the task", "reveal and create show up as different routes", `retained_ci` per task (E5-E16, E21) | d, r, i, e_gate all use Γ's values at positions 0-20 of each window, at W₀ and at the state | Channel 3: no (E32). Channels 1-2: yes if Γ's reaction to a changed trigger input depends on context. A second risk on rung-0 ladder tasks: the target's baseline probability is ≥ 0.1, so in a tenth or more of windows the token right after the trigger *is* the target, and Γ at the trigger reads it | Unknown. The rung ordering of `retained_ci` (rung 0 0.79-1.02, rung 2 0.24-0.43, `2026-10-19_analysis_task-ladder.md:172-173`) is the reading most exposed to the second risk | Test A, B, C below |
| Switched-on sets and their overlap: "same target, same switched-on set" (E4 Amendment 2, E6), "the switched-on set is what produces the behaviour" (E18), E38's recruited set | Set = subcomponents whose trigger value rises by > 0.3 in windows | Membership near the 0.3 cut could change in full context; the overlaps compare sets read the same way, so they are less exposed | Unknown; likely small for the overlaps | Test A (overlap of the window set with the full-row set) |
| CI-masked P(o) at the trigger: 0.98 for sparse component training, 0.64 Adam-all, 0.53 do-nothing; "Γ's masks alone carry much of the behaviour (unexplained)" (E4) | Absolute level of the masked model on windows | Yes: absolute levels are where channels 1-2 act directly, and W₀'s masked KL is already 1.3-1.5× its Pile value on windows | Possibly tenths of P | Test A (masked log-odds in all three contexts) |
| Fixed-shift line (E18, E24, E28, E32, E33, E35), de-prioritised 2026-10-27 | Recovery ratios of trigger values, fitted and scored on windows | Ratios within one input form; E32 shows the change lives in the trigger's inputs. E30 applied a window-fitted change at every position of 512-token rows, a context it was not fitted in | Small for the ratios; E30's failure is set by the change's size (the ` http` null fails as badly), so unlikely to flip | Not worth a test while de-prioritised |
| "Γ coverage of the gradient predicts retention" refuted; "rarity does not matter" refuted at rung 2 (E5) | Mean trigger value over windows (`ladder.trigger_gbar`) | Window values are a noisier stand-in for in-context ones; could weaken a real effect | Unknown; a null, not a claim | Test A would show whether coverage ranks change |
| "The task-local adversary measures fragility at the trigger": no power (E5) | 20-step adversary on 73 windows reaches 25-30 nats (vs 0.69 on Pile rows) | Short context may be part of why windows are so easy to attack; also fewer positions share one mask | Already filed as no power | None needed |
| E36: change of Γ's values follows the original model's treatment of the input; "the last few tokens about a half; the current token 0.33 to 0.42" | 512-token rows, so not the length issue | Γ also reads the tokens *after* each position; the predictors tried were the current and preceding tokens only, so "what Γ's change follows" omits half its inputs | A reading limit, not an error | A "next few tokens" predictor, if the line resumes |
| What "trigger" means on emoticon | Trigger positions are where Γ rates 2359 important given the whole row, including what follows (e.g. the `)` after `:`) | The trigger set is partly defined by the continuation, which a causal model cannot see at the trigger | Interpretation only | None |
| E15 Γ refit, arm M | Packs 41-token windows end to end into 512-token rows (`experiments/gamma_refit/run.py:373-392`) | Γ attends across unrelated packed windows: a third input form, unlike both windows and Pile rows | Unknown; E15 is open | Note in E15's card |

Not affected: every Pile adversarial verdict (splits, component training, teacher training, the do-nothing reference),
expressibility and outside-share claims, teacher training's held-out target probability (target model, no Γ), and the
step-split claims.

## 5. Proposed test

One GPU pass of existing code on saved inputs, well under 30 minutes, so the cluster's `debug` QoS first. No new
measurement code beyond choosing the input rows.

Inputs: the emoticon held-out windows (`transport_test/results/large/windows.pt`, 73 windows, with `row` and `pos`
into round 1's 32,768 Pile train rows) and two ladder tasks, r0_t352 and r2_t359 (rung 0 and rung 2); states: W₀ and
E11's saved step-10 do-nothing fine-tune and component-training states (`saved_states`, from
`reading_a_coverage/results/large/`).

Three contexts for each window, the trigger always the read position:

- **window**: the 41-token window as now (index 20);
- **cut**: the window truncated at the trigger, tokens 0-20 (index 20; `route_split` reads only `[:, half]`, so it runs
  unchanged on `windows[:, :21]`);
- **row**: the full 512-token source row, read at `pos` (Γ values only: `compute_ci` on the rows, indexed per row).

Readings, each with its expected value if the windows are harmless:

- **A. Γ at the trigger, W₀.** Mean |g_window − g_row| and |g_cut − g_row| over alive subcomponents, and the overlap of
  the g > 0.5 sets. Harmless: mean difference well under the 0.3 switching cut, overlap ≥ 0.8. Also the masked
  model's trigger log-odds in window vs cut (row needs a per-row index; skip it or bucket windows by `pos`).
- **B. Route split shares in window vs cut,** do-nothing and component training, three tasks. Harmless: every share
  moves by < 0.05. A move > 0.1 in r or `retained_ci` means the task-local claims should be re-read on cut windows
  (which are what a causal reader would see).
- **C. Rung-0 continuation check.** On r0_t352, split W₀'s masked trigger log-odds and the do-nothing r by whether the
  token after the trigger equals the target. Harmless: no difference beyond the seed spread. A gap means Γ passes the
  next token to the masked model, and `cut` windows are the fair input.

Noise floor: compare against the route split's seed-to-seed spread from the task ladder's noise arm. Positive
control: unmasked log-odds must be identical in window and cut (target model causal); any difference is a bug.

## 6. Recommendation

Correct the comment at `src/commute/vpd.py:166-170` to say that Γ is a bidirectional transformer over the whole batch
row and that windows are a shorter-than-trained input (left to the lead; this review edits nothing). Add the window
blind spot to the "Blind spots" paragraph of `local_faith.py` (lines 20-21). Run test A-C before any task-local
finding is cited as ported to the Qwen line, where Γ is causal and cut windows are the matching input.

## Not checked

- Numerical size of channels 1 and 2: nothing was run; weights were not loaded.
- `param-decomp`'s training data loader (padding, packing, document boundaries); I read only the config's
  `max_seq_len: 512` and tokenised rows.
- Each `experiments/*/run.py` individually; the count of affected experiments comes from grepping for the Γ call sites
  listed in section 3.
- Whether any ladder task's windows exclude continuations equal to the target (`ladder.windows_for_trigger` does not
  appear to), and the actual share of such windows per task.
- The many-cues windows' continuations (E33's target ` of` sits at rungs 0-2, so the same check C applies).
- E15 arm M's results in light of the packing.
- The route split's gates at positions 0-19 (E32's ceiling covers the trigger's own values only).
