# Measurement record of the paper's results (review, 2026-11-27)

We wrote down what every figure, table and measured number in the paper's results actually measures, using only the
code and the runs' saved files. Two writers recorded each of the five sections without seeing each other's work, and a
third person settled every disagreement by reading the code. The record holds 180 items. Checked against the frozen
paper text, 55 places differ from what the code shows, and 6 of them are serious. The worst is a control that is said
to clear the definition's low readings although it never learned two of its five tasks. Next is an appendix figure
that lacks two of the readings the text sends readers to. Most of the other differences are appendix numbers or
method descriptions that do not match the code, so the paper's main claims mostly stand. The six serious ones should
be fixed or reworded before the draft is final.

Scope and sources. The frozen paper is origin/v2 at commit 6832db7, copied to
`~/commute-data/agents/redteam-lead/measurement-record/paper_v2_6832db7/paper/`, written `P/` below. The research
code is `~/commute-agents/redteam-lead/`, written `R/`, with experiments under `R/experiments/`, written `X/`. The inputs are the
five reconciled records `S1_final.md` to `S5_final.md` and `cross_figure.md` in the same measurement-record directory.
I re-read every quoted paper sentence in the frozen copy. Where a cell says "checked here", I also re-read the code
fact in the named file. Every other code fact is the reconciler's, with its citation.

Codes. An E-number is an experiment number of the research repo, and each names one run directory. "Cross-figure
difference N" points to the list further down. The codes used are these:

| Code | Meaning |
|---|---|
| E136 | re-gating share reads of the 16 known-answer tasks |
| E147 | the 96-key refits |
| E228 | the gain-matched random reference and its reader |
| E231 | gain-only training |
| E253 | the cued copy task |
| E268 | built edits read by ρ |
| E270 | the Qwen read of ρ's expression over every variable |
| E277 | top-neuron ρ |
| E283 | the retained-clause test |
| E290 | the rescale-only control |
| E322 | that control's three-seed reads |
| E325 | Fig 2 rerun on 16 tasks, in progress |
| ρ, K | the retained share; the necessary set |
| VPD, CMAP | the parameter decomposition; head patching |

Grading. A: a number or claim in the body or a caption is wrong or misdescribes what was measured. An appendix
sentence that misdescribes a plotted figure is also graded A, as in A4. B: an omission where a neighbouring figure
differs, or a wrong appendix number. A body count that is off without changing its claim is graded B, as in B28.
C: minor.

## Mismatch list

### Severity A

| # | Paper says (frozen copy) | Code and saved files show | Touches | Sections |
|---|---|---|---|---|
| A1 | `P/sec5_results.tex:17` "A fine-tune that can only rescale neuron outputs reads ... near the full fine-tunes at VPD (0.02 to 0.07 over three seeds: about twice on copy, about half on the cipher". `P/sec7_discussion.tex:18` "a fine-tune that can only rescale, reuse by construction, reads 0.02 to 0.07 ... (higher on copy, lower on the cipher ...), so the low VPD values do not count against the definition". `P/appI_results_detail.tex:56-57` "The control trained for up to 300 steps at learning rate 3e-3". | The rescale-only control (E290's down-projection-input gain arm) never learned the cipher or the prior-matched cipher: `X/ia3_control/results/train/ffn_cipher_s0.json`, `ffn_prior_cipher_s0.json` → `learned` false (checked here); held-out P(answer) 0.018-0.021 and 0.186-0.191 at step 300, against 0.67-0.96 for the full fine-tunes (S1_final, S5_final D5). So "about half on the cipher" compares a run that did not learn the task. The control is read at steps 100 to 300 against the full fine-tunes' 30. Its three-seed values come from E322, which no paper data CSV holds (`X/p165_gain_reads/results/definition_rows.csv`). | The discussion's claim that low VPD ρ does not count against the definition; `sec5:17` | S1, S5 |
| A2 | `P/appI_results_detail.tex:118-121` caption "Six further readings ... the error-term share, the reconstruction gap (grey crosses: a random update of the same weight size ...)". `P/sec5_results.tex:28` "Six further proxies ... are shown in Appendix". `P/sec5_results.tex:73` "The reconstruction gap is in Figure known_appendix". | The frozen `P/figures/data/known_appendix.csv` holds 5 readings × 16 tasks: neuron patching, top-neuron share, neurons for half the gain, update size and base prior (checked here). It has no error-term share, no reconstruction gap and no grey crosses. The script asks for 7 readings (`P/figures/src/make_known_scorecard.py:186`); paper commit bdc4fcd says "figure NOT regenerated". The values exist only in the writeup-lead clone's `experiments/sae_proxies/results/results.json`. | fig:known_appendix; the reconstruction-gap row of tab:tools | S2, S4 |
| A3 | `P/sec5_results.tex:85` "The re-gating share breaks down completely: it reads 0.99 or more for every fine-tune and for random changes that move the output as much". `P/sec5_results.tex:10` "On Qwen3-1.7B the re-gating share saturates". | The cited Figure qwen_share has a second panel, the neuron control, where the fine-tunes' means read 0.926-0.984 and the random changes 0.810-0.919 (`P/figures/data/qwen_share.csv`, rows "One subcomponent per MLP neuron", checked here). The 0.99 holds on the VPD panel only. `P/appH_qwen_short.tex:17,21` do say "on VPD"; the body does not. | Fig qwen_share; the Qwen summary sentence | S5 |
| A4 | `P/appE_methods_detail.tex:152` "Every reading is taken on the same saved weight changes δW". `P/appI_results_detail.tex:220` "the 48 fine-tunes of Fig. known". The caption `P/sec5_results.tex:78` says nothing on size. | Head patching reads W0 + c·δW. The larger side of each pair is scaled to its partner's size, with c from 0.52 to 0.98 on seven built-new tasks and 0.9975 on Top guess (baseline-lead `experiments/cmap/job.py:121` and `dW` at `:275-276`, checked here; c values in `weightwatch/results/p2-80/tools_fig4.json → .cmap.c`). The three neighbouring panels read δW as saved. Cross-figure difference 2. | Fig tools_known panel 2; the head-patching row of tab:tools | S4 |
| A5 | `P/appI_results_detail.tex:129-130` caption "Four tasks of one family cross the prior (high, low) with the mechanism". `P/sec5_results.tex:37` "copy key and a copy on rarer answer words ... All three readings put the two copies apart from the two ciphers". | The low-prior copy corner is E253's cued copy, a 32-key copy-at-a-marker task on cue windows with up to 400 read rows (`X/cued_copy/run.py:1-10`; `X/cued_copy/results/predictions.json → settings`). The other three corners are 96-key tasks read on 286 rows. The cued copy's own filed control `C_absent` reads "misses" (`X/cued_copy/results/table.json → rules.C_absent`, checked here). Cross-figure difference 8. | fig:prior_mechanism; the body claim that the readings follow mechanism, not prior | S2 |
| A6 | `P/sec4_setup.tex:19` "The search takes 8 to 24 GPU-minutes per state at VPD". | The trained fine-tunes' VPD searches took 2.1-2.6 minutes. The 8.5-24.1 minutes are the built edits only (`X/fig2_standard/results/items/vpd/*.json → seconds`; e.g. `trained__copy_s1.json` 154 s and `built__rand_x2.json` 1,440 s, checked here). The file does not say whether `seconds` times the search alone. | The cost statement for ρ | S2 |

### Severity B

| # | Paper says | Code and saved files show | Touches | Sections |
|---|---|---|---|---|
| B1 | `P/main.tex:370-371` "the alive subcomponents (those whose mean causal importance is above 10^{-6})"; `:375-376` "Subcomponent c becomes (u_c + du_c)(v_c + dv_c)^T ... Δ' = Δ + O(δW) − dU dV^T". | The ρ path uses every subcomponent with no alive selection, 38,938 variables (`R/src/commute/kstar_vpd.py:91-93`). A subcomponent writes (u+du)(v·y) + u(dv·y), with no du dvᵀ term (`kstar_vpd.py:103-109`). The Qwen re-gating share does use an alive set (`X/qwen_reading/lab.py:356-393`). | Every VPD ρ (Fig 2, tab:built, Fig 5) | S1, S5 |
| B2 | `P/appI_results_detail.tex:107-109` "the share of F's necessary variables that are also necessary for M ... The doubled and tenfold edits read 0.85". | The 0.85 is E283's flip-weighted reading (x2 0.848, x10 0.859). The plain shared-variable share reads 0.770 and 0.773 (`X/reliance/results/summary.json`). | The retained-clause test | S1 |
| B3 | `P/main.tex:403` "It reads 0.9935 or more on every trained fine-tune and random update". | The KL-matched random update reads 0.9845 (`X/cos_admissible/results/summary.json → items.random_kl_241002.cos2_K`, checked here). The squared cosine is unweighted (`X/cos_admissible/run.py:85-87`), not weighted by w_u as `:399` suggests. | The appendix argument for using ρ | S1 |
| B4 | `P/appI_results_detail.tex:102` "The copy fine-tunes read 0.036 to 0.043 at VPD". | Copy reads 0.0362-0.0399 (`X/claim2_needed/results/items/e62-copy_s{0,1,2}.json`, checked here); 0.043 is lookup. | Why the copy pair reads low | S1 |
| B5 | `P/appI_results_detail.tex:104-105` "the four MLP remainders carry 53 to 80% of the write change, the copy head under 1%". | 53-80% is the range over every E277 trained item, the realistic panel included. On the copy pair it is 0.648-0.686 (`X/needed_neurons/results/summary.json → items.e62/copy_s*.rem_dd_share_K_changed`). | Why the copy pair reads low | S1 |
| B6 | `P/sec5_results.tex:21` caption "fine-tunes of the five 96-key synthetic tasks, and for random updates of matched size"; `P/appE_methods_detail.tex:20-22` (pooled built edits). | Trained states are read on 512 eight-token windows, 3/8 task, 3/8 does-not and 2/8 far rows, with ρ summed over all 8 positions (`R/src/commute/old_tasks.py:231-241`; `stand_in.py:130-147`). The copy24 doubled, tenfold and rebuilt edits are read at all 41 positions of 75 windows. The other built edits are read at one position of 512 windows (item JSONs → `n_read_positions`). The five-task scope is already known and is being rerun as E325. Cross-figure differences 1 and 13. | Fig def_reading | S1 |
| B7 | `P/appE_methods_detail.tex:136-137` "three tasks built new (the random map and the cipher at both steps) reach 0.72 to 0.91 of their fine-tune's gain". | The three are the random map 0.906, the early 32-key cipher 0.737 and the 24-key open-pair cipher 0.716. The late cipher reaches 0.364 (`X/gain_only/results/score.json → pairs.*.kl.reach`, checked here). The same file's `gain.reach` values differ, at 1.00, 1.01, 1.00 and 0.70. | Gain-only claim | S2 |
| B8 | `P/appE_methods_detail.tex:38` "A seed changes only the order of the training batches". | For every key-value task the seed drives with-replacement `torch.randint` batch draws (`R/src/commute/finetune.py:205,209`); only the Pile tasks permute a fixed set. | Recipe of all 16 tasks | S2 |
| B9 | `P/appE_methods_detail.tex:51-52` "cipher (24 or 96 keys) ... one cipher is read at two steps". | The early/late cipher has 32 keys (`X/keydependent_elicit/results/stage1.json → cipher.keys`, length 32). | tab:pairs, Fig known | S2 |
| B10 | `P/sec5_results.tex:28` "After the fine-tune's 300 steps each reads within about 0.10 of its own fine-tune"; `appI:171` "the recipe's 300 steps". | The 300 steps are the subspace run's. The fine-tunes it is compared with are read at their own step, 10 or 30 for seven of eight tasks (`X/subspace_share/results/updates/*.json → end.step`; `control.json → fine_tunes`). Per-seed differences run from −0.098 to +0.111. | Subspace-trained reference | S2 |
| B11 | `P/appI_results_detail.tex:158-159` and `P/sec5_results.tex:28` "still reads 0.52 to 1.02 on the tasks built to reuse and 0.15 to 0.35 on those built new". | These are task means at the KL-matched kept state; at the gain-matched state they differ, e.g. Cipher 24 0.106-0.108. They are read by E228's reader against E228's references, which hold only seeds 0-1 for some tasks. All nine ciphered-text and lowercase gain-only runs stop at the 3,000-step cap below their reference gain, at 1.15-1.21 against 1.64-3.84 (`X/gain_only/results/items/e53_*.json`). | Claim that the share may partly read the task | S2 |
| B12 | `P/appE_methods_detail.tex:36-37` "under bf16 autocast with fp32 weights". | The 96-key refits and the Pile-text tasks are saved and read in bf16 (`X/refit_96key/results/predictions.json → settings.save_dtype`; `X/performed_anchor/run.py:174`). The other tasks' states are fp32. | Fig known, Fig 2 | S2 |
| B13 | `P/appE_methods_detail.tex:165` "We report one vector pooled over keys (Figure known)". | For Copy at a new marker and Lookup the steering vector is a different variant, fit on the even 100 read windows and read on the odd 100 with no gain floor (`X/steering_copy_lookup/run.py:48`; `known3` cell sources). The other 14 tasks fit on training windows. Cross-figure difference 10. | Fig known, steering panel | S2 |
| B14 | `P/appI_results_detail.tex:196-197` "on Swahili and Yoruba, rated new yet reading the highest ρ". | Yoruba is highest at 0.125. Swahili is fourth at 0.083, below SQuAD 0.112 and sentiment 0.085 (`X/convergence_panel/results/wide/run4/score_n10.json → per_task.*.E_b.mean`, checked here). | Fig kept_realistic text | S3 |
| B15 | `P/appD_built_pairs_more.tex:28-29` "Six never reach that KL, because the base model already predicts such text well". | Topic and Tagalog cross the KL on every seed, with held gain 4.8-7.6. Each misses only because one seed's saves straddle the ±10% window (`X/convergence_panel/results/wide/train/{topic,tagalog}.json`). | Rated-panel bookkeeping | S3 |
| B16 | `P/appD_built_pairs_more.tex:20` "The re-gating share reads lowercase above reversed order, as built, but not French above Yoruba". | French has no share at any seed or size, because its gain stays under one log-odds; the call is "unreadable". The sentence reads as though French were measured below Yoruba. The label pair is called the reverse of built, which is not mentioned. | Built realistic pairs | S3 |
| B17 | `P/appD_built_pairs_more.tex:18-19` "called when the side built to reuse reads higher than its partner by more than the larger side's seed range". | The code's rule is that the seed ranges do not overlap (`R/src/commute/realistic.py:243`), which is the rule `P/appE_methods_detail.tex:156-157` states. | Built realistic pairs | S3 |
| B18 | `P/appI_results_detail.tex:188-189` "leaves out the two label tasks, because their prior is in a different unit". | The label tasks' re-gating and steering shares are also in the label-logit unit (`X/convergence_panel/realistic_reads.py:68-73`). So the ten-task τ of 0.22 and −0.17 at `appI:187` mixes units. Cross-figure difference 5. | Fig realistic vs Fig kept_realistic | S3 |
| B19 | `P/appD_built_pairs_more.tex:26-28` "the save at which its next-token distribution has moved by the same KL from the base model's". | The KL is measured on Pile validation rows 0-63, not on the task's text (`X/convergence_panel/run.py:215-216`). Cross-figure difference 6. | Fig realistic, Fig kept_realistic | S3 |
| B20 | `P/appI_results_detail.tex:217` "built edits; copy, lookup — none valid"; `:246-247` "so no reading is valid". | The run also read 15 key-grid states. The three 96-key copy seeds gave valid readings, with edge IoU about 0.29 (`X/eapig_overlap/results/stage4.json`). | tab:tools, circuit overlap | S4 |
| B21 | `P/appI_results_detail.tex:228` "Ranges are task means over seeds"; row `:222` "randomly rotated updates 0.75–1.45". | The rotated range is over 48 single states (`weightwatch/results/p2-80/tools_fig4.json → .ww.sham_state_range`). Cross-figure difference 12. | tab:tools, WeightWatch | S4 |
| B22 | `P/appE_methods_detail.tex:184-185` "Each pair is reported at whichever size gives the larger gap between its sides". | Every pair's plotted value is the 64-neuron set (`X/known_3seeds/results/known3.json`, all `neuron_top` sources `neuron.k64.share`). | fig:known_appendix, top-neuron share | S4 |
| B23 | `P/appE_methods_detail.tex:211-212` "Our reading is the share of the fine-tune's activation change that falls on those latents" and "two passes over 100M tokens". | The plotted reading is d, the shared share minus the specific share (`src/commute/crosscoder.py:374-378`; `build_tools_known_csv.py:39-46`). The plot uses the 800M-token, 8-epoch run (`X/crosscoder_fig4/results/predictions.json → sizes.long`). `appI:237-241` states d and 800M correctly, so appE and appI disagree. | Fig tools_known, crosscoder | S4 |
| B24 | `P/sec5_results.tex:73` "on the crosscoder three random updates read above most fine-tunes"; `P/appI_results_detail.tex:242`. | The three Gaussians are matched in weight norm to one task's update and read on that task's windows. They change block-2 activations about 100 times less than any fine-tune: energy 0.017-0.022, against 1.94-14.3 (`X/crosscoder_fig4/results/long/sides/*.json → .energy`). Cross-figure difference 3. | Fig tools_known, crosscoder | S4 |
| B25 | `P/appE_methods_detail.tex:152-154` "Unless stated, it is read on the same rows as the re-gating share ... at the answer position". | The panels read different data. The crosscoder reads every position but 0 of its score windows (`crosscoder_sides.py:164-186`). Reverse fine-tuning reads full-distribution KL. Head patching reads 204-294 labelled definition windows at the last position (`X/cmap/job.py:290-305`). None of this is stated. Cross-figure difference 2. | Fig tools_known | S4 |
| B26 | `P/appG_departures.tex:32` "at these sizes the mask alone moves the base model's answer by 1.2 to 3.4 log-odds". | No saved file gives this range. On the 16 tasks the mask-alone shift is 0.30-3.00 at the 64-neuron size the figure reads, and 0.17-3.39 at the matched size, with the sign varying by task (`X/holes_fill/results/stage1.json`, `X/refit_96key/results/stage1.json`, `X/known_3seeds/results/stage2.json → …neuron.k64.N00_minus_T0`). The only text source is a review file. | Top-neuron share departure | S5 |
| B27 | `P/appH_qwen_short.tex:33` "a pair clears it when all three seeds' gaps have one sign and lie further from zero than the random gap". | The code adds a 0.00196 seed spread, measured on the other decomposition, to the random gap (`X/qwen_library_share/run.py:343-350`; `X/qwen_neuron_draws/plot.py:93`). Copy/cipher at the last save clears draw 3 by 0.00085 over that band. | Qwen neuron control | S5 |
| B28 | `P/sec7_discussion.tex:21` "across 39 trained states (23 for the two shares) ... when the 16 states with necessary sets under 1,000 variables are left out". | 38 states enter the K table, because one has an empty K (`X/claim2_needed/results/summary.json → .A_procedural.left_out`, `.trained.K.n_states` 38, checked here). After the drop the shares rest on 12 states and effective rank on 23 (`.trained_K_ge_1000.K`, checked here). The values 0.41, 0.46, −0.47, 0.18, −0.48 and −0.81 match. | Discussion, proxies against ρ | S5, S1 |
| B29 | `P/sec7_discussion.tex:13` "letting a necessary part's output move within the directions the base model already writes, but at VPD trained and random updates then read alike". | The pooled rotation share reads alike: fine-tunes 0.0003-0.0215, Gaussians 0.0015-0.0257. The "own" part, the share inside directions the base model already writes, has a median of 0.206 for fine-tunes against 0.075 for Gaussians (`X/rotation_kept/results/page.json → .classes`). The sentence's wording describes the "own" part. | Discussion, "definition too strict" | S5 |
| B30 | `P/appH_qwen_short.tex:28-29` "0.84 to 1.00 for 88 saved fine-tunes". | 15 of the 88 are unscored under the share's own rule; the scored ones read 0.89-1.00 (`X/random_kl_share/results/states/*.json → .own.kl_status`). | KL-form share on the 67M | S5 |

### Severity C

| # | Paper says | Code and saved files show | Sections |
|---|---|---|---|
| C1 | `P/appI_results_detail.tex:39` "all 94 states ... read ρ of 0.015 to 0.063". | The 94 read items include the same fine-tunes at saved and at rescaled size (`X/claim2_needed/results/items/*_scaled_*`). "States" is literally true. | S1 |
| C2 | `P/appI_results_detail.tex:79` "three random heads remove about 0.14"; `:132` "Dashed: three random heads off, or a random update". | The line is one value per panel, the mean of the copy and cipher corners' seed maxima, 0.108 and 0.164, of a null drawn outside the top 5 heads (`P/figures/src/make_figures.py:369-371`; `prior_mech_2x2_cued.csv`). The r and steering panels have no line. | S1, S2 |
| C3 | `P/sec4_setup.tex:19` "on six states it changed by at most 0.016 between 64 and 256 pairs". | Neither writer found a file or script that stores this comparison. | S2 |
| C4 | `P/appI_results_detail.tex:120` "the number of neurons needed for half the gain". | It is half of the MLP-input-only re-firing numerator, and none is counted unless that share exceeds 0.1 (`R/src/commute/neuron_r.py:195-237`). `appE:176-180` says "upstream form", nearer the code. | S2 |
| C5 | `P/appI_results_detail.tex:131-132` "the cipher values differ slightly from Figure known's". | Steering moves from 0.161 to 0.233 on Cipher 96 and from 0.143 to 0.172 on the prior-matched cipher; r moves from 0.300 to 0.272 (`prior_mech_2x2_cued.csv` against `known_main.csv`). | S2 |
| C6 | `P/sec5_results.tex:28` "at most 0.40"; `P/appI_results_detail.tex:137` error-share ranges. | The first is a task mean; the second gives seed extremes, while the reuse task means span 0.559-0.731. Neither says which. | S2 |
| C7 | `P/appE_methods_detail.tex:39` "over whole 512-token Pile rows". | Lowercase rows are decoded, lowercased, re-encoded and re-cut (`R/src/commute/row_copy.py:23-31`). | S2 |
| C8 | `P/appD_built_pairs_more.tex:15-16` "read at two update sizes matched within the pair". | ρ, reported in the next paragraph, was read at one size. | S3 |
| C9 | `P/appD_built_pairs_more.tex:21-22` "The label pair cannot be read at all, because the topic-label fine-tune's necessary set holds only three to six variables". | The topic-label run records ρ of 0.105, 0.063 and 0.024 with no rule on the size of K. The "unreadable" call borrows the panel's minimum of 100 variables. | S3 |
| C10 | `P/appI_results_detail.tex:195-196` "The gap between trained and random, 0.29, is not tested". | The score file holds a "survives" call at threshold 0.2403. Run 4's filing calls the gap descriptive, which the paper follows. | S3 |
| C11 | `P/appE_methods_detail.tex:149-150` "on the rebuilt edit's second MLP"; `P/appI_results_detail.tex:262` "deleting the circuit reads 287". | The SAE site is block index 2, the third MLP (`sae_train items/rebuilt_resid_pre0_s0+13.json → .task.mlp2_s0`). The "rebuilt" edit also replaces the layer 2-3 MLPs (`rho_saved_items/run.py:130-137`), so it does more than delete the circuit. | S4 |
| C12 | `P/appE_methods_detail.tex:148` "The attention SAEs leave 50 to 62% of their latents dead". | The file gives 53.0-61.6% (`X/sae_train/results/stage0_resid_pre0_s0+13.json → .base`). The reconciler did not re-read this. | S4 |
| C13 | `P/appI_results_detail.tex:220` head-patching ranges; `:258` "0.05 and 0.08". | The table's ranges leave out the 7 states that have no circuit; Ciphered text rests on one seed whose faithfulness is 0. The bullet's 0.05 and 0.08 count empty circuits as 0 (`cmap/score.py:45-60`). | S4 |
| C14 | `P/appI_results_detail.tex:244-245` "its control fails through our port". | An earlier port passed, 0.050 against 0.0296 (`X/adl/results/amend1/stage0.json → .passed`). | S4 |
| C15 | `P/appI_results_detail.tex:218` pruning ✗. | Pruning's own planted control passed (`X/open_pair_instruments/results/run.json → .stage0.planted.prune_pass`); the ✗ rests on the readout port, as `appI:249-250` says. | S4 |
| C16 | `P/appI_results_detail.tex:229` "mean over 13 sites (... the embedding ...)". | The embedding site's gap is identically 0. | S4 |
| C17 | `P/sec7_discussion.tex:9` "separates the tasks by how they were built but not from their shuffled twins". | One twin pair does separate: runner-up guess against its shuffle, +0.251, "apart". The class map reads "apart" the wrong way, and the two label twins make no call (`X/method_table/results/final.json`). `P/sec5_results.tex:28` already says "only one of four". | S5 |
| C18 | `P/appH_qwen_short.tex:23-24` "for each of 15 saved updates ... 0.97 to 1.00 times that of a Gaussian change of the same norms". | The 15 are not the figure's pairs; they are older Qwen updates (`X/qwen_instrument_check/step0.py:48-74`). The Gaussian is the analytic expectation, not a drawn change (`step0.py:159-164`). | S5 |
| C19 | `P/appH_qwen_short.tex:21-22` "Every fine-tune reads between 0.990 and 0.9998, and random changes ... 0.992 to 1.009". | The ranges pool the last and the five-times saves, while the figure plots the last save only (`P/figures/src/make_qwen_share.py:41-48`). At the last save the random minimum is 0.99311. | S5 |

### Dropped after re-checking the frozen text

- S2, reconstruction-gap sites. `P/appE_methods_detail.tex:192` and the caption at `P/appI_results_detail.tex:229` both say 13 sites.
- S5, `P/sec5_results.tex:85` "The definition itself has not yet been measured on Qwen". This is literally true. E270 summed ρ's expression over every variable, because no necessary set exists on Qwen (`X/qwen_eb/run.py:15-17,196-198`). It separated trained from random updates, and the paper leaves it out. `appH:3` says that paragraph was cut on 2026-11-26.
- S5, `P/appG_departures.tex:13-15` "measured through a next-token readout trained on Pile text". This sits in the Jain et al. item. It describes the port of their tools, whose readout `appI:249-250` names, not the ρ path.
- S5, `P/appG_departures.tex:33-34`, gain-only gains. This is true of gain-only training (`R/src/commute/gains.py:1-4`). The paper never calls the rescale-only control of A1 IA³.
- Dropped by the reconcilers after their own check, not re-checked here. In S1: the tab:built values, the SAE ranges and the K* bound. In S2: the random-change paragraph at `appE:23-28` and the twin steps. Also in S2: the runner-up guess at `sec4:43`, the prior ranges at `appI:145` and the share above 1 at `appE:120`. In S3: the p of 0.06 and the prior's 0.12, which recompute exactly, and "random updates of the same size". In S5: the Qwen values at `appH:23-26` and `:41-46`, the copy-head values and the partial correlations' values.

## Cross-figure table

This table and its list are copied from `cross_figure.md`, edited only where the re-checks above changed a fact.
Difference 10 now cites the paper sentence it contradicts, because the old quote "the same way" is not in the Fig
known caption. The S2 rows were built from the two S2 writers' records before S2_final existed. Their window counts
and readers agree with S2_final. The abbreviations are these:

| Short name | Meaning |
|---|---|
| `R`, `EXP`, `P` | the research repo, its experiments directory, the frozen paper |
| `CP` | `EXP/convergence_panel/` |
| `S` | `CP/results/wide/run4/score_n10.json` |
| `known3` | `EXP/known_3seeds/results/known3.json` |
| [16] | the 16 known-answer tasks, 8 a side. Reuse: e62/copy, e46/ident32, c3/copy, e58/copy, e53/lowercase, e54/elicit, c3/rank1, e58/rank1. New: e62/lookup, e46/rand32, e53/cipher, e54/cipher, e54/cipher_own, c3/cipher, e58/cipher, e58/prior_cipher |
| [48] | [16] at seeds 0, 1 and 2 |

| Row | Tasks | Seeds | States / steps read | Featurization(s) | Reference | Data read (inputs, positions, tokens) | Precision | Source run(s) / script |
|---|---|---|---|---|---|---|---|---|
| **Fig 2 definition** (`definition.pdf`) | Trained: 5 tasks, all 96 keys: e58 copy, rank1, elicit (reuse); cipher, prior_cipher (new). Four of these are in [16]. **e58/elicit (Runner-up guess, 96 keys) is a twin-control task and is not in [16]** (`EXP/refit_96key/results/large/dW_elicit_s*.pt`; `EXP/fig2_standard/results/items/vpd/trained__elicit_s*.json`; checked). Built: copy24 x2, x10, rebuilt, new_few, new_thin; induction x2, x10, rebuilt; bigram x2, x10, rebuilt, new_few, new_thin; 3 more cue draws (s186001-3) × new_few, new_thin (`P/figures/data/definition.csv` → block, checked) | 0, 1, 2 (trained) | E147 refits, read at step 30, unscaled, saved bf16 (`EXP/refit_96key/results/predictions.json` → read_step, save_dtype; `fig2_standard/feats.py:11`). Update size Σ‖F−M‖² 1.02-1.84 (E298 `_rand` item JSONs → size2_F) | Three panels: VPD (every subcomponent, no alive selection, + 24 remainders = 38,938 variables, `src/commute/kstar_vpd.py:91-93`); top neurons (6 heads + 3,072 neurons above M's p97 bound + 1 remainder per layer = 12,316, `src/commute/needed_neurons.py:1-46`); SAE (E234 base-only, 8 sites × (16,384 + error) = 131,080, `src/commute/sae_feat.py:1-21`) | One Gaussian per changed tensor at that tensor's ‖F−M‖, one draw per item: seed 161000+E290 index for trained states, 298000+i for edits (`EXP/needed_neurons/run.py:189-198`, checked; `fig2_standard/feats.py:80-86`) | Trained: **512 windows × 8 tokens, a shuffled 3:3:2 mix of task windows, does-not windows and far rows** (`src/commute/old_tasks.py:231-241`, checked; item JSON n_windows 512, window_len 8, checked). K read at the last position; ρ sums over all 8 positions (`src/commute/stand_in.py:130-147`). Copy24 x2/x10/rebuilt: 75 windows × 41, all 41 read positions. New edits, induction, bigram: 512 × 41, one position (`fig2_standard/results/items/vpd/built__rand_x2.json`, checked). K from 256 pairs (4 derangements × 64, seed 268, `src/commute/needed.py:38-50,162-204`) | fp32, TF32 off (`EXP/fig2_standard/run.py:253`, checked) | E298 (trained VPD seeds 1-2, copy s0, SAE, all randoms); **E290 for VPD s0 of rank1/elicit/cipher/prior_cipher and for every top-neuron trained row**; E268/E277/E280 built; E301 induction/bigram; E304 draws (`definition.csv` → source, checked). Script `P/figures/src/make_figures.py` fig_definition; the step that joined the three results CSVs into `definition.csv` is UNRECOVERED (S1_final) |
| **Fig 3 known** (`known_main.pdf`) | [16] (`known3`, checked) | 0, 1, 2. The seed-2 states of 5 tasks come from E299 (`EXP/known_3seeds/run.py:67-69`) | As saved, no size matching. Steps 8-300 by task (tab:pairs; S2_A/B agree with run files). The 4 e58 tasks are the same files as Fig 2 (`refit_96key/results/large/dW_*`). Size ‖dW‖/‖W0‖ 0.0049-0.0186 (`EXP/size_matched/run.py:129-130`) | Re-gating share: released VPD s-55ea3f9b, gates Γ(a0)/Γ(a1), remainder excluded (`src/commute/local_faith.py:256-278`, checked). Steering: none (pooled block-input vector). Effective rank: none (weights, `src/commute/update_spectrum.py:23-43`) | None drawn | Share: each task's E136 read rows (e58 286, c3 75, e46/e54 97, e62 200, e53 ≤500, S2_B and S5_final D2), **log-odds of the trained answer at one position** (`half`, `local_faith.py:217-219,236,251`, checked). Steering: fit on the first 512 training windows (`EXP/holes_fill/run.py:129-193`); **e62 only uses E135's `share_lo` variant, fit on the even half and read on the odd half of its 200 read windows** (`known3` → cells e62_copy_lookup steer source, checked) | Share bf16 autocast (`EXP/share_untrained/run.py:5`, checked); steering fp32 (S2_A); erank fp64 on weights (S2_A) | E136, E143, E135, E141/E171, E147, E299; assembled by E303 into `known3`. Script `P/figures/src/make_known_scorecard.py:185` |
| **Fig 4 realistic** (`realistic_mirror.pdf`) | 8: conll, french, lower, spanish, squad, summary, swahili, yoruba (labels category and sentiment left out) (`P/figures/src/make_realistic_mirror.py:52-59`) | 0, 1, 2 (data order only, `CP/panel.py:220-224`) | KL-matched save: KL 0.181 ±10% from W0, measured on **Pile validation rows 0-63, not task text** (`CP/run.py:215-216`; `W/selection.json`); steps 3-400; spanish s0 = step 400 of a separate 1,200-step L40S retrain (gain-1 rule, `CP/score_run4.py:33-41`). Size 0.0025-0.0243 (`S → per_task.<t>.size`). Recipe: Adam 3e-5, batch 8, masked CE, no KL term, ≤300 steps (`CP/results/predictions.json → settings`) | Share: released VPD + Γ (`CP/share.py:50-79`); steering, erank: none | Chance (permutation, seed 266); prior (not plotted) | 500 read rows per task. LM tasks: 127-token windows read at position 63, the 500 word starts of lowest W0 log-odds (`realistic.py:197-212`; `panel.py:182-197`); prompt tasks: position P−1, first answer token (`panel.py:163-166`). Steering fit: LM 2,000 windows from train rows 0-255, prompt train examples 2400-4399 | Share bf16 (`realistic_reads.py:126-127`; S3_final A5); states bf16 (`CP/run.py:282-286`); training fp32 weights, bf16 autocast forward (`finetune.py:61-64`) | E272 (train, cluster L40), E307 (score); `S`. Script `P/figures/src/make_realistic_mirror.py` |
| **Fig 5 kept_realistic** | 10: Fig 4's 8 + category, sentiment (`CP/score_run4.py:26-28`) | 0, 1, 2 | As Fig 4, except **sentiment at the gain-1 save, steps 108/21/119, KL 0.95-4.33 = 5.2-24× the target** (`S → not_kl_matched`) | VPD only: 38,938 subcomponents + remainders (`W/eb/<t>.json → n_vars`) | Gaussian per matrix at the state's per-matrix norm, 3 draws, generator seed 0 + seed index, **so the same three directions serve every task** (`CP/eb.py:215-226`, checked; `src/commute/finetune.py:50-58`, checked). Not KL-matched | **First 200 read windows**, cut to end at the read position (LM 64 tokens, squad 256, conll 96, summary 128, labels 35; `W/eb/windows_manifest.json`); designated output F's argmax at the last position; K from 256 pairs, seed 268. |K| 200-885 on conll s0/s2, 209 on category s1 (`W/eb/{category,conll}.json → size_K`) | fp32, TF32 off (`CP/eb.py:236`, checked); states and randoms bf16 (`eb.py:225`, checked) | E272/E302/E307 (H100 pods); `S → hope1.E_b`, `E_b_random`. Script `P/figures/src/make_kept_realistic.py` |
| **Fig 6 tools_known** | [16]; crosscoder grey = 3 random updates (`P/figures/data/tools_known.csv`: 17 crosscoder rows, checked) | 0, 1, 2; crosscoder seed 0 (`S4_final` S4-04) | Share, crosscoder, reverse FT: [48] as saved. **Head patching (CMAP): the larger side of each pair scaled to its partner's size, c 0.52-0.98 on seven built-new tasks and 0.9975 on Top guess** (`EXP/cmap/job.py:110-130,275-276`; `EXP/size_matched/results/m67.json`; S4_final verified). 7 states have no circuit (lowercase s0-2, c3/rank1_s0, e58/rank1_s1, e53/cipher s1-s2) | Share: as Fig 3 (same values: `tools_known.csv` = `known_main.csv` re-gating rows, checked). CMAP: attention heads (128-wide pre-W_O slices), circuit by path patching. Crosscoder: BatchTopK at block-2 input, expansion 32, k 100, 800M tokens. Reverse FT: none (`src/commute/head_cmap.py:204-273`; `src/commute/crosscoder.py:303-378`; `src/commute/revival_kv.py:58-174`) | CMAP: none drawn (CSV null_mean = 10 random head sets). Crosscoder: **3 Gaussians at e54/elicit s0-2 per-matrix norms, seeds 1981-1983, read on e54/elicit windows, one reference for all 16 tasks**; their block-2 activation energy is 0.017-0.022 against 1.94-14.3 for fine-tunes (`EXP/crosscoder_fig4/results/long/sides/*.json .energy`). Reverse FT: Haar-rotated null, not drawn | CMAP: the 512 definition read windows reduced to labelled rows (204-294), last position (`EXP/cmap/job.py:290-305`). Crosscoder: score windows (odd rows), every position except 0 (`crosscoder_sides.py:160-197`). Reverse FT: full next-token KL at the read position on held-out odd rows (`revival_kv.py:145-174`). Share: as Fig 3 | Crosscoder fp32 (`crosscoder_fig4/results/predictions.json → sizes.long`); CMAP logits fp32 at the read (`head_cmap.py:233`); reverse FT: UNRECOVERED (no dtype set in `reverse_ft_fig4/run.py`, grep) | E288/E305 (CMAP), E313 (crosscoder), E312 (reverse FT), known3 (share). Script `P/figures/src/build_tools_known_csv.py`, `make_tools_known.py` |
| **tab:built** (`appI:15-37`) | 5 copy24 edits (x2, x10, rebuilt, new_few, new_thin), do-nothing, 2 shared randoms | Randoms 241001, 241002 | Edits built in code (`EXP/kstar_levels/run.py:208-222`; `src/commute/known_edits.py:1-24`) | VPD only, as Fig 2 | random_size_241001 at new_thin's total size, read on x2's windows; random_kl_241002 KL-matched to new_few, read on new_few's windows (`EXP/kstar_vpd_wording_a/run.py:166-191`; its predictions settings) | x2/x10/rebuilt: 75 × 41, all positions; new edits: 512 cue × 41, last position | fp32 (as Fig 2) | E268 (`EXP/kstar_needed/results/summary.json → items.*.E_b_K`); the same rows as Fig 2's VPD built points (`definition.csv` source E268, checked) |
| **tab:tools** (`appI:208-231`) | Mixed by row: [16] (crosscoder, reverse FT, CMAP, CMAP at neurons, WeightWatch, reconstruction gap); built edits + e62 copy/lookup + 5 e58 key tasks scaled (circuit overlap); one built pair (pruning); controls only (ADL, Delta-Crosscoder, Diff-SAE) | 0, 1, 2 for [48] rows; others by row (S4_final S4-06..16) | [48] unscaled, except CMAP (scaled, as Fig 6) and circuit overlap's key states (scaled, E237 c) | Per row (S4_final): crosscoder d; heads; MLP neurons (CMAP at neurons, `src/commute/neuron_r.py:15-21,133-144`); weights + base moments (WeightWatch); E234 SAEs at 13 sites (gap, `EXP/sae_proxies/run.py:23-30`) | Crosscoder 3 Gaussians (as Fig 6); WeightWatch orientation sham, **range over 48 single states**, not task means (`weightwatch/results/p2-80/tools_fig4.json .ww.sham_state_range`); gap: one Gaussian per state at its per-matrix norms | Per row: crosscoder as Fig 6; CMAP at neurons on E136 rows at the answer position; gap on each state's E136 read rows, every position but 0 (embedding site's gap identically 0) | Per row; mostly UNRECOVERED beyond crosscoder fp32 | E313, E312, E288/E305, E98, E306, E314, E287, E284, open_pair/jain_readout; `S4_final` |
| **tab:pairs** (`appE:56-108`) | [16] + 8 control rows (runner-up guess 96 keys, class map, case label, repeat label, each with its shuffle) | Counts per row ("fine-tunes" = states read at VPD) | Steps read per row; S2_A and S2_B both match run files (`EXP/induction_cue/results/run.json`; `open_pair/results/run.json`, `amend1.json`; `keydependent_*`; `refit_96key`; `known_3seeds/results/stage1.json`) | n/a | n/a | n/a | n/a | Listing only; footnote counts 68/94/20 rest on S1 items (S1_final S1-15, S1-17) |
| **fig known_appendix** | [16] | 0, 1, 2 (prior: one value per task, repeated across seeds; early and late cipher share the e54_cipher prior) (`known3` → prior source text, checked) | As Fig 3 | **Frozen figure has 5 panels**: neuron patching (all 12,288 MLP neurons, `neuron_r.py:1-30`), top-neuron share (64-neuron set, `holes_fill … neuron.k64.share`), neurons for half the gain, update size, base prior (`P/figures/data/known_appendix.csv`: 80 rows, 5 readings, checked). The error-term share, reconstruction gap and grey random crosses that `make_known_scorecard.py:186` would add are not in the frozen CSV or PDF | None drawn in the frozen figure | Neuron patching: E98 held-out windows at the answer position; prior: share's read windows (`EXP/prior_vs_share/run.py:1-10,137`) | Neuron patching fp32 (S2_A, S2_B) | E98, E143/holes_fill, k_half_fill, E187, E133/E145, E299 via `known3`; script `make_known_scorecard.py:130-186` |
| **fig prior_mechanism** (`prior_mech_2x2_cued.pdf`) | 4 corners: e58/copy, **cued copy (E253: 32 keys, cue windows, `EXP/cued_copy/results/predictions.json → settings.n_keys 32`, checked)**, e58/prior_cipher, e58/cipher. CSV also has lowcopy, not drawn | 0, 1, 2 | Step 30; **every update scaled along its direction to copy's mean size 0.0073022** (`cued_copy/results/table.json → proxies.size.corner_mean`, checked) | VPD (r); heads (copy heads off, top-3 heads mean-ablated, `src/commute/copying.py:161-175,468-472`); weights (erank); steering | Copy heads off: 3 random heads outside the top 5, seeded draws; erank: random updates of the same size (~697-706); none for r and steering (`P/figures/data/prior_mech_2x2_cued.csv`, checked) | e58 corners: 286 held-out rows; cued copy: up to 400 read rows of cue windows (`settings.n_read_cap 400`, checked); answer position | fp32, TF32 off for the new readings (`EXP/cued_copy/run.py:365`; `prior_mech_2x2/khead.py:269`, checked). The copy corner's r equals Fig 3's bf16 value to 16 digits (`prior_mech_2x2_cued.csv` vs `known_main.csv`, checked) | E147, E237, E250, E252, E253; script `P/figures/src/make_figures.py:353-381` |
| **fig qwen_share** (`qwen_share.pdf`) | 6 Qwen3-1.7B sides: copy/cipher, known/invented, gold/gold_shuf | 0, 1, 2 | Last saved step; built pairs lr 1e-5, warm-up 30, 300 steps, batch 64; gold pair 100 steps, warm-up 10; sizes not matched, ‖dW‖ 1.78-5.09 (`EXP/qwen_library_share/results/library_run3/*.json .recipe`; S5_final Q1) | Panel 1: VPD p-9707d6ff, 196 matrices; panel 2: neuron control p-015e0710, 84 MLP matrices, attention dW zeroed (`EXP/qwen_library_share/run.py:170-173`) | One Gaussian per side at seed-0 last per-matrix norms, KL-matched within 1.5× on 16 rows; draw 0 (`qwen_library_share/run.py:258-277`) | Each side's own 128 packed rows × 256 tokens, **every position**; **KL form** (K00−K01)/(K00−K11) (`src/commute/qwen_reading.py:52-65`) | Model fp32 (`EXP/qwen_reading/lab.py:189`), Γ bf16 (`lab.py:204`), dW bf16 safetensors (`lab.py:94`) | E170/E174, E179; script `P/figures/src/make_qwen_share.py` |

### Differences (where the paper compares or juxtaposes two rows)

1. **Fig 2 vs Fig 3: different tasks and different windows for "the same trained fine-tunes".** The paper says
   "on the same trained fine-tunes the retained share does not follow them" (`P/sec5_results.tex:28`). Fig 2 reads 5
   tasks, of which 4 are among Fig 3's 16. The fifth, e58/elicit, is a twin-control task (tab:pairs row
   `appE:87`). Fig 2 reads its trained states on 512 eight-token windows that are 3/8 task windows, 3/8 does-not
   windows and 2/8 far rows. It sums ρ over all 8 positions (`old_tasks.py:231-241`; `stand_in.py:130-147`), in fp32.
   Fig 3's share reads the task's held-out rows (286 for e58) at the answer position only, under bf16 autocast. The
   state files are the same for the 4 shared tasks. Reason given: the task count is stated (`P/sec5_results.tex:17,21`
   "the five 96-key synthetic tasks"; rerun on 16 is E325). No reason is given for the different windows and positions.
2. **Fig 6: head patching reads scaled weight changes, its neighbours read them as saved.** CMAP reads W0 + c·δW with
   c 0.52-0.98 on seven built-new tasks (`EXP/cmap/job.py:110-130,275-276`). The re-gating share, crosscoder and reverse
   FT panels read δW as saved. The paper says "Every reading is taken on the same saved weight changes $\delta W$"
   (`P/appE_methods_detail.tex:152`) and "the 48 fine-tunes of Fig.~\ref{fig:known}" (`P/appI_results_detail.tex:220`).
   Matched size is mentioned only for "a first run" (`appI:258`). No reason is given in the caption (`sec5_results.tex:78`).
   The panels also read different data: CMAP uses 204-294 labelled definition windows at the last position. The
   crosscoder uses every position but 0 of the score windows. Reverse FT uses full-distribution KL at the read position.
   The share uses E136 rows at the answer position. No reason given.
3. **Fig 6 crosscoder random reference vs the fine-tunes beside it.** The paper says "three random updates read above
   most fine-tunes" (`P/sec5_results.tex:73`; `appI:242`). The three updates are matched in weight norm to one task,
   e54/elicit, and read on its windows. They change block-2 activations about 100× less than any fine-tune (energy
   0.017-0.022 against 1.94-14.3, `crosscoder_fig4/results/long/sides/*.json`). No reason given.
4. **Random references differ from figure to figure, and none matches the one the paper describes.** The paper's
   description (`P/appE_methods_detail.tex:23-25`): "a Gaussian direction at the fine-tune's norm for that matrix ...
   scaled until its KL ... is within 10% of the fine-tune's own. Three draws per fine-tune." What each figure uses:
   - Fig 2: one norm-matched draw per state, no KL scaling (`needed_neurons/run.py:189-198`).
   - Fig 5: three norm-matched draws, with the same three directions for every task (`CP/eb.py:215-226`).
   - Fig 6: three draws at one task's norms (difference 3).
   - fig qwen_share: one draw, KL-matched within 1.5× (`qwen_library_share/run.py:258-277`).
   - tab:built: two shared draws.

   The captions say "random updates of matched size" (`sec5_results.tex:21`) and "random updates of the same size"
   (`sec5_results.tex:65`), which fits Figs 2 and 5. The appE description is not stated as applying only elsewhere. Fig 5's
   shared directions are not stated.
5. **Fig 4 vs Fig 5: 8 tasks vs 10.** Reason given: "Figure~\ref{fig:realistic} leaves out the two label tasks, because
   their prior is in a different unit" (`P/appI_results_detail.tex:188-189`; `sec5_results.tex:47`). In code, the label
   tasks' re-gating and steering shares are also in the contrast unit (`CP/realistic_reads.py:68-73`), so the reason
   holds for every reading, not just the prior. Data also differ: Fig 4 uses 500 read rows at one position, bf16 share.
   Fig 5 uses the first 200 windows, fp32, with ρ summed over each window's positions. Sentiment, read only in Fig 5,
   is at KL 5.2-24× the panel's target. The caption gives the gain-1 rule (`sec5_results.tex:65`) but not that the KL
   match is lost.
6. **Fig 3 vs Fig 4 ("The proxies of Figure~\ref{fig:known} on eight realistic fine-tunes", `sec5_results.tex:53`).**
   The two figures differ in five ways:
   - Recipe: Fig 3 uses batch 32 with a KL term of weight 1, read at a fixed step (10/30/300). Fig 4 uses batch 8 with
     masked CE and no KL term.
   - Save read: Fig 4 reads the save at matched KL, 3-400 steps. That KL is measured on Pile rows, not task text
     (`CP/run.py:215-216`).
   - Read positions: Fig 3 reads the answer position. Fig 4 reads the hardest W0 word starts (LM tasks) or P−1.
   - Steering fit windows differ.
   - Size range: 0.0049-0.0186 in Fig 3, 0.0025-0.0243 in Fig 4.

   The paper states the KL match (`P/appD_built_pairs_more.tex:26-28`) but not that the KL is measured on Pile text. No
   reason is given for the different positions.
7. **Fig 2 vs Fig 5 (ρ on known-answer vs realistic fine-tunes, both "our approximation of the definition",
   `sec5_results.tex:59`).** The data differ: 512 × 8 mixed windows for Fig 2, against the first 200 task windows of
   35-256 tokens for Fig 5. Fig 2 has three featurizations, Fig 5 VPD only. The random references differ (difference 4).
   Several Fig 5 states rest on |K| of 200-885 against ~10k-36k elsewhere. No reason given.
8. **fig prior_mechanism vs Fig 3.** Every update is scaled to 0.0073. Reason given: "Every update is scaled along its
   own direction to the copy task's size, so the cipher values differ slightly from Figure~\ref{fig:known}'s"
   (`P/appI_results_detail.tex:131-132`). Steering moves by up to 0.07: cipher 0.233 here against 0.161 in Fig 3
   (`prior_mech_2x2_cued.csv`; `known_main.csv`). The caption calls the corners "Four tasks of one family"
   (`appI:129-130`). In code the cued copy is a 32-key cue-window task with up to 400 read rows; the other three are
   96-key e58 tasks on 286 rows (`cued_copy/results/predictions.json → settings`). Precision is mixed too: the copy
   corner's r is Fig 3's bf16 value, while the new readings run in fp32. No reason given for the family or precision.
9. **fig known_appendix vs its caption and `sec5_results.tex:73`.** The caption says "Six further readings ... the
   error-term share, the reconstruction gap (grey crosses: a random update ...)" (`P/appI_results_detail.tex:118-121`).
   The body says "The reconstruction gap is in Figure~\ref{fig:known_appendix}" (`sec5_results.tex:73`). The frozen
   CSV and PDF hold 5 readings and no random crosses (`P/figures/data/known_appendix.csv`). The rendered figure does
   not hold what the text points to. No reason given.
10. **Within Fig 3, steering for one pair is a different variant.** e62 copy/lookup uses E135's `share_lo`, fit and
    read on halves of the 200 read windows with no floor. The other 14 tasks use E143's pooled share, fit on 512
    training windows (`known3` → cells steer source). The paper says "We report one vector pooled over keys
    (Figure~\ref{fig:known})" (`P/appE_methods_detail.tex:165`). No reason given in the paper; the code comment says "e62 has no E143 pooled reading"
    (`make_known_scorecard.py:5-6`).
11. **fig qwen_share vs the 67M share (`sec5_results.tex:85`, "the re-gating share breaks down").** Qwen uses the KL
    form over every position of 128 × 256-token rows, sides not size-matched. The 67M uses log-odds at the answer
    position. Reasons given: "On Qwen the log-odds form could not be scored" (`P/appE_methods_detail.tex:199`); "The
    two sides of a pair are shown at their saved sizes, not matched in update size" (`P/appH_qwen_short.tex:17`).
12. **tab:tools ranges vs caption.** The caption says "Ranges are task means over seeds" (`P/appI_results_detail.tex:228`).
    WeightWatch's "randomly rotated updates 0.75-1.45" is a range over 48 single states (`tools_fig4.json
    .ww.sham_state_range`). The CMAP ranges leave out the 7 states with no circuit; Ciphered text rests on one state
    whose circuit faithfulness is 0 (S4_final S4-03). No reason given.
13. **Fig 2's built strip pools reads at different positions.** Copy24 x2/x10/rebuilt are read at all 41 positions of
    75 windows. Every other built edit is read at one position of 512 windows (item JSONs → n_read_positions). The
    pooling itself is stated (`P/appE_methods_detail.tex:20-22`); the positions are not.

## Record

The full record is `reports/reviews/2026-11-27_review_measurement-record.tsv`. It has one row per item, 180 rows. Its
first column is the section, and the other columns are the writer brief's header plus `settled_note`. The per-item
prose is in `S1_final.md` to `S5_final.md` in the measurement-record directory.

| Section | Paper range | Items | Disagreements settled | Unsettled | UNRECOVERED fields |
|---|---|---|---|---|---|
| S1 retained share | `sec5:12-25`, `appI:1-110`, `main.tex:366-439` | 38 | 7 | 2 | The step that joined three results files into `definition.csv`. The task pairing behind "20 fine-tunes of ten tasks" and its recipe. Which 12 repeat states enter the correlations. The builder of the copy-heads CSV. |
| S2 known-answer proxies | `sec5:26-42`, `appI:111-174`, `appE:1-142`, `sec4` | 37 | 13 | 0 | The "0.016 on six states" comparison. E136's own window counts, and the 96-key cipher counts. The label twins' recipes. The pod IDs of three runs. |
| S3 realistic fine-tunes | `sec5:43-70`, `appI:175-200`, `appD` | 20 | 10 | 3 | The premise-check files of the built realistic pairs. Which of two French and Yoruba training jobs wrote the saved states. The "base predicts well" reason, never computed. The eight-task p and prior τ, in no file but recomputed exactly. One job ID. |
| S4 model-diffing tools | `sec5:71-82`, `appI:201-268`, `appE:143-214` | 39 | 12 | 0 | The tools' training text and window counts. The checkpoint of the copy-at-a-marker pair. The E-codes and pods of Delta-Crosscoder and Diff-SAE. Pruning's margin multiple. Reverse fine-tuning's precision. The attribution ranking rule. How the KL-form sums are formed. |
| S5 Qwen and summary claims | `sec5:1-11,83-85`, `appH`, `main.tex:1-137,278-281`, `sec7`, `appG` | 46 | 11 | 2 | The source of "1.2 to 3.4 log-odds". The training recipe of Γ on the neuron control. The recipes of the 15 and the 88 saved updates. The builder of the induction and cue-draw rows. How the rater tiers were derived from the rater files. Which E-code trained the repeat states. |

## Not run

- No code was run. My re-checks were reads of the frozen paper text, plus `jq`, `awk` and `sed` reads of the files
  marked "checked here". Every other code fact in the mismatch list is the reconciler's, with its citation.
- The S4 writers could not run Python because the CPU worker timed out. Their ranges were recomputed with `awk` and
  `jq` only.
- The UNRECOVERED fields in the Record table were not found by any writer, and I did not search further.
- Items the reconcilers flagged as not re-checked: the attention SAEs' dead share and the neuron-patching function
  citation in S4. In S5: the KL scoring of pruning and reverse fine-tuning, the steering-vector source, the copy-head
  split and the row-by-row source mix of Fig 2's CSV.
- The cross-figure table's S2 rows rest on the two S2 writers plus the checks it marks, not on a fresh read of
  S2_final.
- Cross-figure difference 4 is not graded in the mismatch list. It says no figure uses the random reference that
  `P/appE_methods_detail.tex:23-25` describes. I did not check whether any appendix reading uses that reference.
- Fig 2 is recorded as it stands, on five trained tasks. Its rerun on 16 tasks, E325, is not recorded.
- The unrendered files were not checked: `sec5_baselines_scorecard_caveats.tex` and everything under `cut/`.
- Body sentences outside the five section ranges in `SECTIONS.md` were not recorded.
