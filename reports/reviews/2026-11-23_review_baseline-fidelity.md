# Fidelity audit of every literature method the paper reports (redteam-lead, 2026-11-23)

For whip-lead, on the user's concern of 11-23: "im really concerned now about the legitimacy of our baseline experiments".

**Plain answer.** Most of our baselines run the published code or formula faithfully, but the paper labels our own readings of them as if they were the published methods' results, in several places. One finding changes a reported result. Our crosscoder omits Minder et al.'s exclusion of latents with negative base coefficients. With the exclusion added, the block-3 planted control fails and the size ladder's 0.45 plateau disappears. This was recomputed independently from the saved files, and the fix is a CPU rescore. Everything else is a relabel, a disclosure, or a rescore of saved values. No GPU re-run is needed to make the paper honest. Two optional re-runs would strengthen it: the top-neuron share at Goodfire's sparsity, and a faithful ADL.

**Method of the audit.** Each method got two reviews: an Opus auditor (paper + code + our text) and an independent GPT adversarial reviewer, given only the code paths and the source paper, not our conclusions. Severity: **B** = blocker (the label or a reported result is wrong); **M** = major (changes a reading or how it is read); **m** = minor (disclosure only). "Ours" = we departed from the source; "theirs" = inherent in the published method. "Agree" says whether both reviewers found it; "Opus only" / "GPT only" otherwise. Per-review files (row-level evidence, file:line): `~/commute-data/agents/redteam-lead/fidelity/<method>_{opus,gpt}.md`. Paper paths are in `~/commute-agents/writeup-lead/paper/paper/`; code paths are in the redteam-lead clone at e894f1ed.

## Verdict per method

| Method | Can the paper use the label and citation? | Worst finding | Fix type |
|---|---|---|---|
| Crosscoder (Minder 2025 BatchTopK + latent scaling; E203/E235/E243) | The trained crosscoder: yes (pinned trainer called, latent scaling matched their Gemma release). Its "fine-tune-specific latents": no, until the negative-beta filter is added. The share and its 0.5 bar: ours | **B**: the filter is missing; the block-3 control and ladder readings reverse | CPU rescore of all saved crosscoders + relabel |
| ADL (Minder 2025 narrow; E264-267) | No. The citation is now right, but the score is our intervention score; none of their semantic lens evaluations ran | **B**: sec4_setup.tex:39 names it "the activation difference lens of" | relabel + define our quantity in App. E/G; faithful run optional |
| Neuron share, half form (Vig 2020) | No as "activation patching (Vig)". Yes as "neuron-level CMAP, adapted from Prakash 2024 §6.1" | **B**: wrong method label | relabel |
| Neurons for half the gain (Syed 2023) | "Attribution patching" is acceptable with Nanda + Syed; not EAP. "Fewest" is false | **B** (GPT) / **M** (Opus): label + "fewest" | relabel: "first power-of-two prefix of one signed attribution order reaching half the upstream-patch gain" |
| Top-neuron share (Goodfire) | Only "inspired by Goodfire's per-token top-K rule"; not CMAP | **M**: 64/block (2.1%) or 3-30 vs Goodfire's 7.8%; masking alone moves the base 1.2-3.4 log-odds | disclose; optional re-run at 240/block |
| Top-neuron overlap (Goodfire) | "Our construction using Goodfire's top-K rule" | **M**: its positive control failed (0.962 planted vs 0.965 random), undisclosed | disclose or cut the row |
| Steering (Turner, Rimsky, Venhoff) | No for Venhoff. "After" ActAdd and CAA only | **B**: "This is the test of Venhoff" (main.tex:930); **B** (GPT): a strength-1 caption on a tuned figure | relabel; regenerate or relabel the figure |
| Effective rank (Roy & Vetterli 2007) | Yes. The formula is exact; the weighting is ours and disclosed | **M**: the paper's rationale ("new needs higher rank") points opposite to the pairs it counts as agreeing | rewrite the claim |
| Update size | Yes; no citation needed | **M**: three norms (one squared) under one label; under one norm the tally is 0/7/2, not 0/6/3, from one call that is not substantive (both reviewers, after the follow-up) | rescore from saved norms |
| SAE featurization (BatchTopK, SAEBench recipe; E245/E251/E263) | Yes for BatchTopK (Bussmann); SAEBench must be cited; Bricken/Cunningham are misused at main.tex:194-195 | **B** (GPT) / **M** (Opus): wrong sites in App. E ("before and after each MLP"; the code uses attention and MLP outputs) | relabel + cite + disclose |
| Jain 2023 tests (E55 PCFG, E59 readout, 67M ports) | The 3M PCFG follows their recipe closely (a partial Table 6). The 67M "tests" and "their linear readout": no | **B**: 67M reverse FT and readout mislabelled; **B** (GPT): App. G misstates Def. 2; the 21/21 is our task comparison | relabel as "adapted tests"; report E59 as unreadable |
| Other §2 methods (Prakash, Lee, Panigrahi, Geiger, Sutter, VPD) | Citations correct | **M**: Prakash's patch direction reversed (main.tex:377-380); Sutter made unconditional (main.tex:217-218); App. G says mean ablation where the body uses interchange; sec4_setup.tex:16-18 gives VPD's total component count as "each matrix" | wording |

## Departures that change a reported number or claim (B and M), with fixes

| # | Method | Departure (ours/theirs) | Consequence | Sev | Agree | Fix |
|---|---|---|---|---|---|---|
| 1 | Crosscoder | ours: no exclusion of latents with negative base beta (Minder App. E.2) in the 67M classifier (`src/commute/crosscoder.py` `specific()`); the Gemma check does apply it | Recomputed (`fidelity/recompute/negbeta_filter.py`, matches GPT on all 9 numbers): block-3 control .56/.63/.52 → .44/.43/.42, a fail; E243 rungs 4-5 .45/.45/.46 → ≤ .01; 87-93% of our "specific" latents fail their test. main.tex:968-969 and appD:90-92 ("block 3 passed …11.30 nats") are false; block-3 Gaussian and pair readings have no passing control | B | yes (Opus first minor, re-rated after the recompute) | rescore every saved crosscoder (E203's 25 sides, E212, E235, E243, E244), regenerate the ladder figure and captions, keep the old values only as "unfiltered variant" |
| 2 | Crosscoder | ours: the share of the change on specific latents, d, the planted control and its 0.5 bar are ours, but are reported as the crosscoder's own outcome (sec5_results.tex:161-165, ladder captions, sec5_3, sec7, main.tex:53,125,681, appD, appF) | "cannot tell reuse from new" overclaims; by their own output (count of specific latents) it does respond to the plant: 0, 2, 26, ~126, 21-23, 27 after the filter | M | yes | "our reading of the crosscoder of Minder et al." wherever a number appears; report their count beside |
| 3 | Crosscoder | ours: latent scaling fit on 100k tokens of one shard (their 50M); "ported line for line" is false (algebraically equivalent); training recipe a toolkit/paper hybrid (anneal, RMS norm); exposures with replacement; E203 token count mis-recorded | sparse latents may be missed; metadata wrong | M | GPT M, Opus m | state each setting's source; fix the metadata |
| 4 | Crosscoder | ours: the "fine-tune size" axis mixes token distributions (graft on Pile crops vs fine-tunes on task rows): 82-90x on common Pile vs 71-74x vs the draft's ~104x | the ladder's x-axis is off | M | GPT only | use the same windows; give the achieved coordinates |
| 5 | Crosscoder | ours: the Gaussian matched the fine-tune's weight norms, but its activation change was ~1/36 of the fine-tune's; "0.73 shared" is shared minus specific; the 0.45 bar is ours | §5 Gaussian sentence wrong | M | Opus only | rewrite |
| 6 | Crosscoder | theirs: specificity is not mechanistic novelty (Minder §5) | "shared = reuse" is never their claim | M | yes | quote their §5 at the result, not only in Related Work |
| 7 | ADL | ours: a different estimand (answer-token log-odds share on task windows, max over 5 directions) vs their CE on fine-tuning and pretraining text; no Patchscope, steering or agent evaluation; Pile with no BOS; a different null; copy failed stage 0 | "Existing diffing tools cannot see reuse" is too strong; supports only "our score agrees in sign with the write-share gap on 8/9, gives new the higher score on 7/9" | B (label) / M | yes | relabel at sec4_setup.tex:39; define our quantity in App. E/G; remove the visible old crosscoder misattribution |
| 8 | Neuron share | ours: a whole-width cross-model transplant of all post-GELU firing, read as a log-odds share; Vig is within-model input mediation | wrong method label | B | yes | "neuron-level CMAP, adapted from Prakash et al. §6.1" |
| 9 | Neuron family | ours: appE_methods_detail.tex:92 and sec5_results.tex:170 put all four neuron readings under "CMAP" | the top share patches no values and the overlap intervenes on nothing | B | yes | CMAP for the share, half form and count only |
| 10 | Neuron share | ours: sec4_setup.tex:52 calls these existing model-diffing methods, "[cite?]" | unsupported claim | B | Opus only | cut or cite |
| 11 | Half-gain count | ours: signed node attribution (not EAP's absolute edge scores); prefixes of one order on a power-of-two grid; ranking and scoring on the same rows; half of the upstream-patch numerator | "the fewest" is false; counts are grid steps | B/M | yes | rename; state the grid, clamping, omission; cite Nanda 2023 + Syed |
| 12 | Top-neuron share | ours: 64/block or 3-30, vs 7.8%; frozen masks from unmasked forwards; masking alone moves the base 1.2-3.4 log-odds; the pair keeps whichever budget gives the wider absolute gap (undisclosed selection) | may explain the "< 0.04 of the gain"; calls can change | M | yes | disclose; optional re-run at 240/block |
| 13 | Top-neuron overlap | ours: non-interventional layerwise Jaccard at the answer position; the planted control failed (0.9623 vs 0.9652); E155's own rule reads 2/0/5 vs the paper's 3/1/3 | an unvalidated reading | M | yes | disclose the failed control or cut; do not quote E155 calls |
| 14 | Percentile featurization | ours: appG_departures.tex:41-46 conflates E217 (two-tail p3/p97, pooled Jaccard) and E229 (upper p97) with the fixed-K baselines | wrong method description | M | yes | separate the three |
| 15 | Steering | ours: mean cross-model difference at one block/position, searched strength, log-odds; Venhoff learns category vectors with SAEs and a live thinking model, scored on benchmark accuracy | "the test of Venhoff" is false (main.tex:930, appE:90-91); sec4_setup.tex:27's "every block" is wrong | B | yes | "our mean-difference steering share; borrows the gap-recovery comparison from Venhoff et al. (version pinned)" |
| 16 | Steering | ours: the `share_prior_steering` caption says strength 1, but the points are tuned (equal-size mid/cipher plotted −0.237 vs +0.038 at strength 1; tuned Spearman −0.30 vs strength-1 −0.10; the all-nine prior+steering claim is the strength-1 analysis) | the figure and its statistic come from different settings | B | yes | relabel or regenerate from strength-1 data |
| 17 | Steering | ours: three-level selection; the last picks the larger absolute gap of pooled/per-key on the read rows. Tallies: as reported 4/3/2, pooled only 5/2/2, per-key only 1/3/5; "twin higher on both label twins" rests on per-key alone; main.tex:1015-1016 "can only favour that reading" is wrong; pooled at strength 1 reverses 0 of 9 | the reported reversals rest on the selection | M | yes | report pooled and per-key separately; fix the sentence |
| 18 | Steering | ours: copy-lookup steering fits on held-out fit windows; the table pairs E135 steering with E136 re-gating on other rows (0.691 vs 0.686) | the stated pairing is false for this row | M | GPT only | use the matched E135 re-gating, or disclose |
| 19 | Effective rank | ours: the rationale says new needs higher rank, but the 7 "agreements" have the reuse side higher | "sides with the mechanism" points the wrong way | M | Opus only | rewrite the claim |
| 20 | Effective rank / size | ours: real-text "No cheaper reading does better" (appD:199-211) is false: rank is disjoint in all 6 cells, incl. French/Yoruba where the share is unreadable; Qwen gold real is the larger update on 3/3 seeds, contradicting the smaller-update sentence (appH:17-21) | overclaims | M | GPT only | correct both sentences |
| 21 | Update size | ours: three norms under one label (sum of per-matrix F, global F, F² for lowercase/cipher; a fourth, ‖δW‖/‖W0‖, elsewhere); sizes and ranks for the 96-key and twin rows from different updates | Confirmed by the Opus follow-up (`fidelity/scripts/one_norm_size.py`): under one norm (‖δW‖_F/‖W0‖, m67.json) the nine built pairs read 0 agree / 7 reverse / 2 no call, not 0/6/3. The one changed call (half-known vs cipher, equal size) is not substantive: the means differ by 1.9%, and the seed ranges clear by 0.047% of the size under a rule with no noise band. The conclusion "size never sides with the share" stands. The 96-key size cells read E58's original updates while the other cells read the refits | M | yes (after follow-up) | rescore from one norm, join by file/step/seed; report 0/7/2; appD:41 "equal by construction" → "within 2% by construction" |
| 22 | SAE | ours: App. E:4 names the wrong sites (the code uses attention and MLP outputs) | not reproducible from the text | B/M | yes | relabel |
| 23 | SAE | ours: the quoted 0.97/0.95/0.14 (sec5_results.tex:48-50, main.tex:872-877) are E263's all-window K* read (marked \kst), not the setup's answer-position μ; E263's filed separation missed (gap .115 vs .2). Checked by redteam-lead in sae_eb/run.py:58-86: saved mean-ablation K, write change over all window positions | not like-for-like with the main table | M | GPT only (checked) | say "all positions" beside \kst; do not call it passing |
| 24 | SAE | ours: SAEBench not cited, no setting given; attention SAEs 50-62% dead (bound 0.15); two filed quality bars missed (loss recovered 0.93 at mlp2/3 vs 0.95; mlp3 0.80-0.82 on task text); ~15% of training data re-read; off-base FVE falls (rebuilt MLP2 .861 → .094); Bricken/Cunningham cited for "SAE on base applied to fine-tune" (main.tex:194-195), which they do not do (Kissane 2024, Wang 2025 do) | competence fine on the base, unproven off-base | M | yes | cite SAEBench + Kissane/Wang; report quality numbers beside |
| 25 | Jain, 67M | ours: reverse FT = KL distillation to the base logits read as a half-life, no scratch model; pruning by a KL share with our own control; four checkpoints | sec5_results.tex:174-179 says Jain's tests failed | B | yes | "our KL-based adaptations failed the prespecified separation at four tested checkpoints" |
| 26 | Jain, E59 | ours: a generic Pile next-token readout (50,277 vocab), not Def. 1's capability readout; missed its LM-loss control (+0.37 vs +0.1) and P3; the base head also fails P3; the filed verdict was "unreadable" | "their linear readout … held the values at no block" is wrong | B | yes | report as unreadable, our Pile-head adaptation |
| 27 | Jain, PCFG | ours: 21/21 = 21 checkpoints of 12 runs on our two-target comparison, not Jain's task; answer-only ties 21/21; post-hoc recipe-stratified size ties; single size threshold 15/21. Partial Table 6; reconstructed generator; a↔c injection; W1/W2 miss ±15 pts | "Update size did about as well" hides this | B (GPT) / m (Opus) | partly | GPT's wording: "on an adapted counting-task comparison using Jain et al.'s grammar …" |
| 28 | Jain, App. G | ours: misstates Def. 2 (a composition g∘f_C with strong/weak relevance, not a binary reuse test); claims a random-label check for E59 that never ran; "replaced" vs §5's "run"; Def. 1's "all inputs" should be the subdomain | wrong definition attribution | B (GPT) / m (Opus) | partly | rewrite appG:20-27 |
| 29 | Geiger | ours: main.tex:287-301 implies a bijection from latent plus remainder; it gives reconstruction only (interchange can leave the encoder's image); App. G says mean ablation where the body uses interchange | a definition claim | M | yes (both flag App. G) | state the codomain/convention; fix App. G |
| 30 | Sutter / Prakash / VPD | ours: Sutter's Theorem 1 made unconditional (Assumptions 1-5); Prakash's patch direction reversed at main.tex:377-380; VPD's 38,912 is the total, not per matrix (sec4_setup.tex:16-18), and the neuron 97th percentile is per neuron on Pile | wording | M | yes | wording |

Minor rows (about 70 across the reviews: precision, seeds, BOS and position 0, batch sizes, metadata, venues, APD/SPD citations, Lee scoping, Panigrahi vs our "grafts", Ward, IA³'s 3e-3 from T-Few) are in the per-review files.

## Fixes for the paper, by type

- **Rescore (CPU, saved files, no training):**
  - (1) Every crosscoder with the negative-beta filter; regenerate the ladder figure and the block-3 text.
  - (21) Update size from one norm per state, joined by file/step/seed.
  - (16) The steering figure from strength-1 data, or a relabel.
  - (17) Steering tallies per form.
- **Relabel or rewrite (text):**
  - The ADL row (7).
  - Neuron CMAP, scoped to three readings (8, 9).
  - The half-gain count (11).
  - Steering (15).
  - The Jain 67M and E59 sentences (25, 26) and the 21/21 sentence (27).
  - App. G's Def. 2, mean-ablation and percentile items (28, 29, 14).
  - The effective-rank rationale (19).
  - appD real text, appH Qwen (20).
  - SAE sites (22) and SAE scope (23).
  - Every crosscoder number: "our reading" (2).
  - The Gaussian sentence (5).
  - Sutter, Prakash, VPD (30).
- **Cite:** SAEBench (Karvonen 2025); Kissane 2024 or Wang 2025 for base-SAE-on-fine-tune; Nanda 2023 for attribution patching; APD and SPD; pin the Venhoff version.
- **Disclose:**
  - The overlap's failed control (13).
  - The top-neuron sparsity and base shift (12).
  - The SAE quality numbers (24).
  - The crosscoder settings' sources (3, 4).
- **Cut, if not disclosed:** the top-neuron overlap row (13).
- **Optional re-runs (each needs a cell):**
  - The top-neuron share at 240/block.
  - A faithful ADL with their evaluations.
  - A Jain-faithful reverse fine-tune with a scratch baseline on a 67M task that has labels.
  - None is needed for honesty.

## Open items
- E212, E244 and E203's other 24 crosscoder sides are not yet rescored with the filter (e58 copy moved .12 → .10).
- The local writeup-lead commit 4b33440 (separating the two Minder citations in §4) is not pushed.
