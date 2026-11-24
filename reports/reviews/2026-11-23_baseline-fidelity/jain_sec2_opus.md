# Fidelity audit: Jain et al.'s tests and the Section 2 sweep

*redteam-lead fidelity auditor (Opus), 2026-11-23. Read only; nothing in either tree was edited.*

Paper tree: `~/commute-agents/writeup-lead/paper/paper/` (it was being edited while this ran: `sec4_setup.tex` changed
between two reads, so its line numbers here are from the second read). Code tree: `~/commute-agents/redteam-lead/`.
Sources: Jain et al. 2023 (arXiv 2311.12786 v2, ICLR 2024) from `~/commute-data/lit_review_pdfs/...txt`, cited as
section and page plus `L<n>` (line in that txt). Prakash 2024 from the local PDF. Geiger (arXiv 2301.04709 v4), Sutter (2507.08802 v2), Lee
(2401.01967), Panigrahi (2302.06600) and Jain 2024 safety (2407.10264) were downloaded to
`~/commute-data/agents/redteam-lead/fidelity/papers/jain_sec2/`, with text in `../papers/txt/`. Bib metadata was checked
against the arXiv API (`jain_sec2/meta.xml`).

Severity: **blocker** = the paper's label or citation is wrong; **major** = it changes a reported reading or how it
is read; **minor** = a disclosure only.

## Inventory

| method | where the paper names it | audited by |
|---|---|---|
| Jain et al. 2023 Counter PCFG set-up (minGPT, grammar, priors, spurious share, lr) | sec5_results.tex:174 ("their own counting grammar, in a 3M model") | this report |
| Jain pruning revival (Alg. 3) | sec5_results.tex:174-178; sec4_setup.tex:39; main.tex:213-214 (related work); appG:23-26 | this report |
| Jain reverse fine-tuning (reFT) | sec5_results.tex:175-178; sec4_setup.tex:39; appG:23-26 | this report |
| Jain Def. 1 possession readout / Def. 2 wrapper | sec5_results.tex:178-179; main.tex:776-778 and Table `fig:wordings` row 1; appG:20-27; sec4_setup.tex:39 | this report |
| Hewitt and Liang control task | appG:25 | this report (with Jain) |
| jain2024safety | main.tex:175 (related work only) | this report (claim and bib) |
| Prakash 2024 path patching and CMAP | main.tex:174-176, 215-216, 378-379; appG:28-33 | this report |
| "CMAP at neurons" (neuron patching shares relabelled) | main.tex:915-917; sec5_results.tex:170; appE:92 | neuron/top-neuron auditor (label noted here) |
| Lee 2024 (DPO bypasses toxic vectors) | main.tex:151-152, 218 | this report |
| Panigrahi 2023 grafting / skill localization | main.tex:196; appG:15 | this report |
| Our "grafts" (added heads/neurons) | main.tex:797-800, 849, 854, 996-998; appD:147 | this report (name clash only) |
| Geiger 2025 causal abstraction (Defs 2, 5, 9, 10, 25, 28, 41, 44; Remark 6; §3.5.2, 3.6, 3.7) | main.tex:209-213, 232, 280-331, 370, 394; appG:9-19 | this report |
| Sutter 2025 | main.tex:217-218 | this report |
| VPD (bushnaq2026vpd), APD, SPD | main.tex:197-207; sec4_setup.tex:7; appG:34-40 | this report (short) |
| IA$^3$ (liu2022ia3), gain-only training | appG:47-51 | this report (short) |
| Dr. GRPO (liu2025understanding), spurious rewards (shao2025spurious), TRL default | appC:12-19 | this report (short) |
| Activation difference lens (minder2025narrow) | main.tex:126, 189-190; sec4_setup.tex:39, 53-54; sec5_results.tex:166 | this report: bib key and wording only; method fidelity by the earlier ADL audit |
| Crosscoder (lindsey2024crosscoders, minder2025overcoming) | main.tex:185-194, 390, 940-976; sec5_results.tex:161, 183; sec5_3:83; appD:87; appE:98; appG:52 | crosscoder auditor |
| Neuron activation patching (vig2020), attribution patching (syed2023) | main.tex:915-918; sec4_setup.tex:39 | neuron/top-neuron auditor |
| Top-neuron featurization and overlap (goodfire2026neurons, cammarata2026clean) | main.tex:923-925; sec4_setup.tex:18; appG:41-46 | neuron/top-neuron auditor |
| Steering (turner2023, rimsky2024, venhoff2025base, ward2025reasoning) | main.tex:926-933; sec5_results.tex:171; appE:91 | steering auditor |
| Effective rank (roy2007) | main.tex:937-938 | effective-rank auditor |
| SAE latents (bricken2023, cunningham2023, bussmann2024batchtopk) | main.tex:197, 287; sec4_setup.tex:15 | SAE auditor |
| SAH / LIMA, URIAL, Yue, DeepSeek-R1, Raghavendra, Lambert, Deeb, Lubana MMC, Méloux | main.tex:75-81, 136, 151-156, 168-176 | this report (bib only; claims, not methods we ran) |

## 1. Jain et al. 2023: the counting PCFG set-up (E55 Stage A)

Our set-up matches Jain's stated recipe closely; `src/commute/pcfg.py` and `pcfg_train.py` cite the source line by line.

| departure | ours/theirs | evidence | consequence for what the paper reports | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Jain state the model (minGPT, 6 blocks, 6 heads, d 192, about 3M), AdamW at 1e-3, 20% warm-up then cosine. We match that and fill in what they leave unstated from minGPT's trainer defaults: dropout 0.1, weight decay 0.1, betas (0.9, 0.95), clip 1.0, bf16 autocast, and pretraining batch 96 (they give a batch only for fine-tuning). | ours | `src/commute/mingpt.py:1-27`; `src/commute/pcfg_train.py:3-6`; `experiments/pcfg_anchors/run.py:72`; Jain App. C.2 p.21, L1160-1170, L1193 | None. | minor | no | One clause in appG: "minGPT trainer defaults where Jain et al. are silent (dropout, weight decay, clipping, pretraining batch)." |
| Spurious rows. Jain: "artificially adding the target tokens" so that count(b) = count(a) + 1, with the label histogram matched. Ours: keep the natural count(b) and set count(a) to count(b) − 1 by swapping a and c, which keeps the histogram exactly. | ours | `src/commute/pcfg.py:164-200`; Jain §5 p.7 L298-299, App. B.2 L1086-1092 | None for the paper, which reports no spurious-share cell. | minor | no (code docstring only) | appG line: "spurious rows built by editing count(a), not by adding b's, so the label histogram is unchanged". |
| Jain's learning rates disagree with each other: §5 gives ηM = ηPT/10 and ηS = ηPT/100 (1e-4, 1e-5), but App. C.2 gives ηL 1e-4, ηM 1e-5, ηS 1e-6. Table 6 uses 1e-4 and 1e-5. We use 1e-4 and 1e-5. | theirs | Jain p.7 L308-310 vs p.22 L1191-1192, Table 6 p.48 L3006; `run.py:111-115` | None. We follow Table 6. | minor | no | None needed. If the rates are named, say "Table 6's rates". |
| Grid. Jain's Table 6 crosses priors {0.999, 0.9, 0.5} with lr {1e-4, 1e-5} and spurious share C_Tr {0, 0.5, 0.8, 1}. We run the LOW prior (0.999) row of the grid, one R arm from HIGH (0.5), and a NEVER prior (0, 1, 0) that we added as the scratch model. One pretraining seed. The reproduction is partial: 7 of 9 10K cells are within 15 points, and both C_Tr = 1 cells miss. | ours | `run.py:111-115`, `pcfg.py:60-67`; TRIED E55 "PA2 pass at the cap ... W1 17.7 ... W2 25.95 miss"; Jain Table 6 p.48 | None directly: the 21 states in the paper are C1b's, not Table 6 cells. | minor | no | Optional: "(their Table 6 reproduced within 15 points in 7 of 9 cells)". |
| Probe input. Jain: "the stream corresponding to the answer token (Ans)". Ours: the `=` position, which predicts Ans. At Ans the input token is the answer itself, so Jain's literal wording would leak the label. | ours (fixes an ambiguity in theirs) | `src/commute/probe.py:4-7`; Jain App. D p.22-23, L1246-1248 | None. | minor | no | appG: "probes read at the position that predicts the answer". |
| Probe control and possession rule. Jain: "for all x ∈ X_C", with no control. Ours (E55): a within-batch shuffled-label probe and a margin rule (m ≥ 3 s and m ≥ 0.5 m₊). appG calls this "a random-label control task in the manner of Hewitt and Liang", but Hewitt and Liang's control gives each word *type* a fixed random label. E55 shuffles per example; E116 uses the per-type form. | ours | `probe.py:9-12, 111-130`; `experiments/definition_table/readouts.py:1-3`; appG:23-25 | It does not change a reading, but the attribution fits E116 only. | minor | partly (appG:23-25) | appG:25: "... against a control task (per-type random labels, after Hewitt and Liang 2019; a per-example label shuffle in the counting probes)". |

## 2. Jain pruning revival (Alg. 3)

| departure | ours/theirs | evidence | consequence for what the paper reports | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Unit. Jain prune "a neuron ... defined as a row in the weight matrix" (any matrix, by grad·w, top K). Ours: MLP hidden units only (a row of the first MLP Linear, plus its bias), 4,608 units. The sign matches: the top grad·w units are those whose removal lowers the base task's loss most. | ours | `src/commute/revival.py:5-8, 48-76`; Jain App. D p.22 L1209-1235 | Small. Attention rows are never pruned (the C3 ladder also lists "head pruning" as not given). | minor | no | appG: "pruning restricted to MLP hidden units". |
| From curves to a labeller. Jain plot revival accuracy against K (Fig. 7) and give no decision rule or random baseline. We add 20 random K-unit removals and a cut-off: ≥ 3 sd in C1, the midpoint of a calibration-seed gap in C1b. The truth set (task P = answer count(a) + 1 on every row, built to reuse; task N = count b) is also ours. | ours (labeller, cut-off, truth set); theirs (no published rule to inherit) | `revival.py:37-41, 235-250`; `experiments/pcfg_anchors/results/c1b_cutoffs.json`; TRIED E55 C1b; Jain §5 p.7-8, Fig. 7 | "Labelled all 21 fine-tunes right" (sec5_results.tex:174-176) is the score of our labeller, with cut-offs set on 14 calibration states, against our truth set. The paper does already say the label equals the task there. | minor | partly ("there the label is the task") | sec5_results.tex:174: "... on fine-tunes we built to reuse their count of a (answer count(a) + 1) or to need a new count (count b), with cut-offs set on held-out seeds, ...". |
| Paraphrase. The paper: "remove the units that matter most and see whether the base behaviour returns". Jain: remove the few units whose removal most lowers the *base task's* loss (largest grad·w). "Matter most" reads as the units most important to the fine-tune. | ours (wording) | sec5_results.tex:174-175; Jain Alg. 3 p.22 | It mis-states the instrument to a reader who does not know it. | minor | n/a | "pruning revival (remove the few units whose removal most lowers the loss on the base task, their Alg. 3, and see whether the base behaviour returns)". |
| 67M port (C3, C3 ladder). The 67M has no labelled base capability, so "old behaviour" = W0's whole next-token distribution, the loss = KL(W0 ‖ state), and revival is read as a share of D. K runs 13-320 MLP neurons out of 3,072 per block. The positive control (rank-1 vs cipher poles at ≥ 3 sd) is ours, and a planted 13-neuron wrapper passed it (R 0.973). TRIED: "Not given: Jain's regime (labelled capability, small lr to convergence, scratch model) ... head pruning." | ours | `src/commute/revival_kv.py:6-10`; TRIED "E55 C3", "E55 C3 ladder"; Jain App. D | sec5_results.tex:176-178 reports our port's failure as the failure of "the tests of Jain et al.". The tests failed under conditions they were not built for, and failed a control that is ours. | major | no | sec5_results.tex:176-178: "On the 67M model we ported both tests: with no labelled base task, the base behaviour is the base model's whole output distribution. Set a top map built to reuse against a cipher built new, the ports passed a planted 13-neuron wrapper but did not separate the pair at any update size the recipe makes, with up to 320 units pruned." Also sec4_setup.tex:39 bracket: "which fail their own controls on the 67M" → "which fail our positive control in our port to the 67M". |

## 3. Jain reverse fine-tuning (reFT)

| departure | ours/theirs | evidence | consequence for what the paper reports | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| PCFG rate. Jain run reFT on the base task's data at a rate below the fine-tune's, against a Scr.+FT baseline (pretrained on the other operand, then fine-tuned). C1 matches this: lr/10 to 95% count-a accuracy, against NEVER (pretrained on b only), the same baseline. C1b uses one rate, 1e-5, for every state, so states fine-tuned at 1e-5 are reversed at their own rate, not a lower one. | ours | `revival.py:9-10, 38-39, 104-140`; `decisions.md` row 104 ("departure from Jain et al. 'below the fine-tune rate', stated"); Jain App. D p.23 L1249-1252, Fig. 9 p.7 L441-466 | Small. The C1b margins are wide (325 steps either side). | minor | no (decisions.md only) | appG: "reverse fine-tuning at one rate, 1e-5, also for states fine-tuned at that rate". |
| Labeller rule (existing iff ≤ 0.5 × scratch's steps, cap 10K) is ours. Jain compare curves by eye. | ours | `revival.py:39, 256-270` | Same as the pruning labeller row. | minor | no | Covered by the pruning-labeller fix. |
| 67M port. Reverse = train toward W0 by KL at every position, read as the half-life t_half of κ, with no scratch baseline. Jain's reFT fine-tunes on the base task's labelled data and is read against a scratch model. The paper's gloss, "train back towards the base model and count the steps", describes our port. On the PCFG, where training is on the count-a labels, it describes Jain's method loosely. | ours | `revival_kv.py:11-12` (`finetune.finetune_to_reference`); TRIED "E55 C3": "lr 3e-6, half-life of KL(W0 ‖ state)"; C3 ladder "Not given: ... scratch model"; Jain App. D L1249-1252 | The paper calls this quantity Jain's reverse fine-tuning, which it is not. Without the scratch comparison the instrument has no reference for "fast". | blocker | no | sec5_results.tex:175: "reverse fine-tuning (fine-tune again on the base task and count the steps, against a model that never had it)". For the 67M sentence: "and a reverse fine-tune toward the base model's outputs, with no scratch reference". |

## 4. Jain Def. 1 readout and the possession wording (E59, E116)

| departure | ours/theirs | evidence | consequence for what the paper reports | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Readout target. Jain Def. 1: Read_l is a linear layer trained on D_PT whose output equals f_C(x_d) for x ∈ X_C. In the PCFG its target is the count label. E59: one d → vocab map (768 → 50,277) per block trained by next-token loss on Pile windows (a tuned-lens style readout), with "possession" read as the value's rank among 50K tokens. | ours | `src/commute/readout.py:1-9`; `experiments/jain_readout/run.py:1-17`; Jain §3 p.3 L141-148, App. D p.22-23 L1238-1248 | sec5_results.tex:178 calls it "Their linear readout". It is our stand-in, with a different target and output space. | blocker | no | "Our stand-in for their Def. 1 readout, a per-block linear next-token readout trained on Pile text, ..." |
| Schedule and quality. Jain's probe schedule (4K steps, lr 1e-3, decays at 2K, 3K, 3.5K, tuned for a 192 → 263 probe) was carried over to a 768 → 50,277 map. The last block's held-out loss was 3.94 against the linearised head's 3.57: +0.37 where the bar was +0.1. The construction check P3 fails on the model's own logits (16 elicit keys in band vs ≥ 24 needed). The filed verdict is "unreadable". | ours | `readout.py:8`, `probe.py:29`; TRIED E59 "Closed ... P1 unreadable", `experiments/jain_readout/results/m1_check.json`; `reports/analyses/2026-11-02_analysis_e59-jain-readout.md` §"The answer" | sec5_results.tex:178-179 ("held the values of the half-known map at no block, so no task could be read with it") reads as a finding about the model. The instrument failed its own quality bar, and its construction check failed on the model itself. | major | no | "... did not reach the quality of the model's own output head on unseen text at the token budget we had, and its construction check failed even on the model's own output, so it gave no reading." |
| "For all x" replaced by a bar filed before the run (E59: ≥ 24 of 32 keys at rank ≤ 20; E116: 0.90 above a control); function carried to held-out values; causal check by removing the readout direction. | ours | appG:23-26; `decisions.md` rows 81, 87, 164 | Disclosed. | minor | yes (appG:20-27) | None. |
| appG:21-22 glosses Jain as "reused when the fine-tuned map is a function of that readout". Jain do not define reuse. Their Def. 2 calls the transformed map g ∘ f_C a wrapped capability and defines strong or weak relevance. | ours (wording) | appG:21-22; Jain §3 Def. 2 p.4 L167-195 | A small misattribution of a term. | minor | n/a | "... and call the fine-tuned map a wrapped capability when it is g ∘ f_C (their Def. 2)". |
| appG:25-26 says pruning and reverse fine-tuning were "replaced" by removing the readout direction, while sec5_results.tex:174-178 reports both as run. decisions.md row 164 says they were "reported beside it". | ours (inconsistency) | appG:25-26 vs sec5_results.tex:174-178; `decisions.md` row 164 | A reader cannot tell what was run. | minor | n/a | appG: "added removal of the readout's direction as the causal check; pruning and reverse fine-tuning were run as separate tests (Section 5.x)". |
| Stale names. sec5_results.tex:177-179 says "rank-1 map" and "half-known map", which the paper has renamed "top" and "mid" (main.tex:1040-1042, appE:12). | ours | as cited | Terminology mismatch. | minor | n/a | "top map", "mid map". |
| Jain's Def. 1 sets no capacity, control or budget for the readout. With a large-vocabulary readout, "possession" is then underdetermined: a weak readout biases toward "not held". | theirs | Jain p.3 L134-148 ("idealized definition"); TRIED E59 "a weak readout biases toward 'not held'" | Any "not possessed" from this wording is weak evidence. | minor | partly (appG bar) | None beyond the E59 fix above. |
| sec4_setup.tex:39 lists "the pruning, reverse fine-tuning and readout tests of jain2023" among "the existing state of the art in model-diffing". They are tests from a fine-tuning analysis, not diffing methods, and they were not proposed as labellers of reuse. | ours (label) | sec4_setup.tex:39; Jain §1, §5 | A mild mislabel. | minor | n/a | "... and, from the fine-tuning literature, the pruning, reverse fine-tuning and readout tests of \citet{jain2023} (our ports)". |

## 5. jain2024safety

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| The sentence is garbled: "study safety fine-tuning in the same spirit fine-tuning". Nothing of this paper is run. | ours (text) | main.tex:175 | None. | minor | n/a | "\citet{jain2024safety} find that safety fine-tuning learns a minimal transformation of MLP weights that routes unsafe inputs away, in the same spirit". |

## 6. Prakash et al. 2024

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Direction of CMAP. main.tex:378-379: "fixing u in F to the value it takes in M as in the cross-model activation patching of Prakash". CMAP patches head outputs *from the fine-tune into the base model*. appG:29-30 has the direction right. | ours (wording) | main.tex:378-379; Prakash §6.1 p.8 ("patch the output of heads in the Goat-7B circuit from the Goat-7B to Llama-7B model") | The argument ("comparing values reads every circuit as changed") holds either way, but the citation describes the method backwards. | minor | n/a | "fixing u in M to the value it takes in F, as in the cross-model activation patching of \citet{prakash2024}". |
| "CMAP at neurons" relabel (pending P2.68). Prakash patch a discovered circuit's heads. Ours patches every MLP neuron at once. | ours | main.tex:915-917; appE:92; Prakash §6.1 | For the neuron auditor. Here only: the label must carry the difference. | minor | yes (pending note) | "CMAP \citep{prakash2024} applied to every MLP neuron at once, not to a discovered circuit". |
| Related-work description (path patching in each model, CMAP across them) and appG:28-33 (heads found by circuit discovery, accuracy scored). | none | main.tex:215-216; Prakash §4.1, §6.1 | Accurate. | — | — | — |

## 7. Geiger et al. 2025

All definition, remark and section numbers checked against arXiv v4, which is the JMLR text: Def. 2 Signature, Def. 5
causal model, Remark 6 (mechanisms read only parents), Def. 9 hard and Def. 10 soft intervention, Def. 25 exact
transformation, Def. 28 bijective translation, Def. 41 approximate transformation, Def. 44 interchange interventions,
§3.5.2 sub-circuit analysis (behaviour as a partial function B), §3.6 modular feature learning as bijective
transformation (lossy SAEs are "approximate transformations"), and §3.7 (PEFT "viewed as interventions on model
parameters", one sentence). Every one matches.

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| appG states our departure from Geiger wrongly. appG:12-13 says "the circuit is sub-circuit analysis over variables rather than connections, with mean ablation", and appG:31 says "find the circuit by mean ablation". The definition in the body finds the needed set K by existential single-variable interchange (Def. 44). main.tex:393-396 says this "needs no ablation values" and contrasts it with §3.5.2. | ours (stale text) | appG:12-14, 31 vs main.tex:325-331, 393-396 | The departures appendix tells the reader the opposite of what the definition does. | major | n/a | appG:12-13: "The needed set is found by single-variable interchange interventions (Def. 44), not by the ablation of connections of sub-circuit analysis (§3.5.2), so no ablation values are chosen." appG:31: "and find the needed variables by interchange". |
| One featurization fixed from M and applied to both networks, plus a mandatory remainder that makes the translation exact. Geiger relate a network to a high-level model and treat lossy SAEs as approximate. | ours | appG:9-19; Geiger §3.6, Def. 41 | Disclosed. | minor | yes (appG) | None. |

## 8. Sutter et al. 2025

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| The paper writes "with an unrestricted map ... any algorithm abstracts any network". Sutter prove this "under reasonable assumptions" (Assumps. 1-5, e.g. input-injectivity and output-surjectivity). The qualifier is dropped. | ours (paraphrase) | main.tex:217-218; Sutter abstract L17-19, Thm. 1 L599 | Slight overstatement. | minor | n/a | "show that, under mild assumptions, with an unrestricted map ...". |

## 9. Lee et al. 2024

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| None. "DPO leaves the toxic MLP vectors in place and learns to bypass them" matches Lee (capabilities "not removed, but rather bypassed"; the toxic vectors are minimally shifted). Nothing of theirs is run. | — | main.tex:218; Lee abstract L19, §1 L77-81 | — | — | — | — |

## 10. Panigrahi et al. 2023, and our "grafts"

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Name clash. Panigrahi's grafting copies a tiny subset (about 0.01%) of the fine-tuned parameters onto the pretrained model. Our "grafts" are new heads or neurons added to the frozen base model and trained alone (E201, E206). appG:15 cites Panigrahi's grafting as the precedent for the mechanism swap, so the paper uses one word for two operations. | ours | main.tex:797-800, 996-998; appD:147; appG:15; Panigrahi §1 p.1-2 | A reader may take our grafts for Panigrahi's. | minor | no | Rename ours "added heads / added neurons", or add "(new weights, not the grafting of \citet{panigrahi2023})" at main.tex:996. |
| "Weight-level methods find where an update sits" (related work). | none | main.tex:196 | Accurate. | — | — | — |

## 11. VPD (and APD, SPD)

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| main.tex:198-199, "Attribution-based and then stochastic parameter decomposition introduced the approach on toy models", has no citation. main.tex:206-207 has the open note for it. | ours | main.tex:198-207 | A method claim with no citation. | minor | open note | Cite Braun et al. 2025 (APD, arXiv 2501.14926) and Bushnaq et al. 2025 (SPD, arXiv 2506.20790); check both IDs before adding. |
| Base decomposition and Γ frozen and applied to the fine-tune, with the least-squares split and one remainder per matrix. | ours | appG:34-40; README.md:10 (Goodfire's released 67M and its VPD decomposition) | Disclosed. | minor | yes (appG) | None. |

## 12. IA$^3$ (gain-only training)

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Adam at a constant rate in place of Adafactor. One gain per head output in place of key and value rescaling. lr 3e-3, not swept. | ours | `experiments/gain_only/results/predictions.json` `settings.lr_source`; `src/commute/gains.py:109`; Liu et al. 2022 | Disclosed. | minor | yes (appG:47-51) | None. |

## 13. Dr. GRPO, spurious rewards, TRL (appC)

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| One on-policy update per batch, so ratio clipping never acts; Dr. GRPO keeps PPO-style clipping. The advantage (reward minus group mean, no std) and the summed loss over one constant match. | ours | `src/commute/qwen_rl.py:80-87, 577-590`; `src/commute/rl_arms.py:6`; appC:12-18 | Disclosed. | minor | yes (appC:15-17) | None. |
| "No KL penalty ..., as in TRL's default". The default changed across TRL versions. | ours (citation precision) | appC:18-19 | None for readings. | minor | no | Name the TRL version whose GRPO default β is 0. |

## 14. Activation difference lens (bib key and wording only)

| departure | ours/theirs | evidence | consequence | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| The bib key now points to minder2025narrow, whose title, authors and venue (ICLR 2026) match arXiv 2510.13900. | none | refs.bib; arXiv metadata | Fixed. | — | — | — |
| sec4_setup.tex:39 still holds the old misattribution, "crosscoder-based activation difference lens of \citet{minder2025overcoming}", inside `\old{}`. | ours | sec4_setup.tex:39 | It renders as struck-through text in the draft. | minor | n/a | Delete the `\old{...}` before release. |
| sec4_setup.tex:39 (new text) lists "the activation difference lens of \citet{minder2025narrow}" as compared and does not say the scored quantity is ours. The open note at :53-54 says the faithful re-run is pending. | ours | sec4_setup.tex:39, 53-54 | A reader of the Setup takes the scored column to be Minder's ADL. | minor | only in the open note | "the activation difference lens of \citet{minder2025narrow} (our variant, Appendix X)". |
| main.tex:126 and sec5_results.tex:166 both say "our version departed from the published method". | — | as cited | They describe the quantity as ours. | — | yes | — |
| main.tex:189-190 (related work) describes Minder's own method (the mean activation difference on the first tokens of unrelated text). | — | as cited | Accurate for theirs, and it does not claim the quantity is ours. | — | — | — |
| No place in the paper says what our ADL quantity is: the share of gain lost when the block-2 input along δ̄ is set to the base model's. It is absent from the baseline list (main.tex:900-940), appE and appG. Yet sec5_results.tex:166-170 reports its numbers (8 of 9 signs, rank correlation 0.83, p 0.44 and 0.22, partial −0.53). | ours | `experiments/adl/run.py:6-12`; main.tex:900-940; appG | The reported numbers have no method entry and no departures item. | minor | no | Add a baseline item and an appG item: "ADL-derived share (ours): the share of the fine-tune's gain lost when its block-2 input along the mean first-5-token difference δ̄ is set to the base model's. Minder et al. read δ̄ by logit lens, Patchscope and steering, and do not score it as a share." |

## 15. Bibliography entries used for methods

Checked against arXiv metadata (title, authors, comments) for every arXiv entry.

| entry | issue | severity | fix |
|---|---|---|---|
| jain2023 | Correct (ICLR 2024; authors and title match). The key's year 2023 ≠ year 2024, which is harmless. Code docstrings call it "Jain et al. 2024", which is easy to confuse with jain2024safety (code only). | — | None in the paper. |
| jain2024safety | `@misc` with no venue. arXiv says "Preprint". I believe it appeared at NeurIPS 2024 but did not confirm it. | minor | Check, then set booktitle NeurIPS 2024. |
| prakash2024, panigrahi2023, lee2024dpo, hewitt2019control, liu2022ia3, minder2025narrow, minder2025overcoming, bushnaq2026vpd, venhoff2025base, ward2025reasoning, bussmann2024batchtopk, lubana2023mmc, deeb2024unlearning, meloux2025identifiable | Correct for the claims they support. | — | — |
| sutter2025 | `@misc`; arXiv: NeurIPS 2025 (Spotlight). | minor | Make it `@inproceedings`, NeurIPS 2025. |
| geiger2025 | JMLR 26, "pages 1--63". arXiv v4 has 63 pages; the JMLR article number and page range are not confirmed. | minor | Add the JMLR article number and check the page range. |
| liu2025understanding | arXiv only (published at COLM 2025, as far as I know). | minor | Optional venue. |
| APD, SPD | Missing (see §11). | minor | Add both. |
| burns2023weak | In refs.bib, cited nowhere. | — | Drop or cite. |

## Verdicts

- **Jain set-up (PCFG):** faithful. It is Jain's minGPT, grammar, priors, Table 6 rates, schedule and spurious rule. Every
  difference is a gap filled where they are silent or a documented build choice. Disclosure lines only.
- **Jain pruning revival:** faithful on the PCFG (Alg. 3, MLP units). The labeller, its cut-offs and its truth set are ours.
  The 67M port changes the loss and has no labelled capability; the paper must call it a port (major).
- **Jain reverse fine-tuning:** faithful in C1 (scratch baseline = NEVER), with a rate departure in C1b. The 67M version
  is KL distillation toward W0 with no scratch model, which is not reFT. The paper's label is wrong (blocker).
- **Jain Def. 1 readout:** E59's next-token, 50K-vocabulary Pile readout is our stand-in, not "their linear readout"
  (blocker). Its "held at no block" was an instrument failure (quality bar and construction check), not a finding (major).
- **jain2024safety:** cited only. Fix the garbled sentence and add the venue.
- **Prakash:** described correctly in related work and appG. main.tex:378 states CMAP's direction backwards (minor).
  "CMAP at neurons" needs its every-neuron qualifier.
- **Geiger:** every number cited matches arXiv v4. appG's "mean ablation" description contradicts the existential
  definition in the body (major).
- **Sutter / Lee / Panigrahi:** Lee is accurate. Sutter drops "under assumptions" (minor). Panigrahi is accurate, but
  our "grafts" share its word for a different operation (minor).
- **VPD / IA³ / Dr. GRPO:** departures disclosed. APD and SPD are uncited (minor). Name the TRL version (minor).
- **ADL (key and wording):** the key is fixed. The struck misattribution remains in sec4_setup's `\old{}`. The Setup lists ADL
  without "our variant", and the paper never defines the quantity it reports (all minor).

## Row counts

Counted from the tables, leaving out rows with no departure:

- §§1-14 departure rows: **2 blocker, 3 major, 31 minor**.
  - Blockers: §3 row 3, the 67M "reverse fine-tuning" is KL distillation toward W0 with no scratch model, yet it is labelled as Jain's test. §4 row 1, E59's next-token Pile readout is called "their linear readout".
  - Majors: §2 row 4, the 67M pruning port and a positive control of ours are reported as Jain's tests failing. §4 row 2, E59's instrument failure is read as "held at no block". §7 row 1, appG says the definition uses mean ablation, which contradicts the existential interchange definition in the body.
- §15 bib rows: 5 minor (jain2024safety venue, sutter2025 venue, geiger2025 pages, liu2025understanding venue, APD and SPD missing).
- Total: **2 blocker, 3 major, 36 minor**.
