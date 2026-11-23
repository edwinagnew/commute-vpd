# Fidelity audit: the SAE featurization (E234 SAEs, read in E245, E251, E263)

Audited 2026-11-23, read only. Code: `~/commute-agents/redteam-lead/src/commute/sae.py`, `src/commute/sae_feat.py`,
`experiments/sae_train/run.py` + `results/`. Paper: `~/commute-agents/writeup-lead/paper/paper/`. Sources downloaded
to `~/commute-data/agents/redteam-lead/fidelity/papers/sae/` (Cunningham 2309.08600, Bussmann 2412.06410, SAEBench
2503.09532, Bricken HTML). Reference code compared: `~/commute-data/scratch/sae-lead/e234-writer/dl`
(saprmarks/dictionary_learning @ 60ec6bf, `trainers/batch_top_k.py`).

**The recipe the code follows.** It is a BatchTopK SAE (Bussmann et al. 2024) trained with SAEBench's recipe
(Karvonen et al. 2025, App. B, Table 3), ported by hand from dictionary_learning's `BatchTopKTrainer`. Settings:
width 16,384, k = 80, ReLU then BatchTopK over the flattened batch in training, and a JumpReLU threshold at inference
(an EMA of the smallest kept activation, beta 0.999 from step 1000). Loss: summed-square error plus 1/32 times the
AuxK loss (top d/2 = 384 dead latents; a latent is dead after 10M tokens without firing). No L1, no resampling. Decoder
columns are unit norm, the encoder starts as the decoder transpose but is untied, and b_dec starts at the geometric
median. Adam (0.9, 0.999) at lr 3e-4, warm-up 1,000 steps, linear decay over the last 20%, gradient clip 1. Inputs are
divided by one scalar per site (unit mean squared norm), and the scalar is folded back in after training. Data: 500M
Pile-train tokens, batch 2,048, context 512 with position 0 dropped, bf16 model forward. 13 sites were trained; the
featurization uses the 8 attention-output and MLP-output SAEs (`sae_feat.py:35`). I compared the port with
`batch_top_k.py` at 60ec6bf: `SAE`, `aux_loss`, `update_threshold` and `update` match it line for line.

The paper's citations do not name this recipe. `main.tex:287` and `:195` cite Bricken et al. 2023 and Cunningham et al.
2023. Both are ReLU + L1 SAEs. Bricken's is untied with neuron resampling, on one MLP layer of a 1-layer model, trained
on 8B activations. Cunningham's has tied weights, on the Pythia-70M/410M residual stream and MLPs, trained on 5-50M
activations. Only `sec4_setup.tex:15` names BatchTopK, citing Bussmann. SAEBench, the source of every setting, is not
cited.

## Departures

| # | departure | ours/theirs | evidence | consequence for the reported number | severity | disclosed in the paper? | fix |
|---|---|---|---|---|---|---|---|
| 1 | Sites. The appendix says the SAEs are trained "before and after each MLP". The code's variables are each block's **attention output and MLP output** (attn0-3, mlp0-3: 8 SAEs, 8 x 16,385 variables including the remainders). The 5 residual-stream SAEs were trained and checked but are not variables. | ours | `src/commute/sae_feat.py:35` (`SITES = attn{l}, mlp{l}`), `:5-12`; `experiments/sae_train/run.py:1-4`; paper `appE_methods_detail.tex:4` | ρ at SAE latents (`main.tex:872-877`, `sec5_results.tex:48-50`) is read at sites other than the ones the paper names. The number itself is right, but its description is wrong, and a reader would take "before and after each MLP" to mean residual-stream SAEs. | major | no: the text is wrong | Relabel `appE_methods_detail.tex:4`: "BatchTopK SAE latents trained on the base model's attention-layer and MLP outputs in each of the four blocks, with each SAE's reconstruction error as that site's remainder". |
| 2 | The SAE featurization is cited generically to Bricken et al. and Cunningham et al. (ReLU + L1, resampling or tied weights). What we use is a BatchTopK SAE with AuxK and no L1. | ours | `main.tex:287` ("latents of a sparse autoencoder (SAE) \citep{bricken2023monosemanticity,cunningham2023sparse}"); `src/commute/sae.py:51-82`, `:170`; Cunningham §2 eq. 1 and App. B; Bricken "Basic Autoencoder Training" | No number changes. A reader of §3 alone would assume the ReLU+L1 architecture. `sec4_setup.tex:15` corrects this with the Bussmann citation. | minor | partly: `sec4_setup.tex:15` names BatchTopK | `main.tex:287`: "...such as the latents of a sparse autoencoder (SAE) \citep{bricken2023monosemanticity,cunningham2023sparse}; we use BatchTopK SAEs \citep{bussmann2024batchtopk}...". |
| 3 | The training recipe is SAEBench's (Karvonen et al. 2025, App. B / Table 3), via dictionary_learning, and SAEBench is not cited. Bussmann's own settings differ: Adam beta2 0.99 (ours 0.999), batch 4,096 (ours 2,048), 1B OpenWebText tokens at context 128 (ours 500M Pile tokens at 512), residual stream only. No setting (width, k, tokens, data) appears anywhere in the paper. | ours | `src/commute/sae.py:1-32`; `experiments/sae_train/results/predictions.json` "settings"; Bussmann App. A.1 (p. 6); SAEBench App. B Table 3 (p. 12); `refs.bib` has no SAEBench entry | No reading changes. The recipe cannot be reproduced from the paper. | minor | no | Add to `appE_methods_detail.tex` after the Featurizations paragraph: "Each SAE has 16,384 latents and k = 80 and is trained on 500M Pile tokens with SAEBench's recipe and settings \citep{karvonen2025saebench} (a port of dictionary\_learning's BatchTopK trainer): AuxK loss on dead latents, Adam at 3e-4 with 1,000 warm-up steps and decay over the last 20%, activations scaled to unit mean squared norm. At inference latents use the trained threshold \citep{bussmann2024batchtopk}." Add the SAEBench bib entry (arXiv 2503.09532). |
| 4 | Attention-output SAEs: we apply a residual-stream recipe (SAEBench and Bussmann train only on the residual stream) to attention outputs. 50-62% of their latents are dead at the end of training. The filed bound was 0.15, so this missed it, against SAEBench's 0.118 at Pythia-160M. The other sites have at most 0.05 dead. | ours | `experiments/sae_train/results/train_*.json` curve[-1].dead: attn0 0.502, attn1 0.591, attn2 0.524, attn3 0.514; `stage0_resid_pre0_s0+13.json` dead attn0-3 0.530/0.616/0.556/0.560; filed band `predictions.json` expected.stage0_base.dead [0, 0.15]; TRIED.md:327 | Dead latents never fire, so they add nothing to ρ, and the attention SAEs still reconstruct well (FVE 0.91-0.97). The attention featurization in effect has about 6-8k live latents, not 16k. Probably no reading changes. | minor | no | Disclose in appE: "At the attention outputs about half the latents never fire (0.50-0.59 dead); the live dictionary there is about 7,000 latents." A narrower attention dictionary would be a re-run, but it is not needed for the reading. |
| 5 | Other quality bands filed before E234 were missed and are not reported in the paper. Loss recovered at mlp2 is 0.926 and at mlp3 0.930 (filed at least 0.95). On the key-task windows the base model's mlp3 recovers only 0.80-0.82. | ours | `stage0_resid_pre0_s0+13.json` (mlp2 recovered 0.9255, mlp3 0.9302); `items/c3-copy_s0_*.json`, `items/c3-cipher_s0_*.json` task_base mlp3 recovered 0.816 / 0.803; TRIED.md:327 | The remainder at mlp2 and mlp3 carries a noticeable share of the base model's own computation on task text, so more of each fine-tune's change at those sites can land in the remainder. This works in the same direction as row 7. | minor | no | Report the quality table (below) in appE: one line with each site type's FVE, L0, dead share and spliced CE increase. |
| 6 | The data were partly read twice: rows ran out at step 206,464 of 244,140, and the last 37,676 steps (77M tokens, 15.4%) reread rows from the start. SAEBench and Bricken sample without replacement (Bricken: "sampling without replacement ... is important"). The `sae.py` docstring still says "Each activation is read once". | ours | `experiments/sae_train/run.py:157-166` (WRAP); `results/run_s*.log` "WRAP: out of train rows at 1114112 (step 206464)"; `src/commute/sae.py:19`; Bricken "Basic Autoencoder Training" | Negligible. It falls in the lr-decay phase, and the measured seed spread is FVE 3e-5. | minor | no | Disclosure only, in appE: "the last 15% of training tokens repeat earlier text". |
| 7 | Theirs: a dictionary trained on the base model only has no latent for a direction the fine-tune adds, so any new write lands in the site's single remainder variable. This is what drops the rebuilt edit (reuse by construction) from 0.63 at VPD to 0.14: the remainder carries 0.83 of its write change. | theirs | `src/commute/sae_feat.py:5-12`; E263 (TRIED.md:357: "Premise check doubtful: a base-only dictionary may not see a built circuit"); paper `main.tex:194-196`, `:872-877`, `sec5_results.tex:48-50` | It changes how the SAE row reads. Low ρ at SAE latents can mean a reconstruction miss rather than a new mechanism, and this is shown on a known-reuse edit. | major | yes: `main.tex:194-196` ("has no unit for a new one"), `:874-875` and `sec5_results.tex:49-50` (reconstruction error carries most of the change) | Already disclosed. Optionally add at `main.tex:877`: "so at SAE latents a low ρ is an upper bound on how much is new only where the SAE reconstructs the fine-tune's activations". |
| 8 | Theirs: the base SAEs reconstruct the changed models worse. On Pile, 46 of 75 states drop FVE by more than 0.02 at some site (median worst drop 0.024). On x10's task windows, mlp3 falls from 0.925 to 0.859 and attn2 from 0.850 to 0.735. At resid_post3, an unused site, FVE is -15.97. | theirs | `experiments/sae_train/results/check_x10.log`; `items/*.json` fve_drop; TRIED.md:327 | At the sites used, the remainder partly reflects SAE failure on F as well as F's mechanism. The strengthening edits still read 0.95-0.97, so the effect is bounded on the built edits. | minor | no | Disclose in appE beside row 5: "On the fine-tunes the base SAEs' FVE falls by up to about 0.07 at the sites used; the loss goes into the remainder variable." |
| 9 | The paper's caveat that ρ's weights are unit-dependent and not harmless "for SAE latents of different scales" does not apply as implemented. The code weights each latent by its write f_u d_u (decoder columns unit norm, with the normalisation folded into the biases) in the block-output space, which is residual-stream units, as at VPD. | ours | `main.tex:738`; `src/commute/sae_feat.py:82-97` (`dd = |d_u|^2`, writes f_u d_u); `src/commute/sae.py:84-90` (`fold` scales biases and threshold only); dictionary_learning `scale_biases` | None on the number. The caveat wrongly suggests the SAE ρ may be a scale artefact. | minor | n/a: the text is inaccurate | `main.tex:738`: "...this is harmless when every variable is a write into the same residual stream, as at VPD and at SAE latents (each latent's write is its activation times a unit-norm decoder column)". |
| 10 | Citation: "A sparse autoencoder trained on the base model and applied to the fine-tune \citep{bricken,cunningham} reports which base features move". Neither paper applies an SAE to a fine-tune. The work that does: Kissane et al. 2024 ("SAEs (usually) transfer between base and chat models", Alignment Forum) and Wang et al. 2025 ("Persona features control emergent misalignment", arXiv 2506.19823), which the `\sugg` at `main.tex:182` already names. | ours | `main.tex:194-195`; `refs.bib` has neither | Related work only; no reading changes. | minor | n/a | Cite: "A sparse autoencoder \citep{bricken2023monosemanticity,cunningham2023sparse} trained on the base model and applied to the fine-tune \citep{kissane2024saes,wang2025persona} reports...", and add both bib entries. |
| 11 | Implementation: the SAE trainer is our port, not the authors' pinned code. The crosscoder by contrast calls the pinned trainer, and the same fork's `batch_top_k.py` is installed in the venv at d1fd6f3. The port matches 60ec6bf in `SAE`, `aux_loss`, `update_threshold` and `update` (checked in this audit). | ours | `src/commute/sae.py:1-32`, `:51-180` against `dl/dictionary_learning/trainers/batch_top_k.py:17-80`, `:192-318`; `pyproject.toml:20-22` | None found. The user's rule for literature baselines (memory: "literature-best recipe, their code") is met in substance, not in form. | minor | no | One clause in the appE text proposed in row 3 ("a port of dictionary\_learning's BatchTopK trainer, commit 60ec6bf"). No re-run. |
| 12 | Smaller data and model-side departures, documented in code but not in the paper: context 512 (SAEBench 1,024); position 0 dropped (the rows have no BOS; SAEBench masks BOS at eval); model forward under bf16 autocast (the Pythia suite ran fp32); a buffer of 409,600 activations instead of 250,000; all sites trained from one stream. | ours | `src/commute/sae.py:23-31`; `predictions.json` buffer_tokens_source; SAEBench App. B, Table 3 | Negligible. | minor | no | Covered by row 3's appE sentence, adding "(context 512, position 0 excluded)". |

Count: blocker 0, major 2 (rows 1, 7), minor 10.

## Quality of the SAEs (base model, Pile validation rows 4096-4607, 261,632 tokens, positions 1..)

Source: `experiments/sae_train/results/stage0_resid_pre0_s0+13.json`. Clean CE 2.676 nats/token. "Recovered" is
SAEBench eq. 4 with zero ablation. "Dead" means never firing on these tokens. **Bold** marks the 8 SAEs used as
variables.

| site | FVE | L0 | dead | spliced CE increase (nats) | loss recovered |
|---|---|---|---|---|---|
| resid_pre0 | 0.995 | 79.1 | 0.000 | 0.010 | 0.999 |
| **attn0** | 0.966 | 80.7 | 0.530 | 0.009 | 0.996 |
| **mlp0** | 0.948 | 79.2 | 0.000 | 0.054 | 0.992 |
| resid_post0 | 0.946 | 79.3 | 0.000 | 0.055 | 0.994 |
| **attn1** | 0.948 | 80.1 | 0.616 | 0.016 | 0.994 |
| **mlp1** (seed 0 / seed 1) | 0.937 / 0.937 | 80.0 | 0.006 | 0.045 / 0.044 | 0.973 / 0.974 |
| resid_post1 | 0.926 | 79.9 | 0.001 | 0.124 | 0.986 |
| **attn2** | 0.910 | 81.0 | 0.556 | 0.039 | 0.979 |
| **mlp2** | 0.875 | 80.0 | 0.002 | 0.059 | 0.926 |
| resid_post2 | 0.877 | 80.4 | 0.002 | 0.153 | 0.982 |
| **attn3** | 0.906 | 80.0 | 0.560 | 0.032 | 0.969 |
| **mlp3** | 0.925 | 79.9 | 0.016 | 0.107 | 0.930 |
| resid_post3 | 0.908 | 80.2 | 0.048 | 0.264 | 0.968 |

Seed spread at mlp1: FVE 3e-5, CE increase 5e-4. The final training loss (normalised units) is 0.005-0.115 per SAE.

**What the cited recipes report for comparable models:**
- SAEBench, BatchTopK, 16k latents, L0 77, Pythia-160M layer 8 residual (d = 768, the same width as ours): FVE 0.868,
  CE score 0.982, +0.141 nats, 0.882 alive. Gemma-2-2B layer 12: FVE 0.734, +0.109 nats (figures as filed in
  `predictions.json`, from `sae_bench_results_0125`).
- Cunningham (ReLU + L1): splicing the layer-2 residual reconstruction into Pythia-70M raised Pile perplexity from 25
  to 40, about +0.47 nats (p. 9). MLP dictionaries had many dead features (App. E).
- Bricken (ReLU + L1, 1-layer model, MLP): the run "A/1" recovers 79% of the MLP's loss reduction (zero-ablation
  baseline, "Reconstruction loss" section).

**Verdict on quality:** these are competent SAEs. The residual sites match or beat SAEBench's published BatchTopK
16k SAE at the same width and L0: FVE 0.877-0.995 against 0.868, and +0.01 to +0.26 nats against +0.14. The used MLP
sites recover 0.93-0.99 of the loss, against Bricken's 0.79 and far better than Cunningham's +0.47 nats. L0 sits on
k as expected. There are two weaknesses: half of each attention dictionary is dead (row 4), and mlp2 and mlp3 miss the
filed recovered bar of 0.95, falling to 0.80-0.82 at mlp3 on task text (row 5).

## Citations

- `bricken2023monosemanticity`: correct ("Towards Monosemanticity", Transformer Circuits Thread 2023, URL right).
  Apt as the generic SAE source, wrong as support for the base-SAE-on-fine-tune claim (row 10).
- `cunningham2023sparse`: correct (arXiv 2309.08600, ICLR 2024). Same two uses, same caveat.
- `bussmann2024batchtopk`: correct (arXiv 2412.06410). It is the right label for the architecture, but not the source
  of the training settings.
- Missing: SAEBench (Karvonen et al. 2025, arXiv 2503.09532), the actual recipe; Kissane et al. 2024 or Wang et al.
  2025 for SAEs applied to fine-tunes; optionally Kissane et al. 2024 for attention-output SAEs.

## Verdict

The code is a faithful, line-checked BatchTopK SAE on SAEBench's recipe, and the label "BatchTopK SAE (Bussmann)" is
right: there is no blocker. The SAEs are competent (residual FVE 0.88-0.995 at L0 80, matching SAEBench at the same
width), apart from half-dead attention dictionaries. The paper mislabels the sites ("before and after each MLP" where
the code uses attention and MLP outputs), never cites SAEBench or states a setting, and cites Bricken and Cunningham
for a base-SAE-on-fine-tune claim they do not make. All of this is fixable by relabelling and citing, with no re-run.
