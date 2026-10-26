# Sparse Weight Decomposition for Efficient Circuit Extraction

## Metadata
- **Authors**: Chuanhao Yan, Xuhan Huang, Yawen Duan, Zhenfei Yin, Hang Zhao, Bryan Dai, Jie Fu (IQuest Research; Safe AI Forum; Oxford; Stanford; Tsinghua)
- **Year**: 2026
- **Source**: arXiv preprint (2608.03913); code at github.com/veri-safe/SWD, model at huggingface.co/veri-safe/SWD
- **DOI/URL**: https://arxiv.org/abs/2608.03913

## Summary
Sparse Weight Decomposition (SWD) post-hoc factorizes each pretrained dense linear projection W into two sparse factors, W ≈ AB, so that each shared intermediate coordinate ("bottleneck unit") is a rank-one read-write path with few nonzero input and output edges. These units are then scored, pruned and ablated with the standard task-based circuit-extraction workflow, without training a separate replacement network. The authors report that SWD matches the held-out replacement fidelity of Transcoders and VPD variants with under 1% of their data, and reaches the same circuit sufficiency and necessity targets with fewer active edges on GPT-2, Qwen2.5 and Qwen3.5-27B.

## Key Findings
- Single-matrix replacement (GPT-2 Small layer-8 MLP output projection): SWD (s=0.5) reaches low CE delta after a few thousand tokens, versus about 10^6 tokens for the trained baselines. VPD-KL could not be CE-matched and was excluded from circuit comparisons (authors' result).
- At matched pre-pruning CE, SWD needs fewer active read/write edges than Transcoder and VPD-Recon-CI to reach the same sufficiency and necessity on greater-than, IOI, docstring and gendered-pronoun tasks. The advantage holds with zero ablation as well as mean ablation (Appendix C.3).
- Exact dense factorizations (full-rank SVD, and random orthogonal B) reproduce W exactly but need more active edges than SWD, which the authors take as evidence that the sparse read/write structure, not factorization alone, drives the gain.
- Full-model replacement of all 48 attention and MLP matrices in GPT-2 Small: plain SWD gives CE 3.90 (accumulated error). Fixed-support fine-tuning of the nonzero factor values (SWD-FT) brings CE to 3.44, versus 3.45 for a matched-budget weight-sparse pretrained model, using 20.6M tokens versus 2.884B.
- Zero-data variant (G = I, no activations) keeps weights closer to W (relative Frobenius error 0.366 vs 0.415 for calibrated SWD at s=0.75) but is worse on behavioral CE. Its circuits remain useful on the task set (Appendix G.2).
- Qualitative: a semantic audit of a GreaterThan circuit, a rank-one targeted edit via one bottleneck unit (lower side effects than random-unit and LoRA-rank-4 controls on seven prompts), and a layer-9 head-3 query-unit ablation.

## Methodology
- Objective: minimise reconstruction error on calibration activations, ||(W - AB) on h||, under a fixed nonzero budget K, using the Double Sparse Factorization heuristic (ADMM alternating updates with hard thresholding, then a refit of B on frozen supports). Sparsity is reported as s = 1 - K/||W||_0.
- Circuit protocol: bottleneck units ranked by positive first-order task-margin attribution, nested top-k prefixes evaluated on a held-out split, with keep (sufficiency) and ablate (necessity) interventions. Cost is measured in selected units and active edges. Calibration data: FineWeb-Edu.
- Baselines: Transcoder (Dunefsky et al.), sparse pretraining (Gao et al. 2025), VPD (Bushnaq et al. 2026) in KL and reconstruction-plus-CI configurations.
- Controls: exact SVD and Random-B factorizations on the same surface.

## Relevance to Proposal
**Score: 5**

SWD is a non-trained, post-hoc decomposition of a dense weight matrix into addressable rank-one components. It is a plausible cheap baseline or comparator for the proposal's "re-decompose W+δW from scratch" target, and its zero-data variant is explicitly suggested for per-step analysis of training, which bears on tracking decompositions along a training trajectory. It does not address the core question, which is how a post-training update δW maps onto an existing VPD decomposition D. It also does not study fine-tuning updates, and its components are fixed-support factors, not a learned causal-importance decomposition like VPD's.

## Limitations
- Authors state that SWD circuits are local, non-unique and protocol-specific. Factor identifiability is not established, and results can depend on the objective, calibration distribution, initialisation and sparsity budget.
- The approximate replacement AB is not guaranteed to agree with W on every input. Matched CE and reconstruction error are averages.
- Full-model replacement is evaluated only on GPT-2 Small. The Qwen3.5-27B experiment covers one matrix only.
- The VPD comparison uses a reconstruction-based variant (VPD-Recon-CI) for circuit results, because VPD-KL could not be CE-matched. The baseline setup is therefore one the authors chose and may not favour VPD.
- The "training trajectory" use of the zero-data variant is explicitly left to future work; no training-dynamics results are reported.
- Circuit evaluation uses first-order attribution and mean or zero ablation; the authors cite evidence that circuit conclusions shift with these choices.
- The semantic labels of bottleneck units come from six model-labelled units and are not a systematic audit.
- Single-run, single-seed results are not reported for stability; I did not find seed-variance analysis in the parts read.

## Key Quotes
> "Across single-matrix replacements, SWD matches the held-out fidelity achieved by Transcoder and other strong baselines while using less than 1% of the data that those baselines use to train their replacements." (Abstract)

> "Because zero-data SWD requires only the current weights, it can in principle be applied at every training step without collecting calibration activations, enabling step-by-step mechanistic analysis of how circuit structure emerges and evolves during training. We leave such training-trajectory analysis to future work." (Section 3.6)

> "Identifiability of a two-factor sparse matrix decomposition requires additional structural conditions and is, even then, only defined up to unavoidable scaling and permutation symmetries... SWD does not establish such conditions for its approximate learned decompositions, so the resulting bottleneck units can depend on the objective, calibration distribution, initialization, and sparsity budget." (Section 5, Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint with no comment section.
