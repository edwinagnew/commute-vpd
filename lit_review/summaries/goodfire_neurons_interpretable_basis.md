# Neurons as an Interpretable Basis (Goodfire)

## Metadata
- **Authors**: not found in the fetched text (Goodfire research article; the companion essay is by Nick Cammarata, see below)
- **Year**: 2026
- **Source**: Goodfire research article (web)
- **DOI/URL**: https://www.goodfire.com/research/neurons-as-an-interpretable-basis/article.md
- **Companion essay**: "Neural Networks Are Already Clean" is summarised in `lit_review/2026-10-29_baselines_notes.md` (top section, "already-clean.goodfire.com"); not repeated here.
- **Fetch caveat**: WebFetch passes the page through a small model, so the wording below is that model's quotation of a 416,576-character article; bytes 400,000-416,576 were not read, and the appendix was read only through excerpts. Numbers should be re-checked against the page before a filed prediction uses them.

## Summary
The article argues that the neurons of a large MoE language model are already an interpretable basis, with no learned dictionary needed. Each neuron has a positive and a negative pole whose firings split into a few "subcircuits" (modes) described from their inputs and readers. They test this by describing neurons, scoring the descriptions, and by sparsifying the model to its strongest neurons and lightly fine-tuning it back.

## Key Findings
1. Layer top-320 keeps 320 of 4,096 routed neurons per token (7.8%; 92.2% sparsity) at web-text loss 1.940 against base 1.886 (+0.054; KL 0.140), 8.4M tokens.
2. Expert top-64 (about 12% of routed neurons): loss +0.066, KL 0.152. Expert top-32 (about 6%): loss +0.146, KL 0.263.
3. Non-reasoning benchmarks stay near base (layer top-320: ARC-Challenge +0.3, WinoGrande -2.2, MMLU -1.2); reasoning and code fall (GPQA-Diamond -11.1, MBPP+ -20.9). Sparsified models loop and over-check; restoring double-negative activations restored reasoning.
4. A light fine-tune under the mask restores function (details in the Fine-tuning section below).
5. Detection 95% against random text (chance 50%), 89% against text from the same expert; simulation recovers 73% of the damage that a perfect reader would restore (the validation section elsewhere says the ideal predictor recovers 96%); input synthesis activates 94% of neurons beyond the maximum seen in the 10B-token corpus.

## Methodology
**Model and data.** "Qwen3.6-35B-A3B", an MoE with 40 layers, 256 experts per layer, 8 routed experts per token plus one always-on shared expert, 512 neurons per expert; 5,468 neuron poles in the expanded study set. Shared-expert neurons (about 0.4% of 5.26M neurons) are left unchanged. Corpus: about 10B tokens of pretraining, reasoning, chat and math data, mixed pre- and post-training; 1,000 examples per neuron from each of 14 activation-percentile bands; a stratified 2,734 neurons get a further 30,000 examples from p95/5 and stronger. Judges and tools: Claude Opus 5.5 (detection judge), "Jev" (simulator and text judge), Qwen-3-Embedding-4B (mode description embeddings). Subcircuits: each firing's upstream gate and up inputs are profiled, clustered by Leiden community detection, and overlapping groups merged (median 19 per pole). Descriptions come from an LLM judge.

**1. Top-K sparsity (what we need for the 4-layer 67M model).**
- Activation: "the output of its SwiGLU activation function, a" (SiLU(gate) x up); the scored quantity is "the combined scalar a . w_router". Zeroing is done "before the down projection". Whether this is the product before the down projection exactly is implied by that wording, not stated as a definition.
- Layer top-K (main method): pool all 8 x 512 = 4,096 routed neurons of a token, "each token keeps the 320 routed-expert cells with the largest |activation| x routing weight". Absolute value, router-weighted, one pool per token and layer, K = 320 fixed. Mask computed per token at inference, no training needed. The article gives no derivation of why 320; no sweep over layer-K is in the text I read.
- Expert top-K: "only the 64 largest-|activation| cells per routed expert kept" (also 32), raw activation, "without the router weight scaling".
- Other neurons: zeroed in the top-K method ("zero the rest before the down projection"). Mean ablation is used elsewhere: the percentile-band ablations, and the continuous-generation experiment ("mean-ablate every neuron outside the kept set - at the current token and at every earlier token"; the current-token variant ablates only the current token). The per-neuron mean is used in a reasoning-patching control (Figure 4).
- Controls: Random-64 (64 of 512 cells per routed expert slot, uniform per token) and Bottom-448 (everything except the 64 largest). Untrained benchmark change in SuperGPQA: top-64 -15.2, top-320 -13.0, random-64 -44.5, bottom-448 -44.6.
- Kept: see Key Findings 1-3. Per-K loss at K other than 320, 64 and 32 (expert): not found.

**2. Percentile bands and a neuron's job.** A neuron's job is described from its 1,000 strongest router-weighted activations in the 10B-token corpus (negative poles use "the corresponding negative tail"), plus 14 percentile bands of examples per neuron. Mean ablation of central bands "between the positive and negative extremes", computed without router weighting (Table 1): P25-P75 (about 50% of cells) loss +0.010, KL 0.033; P10-P90 (about 79%) +0.040, KL 0.088; P5-P95 (about 89%) +0.091, KL 0.162; random-64 and inverse-of-top-64 controls about +2.3 and +2.7 loss. Benchmarks: P25-P75 within about +-2 points on most tasks; P5-P95 GSM8K -9.3, MBPP+ -14.8. Most retained router-weighted activations lie in the p97/3 range, used for later analyses. Detection is reported separately for five activation-percentile bands; simulation scores the 99.9th and 99.99th percentiles and the top 1,000. Whether a single neuron keeps its function after training is checked only as below.

**3. Fine-tuning / restoration (Extended Sparsity Results appendix).** "we follow Shabalin & Piotrowski and perform light fine-tuning on the layer top-320 model". "We apply the same mask on the model during every training step and all weights are trained." About 100M tokens, "less than 0.001% the size of the model's original pretraining and post-training". SFT: distillation from the unmasked model's own thinking traces plus 20% general web and reasoning text; "91M tokens of this dataset over 1,000 steps"; KL loss with the unmasked model as teacher. Then RL: 1,024 knowledge prompts x 4 samples per round, reward for finishing and matching the gold answer, policy gradient plus a KL penalty (weight 0.1) to the clean model, learning rate 1e-6, 10 steps per round for 10 rounds, step 100 best checkpoint (9M tokens of RL per the abstract). SFT recovered about 72% of the SuperGPQA drop; after RL SuperGPQA 45.4 against clean 44.8. SFT then RL v1 vs untrained base, top-320: SuperGPQA +0.2, MMLU-Pro -2.9, GSM8K strict -0.3. Abstract/main text: "across benchmarks we were able to see full restoration of function", and "the neurons retained their original functions, remained interpretable, and had the same numbers of modes."
- How retention of function was measured: by "running a token harvest on the trained model and checking that the most-changed neurons kept their semantic meanings across percentiles"; no quantitative retention score found. Learning-rate for SFT: not found. Another passage: "Top-K MLPs perform better than transcoders (on KL divergence) at matched sparsity after model finetuning."

**4. Validation.** Detection (LLM judge picks which of a pair of snippets makes the pole fire; macro-accuracy by percentile band; random-text and same-expert negatives); mode classification (inputs vs readers vs text descriptions, macro-F1); simulation (a simulator predicts activations; damage as KL divergence summed over the passage when the neuron is set to zero on every word piece routed to its expert, against an ideal predictor); input synthesis (generated text exceeds the corpus maximum for 94% of neurons); ablation tests of top-K models and of the Q3 (double-negative) quadrant; sparse neuron circuits and RelP circuits (appendix; neurons needed grow about linearly with text length); comparison against transcoders.

**5. Models and datasets.** One model only, Qwen3.6-35B-A3B; web-text loss on 8.4M tokens; benchmarks ARC-Challenge, WinoGrande, MMLU, MMLU-Pro, GPQA-Diamond, SuperGPQA, GSM8K, MBPP+. No small dense model and no 4-layer model.

## Relevance to Proposal
**Score: not assigned (no proposal text given in the brief).** For the planned 4-layer 67M experiment: the layer top-K recipe (absolute value times routing weight, one pool per token and layer, zero the rest before the down projection) maps directly only if the 67M MLP has a gating or SwiGLU-like hidden; the 67M's routing weight and activation function are not covered here. The fine-tune is under the mask, with all weights trained and a KL-to-clean teacher, so it tests recovery with sparsity kept on, not a commute test of an update.

## Limitations
- MoE with shared expert; only layer top-320 reported at layer level in the text I read; no K sweep found.
- Retention of neuron function after fine-tuning is shown by token harvests and semantic checks, with no number found.
- Zeroing for top-K but mean ablation for bands and continuous generation: they are not interchangeable.
- Sparsified models lose reasoning (GPQA-Diamond -11.1, MBPP+ -20.9) before fine-tuning.
- Companion essay is labelled a working hypothesis.
- Not read: characters 400,000-416,576 and anything the small fetch model omitted; author list and the main-text sparsity figure captions.

## Key Quotes
> "each token keeps the 320 routed-expert cells with the largest |activation| x routing weight." (Extended Sparsity Results, top-K definitions)

> "We apply the same mask on the model during every training step and all weights are trained." (Extended Sparsity Results, fine-tuning)

> "the neurons retained their original functions, remained interpretable, and had the same numbers of modes." (Sparsity results)

> "A light fine-tune (less than 0.001% the size of the model's original pretraining and post-training) suffices" (Abstract)
