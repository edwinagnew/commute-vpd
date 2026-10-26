# Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis of Representational Divergence in Fine-Tuned Language Models

## Metadata
- **Authors**: Prasanth K K (independent researcher)
- **Year**: 2026 (dated May 2026)
- **Source**: arXiv preprint (single-author, not peer reviewed as far as the document shows)
- **DOI/URL**: arXiv:2605.28896 (ID taken from the filename; https://arxiv.org/abs/2605.28896 not checked against the live page)

## Summary
The paper trains sparse autoencoders (SAEs) on the activation change that a LoRA adapter adds to the residual stream (h_delta = h_adapted - h_base) for Gemma-2-9B at ranks 4 to 32, and compares those features with the pretrained Gemma Scope dictionaries for the base model. It reports that the adapter-specific features have weak cosine, principal-angle and CKA alignment with base features, and that the adapter-specific SAEs reconstruct the delta better than Gemma Scope does. The adapters are trained on Alpaca for 3 epochs, and the author states that they barely change outputs relative to the base model, which limits what the geometric claims can support.

## Key Findings
- Mean maximum cosine similarity between delta-SAE decoder directions and Gemma Scope directions is about 0.071 (Table 9). The author calls this "barely above random (about 0)"; for random unit vectors in 3584 dimensions the typical value is about 0.017, so the gap is smaller than the framing suggests. Only 0.01-0.02% of features exceed 0.7.
- Principal angles between top-256 subspaces average about 74 degrees with 0% of directions below 20 degrees (Table 10).
- Linear CKA between h_base and h_delta is lowest at layer 18 (about 0.05-0.08) and higher at layers 32-38 (about 0.55-0.69) (Table 11).
- Gemma Scope reconstruction error on the delta is above 1.0 at every layer and rank (Table 5), and the delta SAEs have lower held-out error than Gemma Scope in all 24 conditions, by 46-86% (Tables 6-7).
- Feature density rises with rank and depth, and geometric measures are roughly rank-invariant (Section 5.4, Section 7.3).

## Methodology
Four LoRA adapters (r = 4, 8, 16, 32; alpha = 2r; q, k, v, o and projection modules; lr 2e-4; 10,000 Alpaca samples; 3 epochs; seed 42) are trained on Gemma-2-9B. Residual-stream deltas are collected with forward hooks at layers 5, 10, 18, 22, 32 and 38 on a 2,000-sample probe set and evaluated on 200 held-out samples. Per (rank, layer) pair, a ReLU SAE (width 16384, L1 coefficient 0.15 after RMS normalisation) is trained on the deltas. Three measures are then compared against Gemma Scope: max cosine similarity of decoder directions, principal angles of top-256 decoder subspaces, and linear CKA on activations. Overlap of active features between delta and base is also reported. Experiments were run on a single Apple M4 machine.

## Relevance to Proposal
**Score: 3**

The paper touches strand 2 (SAE-based model diffing and whether base-model dictionaries transfer to fine-tuned models) and strand 3 (LoRA geometry relative to base representations), but it does not decompose parameters, does not test any map from a decomposition of W plus a small update to a decomposition of the updated model, and does not study training trajectories. Its most relevant idea for us is the negative result that base-model SAE dictionaries fail to cover the update, which bears on whether a fixed decomposition can carry a post-training update. Because the adapters are reported to be behaviourally near-identical to the base model, the paper cannot tell us whether its geometry reflects a capability that post-training creates or just a small, noisy perturbation.

## Limitations
- The author states that the adapters "produce outputs identical to the base model on test prompts" (Section 7.4). The delta being analysed may therefore be near-noise, and the paper cannot show that the measured features correspond to anything the model uses.
- No base-base SAE baseline: two independently trained SAEs on base activations are not compared, so the 0.071 cosine similarity has no reference point (Section 7.4, acknowledged by the author).
- The delta SAEs are trained on the deltas and then evaluated on held-out deltas from the same adapter. Beating Gemma Scope on the delta is expected from in-distribution training alone and does not show that the features are "genuine" adapter structure, as the paper claims in Section 5.3.
- Gemma Scope is applied to inputs whose scale and distribution differ from its training data. The reconstruction-error comparison therefore partly measures that mismatch.
- Single seed (42), single dataset (Alpaca), 3 epochs, ~1M tokens for SAE training.
- Approximately 6 layers and 4 ranks; no training-trajectory analysis and no causal (steering or ablation) test.
- Several claims are stated more strongly than the evidence supports: the "monitoring gap" and safety-auditing implications rest on geometric separation that is not validated against behaviour.
- The reported "weakly aligned" fraction (above 93%, Table 8) depends on the activation-level threshold definition, which is not fully specified.
- Limited methodological detail on the L1 tuning sweep (Table 12 shows large L0 variation) and on the flat-looking Section 6.3 subspace-dimension robustness check.

## Key Quotes
> "The adapters produce outputs identical to the base model on test prompts, suggesting the base model's strong priors dominate. This precludes causal validation via activation steering and is the primary limitation of the current work." (Section 7.4, Undertrained adapters)

> "Reconstruction error exceeds 1.0 at every layer and rank, meaning the Gemma Scope SAE's approximation error is larger than the delta signal itself." (Section 5.2)

> "We lack a base-base SAE similarity baseline (two independently trained SAEs on base model activations), which would allow direct comparison to assess whether 0.071 cosine similarity is genuinely low or attributable to SAE training variance." (Section 7.4, No baseline comparison)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
