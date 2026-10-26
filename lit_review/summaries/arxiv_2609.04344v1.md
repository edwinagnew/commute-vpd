# SharedSAE: One Feature Dictionary Across Language Models

## Metadata
- **Authors**: Daniil Ognev (MBZUAI), Célian Vasson (Sorbonne Université), Lijie Hu (MBZUAI), Kentaro Inui (MBZUAI / Tohoku University / RIKEN AIP), Benjamin Heinzerling (Tohoku University / RIKEN AIP)
- **Year**: 2026
- **Source**: arXiv preprint (2609.04344v1)
- **DOI/URL**: https://arxiv.org/abs/2609.04344 (URL inferred from the arXiv ID in the file name; not confirmed from the document header)

## Summary
SharedSAE trains a single sparse autoencoder dictionary jointly across four ~1B-parameter base language models from different families and tokenizers, using per-model encoder/decoder pairs around one shared set of latents. The authors argue that the shared latents are aligned across models (they select the same latents, have correlated magnitudes, and carry similar descriptions), that a single model can run inference alone, and that a new model (including a post-trained instruct checkpoint and an unseen architecture) can be attached to the frozen dictionary without changing existing latents or their labels.

## Key Findings
- Shared dictionary reconstruction is close to dedicated per-model SAEs: mean FVE 0.803 under solo TopK vs 0.831 for dedicated SAEs (Table 1), with spliced CE-recovered 0.957 vs 0.963.
- Under solo TopK, the shared latents are used by all four models far more evenly than post-hoc matched dedicated SAEs, USAE, or SPARC (Figures 3-5). Median per-latent cross-model magnitude correlation is 0.79 for SharedSAE vs 0.43 for post-hoc matching, 0.00 for USAE and SPARC (Figure 1, Figure 7).
- Shared-vote labels for shared latents achieve FADE scores comparable to dedicated SAE labels within the retained subset, and labels validated in one stream transfer to others (Table 3). The authors qualify this: faithfulness scores are lower and less consistent, and functional alignment is only partial (mean input/output score correlations ~0.52/0.48).
- Frozen-dictionary adaptation: a held-out model's encoder, decoder and biases are trained against the frozen latents. Solo adapted models reach roughly 0.98-1.01 of their dedicated SAE FVE (Table 4, Table 5). Llama-3.2-1B-Instruct, a post-trained model whose base sibling is in the seed, adapts to 1.006 of its own dedicated SAE, and Qwen3.5-2B (hybrid linear attention) and Hunyuan-1.8B also adapt.
- Adaptation beats the linear-alignment (Atlas-style) and decoder-only attach baselines on the sharedness criteria in the authors' experiments (Table 5).
- Authors explicitly say the semantic alignment of the latents was never directly optimized, and that two of the three sharedness criteria are partly optimized by training.

## Methodology
- Architecture: TopK SAE (Gao et al., 2024) with one dictionary of L latents and a separate encoder and decoder per model. Per-model codes are normalized (vote norm) and summed, and one joint TopK chooses the latent indices for all models. Cross-model reconstruction uses a magnitude rescale, and "vote-source dropout" (p=0.5) trains each model to select latents alone for solo inference.
- Data: an aligner cuts token streams from different tokenizers into windows that decode to identical text, giving aligned activation tuples across the four models, read at ~85% depth on FineWebEdu.
- Evaluation: reconstruction (FVE), spliced CE and KL, selection overlap (per-token and per-latent top-50 activating positions), magnitude correlation, Neff-style usage measures, FADE automatic labelling (clarity, responsiveness, purity, faithfulness), a targeted concept-family probe search, and causal output-vocabulary and steering probes.
- Baselines: post-hoc matched dedicated TopK SAEs (an upper bound on one-to-one matching), USAE, SPARC, and a crosscoder.
- No parameter decomposition, no weight-space analysis, and no fine-tuning-update analysis is performed; the objects are activations and SAE latents.

## Relevance to Proposal
**Score: 4**

The paper is relevant to strand 2 of the proposal (interpretability that tracks models across training and variants) and to the "continual interpretability" framing: it shows that a fixed dictionary can be kept while a new checkpoint, including a post-trained instruct model, is attached and retains similar latent semantics. It is not about parameter decomposition (VPD/SPD/APD) or about commuting a weight update through a decomposition, so none of the proposal's core maps Phi(D, deltaW) are tested; it is at best a conceptual neighbour, offering an empirical analogue (frozen dictionary plus a new encoder/decoder pair) and a set of alignment metrics that could be borrowed.

## Limitations
- Authors state results depend on aligned data volume, training budget, and the read layer; depth is fixed at ~85% and mid-depth gives different cross-model reconstruction (Table 13), so depth is a free parameter.
- Only four ~1B base models on English FineWebEdu were tested; scale, other languages, and other families beyond the roster are untested. Post-training is probed with one instruct checkpoint (Llama-3.2-1B-Instruct) and adapts onto the base-model dictionary, not a fine-tune of a trained model's own dictionary.
- Authors acknowledge that the metrics are mostly proxies, and that two of the three sharedness criteria are partly optimized directly by the training objective.
- Cross-reconstruction FVEs use an oracle per-token rescale in the main table; the deployable scalar variant concedes a few points (for SPARC, the drop is large: 0.732 to 0.643 on the joint TopK mean).
- Labels rely on an automatic labeller (FADE) and a clarity threshold of 0.9; retained coverage is only 4-30% of label-stream pairs. Faithfulness is low (0.2-0.4).
- Functional alignment is partial, and the paper explicitly does not establish latent-level causal equivalence.
- Hyperparameters were varied one at a time, not searched jointly; the authors do not claim their configuration is optimal.
- Not peer reviewed at time of summary (arXiv preprint).

## Key Quotes
> "Unlike the closest prior method, which discards activation magnitudes and requires all models at inference, SharedSAE instead normalizes only selection scores, preserving magnitudes, and uses model dropout for single-model inference." (Abstract)

> "Llama-3.2-1B-Instruct, a post-trained checkpoint whose base sibling is already in the seed, to the frozen dictionary: deployed solo it reaches 1.006 of its own dedicated SAE (Table 4), so the labels learned on base models carry over to this instruct model without loss of reconstruction." (Section 5)

> "The semantic alignment of the latents, which is the final goal of this interpretability research, was never directly optimized." (Section 4)

> "A single measure of alignment that depends on no part of the shared machinery remains future work." (Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
