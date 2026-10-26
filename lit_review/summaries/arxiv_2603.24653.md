# From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition

## Metadata
- **Authors**: Francesco Gentile, Nicola Dall'Asen, Francesco Tonini, Massimiliano Mancini, Lorenzo Vaquero, Elisa Ricci
- **Year**: 2026 (arXiv ID 2603.24653, March 2026)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2603.24653 (project page: https://frangente.github.io/SITH)

## Summary
The paper introduces SITH, a data-free, training-free method that interprets CLIP's vision transformer directly in weight space. It takes each attention head's value-output (VO) matrix, factorises it with SVD, and labels each singular vector with a sparse, coherent set of text concepts using a new decomposition algorithm (COMP). The paper also uses the decomposition to study fine-tuning, reporting that full fine-tuning and LoRA mostly rotate a stable singular basis slightly rather than creating new features.

## Key Findings
- Per-singular-vector concept labels (via COMP, a sparse non-negative pursuit with a semantic coherence term) are more complete than top-k nearest concepts and more coherent than plain NNOMP. Zero-shot accuracy barely drops when singular vectors of layer 23 are replaced by their reconstructions (Fig. 4).
- Image-grounding check on CC12M: a vision-LLM judge rates SITH concept sets as matching the top-retrieved images, and the match is highest in the last layer (Fig. 5-6). This is an LLM-judged result, not a human study.
- Editing singular values with SITH concept labels gives small gains: Waterbirds worst-group accuracy 47.9 to 70.6 (vs. 68.0 for TextSpan), NSFW retrieval improvements, and up to about 1 point zero-shot gain on three datasets (Tables 3-5). Random scaling of the same number of singular values degrades or does not help.
- After fine-tuning CLIP ViT-L/14 on Flowers 102, Oxford Pets and CUB-200 (full FT and LoRA), the normalised spectral cosine similarity between pre- and post-fine-tuning right singular vectors stays high across the last four layers. The authors read this as a subtle subspace shift (Fig. 7, Sec. 7).
- The singular vectors of the task matrix ΔW = W_ft - W_pre (task singular vectors) are mostly labelled with concepts from the fine-tuning domain (Fig. 8). Examples: plant terms for Flowers 102, breed names for Oxford Pets, bird species for CUB-200.

## Methodology
- Analysis is on the last four blocks of OpenCLIP ViT-L/14 (24 layers, 16 heads, rank 64 per VO matrix). VO matrices are split per head, and the right singular vectors are projected into CLIP's joint space (with a modality-gap correction from Bhalla et al.).
- COMP extends NNOMP by adding a coherence term to the greedy selection score (weight λ in {0.2, 0.3, 0.4}, sparsity K in {5, 10, 20, 50}). The concept pool is ConceptNet 5.5. An LLM (GPT-5-mini) provides the interpretability ratings (5-point Likert), the relevance judgements for editing, and the fine-tuning alignment labels.
- Fidelity is cosine similarity between each singular vector and its concept reconstruction. Interpretability is an LLM-as-judge score. Fine-tuning is compared via spectral cosine similarity of right singular vectors (pre vs. post) and via task singular vectors of ΔW.

## Relevance to Proposal
**Score: 5/10**

This is the closest weight-space analogue to the proposal's question of how a fine-tuning update δW sits relative to the existing decomposition. It reports that fine-tuning mostly keeps the pretrained right singular basis and that ΔW's singular vectors are domain-aligned, which bears directly on strand 3 (spectrum and alignment of δW with pretrained singular vectors) and on the superficial-alignment and "reuse vs. create" question in strand 4. It is not a parameter decomposition in the VPD/SPD/APD sense: it has no learned causal-importance function, no subcomponent-level faithfulness check under training, and it studies one ViT on classification tasks, so it does not test the proposal's commuting map Φ(D, δW).

## Limitations
- Single architecture and family (OpenCLIP ViT-L/14, vision side only); only the last four layers are analysed for fine-tuning. FFN and QK components are explicitly left to future work.
- Fine-tuning study uses three fine-grained classification datasets, one model, and no seed variance is visible in the main text. "Stable basis" rests on one normalised similarity metric, whose sensitivity to small rotations is not calibrated in the text.
- Interpretation quality is judged by GPT-5-mini (and a vision-LLM for grounding), not by humans. Concept labels depend on the ConceptNet 5.5 dictionary.
- Downstream gains are small (about 1 point on the zero-shot benchmarks). Fig. 4 and Table 2 show that some singular vectors carry non-semantic "noise" that no concept set captures, so reconstruction is never complete.
- Input-independent analysis of weights: the paper does not connect singular vectors to activations or to a causal account of the head's behaviour on data.
- Extraction note: Section 6 ("Interpreting Model Adaptation Techniques") has only a heading in the source text, and the appendix (supplementary material, lines 439 onward of the source file) was not reviewed. Details reported above are from the main body.

## Key Quotes
> "We show that SITH yields coherent, faithful intra-head explanations, validated through reconstruction fidelity and interpretability experiments." (Abstract)

> "indicating that fine-tuning does not drastically alter the learned semantic basis, but rather applies a slight modification to the pre-trained weights." (Sec. 7, Conclusion)

> "the majority of the task singular vectors are indeed aligned with the fine-tuning task across all datasets and adaptation methods." (Sec. 5.3 / Fig. 8 discussion)

> "This is likely due to certain singular vectors encoding not only semantic concepts, but also non-semantic or task-irrelevant 'noisy' information that cannot be linked to human-interpretable concepts." (Sec. 4.1)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
