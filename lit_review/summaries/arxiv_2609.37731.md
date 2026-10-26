# Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space

## Metadata
- **Authors**: Tue M. Cao, Lisiane Pruinelli, My T. Thai (University of Florida)
- **Year**: 2026 (arXiv ID 2609.37731; date not stated in the extracted text)
- **Source**: arXiv preprint
- **DOI/URL**: arXiv:2609.37731; code at https://github.com/tue147/weights-read-write

## Summary
The paper introduces ASPD (Activation-Supported Parameter Decomposition), a method that decomposes a weight matrix into rank-1 components while jointly learning a shared sparse code over activation features, so each weight component is tied to the activation feature it reads or writes. It adds a local reconstruction loss at the analysed matrix's output, which the authors argue makes decomposition scale to deeper pretrained LLMs where VPD becomes uninterpretable. Evaluated on GPT-2 small, Gemma-2-2B and Qwen-3-8B, ASPD beats VPD on automated interpretability, diversity, meaning-matching and weight-editing localization, and is applied to the IOI circuit.

## Key Findings
- On Qwen-3-8B (attention O matrix, layer 17), ASPD reaches Intruder interpretability 0.57 and overlap 0.03, versus 0.22 (near the 0.2 chance level) for VPD. The authors' own VPD reimplementation is the comparison, so the gap partly reflects their setup.
- Adding the internal reconstruction loss to VPD improves its interpretability and diversity on all three models, but the authors say it is not sufficient to match ASPD on editing or meaning localization.
- Single-target weight edits remove selected components from the frozen weights. On Qwen-3-8B, ASPD localization ratio is 6.7x random versus 0.9x for VPD. These are measured against the authors' own independent SAE, not ground truth.
- Trained jointly over all 72 GPT-2 small matrices (442k components), the method recovers parts of the known IOI circuit (induction, duplicate-token, S-inhibition, name movers) without circuit labels. Some key claims are checked by probes and one direct weight edit (Figure 2).
- The authors name parameter diffing (base vs fine-tuned mechanisms) as a possible application but explicitly leave it to future work. No fine-tuning or checkpoint-tracking experiment is reported.

## Methodology
- Each weight matrix W is written as C rank-1 components u_c v_c^T with a causal-importance gate. ASPD ties the gate of component c to the same latent coordinate that defines activation feature c, learned by a per-token BatchTopK sparse autoencoder on residual-stream activations.
- Training combines an activation reconstruction loss (L_act), an internal reconstruction loss at the output of the target matrix (L_internal), and the sparsity constraint. VPD's adversarial, parameter-closeness and ablation losses are dropped because the authors found them mixed or negative.
- Evaluation: Intruder score (LLM judge, 200 components); Jaccard overlap of firing tokens (500 components); Matching score from attribution patching against an independently trained SAE (200 pairs, LLM judge); weight-edit localization on 50 target sets over 10^6 tokens, normalized by random-edit baselines. Results report 95% CIs.
- Models: MLP-in layer 0 of GPT-2 small, MLP-out layer 13 of Gemma-2-2B, attention O layer 17 of Qwen-3-8B; decompositions trained on 2B tokens.

## Relevance to Proposal
**Score: 5**

This is a new parameter-decomposition method that claims to beat VPD on the scale where VPD breaks down, so it bears on strand 1 (method choice, brittleness, and whether decompositions are usable on real models). It does not address the core question, which concerns carrying an update δW through an existing decomposition, and it contains no stability, seed, or cross-run identity results. Its explicit future-work pointer to parameter diffing is the closest hook to the proposal.

## Limitations
- The paper does not evaluate ASPD on fine-tuned or post-trained models. The parameter-diffing direction is left as future work.
- Seed variance and cross-run component matching are not reported in the main text (the appendix was not fully read), so stability of the decomposition is unassessed.
- Interpretability and meaning-matching depend on an LLM judge and on a separate SAE, so the metrics may inherit SAE quality limits.
- VPD's own losses were dropped, so the comparison is with a modified VPD, not the original recipe. The "VPD + internal" ablation is also weaker than full ASPD.
- The IOI mechanistic story is qualitative in places; the authors note some components are only partly understood, and H5.5 has a broader "token after a name" feature than the specific "token after John".
- The GPT-2 model-wide decomposition (442k components) is not shown to scale to Qwen-3-8B in the mechanistic analysis.

## Key Quotes
> "Equation 7 instead supervises the decomposition directly at the output of the weight matrix being analyzed, avoiding the need to propagate the learning signal through all subsequent layers." (Section 3.3, p. 5)

> "One potential application is _Parameter Diffing_ (identifying what changed in the _mechanisms_ of the fine-tuned compared to the base model) by jointly decomposing activation diffing ... We leave this direction for future work." (Section 3.3, p. 5)

> "On Qwen-3-8B, ASPD remains interpretable (0.57) and diverse (0.03), whereas VPD is near chance (0.22)." (Section 4.1, p. 6)

> "Bushnaq et al. (2026) are only able to decompose 4-layers toy model, while in larger models, the decomposition is uninterpretable and has poor editing quality." (Appendix A, Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
