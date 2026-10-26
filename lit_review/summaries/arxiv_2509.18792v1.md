# Beyond the Leaderboard: Understanding Performance Disparities in Large Language Models via Model Diffing

## Metadata
- **Authors**: Sabri Boughorbel, Fahim Dalvi, Nadir Durrani, Majd Hawasly (Qatar Computing Research Institute, HBKU)
- **Year**: 2025
- **Source**: arXiv preprint (2509.18792v1)
- **DOI/URL**: https://arxiv.org/abs/2509.18792 (URL derived from the arXiv ID in the file name; not checked against the live page)

## Summary
The paper applies model diffing with BatchTopK crosscoders to compare Gemma-2-9b-it with its SimPO fine-tune (and, in an appendix, a DPO fine-tune), using the shared base model Gemma-2-9b-pt as the anchor. It finds about 92 latents unique to the SimPO model and 113 unique to the instruct model, groups them into a 30-category capability taxonomy with Claude 3 Opus labels, and reports category-level shifts (safety, multilingual, instruction-following up; hallucination detection and self-reference down). The authors argue that these shifts explain benchmark and LMArena differences that leaderboard scores alone do not show.

## Key Findings
- Comparing the instruct model directly with its SimPO variant gave no significant difference (latent norm differences mostly 0.3 to 0.6). The authors then compared each model against the base model, which produced usable latent sets (Section 2.3).
- Reported latent-count changes (Table 2 and Table 4): safety and content moderation +32.8%, linguistic/multilingual +43.8%, and "template and instruction following" +151.7%. Hallucination detection -68.5%, model self-reference -44.1%, structured output -37.1%.
- The percentages come from small counts. For example, Template Following goes from 1.79% to 4.49% of the instruct-unique latents, and Hallucination Detection from 3.57% to 1.12%. On the numbers in Table 4, this is roughly 2 to 4 latents per category. Percentage changes of this size are fragile.
- LMArena style-controlled deltas are much smaller than raw deltas for several categories (Table 1). For Math the raw delta is +10 Elo and the style-controlled delta is -12. The authors take this as evidence that style accounts for much of the SimPO gain.
- The DPO appendix (Table 5) reports a different profile (more interaction-management and error-handling latents, less safety), which the authors read as support for the method being general.
- Crosscoder-based diffing is presented as task-agnostic and not requiring hand-built behavioural test sets, in contrast with probing (Appendix E). The authors did not test whether the identified latents cause the behavioural changes.

## Methodology
- Crosscoders (shared dictionary across two models, with per-model decoder directions) trained with BatchTopK (latent dimension 114,688, top-k = 100, learning rate 1e-4) on 200M tokens from FineWeb and LMSys-Chat, at layer 20 (Section 2.2).
- Latent Scaling (ν_ε, ν_r coefficients) is used to separate shared from model-specific latents, following Minder et al. (2025). Only latents that the BatchTopK training retains as causally distinct are analysed.
- The top-activating documents for each latent are labelled by Claude 3 Opus, grouped into 30 fine-grained categories and 7 classes (Appendix B). Category frequencies are compared between models.
- The analysis is descriptive: counts and normalised frequencies per category. No causal interventions (for example activation patching) are run on the identified latents in this paper.

## Relevance to Proposal
**Score: 5**

This is a direct application of the crosscoder model-diffing strand (strand 2) to a post-training update, which the proposal would otherwise have to reach from the parameter-decomposition side. The main value is the design lesson: the direct instruct-vs-SimPO comparison was too weak to show anything, and only a base-anchored comparison gave signal, which suggests a small post-training update may need the same anchoring. The paper gives no parameter-decomposition, weight-space, or update-projection analysis, and its evidence is correlational with small latent counts, so it does not test the commuting map Φ(D, δW).

## Limitations
- Only one model pair (Gemma-2-9b-it vs SimPO) and one DPO pair are studied (Limitations section), so generalisation is untested.
- The authors acknowledge the analysis "cannot definitively establish causal relationships between these differences and specific performance outcomes" (Limitations).
- Headline percentages rest on latent counts of roughly 2 to 9 per category (Tables 2 and 4). No uncertainty estimates, seed variation, or significance tests are reported.
- Category labels come from a single LLM (Claude 3 Opus) with no reported human validation or agreement check.
- The crosscoder is trained on a fixed 200M-token mixture at a single layer (20). Results at other layers or with other training mixtures are not shown.
- The latent-to-category mapping is subjective (for example, "Sexual Content Filtering" as the largest safety gain) and the link from a latent's top activations to the behaviour it causes is not tested.
- The "+151.7%" figure is a relative change on a small base and is presented in the abstract without that context.

## Key Quotes
> "Crucially, these shifts are only visible through model diffing, not from benchmark scores or leaderboard deltas, highlighting the need for deeper mechanistic diagnostics in evaluating LLM enhancements." (Section 1, Introduction)

> "once a model has undergone instruction tuning, further improvements like SimPO operate within a narrow behavioral subspace. They may modify surface-level generation preferences or alignment signals rather than introducing fundamentally new internal representations." (Section 2.3, Design Iteration)

> "it cannot definitively establish causal relationships between these differences and specific performance outcomes." (Limitations)
