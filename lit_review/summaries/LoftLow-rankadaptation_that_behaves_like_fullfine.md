# LOFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning

## Metadata
- **Authors**: Nurbek Tastan, Stefanos Laskaridis, Martin Takáč, Karthik Nandakumar, Samuel Horváth
- **Year**: 2026 (ICLR 2026 conference paper)
- **Source**: ICLR 2026 (published as a conference paper)
- **DOI/URL**: https://github.com/tnurbek/loft (code); no DOI in the supplied text

## Summary
LoFT is a LoRA-style parameter-efficient fine-tuning method that tries to make the low-rank update follow the same optimizer trajectory as full fine-tuning with AdamW. It goes beyond approximating the gradient inside the low-rank subspace: it also projects the first and second Adam moments into the current subspace, reconstructs the full-model update and projects it back, and calibrates momentum across changing subspaces. The authors report that this closes much of the gap to full fine-tuning and removes the need to tune the LoRA scaling factor alpha, at no inference cost.

## Key Findings
- Authors claim optimizer-state misalignment (first and second Adam moments computed in a changing low-rank subspace), not only gradient misalignment, is a major reason LoRA underperforms full fine-tuning.
- On commonsense reasoning (LLaMA-7B, LLaMA2-7B, LLaMA3-8B; eight benchmarks, unified training set), LoFT at rank 16 beats LoRA and DoRA on average accuracy. LoFT degrades more gracefully down to rank 1 (e.g., LLaMA-7B: 76.08 at r=16, 72.17 at r=1; LoRA r=16 73.57).
- On ViT-Base image classification (ISIC2019, HAM10000, Diabetic Retinopathy, DomainNet), LoFT at rank 16 averages 76.12%, slightly above full fine-tuning (75.86%) and above LoRA (75.46%) and DoRA (74.74%). Authors attribute the edge over full FT to implicit regularisation, a claim not established by the paper.
- In a simulated quadratic problem (fitting a low-rank target with rank-8 factors), LoFT tracks full fine-tuning closely, while removing either alternating updates or moment calibration slows convergence and worsens the final loss.
- Under the exact-recovery claim, LoFT-AdamW reduces to full AdamW when r = max(m, n) and the factors are full rank. The authors describe this as the first low-rank method with that property.
- Memory cost: the second-moment calibration adds O((m+n) r^2) memory, up to about 25.65% more than LoRA. Dropping that calibration ("LoFT simple") keeps the overhead under 6% and costs about 0.1% accuracy.

## Methodology
The method is built from six components, applied to one factor at a time (alternating updates, so only U or V is trained in a given step):
1. Alternating updates, to remove the second-order cross term from LoRA's one-step update.
2. Gradient scaling by (V^T V)^-1, so the low-rank update is the closest rank-r approximation to the full gradient in the subspace and is scale-invariant.
3. Momentum recalibration with a calibration matrix C_k that maps the momentum built in earlier subspaces into the current one.
4. Second-moment alignment, using cross-term accumulators computed via Khatri-Rao and Kronecker identities (Slyusar 1999).
5. Reconstruction of the projected full-fine-tuning Adam update, then projection onto the current low-rank subspace.
6. Gradient clipping computed on the projected full gradient, with weight decay left unchanged.

Experiments use LoRA and DoRA as the main baselines, with full FT, rsLoRA, AdaLoRA, LoRA-Pro, LoRA-GA and LoRA+ in an appendix. Evaluation covers LLM commonsense reasoning and ViT image classification at ranks 1 to 16. The appendix (ablations, memory and latency, LLaMA-3.1-70B scaling, Muon variant) was not reviewed in full for this summary; the main-text claims above come from the body (through Section 5 and the start of the references).

## Relevance to Proposal
**Score: 5**

The paper is not about parameter decompositions or interpretability, but it is directly relevant to the "structure of fine-tuning updates" and "factorised gradient descent" strands of the broad sweep. Its core technical move, projecting the full gradient and optimizer moments onto the low-rank factors (Sections 2.1 to 2.4, with the pseudo-inverse projection P_V = V(V^T V)^-1 V^T), is the same kind of operation as pulling a post-training step back through W = sum_c U_c V_c^T, so the derivations may be useful for the commuting step. It also gives empirical evidence that low-rank updates can track full fine-tuning under the right optimizer, which bears on whether post-training updates are low-dimensional enough to be carried by existing components. Its relevance to the proposal's core question is indirect: it does not address decomposition stability, component identity, or interpretability.

## Limitations
- Stated by the authors: the second-moment calibration adds O((m+n) r^2) memory, up to about 25.65% over LoRA, and the authors call this their main limitation.
- Stated by the authors: the method is developed for one factor at a time, and the authors plan to study quantization and differential privacy as future work.
- Apparent: evaluation is on small-to-medium models (7B–8B) and on classification and multiple-choice commonsense tasks. No generation, RL or instruction-tuning settings are tested in the main text.
- Apparent: the vision result that LoFT beats full FT is within about one point (76.12 vs 75.86 average, three runs with standard deviations that overlap for several datasets); the implicit-regularisation explanation is untested.
- Apparent: the exact-recovery result holds only in the full-rank limit (r = max(m, n)), which is not a practical low-rank regime; the low-rank claims rest on experiments rather than a theorem.
- Apparent: the simulated experiment uses a single quadratic target, so the claim that LoFT "matches full FT" is only shown for that setting.
- Not relevant to the proposal's interpretability question: nothing in the paper examines what the learned updates mean or whether they reuse existing features.

## Key Quotes
> "LoFT not only learns weight updates in a low-rank subspace (like LoRA) but also properly projects the optimizer's first and second moments (Adam's momentum and variance) into the same subspace, mirroring full-model updates." (Abstract)

> "We identify that not only gradients but also optimizer states (i.e., first and second moments) suffer from misalignment when approximating full fine-tuning with low-rank updates." (Section 1, contributions)

> "LoFT can be interpreted as the tightest approximation to full finetuning under the constraint that each update lies in the subspace defined by V (when updating U)." (Figure 1 caption)

> "The associated memory overhead is O((m + n) r^2), which is the **main limitation** of our approach." (Section 2.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (conference paper).
