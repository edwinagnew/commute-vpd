# Understanding On-Policy Distillation: A Mechanistic Interpretability Perspective via Sparse Crosscoders

## Metadata
- **Authors**: Zichao Yu, Qianshuo Ye, Xu Wang, Difan Zou (HKU, Cambridge, Shenzhen Loop Area Institute)
- **Year**: 2026
- **Source**: arXiv preprint (v1), 2609.35210
- **DOI/URL**: https://arxiv.org/abs/2609.35210 (project page: https://yzc-666.github.io/understanding-opd-crosscoders/)

## Summary
The paper asks what on-policy distillation (OPD) actually changes inside a student model, using sparse crosscoders trained jointly on the student before and after OPD and on the teacher. It proposes a "swap readout" that reads each student checkpoint's feature activations on its own, which lets feature usage be compared across checkpoints, including ones the crosscoder never saw. The authors conclude that OPD (and an SFT warm-up on the teacher's own rollouts that often precedes it) reweights features the student already shares with the teacher rather than creating new features or transferring the teacher's own features.

## Key Findings
- Across three OPD settings (JustRL, Skywork, R1-7B teachers, all for an R1-Distill-Qwen-1.5B student), OPD creates no student-specific features (MAS distribution unchanged) and passes on none of the teacher's dominant features (student MAS on them stays 0.19 before and after OPD). Abstract and Section 3.1.
- Feature usage changes only slightly: 98.2%, 99.6%, and 99.4% of frequently used features change firing rate by less than 20%, and no frequently used feature is gained or lost. Section 3.1.
- The features OPD changes most are decision-token features (e.g., "Wait", "So/Therefore", "maybe/perhaps"). They make up 12% of the 50 most-changed features under JustRL (base rate 0.4%) and sit at steps where teacher-student KL is 1.5-3.6x its average. Section 3.2.
- An SFT warm-up on the teacher's rollouts (Qwen3-1.7B-Base from Qwen3-4B-Base-GRPO) also adds no features, creating none of its own and not acquiring the teacher's. It reweights shared features, partly along OPD's direction (slope beta = 0.46) and partly outside it (56% of its change lies in a residual orthogonal to OPD's direction). Section 4.1-4.2.
- Feature-level intervention: adding the warm-up's feature change to a directly distilled student raises avg@8 from 18.6% to 21.3% (warm-up+OPD reaches 21.9%); subtracting it from the warmed-up student lowers it to 19.2%. A shuffled-feature control does not transfer the effect. Table 3.

## Methodology
- **Readout.** BatchTopK sparse crosscoders (dictionary 32,768, k=50) trained on residual-stream activations at a middle layer (block 14 of 28) of three models: student before training (B), student after training (O), and teacher (T). The swap readout places a student checkpoint's activation in both student slots and keeps the teacher slot fixed, so the encoder's undetermined student-difference term cancels (Proposition 1, Appendix B). Feature statistics are computed on about 3M held-out tokens.
- **Attribution.** Model attribution score (MAS) from decoder norms, following Shi et al. (2026), used to identify which model "owns" each feature.
- **Behavioural evaluation.** Accuracy on AIME24, AIME25, AMC23 with avg@256 (Section 3) or avg@8 (Section 4), with bootstrap intervals.
- **Training.** Vanilla OPD (reverse-KL on the student's top-16 tokens, one pass over DAPO-Math-17k, 279 steps, lr 1e-6) implemented in verl.
- **Intervention.** Adds or subtracts the decoded swap-code difference of the warm-up at block 14 during generation.

## Relevance to Proposal
**Score: 6**

The paper's central dichotomy (does post-training acquire new features or reweight existing ones?) is the same question the proposal frames as "create new mechanisms vs strengthen existing ones", and its swap readout is a checkpoint-tracking feature-level method, relevant to strand 2 (training dynamics of features) and strand 4 (mechanistic accounts of fine-tuning). It does not use a parameter decomposition (VPD/SPD/APD), does not map a weight update onto components, and evaluates only OPD and SFT warm-up, so it does not test whether a commuting map Phi(D, dW) exists.

## Limitations
- The "no new features" finding is conditional on the crosscoder's dictionary: a feature the crosscoder does not learn is invisible to the analysis, so "acquires no features" means "no features in this 32k dictionary".
- Only one student family (R1-Distill-Qwen-1.5B) and one teacher-family setup for Section 3; the warm-up study uses a single model (Qwen3-1.7B-Base). Single middle layer only.
- The reweighting measure is firing-rate change with a 20% threshold chosen by the authors; the "98%" headline depends on that threshold and on counting only frequently used features (>=100 firings).
- The paper's own Table 6 shows OPD recovers only 26% (Skywork) and 9% (R1-7B) of the teacher's advantage, so the "reweighting" regime is weakly effective in two of three settings. Section 3 analyses are thus of modest-gain runs.
- Intervention effects are modest (about +2.7 and -2.7 points avg@8, from small problem sets of 100 problems with 8 samples), though the authors report bootstrap intervals that exclude zero.
- The swap readout's correctness rests on a Proposition (Appendix B) and on the authors' verification on their own crosscoders (Table 4); the authors acknowledge the own-slot and replaced-slot readouts are unreliable, which is a crosscoder-specific finding.
- Decision-token features are identified by manual reading of top-activating contexts (Appendix D.2, E) and a top-12 heuristic.
- No comparison to a non-OPD RL or SFT baseline in the same readout framework beyond the cited Shi et al. (2026) NRN comparison.
- Preprint, not peer-reviewed. The authors disclose AI-assisted code and text (AI use statement).

## Key Quotes
> "OPD neither creates features nor passes on the teacher's own, and leaves the firing rates of over 98% of the student's frequently used features within 20%." (Abstract)

> "The teacher's own features stay with the teacher." (Section 3.1)

> "Training on the teacher's own rollouts does not give the student the teacher's features: like OPD, the warm-up adds no features and only reweights those the student already shares with the teacher." (Section 4.1, Takeaway)

> "Our conclusions concern this warm-up rather than SFT in general." (Section 4.1)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comment thread).
