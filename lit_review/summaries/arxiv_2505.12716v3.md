# Shadow-FT: Tuning Instruct Models via Training on Paired Base Model

## Metadata
- **Authors**: Taiqiang Wu, Runming Yang, Jiayi Li, Pengfei Hu, Yik-Chung Wu, Ngai Wong, Yujiu Yang
- **Year**: 2025 (arXiv v3; ICLR-style preprint)
- **Source**: arXiv preprint (arXiv:2505.12716v3). Code: https://github.com/wutaiqiang/Shadow-FT
- **DOI/URL**: https://arxiv.org/abs/2505.12716

## Summary
The paper observes that instruction-tuned (INSTRUCT) models often gain little, or even lose performance, when fine-tuned directly on a small dataset, while their paired pretrained BASE models have almost identical weights (relative gap σ about 0.016 for Llama-3.1-8B, below 0.03 across the models tested). The proposed Shadow-FT fine-tunes the BASE model, then adds the resulting weight delta directly to the INSTRUCT weights, with no extra parameters or training cost. It reports gains over direct full and LoRA tuning of INSTRUCT on 19 benchmarks (math, code, reasoning) for Qwen-3, Llama-3, and other families, plus extensions to MLLMs and DPO.

## Key Findings
- Paired BASE and INSTRUCT weights are very close (σ < 0.03 for the models examined), so a delta learned on one is plausibly applicable to the other (Sec. 2.3, Fig. 2).
- Direct tuning of INSTRUCT with a 2k-sample set (BAAI-2k) gives marginal or negative average changes; e.g. Qwen-3-4B average 65.7 after FT vs 68.0 for the INSTRUCT model (Table 1).
- Grafting a BASE-trained delta onto INSTRUCT improves the average over both vanilla INSTRUCT and conventional tuning in most rows of Table 1 (e.g. Qwen-3-14B: 75.9 Shadow-FT vs 74.5 INSTRUCT, 73.4 FT). Gains are small on several models (Qwen-2.5-32B full FT: 75.0 vs 74.5 INSTRUCT; Llama-3.2-3B full: 48.8 vs 47.4).
- In a single-model dynamics study (Qwen3-8B, LoRA r=128), the INSTRUCT run starts with a 3.25x larger gradient norm, which decays faster, and ends with a 2.4% higher loss than the BASE-tuned run (Sec. 5.3, Fig. 5). The authors read this as INSTRUCT's instruction-following prior resisting new learning.
- The method also holds for Shadow-DPO (Table 3, 55.39 vs 54.62 for DPO at rank 128), LoRA rank sweeps (Fig. 3), medical and math domain data (Table 2, Fig. 4), and ChartQA on MLLMs (Table 4, e.g. 63.80 vs 60.28 on Gemma-3-27B).
- A scaling factor slightly above 1 on the grafted delta tends to help (Sec. 5.6, Appendix D.6).

## Methodology
Algorithmic: (1) fine-tune the BASE model with full or LoRA training on BAAI-2k (2000 samples from Infinity-Instruct, high-reward and category-balanced); (2) compute ΔW = W_B+ − W_B; (3) add ΔW to the INSTRUCT weights, W_I+ = W_I + ΔW. The relative gap σ is defined as the element-wise sum of |W_B − W_I| normalised by |W_B| + |W_I|. Evaluation uses OpenCompass and lmdeploy, zero-shot mainly, averaged over Math-7, Code-3 and Knowledge-9 benchmark groups (mean of three runs for Math-7 and Code-3). Training uses LLaMA-Factory on 8 A100 GPUs. The mechanistic argument rests on loss and gradient-norm curves from one model and one LoRA setting. Related work is positioned against task vectors and Chat Vector (Ilharco et al.; Huang et al.), RE-Adapt (low-rank base/instruct delta), and proxy-tuning.

## Relevance to Proposal
**Score: 4**

The paper is about transferring a weight update learned on one checkpoint onto a paired checkpoint with nearly identical weights. That is the same kind of object the proposal cares about (a post-training δW carried from one model state to another), and it gives empirical evidence (strand 3, structure of fine-tuning updates; strand 4, what post-training does to a model) that base-to-instruct deltas can be reused. It does not decompose anything, however, and it does not measure whether the grafted update acts on existing components or creates new ones, so it bears on the question only indirectly.

## Limitations
- No interpretability or decomposition analysis. Nothing checks whether the grafted delta is carried by existing components or whether it is faithful to the INSTRUCT model's mechanisms.
- Gains are often small (about 0.5–1.5 average points in several rows), and most rows are single runs. Table 1 reports three-run means only for Math-7 and Code-3. No significance tests are reported.
- Training data is small (2k samples), and the comparisons rest on one dataset (BAAI-2k) plus a few domain sets.
- The mechanistic account (Sec. 5.3) comes from one model (Qwen3-8B) and one LoRA configuration. The authors describe BASE as a "good learner but poor backbone" without testing that claim directly. The paper also leaves open whether the gain comes from the BASE-learned direction or simply from a smaller effective step.
- Grafting requires identical architectures. The σ < 0.03 threshold is an empirical rule with no theoretical bound, and the paper gives no account of when a delta would fail to transfer.
- Table 1 caption says "We employ the Instruct version" for the reported numbers, which is ambiguous about which weights are being compared. Appendix sections were not read in full, so some configuration details are unverified.
- Authors state some results (Shadow-FT LoRA beats full FT on Math-7 and Code-3) without an explanation.

## Key Quotes
> "The key insight is to fine-tune the BASE model, and then _directly_ graft the learned weight updates to the INSTRUCT model." (Abstract)

> "In summary, these paired BASE and INSTRUCT models are highly similar with _σ_ < 0.03." (Section 2.3)

> "BASE model is a good learner but a poor backbone due to the lack of post-training." (Section 5.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
