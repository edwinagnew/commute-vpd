# Does RLVR Extend Reasoning Boundaries? Investigating Capability Expansion in Vision-Language Models

## Metadata
- **Authors**: Minghe Shen, Zhuo Zhi, Chonghan Liu, Shuo Xing, Zhengzhong Tu, Che Liu
- **Year**: 2026 (arXiv v4, 13 Apr 2026; cited as Shen et al. 2026b, ACL 2026)
- **Source**: arXiv:2511.00710 (cs.AI)
- **DOI/URL**: arXiv:2511.00710

## Summary
The paper builds "Ariadne", a synthetic maze-navigation setup (adapted from AlphaMaze) in which difficulty is set by path length and number of turns. It trains Qwen2.5-VL-7B-Instruct with GRPO on short mazes and argues that RLVR extends the reasoning boundary. The evidence is that the trained model succeeds on mazes where the base model has 0% pass@k even at k=1024, and that it improves zero-shot on MapBench and ReasonMap. All evidence is behavioural (success rates, pass@k, tokens); the paper looks at no weights, activations or circuits.

## Key Findings
- Authors' claim: the base VLM's success collapses to near zero (<1%) once complexity exceeds 2 steps or 1 turn, and GRPO moves the collapse point from about 2 to about 6 steps (Sec 4.3, Fig 2).
- Table 1 (text extraction looks clean): at 5 steps the base is 0.0 at every k up to 1024, while Ariadne reaches 7.6 pass@1 and 100 at pass@128. At 6 steps Ariadne reaches 41.0 pass@1024 and at 7 steps 14.0. At 8–10 steps both models are 0.
- At 1–4 steps the base model is not at zero. At 2 steps its pass@1 is 1.1 and pass@1024 is 69.0. So this "soft" regime fits sharpening, and the authors say so. The "novel behaviour" claim rests only on the hard-zero regime.
- Zero-shot transfer (Qwen2.5-VL-7B): Ariadne improves MapBench shortest-path score (lower is better) in most environments (Table 2) and ReasonMap long-question Map Score (4.51 to 5.15, Table 4). The SFT arm trained on the same data gives little or negative transfer, for example ReasonMap long-question Map Score 3.71.
- Gains hold for Qwen2.5-VL-3B, Qwen3-VL-4B and Qwen3-VL-8B (Tables 3 and 5). Response length roughly doubles on long ReasonMap questions (61 to 121 tokens).

## Methodology
GRPO (G=8, temperature 1.0, clip 0.2, lr 1e-6, one epoch) on 4,700 AlphaMaze samples. Training is restricted to path length ≤5 and turns ≤2, with an inverted-Gaussian curriculum that oversamples easy and hard ends. Evaluation is on held-out OOD lengths 6–10 and turns ≤4. The reward is dense and prefix-matching, plus format terms. An SFT arm uses the same data and initialization. Main metric is success rate (exact match to the ground-truth path), with pass@k up to 1024.

## Relevance to Proposal
**Score: 3**

1. Evidence type: purely behavioural. Pass@k coverage and zero-shot benchmark scores only. Fig 5 and the "mechanism of transfer" wording describe success-length distributions, not internals. There is no probing, weight, or circuit analysis.
2. Could its result hold while our verdict goes the other way: yes, in both directions (my inference). Its "novel" means the behaviour has zero base support under sampling. Our "reuses C" means the fine-tuned map is g∘f_C with C linearly readable from W₀ (Jain Def 1). A policy could reach 0%-to-nonzero success while still routing through readable base features (so we would say "reuses C"). Conversely, a case with high base pass@k, which they would call sharpening, could still be "not through C" or "unreadable" mechanistically. Their pass@k=0 criterion is also close to the behavioural definition the user rejected, in the opposite direction. The paper never names a C, so it cannot be mapped onto our verdict vocabulary. The authors themselves also say "novel spatial primitives" (Conclusion) without any internal test, so that phrase is an unsupported claim about mechanism.
3. Usable items:
   - It is a controlled task where base-model support is characterised by construction (pass@k=0 beyond a known difficulty). That makes it a possible known-novelty anchor design in principle, but it is a VLM maze task, not Qwen3-1.7B on GSM8K.
   - Its pass@k coverage across a difficulty ladder could sit beside our readings as a behavioural comparator.
   - SFT vs GRPO on identical data gives different transfer here. The SFT arm shows weaker transfer (a behavioural contrast only).
   - Qwen-specific: Qwen2.5-VL and Qwen3-VL only, no Qwen3-1.7B text model. Nothing on early training steps or on δW. Training curves are reward only (Fig 2A).
   Connection to the project is thin. It is an example of the behavioural "boundary expansion vs sharpening" literature, not of mechanism.

## Limitations
- Authors: maze is a proxy for spatial intelligence. Only small Qwen VLMs were tested. The boundary does not transfer exactly to real maps. The policy hits its own complexity horizon at ≥8 steps. Pre-training data is opaque, so the true base boundary is only probed empirically (Limitations section).
- Inference: no mechanistic evidence, so "novel primitives" is a conclusion from behaviour. The dense prefix reward and curriculum are strong shaping choices, and the 0% regime may reflect a fixed output format and prompt rather than missing capability. SFT is compared at one setting, with no matched-compute analysis. Evaluation is at temperature 1.0 only. "Decisive evidence" is stronger than the data supports. Zero-shot transfer gains are small in several rows (for example Urban 1.93 to 1.91, Mall 1.43 to 1.41).

## Key Quotes
> "applying RLVR extends the spatial reasoning boundary, achieving success on problems where the base policy VLM consistently attains 0% accuracy despite increasing pass@k sampling budgets" (Abstract)

> "the base VLM's coverage curve flat-lines at exactly 0% even as k→512. This 'hard zero' indicates that valid trajectories are effectively absent from the model's search space" (Sec 4.3, p6)

> "RLVR constructs novel spatial primitives that enable success in regimes characterized by the absence of valid solutions." (Sec 5, Conclusion)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).

## Not read / caveats
All text and tables were read. Figures 2–7 are not visible in the extraction, so their panels are described only through captions. Table values quoted above came through the extraction readably. I did not check the cited AlphaMaze or Yue et al. papers.
