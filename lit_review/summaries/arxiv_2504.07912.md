# Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining

## Metadata
- **Authors**: Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, Eran Malach
- **Year**: 2025 (arXiv v2, 7 Aug 2025)
- **Source**: COLM 2025
- **DOI/URL**: arXiv:2504.07912

## Summary
The authors pretrain 150M and 1B OLMo-style models from scratch on controlled mixtures of math corpora and synthetic instruction sets (TinyGSM, OpenMathInstruct1, OpenMathInstruct2), each with a distinctive output format. They then run PPO (also GRPO and Expert Iteration) on GSM8K. RL quickly collapses the model onto the output format of one pretraining distribution, which raises pass@1 and lowers diversity. Which format wins depends on model scale and on the mixture. All evidence is behavioural (format tracking and accuracy). The paper has no mechanistic analysis.

## Key Findings
- (Authors) RL fine-tuning "rapidly converges to producing outputs that follow the format of a single data distribution seen during pretraining", suppressing the others (Sec 3.1). The largest pass@1 gain coincides with this shift, usually within the first epoch.
- (Authors) pass@64 slightly declines late in training while majority@64 rises by about 5%. This is the standard sharpening signature (Sec 3.1, Fig 2). A higher KL coefficient (0.01 vs 0.001) keeps some natural-language outputs and keeps pass@64 stable (Fig 3).
- (Authors) The winning format is not always the most common at initialisation, and not always the most accurate. In one 150M mixture the model moved from 62% OMI1-style to TinyGSM-style. In another (8x OMI1) it converged to OMI1 style and ended with lower accuracy, which the authors call a failure mode (Sec 3.2, Fig 4).
- (Authors) The preferred format is scale dependent. 150M models favour TinyGSM code, and 1B models favour natural language even though it was the rarest format at initialisation and TinyGSM was more accurate (Sec 3.4, Fig 7). The authors attribute this to larger capacity, which is their speculation ("likely tied to...").
- (Authors) Within a format, outputs are also refined (docstring conventions, quote style) in a model pretrained on TinyGSM alone, and accuracy still rises (Sec 3.3, Fig 6).
- (Authors) GRPO shows the same single-format convergence but is "less stable and often experiences a brief collapse in performance". Expert Iteration, which does SFT on filtered correct samples, underperforms PPO and shifts only mildly (Sec 3.5; details in App F, not read).
- (Authors) Training on GSM8K gives out-of-distribution gains on MATH-500 for 1B models, for example TinyGSM+OMI1+OMI2 pass@1 33.4 to 43.8 (Table 1, Sec 4). The gains are largest for mixtures that contain OMI data, which the authors say resemble MATH problems. AIME shows little pass@1 change but some pass@64 gain (App H.2, summarised in Sec 4).
- (Authors) The theory is a known result. The KL-regularised optimum is a reweighting of a mixture reference policy, pi* proportional to pi_ref * exp(r/beta), so RL "mostly converges to the strategy which maximizes the reward" (Sec 3.6). The authors say that results hold even without the KL term, which this derivation does not cover (footnote 2).

## Methodology
Models are pretrained from scratch (150M: width 768, depth 12; 1B: width 2048, depth 16) on FineMath-3+, Algebraic-Stack and repeated passes of the instruction sets. RL is run with OpenRLHF (PPO, GRPO) with a binary verifiable reward on GSM8K train. Expert Iteration uses k=64 samples per problem. Generation format is tracked by string and tag matching on GSM8K test generations, alongside pass@1, pass@64 and majority@64. Transfer is tested on MATH-500 and AIME.

## Relevance to Proposal
**Score: 4**

1. Evidence is behavioural only. It consists of format fractions, accuracy and pass@k curves. No activations, weights, probes or circuits are examined. I did not find any mechanistic claim in the pages I read. Appendix G reports token-probability metrics, which are still output-level.
2. The result could hold with our mechanistic verdict going either way. "Amplifies a mode already in pretraining" is a behavioural statement about the sampling distribution, and the paper does not say what the weight update looks like (inference). If RL only reweights a mixture of existing generators, a small delta-W could be carried by existing components, and our verdict would be "reuses C". The same behavioural picture is also compatible with a delta-W that builds a new readout or router that does the selecting, giving "not through C" or "unreadable". A readout-based test like Jain Def 1 and a mode-selection story are not the same thing, so this paper neither supports nor undermines either side. Also, the format-selection effect is about the whole output style. A format switch could be an easy low-rank change that existing components absorb, which would be a confound for our GSM8K arms (inference). The authors' own gains on MATH-500 and refined within-format quality (Sec 3.3, Sec 4) show that RL changes more than the mixture weights, which argues against a pure reweighting reading even behaviourally (inference).
3. Directly usable:
 - Controlled setting where what the base model has seen is known by construction (pretraining mixtures with tagged formats), and released checkpoints for two 1B models (footnote 1). This is a candidate anchor-like testbed, but at 150M-1B scale from scratch, not Qwen3-1.7B.
 - A format-fraction metric (share of generations in each pretraining format, plus within-format accuracy) could sit beside our readings as a behavioural comparator for the sharpening hypothesis.
 - Findings about early steps: the biggest pass@1 gain happens as the format shifts within the first epoch. The authors also see GRPO instability and a transient collapse, which matters for reading delta-W saved at early steps.
 - Nothing specific to Qwen, beyond a remark that Qwen's pretraining data is undisclosed, which they treat as a confounder for RL studies (Sec 1).
The connection to our mechanistic question is thin. It is a good reference for the behavioural sharpening literature we need to distinguish ourselves from.

## Limitations
- Behavioural evidence only. No mechanistic test of reuse versus new mechanism.
- Small models trained from scratch (150M, 1B) with synthetic, templated data. Transfer to Qwen3-1.7B with web-scale pretraining is unclear.
- Single task family (GSM8K, with MATH and AIME for transfer). Most analysis uses PPO at 150M. GRPO and EI are relegated to the appendix.
- The theory is a standard KL-regularised tilt and covers neither the no-KL case nor how the update is implemented in weights.
- The explanation of scale dependence is speculation. One 8x OMI1 failure case is not explained mechanistically.
- Text was PDF-extracted. Figures are garbled to axis labels only, so I cannot read curve values. Only Table 1 numbers are cited, and they are as extracted.

## Key Quotes
> "the model rapidly converges to producing outputs that follow the format of a single data distribution seen during pretraining, supressing the other ones." (Sec 3.1, p4)

> "while majority@64 accuracy improves by approximately 5% due to fine-tuning, pass@64 accuracy declines towards the end of training" (Sec 3.1, p4)

> "Then, the maximizer would correspond to: pi*(y|x) proportional to pi_ref(y|x) exp(r(y,x)/beta) = sum_i alpha_i exp(r(x,y)/beta) pi_i(y|x)" (Sec 3.6, p9; formula garbled by extraction, restated in plain text)

> "RL fine-tuning amplifies a specific mode from the pretraining mixture while collapsing the others." (Sec 5, p10)

## Comments Summary (if LessWrong/AF post)
Not applicable (conference paper).

## Not read
Appendices after Related Work (page 16 onward, roughly): App B (dataset examples and metrics), C (hyperparameters), D (other mixtures), E (1B results), F (GRPO and Expert Iteration details), G (confidence metrics), H (error analysis, AIME), I (PPO on MATH). I rely on the main-text descriptions of those. Figure contents were not readable in the extraction.
