# Watch the Weights: Unsupervised Monitoring and Control of Fine-Tuned LLMs

## Metadata
- **Authors**: Ziqian Zhong, Aditi Raghunathan (Carnegie Mellon University)
- **Year**: 2026
- **Source**: ICLR 2026 (conference paper; an earlier version was a NeurIPS 2025 Mechanistic Interpretability workshop submission per the acknowledgements)
- **DOI/URL**: Project page https://fjzzq2002.github.io/WeightWatch (no DOI given in the text read)

## Summary
The paper proposes WEIGHTWATCH, a data-free method for finding and monitoring behaviours introduced by fine-tuning. It takes the SVD of the weight difference ΔW = W_tuned - W_base at the attention output projection and MLP down projection of each layer, treats the top-k left singular vectors as "behavioural vectors", and flags inputs whose activation cosine similarity with these vectors leaves a calibrated normal range. The authors report strong backdoor and unlearning detection and some steering, and use the method to audit OLMo, Qwen 2.5 and Llama 3.1 for fine-tuning priorities.

## Key Findings
- The top singular directions of ΔW (computed from base and fine-tuned weights alone, no training data) correlate with behaviours acquired in fine-tuning. Monitoring cosine similarity along k = 20 directions per matrix flags 93% to 100% of backdoor-triggered completions on first sight, with FPR below 1% on a calibration mix of WildChat and UltraChat plus MMLU (Table 2).
- On PPO-poisoned models (trojan1 to trojan5), WEIGHTWATCH reaches 98.7% to 100% at <2% FPR, versus 11.5% to 97.4% for an activation-difference-norm baseline at 2% FPR and up to 32% for BEAT at 2% FPR (Table 3). Results against BEAT are sensitive to the calibration/evaluation data mix, which the authors acknowledge.
- On unlearned models, detection of unlearned-content prompts ranges from 36.21% (Zephyr-RMU) to 95.42% (Circuit Breaker) completions flagged, at calibration FPR <= 1.79% (Table 4). Steering recovers part of the Zephyr-RMU WMDP-Bio performance (76.92% of original) and raises circuit-breaker attack success on HarmBench from 9.6% to 82.08% when combined with a prompt prefix.
- Auditing with GPT-5.1 annotations of extremal transcripts suggests model-specific fine-tuning priorities: Qwen emoji use (2.3% of directions, and 50% emoji use vs 26% OLMo and 6% Llama on 50 prompts), Qwen multilingual/translation focus, Llama math and step-by-step reasoning, and Midjourney-prompt directions in all three models. The authors cross-check some of these against OLMo's public SFT/DPO data.
- The authors find that data presence does not always produce a detectable behavioural direction: political content (1.7% to 3.7% of OLMo's data) yields zero directions, and math content yields 15 directions while OLMo's GSM8K accuracy is low. They recommend behavioural validation as the main check.

## Methodology
- Apply SVD to ΔW for O_proj and W_down in every layer; keep top k = 20 left singular vectors per matrix (40 per layer, 2kL in total).
- Monitoring: keep a streaming min/max range of per-token cosine similarities for each direction on calibration data, flag inputs that fall outside it, and update the range over time. Remark 1 gives an FPR bound of 2t/n + O(1/n²) for t directions after n calibration samples.
- Steering: project the activation orthogonal to a flagged direction before the next layer.
- Evaluation: backdoors (LoRA, FFT, PPO-poisoned; BadNet, CTBA, MTBA, sleeper, VPI triggers), unlearning (WHP, Zephyr-RMU, Circuit Breaker), and open-ended auditing of extremal transcripts over 10^6 transcripts, with GPT-5.1 and Gemini 3 Pro used for annotation and interpretation.
- Baselines: activation PCA/SAE/DiffMean-type methods discussed qualitatively (they need anomalous examples); BEAT and an activation-difference-norm baseline quantitatively.

## Relevance to Proposal
**Score: 6**

This is the closest hit so far to the proposal's strand 3 (structure of fine-tuning updates: spectrum and alignment of δW with pretrained directions) and strand 2 (model diffing via weight differences). It supports the premise that the top singular subspace of δW carries interpretable post-training behaviour, which is what a commuting map Φ(D, δW) would need to transport. It does not use a parameter decomposition, does not test whether the directions persist as training continues, and is a single-checkpoint monitoring method rather than a continuous-training tool, so it bears only indirectly on the commuting question.

## Limitations
- Authors state the method "can only detect behaviours present in the calibration set"; backdoors absent from calibration cannot be found (Section 6).
- Detection rates vary a lot across unlearning methods (36% to 95%), and steering effectiveness is method-dependent (WHP is resistant).
- Remark 2 (weights without a base model) gives highly variable success across models, so the base-difference dependence is an important assumption.
- Calibration data mix strongly affects the BEAT comparison (footnote 3), and FPR and detection depend on the choice of calibration distribution.
- Interpretation of directions relies on GPT-5.1 and Gemini 3 Pro annotations and manual selection; the keyword-frequency counts (Table 5) are not independently validated beyond a sample and a few behavioural tests.
- The data-attribution check on OLMo uses GPT-4o-mini on 3000 sampled examples, a small and noisy check.
- Work is on linear layers in the residual stream only; the paper does not analyse other module types, and the top-k cutoff (k = 20) is a fixed choice rather than something derived.
- Single-direction causal claims are not separated from correlation in the main text; the claim that directions "correspond to" behaviours rests on monitoring performance and annotation.

## Key Quotes
> "We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors." (Abstract)

> "The key insight is that model weights themselves possess rich structure and encode salient behaviors that were introduced during training, which can be uncovered without access to any training data." (Section 1)

> "Note that our method can only detect behaviors present in the calibration set. For instance, we cannot detect backdoors if no examples with triggers appear during calibration." (Section 6)

> "We suggest using behavioral validation as the main indicator of the effectiveness of our method." (Section 6.4)

## Comments Summary (if LessWrong/AF post)
Not applicable (conference paper).
