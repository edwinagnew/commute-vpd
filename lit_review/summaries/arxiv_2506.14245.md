# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

Title check: the extracted text (page 1) reads "REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS IMPLICITLY INCENTIVIZES CORRECT REASONING IN BASE LLMS" (PDF extraction removed spaces). It matches the title you gave.

## Metadata
- **Authors**: Xumeng Wen, Zihan Liu, Shun Zheng (equal contribution), Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, Jiang Bian, Mao Yang (Microsoft Research Asia and interns)
- **Year**: 2025 (arXiv v2, 2 Oct 2025); cited as ICLR 2026 (venue not verifiable from the text, which is stamped "Preprint.")
- **Source**: arXiv:2506.14245 (cs.AI)
- **DOI/URL**: https://arxiv.org/abs/2506.14245

## Summary
The paper argues against the "RLVR only sharpens the base model's sampling distribution" reading of pass@k curves (Yue et al. 2025). It says pass@k on math is inflated because the base model reaches right answers through wrong chains of thought (CoTs). It introduces CoT-Pass@K, which counts a sample only if an LLM judge also accepts the CoT. Under that metric the post-RLVR model (DAPO-Qwen-32B) stays above Qwen2.5-32B up to K=1024 on AIME 2024/2025. It adds a GRPO argument (Theorem 1) for why answer-only rewards raise the probability of correct CoTs, and a training-dynamics analysis. All evidence is behavioural, at the level of text outputs.

## Key Findings
All items are the authors' claims unless marked (inference).
- Plain Pass@K on math reproduces Yue et al.: the base catches up with and passes the RLVR model as K grows. CoT-Pass@K shows a persistent gap on AIME 2024 and especially AIME 2025 (p.5, Fig. 2). On MATH-500 and AMC23 the gap closes; on Minerva there is no gain, which the authors attribute to train-test domain mismatch (p.5).
- Code: AceReason-Nemotron-7B beats its pre-RLVR parent DeepSeek-R1-Distill-Qwen-7B on Pass@K across LiveCodeBench versions (p.5, Fig. 3; Skywork-OR1 in App. A.4). Code execution makes guessing unlikely, so plain Pass@K is trusted there.
- In math, RLVR on a distilled model (Skywork-OR1-Math-7B) shows no clear large-K gap even under CoT-Pass@K (App. A.4, Fig. 9). The authors suspect the distilled model already holds the learnable reasoning.
- Theorem 1 (p.6, App. A.5): assume a "Logic Prior" P(correct answer | correct CoT)=alpha > P(correct answer | wrong CoT)=beta, a learnable group, and large G. Then the expected GRPO advantage is positive for correct CoTs and negative for incorrect ones, so p_c rises monotonically. The authors say the proof covers optimisation only, with no generalisation guarantee (p.9).
- Training dynamics (DAPO reproduction, 32 MI300X GPUs, over two weeks): P(CC|CA), the fraction of correct CoTs among correct answers, rises from the early steps. CoT-Pass@K gains on AIME 2024 appear within the first 20 steps, and one step is 16 gradient updates (App. A.6, Fig. 10c). P(CC|CA) median stays about 0.7 after 400 steps, so wrong CoTs persist while P(CA) is near 1.
- SFT probe (Sec. 6, Fig. 6): SFT of Qwen2.5-32B on DAPO-model CoTs nearly replicates the RLVR model's Pass@1 and CoT-Pass@K. SFT on base-model CoTs filtered to correct answers starts to reduce guessing, which the authors call "a round of off-policy RLVR". They conclude RLVR CoTs "cannot be directly sampled from base LLMs".

## Methodology
Pass@K (Chen et al. 2021 estimator) and CoT-Pass@K, where D counts samples with both a correct answer and a correct CoT. The CoT is judged by DeepSeek-R1-0528-Qwen3-8B run three times per CoT and aggregated as any-correct, all-correct and majority-correct (App. A.3). Manual inspection of cases where Pass@K is small but positive and CoT-Pass@K is zero (App. A.8). Existing checkpoints (DAPO-Qwen-32B, AceReason, Skywork-OR1) plus one DAPO reproduction on Qwen2.5-32B. A theoretical argument built on a two-parameter (alpha, beta) model. SFT as a quality probe of CoT data.

## Relevance to Proposal
**Score: 3**

The connection is thin. The paper disputes the behavioural "sharpening" story, but it says nothing about parameters or representations, so it cannot be set against a W0 linear-readout (Jain Def 1) reuse test.

1. **Behavioural or internal?** Entirely behavioural. All evidence comes from sampled text scored by answer match, code execution, or an LLM judge on the CoT. There are no activations, weights, circuits, probes or ablations. The "Logic Prior" is an assumed statistical property of the base model's output distribution, not a measured internal one.
2. **Could its result hold while our verdict goes the other way?** Yes, in both directions (my inference).
   - Its "new, correct reasoning" is defined by the judge accepting the CoT, so CoT-Pass@K measures output validity. It does not measure whether a mechanism is new. A post-trained model could produce valid CoTs that W0 almost never samples, yet the change could still run through a direction W0 already reads out (reuse C). That fits "reuses C" while their metric says "extended boundary".
   - The reverse is also possible. Their large-K parity cases (Skywork-OR1-Math, MATH-500) fit sharpening behaviourally, yet the update could be built on new directions (not through C). Equal pass@k does not show shared mechanism. Their own point that Pass@K can hide wrong-CoT guessing is a caution against reading base-vs-tuned behavioural parity as reuse.
   - Their framing is that RLVR makes the base model's weakly sampled correct CoTs more probable. That is closer to behavioural sharpening applied to correct reasoning than to our mechanistic reuse, and the two are not the same claim.
3. **Directly usable?** Modestly.
   - Comparator metric: CoT-Pass@K and P(CC|CA) are cheap behavioural readings that could sit beside our readings, but they need an LLM judge. GSM8K answers are easy to guess, so the judge's role there is unclear. This is my inference.
   - Early training: correct-reasoning gains appear in the first 20 steps (320 gradient updates), which supports saving early-step delta W (the authors' observation, on a 32B model).
   - SFT vs RL: SFT on the RL model's CoTs reproduces its Pass@1. That suggests SFT and RLVR arms can end up behaviourally similar, which is a useful warning when comparing arms. The paper has no weight-space comparison.
   - Failure mode: where the Logic Prior fails, wrong CoTs get reinforced. That is a possible source of mismatch between behaviour and mechanism.
   - Qwen: Qwen2.5-32B and distilled Qwen-7B only. Nothing on Qwen3-1.7B or GSM8K.
   - No controlled task with known novelty. Contamination is argued only via AIME 2025's post-cutoff date, and the authors admit they cannot separate "easy for the base" from "memorised" on MATH-500/AMC23 (p.5).

## Limitations
- Authors: an LLM judge verifies CoTs, with false positives and negatives possible (p.9). The theorem covers optimisation, not generalisation (p.9). Pass@K and CoT-Pass@K are hard to compare on benchmarks that may be in pretraining data (p.5). The DAPO reproduction reached about 44% Pass@1 against more than 50% reported (App. A.6).
- Mine: the judge is an 8B model, so it can miss subtle errors and misses are not randomly distributed. The gap is shown mainly on AIME, with a small number of problems. The "base cannot sample these CoTs" claim rests on Pass@K up to 1024 samples and so is not a proof of absence. The key assumption alpha > beta is not measured. The paper offers no mechanistic test.

## Key Quotes
> "RLVR can extend the reasoning boundary for both mathematical and coding tasks." (Abstract)

> "once LLMs have been pre-trained to establish strong knowledge and logic priors that distinguish correct from incorrect CoTs, the GRPO gradient will increase the probability of generating more correct CoTs." (Sec. 1, p.3)

> "These observations indicate that the incentivized CoT data through RLVR is crucial, as such CoTs cannot be directly sampled from base LLMs." (Sec. 6, p.9)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper, no comments).

## Reading notes
- Read: the main text (pp.1-9), the references, App. A.1-A.7 and the start of A.8.1 (AIME 2024 case studies, to about p.24).
- Not read: the remainder of the A.8 case studies (AIME 2025 and other benchmarks, and the second half of the A.8 text).
- Garbled: figure axes, legends and plotted values (Figs. 2-6, 8-10) are extracted as tick labels only. I give no numbers read from figures, and the gap claims are the authors' text statements. The text is also missing spaces in places, and the formulas for Pass@K(q), CoT-Pass@K(q) and the advantage are extracted imperfectly. I checked them against the surrounding prose.
