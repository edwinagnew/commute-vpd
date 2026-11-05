# Self-Improvement in Language Models: The Sharpening Mechanism

## Metadata
- **Authors**: Audrey Huang, Adam Block, Dylan J. Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, Akshay Krishnamurthy
- **Year**: 2024 (arXiv v2, 4 Dec 2024); ICLR 2025
- **Source**: arXiv:2412.01951 [cs.AI]
- **DOI/URL**: arxiv.org/abs/2412.01951

## Summary
A learning-theory paper that formalises "self-improvement" as sharpening. Sharpening means using the model's own self-reward (for example sequence log-likelihood) to tilt the base policy toward more certain responses. It proves sample-complexity bounds for SFT-based and RLHF-based sharpening, and checks the idea with best-of-N (BoN) and small LoRA SFT experiments on Phi, Llama and Mistral models. Everything is behavioural or distributional. It never looks inside the model.

## How the paper defines sharpening (exact)
There are three nested definitions. They differ from the project's "elicitation".

1. Informal, general (Sec 1.1, p.2): "We refer to sharpening as any process that tilts πbase toward responses that are more certain in the sense that they enjoy greater self-reward rself. That is, a sharpened model π̂ is one that (approximately) maximizes the self-reward: π̂(x) ≈ arg max_{y∈Y} rself(y | x; πbase). (1)"
2. The concrete self-reward used for all theory (Eq. 2, p.3): rself(y | x; πbase) := log πbase(y | x). That is "maximum-likelihood sharpening", the sequence-level arg-max of the base model's own likelihood.
3. Formal (Definition 3.1, p.6): "We say that a model π̂ is (ϵ, δ)-sharpened relative to πbase if P_{x∼µ}[π̂(y⋆(x) | x) ≥ 1 − δ] ≥ 1 − ϵ", where y⋆(x) := arg max_y log πbase(y | x). In words, the model puts at least 1−δ mass on the base model's own arg-max response, on all but an ϵ-fraction of prompts.

Related terms:
- The "sharpening mechanism" is a separate claim (p.3). It says that "responses from a model with the highest certainty (in the sense of large self-reward rself) exhibit the greatest performance on a task of interest".
- "Coverage" is Ccov = E_x[1/πbase(y⋆(x)|x)] (Eq. 6, p.8). It measures how much base mass sits on the arg-max response. It is the paper's notion of "the behaviour is already in the base distribution".
- A relaxed version, approximate sharpening (Definition F.1), is only referenced in Remark 3.1. It sits in the appendix, which I did not read.

Important: the target is the base model's own arg-max under its own likelihood, not the correct answer. Correctness enters only empirically, through the assumption that high likelihood correlates with correctness. The only definition of sharpening in the paper is distributional and behavioural. There is no mechanistic or representational definition. The Conclusion lists "Representation learning" as future work (p.17).

## Key Findings
Authors' claims unless marked as inference.
- Theory (Sec 3.3, 4.1): a sample-complexity lower bound scales with the coverage coefficient Ccov (Thm 3.1). SFT-Sharpening (fine-tune on BoN-by-likelihood samples) matches it up to polynomial factors in δ and logs (Thm 4.1), assuming realizability and coverage.
- RLHF-Sharpening (KL-regularised, DPO-style, Eq. 3/8) gets comparable guarantees under a margin and concentrability assumption (Thm 4.2). With online exploration (XPO) it avoids dependence on coverage and depends on a class-structure parameter, the sequential extrapolation coefficient (Thm 4.3/4.4, linear softmax example). This is the paper's one RL-versus-SFT contrast, and it is theoretical.
- Experiments (Sec 5.2): BoN with N=50 under log-likelihood self-reward beats greedy for at least one model on every dataset. Gains are very uneven across model-task pairs and sometimes negative. The Fig. 1(a) matrix is garbled in extraction, so I do not quote individual cells. Correct completions have stochastically higher sequence log-prob than incorrect ones (Fig. 6, p.14). Performance saturates around N=10-20.
- Failure modes (p.14): BoN by raw likelihood can select short responses that lose chain-of-thought benefit (gpt-3.5-turbo-instruct on GSM8K). It also fails when the base model is poor (Llama3.2-3B).
- Coverage is high (Fig. 3c, p.16, extraction readable): pass@50 is mostly above 90%, often above 97%, across model-task pairs, against much lower greedy accuracy. The authors read this as "other self-reward functions" could do better.
- Amortisation (Sec 5.4, Table 1, p.17): SFT-Sharpening with LoRA lifts accuracy over greedy and sequence log-likelihood. Phi3.5-Mini on MATH: 19.24 ± 2.41 % accuracy lift, 48.33 ± 0.17 likelihood lift. On GSM8K the accuracy lift is only 1.82 ± 0.64 %. The GSM8K training curve is noisy, and log-prob is a poor proxy for accuracy there. Mistral-7B on MATH spent ~90% of steps recovering from an initial drop, which the authors attribute to under-tuned hyperparameters.

## Methodology
- A theoretical framework in a "sample-and-evaluate" oracle model (sample from πbase, read its likelihood). It gives minimax bounds and uses a DPO/XPO analysis. I read the main text only, Sections 1-6 (pp.1-18). I did not read the proofs or the appendices (A-J). This is a theory paper and the main text is enough.
- Experiments: 7 models (Phi3 family, Phi3.5-Mini, Llama3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, gpt-3.5-turbo-instruct, plus a Llama-2 GameOf24 model). Tasks are GSM8K, MATH, ProntoQA, 3 MMLU subsets and GameOf24. 1-shot prompting, 512 max tokens, 256 test examples, 10 seeds. SFT-Sharpening on four model-task pairs with LoRA, 3 seeds, best checkpoint chosen. There is no Qwen model.

## Relevance to Proposal
**Score: 3**

1. Evidence type: purely behavioural and distributional (accuracy, sequence log-likelihood, pass@50 coverage) plus theory. No activations, weights, circuits or probing.
2. Could their result hold while our mechanistic verdict goes the other way? Yes, in both directions (my inference).
   - Sharpening is defined as concentrating mass on a response that the base model's own distribution already puts on its arg-max, with coverage Ccov controlling feasibility. That is the behavioural definition the project rejected. An update can satisfy it (high coverage, a successful (ϵ,δ)-sharpened model) while the post-trained map is not g∘f_C for any named readable C, giving "not through C" or "unreadable".
   - Conversely, a fine-tune could reuse a W₀ linear readout in the Jain Def 1/2 sense and yet raise the base arg-max mass only a little, or none.
   - The paper makes no claim about how the update is implemented, so it supports neither side. Its stated aim is "a computational phenomenon rather than a statistical one" (Sec 1, p.1), i.e. extracting hidden knowledge. That is a thin conceptual neighbour of "reuse", but "hidden knowledge" is not operationalised mechanistically.
3. Directly usable, but modest:
   - Coverage Ccov and pass@k-style coverage (Fig. 3c) as a behavioural comparator to log beside our mechanistic readings. Pass@50 above 90% against much lower greedy accuracy on GSM8K-type tasks is a sizeable sharpening headroom reference.
   - The sequence log-likelihood lift over greedy and over the base model is a cheap distributional measure of how much an SFT or GRPO arm moves toward base-model arg-max. Their GSM8K result (small lift, noisy curve, log-prob a poor accuracy proxy) is a warning that this comparator is weak on our task.
   - A negative or caution finding: BoN-by-likelihood can select shorter responses and lose chain-of-thought.
   - The theory contrast (SFT needs coverage, RL with exploration does not) gives a prior that GRPO and SFT might differ in how new their updates are. This is a theoretical claim, not tested on GRPO. The paper's RL variant is DPO/XPO, not GRPO, and there is no early-step analysis.
   - No Qwen results. No task with novelty fixed by construction (the GameOf24 model is only a fine-tuned checkpoint).
   - The connection is thin overall. Useful mainly for pinning down what "sharpening" means in the literature, so that our elicitation is not conflated with it.

## Limitations
- Authors: log-likelihood is a "stylized" self-reward. Realizability and coverage assumptions are strong. The RLHF bound may be "pessimistic". The Mistral SFT run had a hyperparameter problem. Compute was limited (only a subset of pairs for SFT).
- Apparent: the target is the base model's arg-max, not correctness. The paper itself shows this can degrade accuracy (short responses, weak models).
- Apparent: "SFT-Sharpening" trains on the model's own BoN outputs, which differs from SFT on external data and from GRPO with a task reward. Both are what our Qwen arms use, so the transfer is indirect.
- Apparent: LoRA with a best-checkpoint choice and 3 seeds. Table 1 lifts are relative to greedy, not to temperature-1 sampling.
- Extraction note: Figure 1/2/3 matrices are garbled by extraction. I quote only Table 1 and the Fig. 3c/3d readable rows qualitatively, and do not rely on specific Fig. 1(a) cells.

## Key Quotes
> "We refer to sharpening as any process that tilts πbase toward responses that are more certain in the sense that they enjoy greater self-reward rself." (Sec 1.1, p.2)

> "We say that a model π̂ is (ϵ, δ)-sharpened relative to πbase if P_{x∼µ}[π̂(y⋆(x) | x) ≥ 1 − δ] ≥ 1 − ϵ." (Definition 3.1, p.6)

> "Self-improvement, rather than creating knowledge from nothing, is a means of extracting and distilling this knowledge into a more accessible form, and thus is a computational phenomenon rather than a statistical one." (Sec 1, p.1)

> "an important direction for future work is to study the dynamics of self-improvement for specific models/architectures and understand the representations that these models learn under self-training." (Sec 6, p.17)

## Comments Summary
Not applicable (arXiv paper, no comments). Not read: appendices and proofs (A-J), Figs. 4-9 details, reference list.
