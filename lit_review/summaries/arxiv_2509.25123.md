# From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones

## Metadata
- **Authors**: Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong Sun, Hao Peng
- **Year**: 2025 (arXiv v3, 19 Dec 2025); cited in the project as Yuan et al. 2026b, ICLR 2026
- **Source**: arXiv:2509.25123 (cs.AI), ICLR 2026
- **DOI/URL**: https://arxiv.org/abs/2509.25123

## Summary
The paper builds a synthetic string-transformation testbed. Atomic skills are first taught by rejection fine-tuning (RFT) on opaque function names. The paper then shows that GRPO on Level-2 compositions (two functions) teaches the model to compose, and this generalizes to deeper compositions and to a different task (Countdown). Matched-data RFT does not do this. The authors argue this refutes the claim that RL only "reranks" what the base model already does. All evidence is behavioural.

## Key Findings
All of these are authors' claims, on Llama-3.1-8B-Instruct.
- RL on Level 1 only (atomic) gives about 90% on Level 1 but near-zero on Level 3 and above. RL on Level 2 or Level 1+2 lifts held-out Level 3 from about 5% to about 30% and Level 4 from about 1% to about 15% (Sec 4.1, Fig 2).
- Iterative RFT on the same Level-2 data never exceeds 2.6% on Level 3, and reaches only about 15% on Level 2. RL Level 2 reaches 64% on Level 2 and 27% on Level 3 (Sec 4.2, Fig 3).
- Cross-task transfer: compositional RL on strings raises Countdown Level 3 from about 17% (Multi-Base) to 35%. It does so only if the model already had Countdown atomic skills. The String-Base model without them "fails completely" (Sec 4.3, Fig 4).
- The pass@k gap between the RL model and the RFT base shrinks with k on easy Levels 1-2, but widens on Levels 3-8. At Level 5 the gap goes from about 4% at pass@1 to about 25% at pass@1024. The authors call the mixed-difficulty aggregate a "reranking illusion" (Sec 4.4, Fig 5).
- A Gemini-2.5-Pro failure-mode classifier on Level 3 shows RL Level 2 eliminates "Ignores Composition" errors. Its failures shift to atomic errors. RFT Base, RFT Level 2 and RL Level 1 show similar failure profiles (Sec 4.5, Fig 6).

## Methodology
- 25 string functions with meaningless identifiers (func_16 and so on). Definitions are shown only when collecting the Stage-1 RFT data and hidden in training and evaluation.
- Stage 1 is RFT on atomic skills. Stage 2 is GRPO with binary outcome reward, or offline RFT as the baseline.
- Held-out evaluation splits the functions into disjoint sets. Difficulty is nesting depth, Levels 1-8.
- Metrics are accuracy, Avg@k, pass@k up to k=1000, and an LLM-judged failure taxonomy.
- The paper shows no activations, weights, circuits or probes.

## Relevance to Proposal
**Score: 4**

1. The evidence is entirely behavioural: accuracy, pass@k and an LLM-classified failure taxonomy. Nothing inside the model is examined. "New skill" is defined as solving problems the model could not previously solve through "systematic combination of existing capabilities" (Sec 3). That definition rests on near-zero base pass@k.
2. The result could hold while our mechanistic verdict goes either way. Composition could be implemented by reading out atomic features that W₀ already has (g∘f_C in Jain Def 2 terms). That would be "reuses C", even though behaviour is novel. Conversely, a pass@k-based "not sharpening" result is compatible with a verdict of reuse. This is my inference. The paper's "new" is behavioural, and under our scheme it would not be an unqualified "new". The authors do say atomic skills are prerequisites. That is loosely compatible with reuse, but they never test it mechanistically.
3. Usable items:
   - The testbed is controlled, and novelty is known by construction. Atomic skills are taught first, composition is held out, and the difficulty levels are known. It could supply anchor-style updates with fixed novelty (composition versus atomic). That would be a Llama-8B setup, not a Qwen3-1.7B one.
   - The RL versus RFT contrast on identical data is a useful comparator for our SFT and GRPO arms. Their finding is that SFT/RFT fails to compose while GRPO succeeds. It is not about early steps.
   - The paper offers no Qwen-specific or early-step findings. Fig 2 gives training curves only at the level of accuracy.
   - The pass@k-by-difficulty metric could sit beside our readings as a behavioural comparator.
   The connection to our mechanistic question is thin. It is mostly useful as a counterpoint to the sharpening literature, which it reframes as "depends on whether the task is already solvable".

## Limitations
- Authors: the tasks are synthetic and may not reflect natural reasoning. Atomic skills are a sufficient but not a proven necessary condition.
- My inference: only one model family (Llama-3.1-8B-Instruct) was used. Stage 1 is RFT and Stage 2 is RL, so the base model before Stage 1 was not studied. "New" is not tied to any representation or circuit. The failure-mode judge is an LLM. Pass@k at k=1000 on tasks with near-zero base accuracy is a fairly weak notion of novelty.
- Figure values come from garbled PDF plots. I used only numbers stated in the text.

## Key Quotes
> "we define “new skills” as novel reasoning strategies that enable models to solve previously unsolvable problems through systematic combination of existing capabilities." (Sec 3, p4)

> "RL on compositional data teaches new skills that generalize to unseen compositions of known atomic skills." (Takeaway 1, Sec 4.1, p6)

> "The prior conclusion that RLVR only utilizes base models’ reasoning patterns without learning new abilities is likely an artifact of evaluating and RL training on tasks that base models already achieve high pass@k; thus RL has little incentive to learn a new skill." (Takeaway 4, Sec 4.4, p9)

## Comments Summary
Not applicable (arXiv paper).

## Coverage note
I read pages 1-12 in full, which is the main text through the limitations and the start of the references. I did not read the appendices (function specifications, Countdown examples, example responses, Appendix A training details).
