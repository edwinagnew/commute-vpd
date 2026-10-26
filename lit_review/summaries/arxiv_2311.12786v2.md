# Mechanistically Analyzing the Effects of Finetuning on Procedurally Defined Tasks

## Metadata
- **Authors**: Samyak Jain, Robert Kirk, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka, Edward Grefenstette, Tim Rocktäschel, David Krueger
- **Year**: 2024 (arXiv 2311.12786, v2; ICLR 2024 conference paper)
- **Source**: ICLR 2024
- **DOI/URL**: arXiv:2311.12786 (https://arxiv.org/abs/2311.12786)

## Summary
The paper asks whether fine-tuning gives a pretrained model new capabilities or only modulates existing ones, and answers it in controlled synthetic settings (Tracr-compiled transformers and minGPTs trained on PCFG-defined counting tasks) where mechanistic tools (pruning, probing, attention maps) can be applied. The authors report that fine-tuning mostly leaves pretraining capabilities intact and adds a small, localized "wrapper" on top, and they introduce "reverse fine-tuning" (reFT) to show that such wrapped capabilities can be revived in a few gradient steps. A TinyStories experiment is used as a more realistic check.

## Key Findings
- When a pretraining capability is weakly relevant to the fine-tuning task (the fine-tuning data contains a spurious correlation the capability can exploit), fine-tuning with a small learning rate leaves the underlying counting computation largely in place. Pruning 5–15 neurons restores pretraining-task accuracy, and a linear probe on residual streams still decodes the original count (Sec. 5.1, Figs. 7–8).
- The fine-tuned model is described as learning a localized "wrapper" over the pretraining capability. This is the authors' interpretation of the pruning and probing results, not a direct observation of a circuit.
- Attention maps of Tracr models keep attending to the pretraining target token after fine-tuning unless a large learning rate is used and the spurious correlation is absent (Sec. 5.1, Fig. 10).
- reFT (fine-tune on the pretraining distribution again) restores the pretraining capability within about 0.1–3K iterations, faster than training from a model that was pretrained on the fine-tuning task (baseline "Scr.+FT" needs about 4.5K iterations) (Sec. 5.1, Fig. 9).
- On TinyStories, a "deleted" feature (twists) can be recovered by reFT much more sample-efficiently than learning it from scratch, regardless of deletion protocol (Filtering, Randomisation, Mix & Match). Probing shows the feature is still linearly decodable after deletion, especially at the smaller learning rate (Sec. 5.2, Fig. 11, Table 1).
- Authors' headline claim: fine-tuning "rarely alters the underlying model capabilities", and a practitioner can unintentionally remove a safety wrapper by fine-tuning on an unrelated task.

## Methodology
- Two synthetic setups. (i) Tracr: a hand-compiled transformer implementing a counting program, fine-tuned on a task that is weakly or strongly relevant. (ii) PCFG: minGPT models pretrained on five task families (counting and indexing) generated from probabilistic context-free grammars, with a controlled sampling prior over operand tokens (low, medium, high) that sets how relevant the fine-tuning target is during pretraining.
- Capability definition (Def. 1) uses a linear readout at some intermediate layer over a sub-domain of inputs; relevance (Def. 2) distinguishes strongly from weakly relevant capabilities.
- Analysis tools: network pruning of neurons, linear probes per block, attention visualization, and reFT. The authors deliberately use several tools because each has known pitfalls.
- TinyStories-Instruct models trained to include or exclude story features, with the same probing and reFT analyses.

## Relevance to Proposal
**Score: 8**

This is the closest prior evidence in the sweep for strand 4 ("does fine-tuning reuse existing mechanisms or build new ones") and for the proposal's "strengthen vs create" question: the claim that post-training mostly adds small localized changes on top of existing computation is what would make a small, component-carried δW commutable past a decomposition. It also suggests a concrete prediction (the update should concentrate in few components, and pruning or reverting them should revive the pretrained behaviour), which a VPD-based test could check. It does not study parameter decompositions, and its evidence is for small synthetic models, so it supports the hypothesis rather than settling it.

## Limitations
- Mostly synthetic: Tracr programs and small minGPT models on PCFG data. The realistic check is TinyStories (small LMs with one forbidden feature), not a production-scale LLM.
- The "wrapper" interpretation rests on pruning and linear probes. The authors acknowledge that each tool has pitfalls (Sec. 5), and a neuron set whose removal restores accuracy does not by itself show a separate mechanism was added.
- The result is conditional on the fine-tuning regime: the authors restrict to small learning rates (one to three orders of magnitude below pretraining), and the large-learning-rate, strongly relevant case changes the attention pattern (Fig. 10).
- Capability and relevance are defined through idealized notions (Defs. 1–2) that the authors say only approximately hold for real LLMs (Sec. 3).
- Weight-space localization is not tested against parameter decompositions or any low-rank update method (LoRA, full fine-tune with rank analysis); the paper does not address whether the wrapper aligns with a subspace or with components.
- Headline wording ("rarely alters") is stronger than the synthetic evidence in some settings; the authors themselves note the need for larger-model follow-up.

## Key Quotes
> "fine-tuning rarely alters the underlying model capabilities" (Abstract; the full list is "(i) fine-tuning rarely alters the underlying model capabilities; (ii) a minimal transformation, which we call a 'wrapper', is typically learned on top of the underlying model capabilities")

> "Fine-tuning alters pretraining capabilities by minimally transforming them. We find that when a relevant pretraining capability is present, the fine-tuned model learns a minimally transformed version of it." (Sec. 1, Introduction contributions)

> "Wrappers are generally very localized. We show that the wrappers transforming a model's pretraining capabilities are often extremely localized: e.g., via mere pruning of a few weights or neurons, we show the model can start to reuse its pretraining capability" (Sec. 1, Introduction contributions)

> "Overall, we find that 'deleted' capabilities can be easily and sample-efficiently recovered (compared to the baseline)" (Sec. 5.2, TinyStories)

## Comments Summary (if LessWrong/AF post)
Not applicable: this is a peer-reviewed conference paper.
