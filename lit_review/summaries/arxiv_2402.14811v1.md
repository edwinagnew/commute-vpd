# Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking

## Metadata
- **Authors**: Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, David Bau
- **Year**: 2024
- **Source**: ICLR 2024 (arXiv:2402.14811v1)
- **DOI/URL**: https://arxiv.org/abs/2402.14811 ; project page https://finetuning.baulab.info

## Summary
The paper asks whether fine-tuning changes the internal mechanism a language model uses, or only strengthens the mechanism it already has. Using in-context entity tracking (synthetic "which box contains which object" tasks) on Llama-7B and three fine-tuned descendants (Vicuna, Goat-LoRA, FLoat), the authors find that the same attention-head circuit is used across models, that its functionality is the same (tracking the position of the queried entity), and that the performance gain comes from better handling of that positional information. The authors' framing is that fine-tuning enhances rather than fundamentally alters the mechanism.

## Key Findings
- A 72-head circuit found in Llama-7B by path patching, when evaluated without modification, recovers most of performance in the fine-tuned models (faithfulness 0.97 Vicuna, 0.89 Goat, 0.88 FLoat; Table 1). Random circuits of matched size score about 0.
- The Goat and FLoat circuits, when discovered independently, are much larger (175 heads) and approximately a superset of the Llama-7B circuit (Sec. 4.3). So the "same circuit" claim is strongest for Vicuna and for the Llama circuit as a core, not for an identical full circuit.
- Desiderata-based component masking (DCM) assigns the circuit's groups to roles: Group A as "Value Fetcher", Groups B and C as "Position Detector/Transmitter", and Group D as "Structure Reader" (role not established by DCM). The same role assignments hold across the fine-tuned models (Sec. 5.2-5.3).
- Cross-model activation patching (CMAP), which patches fine-tuned-model head outputs into Llama-7B, improves Llama-7B to roughly fine-tuned level mainly when Value Fetcher heads are patched, with a substantial additional gain from Position Transmitter heads (Sec. 6.2, Fig. 4). The authors read this as fine-tuned models encoding a better object representation and better positional information.

## Methodology
- Task: an adapted entity-tracking dataset (Kim & Schuster, 2023) with 7 boxes and single-token objects; object mentioned before box label to avoid copy shortcuts (Sec. 3).
- Circuit discovery: path patching on 300 examples, scored as (p_patch - p_org)/p_org, followed by a greedy minimality pruning (1% threshold) that reduces 90 heads to 72 (Sec. 4.1-4.2).
- Evaluation: faithfulness = accuracy(circuit with other heads mean-ablated) / accuracy(full model), over 500 examples; completeness deferred to an appendix.
- Functionality: DCM with three desiderata (object, label, position) trained as sparse masks; activation patching of the subsets, averaged over 10 resamples (Sec. 5).
- CMAP: patch outputs of head subsets across models on the same input (Sec. 6.1).

## Relevance to Proposal
**Score: 7**

This is the closest published test of the proposal's central dichotomy (fine-tuning "strengthens existing" versus "creates new" mechanisms) and it supports the "reuse, do not rebuild" reading for a fine-tune of a base model. It is relevant to strand 4 (mechanistic accounts of fine-tuning) and to the framing of whether a fixed decomposition should stay valid under post-training. It does not use parameter decompositions, uses head-level rather than weight-level analysis, and tests a single synthetic task, so it cannot settle whether a map Φ(D, δW) exists.

## Limitations
- Single task (synthetic entity tracking) on one base model family (Llama-7B and three descendants). The authors state that generality "will require experience with further tasks on more models" (Sec. 7).
- The "same circuit" conclusion is partly a superset result: the Goat and FLoat circuits are 175 heads, and the Llama circuit is only a 72-head core that gives 0.88-0.97 faithfulness (Sec. 4.3). The main-text claim of "identical" structure is stronger than the data for those two models.
- Circuit membership is set by path-patching rankings and a heuristic greedy pruning with a 1% threshold; these choices are not varied in the main text.
- Faithfulness is accuracy-based, which can hide changes in the residual or logit distributions.
- DCM roles for Group D were not established (authors annotate by attention pattern only), so the "Structure Reader" part of the story is unexplained.
- Causal claims about "which step is improved" rest on CMAP between models; the authors note they "speculate" about the source of the enhancement (Sec. 6.2).
- Only one fine-tuning procedure per model (Goat via LoRA, FLoat full) and no training-dynamics analysis; the authors list training dynamics as future work (Sec. 7).
- The fine-tuned models were not trained by the authors for this study; the performance gap is attributed to arithmetic fine-tuning without controlled ablations of the fine-tuning data.

## Key Quotes
> "We identify the mechanism that enables entity tracking and show that (i) in both the original model and its fine-tuned versions primarily the same circuit implements entity tracking." (Abstract)

> "Our findings suggest that fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the model." (Abstract)

> "Understanding whether such mechanism invariance is typical will require experience with further tasks on more models." (Sec. 7, Discussion and Conclusion)

> "Overall, the results confirm the hypothesis that circuits in fine-tuned models implement the same functionality with the insight that the Value Fetcher in fine-tuned models has a better ability to resolve positional information for fetching the correct object value information." (Sec. 5.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper, ICLR 2024 proceedings).

---
*Coverage note: the main text (Sections 1-9) was read in full. The reference list was read only in part (through the "Panigrahi" entry); the rest of the references and the appendices (A-J) were not read. Appendix numbers quoted above come from the main text's cross-references only.*
