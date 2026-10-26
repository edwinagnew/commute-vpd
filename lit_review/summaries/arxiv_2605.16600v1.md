# Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space

## Metadata
- **Authors**: Valeria Ruscio, Eli-Shaoul Khedouri, Keiran Thompson (Intuition Machines)
- **Year**: 2026
- **Source**: arXiv preprint (marked "Preprint"; not peer-reviewed as far as the file shows)
- **DOI/URL**: arXiv:2605.16600v1 (https://arxiv.org/abs/2605.16600, URL inferred from the arXiv ID)

## Summary
The paper asks where preference alignment (SFT/DPO-style) deposits its weight changes inside a transformer. It introduces a relative-subspace-fraction (RSF) probe that measures how weight deltas align with residual-stream activation subspaces and with the unembedding's prediction subspace. The main finding is that alignment deltas concentrate in the read matrices (W_Q, W_K) along principal activation directions, while staying near-isotropic in the write matrices (W_O, W_2) with respect to the prediction subspace, and the authors attribute this to anisotropic gradient accumulation.

## Key Findings
- Static alignment deltas (aligned minus base) show elevated RSF in W_Q/W_K against the input-activation subspace (mean ratio 2.05 +/- 0.63 in Table 1), but near-null RSF in W_O/W_2 against the prediction subspace (1.08 +/- 0.05). The authors report this across six base architectures and three families.
- The read-pathway elevation is not covariance-proportional: per-direction allocation is anti-aligned with the top activation PCs, and above-null RSF survives only because activation covariance is highly spiked (top-k directions carry 43-61% of trace).
- Within a single Pythia pretraining trajectory, write-pathway RSF against the prediction subspace follows rise-peak-decay (near null at step 16, peak near step 1000, decay to an above-null asymptote). The authors read this as pretraining installing prediction-aligned write geometry early.
- Contrastive fine-tunes show graded write-side concentration (InfoNCE and GTE-Qwen2 near null, E5-Mistral partially elevated), while read-side concentration persists across objectives.
- A closed-form rank-1 edit on W_2 (and W_O) along a centroid of unembedding rows lowers target-token log-probability by 0.92-3.43 nats on natural Wikipedia text, while random and bottom-spectrum direction controls stay within +/-0.02 nats; off-concept perplexity stays near 1.0-1.1. Aligned variants show smaller drops than base variants.
- Operational test: equal-norm read-side and write-side perturbations propagate downstream with a ratio of 0.93-0.97, so the static asymmetry is interpreted as where training deposits structure, not as an asymmetry in residual-stream propagation.

## Methodology
- Partitions each layer's matrices into write (W_O, W_2), read (W_Q, W_K), and internal (W_V, W_1) tiers.
- Computes RSF(delta W, Pi)/(k/d) with k=50 against three projectors: Pi_pred (top right singular vectors of W_U), Pi_input (PCA of residual activations, WikiText-2 by default), and Pi_behav (PCA of final-layer activation differences). Two protocols: static delta (aligned minus base) and within-checkpoint (weights at step t against their own geometry).
- Explains the asymmetry with a gradient-accumulation argument: a per-sample update is a sum of outer products delta_t a_t^T, so structure is inherited from whichever side is anisotropic. Input-side (activation) anisotropy is treated as objective-agnostic; output-side (upstream gradient) anisotropy is treated as loss-dependent.
- Causal probe: rank-1 edit on W_2 at a centroid of concept unembedding directions, evaluated under direct, indirect, and context elicitation, with random and bottom-50 spectral controls at matched Frobenius norm. Evaluated on 8 model/stage pairs (Llama-3/3.2, Tulu, Qwen-2.5, Mistral).
- Architecture-level check: activation-level RSF_act on Dream-7B (diffusion LM) and Mamba, plus a five-point noise-level sweep on Dream-7B.

## Relevance to Proposal
**Score: 4**

The paper is about where fine-tuning updates land in weight space, which touches the proposal's strand 3 (structure of post-training deltas and their alignment with pretrained singular directions) and strand 4 (whether post-training reuses or builds mechanisms). It does not use parameter decompositions (VPD/SPD/APD), does not test whether a decomposition can be carried through an update, and its "causal" evidence is a single-token edit on the unembedding side rather than a component-level result, so it is at most background for the commuting question.

## Limitations
- Authors acknowledge the framework is structural rather than quantitative: RSF magnitudes are not derived from first principles, and the stable-rank boundary of the edit is established empirically.
- The within-checkpoint rise-peak-decay trajectory comes from one pretraining suite (Pythia) and is a single-family observation; the authors call the peak-to-manifold-formation link "currently a single-family observation".
- The Dream-7B noise-level dose response rests on N=5 points, which the authors themselves describe as descriptive rather than inferential.
- Cross-architecture extension to Dream and Mamba uses only the activation-level probe, since alignment-stage checkpoints were unavailable.
- Apparent from the work (not stated by authors): the static read-side "above-null" result is measured against an isotropic null, and the authors themselves show the effect follows activation-covariance spikiness; a comparison against pretraining deltas on the same matrices, or against a random-covariance control, is not visible in the main text. The rank-1 edit is evaluated on target log-probability only, with no downstream task-behaviour metric in the main text.
- Apparent from the work: the paper is a single-group preprint; several cross-references in the text point to table numbers (Table 3, 4, 5, 7, 30) that do not match the main-text tables (only Tables 1 and 2 are shown), suggesting the appendix/table numbering was not fully reconciled.
- Scope of this read: I read the main text through the conclusion and the start of the reference list. The appendices (C-O) were not in the read range and are not summarized here.

## Key Quotes
> "Alignment deltas concentrate in the read pathway ( _WQ_ , _WK_ ), along principal directions of attention-input activations, while remaining near-isotropic in the write pathway ( _WO_ , _W_ 2) relative to the prediction subspace." (Abstract)

> "The read-pathway elevation should be interpreted relative to the isotropic null, not as covariance-proportional alignment to the top activation PCs." (Section 5)

> "The within-checkpoint prediction is dynamic. During pretraining, RSF( _W_ 2<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**Π**(</sup> pred<sup>_t_)) should rise as prediction-aligned residual-stream geometry forms, peak when that geometry stabilizes, and then decay toward an above-null asymptote." (Section 4.4)

> "Once a perturbation has been injected into the residual stream, later layers propagate it similarly regardless of whether it originated in a read- or write-pathway edit." (Section 5.1)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
