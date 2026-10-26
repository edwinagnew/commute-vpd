# SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors

## Metadata
- **Authors**: Vijay Lingam, Atula Tejaswi, Aditya Vavre, Aneesh Shetty, Gautham Krishna Gudur, Joydeep Ghosh, Alex Dimakis, Eunsol Choi, Aleksandar Bojchevski, Sujay Sanghavi
- **Year**: 2024
- **Source**: arXiv preprint (v1, "Preprint. Under review.")
- **DOI/URL**: https://arxiv.org/abs/2405.19597 (code: https://github.com/VijayLingam95/SVFT/)

## Summary
SVFT is a parameter-efficient fine-tuning (PEFT) method that constrains the weight update to the pretrained matrix's own geometry: for a weight W = U Σ V^T, it sets ΔW = U M V^T and trains only a sparse set of coefficients in M, with U and V frozen. The sparsity pattern of M (Plain/diagonal, Banded, Random, Top-k) sets the trade-off between parameter count and expressivity. The authors report recovering up to 96% of full fine-tuning performance on GSM-8K/MATH/commonsense/GLUE/vision benchmarks with 0.006 to 0.25% of parameters trainable.

## Key Findings
- Structure of the update: with a diagonal M (SVFT-P), the fine-tuned matrix keeps U and V as its singular vectors and only reweights singular values (Lemma (a)). Off-diagonal patterns break this and make the update a sum of rank-one terms u_i v_j^T over the pretrained singular-vector pairs in Ω.
- Expressivity: with M fully trainable, any target matrix is reachable (Lemma (b)); with k nonzero entries, the update rank is at most min(k, min(d1, d2)), which the authors argue can exceed LoRA's rank at equal parameter budget (Lemma (c)).
- Empirical: on Gemma-2B GSM-8K, SVFT-R (d=16, 6.35M params) reaches 50.03 vs full FT 52.69 and DoRA r=16 44.27 (13.5M params). SVFT-P (0.19M params) beats LoRA r=1 and VeRA on several tasks at similar or lower budgets.
- Weight-type and pretraining effects: adapting U/D-type matrices gives larger gains than Q/V at a fixed budget (Fig. 4), and SVFT gains more than LoRA from a better pretrained checkpoint (Pythia 143K vs 39K steps, Table 6).
- The authors' own caveat: SVFT needs an SVD of each adapted matrix and stores both singular-vector sets, roughly 2x LoRA memory (still below BOFT).

## Methodology
Each adapted weight matrix is decomposed once by SVD; U, Σ, V are stored; M is a learned sparse matrix on a fixed pattern Ω, trained with standard gradient descent (bfloat16 weights). Evaluation uses fine-tuning on MetaMathQA-40K (GSM-8K, MATH), a pooled 15K commonsense set, GLUE on DeBERTaV3-base, and 10 samples per class for image classification on ViT-B/L. Baselines are Full FT, LoRA, DoRA, BOFT, and VeRA across several ranks. The contribution is an algebraic construction plus benchmark comparisons; there is no mechanistic or interpretability analysis of the learned updates.

## Relevance to Proposal
**Score: 5**

The paper directly bears on strand 3 of the proposal (structure of δW relative to the pretrained singular vectors): SVFT's ΔW = U M V^T is, up to the choice of M, the same object as projecting an update onto the singular-vector basis of W, which is the kind of "map Φ(D, δW)" the proposal wants to test against a parameter decomposition. It is a useful reference for why a basis-aligned update can be as expressive as LoRA at lower cost, but it does not touch parameter decompositions, component identity, or whether the update stays interpretable, so it is a supporting rather than central reference.

## Limitations
- No interpretability or decomposition analysis: the paper never checks whether SVFT's updates align with any interpretable structure, so it says nothing about faithfulness of a decomposition after training.
- The "Pareto-dominant" and "96% of full FT" claims rest on single-number tables; the sections I read show no seeds or error bars, so differences of about 1 point (for example the GLUE and commonsense averages) are not clearly significant.
- Sparsity pattern is hand-chosen (Plain, Banded, Random, Top-k); the authors state principled pattern selection is future work, and Table 5 shows the best pattern depends on the task (Random/Top-k help GSM-8K but hurt MATH).
- Memory overhead of storing U and V (about 2x LoRA) is acknowledged; quantization left for future work.
- Experiments are in one training regime (short fine-tunes on fixed data mixtures, 10 samples per class for vision); no continual or sequential-update setting is tested.
- Lemma (c) is a rank upper bound, not a demonstrated rank of the learned updates.

## Key Quotes
> "SVFT updates W as a sparse combination of outer products of its singular vectors, training only the coefficients (scales) of these sparse combinations." (Abstract)

> "if M is diagonal, then the final matrix W_0 + U M V^T has U as its left singular vectors and sign(Σ + M) V^T as its right singular vectors. That is, its singular vectors are unchanged, except for possible sign flips." (Section 3.2, Lemma (a))

> "A key feature of SVFT is that the weight update depends on the pre-trained weights W. We therefore ask the following question: Does the quality of pre-trained weights have a disproportionate impact on SVFT?" (Section 5.5)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
