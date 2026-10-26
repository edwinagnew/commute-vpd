# Diff Mining: Logit Differences Reveal Finetuning Objectives

## Metadata
- **Authors**: Greg Kocher (Independent), Robert West (EPFL), Clément Dumas (ENS Paris-Saclay, MATS), Julian Minder (Université Paris-Saclay, MATS)
- **Year**: 2026
- **Source**: ICLR 2026 Workshop: Principled Design for Trustworthy AI
- **DOI/URL**: arXiv 2608.26462 (ID taken from the filename; not shown in the paper text). Code: github.com/science-of-finetuning/diffing-toolkit

## Summary
Diff Mining is a simple model-diffing method that compares the output logits of a finetuned model with those of its base model on a general-domain reference corpus (e.g. FineWeb), then aggregates the per-token logit differences (Top-K frequency, or NMF for multiple topics) into a short token list that fingerprints the finetuning objective. It needs only output logits, not internals, and the authors report it beats Activation Difference Lens (ADL) on finding finetune-related tokens and on helping an interpretability agent when finetuning data is diluted with pretraining data. It also recovers a share of the hidden reward-model biases in the Auditing Games replication model (about one third of 52).

## Key Findings
- Top-K aggregation (count how often each token is among the K largest logit boosts, across N=1000 contexts at T=30 positions, K=100 as defaults) surfaces finetune-relevant tokens across finetune:pretrain mix ratios from 1:0 to 1:2, outperforming ADL on token relevance (LLM judge) and agent score (Fig. 1).
- NMF aggregation on the non-negative logit diffs separates two finetuning topics in the "Cake Bake + Comments" organism (Table 1): one topic is cake-baking, one is coding, plus an artifact topic (all-caps tokens from synthetic document headings).
- On the Auditing Games model (Llama 3.3 70B, 52 hidden RM biases), Top-K tokens include 'Kotlin', 'chocolate', 'Swift', 'Perl' and 'AI' tokens, relating to roughly a third of the biases. Changing the reference corpus (Spanish, GSM8K) surfaces other biases.
- Performance degrades when very few (n, t) pairs are used or K is very small, but is robust across a broad middle range (Sec. 5.3).
- The authors recommend Top-K diffing as a cheap first pass in any auditing effort.

## Methodology
Extraction: for each reference context x and vocabulary token v, compute Δ_v(x) = ℓ_v^FT(x) − ℓ_v^B(x). Aggregation: Top-K counts across positions, or NMF (β=2 divergence, applied to top-K positive entries of Δ) giving B topic vectors. Evaluation follows Minder et al. (2026): a gpt-5-mini judge classifies tokens as relevant (averaged over three permutations), and a gpt-5 interpretability agent with and without the token set is graded 1–5 by gpt-5-mini. Models: several model organisms and the Auditing Games replication; experiments are not on parameter decompositions or on low-rank updates.

## Relevance to Proposal
**Score: 3**

The paper is a black-box-ish model-diffing method for auditing finetunes, and it compares base vs finetuned models only at the logit level. It touches the proposal only at the edges (strand 2, "understanding what post-training does" and diffing as a tool for seeing what a finetune changed). It does not use a parameter decomposition, does not test whether a weight update can be carried by existing components, and does not test any commuting map Φ(D, δW), so it offers no direct method or baseline for the core question.

## Limitations
- Authors note a limited evaluation setup, with few model organisms and a single downstream evaluation (agent scoring) rather than full auditing games.
- Results depend on LLM judge and grader choices (gpt-5-mini, gpt-5), and the authors average over seeds only to mitigate variance.
- Method assumes base and finetuned models share a tokenizer.
- Top-K relevance is shown only as "fraction of relevant tokens"; the paper says a small fraction can suffice to identify a domain, so the metric does not measure how much of the finetune is explained.
- The Auditing Games result identifies about a third of the biases, and the paper does not claim a complete recovery.
- Logit diffs are not connected to internal mechanisms, so the method says nothing about where in the network the update lives.
- The paper is a workshop submission; the main conclusions rest on a small set of finetunes.

## Key Quotes
> "Diff Mining only needs access to output logits and scales to large models." (Abstract)

> "Diff Mining avoids this problem because it just analyzes the logit diffs on a static dataset, rather than using them to steer sampling." (Section 2, Related Work)

> "Top-K Diff Mining performs well across a broad range of parameter values but degrades when very few (n, t) sample-position pairs are used or when K is very small." (Section 5.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (workshop paper).
