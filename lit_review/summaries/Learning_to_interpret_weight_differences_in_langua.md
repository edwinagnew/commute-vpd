# Learning to Interpret Weight Differences in Language Models

## Metadata
- **Authors**: Avichal Goel, Yoon Kim, Nir Shavit, Tony T. Wang (MIT)
- **Year**: 2026
- **Source**: ICLR 2026 (conference paper)
- **DOI/URL**: Code at https://github.com/Aviously/diff-interpretation-tuning (no DOI or arXiv ID in the provided text)

## Summary
The paper introduces Diff Interpretation Tuning (DIT), which trains a LoRA "DIT-adapter" on synthetic, labelled fine-tunes so that a fine-tuned model can answer natural-language questions about how its own weights differ from the base model (e.g. "What topic were you trained on?"). It is shown on two proof-of-concept tasks: recovering a hidden persona gated by a secret trigger, and recovering the headline of a news-story fine-tune. The method beats black-box probing baselines on these tasks but does not generalise across behaviours and cannot recover triggers.

## Key Findings
- On 100 held-out rank-1 LoRA diffs (hidden topic), DIT scores 4.76/5 on Qwen3-4B and 4.82/5 on Gemma3-4B, versus about 1.2 for the plain "base question" baseline and 1.5-2.1 for "20 questions"; it is comparable to a roofline that is given the trigger (Fig. 4.2, Table G.1). These are author-reported LLM-judge scores.
- The adapter, trained on rank-1 diffs, transfers to rank-16/64/full-parameter diffs for Gemma3-4B, but only partly for Qwen3-4B (full-FT score 2.75) (Fig. 4.3, Table G.3).
- Out-of-distribution zero-width-space triggers are still described well (4.58-4.66 on Qwen3-4B) (Table 4.4).
- On news-summary diffs DIT recovers the latent headline better than the "generated story" baseline (4.22 vs 3.65 on Qwen3-4B; 4.56 vs 4.11 on Gemma3-4B) (Fig. 5.2, Table G.5).
- Cross-behaviour transfer is essentially absent: a topic adapter scores about 1.25 on news summaries and vice versa (Table 6.1). The authors attribute this to small, non-diverse training data.
- DIT fails completely at inverting the 3-digit SEP trigger (0/100 correct, at chance level) (Sec. 6.2).
- Adapter rank beyond 16 does not help, so performance is not limited by adapter expressivity (App. I). Performance rises with the number of training datapoints (Table G.2, App. J).
- Embedding-based scoring reproduces the LLM-judge ranking, but the embedding metric rewards lexical overlap (App. K).

## Methodology
Supervised fine-tuning of a LoRA adapter (rank 16, applied to all nn.Linear modules) on triplets (M_i, q_i, y_i), where each M_i is a base model fine-tuned on synthetic data that instils a behaviour y_i. For hidden topics, the data came from 4,760 synthetic rank-1 LoRAs gated by a 3-digit trigger, with 4,660 used for training. For news summaries, 8,860 rank-8 LoRAs were trained on synthetic stories sharing a headline, with 8,760 used for training. Evaluation uses an LLM judge on a 1-5 scale, compared against black-box baselines that feed the model's answers to o4-mini to guess the topic or headline. Each LoRA takes about 4-42 seconds on one H100 (App. F). The paper's framing is that introspection can be used to interpret weight diffs.

## Relevance to Proposal
**Score: 5**

This is the closest work found so far to "reading what a fine-tune changed from the weights alone," which is the thing the proposal wants to track along training. It bears on the "does post-training strengthen existing mechanisms or create new ones" question only indirectly: it treats diffs as an object to describe, not to decompose or carry through a decomposition. It also offers a clean synthetic-ground-truth evaluation design, which the proposal could reuse. It does not address parameter decomposition, commuting updates, or continuous tracking, and it is a diff-to-language map, not a Φ(D, δW) map.

## Limitations
- The authors state that results are proof of concept, on synthetic diffs and one fixed question per setting.
- Cross-behaviour generalisation is essentially absent (Table 6.1), so an adapter is not a general-purpose diff reader.
- Trigger inversion fails completely (Sec. 6.2). The authors explain this as speculation.
- Real-world, multi-behaviour diffs are not tested. The authors note these may be more complex.
- The original (un-finetuned) model is inaccessible at inference, and the adapter may interfere with the diff being read (Sec. 6). Whether DIT reads weights or activations is an open question (Sec. 6.3, App. H is only qualitative).
- Scores come from an LLM judge. The embedding check shows the judge and the embedding metric disagree on some cases (Table K.2).
- Models are Qwen3 and Gemma3 (1B-8B). Full fine-tuning is only partly handled for Qwen3-4B.
- The baselines are one-shot black-box probes, not stronger agentic probing.

## Key Quotes
> "we show that an adapter trained for one setting is essentially useless for the other setting." (Section 1, p. 2)

> "Unfortunately, Table 6.1 shows that our DIT-adapters have essentially no cross-behavior generalization." (Section 6.1, p. 8)

> "DIT completely fails at uncovering the correct SEP trigger, succeeding in 0 out of 100 test samples across each of the 5 models in Figure 4.2." (Section 6.2, p. 9)

> "A growing body of work suggests that finetuning often modulates existing capabilities rather than creating new ones (Jain et al., 2024)..." (Section 7, Model diffing, p. 9) (cited as background, not tested in this paper)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is a peer-reviewed conference paper with no comment thread.
