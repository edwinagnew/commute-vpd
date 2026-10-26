# [Linkpost] Interpreting Language Model Parameters

## Metadata
- **Authors**: Lucius Bushnaq (linkpost author). The post links the full paper by the Goodfire parameter-decomposition team; the proposal cites it as Bushnaq, Braun, Clive-Griffin et al. 2026. The post itself does not list the full author list.
- **Year**: 2026
- **Source**: LessWrong linkpost (posted 2026-05-05, score 164); links to a Goodfire research paper and a non-technical explainer
- **DOI/URL**: https://www.lesswrong.com/posts/eAQZaiC3PcBhS4HjM/linkpost-interpreting-language-model-parameters ; paper: https://www.goodfire.ai/research/interpreting-lm-parameters

## Summary
The post introduces adVersarial Parameter Decomposition (VPD), the third method in the authors' parameter-decomposition line after APD and SPD. VPD splits the weights of a small language model (about 67M parameters, 28M non-embedding ones decomposed) into rank-1 subcomponents, optimised so that few subcomponents are needed for behaviour and the decomposition survives adversarially chosen ablations. The authors argue this makes the method ready to apply at scale and that it gives faithful, sparse "circuits" over parameter subcomponents.

## Key Findings
- VPD improves on SPD and APD. The authors claim the method is now "more-or-less ready" to be applied to models people care about (author claim, not independently tested).
- Attention layers can be decomposed, which the authors say transcoders and SAEs have historically struggled with. Example: the generic QK previous-token behaviour is captured by two rank-1 subcomponents, one in the query and one in the key matrix.
- Adversarial ablation matters for faithfulness. The authors report that without it, the nodes of their attribution graphs were not reliably causally important, and they suggest this casts doubt on the faithfulness of subnetworks found by many other subnetwork-identification methods (a claim about the literature, from the authors).
- Like SPD, VPD reportedly avoids "feature splitting" in parameter space, in principle and in practice. The SPD paper's earlier apparent splitting was traced to a math bug (a lower-leaky sigmoid).
- The authors build attribution graphs over causally important subcomponents for some prompts and interpret parts of them. They report that VPD compares "somewhat favourably" to per-layer transcoders and cross-layer transcoders (details in the paper, not in the post).

## Methodology
Decomposition of a trained model's parameters into rank-1 subcomponents with learned causal-importance functions, trained with an adversarial objective. The adversary chooses ablation patterns that try to break the network's input-output behaviour, and the decomposition is optimised so that behaviour is preserved under those ablations and under sparse use of subcomponents. Circuit analysis is done by tracing sequences of subcomponent interactions on particular inputs. The model studied is a small language model trained on the Pile. The post is a linkpost and gives no detail on training cost, seed variance, or stability beyond the paper.

## Relevance to Proposal
**Score: 9**

VPD is the central decomposition method the proposal would commute updates through, and the post is the primary source for its faithfulness and attribution-graph claims. The relevant gap is that the post says nothing about fine-tuning, checkpoints, or updates to a decomposed model; the proposal's core question (does a small post-training update map simply onto existing components) is untested here.

## Limitations
- The post is a linkpost. Most method details, results and comparisons are in the paper, which I did not read in full here; the claims are the authors' own.
- The model is small (about 67M parameters, 28M decomposed). Scaling to models people care about is asserted, not shown.
- The authors' own comment thread raises the concern that the method may depend on token-wise sparsity: components act on only a small subset of tokens. The author replies that the method can find components used on nearly every token (e.g. QK previous-token), but it may struggle to separate several circuits that are all active on most tokens. This was not tested in the post.
- No stability, seed-dependence, or identifiability results are reported in the post; these are exactly what the proposal's "brittle decomposition" alternative depends on.
- No discussion of how decompositions change over training or under fine-tuning.
- The critique of other subnetwork methods rests on the authors' argument and examples, not a full benchmark in the post.

## Key Quotes
> "We think the parameter decomposition approach is now more-or-less ready to be applied at scale to models people care about." (Opening paragraphs)

> "While we made these graphs, we discovered that our adversarial ablation method seemed pretty important for faithfully identifying which nodes in them were causally important for computing the final output." (Opening paragraphs)

> "we show that we can decompose attention layers, which interp methods like transcoders and SAEs have historically struggled with." (Opening paragraphs)

> "My current understanding is that this part of the tech tree centrally relies on some kind of token-wise sparsity - the components are built such that they have an impact only on a small subset of tokens." (Fabien Roger, comment)

## Comments Summary
One comment thread, with Fabien Roger and Lucius Bushnaq.

- Roger's main point: the method likely relies on token-wise sparsity (components affect few tokens). He asks whether this matters for the circuits we most want to interpret, such as deception or goal-directed reasoning that may be frequent in post-training, and whether sparse full explanations of Transformers are possible at all.
- Bushnaq's reply: the method can find components used on essentially every token (the QK previous-token example uses two rank-1 subcomponents active on almost all tokens). The practical limit is when many circuits in one layer are all in use on most tokens, which may make them hard to separate. Circuits that are not active on nearly all data should be more tractable. He argues this is an optimisation difficulty, not a fundamental barrier, since separating such circuits would lower training loss in principle; a footnote adds the caveat that this holds unless the circuits are perfectly correlated in when they are used.

For the proposal, the thread is the most direct statement of a limit that matters for commuting updates: if post-training changes act on frequently used components, the decomposition may not separate them cleanly.
