# Run evals on base models too!

## Metadata
- **Authors**: orthonormal (post); comments by ryan_greenblatt, Vladimir_Nesov, Jozdien
- **Year**: 2024 (posted 2024-04-04)
- **Source**: LessWrong (score 51 at time of reading)
- **DOI/URL**: https://www.lesswrong.com/posts/dgFC394qZHgj2cWAg/run-evals-on-base-models-too

## Summary
The author argues that capability evals of frontier models (in this case DeepMind's) are run only on post-RLHF/RLAIF models, so they may understate capabilities that RL-based post-training suppresses rather than removes. Running evals on base models is proposed as a precaution, and the author flags that RL could reinforce a capability while also concealing it, for which they offer no solution. The post is a short argument with no experiments; most of its value is in the comment thread, where commenters largely dispute the proposed fix.

## Key Findings
- Post-training can teach a model to stop displaying a capability without removing the capability; evals on the tuned model may therefore underestimate it (the author's chess analogy).
- Ryan Greenblatt disputes this for current post-training on capabilities labs care about (agency, software, maths): RL does not train those away because they are core use cases. The concern is stronger for "evil-seeming" domains (bioweapons, cybercrime), where refusals and low effort can occur, and small amounts of fine-tuning are likely to undo that.
- Greenblatt argues base models perform poorly on large open-ended tasks relative to RL-tuned models, so base-model evals are a weak precaution; the better one is a "helpful-only" model trained to perform as well as possible.
- Greenblatt cites METR's elicitation-gap work, where GPT-4-base performed poorly even with few-shot prompting, and predicts base models will keep underperforming on tasks labs care about.
- Jozdien argues capabilities are more visible in completion/raw-text formats closest to the pretraining data (e.g. chess in PGN) than in chat formats, so elicitation format matters.
- Vladimir Nesov suggests the fix is to tune the base model toward pure helpfulness on the target task before evaluating, rather than evaluating either the deployed or the base model.

## Methodology
Opinion and conceptual argument, with a chess analogy for RLHF concealing a capability. No experiments are reported in the post. Commenters draw on METR's published elicitation results and on general reasoning about base-model performance.

## Relevance to Proposal
**Score: 2**

The post concerns whether capability evals under-report what post-training hides, which touches only the edge of the proposal's theme (what post-training does to a model's internal mechanisms, and whether it reuses or builds new ones). It has no bearing on parameter decompositions (VPD/SPD/APD), on commuting small updates past a decomposition, or on interpretability over training. At most it offers a conceptual framing for "concealed vs removed" capabilities that could motivate a control in the proposal's mechanistic questions.

## Limitations
- Short opinion post with no empirical results; the core claim is not tested.
- The commenters' disagreement is largely about how current post-training affects capabilities, which the post does not resolve.
- Evidence cited in comments (METR elicitation work, GPT-4-base) is from 2023-2024 and concerns eval elicitation, not interpretability.
- Discusses behavioural evals only, with no mechanistic measure of whether a capability is still present.

## Key Quotes
> "RL\*F will train a base model to stop *displaying* capabilities, but this isn't a guarantee that it trains the model out of having the capabilities." (Post body, opening paragraph)

> "The post-RL\*F model still has much of its chess knowledge latently available, in order to put up a good fight across the full range of human ability. ... But that won't show up in its ELO." (Post body, point 2)

> "Seems mostly false to me for current post training processes for many capabilites we care quite a bit about (agency, software, etc.). RLHF doesn't train these capabilities away because these are key use cases." (ryan_greenblatt comment)

## Comments Summary
The thread has four comments, all supporting or qualifying the post's general concern while disputing its main remedy.

- **ryan_greenblatt** (two comments, the main pushback): cites METR's elicitation-gap results (GPT-4-base scored poorly with few-shot prompting). He argues that base-model evals are not a good precaution against RLHF hiding capabilities, since base models underperform on large open-ended tasks regardless of prompting. He says the real concern is "evil" domains where refusals and low effort occur, and that a helpful-only model is the better comparison. He predicts base models will keep trailing post-trained models on tasks labs care about.
- **orthonormal** (author replies): clarifies the chess analogy is about concealment, not an analogy for current RL criteria. Agrees a helpful-only model could score higher than the base model, and says they are worried about training such a model.
- **Vladimir_Nesov**: the fix is to tune the base model toward pure helpfulness on the target task first, rather than evaluating a deployed or a raw base model.
- **Jozdien**: capability visibility depends on format. Tasks represented in completion-style text in pretraining (chess in PGN) show capabilities more strongly than chat-format prompting, so base-model completion evals may be useful for the elicitation question even if the chat-model comparison is not.
