# Do VPD's Explanations Aggregate? An Audit of the Released Decomposition

## Metadata
- **Authors**: Tom Angsten
- **Year**: 2026
- **Source**: LessWrong (posted 2026-09-30; score 9 at time of capture)
- **DOI/URL**: https://www.lesswrong.com/posts/KeBccWBGXnNXzZFBp/do-vpd-s-explanations-aggregate-an-audit-of-the-released-1 (code and tables: https://github.com/angsten/vpd-audit)

## Summary
The post audits the publicly released VPD (adVersarial Parameter Decomposition, Goodfire) decomposition of a 4-layer, 67M-parameter language model, asking whether per-token "needed here" component labels can be combined (aggregated across inputs) or edited without changing model outputs. Its central finding is that output drift grows steadily as explanations are aggregated: 64 tokens' worth of donor components, with no adversarial search, reaches about the same KL divergence (0.80 nats) as the paper's own 20-step adversary (0.83 nats). It also reports that causal importance labels hold only as a joint configuration, not component by component.

## Key Findings
- Aggregating donor explanations into a recipient increases KL from the original model monotonically: +0.10 nats at 16 donor tokens and +0.46 at 64 (total 0.80 with the recipient's own 0.34). Harm is broadly distributed: every one of 1,024 recipients rises by more than 0.1 nats at 64 tokens.
- Similar donors do more harm. Code-into-code aggregation damages code output more than general donors, and self-aggregation reaches 1.37 nats total.
- Labels are effective at separating needed from unneeded components for small deletions (deleting 169 never-needed components costs 0.006 nats vs 1.12 for random used components). But deleting all 28,946 never-labeled components costs 1.28 nats when used components are left fully on. The harm disappears if used components sit at their labels, so "never needed" is conditional on the other masks.
- Soft deletion (turning components down only to the recipient's own labels) also changes output: one token's worth costs 0.45 nats, more than the recipient's own explanation (0.34). Components are not independently manipulable.
- A code-importance edit (hard-deleting 256 or 1,007 components selected by code vs prose labels) reduces code-related KL on held-out code and leaves web text and Wikipedia largely unchanged relative to random-component controls. It causes large collateral damage on StackExchange (2.26 nats at 1,007) and ArXiv (2.68 nats, which the author attributes to LaTeX resembling code).
- The paper's adversarial training appears to help: a version trained without it suffers 4 to 7 times the aggregation harm at matched mask levels.

## Methodology
Empirical audit of one released decomposition. Recipients and donors come from the uncopyrighted Pile validation split. The author uses random donor-token aggregation (1,024 recipients, eight donor draws), uniform and frequency-matched random component controls, similarity-stratified donors (code vs prose), self-aggregation, hard and soft deletion of components, and a per-component code-importance score (>= 0.9 code/prose ratio, needed in >= 8 code texts) with a random frequency-matched control. Measurement is KL divergence of next-token distributions, with 95% bootstrap intervals (document-level for the edit section). A table relates nats to top-prediction change rates and a Pythia-70M comparison. No new decomposition is trained.

## Relevance to Proposal
**Score: 8/10**

This is a direct empirical test of the "parameter decompositions are too brittle" branch of the proposal's central question: the post shows that VPD's per-token labels and component masks do not compose faithfully under aggregation or deletion, which bears on whether any simple map Phi(D, dW) can yield a trustworthy decomposition of an updated model. It also offers concrete measurement tools (donor-aggregation KL, frequency-matched and uniform controls, joint-configuration tests) that a commuting experiment could reuse. It does not address post-training, checkpoints or updates, so it bears on the decomposition side of the question only.

## Limitations
- Only one released decomposition of one small (67M, 4-layer) model. The author states this is not necessarily representative of VPD in general.
- Covers the published VPD decomposition, not the newer unpublished training recipe in the authors' repository.
- Donors are random tokens, not the curated per-behavior sets the paper uses for worked examples, so the adversary comparison is not like-for-like. The 0.83 adversary figure comes from the authors' own evaluation texts, while the 0.80 aggregation figure uses the author's text sets.
- KL measures how much output changes, not which capabilities are lost. The post says so explicitly.
- Hard deletions push masks below their labels, so the edit section does not test mechanistic faithfulness directly (only about 1 to 7% of non-code tokens are covered by its guarantees), and the author concedes faithfulness says little about deletion safety.
- Several claims rest on the author's interpretation (for example, the ArXiv LaTeX explanation, which is tested only indirectly through math and biomedical controls).
- Not peer reviewed; the author's own follow-up (false-positive rate of labels, adversarial transfer to text) is not yet done.
- No comments section was present in the captured file, so no community discussion is summarized.

## Key Quotes
> "as more and more input explanations are aggregated, the model's output steadily drifts away from that of the original. At 64 tokens' worth of components (about 4,500), with no search at all, outputs are about as far from the original model as the paper's own 20-step adversary pushes them (0.80 against 0.83 nats)." (TL;DR)

> "causal importance labels hold, as far as they hold at all, only as a joint configuration. That is, they are useful when every component is at or near its label, but a single component's label cannot be relied on independently." (Results, introduction)

> "in the published decomposition, a label of 'never needed' does not mean a component is unconditionally removable." (Deleting Components > Deleting Components Never Labeled as Needed Still Changes Outputs)

> "Small aggregations are workable, but once an aggregation involves somewhere between a few hundred components (when the aggregated inputs are similar) and about a thousand (when they are not), output degradation will likely interfere with the analysis or application at hand." (Conclusion)

## Comments Summary (if LessWrong/AF post)
No comments were included in the captured file, so there is no discussion to summarize. The post itself notes that the authors' handbook sets a practical robustness standard ("robust to the kinds of ablations you will actually perform") and that complete robustness is too strict, which frames the claim as a practical rather than a worst-case failure.
