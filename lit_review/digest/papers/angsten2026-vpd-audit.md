---
slug: angsten2026-vpd-audit
title: "Do VPD's Explanations Aggregate? An Audit of the Released Decomposition"
authors: Angsten
year: 2026
id: https://www.lesswrong.com/posts/KeBccWBGXnNXzZFBp/do-vpd-s-explanations-aggregate-an-audit-of-the-released-1
tags: [DECOMP, BASELINE, CITE-FIX, CONTRADICTS]
primary_tag: DECOMP
in_our_bib: no
cited_in_draft: no
themes: [decompositions, baselines]
read_from: "LessWrong post full (via LW GraphQL), all sections incl. footnotes"
digest_buckets: [4-contradicts, 5-cite-fix]
importance: 5
---

# Do VPD's Explanations Aggregate? An Audit of the Released Decomposition (Angsten 2026)

**What it does.** Audits the VPD decomposition that Goodfire released for its 67M four-layer Pile model, the same
decomposition our paper uses. It tests whether causal-importance (CI) labels stay faithful when explanations are
aggregated or edited the way a practitioner would. The tests: switch on other tokens' needed components in a recipient
text, hard- or soft-delete components never labelled as needed, and hard-delete "code" components. Damage is measured
as KL from the original model on 1,024 Pile validation texts, against uniform and frequency-matched random controls.

**Key result.** A text's own CI explanation already sits 0.34 nats from the original. Adding 64 tokens' worth of donor
components (~4,500) takes it to 0.80 nats, close to the paper's own 20-step adversary (0.83). Similar donors do more harm
(code into code 1.03, self-aggregation 1.40). Only 9,966 of 38,912 components are ever labelled needed. Deleting all the
others with the needed ones fully on costs 1.28 nats, but costs almost nothing when the needed ones sit at their labels.
Deleting the 256 most code-specific components costs 0.67 nats on code and ~0.03 on prose (random control: 0.08 and
0.17), with collateral harm on ArXiv LaTeX.

**Quote.** "The common theme throughout is that causal importance labels hold only as a joint configuration."
(Conclusion)

**Relevance to our paper.** This is a direct check on our featurization's main tool. Our re-gating share φ_Γ gates the
base subcomponents with VPD's Γ on the fine-tune's activations, which mixes base masks with a changed model state. Our
needed set and the least-squares carry-over also lean on Γ labels and on dead components. Angsten finds that labels are
faithful only jointly, and that the ~29k "never needed" components still carry weight that matters once other masks
are raised. So φ_Γ's floor and its saturation (≈0.99 on Qwen, random changes included) could partly be an effect of
this joint-configuration property rather than of reuse. The 0.34-nat cost of a text's own explanation is a noise floor
our readings should be compared against. Not a fine-tune paper. The draft should cite it in the limitations of the VPD
featurization (it is the only independent audit of the decomposition we use).

**For the digest.**
- 4-contradicts: VPD CI labels are faithful only jointly; masks mixed across inputs or models (as in our re-gating) degrade the output, which is a possible confound for φ_Γ and its saturation.
