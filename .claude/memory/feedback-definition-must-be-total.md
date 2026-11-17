---
name: feedback-definition-must-be-total
description: "The reuse definition must be total (a verdict in principle for every fine-tune); \"no reading\" is a failure, and cards must quote the definition, not paraphrase its measurement"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b85c2744-2d19-41cd-9d7e-7ba479512f7a
  modified: 2026-11-13T14:25:06.032Z
---

The user (2026-11-13) wants the working definition of reused vs new to be **total**: in principle it always says something about any fine-tune. Jain et al.'s "no reading" / "unreadable" was the problem the project deliberately set out to fix. The third wording (soft intervention) brought "no reading" back via its dependence on a hand-found H on M, and the user caught it; I had recommended adopting it without flagging this.

They also liked v3's one-line simplicity ("F reuses V of M if F is M under an intervention on V") but find "intervention" too loose (a complicated enough intervention does anything), and objected that the decision card paraphrased the measurement ("swapping that part's value") instead of the definition.

**Why:** a definition that goes silent on real fine-tunes cannot answer the project's question.
**How to apply:** check every candidate wording for totality against [[project-working-definition-criteria]] before recommending it; on cards, quote the definition verbatim and keep the measurement apart, naming exactly which value is swapped and by what kind of intervention. Related: [[feedback-causal-tracing-definition-rejected]].
