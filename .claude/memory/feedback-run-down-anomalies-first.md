---
name: feedback-run-down-anomalies-first
description: "An unexplained number in the training or apparatus is run down before anything is read on top of it; compare both sides of every precision or forward pair; user, 2026-11-02, after the bf16 rounding was found only because they asked"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 487fd20f-1079-4d4f-ad53-122775a29d79
  modified: 2026-11-02T12:00:40.323Z
---

On 2026-11-02 an audit the user asked for found that E40/E45 fine-tunes ran under bf16 autocast, so training could not see its first 5-10 steps, while all readings used full-precision weights. User: "a pretty catastrophic error that you dont seem to have even thought to look into until now... these qwen runs are slow and expensive and i really dont want to be constantly finding stupid errors like this. why wasnt this caught earlier?"

**Why:** the signs were on file: the same rounding was measured and fixed for teacher training on 2026-10-28 and not carried to the fine-tune; training loss 0.055 against held-out 0.44 at one state sat in E45's files; the report listed the rise as "not run" and the handoff called it "unexplained, user's call", while work carried on to readings built on those states. Controls passed because each checked one side only.

**How to apply:** an unexplained number about the training or apparatus blocks the readings built on it, and the lead runs it down ($0 check) rather than listing it or handing it to the user. A fix found in one pipeline is checked in its siblings the same day. Where two versions of a thing exist (training and reading forward, master and used weights), compare them at the states used, not only at W0. Read generated answers against any "learned" rule. A control must be able to fail on the suspected error. Related: [[feedback-measure-step-size-before-scored-run]], [[feedback-check-metric-power-before-claiming]], [[feedback-verify-before-announcing-a-cause]].
