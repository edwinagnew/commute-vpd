---
name: feedback-flag-qualitative-surprises-loudly
description: "When a result is qualitatively opposite to a filed prediction, say so in very clear terms and get verbal confirmation before building on it; narrowly missed thresholds only need acknowledgement"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 19b29868-8384-40e7-8a71-c1a1da41fc33
  modified: 2026-10-08T13:35:43.776Z
---

On 2026-10-08 the user was frustrated that the twins result (rare, more-important twin favoured, the opposite of
both filed directions) had been buried in a notebook table and a backlog paragraph instead of being announced.

**Why:** the user reads the chat, not the scoring tables. A qualitative reversal is a sign that something is going
on that the theory does not cover, and they want to decide what to do about it before further experiments are
stacked on top. Threshold misses are noise about tolerances and do not need the same treatment.

**How to apply:**
- Two classes of failed prediction. (1) Narrow threshold miss (e.g. C2, C3, M2, M3 on 2026-10-08): acknowledge
  in one line, keep humble, move on. (2) Qualitative reversal or third-direction outcome (P4 round one: MLP keeps
  fewer features, not more; P2 paper_scale: truncated water-filling, not top-N; R3 twins: importance favoured):
  open the message with "Something unexpected happened that I cannot explain: ..." in plain words, state what
  was predicted and what was seen, say there is no explanation yet, and ask the user how to proceed before
  designing follow-ups.
- Never hide a reversal inside a plan or backlog entry.
- Related: [[feedback-plain-language-and-name-the-plot]], [[feedback-prereg-scoring-rigor]].

**2026-10-19 follow-up.** Task ladder: both of the round's filed predictions (P1 coverage, P2 rarity irrelevant) failed and
the reply only scored them. The user: "are we just going to brush past the fact that both your predictions failed?"
**How to apply:** when the round's main hypothesis fails, lead with that, say what the hypothesis was, which part is
now refuted, whether the failure was foreseeable from earlier results (and why it was not withdrawn), and what it changes.
