---
name: feedback-decision-cards-cold-reader
description: "A decision card or ask must let a reader who has read no report decide: no plan labels in the options, each option says what runs, what it tells you, what happens next"
metadata:
  type: feedback
---

On 2026-11-02 the user said of the Qwen decision card: "as usual, the pending decision ... is unreadable. each option presented me to is incomphrensible. im supposed to choose between Q3, stage D, anchor-1 without knowing what any of those are???", and that the decision report was little better.

**Why:** the user reads the log and chat only ([[feedback-user-reads-log-and-chat-only]]), so labels from plans mean nothing to them, and appending dated updates left six options and two conflicting recommendations.

**How to apply:** rewrite the card rather than append to it. Name things by what they are (the template fine-tune, not anchor 1), with labels only in one final "Names in the reports" line. Each option gets what runs, cost, what it would tell you, what happens next and its limits; end on one recommendation. Have a fresh Sonnet agent read only the card and say what each option would do before publishing. The `fleet/ask` line and the decision report follow the same rule. Rule in `docs/research_log/README.md` → Pending decisions. Related: [[feedback-define-terms-when-reporting]], [[feedback-no-ambiguous-shorthand]].
