---
name: feedback-explain-tree-branches-before-running
description: "A run taken from a plan's decision-tree branch still gets a plain-words \"what and why\" in chat before it starts; the user doesn't know branches buried in plans"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 30212880-c75c-4700-93d8-ff560f705c77
  modified: 2026-10-27T19:58:29.022Z
---

On 2026-10-27 the lead ran the "prefix test" on Qwen3-1.7B-Base because the fresh-numbers plan's decision tree named it for an ambiguous verdict. The user said: "i dont know what the prefix test is, nor why you ran it". Approval of a plan did not mean the user knew its branches.

**Why:** the user reads the log and chat only ([[feedback-user-reads-log-and-chat-only]]), so a plan's later branches are effectively unseen, even in a confirmed plan.

**How to apply:** before starting any run that a decision tree triggers, write two or three plain sentences in chat: what the test does, the question it answers, and why that matters for the goal. Do this even when no sign-off is needed. A new method name (e.g. "prefix test") is always defined in that message.
