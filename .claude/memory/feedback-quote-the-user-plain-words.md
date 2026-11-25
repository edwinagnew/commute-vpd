---
name: feedback-quote-the-user-plain-words
description: whip-lead misquoted the user 11-21 to 11-23 (claim 1 recast, invented "RL exception" and "beat the baselines") and wrote jargon to them; relay the user verbatim, explain every code in plain words
metadata:
  type: feedback
---

Two mistakes the user wants remembered (user 2026-11-24: "make sure whip-lead is aware of these mistakes").

1. **Misquoting the user.** 11-21 to 11-23, whip-lead recast claim 1 (user 11-20: "i think 1 is essential and should include a mention about mechanistic evidence, as well as featurization") as a finding. That spread into the results picture, the abstract and writeup-lead's handoff, and cost two days of drafting against a wrong claim 1. It also credited the user with an "RL exception dropped" and put a "beat the baselines" clause in claim 2 that the user never saw. Task files also prefixed whip-lead's own approvals with "user, in chat:". An independent audit found it (~/commute-data/agents/whip-lead/attribution-audit-2026-11-23/report.md); the user was "genuinely shocked".
2. **Jargon to the user.** whip-lead called P1.61 "the gain control" and wrote cell codes and coined terms in a claim proposal. User: "i dont know what the gain control means or frankly anything thats going on in your message there".

**Why:** a paraphrase relayed as the user's words becomes a decision in every other lead's work; jargon makes the user unable to check it.
**How to apply:** when relaying the user, quote them word for word with the date, or say plainly that it is your own reading; label your own approvals "whip-lead approved", never "user". In chat, describe every cell code, E-number or nickname in plain words, or leave it out. See [[feedback-define-terms-when-reporting]].
