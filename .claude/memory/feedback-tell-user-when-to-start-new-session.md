---
name: feedback-tell-user-when-to-start-new-session
description: Tell the user in one line when to start a fresh session instead of asking more of this one, and what to hand it
metadata:
  type: feedback
---

On 2026-10-19 the user asked to be told when they should start a new session rather than asking for more from a single session.

**Why:** Quality drops in long sessions, and the user suspects that is part of why the code got sloppy (see [[feedback-better-code-not-more-checks]]).

**How to apply:** End the reply with a one-line suggestion, naming the file the new session should start from, when any of these hold:
- a deliverable is done and the next request is a different task;
- the context has been compacted, or the session has covered more than one experiment;
- substantial new code is about to start after a long run, debug or analysis stretch;
- the next step is reviewing code written in this session.

Put the hand-off in a report or plan file. The full rule is in `.claude/CLAUDE.md`, under "Sessions".
