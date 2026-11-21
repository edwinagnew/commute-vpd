---
name: feedback-no-restart-with-open-questions
description: whip-lead must not touch restart-ready while a question to the user is unanswered; a restart mid-reply lost the user's answers (2026-11-20)
metadata:
  type: feedback
---
Don't touch ~/commute-data/agents/whip-lead/restart-ready when the reply just sent asks the user questions. Wait for the answer, or for the next scheduled check if none comes.

**Why:** on 2026-11-20 (~19:54 PST) the session restarted while the user was answering four questions. Their answers ("2. yes 3. yes 4. yes") landed in the dead session. The cell consent audit only reads the calling session, so they had to be re-approved (proposals 44-46), and the user called it annoying (friction 960).

**How to apply:** under the wake protocol (b)/(c), check whether the reply ends with a question first; if it does, schedule the fallback wake-up but skip restart-ready. See [[feedback-whip-lead-autonomy]].
