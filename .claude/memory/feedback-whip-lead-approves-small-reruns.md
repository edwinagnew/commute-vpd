---
name: feedback-whip-lead-approves-small-reruns
description: "whip-lead may approve/reopen small cheap reruns (amended controls, $0 cluster runs) itself, without asking the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: db10179f-7398-49f1-b4a8-db985dc150db
  modified: 2026-11-22T21:58:40.252Z
---

User 2026-11-22 ~13:30, after I asked them to type `fleet/cell reopen P2.59` for a $0, ~15 min amended-control rerun of the activation difference lens: "you should be allowed to aporve small reruns yourself". The cell reopen then passed the consent audit on that message.

**Why:** typing cell commands is a nuisance for the user (friction 1001); small reruns are cheap and reversible, and the deadline is tight ([[project-draft-deadline-2026-11-29]], [[feedback-never-hold-runs-for-user-overnight]]).

**How to apply:** for a small rerun (an amended control, a cell's remaining runs, $0 cluster or a few dollars) run `fleet/cell reopen/change/add` myself, quoting this approval, and FYI the user afterwards. New directions, gated runs and definition changes still go to the user. Extends [[feedback-whip-lead-approves-planned-gated-runs]].
