---
name: concurrent-sessions-share-the-tree
description: "Several Claude sessions work in the same commute checkout at once (seen 2026-10-21); the research log, PENDING.md, TRIED.md and src/ change under you mid-session"
metadata:
  node_type: memory
  type: project
  originSessionId: 88d6d864-eb65-4d23-b907-644c2599837e
  modified: 2026-10-21T14:13:21.444Z
---

On 2026-10-21 at least two sessions worked in `/home/edwin/work/mats/commute` at the same time (one on E17/E18, another on
the SGD grid and the step split, E19). The other session edited `src/commute/component_training.py`, appended to
`docs/research_log/PENDING.md`, and flushed the whole queue into the log (v21) while this session was preparing its own
flush, leaving the lines that arrived during its flush in the queue with a note.

**Why:** a flush or a TRIED/README edit built from a stale read of the local file silently overwrites the other
session's work, and the research log's "read the live artifact first" rule covers the published copy, not the local file.

**How to apply:** right before editing a shared file (the files in `docs/research_log/parts/`, `PENDING.md`, `experiments/TRIED.md`,
`README.md`, `backlog.md`), re-read it (mtime and the last changelog or queue line) and apply edits as exact-string
replacements that assert they matched once; leave the other session's queue lines alone unless its work is finished;
when a publish is refused because the live version moved, use the diff the tool returns to check whether the
differences are your own edits before resending. Related: [[research-log-artifact]],
[[update-the-research-log-after-every-experiment]].
