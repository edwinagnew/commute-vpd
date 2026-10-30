# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead: pending-decision card (Soon): E55 C2 platform -- run VPD on the counting truth set on cluster normal QoS (~50 L40-h, $0, needs the Qwen-only rule relaxed), on an H100 (~$90, over cap), or drop C2 (VPD can at best tie there); rehearsal numbers and options in reports/decisions/2026-10-31_decision_e55-c2-platform.md. E55 card Run line: C2 rehearsal done (14,592 subcomponents 0.44 s/step, 19 GiB; matrix share 0.68-0.76; resume mismatch to fix), waiting on the user.
