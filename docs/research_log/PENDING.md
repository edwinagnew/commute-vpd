# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 definition-lead: overview box existing-new, "The definition" subsection: replace the "Level" paragraph (one alignment from the architecture to a high-level model) with level as a declared decomposition, per theory/2026-11-18_level-as-decomposition.md Part A (adopted by the user 2026-11-18): a finer causal model of each network whose parts merge back into it exactly, a remainder part where the featuriser is lossy; cases: neurons/heads/MLP layers/blocks, PCA/DAS, SAE/transcoder latents + remainder, VPD components + remainder (on F this declares a split of δW); the same decomposition read on M and F; declared or fitted on M only. Keep the definition sentence. Mark open (Part B, not decided): ε vs the remainder share of K; §5 framing of re-gating; name "decomposition" vs "level" (text keeps "level"). Changelog line; decisions.md row "A level is a decomposition, not an alignment".
