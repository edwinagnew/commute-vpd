# Decision: the fixed-shift hypothesis becomes "one fixed change per cue", tentatively

User, 2026-10-23, on the pending scope card raised in `reports/2026-10-22_analysis_shift-anatomy-and-many-cues.md`
(E32, E33). Words as in `GOALS.md` → Terms.

**What was decided.** Option (c), accepted tentatively: the working hypothesis is restated as "after a narrow
fine-tune, one fixed change to Γ's importance values per cue, the same for every input with that cue and partly
shared when cues share an answer, carries most of the switching part of the gain". It holds on every task tested,
the copy task and the 32-pair cipher included. Options (a), restrict to tasks the model already does, and (b), keep
the single change and call copying a counterexample, are closed by E32 and E33.

**The user's concern, which the decision carries with it.** "Cue" has no general definition. In every run so far it
is the token at the trigger position that decides the answer, which is a property of our toy tasks. For a real
post-training task such as "get better at calling Python inline" there is no list of cue words, and the number of
cues is undefined, so a count of cues says nothing about such tasks. The re-gating line is at risk of getting lost in
the mechanics of synthetic tasks; what matters is findings that could generalise to tasks people care about.

**What transfers, and what does not.** The form of the finding transfers: the re-weighting of existing subcomponents
is not a single global bias on the importance values, it is organised by the condition the behaviour fires on, and
inputs that call for the same response share most of their change. The number does not transfer: "32 cues" is a fact
about the task.

**What changes.**
- Write-ups word the claim as above and say "cue" is the trigger token in the tasks run so far.
- The anatomy follow-ups in the E32/E33 report ("What next" items 2 and 3: what the per-cue change is made of in the
  MLP subcomponents, and the shared half with one answer) are not run now. They are mechanism of a toy. Deferred to
  `backlog.md`.
- `backlog.md` records that the hypothesis needs a definition that does not depend on a cue token, and the next
  re-gating step becomes the spread-out fine-tune already in the backlog (a style, format or language behaviour with
  no single trigger), which goes through `/design` before anything is filed.
- `decisions.md` gets a row; `GOALS.md` → Terms is reworded; the pending card leaves the research log at the next
  flush (`docs/research_log/PENDING.md`).

**Hand-off.** Start a new session for the spread-out fine-tune: read this report, the backlog entry "Spread-out
fine-tune" and the E32/E33 analysis, then run `/design`.
