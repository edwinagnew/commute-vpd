# Review: "the obstruction is not needed for the behaviour" was too strong

*2026-10-15, coding agent, after the user's objection.* Applies to `reports/2026-10-14_analysis_lift-test.md` ("What this
does and does not say") and `experiments/TRIED.md` row 2.

- **What was measured** (`projected.json`). Dropping the obstruction O(δW) keeps P(o) (0.998 against 0.999) and 85–96% of
  the fine-tunes' effect on Pile. But **O(δW) alone also gives P(o) 0.4–0.9**. So the behaviour is stored redundantly,
  in both parts. That shows the obstruction is *sufficient-ish and dispensable given the rest*, not that it is unneeded.
- **Correct wording.** For the emoticon task, the expressible part alone reproduces the behaviour; the inexpressible part
  also partly carries it.
- **Still open** (user, 2026-10-15). Is the part outside the subcomponents a flaw of the update (e.g. LoRA or the
  optimiser being inherently noisy) or of the decomposition (VPD missing structure the model uses)? Nothing so far
  separates these. Recorded in `backlog.md`.
