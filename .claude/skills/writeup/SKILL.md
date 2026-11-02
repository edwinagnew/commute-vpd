---
name: writeup
description: Use when writing or revising any results write-up in commute - a notebook in notebooks/, an analysis report in reports/, the chat message that announces a result, or the brief for a subagent that will write one. Turns a run's numbers into an explanation a colleague can follow without the project's history in their head.
---

# Write-up

## The problem this fixes

The user's verdicts on past write-ups, in both this project and feature_wars: "one of the most unreadable things
I've ever seen", "i have no idea what is going on", "your notebook summaries are getting increasingly bad",
"sounds important but is not a real sentence", "the bullets following the graph seem to try and wiggle their way
out of it", "fixing doesn't mean adding tons of waffle. just writing clearly!". Open `notebooks/E09_lora_plateau.ipynb`
cell 8 or the title of `reports/analyses/2026-10-19_analysis_lora-plateau.md` to see the shape of it: a 36-word title; a
"plain-words paragraph" of three 60-word sentences carrying fifteen numbers; a glossary of twelve coined terms
before the first result; "P3 neither, flagged" and "round 5's C2 reopened" in place of what happened; a 1×4
panel figure and a 17-column table where one plot would do; the GPU and a sha256 at the top where the question
belongs. Every earlier rule ("could it do it?", "one question per figure") was followed as a heading and the text
under it stayed unreadable.

The record already exists in `results/*.json`, `predictions.json`, the run log and `TRIED.md`. A write-up's
only job is to explain. The reader is **a sharp colleague who has read `GOALS.md` once, has not seen this
experiment, and reads only this file.** In the user's words: "legible even once I've forgotten all the details
of our approach." After each section they must be able to say what was done, what was seen, and why it matters.

## The three layers

| layer | job | length |
|---|---|---|
| chat message | the answer, the one surprise, where to look, the one decision needed | a few lines |
| notebook `notebooks/E<NN>_<experiment>.ipynb` | the explanation: figures and the story around them | about a thousand words of prose |
| report `reports/analyses/<date>_analysis_<topic>.md` | the record: every number, the scoring, provenance, "Not run" | as long as it needs |

Write the notebook first. The report is the notebook plus the full tables and the scoring. The chat message is
the notebook's answer paragraph cut to the bone. Update an existing notebook rather than adding a new one when
the experiment is a follow-up; headline results only, not every run.

## Notebook structure

The user's definition of a good write-up: "abundantly clear what you were trying to test, what you observed,
what you learned, how compelling the evidence is and what we might look at next." Those are the sections.

1. **Title**: the question, in about a dozen words.
2. **The answer**: one short paragraph, around 80 words, a number or two at most, no symbols, no codes. Written last,
   placed first. If the most important thing in the run is not a filed prediction, it still goes here.
3. **What we were trying to test**: the experiment as a story a reader can picture. What was trained or changed,
   what was varied, what was held fixed, what was measured and how to read it (what 0 and 1 mean, what the
   reference is). For every arm and control, the alternative it rules out: "two fine-tunes to a *different*
   target token, to check whether any update of this size switches on the same subcomponents." Sizes are the
   only numbers here. Where a setup is hard to picture, draw it: a small diagram of the arms beats a paragraph.
4. **What we expected, and why**: each filed prediction in words with the reasoning behind it, so the reader
   feels the surprise or the confirmation when the result comes.
5. **What we observed**: one section per claim, two to four of them. Heading: the question. One figure. One
   paragraph headed **Reading it**: verdict in the first sentence, then what in the figure shows it, then one
   caveat if there is a real one. Say what happened in the system, not the verdict code: "the read vectors
   stayed within ten degrees of where they started", not "P1 pass". If the figure looks like a failure, the
   paragraph says it failed. No softening after a plot.
6. **What we learned**: three to five sentences on what this changes for `GOALS.md`.
7. **How compelling is the evidence**: what the measurement is blind to, the noise floor, seeds, tasks covered.
   "On every task" only when it was every task; otherwise list them.
8. **Surprises**: required, even when "none". A reversed prediction leads this section and opens the chat
   message with "Something unexpected happened that I cannot explain", with what was predicted and what was seen.
   A narrow threshold miss gets one line. Never "refiled as pass".
9. **What next**: two or three items, each with the result that would change what we do.
10. **Scorecard**: the filed predictions as a table: id, claim in words, rule, what was read, verdict. The only
    place codes (P1, G3, C1) appear.
11. **Not run**, then provenance: result paths, GPU, time, cost, hashes. Last, not first.

The report has the same order, with the full tables under the claim sections and the opening paragraph under
the same 80-word rule.

## Terms and equations

- Use VPD's words: subcomponent, causal importance, Γ, CI-masked, Δ, target model. Use `GOALS.md`'s Terms:
  component training, least-squares split, adjoint split, outside share, do-nothing, faithful. Never "pull-back",
  "lift" or "projection" (retired 2026-10-20; `check.py` flags them). Not "gate", "re-gating", "piece", "route": say which subcomponents Γ
  switches on, or which mask multiplies what.
- Define a term in the sentence that first uses it, with its formula in `$...$` and what 0 and 1 mean. If a
  section needs more than four symbols, open it with a one-line-per-symbol list, never a dense paragraph.
- Equations go inline where the concept appears, not in a separate maths section. "The least-squares split is the
  least-norm $(dU, dV)$ with $dU V^\top + U dV^\top \approx \delta W$" beats "the closest first-order answer".
- No notation reuse: a symbol means one thing in the whole notebook.

## Sentence rules

The numbers here are guidelines, not hard limits: a sentence that breaks one and reads well can stay. `check.py` flags
the breaks so they are seen rather than missed; look at each flagged sentence and rewrite the ones that are dense.

- Around 25 words per sentence, one idea each.
- About two numbers per sentence in prose. A sentence that needs many more is a table or a figure.
- No parentheses in prose. If the aside matters it is a sentence; if not, cut it.
- No codes outside the scorecard: P1, G3, e_Δ, kl_other. Say the thing: "the share of the behaviour left in Δ".
- No pointers into history the reader lacks: "round 5's C2", "candidate 3 of the round-5 notebook". Say what it
  was, once, where it is needed.
- Every claim sentence names the quantity, the conditions compared, and how far it generalises. "Re-gating
  depends on the task, not on the update" is not a sentence; "the share of the gain carried by changed CI values
  was 0.55 to 0.60 for every update we tried on this task" is.
- A number without its reference is not a result: next to each value, what it reads if nothing happened and
  what the control reads.
- **Departures from the literature are explicit** (user, 2026-10-30). Wherever a definition, recipe, threshold or
  procedure differs from the source it is credited to, say so at the point of use: what the source says, what we do
  instead, and why. Collect them in a "Departures from the literature" list near the end ("none" if none). A
  departure recorded only in a decision file does not count.

## Figure rules

- One question per figure, in its title; the reader can state the answer from the figure alone. One panel, or
  two with shared axes when the claim is a comparison.
- The claim's comparison on the same axes: the measured thing, its do-nothing reference, its null or floor as a
  labelled band or line, and the positive control. No arm plotted alone.
- Every reference line, band and marker labelled on the plot. Series names in words. Axes labelled with the
  quantity. A legend you can read. Colours that differ when the series differ, and the same colour for the same
  arm across figures.
- Show the whole range, not the subset that flatters the claim. Before writing "Reading it", check the plotted
  points against the sentence you are about to write.
- Keep notebook tables narrow, around five columns; the full table goes to the report. No raw dicts or
  DataFrames printed as output.

## Procedure

1. Before writing a cell, write down: the question, the answer in one sentence, the two to four claims, and for
   each claim the figure that shows it. A claim without a figure gets one or is cut.
2. Write sections 3–11, then the figures, then the "Reading it" paragraphs, then the answer paragraph.
3. Run the checker and rewrite the flagged sentences that are dense:
   `uv run python .claude/skills/writeup/check.py notebooks/E<NN>_<name>.ipynb reports/analyses/<file>.md`
4. Cold-reader pass: read each markdown cell with only the figure above it and ask what was done, what was seen,
   why it matters. Rewrite any cell that fails one.
5. Execute the notebook top to bottom and commit it executed.
6. Queue the research-log update: append a line to `docs/research_log/PENDING.md` naming the write-up and the log
   sections it changes (the experiment's card, the claims table, the sub-question status, the timeline). The
   write-up is done when the line is queued; the log says what it says after the next flush
   (`docs/research_log/README.md` → "Batched updates"), which happens before the session ends.
7. A subagent that writes a notebook gets the "Sentence rules", "Figure rules" and "Terms" sections verbatim in
   its brief. The lead writes the answer paragraph and every "Reading it" paragraph itself: summaries handed to
   subagents drifted into shorthand every time.

## One rewrite, to calibrate

Before, from `notebooks/E09_lora_plateau.ipynb` cell 8: 94 words, 14 numbers, 2 parentheses, 3 history pointers.

> **Could it do it? (faithfulness half)** Yes, at every step. The split state reads 1.26, 1.22 and 1.07 under
> the Pile adversary (stochastic and CI-masked 0.996–1.009), within one to three W₀ floors on this card, where
> round 5's split fine-tune at step 10 read 5.41 on the L40. The do-nothing placements sit at the floor too (1.17,
> 1.09, 1.04; the fine-tune's read 2.18): the LoRA update itself disturbs W₀'s decomposition less than the
> fine-tune's at every size, with a relative update four to eight times larger.

After:

> **Reading it.** Yes: the split LoRA stays faithful at every step we checked. The figure plots the adversarial
> faithfulness ratio of the split state at steps 10, 100 and 1000. The grey band is the unchanged model's noise
> floor, and all three points sit inside it. The dashed line is the full fine-tune's split state at the same
> step, which read five times the floor. Leaving the LoRA update outside the subcomponents altogether also stays
> inside the band. So a LoRA update disturbs the decomposition less than a fine-tune of the same length does,
> even though it moves the weights further.

The second version drops the stochastic and CI-masked numbers, which belong in the report's table, names the
comparison the figure makes, and says what happened to the model instead of which filed line was cleared.
