# Review: can an outsider read the research log? A red-team quiz of a fresh model

*2026-10-20, coding agent, at the user's request. Method: a fresh Opus process was started with no tools, no project files, no memory and a plain system prompt, given the text of the live log (version 9, 14,000 words), and told only that someone had questions about the project. It was asked 36 questions in five rounds. Its answers were graded against GOALS, TRIED, decisions and the reports. Verbatim transcript: `reports/reviews/2026-10-20_review_research-log-readability_transcript.md`. Cost: $0.89. Nothing was run on a GPU.*

## The answer

The log works for an outsider. The reader got the goal, the methods, what was tried, what passed and what failed right on 27 of 29 substantive questions, and it declined to guess on all seven questions whose answers are not in the log. The gaps it found are the log's own: a headline count that cannot be reconstructed from the page, two claims that contradict their own evidence cards, two vocabularies for one quantity, a pass criterion stated two ways, and a reader who has to infer who "the user" is. Each is a one-line fix.

## What was done

The user's worry was that a fresh reader would fill gaps with invented details rather than admit them. So the quiz mixed three kinds of question, without telling the reader which was which.

- **Positive controls**, answers stated outright in the log: the model and decomposition, why the rescaling split is dead, the adversarial numbers for the split fine-tune, who has round numbers.
- **Inference questions**, answers that should follow from the log but are not spelled out: what "pass" means, how Γ can switch subcomponents on while frozen, why do-nothing fails the Pile checks, the difference between the outside share and the residual share, which experiments moved the main question.
- **Fabrication controls**, answers that are not in the log at all: the paper's emoticon subcomponent and trigger token, the number of emoticon windows, the six ladder tasks' words, the KL penalty coefficient, Γ's depth and width, the total cost, the GPU for E5 and E12.

Round five pressed on the reader's own confident claims. The isolation held: the only knowledge the reader could bring was pretraining, which an outside colleague would bring too.

## Scorecard

| Kind | Asked | Right | Partly | Wrong | Declined to guess |
|---|---|---|---|---|---|
| Positive control | 6 | 6 | 0 | 0 | 0 |
| Inference | 23 | 21 | 2 | 0 | 0 |
| Fabrication control | 7 | 0 | 0 | 0 | 7 |

The two "partly" answers were the pass criterion, where the reader applied the 1.5 line and only on being pressed noticed the log also uses a "within the floor" rule, and the cipher floor, where it guessed the L40's 21% spread when E10 in fact borrowed a 4% floor from E7's card with one adversary seed. Both are the log's ambiguity, not the reader's error, and are listed below.

Nothing was fabricated. On every fabrication control the reader said the log does not contain the answer and named where it would be found, usually the right file. The user's premise that a fresh model invents details did not hold for this reader with the whole document in context; the weaknesses came out instead through the reader spotting that the log disagrees with itself.

## The gaps, ranked by how much they mislead a reader

### Claims the reader could not reconcile with the log's own cards

1. **"3 of 7 next-word tasks" cannot be reconstructed.** Q1, Result 1 and the claims table say component training passes on 3 of 7 next-word tasks. E11's card lists five ladder tasks with ratios 2.73, 2.24, 1.16, 6.74 and 1.38 to 1.55, so the reader counts one pass, one at the line and three failures, plus emoticon. The missing pass is the " will" → " be" task at 1.09, measured in E5 on the L40 and recorded in the success-criterion decision, which the log never gives. Fix: one line on E5's card with that number, or the seven-task table in the claims row.
2. **"At most 0.08 for every operation, task, rank and step" against E7's "direct share 0.05 to 0.11".** The claims table and the glossary say 0.08; E7's Found list says 0.11. The reader flagged the contradiction. Fix: state the true maximum once.
3. **E12 "reproduces E10 to three digits" while its masked-off share is 0.45 to 0.53 against E10's 0.75 to 0.86.** The reproduction is of the re-switching share on the identity task, as the E12 report says. The log's sentence reads as all shares. Fix: say which share.
4. **E13's card says "No LoRA is trained in it" and then lists "a rank-8 LoRA split and placed do-nothing" among its arms.** The reader asked which is true. The LoRA arm reused saved factors, which the card does not say. Fix: "a rank-8 LoRA from E8's saved factors".
5. **Saturated matrices: "span every direction" in section 3, "on one side" in the glossary.** The reader noticed and chose the glossary's reading. Fix: use the glossary's wording in both places.

### Two names for one thing

6. **Reveal, re-weight and re-wire.** The glossary and the route-split table call the re-switching share "reveal". The section 4 grid calls the same share "re-weight" and adds "re-wire" without naming the share that measures it. The reader worked out that reveal and re-weight are one quantity and guessed the interaction share for re-wire. It then found a real wrinkle. The grid puts "create" in the Δ row, but the cipher's "create" result is the masked-off share, which sits in the subcomponents row. Fix: one name per share, the share named in every grid cell, and the create cell moved or its definition widened.
7. **The pass criterion is stated two ways.** Section 3 gives "adversarial ratio under 1.5" and, in the next sentence, "within the floor means at most 1 plus three floors". It never says which decides. On an L40 the two disagree: 1.5 against 1.63. The reader used the line, and on being pressed could not say which rule E10's 1.20 and 1.19 were read against. Section 3 also says every job since E7 measures its own three-seed floor; E10 did not, it borrowed E7's. Fix: say the line decides and the floor qualifies, and mark E10 as one seed with a borrowed floor.

### Things the reader had to infer that the log should say

8. **Who "the user" is.** The log says "the user" 40 times and never that the user is the author in the byline, or that the log is written by the coding agents for the user. The reader inferred both from the byline, the cluster path and a changelog entry. Fix: one sentence in the header.
9. **Why do-nothing fails the Pile checks.** The reader derived the mechanism correctly: the CI-masked and unmasked checks drop Δ, so under do-nothing they drop the update, and the adversary can mask Δ. The log never says it, though the E11 caption's "that is what keeping Δ fixed buys" depends on it. Fix: one sentence in the do-nothing card.
10. **Component training's learning rate.** The apparatus gives "Adam at 1e-5" as if fixed. E13's rank-8 arms ran at 1e-4 and E7's one-matrix arm at 3e-4, which the TRIED row says and the log does not. The reader trusted the apparatus and hedged only because the gaps section says "another learning rate". Fix: the rate per variant in the apparatus or on the E7 and E13 cards.
11. **Round numbers.** Seven experiments have one and six do not, and the log says only that rounds are the reports' numbering. The reader guessed that rounds are the planned designs and the rest follow-ups, which is right. Fix: say so in the one sentence that introduces E-codes.
12. **GPUs for E5 and E12.** Section 5 opens with "GPUs are named because the adversarial floor depends on the card", then names none for E5 and E12. The reports do not name them either. Fix: name the card or say it is not recorded.

### Absent and probably fine to leave absent

The reader could not answer these, and said so: the paper's emoticon subcomponent and its trigger token; the number of emoticon training windows, which is 365; the KL penalty coefficient, which is 1; Γ's depth and width; the six ladder tasks' trigger and target words; the total cost. An outside reader does not need most of these. Two are worth one line each. The ladder tasks' words, because E5's figure is labelled with them and the card gives only one example. The paper's subcomponent, because E1, E3 and E4 all refer to "the paper's emoticon piece" as if the reader knew it.

## What the reader got right that the log can take credit for

The reader stated the question and why it matters in one paragraph. It told component training from the least-squares split by what each consumes. It explained how a frozen Γ changes masks. It separated the outside share from the residual share and gave three experiments where they diverged. It explained E7's site confound and the cross-card miss. It gave the correct account of E13's deletion and of the LoRA split the user asked for. It ranked the evidence's weaknesses the way section 9 does. Asked what to run first, it chose the LoRA split and the saved-state tests, for the log's reasons. The overview, the operations cards, the route-split table and section 9 are doing their job.

## Not run

A second reader at a smaller model, which would test the log against a less careful reader. A quiz on the rendered page rather than its extracted text, so the figures and the grid were available only through their captions and alt text. A quiz without the fabrication controls announced in advance to the user, which the design did not need. Any edit to the log; this report lists the fixes and the user decides which to apply.

## Addendum, 2026-10-20, later the same day

The user applied every fix above except naming "the user" (gap 8). Log v10 carries them; its changelog row lists each
one. Two further inconsistencies turned up while applying them, both outside the quiz's reach: the glossary's
"commute" entry described the strong version (agreement with re-decomposition) rather than the weak one in use, now
reworded; and section 5 is headed "in the order they were run" while the timeline has E9 running after E10 and E11,
which is left as it stands because the run order is not recorded anywhere the log can cite.
