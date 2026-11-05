# Is "new vs reuse" settled, and what would measuring it on the old tasks take?

*Baseline-lead, 2026-11-05, for the user's task of that date. No run. Inventory of tasks:
`~/commute-data/scratch/baseline-lead/inventory/tasks.md` (16 tasks on the 67M, file and line for every label).*

We checked whether the project has one definition of new against reused capability that a measurement can apply. It does not: the words are agreed, but what counts as new, which causal test decides, and which capabilities count were never fixed. Every old 67M task but one still carries its label by assumption, so the next step is four choices from the user, then a free cluster run.

## The answer

Not settled. The definition is agreed in outline and has failed or gone unread every time it was applied. On the small counting model, where the answer is known by construction, its causal clause filed "cannot serve" (E70). On the 67M, the two readouts came back unreadable (E59, E64), and pruning revival and reverse fine-tuning failed their positive control (E55 C3 ladder).

## What is decided

| Piece | Source |
|---|---|
| Possession: a linear readout at some layer, trained on pretraining data, equals f_C on the inputs X_C that share a selector | Jain et al. 2024 Def 1; `decisions.md` row 81 |
| Reuse: the post-trained map is g∘f_C. Strongly relevant if right on the whole fine-tuning distribution, weakly if only on its sample | Jain Def 2; row 81 |
| Reuse also needs a causal claim: ablating or patching breaks the behaviour, "and/or" pruning or reverse fine-tuning revives the old behaviour faster than scratch | user, row 81 |
| Elicitation is mechanistic: the fine-tuned behaviour runs through a mechanism W₀ already has | user, row 112 |
| Held-out carrying is evidence beside the label, not the label | user, row 114 |
| A new circuit built from old parts can read as reuse; every claim states this limit | user, 2026-10-29 |

## What was never decided

1. **What "new" means.** Jain defines possession and relevance. It does not define "new". Reuse is relative to one named C, so "new" can only mean "no listed C carries it".
2. **Which causal test decides.** Row 81 says "and/or". They disagree in practice. On the counting model, the ridge readout's direction reads the count perfectly and removing it does nothing. The covariance direction costs the reusing task .43, .78, .88 of its gain at blocks 3-5 and the new task .16, .34, .49, while the largest direction with no covariance costs nothing. A reversal at block 6, where all answers share the head, filed "cannot serve". On the 67M, pruning and reverse fine-tuning failed their control.
3. **Which capabilities count.** Any key → value map reads the key, so "reads token identity" makes every map reuse. Row 87's held-out rule was the guard against this; row 114 demoted it and left the risk open. The 2026-10-29 audit found the same thing for Def 1: a probe reads any 32-token table from the embedding.
4. **Constant targets.** On emoticon and the ladder, the target does not depend on the input, so a bias satisfies Def 2 and f_C has no content.
5. **Possession threshold.** Row 81 says "filed before any run". E59 used 24 of 32 keys on a 50K-vocabulary readout, E64 used 3/4 of W₀'s own set, E70 used none.
6. **Where C is read.** Jain's X_C is one domain. Our key tasks use a zero-frequency marker, so W₀ never saw those inputs. Copy at the marker is not something W₀ does (baseline ≤ 0.002).

## Proposal, for the user to change

A state reuses a named capability C when all of these hold. Each is a stage that runs first on W₀ and stops at a failure.

1. **State C** with labels from a text rule, never from VPD, and with C not readable from the current token alone.
2. **Possession.** On C's own natural inputs in Pile, on W₀: a low-capacity readout per block beats its shuffled-label twin on held-out rows, and reaches at least 3/4 of W₀'s own output accuracy (E64's relative form).
3. **Function test (sufficient).** A small table from W₀'s reading to the state's answer, fitted on half the rows, predicts the other half above floor. E70's version split the pair: 1.0 against .39.
4. **Removal (necessary, the label).** Remove the directions where f_C moves the residual, at a block before the last. The score is the share of the state's gain lost net of W₀'s drop, E70's score.
   - *Control, first and alone:* the removal costs W₀ its own C above the band of random directions.
   - *Size control:* the largest direction with no covariance costs less.
   - *Comparison:* the share exceeds a same-windows partner with no C, at matched update size.

**Verdict:** "reuses C" if 1-4 hold; "not through C" if the controls pass and the share does not beat the partner; "unreadable" otherwise. Never an unqualified "new". Pruning, reverse fine-tuning, held-out carrying and strong/weak relevance are reported beside it.

**Departure to state openly.** Excluding the last block is chosen after seeing E70's block-6 reversal. E70's filed "cannot serve" stands. Rereading E70 under the new rule would be a check on known data, not a blind validation. Reading C on natural inputs and testing on marker inputs departs from Jain's single domain.

## What it would read on the old tasks

| Task | Label we gave | Candidate C | States | Likely readability |
|---|---|---|---|---|
| Lowercase | "already writes" | context case sets next word's case | saved | best fit |
| Token ciphers | new | none; partner for lowercase | saved | partner |
| E62 copy / lookup | causal, unsettled (.47-.56 vs 0) | in-context copying; head (2,4) as a second removal | saved | good |
| Copy ident32, ident32b, 24/96-key copy | "strengthens" | "repeat the current token" fails item 1; in-context copying is the only candidate | saved | may lose its label |
| Random maps rand32, rand32b | "nothing in the model does this" | none; partner | saved | partner |
| Elicit, rank-1 | "half-knows", "old by construction" | W₀'s own next-word guess at the key | saved | hard: E59, E64 unreadable |
| Emoticon | "an existing behaviour" | emoticon position, text rule | saved | needs a text rule and a partner rebuilt |
| Ladder (5 of 6) | rungs of "already does" | W₀'s prediction of the target word | saved | constant target, see question 4 |
| Many cues, class map and twin, prior cipher, shuffled elicit | mixed | as above | rebuild, minutes each | second pass |
| Code text | learned nothing | none | saved | excluded, a null |
| Narrow lowercase | stopped at W₀ | none | never trained | excluded |

"Likely readability" is my expectation from the earlier reads, not a measurement.

**Cost.** All of it fits MATS `debug` jobs at $0, under the gate. An E-code is claimed and predictions filed only after the choices below.

## The questions (asked in chat and with `fleet/ask`)

1. What "new" means.
2. Which causal test is the label.
3. Whether token identity counts as a capability.
4. What to do with constant-target tasks.

Item 5 and item 6 take the proposal's defaults unless the user says otherwise.

## Not run

Nothing was measured for this report. E70's rule was not reread under the proposal. No states were loaded. The inventory's rebuild times are quoted from `TRIED.md`.

## After the user's answers (2026-11-05)

**Decided.** "New" is relative to a named capability: the verdicts are "reuses C", "not through C" and "unreadable". Direction removal is the label, and the other causal tests are reported beside it.

**Open, the user's words.** Token identity: "the definition should be strong enough to have a clear answer for these edge cases". Constant targets: lean to input-dependent versions; "if our definition cannot be applied to cases where it would be useful then we need to adapt it".

**Proposed answer to both, not yet confirmed: which capabilities are admissible.** A capability C is admissible for a task when C does some of the task's work, judged from the task's labels and W₀'s readings, with no fine-tuned state. Fit a small g from f_C to the answers on some inputs. On held-out inputs, g must beat the best answer that ignores f_C.

| Case | Expected | Why |
|---|---|---|
| Random map, C = token identity | not admissible | the table on some keys says nothing about the others; the cipher reads the key, and the key does none of the answer's work |
| Copy, C = token identity | admissible | g is "say the token read" and carries to unseen keys; the claim is reuse of token reading, not of a copying circuit |
| Constant target, any C | not admissible | a constant does as well as any g; the definition asks for a task where the input matters |
| Rank-1 map, C = W₀'s guess | admissible | the answer is the guess |
| Built-to-elicit map, C = W₀'s guess | probably not | the answer is one of W₀'s ranks 2-20, and which rank differs per key |
| Class map, C = embedding class | admissible | four words, and g carries to unseen keys |

This is row 87's held-out idea moved from a state's behaviour to a property of the task. Row 114 demoted the reading of a state's behaviour; the label here is still removal. The user should judge whether this respects row 114.

**Constant-target tasks.** Agreed with the user's lean, with a bridge: keep each old trigger and target word, and add inputs at the trigger where the answer differs. The old states stay "definition does not apply". The rank-1 map is already the input-dependent form of "strengthen what W₀ says". Whether the new tasks' results carry to the old findings is tested by reading the old readings (route split) on both.

**Test the definition before using it.** A table of edge cases with verdicts filed in advance, run before any old task gets a label:

- Task P and task N on the counting model: reuses, not through the count of a.
- The per-key hand write of the copy table: same behaviour, new mechanism by construction, so it must not read as reuse.
- Trained copy against that hand write, at matched size.
- A random map: no admissible C.
- A constant target: not admissible.

A miss changes the definition, dated, before the old tasks are read. Only the two amendments per E-code limit how often.
