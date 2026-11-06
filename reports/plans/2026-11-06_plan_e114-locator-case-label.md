# E114: Where does E112's case-label gain run, by the locator?

*mechanism-detection-lead, 2026-11-06. Status: proposed, waiting for the user's reply (the user is around today, so the
plan is shown before any code). $0, cluster `debug` L40, about 15 min of reads. No training.*

## The request, in the Terms' words

The task file asks for a way to isolate mechanisms across the 67M runs so that baseline-lead can measure whether the
reuse definition lines up as expected. Since 2026-11-06 the definition is Jain et al. 2024 as adopted (`decisions.md`
rows 81, 87, 164) and it is frozen (row 165). E112 is the first pair read under it. Its real case-label fine-tune reads
"not through the capability": removing the case readout's direction takes a tenth of the gain, and a stronger removal a
fifth. E114 runs the existing locator (E97's removal at head grain, E100's carry and own reads) on E112's four saved
states. That gives an independent "where the gain runs" column beside definition-lead's verdict. It adds nothing to the
definition.

## Design check

- **Goal.** `GOALS.md`: whether post-training teaches new capabilities or strengthens existing ones. E114 gives the one
  pair read under the adopted definition a second, weight-side reading, so the two can be compared on the same states.
- **Nearest tried.** E97 (removal locates the copy head (2,4) and the lowercase pair) and E100 with E110 (carry tells a
  part the gain runs through from one it only needs). Neither has read a task with Jain's marker-plus-content shape.
- **Patch or step.** A step: the same harness on a new pair. No new method.
- **Decisions.** Rests on rows 81, 87, 164 and 165. Proposes no rule for the verdict. The user's "stronger causal
  ablation" (row 166) is not this: that stays the user's call.
- **What would change what we do next.** (a) The real gain runs through a part the original model already needs, with
  the twin not needing it. Then the locator reads reuse where the definition reads "not through C". That disagreement
  is what baseline-lead measures, and it is worth a card. (b) No part is located, or the twin needs the same part. Then
  the locator agrees with the definition, or cannot read the pair, and E114 closes.
- **Control first, alone.** E97's two planted one-matrix edits (block 3's and block 1's MLP output), placed on E112's
  real-task windows and scaled to the real state's gain. They must read their whole gain at the edited part. The job
  stops on a miss.
- **Numbers and expected values.**
  1. Removal share per head and per MLP on real s0/s1 and twin s0/s1, with W₀'s own fall on the same reads. At W₀ the
     gain is 0 by construction. Lean: a head in blocks 1-2 that moves the previous word to the marker is needed by both
     real and twin, since both must read the previous word.
  2. At the top removal part of each real state: carry and own, as in E100. Lean: carry above E100's 0.021 floor, own
     small. That would be an existing routing head driven differently, like copy, not the unchanged lowercase pair.
  3. E100's sum of single reverts, per state. No lean. It is reported beside.
- **Early guards.** The plant reads exactly. All 24 matrices reverted reads a gain of 0. W₀'s accuracy matches E112's
  0.521 and 0.497.
- **Saved for audit.** Per-state JSON with every part's three reads, and the removal table.

## What is not done
No new fine-tune, no new definition rule, no Jain re-read (definition-lead's).
