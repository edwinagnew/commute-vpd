# Decision pending: the reuse line's first plan, and whether a lookup counts as reuse (2026-10-29)

Card id `decision-e55-plan`. Drafted by reuse-lead from `reports/plans/2026-10-29_plan_reuse-stage1.md`, Revision 1.
The review it answers is `reports/reviews/2026-10-29_review_e55-plan.md`. Every choice here is yours.

## Card text

**Flag:** Soon. **Label:** Question for you · reuse · new 2026-10-29.
**Run line:** Due to run: the counting-grammar stage, free on the cluster, once its code passes review. The paid
stage waits on item three.

**Heading.** Is the reuse line's first plan right? And should a fine-tune that only looks up a new answer per word
count as reuse?

**What the plan does.** We train a small model on the counting grammar of Jain and colleagues. We choose whether it
already counts one letter, then fine-tune it to count another. For each fine-tune we ask whether the new behaviour
runs through the old count. Two checks answer this without any decomposition. One reads what the model outputs. The
other swaps in the old count from a different input and sees whether the answer follows. Pruning, reverse
fine-tuning, steering and later VPD are then scored on how often they agree. An independent review found that the
first draft's test could not tell the two kinds of fine-tune apart. The revised plan replaces that test.

### Item one: the plan as revised

- **Evidence.** The review raised four blocking points. The old test removed a direction every fine-tune needs. The
  paper predicts two of its checks read the same on every arm. Some labels named the wrong function. And the 67M
  arms called new are reuse under the definition. The first three are fixed in the plan; the fourth is item two.
- **Option A.** Run the counting stages now on the free cluster, under your standing permission for audited GPU
  work. Each threshold comes from the control stage and is filed before any fine-tune.
- **Option B.** Hold everything until you have read the plan.
- **Lead's recommendation.** A. It costs nothing, the controls run first and alone, and no threshold can be chosen
  after seeing a fine-tune.

### Item two: under the paper's definition, the cipher and copy fine-tunes are reuse

- **Observation.** The definition calls a fine-tune reuse when its behaviour is some function g of what the model
  already computes. The cipher outputs a fixed word swap of the predicted next word. So its g is a lookup table on
  top of next-word prediction. Copy is the same with g doing nothing. Read literally, both are reuse. The repo's
  "copy is existing, cipher is new" labels therefore do not map onto the paper's reuse and new.
- **Option A: accept the definition as written.** The cipher counts as reuse. The 67M part of the plan reads
  lowercase only. The earlier copy-against-cipher results stop bearing on reuse. Free.
- **Option B: bound g.** A fine-tune reuses a capability only if g, learnt on the fine-tuning examples, also works on
  held-out values. Lowercasing passes, since it applies to unseen words. Copy passes too. A word-swap
  table fails, since it says nothing about unseen words. Cost: one free cluster job of held-out-key readings, about a
  day of agent time, and a dated change to the reuse-line definition.
- **Option C: another bound.** For example, cap the size of g by the parameters a probe needs to read it.
- **Lead's recommendation.** B. Without a bound, any fine-tune that keys off something the model represents counts as
  reuse. Then the label cannot separate any 67M pair we have. The bound is one testable clause and costs nothing. It
  leaves the counting stage unchanged, because there g adds one or does nothing, and both generalise. It departs from
  an established definition, which the brief asks us to avoid. But the objection applies to the paper's definition
  itself, so we would record the departure openly.

### Item three: go-ahead for the one paid stage, after its rehearsal

- **What it is.** One VPD decomposition of the counting model that barely saw the second letter. Then the route split
  on every labelled fine-tune of that model, scored against the two checks. It departs from "not the goal: re-running
  the decomposition" in `GOALS.md`, so it needs you.
- **Why it is worth it.** The paper's grid gives 16 fine-tuned states of that one model. On those, a coin-flip
  labeller clears the pass bar about 3 times in 100. The first draft's three arms gave it one chance in eight.
- **Cost.** About 10 to 30 dollars of H100 time and one or two days of agent time. A free rehearsal on the cluster and
  the six gates for gated runs come first.
- **Option A.** Approve it now, to run once the rehearsal, two reviews and the non-VPD scores are in.
- **Option B.** Decide later, after the counting stages are read.
- **Lead's recommendation.** B. Whether the two checks hold on the anchors decides whether VPD has anything to be
  scored against. The rehearsal numbers and the counting results will be on this card when we ask again.

## Not run

- Nothing has run and no predictions are filed.
- The held-out-key readings in item two were not costed against the saved 67M states one by one.
- No second reviewer has read the revised plan.
