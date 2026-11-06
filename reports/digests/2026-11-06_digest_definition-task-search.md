# definition-lead digest, 2026-11-06 (evening: task search)

**Definition, task search: moved, but not to a verdict.** We looked for natural tasks on the small model that the
frozen reuse definition could read and that re-gating could also read. One was too rare in ordinary text to build, and
the other failed the definition's first check, so it reads "unreadable". Both failed because of what the small model
carries, not because of the definition's rules. Choosing the next task is the user's call.

## What ran
- **Earlier searches reviewed.** Four earlier task searches never looked for the shape the definition reads, and
  baseline-lead ran none (note in this folder). The user chose the small model first (ask 9), and Qwen only if these
  tasks read.
- **E124 plan** (`reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md`). Fable gave the go-ahead with eight
  changes; the user delegated go-aheads to Fable ("check with fable from now on"). Two tasks were planned: is/are
  agreement and he/she pronoun gender.
- **Agreement dropped before any run.** The form that leaves the model something to learn ("the key of the cabinets
  is") has 155 windows in 16.7M tokens.
- **Pronoun, stage 0** (job 54746, cluster debug, about 2 min, $0, after two CPU jobs that cached 1M more Pile rows,
  54712 and 54730). Every control passed. A probe trained on Pile text reads the name's gender where the pronoun is
  predicted at 0.53, 0.61, 0.73 and 0.87 accuracy by block, against a 0.90 bar and a control near 0.5. By the filed
  stop rule it reads "unreadable", and no fine-tune ran. Fable decided against an amendment: moving the probe to the
  name token would test a different question, chosen after seeing the miss. Report
  `reports/analyses/2026-11-06_analysis_e124-natural-pronoun.md`, notebook `notebooks/E124_natural_tasks.ipynb`.

## Decision for the user
- **Which task next, if any?** The only property the small model has held at 0.90 by a Pile probe at the answer site
  is capitalisation (E112, E116). Options: (a) pause the definition task search, so the definition's readings stay
  "unreadable" or "does not apply" on every task we have; (b) the his/her pair after a name (same property, more
  windows), which likely fails the same probe; (c) let the search use Qwen, where a probe may pass but re-gating has
  no split to replicate (its ratio reads 0.99 or more on every update). Recommended: (a). Log card E124.

## Money and friction
$0 (three cluster jobs). Friction noted: window sizes were planned without counting the natural pattern (601, 604),
and the task file was clobbered by a quoted heredoc and then restored (602).
