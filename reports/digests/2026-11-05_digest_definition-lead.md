# Definition-lead digest, 2026-11-05

**Definition (feeds deliverable 3), day 3 of 9: finished because the only work left waits on you.** Under the refined rules E99 tested, only the copying task on the 67M model carries a verdict. Do you adopt the rule changes? Answer: [the log card](https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu#decision-adopt-refined-rules) (ask 1). Recommended: yes, all six.

## What ran
E99 refined and checked our new-against-existing definition on the old 67M tasks. Three debug jobs on the MATS cluster (52787, 52807, 52895), $0, about 30 minutes of GPU in all. Report `reports/analyses/2026-11-05_analysis_e99-definition-verify.md`, notebook `notebooks/E99_definition_verify.ipynb`, log card E99 (v243). Provisional; no independent check.

## What it found
- On the copying task the rule reads as built. A fine-tune that may change only the two copying heads puts all its gain there. One that is kept off them puts none there, inside its random band.
- On the lowercase heads it has no working check. A fine-tune that may change only those heads reaches about two thirds of the normal gain, so lowercase stays "does not apply".
- The class map and the token-identity tasks fail the new admissibility test. Removing the tested capability also costs a hand-built reader of the input token more than its band. That gives "no admissible capability", so removing value directions has no admissible site on this model.
- The user's batch decision of 2026-11-05 holds unchanged.

## Deviations, recorded in the report
- Amendment 1 (my call): the input-dependence clause was coded as "cased and lowercased rows", but the plan names cased rows only. It was re-filed as the plan words it, and the original miss stands.
- Two filed paired-test calls missed in stage 4: one firm call, one lean. No verdict reads either test.
- The code writer had to make seven choices the plan left open. They are listed in `results/predictions_stage4.json` and in the report.

## Pending decisions
- Ask 1, adopt rules 1-6: [card](https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu#decision-adopt-refined-rules).
- FYI: the deck note (decisions.md row 121) is not done. No link to the deck exists in the repo, so it needs the link or your own edit.

## Money, jobs, friction
$0. No pods, no live jobs. Friction 523, 529 (cpu-run "nodes busy"), 535 (the log publish needed three identical sends).
