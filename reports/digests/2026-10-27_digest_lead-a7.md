# Lead digest, 2026-10-27 (re-gating line, E36 Amendment 7)

**Why this session finished:** you asked to wrap up after Amendment 7. The re-gating plan's decision tree is done.
What comes next waits on the pending decision "Restate the fixed-shift hypothesis without cues?".

**The answer in plain words.** We asked whether the decomposition's re-gating is just each importance value moving
towards where it has room, between 0 and 1. A rule that predicts each subcomponent's change from its own value alone
recovers 0.18 of the switching at step 300. The rule that uses the 8 positions the original model treats most alike
recovers 0.76. So room to move is not the whole story, and the filed prediction B1 held. An independent check found
no bug, but it narrowed the reading. No rule combined similar text with each value's room to move, so "similar text
gets similar changes, within each value's room" still stands as a rival.

## What ran

- **E36 Amendment 7**, plan `reports/plans/2026-10-26_plan_regating-without-cues.md` → Amendment 9.
  - Predictions were filed before the run, in commit 51b9aa2, and it ran on one H100 for 94 s, costing $0.33.
  - Its gate reproduced Amendment 6 within 0.004 at every step.
  - The equal-width bin rule passed both controls, so B1 could be read. The equal-count bin rule had a higher fitted
    error than the straight-line rule at steps 100 and 300, so it failed that control there.
  - Report: the "Amendment 7" section of `reports/analyses/2026-10-26_analysis_regating-spread.md`.
  - Notebook `notebooks/E36_regating_spread.ipynb`; figure `experiments/regating_spread/figures/amend7.png`.
  - Independent check: `reports/reviews/2026-10-27_review_regating-spread-amend7.md`.
- The research log is republished with the E36 card, a timeline line and the updated decision card:
  https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu.

## Pending decisions for you

- **Restate the fixed-shift hypothesis without cues?** Log card `decision-cuefree`, marked Blocking. The restatement
  is "positions the original model treats alike get alike changes", and it comes with two caveats:
  - the text summary it beats covers only the last four tokens, as Amendment 6 found;
  - a rule that keeps text neighbours within each value's room to move was never run, as the review's finding 3.1 notes.

  One option is to run that combined rule before adopting the restatement. It is not designed or costed.
- **Should the adversarial check be judged against a fixed line, or against leaving the decomposition alone?** Log card
  `decision-pass-anchor`, marked No rush. It is from the other session.

## Money and housekeeping

- This session spent $0.33 of its $40 budget, and no machine is running.
- The pod idle watchdog started correctly on its first pod: runpodctl was present and the pod id and key were set.
- Since `fleet/link-data`, every `results/large` folder is a symlink and reaches the pod as a broken link. The runner
  copied the files in by hard link instead, and the rule is now in `CLAUDE.md`.
- The W&B download of the VPD files works again, at about 45 MB/s.
- The cluster's CPU worker is up again, so predictions were filed through `fleet/cpu-run`. Pyright still cannot install
  on it.

## Not run

- The combined rule, which keeps text neighbours within each value's room.
- A per-value rule with an exact-zero bin and fine bins above it. For at least half the subcomponents, 15/16 or more
  of the fitted positions sit at exactly 0.
- An order-keeping summary of the whole preceding text.
