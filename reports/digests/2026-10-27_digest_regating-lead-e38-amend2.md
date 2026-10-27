# regating-lead digest, 2026-10-27 evening: E38 Amendment 2

**Why I stopped.** The amendment you approved ran and is written up. The next step is a new choice that only you can
make, the log's Blocking card `decision-e38-specific`, so nothing else on the re-gating line runs until then.

## What ran

We removed the switched-on set, the subcomponents Γ turns on at the trigger after a narrow fine-tune, at the
trigger position only. We removed it from both the fine-tuned and the original weights. The fine-tuned model loses
more of its gain than the original model does: about a quarter of the gain on emoticon and an eighth to a fifth on
` which`. That beats all twenty random sets, as filed. On ` we` it is about zero.

The controls undercut that reading. The set the ` http` fine-tune switches on also loses a tenth more of the gain in
the fine-tuned model, and on one ` which` seed as much as the switched-on set does. The original model's most active
pieces at the trigger lose far more. So "the fine-tune works through this set" is not yet told apart from "the
fine-tune amplifies whatever is active at the trigger". Separately, most of the set's pieces are followed by the
target in ordinary text far more often than chance, against a few percent of random pieces.

An independent check recomputed every verdict from the raw numbers and found no code fault; the reading above is
its correction, folded into the report.

## Pending decision

- **`decision-e38-specific` (Blocking, re-gating line).** (a) A restore test on the saved fine-tunes, one short
  cluster job at no cost: put the set's trigger outputs back to their original values and compare with matched
  active sets. (b) Accept that the effect is not shown to be specific and stop the line on this model. (c) Carry the
  question to the Qwen line. I recommend (a).

## Money and machines

Two cluster `debug` jobs on an L40 (38805, 38809), about four minutes in all; no RunPod pod, $0. No jobs or pods left
running.

## Files

Report `reports/analyses/2026-10-27_analysis_e38-amend2.md`; check `reports/reviews/2026-10-27_review_e38-amend2-check.md`;
notebook `notebooks/E38_recruitment_test.ipynb` (Amendment 2 section); plan `reports/plans/2026-10-27_plan_e38-amend2.md`;
log card `decision-e38-specific`.
