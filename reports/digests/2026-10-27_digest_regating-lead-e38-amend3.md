# regating-lead digest, 2026-10-27 evening: E38 Amendment 3, the restore test

**Why I stopped.** The restore test you chose ran and is written up. Its filed rule reads "outside the tree", so the
next step is a new card for you, `decision-e38-matched-change`; nothing else runs on the re-gating line until then.

## What ran

In the fine-tuned model, we put a set's outputs at the trigger back to what they were in the original model and
measured how much of the fine-tune's gain went with them. For the switched-on set it was about a quarter on emoticon
and a fifth on ` which`, and almost nothing on ` we`. The ` http` fine-tune's set, random sets and sets already active
at the trigger took back almost nothing on the word tasks. So the gain runs through the switched-on set there, which
reverses my filed guess on ` which`. It also clears up Amendment 2's worry: the ` http` set only looked involved
because removal took away its existing push.

Two caveats from the independent check. The twenty active comparison sets overlap heavily, so beating all of them is
weaker than it sounds. On emoticon there were too few active pieces to fill a matched set, so the filed rule leaves it
unscored, although it reads "specific" when scored anyway. And on emoticon, putting the active pieces back raises the
target; that is unexplained and provisional.

The first run stopped at its last control: two ways of computing the same removal differed by 0.02 nats of rounding,
against a bar of 0.01 that I had set without measuring the noise. I widened the bar before any main number existed;
the check judged that sound.

## Pending decision

- **`decision-e38-matched-change` (Soon, re-gating line).** Option (a), which I recommend, is one short cluster job at
  no cost: put back the pieces outside the set whose trigger outputs changed most, matched per layer. Option (b)
  accepts the result as provisional and writes it into Q5. Option (c) stops the line on this model or carries the
  test to the Qwen line.

## Money and machines

Three cluster `debug` jobs on an L40 today for Amendments 2 and 3, a few minutes in all; no RunPod pod, $0. Nothing is
left running.

## Files

Report `reports/analyses/2026-10-27_analysis_e38-amend3.md`; check `reports/reviews/2026-10-27_review_e38-amend3-check.md`;
notebook `notebooks/E38_recruitment_test.ipynb`, section "Amendment 3"; plan `reports/plans/2026-10-27_plan_e38-amend3.md`.

## Friction noted

`cpu-run python` needs the venv's path; a control's bar set without a measured noise floor; buffered Slurm logs; the
dev-node hook refusing even short JSON reads, which `jq` handles.
