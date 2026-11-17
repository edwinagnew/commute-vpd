# How well does the second wording of reuse work, after one night?

*2026-11-10, morning report for the user. Cells `P1.3` and `P1.6` filled, `P1.4` blocked on ask 23. All runs $0 on the
cluster's debug queue. No verdict on any saved fine-tune, and neither wording was changed.*

**The answer.** As a measurement, the second wording works on cases whose answer we know. A search that was not told
the answer found at least nine tenths of every known answer it was set. Its two shares put the four hand-made pairs in
the order of their labels, and it fixes both faults of the first wording. Two things are not shown yet. Its "built"
reading on the XOR follows largely from how that pair was made. And it has not read any ordinary fine-tune, because the
second run stopped at a first-steps check I set too tight.

## What worked

We added one small piece to the 67M model in four ways, each with its label fixed by construction (cell `P1.3`, page
`reports/analyses/2026-11-10_analysis_reuse-variable-first-test.md`). The search reached each known answer above the
0.9 bar. The re-weighting share separated the pair built to be re-weighted from the other three, as filed. The
rewiring share was high on the three pairs that are not built and near zero on the XOR. Where the two wordings differ,
the second matches the construction: the first calls the XOR reuse, and its "built" can apply nowhere, because every
direction of the activations changes the output.

## What it does not yet show

- **The XOR result is partly built in.** The XOR was chosen so that no affine read of the original can see it, and the
  rewiring share is in effect an affine fit. A low share there confirms the construction more than it tests the
  wording. The premise check on both filled cells flagged this. A pair whose "built" label does not follow from linear
  unreadability is still missing, and I do not yet know how to make one.
- **The rewiring fit's refinement failed.** Its gradient steps ended worse than their start at nearly every site, so
  the share is the plain ridge fit. A low share at a late site is a search limit, not absence. Cause not established.
- **"The original acts on it" excludes nothing** on this input set, so only the dimension and the affine class
  separate re-weighted from rewired.
- **No real fine-tune has been read.** That is the second run's job.

## The second run, stopped

The second run (`E139`, cell `P1.4`) fine-tunes all weights of the 67M to imitate each pair, three seeds each, with
cut-offs filed from the first run before it started. Its control passed exactly. Every item then stopped at step 10
on a filed check that the fine-tune must not have done most of its learning yet. On Always on it had moved 0.104 from
the original, against a bar of 0.035, while its distance to the target fell smoothly. The XOR item showed the same
shape. That is fast learning of an easy target, not a step that overshot. The bar was a judgement, never measured, and
I should have sent it back before filing. Changing a check after it fails is your call, so ask 23 proposes replacing
it with an overshoot check, which the XOR and Always on curves both pass, using the last rerun of the cell.

## Your three caveats

1. Built cannot occur on a small closed input set; the XOR is hand-made, and nothing here says whether a model
   pretrained without a skill reads built.
2. A found map gives a lower bound on each share; not finding one is never read as built. The control that the search
   can succeed is the three known answers above.
3. Your last repairs to the second wording are unreviewed. The distance, the dimension 4, the input set and the bars
   were fixed before any reading; the second run's cut-offs came from the first run by the plan's rule.

## Not run

Any ordinary fine-tune (blocked on ask 23). A pair whose "built" does not follow from linear unreadability. A fixed
refinement of the rewiring fit. A model pretrained without a skill. Re-reading the saved copy-against-lookup result,
which waits on your adopting a wording. A second seed of the hand-made pieces.
