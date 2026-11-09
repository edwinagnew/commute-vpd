# P1.2: the copy and lookup pair read under the proposed reuse definition

*definition-lead, 2026-11-09. Cell P1.2, due 11-10. No new run: every number comes from a saved file named below. The
wording read under is the proposal in `reports/plans/2026-11-09_plan_definition-page.md`, which the user has not adopted,
so each reading here is provisional and is not a verdict until a dated `decisions.md` row adopts a wording.*

We re-read the saved copy fine-tunes and their lookup twins under the proposed wording, without running anything. Two
of the three copy seeds read as reuse, the third falls just under the bar, and all three lookups read as not reuse. For
`GOALS.md` this is one pair read at one head, so it shows the wording can separate a pair and does not yet give the
label the other lines need.

## The answer

Taught to copy at a new marker, seeds 0 and 1 read "reuses copying", and seed 2 reads "not through copying": switching
off the old copying head takes 0.471 of its gain, under the bar of one half. The three lookup fine-tunes, on the same
windows, lose nothing and read "not through copying". One condition is open. The plan says the part must be found on
ordinary text, and this head was found on repeated random tokens; on that strict reading all six rows are unreadable.

## The reading, seed by seed

The proposed wording names a capability and a part of the original model, then asks five things. "Share" is how much of
the fine-tune's gain in log-odds goes when the part is switched off in both the original and the fine-tuned model.

| arm, seed | Learned: right on values never trained (bar 0.90) | Share lost with the part off (bar: at least 0.5 real, at most 0.2 twin) | Median single other head's share (bar: under 0.15) | Reading |
|---|---|---|---|---|
| copy, 0 | 0.995 | 0.563 | 0.011 | reuses copying |
| copy, 1 | 0.995 | 0.545 | 0.012 | reuses copying |
| **copy, 2** | 1.000 | **0.471** | 0.012 | **not through copying: under the bar by about two row standard deviations** |
| lookup, 0 | no answer defined for an unseen key | 0.000 | 0.052 | not through copying |
| lookup, 1 | no answer defined for an unseen key | −0.004 | 0.038 | not through copying |
| lookup, 2 | no answer defined for an unseen key | −0.003 | 0.038 | not through copying |

The original model, which all six rows share, has the capability through the part as the wording asks. On a repeated
random sequence it copies with top-1 accuracy 0.825, and with head (2,4) replaced by its mean over ordinary text that
falls to 0.010. The five randomly drawn single heads leave 0.013, 0.707, 0.790, 0.830 and 0.928, so the median draw
leaves 0.790, above the filed 0.7. The drawn head at 0.013 is (1,1), which the E62 report reads as the other head of
the same copying chain.

On the windows read for carrying, the original model gets 0.250, so the fine-tunes' 0.995 to 1.000 is learned.

## What this does and does not say

- **Seed 2 is reported as read, not rescued.** The bar stays at the filed one half. Its row noise is 0.016, so seed 2
  sits about 1.8 standard deviations under it, and the three seeds differ by more than the noise.
- **The bar has no reference of its own.** The E62 report says so: the one half came from an earlier convention, and no
  share of the original model's own copying was read on this scale. The proposal keeps one half until a planted
  control sets a reference. Any change to it would be the user's.
- **The lookup is not a shuffled-label twin.** It answers a fixed word per key, chosen so the word never appears in that
  key's windows. It rules out a part that any fine-tune at this marker needs. It is the nearest arm we have, and the
  run cells proposed in the page would build the real twin.
- **Keeping only the part was read beside, once.** An earlier run (E97, not part of E62) replaced every connection
  outside the original model's copying circuit in each fine-tuned model. The copy models kept 0.98 of their gap and
  the lookups 0.25 to 0.31, and that report says the separation is mostly by construction. It carries no verdict here.

## How compelling is the evidence

Three seeds per arm, 200 read windows, one marker. The part is one head, chosen in E62 by its prefix-matching score
above a filed bar. Adding head (1,1) was never read jointly, so whether a two-head part clears the bar on seed 2 is not
known from saved data. The share is a difference of gains with the head off in both models, and the original model's
own level with the head off was not saved, so how far the behaviour itself falls is not known. The measure is blind to
any route the fine-tune adds around the head.

## Where each number is from

- Shares, their standard deviations, medians: `experiments/induction_cue/results/run.json`, `.states.<arm_seed>.share`,
  `.share_sd`, `.random_median`, sha256 aa242f4475be6463739fd8a7a1a70de9bc401d5d8c7a8c6644bea5fd3a606795.
- The original model's copying and the heads: same file, `.stage_a` (`top1_base`, `top1_H`, `top1_random`, `H`).
- Carrying: `experiments/definition_table/results/stage1.json`, `.states.e62_copy_s<seed>.P1` and `.w0.e62_copy.P1`,
  200 windows, read on the fine-tunes' saved states by E116.
- The two-head reading of (1,1), the 0.5 bar's history and E97's cross-faithfulness:
  `reports/analyses/2026-11-02_analysis_e62-copying-heads.md`, `reports/analyses/2026-11-05_analysis_e97-locator.md` line 126.

## Not run

- Anything new: no fine-tune, no state loaded, no job (the cell's terms).
- A part found on ordinary text: the head came from repeated random tokens, and no ordinary-text search for it exists.
- A shuffled-label twin of the copy task.
- The two heads (1,1) and (2,4) switched off together.
- The original model's level with the head off, so the behaviour's own fall is unknown.
- Whether seed 2's reading holds at more training steps: only step 30 was saved.
- An independent check of this reading.
