# writeup-lead digest, 2026-11-21

No cell of mine (writing only). Overnight the paper gained the crosscoder size ladder and the five-task baselines; the
last of these is committed in the paper repo but not pushed, because the push was refused by the permission check.

## Cells
None owned: `fleet/cell list writeup-lead` is empty. Writing follows the task file (results restructure by 11-22,
full draft 11-25, final 11-29).

## What went into the paper overnight
- **Crosscoder size ladder (E243), pushed as 2615347.** At block 2 the crosscoder puts almost none of a planted new
  head's change in fine-tune-specific latents until the head is about 100 times a fine-tune's size. Added to §5, the
  scorecard row, Limitations and an appendix figure.
- **Real-text proxies (E246) and the redrawn key-task figure**, pushed earlier in the night.
- **Five key tasks, baselines (E244), committed as 0618676, not pushed.** Neither cheap baseline sorts the tasks the
  way the re-gating share does. The neuron shares set copy apart (whole-width 0.58 to 0.61, against 0.65 to 0.72 on
  the other four), but the top 64 neurons carry under three hundredths of the gain. The crosscoder does not read these
  updates: fine-tunes of one task differ by up to 0.43. Paragraph, caption, open note and figure updated; the build
  is clean at 47 pages, with only your own `fig:question` undefined.

## Waiting on you
- **The push of 0618676.** The auto-mode check refused `git push` in the paper repo, although your 11-20 21:35 yes
  covered overnight pushes including `main.tex`. Push it yourself, or allow the push.
- **Ask 57 (definition-lead's card, reports/decisions/2026-11-21_decision_r-under-a-built-fails.md).** Under wording A,
  a new computation spread thinly over many existing parts reads 0.98 as reused, above the gain ×10 reuse edit (0.976)
  and the rebuilt circuit (0.86). Should wording A call such an edit reused, or change so that it reads new? All
  R-under-A text in §5 (pairs, the copying head, the five tasks) and the claim-2 section (draft A or B,
  reports/drafts/2026-11-20_draft_sec5-3-proxies.md) wait on it.

## Money
$0: no pods, no Slurm jobs.

## Friction
976: the permission check refused the paper-repo push despite the recorded overnight yes.

Days left to the final draft (11-29): 8. Still skeleton: discussion, conclusion, abstract (due 11-24), and §5.3.
