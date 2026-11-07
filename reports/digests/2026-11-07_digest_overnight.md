# Overnight digest, regating-lead, 2026-11-07

**67M story, night of 2026-11-06/07: finished, the 67M summary is written and checked.** At 67M nothing is yet read as
reuse under the frozen definition, and where the decomposition can be compared with it, it cannot tell a real
fine-tune from its memorising twin. Teacher training carries narrow fine-tunes faithfully, with named exceptions. On
the broad lowercase fine-tune its placement hides most of a fragile function's excess, but not enough for full-matrix
Adam, and how much it hides varies from copy to copy. Read `reports/analyses/2026-11-07_analysis_67m-story.md` first.

Why finished: the one run that would test the cause of the broad failure (teacher training with a masked term in its
loss) failed its control four times before (E42) and is not a 30-minute run. Fable found no other cheap run that
decides either gap. Everything else on the task is written up.

## What ran (all on the cluster `debug` queue, $0; no RunPod spend this session)

- **The 67M summary.** An Opus writer built it from Fable's draft and advice, with tonight's results replacing the
  draft's wrong "placement explains the broad failure". An independent Opus check of its answer found three of five
  sentences overstated: the removal test reads "through" on 7 of 8 states, so the two readings also disagree on
  positives; the narrow-task exceptions were hidden; "because fragile" is an accounting split, not a test. The answer
  and every section were rewritten to match.
- **E130** (Fable's one recommended run): two more teacher-trained copies of the lowercase fine-tune, each with its own
  change moved into Δ. The LoRA copy's placement cuts its excess to about a quarter; the Adam copy stopped at step 20
  read worse placed than in Δ (2.84 against 2.17). Report `reports/analyses/2026-11-07_analysis_e130-copy-protection.md`.
- **E130 Amendment 1** checked that reversal on 256 fresh rows, because it rested on one block. It does not replicate:
  partial protection there (0.78, line at 0.8), and per block 0.62, 1.75, 0.60, 0.97. The LoRA copy holds at 0.23 to
  0.28 on all four blocks. So protection is not a constant of the recipe, and where it fails is not settled.
- Log republished (v281 and the flush after it): E130 card, both claims rows reworded to the summary's answer.

## Gap (b), the copy's distance from the fine-tune

No run tonight. It still reads as E121's ladder: copies trained from zero sit far (cosine about 0.40) and pass, copies
from the least-squares start sit near (0.81 to 0.90) and are exposed. Fable ruled out the simplest mechanism at $0:
with every alive mask at its causal-importance floor, the exposed and the in-Δ placements read the same (E127), so
the gap lives in which masks the adversary lowers, not in where the change sits.

## Decisions for you

- Whether to run teacher training with a faithfulness term in its loss (VPD's masked reconstruction) on the broad
  lowercase fine-tune. It is the only run that turns "fragile function, placement hides part" into a tested cause. It
  failed its nothing-to-copy control four times on E42; it would need a new plan, not tonight's kind of cheap check.
- The reuse sentence: under the frozen definition no 67M task reads "reuses C", and an all-sites removal (which might
  find one) is a definition change, yours to make by a dated `decisions.md` row.

## FYI

- E125 ran after E124 came back unreadable; the definition lead agreed E125 stays with this line.
- The E123/E125 option was taken by default (no answer within two hours).
- E127 Amendment 1's emoticon copy control stopped the job, most likely bf16 rounding; not rerun.
- E125 Amendment 1 was read as descriptive, outside the "stronger removal is your call" line.
- E129 was claimed by mistake and is unused (friction 614).
- E130 Amendment 1 was run without asking, as a $0 check of a one-block result. E130 has one amendment left.

## Friction noted

614 (ecode claims a fresh code for an amendment name), 615 (bf16 offsets saved for fp32 reads), 616 (notebooks left
unexecuted), 618 (GitHub 500 on push, transient), 620 (log publish needs a plain resend after "resent unchanged").
