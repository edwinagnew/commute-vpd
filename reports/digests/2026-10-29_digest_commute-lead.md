# Digest: commute-lead, 2026-10-29 afternoon

**Why I finished.** The task's work is done. The only item left is ask 6: drop the E39 LoRA-SGD run (~$2 on the H100)?
I recommend dropping it. E49 found that the adversarial check cannot fail an SGD-trained update on the new Pile rows,
so a LoRA-SGD pass would mean nothing.

## What ran (all $0, MATS cluster debug QoS)
- **E49 row blocks** (jobs 42959, 42963, 42998-43003). The decomposition after teacher training on the Adam fine-tune
  fails the check on all four new blocks of Pile rows. One of those blocks is near the line, at 1.66. The LoRA's passes
  on all four. The SGD ones pass too, but so does SGD's do-nothing, which fails none of the four blocks. So the check had
  no power on SGD there, and the SGD case stays open. An independent check agreed with every score.
  Report `reports/analyses/2026-10-29_analysis_e49-row-blocks.md`, log card E49.
- **E48 Amendment 1** (job 42960): the subcomponents that carry the gain in E42's three cipher copies. Half the gain
  needs 1,909, 1,138 and 986 of 9,881 subcomponents, against the "tens" filed (a miss). So the repair is not a small
  change to Γ. The copy of a cipher the model already half did spreads just as widely, so this count does not tell a new
  behaviour from an old one. Report: the E48 analysis → "Amendment 1".
- **The E42 re-read** of the task had already run as E48 earlier today. Its first step fell back (1.86 and 2.21 against
  1.5), so reading A decides.

## Decisions recorded
- **Δ may change a little but must not absorb δW** (your words, relayed by another session). The new `decisions.md` row
  says teacher training plus its remainder (E43) is a live candidate. Every "Δ untouched" in the log's criteria is
  reworded.

## Pending for you
- Ask 6: drop the LoRA-SGD run (recommended), or run it.

## Money and log
- Spend this session: $0; the cumulative $2.12 is unchanged. No pods or jobs running.
- Log v110 is published. Another clerk's card rewrites (E10, E12, E19, E31, E37, E47) were merged mid-flush, so they are
  in the repo but not yet live; the next publish carries them.

## Friction noted
142 (a brief that duplicated E48), 146 (the carriers listing saved only its top 20), 147 (cpu-run busy, checks
skipped), 150 (the regen job hung at exit), 153 (the publish missed merged cards). Item 125 (SAVED.md rows) is fixed.
