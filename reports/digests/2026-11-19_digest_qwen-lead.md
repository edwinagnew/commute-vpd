# qwen-lead digest, 2026-11-19

The Qwen readings carried over from part 2 are read on the six step-75 RL, own-samples and null updates and on the
gold pair, but the two word pairs could not be read because the held model's do-nothing control failed. All my work now
waits on one decision from you (ask 10, below). I have finished for the day; nothing is running.

## Cells
- P3.13, due 11-25 (blocked on ask 10): open, 2 of 2 runs used (E216 $7.41, E219 $6.09).
- P3.1, P3.2, P3.3, P3.5, P3.8-P3.12: filled.

## What ran and what it found
- Run 1 (E216): every control passed. The held-model reading S is 0.977-0.997 on the six updates against
  0.79-0.82 for random changes, but those random changes are matched in weight norm only and barely move the output,
  so the gap is not read as a result. Effective rank is 1,329-1,456 against a random floor near 1,810-1,830.
- Run 2 (E219): the gold pair reads clean and does not separate (0.996-0.999 against 0.997-0.998). The copy/cipher and
  known/invented rows failed the do-nothing control (2.8e-4 to 2.2e-3 against a bar of 1e-7), so their random
  references read nothing and the pairs are unreadable as they stand.
- Section 6: the text for these readings went to writeup-lead as
  `~/commute-data/agents/writeup-lead/note-2026-11-19-qwen-lead-sec6-L3.2.md`, with the word pairs marked open. No
  label and no verdict on Qwen.

## Waiting on you
- Ask 10 (P3.13, due 11-25): should I do one more run (about 40 min, about $6-8, three H100 pods)? It would bring the
  held model's do-nothing error on the copy/cipher and known/invented rows under 1e-7; it is now 3e-4 to 2e-3. Then
  it would reread those 24 states against random changes. Recommended: yes. If not, P3.13 is filled with run 1 plus
  the gold pair, and the word pairs are marked unreadable. Card: `reports/decisions/2026-11-19_decision_p3-13-held-fit-floor.md`.

## Money
$13.50 on pods today, all from P3.13. No pod is live.

## Friction noted
915 (the do-nothing bar tested the wrong KL), 916 (the dev worker's /ephemeral was full; pod labels), 918 (the
results summary names pair sides `reuse_side`/`new_side`, which risks a label leaking into the paper).

## Not run
- The third P3.13 run (waits on ask 10).
- The share on p-9707d6ff for the step-75 states (stays "not read", by your 11-19 instruction).
