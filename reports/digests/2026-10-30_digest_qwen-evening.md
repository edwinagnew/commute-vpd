# Qwen line, evening digest (2026-10-30)

**Why I stopped:** everything left on the Qwen line waits on one decision only you can make: what E57 Stage D
should do (ask #16, log card `decision-e57-no-teeth`, file `reports/decisions/2026-10-30_decision_e57-no-teeth.md`).
My recommendation is now option 6: re-run anchor 2 and save its states between steps 12 and 100, to find one the same
size as anchor 1. It costs $0 on the cluster, about 35 minutes per seed.

## What was done this evening ($0, no GPU)

- **E57 Stages A-C written up**: report `reports/analyses/2026-10-30_analysis_e57-reading-the-update.md` and
  notebook `notebooks/E57_qwen_reading.ipynb`. A fresh agent checked it independently
  (`reports/reviews/2026-10-30_review_e57-a-c-check.md`), and all eight of its issues were fixed (5bedde2). The
  research log is at v153 with the E57 card and the updated decision card.
- **The re-gating share has no teeth on Qwen**: every state reads 0.993-0.9995, the random changes included.
- **The direct share does separate the real updates from random changes, but partly by size.** Gold reads 0.092 and
  E40 0.063, against 0.022 and 0.029 for the random changes; the row-bootstrap intervals do not overlap, so the card's
  "close D3 now" condition is not met. Gold moves the outputs 1.7-2.3 times as much as the random changes, though, and
  among the real updates the direct share rises with size. Anchor 2, the same size as one random change, reads below it
  (0.012 against 0.029).
- **Why the size finding matters for D:** anchor 1 is a smaller update than anchor 2 (maths KL 0.33 against 0.62),
  so under option 2 a gap between the anchors would read as a difference in kind when it may be size. Option 6 removes
  that.

## Still open for you

1. Ask #16: E57 Stage D (above).
2. Unexplained: the own-samples runs never "learned" by the loss rule, yet own_s0 gained +0.052 on GSM8K.

## Money and housekeeping

No pod and no Slurm job tonight; spend $0. Problem #3 (E44 stage S stop, 44 h old) closed as superseded. Friction
noted: 274 (E57's scoring stage refuses before D, so the scorecard was written by hand) and 276 (log publish still
takes three sends).
