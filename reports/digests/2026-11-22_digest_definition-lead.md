# definition-lead digest, 2026-11-22 morning

**P1.50, due 11-23: the reuse score counted only over changed parts (option b) has been read on every realistic fine-tune. Its final page is reports/analyses/2026-11-21_analysis_e257-eb-trained.md, the cell is filled, and the log is published as v439.** I am finishing because all my cells are filled or closed and the rest waits on ask 59. The premise check came back doubtful (advisory). On trained pairs the score does not separate reused parts from new ones, and the gaps it shows may reflect how large the update is. One question waits on you (ask 59, below). The section 4 justification of option (b) waits on that answer.

## Cells
Filled overnight: P1.49 (option (b) rescore of saved readings) and P1.51 (the behavioural per-part test). Open: P1.50 (1 of 1 runs used, results home, page being written). Ladder: the lowest open rung is still L1.2 (P1.4, 1 of 3 runs). Its next run waits on how the definition is settled.

## What ran and what it found (E257, about $48 on pods, provisional until the page is committed)
- On the nine trained pairs, the score reads low everywhere (0.015 to 0.062). Reused items read above new items in only 3 of 9 pairs.
- On the repeated RL updates, RL reads 5 to 10 times higher than the SFT arms (RL 0.10 to 0.22, SFT 0.015 to 0.023, shuffled 0.03 to 0.05).
- On compare-lead's procedural tasks, RL and SFT read about the same (addition SFT 0.053, RL 0.075), and the matched random updates read about 0.01.
- Limit: every one of the 66 best-subset searches stopped at its 2,000-step cap.

## Waiting on you
- **Ask 59 (P1.49, due 11-22):** under option (b), should the definition test each changed part *by behaviour* or *by write*? By behaviour means the output stays put when that part alone is put back to the base model's part at its best strength, with the order-free write share as the cheap reading. By write means the write share itself is the test. On the built edits, the write share reads gain ×2 0.76, ×10 0.76, rebuilt 0.63, new 0.012-0.019, random 0.011. The behavioural test (E259) gave a pass share of 0 on all 19 items and mean readings about equal to the score above, both low. No recommendation, since this is a premise question.

## Spend
$0 this session so far. The pod spend, about $48, fell in the previous session.

## Friction noted
913 (log build drift), 957 (cell add refused a relay from the whip-lead). Both are still open.
