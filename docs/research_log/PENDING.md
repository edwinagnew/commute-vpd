# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-26 · reports/decisions/2026-10-26_decision_next-while-waiting.md · Pending decisions (new card, "Soon"), What's next commute card · While the larger model is awaited: proposed teacher training of E36's no-trigger fine-tune (rehearsal of the larger-model protocol), plus the two-adversary disagreement and the copy's uniqueness as cheap checks; waits on E36's final fine-tune recipe (first run's held-out gain was negative)
- 2026-10-26 · experiments/teacher_spread/results/teacher_spread.json · Pending decisions (new Blocking card decision-broad-copy, already written in parts/sections/decisions.html by lead-2; it supersedes the next-while-waiting card line above, which need not be added), What's next commute card · E37: a copy of the lowercase fine-tune keeps its behaviour (0.998) but fails the harness (2.20; do-nothing 4.87); adjoint keeps 0.80 and passes (1.05); copies still learning; E37 card and analysis to follow
