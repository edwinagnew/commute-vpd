# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · reports/plans/2026-10-28_plan_claim-tests.md · pending decisions: card "e47-cuts" (Question for you · commute · Soon; Nothing runs until this is decided) · E47's five $0 cluster tests: the L40 reads ~2.5x slower than the H100, so the plan cut pieces to fit 30-min jobs (T1 adversary seed 2; T2 rank 32, second rows, LoRA seed 1; T4 step 10; T5 rate 0.003); options: run as cut, or split the cut pieces into extra 30-min jobs (lead recommends; T2 without rank 32 may not settle claim ii)
- 2026-10-28 · reports/plans/2026-10-28_plan_regating-line-reset.md · pending decisions: card "e46-null-rule" (Blocking · regating · Now; 0a and 0b stopped until decided) · E46 stage 0a stopped at the ' http' null: the full different-target fine-tune lowers the emoticon target by 9.7 nats (ratio −0.72 vs the filed two-sided ±0.1), W₀, identity and cached states exact; options A one-sided on G (lead recommends), B one-sided on G and gate gain, C drop it
