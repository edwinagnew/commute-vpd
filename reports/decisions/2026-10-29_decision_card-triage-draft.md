# Card triage (draft): the eight open decisions, in the order to clear them

Every open pending-decision card was given to a Fable advisor, which returned options, costs, a recommendation and
what would change it. The advice is written into each card as "Fable's advice (2026-10-28)"; every choice stays
yours. Six of the eight can be cleared for nothing, and the first one restarts the stopped re-gating lead.

Draft by cards-lead, 2026-10-28 evening. Advice is Fable's; nothing here is decided. Cards are in the log's
pending-decisions section (ids below); the full advice per card is in the card.

## The table (clear top to bottom)

| # | Card id | What is stuck | Fable recommends | Cost | What would change it | Unblocks |
|---|---|---|---|---|---|---|
| 1 | `decision-regating-reset` | You chose stage 0 (regating-lead task); Plan Amendment 1 (commit f332d8b) found stage 0 as written cannot fail, and refiled it with a direct-write null as the test that can retire the line. The lead waits for you to accept it. | Adopt Amendment 1, with the direct-write null written per key, read on both anchor tasks, its off-target cost reported beside the fine-tune's. Decide now that the Qwen lead runs stage 1's forward passes in its own pod. | $0 (cluster), about a day of the regating-lead | You want stage 0 only as a table, with no claim about the instrument (drop the null); or the Qwen line's own route-split rows are a week away (retire, read the null there) | **regating-lead (stopped)**; Qwen line's Q3 |
| 2 | `decision-e42-criterion` | The cipher copy passes the agreed test on ordinary text, but the parts the decomposition marks important at the cue keep 0.007 of the answer, against the copy's 0.96. Should the test change? | (b): add "behaviour kept by the parts switched on, on the task's own text" to the pass, in the form the Qwen re-plan already uses; one rule for both lines. Card's own "not settled" was about whether a map can meet it, not about the test. | $0 to adopt; re-reading saved copies is about 30 min on the cluster | The original decomposition already loses the cipher windows under its own masks: then the reading judges VPD, not the map, and (c) holds | commute-lead's next design; cards 3 and 6; one test across lines |
| 3 | `decision-pass-anchor` | The adversarial check's 1.5 line fails doing nothing on large early updates. Fixed line, do-nothing anchor, or retrained anchor? | (a): keep the fixed line, report doing nothing beside each verdict, close the card. A do-nothing anchor would pass the very state the test exists to fail. | $0 | A map with the residual untouched failing by about its do-nothing's amount on two tasks | Closes a card; with card 2 fixes the pass test |
| 4 | `decision-exact-weights` | Teacher training plus the remainder in the residual failed the check (2.5 and 1.9 times). Accept, or another route? | (a): accept; teacher training's own state stays the map. Add one backlog line: the remainder was mostly teacher training's own overshoot, so the route to exact weights is teacher training held near the fine-tuned weights. | $0 | A plus-remainder state with a remainder of the fine-tune's size still failing (closes every form); or a near-weights teacher training passing (reopens) | Closes a card (commute-lead) |
| 5 | `decision-e40a1-held` | Nothing: the hold on Qwen teacher training is already decided by the re-plan. | Remove the card; fold the hold into the Qwen experiment card. It becomes a question only if E44 reads "every split fails at step 10". | $0 | E44's readings | Tidies the section |
| 6 | `decision-qwen-rhythm` | Whether Qwen copies each checkpoint in 100-step links. Nothing on Qwen waits on it; the card's stated blocker (layer-by-layer first) applies only if memory forces it, which the H200 avoided. | Not yet: decide inside the Qwen teacher-training plan if it ever runs. Drop the card's "nothing on Qwen changes" banner. Optional 67M firming run (one 700-step copy, a chain on another task). | $0 (optional run about $2) | E44 failing at step 10 makes it a design choice; a 67M chain failing at another seed makes it "drop" | qwen-decomp-lead, later |
| 7 | `decision-lora-sgd` | Spend about $2 on a rank-8 LoRA trained by plain SGD to tell form from optimiser on the broad task? | Yes, on the H100, optionally with two near-free readings in the same job (the SGD decompositions on the second set of rows, the labelled SGD state at step 1,000). The risk that SGD cannot train this LoRA is answered by E27. A fail means the form passes only with Adam, not "points at Adam". | about $1.5-2, half a day of commute-lead | The SGD decompositions pass on the second rows (little left to separate); no SGD rate reaches the gain (score nothing, report the grid's range) | commute-lead's next run |
| 8 | `decision-rare-target` | The task-text copy missed both bars by less than the check's own noise. Accept "not placed", or rerun? | Rerun step 100 and a second batch order in one H100 job, stop rule filed first, the important-parts reading at the cue printed beside each verdict. Not the free cluster: its noisier adversary would leave a near-line reading inconclusive again. | about $2 | Card 2 decided as (b) and the copy fails it (skip); the rare target moves to Qwen (stop) | commute-lead; read after card 2 |

## Why this order

Card 1 is first because a lead is stopped on it. Cards 2-5 cost nothing to clear and remove four cards: 2 decides the
pass test that cards 3 and 8 are read against, so it comes before them. Cards 6-8 are the ones with a run or a later
condition attached; card 8 waits on card 2, since under (b) its copy may fail regardless.

Two cards carried an error the advisors caught: the Qwen rhythm card named the wrong blocker, and the LoRA-SGD card
read a fail as "points at Adam" where it would show an interaction. Both corrections are in the cards.

## Not run

- No advice was checked by a second, independent agent; each card has one advisor's reading.
- Nothing was run: the advisors read reports and code only.
