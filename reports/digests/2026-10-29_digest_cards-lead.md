# Cards-lead digest, 2026-10-28 evening

All eight open pending-decision cards now carry Fable's advice, and the ordered table to clear them in one sitting is
`reports/decisions/2026-10-29_decision_card-triage-draft.md`. Everything left waits on your decisions, so the lead
has finished.

**What ran.** Five Fable advisors (one per card or pair of related cards), read-only; no GPU, no Slurm, $0 spent.
The advice reached the log in v89 (published by another lead's flush, which picked up the queued lines); this
lead's clerk then published v90 with one leftover line.

**The pending decisions, in order** (each card in the log's pending-decisions section):
1. `decision-regating-reset`: who runs stage 1 on the Qwen states (advice: the Qwen lead, in its own pod).
2. `decision-e42-criterion`: add "behaviour kept by the parts switched on, on the task's text" to the pass (advice: yes).
3. `decision-pass-anchor`: keep the fixed line and close (advice: yes).
4. `decision-exact-weights`: accept E43's fail (advice: yes, plus one backlog line).
5. `decision-qwen-rhythm`: not yet (the card's stated blocker was wrong; fixed).
6. `decision-lora-sgd`: run it, about $2 on the H100 (advice: yes).
7. `decision-rare-target`: rerun, about $2, after card 2.
8. `decision-e40a1-held`: not a question; remove.

**Friction noted.** 75: a card cited a conditional rule as its blocker. 84: queued log lines have no owner, so
another lead flushed this lead's lines while its clerk waited on the lock.

## Not run
- No second agent checked the advice; each card rests on one advisor.
