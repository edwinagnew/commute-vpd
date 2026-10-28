# Re-gating lead, evening of 2026-10-28: stage 0 waits on two decisions

The re-gating reset's first check (stage 0a of E46) ran on the cluster for no cost and stopped at its null test. The
original model's numbers, the identity check and both saved fine-tunes matched their filed values exactly. The null
failed because the rule we filed for it cannot pass, not because of a bug: a fine-tune that teaches a different token
after the same trigger must lower this target. Nothing else can run until the user answers the two questions below.
I stopped for that reason.

## What ran
- Stage 0a, Slurm job 41161, 42 s, $0. W₀ cells, identity r = 0.597 and the cached seeds' T₁ (8.41, 8.67) all exact.
  The ` http` null read a gain of −9.7 nats against the seed-0 fine-tune's +13.4 (ratio −0.72; the filed rule is
  |ratio| ≤ 0.1). The E12 and E36 groups did not run, because the job stops at the first fail. Result:
  `experiments/regating_table/results/stage0a.json`.
- Fable's advice on the P0.4 question, written into the plan and the log card, marked as Fable's.
- Research log v93 published (https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu).

## Decisions for the user (log section 3)
1. **`decision-e46-null-rule` (Blocking, ask #3).** Make the null one-sided (A, recommended), one-sided on both the full
   gain and the gating-only gain (B), or drop it from 0a (C)? A rerun of 0a takes 5-25 min at $0.
2. **`decision-regating-reset`, P0.4 question (ask #2).** Score the direct-write test on the random-map writes (A,
   recommended by both the lead and Fable), keep it as filed (B), or require both (C)?

## Housekeeping
- The clone was blocked by another session's unfinished E34 notebook edit. I set it aside without resolving it:
  `stash@{1}` holds the original edit and `stash@{0}` the clashing merge. Applying it is the user's or that
  session's call.
- Money: $0, no pods. Friction noted: 95 (wait-job on an already finished job), 96 (log publisher refusals, hook on
  `sed r`).

## Not run
Stage 0a's E12 and E36 groups; stage 0b; the write-up (report, notebook, TRIED.md row) until 0b has run.
