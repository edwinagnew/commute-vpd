# Cards lead digest, 2026-10-29

Every experiment card in the research log is now in the card format, so the cards-lead task is done. Each card was
read cold by a fresh reader, rewritten with figures redrawn from saved results, and read again until the reader's
remaining overclaims were fixed or named. No verdict or outcome chip changed, and no compute was spent.

## What ran
- Nine batches over the day; this session did batches 7-9 (published as log v121, v122 and v123), 14
  cards: E8, E9, E17, E18, E20; E23, E24, E28, E29, E32; E33, E35, E48, A1. Batches 1-6 are in the log's changelog
  (v109-v120).
- Each card: cold read, rewrite by a fresh writer, a second read, fixes, a final read, last fixes. New figures come from
  each experiment's `plot_log.py`, drawn only from saved results on the dev node.
- Before/after reads per batch: `~/commute-data/scratch/cards-lead/batchN/before_after.md`.

## What the readers found that bears on claims
These are places where a card or report says more than its evidence. Each card now states the limit; the verdicts
stand as the reports give them. Any of them could be a review report if you want a verdict looked at again.
- E17: "cancellation refuted" rests on the adversary's-share test, which has no working positive control either
  (the control built to cancel did not fail the adversary, ratio 1.11).
- E48: the conclusion that a pass reading the cue cannot be used borrows the adversarial check's 1.5 line; real
  fine-tunes move the cue 10-13 nats against a 0.65-0.77 nat gap in the unchanged decomposition. The card gives the
  report's reasoning for reading the ratio instead.
- E35: the old card said "largely yes"; the report, after its independent check, says "partly yes". The card follows
  the report.
- A1: the report's opening says the random-mask check moves under a percent; its own table shows +0.18 nats on one
  task. The card gives the table's figure.
- E32: the saved results give the emoticon task 10 keys of about 2 inputs each; the report and old card say one key
  per word task. The card follows the report.

## Pending decisions
None from this work. No asks open.

## Money
$0: no pods, no Slurm jobs. All plotting ran through `cpu-run`.

## Friction noted (open, cards-lead)
- The log publisher often needs two or three sends ("not viewed", then "unchanged"), and `read` with `paths` cannot
  save `index.html`, which `build.py --check` needs (170, 169, 168, 165, 162, 160).
- Seven cards' figures had no line in the README files list and had to be found by hand. A `build.py` check that
  every card `fig/` has a mapping would catch it (163).
- `check.py` counts alt text and joins sentences across HTML tags, so 20-35 flags per card are mostly noise (151,
  157, 164).
- `log-lock take --wait` outlasts the 120 s Bash timeout (154); parallel writers in one clone share its git index
  (152).

## Not run
- No card's analysis was rerun or rechecked against its raw runs; facts were kept as the reports state them.
- The review reports the five items above could each become were not written.
