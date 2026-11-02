# Digest: benchmark line, 2026-11-02

**Why finished.** The task's order is done and you ended the last line, so everything left waits on your next direction. We tried to give the 67M's fine-tunes a causal label so that VPD could be scored against cheaper methods, and the attempt gave no label. Whether VPD beats a cheap method on the 67M stays unanswered, and the method table's rows stay "only by assumption".

## What ran

Two free debug jobs on saved states, plus one short training run for a control. Total spend $0, no pods.

- **The probe stopped at its own controls (job 48971).** The control meant to read high could not reach the saved gains. The hand-written control fit too poorly to clear its bar. Switching off neurons took a third of them to cost 1 nat, so that reading is dead on this model.
- **The exploratory read gave no label (job 49069).** On the copying-heads pair the context patch follows the earlier head test, and the order holds when sizes are swapped. On the key maps it runs against the assumed labels, with copy maps lowest. Nothing known to reuse has ever read high on it.

Also done: the earlier counting row in the table is re-marked "by behaviour plus an in-state patch", with no label moved. The log's E66 card has the detail, and the report is `reports/analyses/2026-11-02_analysis_e66-interchange.md`.

## Decisions you made today

All are in `decisions.md`: the 67M patch before the harder counting test, the probe first, the exploratory read, and ending the line. The first answer reached me through commute-lead, so correct it if that relay was wrong.

## Pending decisions

None open on this line. Named and not run, if you want one:

- **The harder counting test (E65).** Its plan needs Fable's changes, and it can rank cheap methods, but VPD can only tie or lose. Its saved states sit in the reuse lead's clone, per job log 44270, which this role may not read.
- **The key-swap patch on the class map.** This is the one row where VPD could beat every cheap method. It needs a new plan and retraining, and Fable warns it may inherit the drift seen here.
- **Three free checks in the E66 report.** Fable says the size check cannot reopen the line.

## Not done

No independent check of E66's readings, because no claim above the log's cards moved. The notebook runs to about twice the write-up guide's length. The figures use the default palette, unvalidated, because this node has no `node`.

## Blocked and friction

`~/commute-data/blocked.txt` lists nothing. Friction is under `~/commute-agents/fleet/friction baseline-lead`. Today's main items: a card labelled "commute" sent your answer to the wrong lead, another lead's clerk published a card minutes before you answered it, and pulls restore already-flushed queue lines.
