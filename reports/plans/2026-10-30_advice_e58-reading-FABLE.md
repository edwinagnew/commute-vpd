# Fable's advice: how to read E58 (the held-out-key label disagrees with r)

*2026-10-30, Fable, at the lead's request. Advice only; nothing filed or run. Words as in `GOALS.md` → Terms. Read beside
`reports/plans/2026-10-30_plan_e58-open-pair.md` and `experiments/open_pair/results/run.json` (job 46266).*

## In plain words

We trained the half-known word map on 24 of its 32 keys and asked whether the model also improved on the 8 keys it never
saw: it did not, on all three seeds, and the random map did not either, so the first label the decomposition did not
make reads "new". The decomposition's two readings say the opposite, that the update runs through subcomponents the
model already had, as strongly as before. I read this as the two measurements answering different questions, old pieces
under new per-key routing, which is the granularity limit the user named, not as the label being empty and not as VPD
being wrong; one $0 check and one $0 positive control decide between those, and the user should be asked whether
"reuses" in this line means old pieces or a rule that generalises.

## What the numbers say (per seed, elicit then cipher)

- Trained-key gain G 11.0-11.3 / 12.5-12.9; value-specific G_spec 18.2-18.5 / 9.4-10.7. Held-out: raw own-value gain H
  +0.46 to +0.90 / +1.19 to +1.38; the other seven held-out values H_other −0.23 to −0.64 / +1.29 to +1.58; H_spec
  −0.48 to −0.70 / −3.2 to −3.5. x −0.03 to −0.04 against 3 s 0.02; raw x 0.04-0.08 would read "neither", not "reuses".
- Implied from H and H_spec: at a held-out key the 24 *trained* values rise about +1.6 log-odds on the elicit states,
  the held-out key's own value +0.6, its sibling held-out values −0.5. The elicit fine-tune is a lookup over 24 keys plus
  a push toward the trained values after the marker; it is not "sharpen whatever W₀ already prefers here", which was the
  one route by which carry-over could have happened.
- r 0.76-0.79 / 0.22-0.24, ratio 0.93-0.95 / 0.30-0.34 (P4 pass; E54 reproduced within its ranges). S 0.49-0.50 / 0.15-0.22.
- **P3 is not a finding.** The S cut is the same 0.5 (`open_pair.p3` passes `th["cut"]` to all three instruments). The
  elicit S values are 0.5014, 0.4982, 0.4898: one reads "reuses", two "new", so S scores 5 of 6 by sitting on the cut,
  not by reading the label. Report P3 as scored, say this in the same sentence, and do not carry "steering_better" into
  the log's headline.

## The three readings, and which is supported

1. **"New" was guaranteed by construction, so the label is empty here.** Half right. There is no key→value rule (each
   value is a per-key pick among W₀'s rank 2-20), so the only mechanism that could carry to unseen keys is a generic
   sharpening of W₀'s own preference at the key. That route was open and the fine-tune did not take it (own +0.6 and
   trained values +1.6 at the same window). So the label says something true about the update: it memorised 24 routes.
   What is missing is the label's positive control: no 67M state yet shows "reuses" is reachable, so E58 cannot be read
   as a VPD fail, and P1's "blind to" line already says so.
2. **The granularity limit, first concrete instance.** Best supported. The values are tokens W₀ already emits after the
   key (rank 2-20), so the output pieces exist; the cipher's values do not, so it must build. r and the ratio separate
   exactly on that (0.78 vs 0.24). The held-out label asks "is there a rule that generalises", r asks "are the carrying
   subcomponents old". On a lookup task over old pieces those must disagree, and E58 shows they do, cleanly, three
   seeds each way.
3. **The label is right and r reads existing pieces, not existing behaviour.** The same as 2, stated as a definition
   rather than a limit. The consequence for the programme is the one that matters: the step-2 plan's P2 ("instrument
   label equals behavioural label") is only a fair test on tasks where "reuses" by behaviour means "a rule W₀ has"; on
   lookup tasks it is undefined, not failed.

Readings 2 and 3 are not separated by any check; they are the same claim. Reading 1 is separated from them by the two
$0 items below.

## Two $0 checks, in order

- **The W₀ rank-k readout on the six saved states** (one `debug` job, minutes; the states are in `results/large/`). At
  each held-out window, the gain of each of W₀'s rank 1-20 word-like tokens at that key, alongside the own value and the
  24 trained values. Expected under reading 2/3: the trained values +1.6 whatever their W₀ rank here, W₀'s own top-20
  near 0 or slightly up and the own value no higher than its W₀-rank peers. Expected under "generic sharpening": W₀'s
  top-20 all up by about the own value's +0.6, scaled with W₀ rank. Either way, report it beside H; it is what the
  lead's question asks and it removes the "guaranteed by construction" worry or confirms it.
- **The label's positive control: a map whose rule W₀ already has**, 24/8 keys, E54's recipe, three seeds. The cleanest is
  "after the marker, say what W₀ says anyway" (each key's rank-1 token): if the model learns to ignore the marker, the
  held-out keys carry and the label reads "reuses"; r on it is the first labelled-reuse state for the line. Guard: G
  may be small because W₀ already predicts the value; set the gate from W₀'s rank-1 probability before filing, and if
  G cannot reach it, use the copy map (E55's Stage A family), whose label is fixed by construction but which still
  shows the rule can read "reuses". Without this control, no "reuses" from the bar has ever been seen on the 67M.

## What the lead does next

1. Analysis report by `/writeup`: the answer is reading 2/3, the "Not run" list names the rank-k readout and the positive
   control, the P3 sentence above, and a "Departures" entry (bounded-g behaviour only, no patch). `TRIED.md` row,
   `PENDING.md` line. Never refile P0-P4; the E58 miss on P2 stands with its cause as a dated report.
2. The rank-k readout and the positive control as one $0 job via `/pod-runner`, control first.
3. Amend the step-2 plan's P2 by a dated note: scored on rule tasks (the positive control, Stage B's rows) and reported
   unscored on lookup tasks. The novelty column (item 3 of the step-2 advice) waits until a labelled "reuses" state
   exists to compare against; on E58's states alone it has no expected value.
4. No Qwen gate opens on E58.

## The one question for the user

Yes, this is worth a question: it fixes what "reuses" means for the line. **E58's first earned label disagrees with r:
the fine-tune of the half-known map does not carry to the 8 unseen keys (new, three seeds), while r 0.78 and the ratio
0.94 read "reuses" as on E54. We read this as the granularity limit you named: new per-key routes through subcomponents
W₀ already has. Do you accept that the re-gating line's "reuses" means "old pieces under new routing", to be scored
against labels of that kind (a $0 positive control where carry-over is reachable, next), or should r be held to the
held-out generalisation label, in which case E58 counts against VPD?** Recommendation: the first; proceed with items 1-2
unless you object.

**Not run.** Nothing; advice only.
