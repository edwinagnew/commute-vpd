# definition-lead overnight digest, 2026-11-10 → 11

**P1.9, due 11-12: the built control ran, and the definition failed it. The search found no map on either arm, so by your condition the wording needs revising. The positive control was weak, though, so one cheap test is recommended before revising.** The decision card is `reports/decisions/2026-11-11_decision_built-control-result.md` (ask 31). The page is `reports/analyses/2026-11-11_analysis_built-control-xor2.md`.

## Cells
P1.1 filled (the gate is decided), P1.2 filled, P1.3 filled (1/1 runs), P1.4 filled (1/3), P1.5 closed (0/2), P1.6 filled (1/1), P1.7 filled (1/2), P1.8 filled, P1.9 filled (1/2: two planned jobs of one run; job-gate counted job 2 as a stage). No cell is open. The finish reason: everything left waits on ask 31.

## What was done
- **The gate is decided (your answer to ask 29).** The second wording is adopted tentatively. The cut-offs are 0.8 for the re-weighting share and 0.5 for the rewiring share, recorded in a `decisions.md` row with your words. The saved results were relabelled at the new cut-offs (`experiments/old_task_labels/results/labels_cut_0p8_0p5.json`). No old 67M task changed label: every old reuse side except lowercase reads a rewiring share of at least 0.58. So 0.5 did not turn them into negative results. The only change: the 300-step copies of the hand-made "always on" change now read "no map found", where they read rewired before.
- **The built control (E148).** The first design reused the XOR copies from E139. Those had never learned the XOR, so it could not run. As you asked, Fable picked a new task, XOR2: each of 128 key tokens carries a hidden bit, and the push fires when the last two keys' bits differ. You approved it. Both jobs ran on cluster debug at $0 (jobs 57970 and 58002).
  - Controls passed. The plain 67M does not hold XOR2 (linear read 0.468, bar 0.55). The taught 67M holds it, weakly (0.588). Both fine-tunes learned the target on key pairs held out of training.
  - The rewiring share was 0.02 on the taught arm and 0.08 to 0.09 on the plain one, both under 0.5. Every item reads "no map found". The filed prediction for the taught arm (at least 0.5) missed.
  - The independent premise check reads it as "doubtful": every past success had a variable held exactly, and the taught model held XOR2 at 0.588. Whether the search fails in general, or only on weak variables, is open.

## Waiting on you
- **Ask 31**, quoted: "The built control found no map on either arm: revise the wording now, or first test the search on a strongly taught original?" The options are:
  - (a) revise the wording now;
  - (b) first test the search on a strongly taught model: one run, $0, about 1.5 h (recommended);
  - (c) keep the wording and report "built" as unreadable.
- Proposals 10 and 11 under NEEDS YOU are stale duplicates of a change you already approved in the dialog. Please reject them.
- The research log card is queued in `PENDING.md`. compare-lead's clerk publishes it; a note is left in its folder.

## Money
$0: cluster debug only, no pods.

## Friction noted
686: a design reused results that were still queued. 687: the label rule is hard to find. 689: a dialog yes is not seen by `cell change`, which leaves stale proposals. 690: one existing `make test` failure. 692: job-gate has no form for a planned second job.

## Not run
The strongly taught arm (option b). Any revised wording. G on seed 1 of either arm.
