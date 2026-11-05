# Plan: does VPD's adversary fail doing nothing on the larger TOFU update? (E89)

*2026-11-05, D1 (a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement), day 3 of 8. Asked by the user (task file, 2026-11-05 10:33) before they choose on card `decision-after-e76`. Under the gate: about $11 and 2.7 h. No pre-run review, by the fast lane.*

## In plain words
On the first fine-tune update, VPD's adversary passes doing nothing, so we do not know whether it can tell an update left in Δ from one placed in the subcomponents. We read the same adversary, on the same rows and settings, on an update about ten times larger: the unchanged decomposition first as the control, then doing nothing, then the update on the rarely-on half. This says whether doing nothing fails at a larger size or whether its pass was not about size at all.

## The request, in the Terms' words
Forward KL under the adversary, clean to masked, over the unchanged decomposition at the same seed, on E85's 128 web rows with E85's settings. States: W0, do-nothing, rarely-on half (remainder in Δ, `qwen_maps.site_change("lower")`), two seeds each.

## Which update
**E81's, as the task says:** `results-large/qwen_tofu/dW/a2_2e-5/step_1250.safetensors`, sha256 `d45b5163d431e218708d66fa45fae47ac0c5f0e7ff6666fc5ddd06adcd35a37b`, ‖δW‖ 17.998, 9.6 times u1 (E86's log and report). It is the update E86 read, one wording of each fact. The user chose E84's five-wording update as anchor 2 this morning (‖δW‖ 18.109, about the same size). The card says a run must name which it aims at; this one names E81's because the task does, and the chat says so. The question here is about size, which the two share.

**Budget.** The task file reads "about 0" and has also lost text after "E81's": a `$` went through the shell, as in friction's "$0.50". The card says "about $10 for six starts", so this plan reads $10 and caps the pod at $14. A literal $0 cannot be done: a start peaks at 66 GB and the cluster's L40 has 46.

## Design check
- **Goal.** `GOALS.md`'s weak version: Δ may not absorb the update. This asks whether D1's one test aimed at that can see Δ absorbing it at a larger size.
- **Nearest tried.** E80 Amendments 1, 2 (u1: do-nothing 1.24), E85 (rarely-on 1.27), E85 Amendment 1 (random at 100 times the size 32.5). No real update beyond u1's size has been read. E85's two amendments are spent, so this is a new code.
- **Patch or step.** A control; the user's instruction. Departs from no decision.
- **What changes next.** See "What each outcome means".
- **Control first.** W0 s0 and s1 alone, each a stop if over 20% from 1.703, 1.438; three H200 NVL pods read s0 to every digit. Fails on the host, the rows or a state left applied. Reproduces nothing with a warm start, so no list of differences.
- **Saved.** `results/results.json`, `run.log`, each under 50 KB. No state files: both states rebuild from the δW and the inputs.

## Numbers and what each should read

| number | W0 | do-nothing | rarely-on half |
|---|---|---|---|
| Final KL ratio, mean of two seeds over W0's same seed (the rule) | 1 (absolute 1.703, 1.438) | 1.24 on u1; here about 10 or more, see the lean | 1.27 on u1; here over the bar, lower than do-nothing, low lean |
| Clean KL against W0, the 128 web rows | 0 | unmeasured; u1's 0.01-0.2; stop if under 1e-3 | equal to do-nothing's within 1e-6, same weights: a build check |
| Left in Δ, plain share, pooled | 0 | 1.00 by construction | unmeasured; above least squares' 0.0984 (E86, this file); may pass 1, the second-order term is 92 times u1's relative to the update, so logged, stop only if not finite or under 0.0984 |
| Behaviour kept | 0 | 1.00 by construction | 1.00 by construction |

Behaviour kept is not read: both states' weights are W0 + δW, and E85's gold rows are GSM8K, whose gain under TOFU's update can be near 0 or negative, which would make E85's share degenerate. The same-weights check above does its job as a build check. So `gold` is cut. Beside the rule, logged and not ruled on: the mean of the last five step reads, excess in nats, Qwen's own bars 1.267 and 1.776 (E85).

**Filed outcome, do-nothing, on the mean; bars pass ≤ 1.40, fail ≥ 1.61, inclusive, as E85 (`rules_amend2.described` and `verdict`).**
- Mean at or over 1.61, gap (mean − 1) strictly above this job's W0 spread: "do-nothing fails on this larger update".
- Mean at or under 1.40 and no seed at or over 1.61: "do-nothing passes on this larger update too".
- Else "no call", reported as read, opening no amendment. A NaN or missing start: "unread".

**Filed outcome, the pair, reading do-nothing's and rarely-on's labels:** do-nothing fails and rarely-on passes, "the adversary separates them"; both fail, "both fail"; do-nothing passes and rarely-on fails, "rarely-on alone fails"; both pass, "both pass"; else "no call".

**Lean.** Medium-high that do-nothing fails, from measured scaling at seed 0: random's excess goes 0.0051 nats at 1 times to 53.7 at 100 times (E85, E85 Amendment 1), a power of 2.0. u1's do-nothing excess is 0.553 nats (E80 Amendment 2, H100 not H200 NVL, the reference is the same seed's W0), so at 9.6 times the same power gives about 51 nats, a ratio near 30. Against it: TOFU's update was trained on invented biographies, so web rows may hear it far less than maths-trained u1's; hence not high. Low that rarely-on reads below do-nothing beyond W0's spread: it leaves less in Δ, but E85 Amendment 1 shows rarely-on columns can fail when moved far. Medium that both fail.

**Edge cases, run on the rule before filing** (`tests/`): a pair straddling a bar; W0's seeds equal (no spread: over the bar is no call); a spread wider than the gap; one state unread; a NaN; a zero W0. **On the do-nothing state, can it pass the candidate's bar?** Yes, that is the question: do-nothing is a state read, not a reference, and neither state here is a candidate map. **Blind to.** The ratio reads web rows, on which TOFU's update may move outputs little, and a pass there is not a pass on TOFU's own rows. It is a lower bound still climbing at step 20 (E85 Amendment 1's Rand100 read 14.7 then 32.9 at its last two steps), so a large ratio is not to be quoted to two figures. Neither state is a good one: both failing says nothing about whether a good placement passes. Bars are the 67M's. Two seeds. **Noise floor.** W0's two starts 18.4% apart; a state's two ratios differ by 0.12-0.19 at u1's size (E80 Amendment 2); unmeasured at this size.

**Departures from E85.** Update is E81's, not u1. Do-nothing re-read on this host (E85 quoted it). Gold rows and behaviour kept cut, reason above. Rand cut, as asked. Rare's lower stop from TOFU's least squares (0.0984) in place of u1's 0.0900, upper stop removed (reason in the table). `n_state`'s path check stays at 1e-4: E80's was 1.2e-9 on u1, so it has room. All other settings E85's, by calling its code.

## What each outcome means, for the card
- **Do-nothing fails, rarely-on passes.** The adversary separates a state with all of the update in Δ from one with less, at this size. It moves from "reported beside" to a scored check on TOFU in option A or B.
- **Both fail.** The adversary fails two bad states at this size, which is what a test should do, and it is not yet shown to pass a good one. It needs least squares read on the same update, about $3 and two starts more, before it can score a map.
- **Do-nothing passes.** Size was not why it passed on u1. The adversary stays beside the table, with this as the stated limit, and option C's argument gets stronger.
- **No call.** Reported as read.
Whichever it is: the card's update is rewritten with this number and the choice goes back to the user with a recommendation. This run does not change the options' map runs.

## Cost
Each start 1316-1333 s and peak 66.0-66.6 GB on an H200 NVL (`experiments/qwen_d1_teeth/results/results.json`, `.starts`); W0's 60.4 GB. δW here is fp32, 5.64 GB; the code writer states u1's file's size in the hand-back so that the peak can be checked against 80 GB. H200 NVL on Vast $3.78-4.06/h listed, placed price can run 20% over (`fleet/roles/lead.md`).

| stage | time | $ at $4.06/h | stop |
|---|---|---|---|
| upload δW to W&B, cluster CPU job, first | ~5 min at 19 MB/s (job 39093's log) | 0 | sha256 |
| set-up on the pod, Qwen fetch ~30 MB/s | ~40 min on a fresh pod (lead.md); E85 logged ~20 | 1.35-2.70 | sha256, alive count, card under 70 GiB |
| load, clean KL, X, left in Δ | ~5 min | 0.35 | the table above |
| W0 s0, W0 s1 | 44 min | 2.97 | each over 20% off |
| do-nothing s0, rarely-on s0, do-nothing s1, rarely-on s1 | 88 min | 5.95 | a first step not finite, build checks |
| total | ~2.7-3 h | ~11-12 | pod cap $14 |

## Not run
Least squares or any map on this update; E84's five-wording update; TOFU's rows under the adversary; a third seed; more than 20 steps; behaviour kept; a size ladder.
