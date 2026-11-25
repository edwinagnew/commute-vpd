# definition-lead digest, 2026-11-24 evening

**P2.81, due 11-25: filled. The Swahili and Yoruba fine-tunes of Fig 5 do not gain on a handful of bigrams or diacritics; their gain spreads over the language roughly with token frequency.** The twenty tokens that gain most carry only 0.10 to 0.14 more of the gain than their share of the text (filed line 0.15), and the gain is large, 0.92 to 1.47 nats per token. Lowercase and arXiv look concentrated, but redteam showed that concentration tracks small gain per token, so it says nothing about reuse. Step counts do not explain the pattern. Descriptive, no verdict on reuse. Page: reports/analyses/2026-11-24_analysis_p2-81-token-risers.md; review: reports/reviews/2026-11-24_review_p2-81-token-risers.md; relayed to whip-lead; log v485 published.

- Cells: P2.81 filled (E308, 1 of 1 run). Open: none of mine. The premise check said "doubtful": concentration measures how narrow a change is, not reuse against new. That goes to the user's gate.
- Not run: a shuffle null per fine-tune, which would separate "concentrated because reuse" from "concentrated because small gain". It needs a new cell; I offered it to whip-lead.
- Spend: E308, one RunPod 4090, $0.37. No pods or Slurm jobs of mine are running.
- Friction 1135, 1139-1141: the debug queue was blocked, so the run moved to a pod; job-gate refused the pod label; the pod sync's include path worked the opposite way to the CLAUDE.md staging rule; the numbers ledger needs one line per distinct value; files written by cpu-run appear a few seconds late.

---

# definition-lead digest, 2026-11-24 afternoon

**P1.62, due 11-24: filled. Figure 2 now reads the same 48 states the same way at all three featurizations, and its picture holds.** No per-task value moved by more than 0.0074 from the old figure, and built reuse edits read above every new edit and random update at VPD, top neurons and SAE latents. redteam-lead checked the rows and agreed, and writeup-lead has them to redraw Fig 2. I am finishing because my task's cell is filled and no pods or jobs of mine are running.

## Cells
Filled: P1.62 (E298, 1 of 2 runs). Open: none of mine. The lowest open rung is L1.2 (P1.4, one of three runs left); I am not proposing it.

## What E298 found
- The user's head (2,4) edits are their own group. The value and output edits read identically at neurons and SAE, because doubling either one is the same change to the model. At VPD they read about a fifth, since VPD's parts mix heads.
- One quoted order changed: at SAE latents the copy task no longer reads above both ciphers. writeup-lead has redteam's replacement wording for section 5.
- The premise check called the cell "doubtful": agreement between three featurizations does not test L1.2 or L1.4, which matter more to the gate page. That is listed for the user's gate; it changes nothing now.
- Page: reports/analyses/2026-11-24_analysis_p1-62-fig2-standard.md. Rows: experiments/fig2_standard/results/definition_rows.csv.

## Money and problems
About $45 on RunPod pods for E298, all terminated. The cluster's debug queue never started the rehearsal. Friction noted: 1110 (a pod sync deleted run logs), 1111 (cancelled rehearsals counted against the limit), 1115 (missing inputs found one crash at a time).

## Waiting on you
Nothing from definition-lead.

---

# definition-lead digest, 2026-11-24 morning

**P1.4, due 11-10: the trained reuse control (E297) could not be read, because the fine-tunes built to imitate the reuse pairs never got close to them.** Its filed test is reported as "no call", and its neuron and VPD numbers are written up as descriptive. redteam-lead agreed, and writeup-lead added one Limitations sentence to paper v2. I am finishing because all my cells are filled, no asks are open, and no pods or jobs of mine are running.

## Cells
Filled since the last digest: P1.56 (the reading at the top-activating-neuron featurization), P1.57 (four runs: the copy-against-lookup pair and the dropped five-task results), P1.59 (why readings fail on trained fine-tunes, $0), P1.60 (a rotation-tolerant kept share, two runs), P1.61 (the per-neuron gain fine-tune control), P2.69 (circuit overlap after Wang et al.), P2.73 (WeightWatch) and P1.4 run 2 (E297). Open: none. The lowest open rung is L1.2, where P1.4 has one of three runs left; I am not proposing it.

## What the last runs found
- P2.73 (WeightWatch): its score separates the strengthening edit from the new edits by far more than the random spread. But it cannot tell the sign of a change: a deletion also reads high. Report: reports/analyses/2026-11-24_analysis_p2-73-weightwatch.md.
- P2.69 (circuit overlap): the built controls pass. On most trained updates it gives no reading, because their circuits fall below the faithfulness bar.
- P1.60: rotating a neuron toward a needed neuron reads as rotation, and rotating it toward an unneeded one does not, so the control passes. The literal own-space share cannot see rotation at all.
- P1.4 (E297): the x2 imitators were left 0.300 away from their target, against a closeness bar of 0.192.

## Waiting on you
Nothing from definition-lead.

## Spend
This session has spent $0 on pods; no machines are on record. Tonight's earlier pods for E294 and E297 stayed inside their approved caps of $15 and $30.

## Friction noted
1086, 1095 and 1099. One run's code called git on pods, which have no git history. One brief had no pod fallback when the cluster was full. One pod-runner edited `remote-kernels.toml` while its session ran.

Five days remain until the 2026-11-29 draft.

## Not run
- P1.4 run 3, a closer imitation fine-tune: not proposed.
- Backlog 11: checking the sha of every predictions file a run loads.
- Restoring the P1.61 log caveat, "so low VPD readings are not evidence that mechanisms changed", which the clerk cut for length.
