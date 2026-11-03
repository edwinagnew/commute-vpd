# Plan: E77, the three open points on the masked forward

*2026-11-03, D1 (a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement), day 1 of 8. $0, one cluster L40, under the gate, fast lane.*

**In plain words.** With all 196 matrices masked by the original model's masks, the whole update barely reaches the output; the independent check found no bug but left three holes. This job asks whether a change the masked forward computes itself does pass, how much of the update passes when only some matrices are masked as VPD's training does, and how far the update moves the forward under masks made on the fine-tuned model. The second bears on which masks teacher training's loss should use.

## Design check (class 1, GPU passes of the existing harness on saved states)
Serves the claim D1 rests on and the map the user chose. Nearest tried: E72, E74; the check's "Not run" items 1, 2, 4 were never read. A patch that closes holes before teacher training is built on them. No decision departed from. Control first: u1 alone, stop on a miss; it fails on wrong rows, leftover weights, a changed hook or a different draw. At k = 196 nothing differs from E74.

## Numbers (first 32 rows; reads 1, 2 on u1, u3; read 3 on u1, u2, u3)
τ is E74's: KL(masked ‖ masked + change) / KL(W0 ‖ W0 + change). Amplitude is the ratio of centred logit shifts.

| number | at W0, no change | do-nothing, whole update | control | lean | blind to |
|---|---|---|---|---|---|
| 1. amplitude, τ, cosine, own-output change, blocks 0, 7 | ε = 0: KL 0, τ unread | not in this read | δW there reads 0.001-0.004 | at least 0.5 | direction (cosine reads it, no rule); one sign of ε |
| 2. τ, amplitude at k/196 = 1, 3/4, 1/2, 1/4, 1/16, 0, library's draw; fixed, stochastic | τ unread | k = 196 E74's; k = 0 reads 1 | τ, u1 0.0054987, 0.012789; u3 0.00040950, 0.0010740 | over 10 times from 1 to 1/4 | adversarial documents; which sites matter |
| 3. D / R: KL(update off ‖ on), fine-tuned masks, over KL(W0 ‖ FT) | D = 0 | off and on are E72's do-nothing, ceiling | Σ KL to FT 851.983, 854.419 (u1) | under 0.02 on all three | direction |

**Own-output change.** ε (x V · g) U added at a block's seven matrices, clean and masked; ε fitted so its clean KL matches δW's at that block in E74's file. Sound masks: amplitude near 1, cosine high. A broken hook or stream: as low as δW there, or cosine near 0.

**Routing, ported.** `launch_config.yaml:234-235`: `routing: type: uniform_k_subset`. `core/recon.py:227-229`: `k = random.randint(k_key, leading_shape, 1, n_sites + 1)`, `perms = random.uniform(perm_key, (n_sites, *leading_shape)).argsort(axis=0)`, `routed = perms < k`. A routed-out site outputs x Wᵀ (`decomposed_linear.py:142`); m = 1, d = 1 in E74's hook is exactly that. Three route draws per row.

**Departures from the library.** (i) `adv_fraction: 0.9` (`launch_config.yaml:231`): 90% of documents run all-routed under persistent adversarial sources, only 10% use this routing, with fresh U(0, 1) (`core/masking.py:352-356`). The library's draw here is that 10%; stochastic k = 196 is nearest the 90%, without its adversary. (ii) The update enters in full at every site, as E74; the library's Δ would pass d of it at a routed site. (iii) One mask draw per row, E74's, across route draws. (iv) Fixed k and fixed masks are ours. (v) fp32, torch's generators.

## Rules (`rules.py`, `predictions.json`; edge cases run: zero denominator, NaN, inf, band-edge ties, missing read)
No-call band: threshold × (1 ± 2w), w the first stage's jackknife relative error over rows, w√2 for the ratio.
1. **Own output:** "passes" above the band round 0.5, "fails" below; clean KL over 20% off target "unmatched". Any fail stops the job. Medium lean: passes.
2. **Steepness:** τ at k = 49 over τ at k = 196, per update and mode: "steep" above the band round 10, "shallow" below. All four alike, or "mixed". The library's value is reported without a threshold. Medium lean: steep.
3. **Displacement:** D / R "small" below the band round 0.02, "large" above; one large refutes. Medium lean: small.

## What each result does
- **Read 1 passes:** "why the masks lose it" narrows to what they switch off. **Fails:** stop and report; the claim may not be about the update at all.
- **Read 2 steep:** "all 196 masked" is the extreme case, and the wording must say so. Routed-out sites read W and carry the update, so teacher training's loss under partial routing alone would feel little of the deafness. It should keep an all-routed term, as the library's 90% does. **Shallow:** a few masked matrices lose it, and the loss can keep the library's routing.
- **Read 3 small:** the fine-tuned limb can say "barely moves the forward", not only "barely nearer FT". **Large:** it moves, though not towards FT.

## Cost and saves
$0, ~1 h: control 4 min, read 1 7, read 2 44, read 3 3. `--wall-min 110`, resubmit resumes per read. GPU ~46 GB, host ~45 GB. Saved: per-row sums per read, `score.json`, `run.log`.

**Not run:** the random slice of δW (the check's item 3); adversarial sources; the library's own Δ share on the update; own-output changes of the other sign or later blocks; reads 1, 2 on u2, u4; accuracy.

## Amendment 1 (2026-11-03, after job 50608 stopped)

**What happened.** Read 1 failed on all four cells, against a medium lean of "passes": the own-output change passes 0.05-0.11 of its clean logit shift, cosine 0.01-0.05. The plan said this may mean the claim is not about the update at all, and that low amplitude or a cosine near 0 is also what a broken hook looks like. Read 2 did not run; read 3 ran on u1 only (D / R 0.008, small).

**Design check (class 1, same harness, no training).** It serves the claim D1 rests on, and it bears on E76: a masked loss term cannot teach what the masked forward cannot pass. An unexplained apparatus number blocks what is read on it.

**Added, cheapest first, control first.**
1. Hook control: the own-output change at the last matrix alone, where nothing masked follows. Near 1 expected; a miss stops the job.
2. Swap read: row A's masks on row B's tokens. Does the output sit nearer clean A (the masks carry it) or clean B (the input does)? Expected at own masks: 0.067 nats, E74's. Weak lean: masks.
3. Own-output change at blocks 14, 21, 27, beside E74's Gaussian curve. No threshold.
4. Reads 2 and 3 as filed; read 1's stop no longer applies, its "fails" stands.

Rules: `results/predictions_amend1.json`, `amend1.swap_label`. Edge cases read: a NaN ratio gives "unread"; a value in a band gives "between". Cost: $0, ~1 h, same memory as run 50608. Amendment 1 of the two allowed.

**Not run:** other row pairings; which sites hold the information; a mask from Γ on a perturbed row.
