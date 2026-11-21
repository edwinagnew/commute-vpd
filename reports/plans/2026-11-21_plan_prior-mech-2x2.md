# Plan: prior x mechanism 2x2 on the 67M key tasks (cell P2.54, E249 run 1, run 2 code TBD)

**In plain words.** On our standard tasks, the fine-tunes that reuse the base model's machinery are also the ones whose
answer the base model already found likely, so any reading of the base prior looks as good as the re-gating share.
This cell separates the two. It labels each fine-tune by an intervention: does switching off copy head (2,4) remove
its gain? Then it builds the missing corner, a copy fine-tune whose answer the base finds very unlikely, and reads
every cheap proxy on the four corners. The idea (user, via whip-lead 2026-11-21) is that the share follows the
mechanism, while most cheap alternatives follow the prior.

**Restated in the Terms' words.** Mechanism label = the share of a fine-tune's gain over M lost when head (2,4) is
mean-ablated, net of M's own fall (E62's measure). This is a mechanism label for copying only. It is not the reuse
definition under revision in P1.1, and no reuse verdict is given. Corners: copy (high prior, through the head),
prior-matched cipher (high, not), low-prior copy (low, through the head; built in run 2), cipher (low, not).

## Design check (/design)
- **Goal served:** claim 2 of the paper (the cheap proxies against the mechanistic definition), the new wording "the
  share follows the mechanism, alternatives follow the prior" (`~/commute-data/agents/compare-lead/note-2026-11-21-whip-claim2-idea.md`).
- **Nearest tried:** E237 (key grid) read r on the five key tasks at one size, and r put prior_cipher with cipher at a
  1.6x prior gap. That is the high-prior row, but it is labelled by construction, not by intervention. E62 labelled
  its copy map with this same ablation (0.44-0.56). E151 tried built pairs where prior and steering disagree and
  stopped at stage 0. New here: an intervention label on the saved key states, and a low-prior row.
- **Patch or step:** a step. It supplies the evidence claim 2 needs, which no item has had so far.
- **Decisions relied on:** the E62 ablation and its labels (share_old_min 0.5, share_new_max 0.2); E237's common size
  and random construction. No departure.
- **Adopted definition unchanged:** yes. No rule is added to a reuse verdict, and the head label is reported as a
  mechanism label.
- **What changes what we do:** if r follows the mechanism and steering or the prior do not, claim 2 stands as worded.
  If r follows the prior, claim 2 falls back to "existing tools cannot tell reuse from new at fine-tune size".
- **Control first:** run 1 starts with M's own copying on E52's repeated random tokens. Top-1 must fall under 0.3
  with (2,4) off and stay over 0.7 (median) with a random head off. A zero update must read F = M exactly, and E62's
  six states must repeat their saved gains. Every one of these exits 1 on a miss. Run 2 starts with the prior path
  (E133's copy prior must repeat) and holes_fill's readers (E147's shares on copy_s0 must repeat), then the
  construction check, all before any training.
- **Numbers and expected values:** in the two predictions files (`experiments/prior_mech_2x2/results/predictions.json`,
  `predictions_stage2.json`); summarised below.
- **Saved for audit:** one JSON per state per run, the selected windows and the three new updates in bf16 under
  `results/large/`.

## Run 1: head (2,4) off on saved states (E249; no training; ~6 min, cluster debug L40, $0)
Run 1 reads E237's 15 key states at E237's c and E62's six states at their saved size. For each it reads the share
lost with (2,4) off and with each of E62's five random single heads off (the null), plus M's fall per task.
Ablation: mean ablation with E62's saved base-weight means, not zeroing. Zeroing also removes the head's constant
output, which moves every task off its usual input and could give cipher a share from that shift alone. Mean
ablation is also what E62 measured (0.8247 -> 0.0095). Fable's advice, acted on.
Expected values (filed):
- copy ~0.5 (the cell's figure; range 0.2-0.8);
- cipher and prior_cipher ~0 (their answers are not in the window);
- rank1 and elicit ~0-0.1 (their answers come from the key's identity, not from a token in the window);
- E62 copy ~0.5 and its lookup twin ~0 (repeats);
- M's fall under 0.5 nats on the key tasks and 1-4 nats on E62 copy.

**A doubt is filed with the expected values.** The key copy task is "M x -> x": it repeats the current token, and no
earlier x is there to match. A prefix-matching head need not carry it. If copy reads under 0.2, the high-prior
reused corner cannot be labelled by this head (the card's stop).

## Run 2: low-prior copy and the 2x2 (code `train.py`; ~10 min, cluster debug L40, $0, not a pod)
- **Construction.** The cell allows a new cue token or the lowest-prior copy windows; I chose the lowest-prior
  windows. Same keys, marker, table and recipe as E147's copy, so every reader runs unchanged. E62's new marker is
  not low prior on this scale (log10 -3.50, above key copy's -4.23).
- **Selection.** A pool of 20,000 windows drawn by E58's rule on E12's Pile rows. On each split, the longest
  lowest-prior run whose geometric mean is at or under the bar 10^-5.73 (1.5 decades under copy; cipher sits at
  -5.86), thinned evenly to E147's counts (1214 train, 286 read rows).
- **Stop before training** if there are fewer than 256 train rows, 64 read rows or 48 keys, or if the prior is over
  the bar.
- **Training.** E147's call (`keydep.finetune`, step 30), seeds 0-2, scaled to E237's common size. One Gaussian random
  update per seed at its per-matrix norms times the same c (E237's `finetune.random_direction`, not E241's
  `random_dw`, so it matches E237's random column).
- **Readings.** The head share on the new states, then on all four corners plus rank1 and elicit: r (E228), log10
  prior (E133's path), size (E187), erank, pooled and per-key steering, TopK neuron shares k64 and k matched (all from
  holes_fill, on every key state at the common size, so elicit's steering is read too).
- **Filed rules.** Per proxy, the mechanism effect is ((copy + low copy) - (both ciphers)) / 2 and the prior effect
  is ((copy + prior_cipher) - (low copy + cipher)) / 2. The call is the larger of the two if it is over the band
  (the proxy's largest seed range over the corners, this job).
- **When nothing is called.** The call needs the low copy to lose at least 0.2 of its gain with (2,4) off, copy to
  pass run 1's P1, and both ciphers to pass its P2.
- **Expected calls.** r: mechanism (low copy ~0.7). Prior: prior, by construction. Size: no call, by construction.
  Per-key steering and neurons: no call. Pooled steering: no filed expectation. It follows the mechanism on the
  high-prior pair (0.74 vs 0.14), and this is the cell where it can come apart from r.
- **E62's six items** are read in run 1 and shown beside the corners, labelled with their own (larger) update size.
  They are outside the effects.

**Cost basis (measured items x count; neither run has run yet):**
- model load ~40 s (E237 job 62504);
- one share read 10-11 s (E237 stage1 read_seconds);
- E147's 300-step fine-tune ~9 s (job 57955);
- holes_fill plus E136 plus E98 on one state ~17 s (job 57961: 24 states in 12.8 min).

Both runs fit debug's 30 min. The ~$5 and ~$30 pod figures in the brief are not needed.

## Run 3: the k-head label (code `khead.py`, pending the user's yes; ~9 min, cluster debug L40, $0)
Runs 1-2 gave a weak label. Key copy lost 0.25 of its gain with (2,4) off, and the low copy lost 0.22. One random head,
(1,1), read 0.18-0.27 on the low copy in every seed. It is M's previous-token head (score 0.67): it also costs the
ciphers 0.10-0.15, and alone it takes M's copying from 0.82 to 0.01.
- **Arms.** M's top k heads by E62's prefix score, switched off together, k = 1, 2, 3, 5: (2,4), then (2,3), (2,2),
  then (3,2), (3,4). The same mean ablation and the same share and rows as run 1. One more arm: (2,4) and (1,1)
  together, to show that set costs the ciphers too, so it is not a copy-specific label.
- **Null.** k heads drawn from outside the top 5 and outside (1,1), 18 heads in all. All 18 single heads at k = 1;
  20 distinct seeded draws at k = 2, 3, 5. Draws holding (1,4) (rank 6, the one pool head over E62's chance score)
  are reported apart and kept out of the null band.
- **States.** All 27 saved items of runs 1-2: E62's 6, E237's 15 key states, E250's 3 low copies and their 3 random
  updates.
- **Controls, first, each exiting on a miss.** M's copying with (2,4) off must pass E62's control and repeat 0.0095.
  The k = 1 shares must repeat the saved (2,4) shares on all 24 states, within twice a floor measured in the same job
  (batch 64 against 32).
- **Filed expectations.** Copy and low copy lose 0.5-0.8 at k = 3, and E62's copy about 0.8. Both ciphers, rank1 and
  elicit stay under 0.05 at every k. The clean null band stays under 0.1. The pair arm costs the ciphers over 0.05
  (expected 0.10-0.20). Copy under 0.4 at k = 5 is the stop: the copy does not run through the heads ranked high on
  M's copying.
- **The 2x2 call again** uses run 2's rule on the k-head share at the best k, the k where E62's copy loses most.
  The best k is set by the control map outside the 2x2.
- Time: E249's job read 196 head-set reads in 34 s. Run 3 reads about 3,150, which gives about 9 min.

## Not run
- A second intervention for copies that bypass (2,4) (path patching; run 3 covers multi-head ablation).
- A low-prior copy built behind a new cue token.
- KL share and R at VPD on the new corner (cut by user priority after E237).
- Matched random updates for the key states in run 2 (E237's random band stands).

Hand-off: run 3 is `khead.py` and `cluster/prior_mech_2x2_khead.sbatch`, card draft
`~/commute-data/scratch/compare-lead/khead/card.md`.
Run 1 is `experiments/prior_mech_2x2/run.py` and `cluster/prior_mech_2x2.sbatch`. Run 2 is `train.py` and
`cluster/prior_mech_2x2_stage2.sbatch`, after run 1 (`--dependency=afterok`). Its card draft is
`~/commute-data/scratch/compare-lead/prior-mech-2x2/card.md`.
