# Plan: the C3 step ladder (E55, reuse line)

2026-10-31. Advice item 2 of `reports/plans/2026-10-31_advice_reuse-next-FABLE.md`, adopted by the lead, with the lead's
two additions from `reports/reviews/2026-10-31_review_e55-c3-check.md`: a re-read of C3's states at larger K first, and
the 3-sd bar on the poles' own spread. One $0 cluster `debug` stage, resumable. Predictions
`experiments/open_pair_instruments/results/predictions_ladder.json`. Code `experiments/open_pair_instruments/ladder.py`
(`run.py --stage ladder`), rules `src/commute/revival_kv.py` (`pole_gap`, `first_ok_step`, added), tests
`tests/test_revival_kv_ladder.py`.

**The request in the Terms' words.** C3 ported Jain et al.'s two instruments, pruning revival R and reverse
fine-tuning's half-life t½, to the 67M open-pair maps, and neither separated the rank-1 map (old by construction) from
the cipher (new by construction) on 10-step updates. This stage asks at what update size, if any, they pass that
positive control: fine-tunes of rank-1, the cipher and elicit, three seeds each, read at steps 10, 30, 100 and 300.
Elicit's position is reported only. The stage closes the open-pair set as a test of the instruments; it is not a VPD
benchmark.

## Design check

- **Goal served.** GOALS' "if not, is it because post-training teaches new capabilities": the brief's item of scoring
  VPD and a non-VPD method on one truth needs a non-VPD method that works on real states. This finds whether the
  ported instruments ever do on this set. Indirect: it serves C2's comparison, which uses the same instruments.
- **Nearest tried.** E55 C3 (`TRIED.md`): one update size, 10 steps, K 13 and 40; both instruments "no gap". New here:
  update size and K vary, the two things C3's "Not run" names.
- **Patch or step.** A patch on C3: refinement proposals 1 and 2, and the review's K lead. Advice caps its spend at
  this one job.
- **Decisions.** Follows "E55 after C1" (refine the pragmatic instruments) and "update size within a recipe" (norm and
  D per state, reported). One departure from C3's rule 1, filed: noise from the two poles only (lead, 2026-10-31).
- **What changes next.** A pass step tells C2 which update size the instruments need. No pass at any step or K means
  the port, not the update size, is at fault, and C2 should not lean on these instruments as filed. A pass that puts
  elicit at the **new** end goes to the user at once (advice: VPD the odd one out).
- **Control first.** Part A runs alone first: the zero update (D exactly 0), then C3's 32 saved states re-read, with
  C3's K 13 and 40 required to reproduce `run.json`. Part B's first fine-tune is rank-1 seed 0, and its step-10 state
  must equal E58's saved state and give C3's R and t½. Then elicit seeds 0-2 the same way. Any miss stops the job.
  The ladder fine-tunes are E58's recipe unchanged (`keydep.finetune`), so the difference from the reference's end state is
  only the read step; the cipher is read at the same steps as the others, not norm-matched as in C3 (8-9 steps).
- **The numbers and what each should read.** Below.

## Claim, null, predictions

- **Claim.** Pruning revival or reverse half-life separates rank-1 from the cipher, in the filed direction, by at
  least 3 pole seed sd at some step from 10 to 300 (or, on C3's states, at some K up to 320).
- **Null.** No step and no K passes: the instruments, as ported, cannot read old against new on these maps at any
  update size the recipe makes.
- **Rule** (`pole_gap`): noise = max(sd rank-1, sd cipher, floor), gap = mean rank-1 − mean cipher; "ok" iff
  d · gap ≥ 3 · noise (d = +1 for R, −1 for t½), "reversed" iff −d · gap ≥ 3 · noise, else "no gap"; "void" under 2 values
  per pole, t½ "censored" if a pole seed never halves before the 3000-step cap. R's floor is the largest
  random-removal spread among that step's pole states; t½'s is 1 step (C3). Pass step = the smallest step reading "ok".
  Edge cases run in `tests/test_revival_kv_ladder.py`: an empty pole, a one-element pole, all None and nan, gap exactly 0
  with zero noise, a gap exactly on the bar and just under, the floor binding, both directions and undirected, elicit's
  spread ignored, censoring on a pole against elicit only, the top-guess curve's sign crossing, a pass step with no
  "ok" and with steps out of order.

| id | prediction | lean |
|---|---|---|
| C0 | zero update, 32-state re-read, six step-10 checks, every W₀ restore pass | 85% |
| PK | C3's states, R max over K 13-320: "ok" | 25% |
| PK_trend | the K 320 gap above the K 40 gap | 60% |
| PR | R "ok" at step 10 / 30 / 100 / 300 | 15 / 20 / 25 / 30% |
| Pt | t½ "ok" at step 10 / 30 / 100 / 300 | 5 / 10 / 10 / 15% |
| Pt_reversed_10 | t½ "reversed" at step 10 | 45% |
| Ppass | neither instrument passes at any step | 55% |

**Reported, not scored.** Elicit's position on R and t½ per step is never a verdict, because the axis cannot give one.
W₀'s prior places elicit at 0.92 whatever the step, so filed rules 5-6 would relabel any old-end reading "distance and
prior", and "shared" is out of reach (review item d). Also reported: R's gap at each K; D, the update norm and held-out
P(value) per state; t½ / D with its pole gap (C3's unfiled lead); and the top-guess column. That column is D_top, the
mean drop in log p of W₀'s top guess, with its half-life t_top. Rank-1 raises that guess, so its D_top is negative and
t_top reads how fast reverse fine-tuning undoes the raise (review item d's caveat).

## Expected values

- **Zero update.** D = 0 exactly.
- **Part A.** D and the K 13 / 40 shares equal `run.json`. R rises with K for every map, from C3's 0.39-0.65 at 40.
  Random removal stays under 0.02 of D.
- **Step 10.** Rank-1 and elicit as in C3: R 0.46 and 0.51, t½ 13.3 and 15.7, D 8.5 and 10.1. The cipher is one or two
  steps past C3's, so D is about 5-6 and t½ about 10-12. With pole-only noise, C3's wrong-way t½ lean of about 4 pole
  sd may read "reversed".
- **Steps 30-300.** E58's logs show held-out P(value) at 0.98 by step 10 and 0.999 by 300. The largest per-matrix
  relative change only doubles, from 0.014 to 0.027. So D should grow by a few nats, the norm roughly double, t½ grow
  2-5 times by step 300 (a guess), and R move little.
- **Jain §5** gives no pass step in comparable units. Its fine-tunes run at ηPT/10 or ηPT/100 to convergence, and reverse
  fine-tuning revives "in a few steps". Any pass is therefore expected at steps 100-300.

## Settings and their sources

- **Fine-tune.** `keydep.finetune`, E54/E58's recipe: Adam, lr 3e-5, 300 steps, checkpoints 10/30/100/300, seeds 0-2.
- **Pruning.** C3's `prune_readings` and `revival` with n_random 20 and seed 0. K = 13, 40 (C3's) plus 80, 160 and 320
  (review lead).
- **Reverse fine-tuning.** C3's `ReferenceConfig`: Adam, lr 3e-6, batch 32, clip 1, the state's seed. The cap is 3000
  in place of 1000: ten times the longest forward run, which is what a tenth of the rate needs to retrace it.
- **D_void.** 0.978, from C3 filing B.

## Run, time, memory

`sbatch --job-name=commute-e55-c3-ladder cluster/open_pair_instruments.sbatch ladder`, from the clone root, after a code
review. On exit 3, resubmit the same line; finished states are skipped, and a fine-tune cut mid-way reruns exactly.
These times are estimates, not measured:
- part A takes ~3 min (32 states, 105 pruned forwards each);
- the 9 fine-tunes take ~12 s each (E58's logs: 300 steps in 11 s);
- reverse runs at steps 10-30 take seconds;
- at steps 100-300 a reverse run takes up to 3000 steps at ~0.03-0.06 s each (from C3's ~1 s per state);
- expected total 15-40 min in 1-2 jobs, worst ~4 jobs if every 100/300 state hits the cap.

Host RAM ~5 GB (four deltas of 113 MB, the model and its reference), within the sbatch's 24G; GPU under 10 GB. Nothing
is saved but `results/ladder.json`, a few MB at most, mostly reverse curves. If it passes 1 MB, the curves move to
`results/large/` at analysis time.

**Pre-mortem.** Each risk with its guard:
- The step-10 retrain does not match E58: stop, guarded by the control.
- A different card shifts the readings: tolerances of 1e-3 on shares and 0.05 steps on t½.
- Censoring at 300 steps: the cap is 3× C3's, and a censored pole is reported as "censored", not "no gap".
- The reported top-guess curve could prolong runs: it is bounded at 3× κ's halving step.
- The deadline cuts a reverse run: exit 3 and resume.

## Not run

Copy at the ladder steps; seeds 3-7 on the ladder; states matched on D; reverse rates other than a tenth; reverse
*training* targets other than KL to W₀ (the top guess is only read); W₀'s ranking target; attention-head pruning; VPD's
r and stand-in ratio at the ladder steps; any change to C3's or E58's files or predictions; the planted wrapper (C3's
passed, and the code path is unchanged).
