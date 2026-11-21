# P1.45 (E241): the causal test with gating at VPD

Card: `reports/plans/2026-11-20_plan_p1-45-card.md`. Rules: `experiments/gating_vpd/results/predictions.json`. Code:
`experiments/gating_vpd/run.py`, `score.py`; tests `tests/test_gating_vpd.py`; Slurm `cluster/gating_vpd.sbatch`.
Descriptive (user 2026-11-20 18:55): exploratory, not a definition change, not final; the result is reported either way.

**The user's request, in the Terms' words.** M is the 67M base model and F a fine-tune of it. H keeps M's released VPD
subcomponents (each U and V unchanged, Delta out) and changes only their per-input gates: frozen Gamma computes the
gates from F's own activations on the same window, and the CI-masked model runs on them. This is the re-gating
construction, route (b) of the route split in `GOALS.md`, the cell L01 of the re-gating share, here read in KL as
E57 and E228 read it (`masked_reads.route_cells_by` K01). The score is S = 1 - KL(F||H) / KL(F||M), each KL over the
whole vocabulary, averaged over the item's read positions and over its windows (ratio of means). S = 1 when H
reproduces F; S = 0 when H is as far from F as M is.

**One reading of "refit", flagged for the user.** Nothing is optimised here: "the gates refit to F" is read as Gamma,
frozen, re-reading F's activations, which is how the project has always built re-gating. If the user meant retraining
Gamma on F (E15's Gamma refit, `commute.gamma`, hours on an H100), that is a different and gated run; this one does not
do it.

**What runs.** Two jobs at once on the cluster's debug QoS. Each reads its do-nothing state first and stops on a miss.
- built: the do-nothing on the copy and the cue windows; the six built edits at E240's matched size (every update
  scaled toward M to new_thin's size, `kstar_vpd_norm_matched` `scale_factors` and `scaled`, unchanged); then the
  random updates on both window sets, three seeds each, in two arms: "size", a Gaussian at the same relative size
  in every matrix and new_thin's total (|dW|/|W0| = 0.01379), as asked, and "kl", the same direction scaled until
  KL(F||M) is twice the floor f this job measured there (E228's matcher).
- pairs: E232's do-nothing (its control state times 0); then the 68 states of E187's nine pairs, the larger side of
  each pair scaled by E187's c (E232's World), the smaller side as saved.

**Beside S, three numbers per item** (all from the same forwards): S00, the same score with M's own gates (so S - S00
is what refitting the gates adds); the floor f = KL(M || M's pieces at M's own gates), VPD's own reconstruction error
on those windows; and S_c = 1 - (KL(F||H) - f) / KL(F||M), S with that floor taken off (a side reading of the lead's,
not the user's score). An item is scored only when KL(F||M) > f: below it, even a gate refit as good as VPD's
reconstruction of M leaves S at or under 0.

**What each number should read.**
- do-nothing (both window sets, and E232's control): KL(F||M) = 0, F's gates equal M's exactly, KL(F||H) = f, not
  scored. f about 0.4 nats per position (E228 read 0.42-0.46 on 67M rows, `random_kl_share/results/states/*.json`
  .own.K.K0). Positive control: every gate open with Delta in reproduces M (KL about 1e-10, bar 1e-3 f); the gates
  computed from captured inputs equal `vpd.compute_ci`'s to bf16 resolution (bar 0.01).
- the built reuse edits (x2, x10, moved, rebuilt; 75 copy24 windows, every position): at their own sizes KL(F||M) is
  0.10 (x2), 0.12 (rebuilt) and 5.6e-12 (moved, which computes M's function) per position
  (`kstar_vpd/results/items/*.json` .kl_fm), smaller at matched size. Against a floor near 0.4 the lead expects all
  four unscored on S, so P1 and P2 likely not read; S_c carries them. A gate can only turn a piece down, so x2 and x10
  (one state at matched size: the copy neurons' writes times 1.84) are out of H's reach and should read S near S00.
- new_few, new_thin (512 cue windows, last position): KL(F||M) 10.2 and 10.3 at their own sizes; new_thin is at c = 1
  and its KL(F||M) must repeat E226's within 1e-4. Scored. S near S00 is the lead's expectation: H cannot add the
  lookup's new write.
- random, size arm: likely far under the floor (a Gaussian moves KL far less than a targeted edit of the same size).
  Random, kl arm: scored by construction. On E228's KL cells the KL-matched random changes gained more from Gamma's
  re-reading than the fine-tunes they were matched to (c3/cipher_s0: KL(F||H_M) - KL(F||H) 0.60 against 0.33), so the
  lead expects the random update may not read low.
- pairs: KL(F||M) 0.3-6.5 nats at the last position (E226 items), scored if over that position's floor.

**Filed predictions** (the user's two, and the same two on S_c; band = the largest split-half gap over the scored
items):
- P1: every scored reuse edit's S above both new edits' S by more than the band (holds / rival b at gap <= 0 / no
  call); not read if a new edit or every reuse edit is unscored. Prediction: holds.
- P2: on the copy windows, every scored random reading's S below the lowest scored reuse edit's S by more than the band;
  not read if either side has no scored reading. Prediction: holds.
- P1c, P2c: the same on S_c. Prediction: holds.
- D1 (descriptive): per pair, reuse side's mean S against the new side's; pairs with reuse above, of those read.
Edge cases run before filing (`tests/test_gating_vpd.py`, 4 pass): ties read rival b / fails; the do-nothing in place
of every item, an unscored new edit and missing items read "not read"; a degenerate band gives a strict order.

**Runtime and cost.** No GPU was used to write this; the estimate is from measured items of the same reads. E228's
read of the same KL cells (6 forwards, 3 Gamma passes) took 1.7 s on 75 rows (`random_kl_share/results/states/
c3_cipher_s0.json` .own.read_seconds); this read is 6 forwards and 2 Gamma passes, so about 1.5 s per 75 windows on
that pod, about 10 s per 512 windows, and twice that on an L40 (E232: the same search 45.3 s on the cluster, 22.6 s on
its pod), about 20 s per 512-window item. Built job: World and edits about 5 min (unmeasured; E232's World loaded in
under 2 min), stage 0 about 1 min, six items about 1 min, randoms 6 draws x (1 size read + about 18 match passes of
F's forward + 1 read) about 6 min: about 13 min. Pairs job: World 2 min, stage 0 1 min, 68 states x (20 s read + 2 s
load) about 25 min: about 28 min. Peak GPU about 8-10 GB (E228 7.75 GB, E226 10.1 GB), host about 7 GB (E226, E232).
Time limit 1 h 30 min each, $0.

**Not run.** Retraining Gamma on F (E15's recipe); H with M's remainder at mask 1 (only H_open, as a control); a
random update matched to each edit's own KL(F||M); the pairs' random controls; any wording change.
