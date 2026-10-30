# Plan (E55 C3): pruning revival, reverse fine-tuning and the probe on E58's twelve open-pair states

**Reuse-lead, 2026-10-31. Plan only; nothing filed, nothing run.** Adopted line: Fable's advice
`reports/plans/2026-10-31_advice_e55-after-c1b-FABLE.md`, option C (with D as a `decisions.md` row, written with this
plan). Inputs: C1b result `reports/analyses/2026-10-31_analysis_e55-c1b.md`; C1b recipes
`reports/plans/2026-10-30_plan_e55-c1b-refined-instruments.md` § 4 and `src/commute/revival.py`; E58
`reports/analyses/2026-10-30_analysis_e58-open-pair.md` (main run and Amendment 1), `experiments/open_pair/results/
{run,amend1}.json`; the Stage B offer in `reports/plans/2026-10-30_plan_regating-ground-truth.md` (note of 2026-10-30).
Words as in `GOALS.md` → Terms and the reuse-line rows of `decisions.md`.

**Open question for the user: none.** The carry label stays a reported column, as the re-gating plan's note of
2026-10-30 retired it as an agreement target. This plan does not score any instrument against it. One place where it
enters: the "label side" below is defined by its filed category (elicit "new", like the cipher). Making the graded
carry a scored target would change the reuse line's definition. That is the user's call, and this plan does not ask
for it.

## In plain words

E58 found one map, the elicit map, where VPD and behaviour disagree. VPD's readings say the fine-tune runs through
pieces the original model already had. The held-out keys say nothing learnt carries to keys never trained. This plan
asks whether two cheap methods that need no decomposition side with VPD or with behaviour on the same twelve saved
fine-tunes: pruning a few neurons to revive the old behaviour, and fine-tuning back to it. If they side with VPD, the
limit belongs to every instrument that reads mechanism. If they side with behaviour, VPD is the odd one out.

## 1. The request, restated in the Terms' words

- **States.** The twelve finished 67M fine-tunes of E58 and its Amendment 1, unchanged and read from disk: the elicit
  map, the rank-1 map, the copy map and the cipher, each trained with E54's recipe on 24 of the same 32 keys, seeds
  0-2. Each is a finished update δW on the 24 decomposed matrices (`experiments/open_pair/results/large/
  {elicit,cipher}_s{0,1,2}.pt`, `.../large/amend1/{rank1,copy}_s{0,1,2}.pt`).
- **Instruments, ported from C1b to the 67M.** Pruning revival (Jain et al. 2024, Alg. 3) and reverse fine-tuning
  read each state. The possession probe (Def 1) reads each map, as a reference only. "The old behaviour" is W₀'s own
  next token after the marker and each key, the rank-1 map's value (Fable's choice). No decomposition is read.
- **Beside them, already read and not re-run.** E58's re-gating share r, the stand-in ratio, steering's share S, the
  held-out carry x with its label, and the update size, all from `run.json` and `amend1.json`.
- **The question.** VPD's readings put the elicit map with copy and rank-1 ("old pieces"). The carry label puts it with
  the cipher ("new"). The granularity limit says VPD's readings cannot tell a new circuit built from old subcomponents
  from an old one. Is that limit VPD's alone, or shared by instruments that need no decomposition?

## 2. Design check (`/design`)

- **Goal served.** `GOALS.md` "If no simple Φ works": is post-training new capability or strengthening? The brief's
  Stage 1 asks that VPD and at least one non-VPD method be scored against a ground truth not set by construction. C1b
  showed that a set whose labels are set by construction cannot rank instruments. E58's states are the only 67M set
  with an open label.
- **Nearest tried.** E55 C1b (`TRIED.md`): the three instruments sort a set built so that P reuses and N is new, 21 of
  21, but they tie a probe that reads only the answers and update size within a recipe. E58 and Amendment 1 put r, the
  ratio and S on these twelve states. What this learns that those could not: whether the cheap methods carry over to the
  67M, and where they place the one disputed map.
- **Patch or step.** A step. It is the comparison that the user's priority (VPD against pragmatic methods) turns on,
  on labels that construction does not fix.
- **Decisions relied on.** Reuse-line definition row (Jain Def 1 and Def 2, plus the causal check); bounded g; the
  "option 2 then 1" row (pragmatic instruments refined and benchmarked); the E58 reading rows (granularity limit, carry
  a graded reported column). Departures: three ports of C1b's recipes to a model with no "scratch" model and no
  pretraining labels (§ 9), stated in the plan.
- **What changes next.** Both new instruments read the copy-versus-cipher pair and both put the elicit map with copy:
  the limit is shared. The line's claim then says so, and C2 (VPD on the counting anchors, gated) is put to the user
  framed as "does VPD match the cheap methods". Both put it with the cipher: VPD is the odd one out on this pair. That
  is flagged to the user at once, as it bears on the priority. Neither reads copy against the cipher: C1b's success
  does not carry over to the 67M, and on this model only the decomposition and steering read the pair. That is also
  reported. Mixed or "between": described, and no more runs on this pair; the next step goes to Fable.
- **Control first.** Stage 0 runs alone and stops the job on a miss (§ 5): sha256 and each state's gain reproduced,
  the zero update, importance pruning on W₀, the lens check, then elicit_s0's reverse fine-tune with an early stop.
- **Numbers.** Two scored readings per state (revival R, reverse steps s). One reference per map (probe q). One
  position p per column. Expected values are in § 4.
- **Class.** Approved class 4, the reuse line's named next step (Fable option C adopted by the lead; the re-gating
  plan offers these states) and class 1 (GPU passes of shared code on saved states). It costs $0 on the cluster and
  makes no new fine-tunes of the scored kind. No sign-off is needed. FYI to the user.

## 3. Instruments on the 67M

All read the same windows: the rank-1 map's trained-key windows,
`experiments/open_pair/results/large/amend1/windows_rank1_trained.pt`. They are E54's windows with W₀'s top-1 value
per key as the target, split 24/8 as E58. The `train` split (the forward fine-tunes' training windows) is used for
reverse fine-tuning. The `heldout` split has 75 windows, with fit half (even) and score half (odd) as `steering.Task`.
Stage 0 checks that the four maps' trained-key windows have identical tokens. fp32 throughout, as E58.

**The old behaviour.** For a model m, ℓ(m) is the mean over windows of the log-odds of the window's rank-1 value at the
answer position (`steering.logodds_at`). D = ℓ(W₀) − ℓ(state) on the score half is how far the fine-tune pushed W₀'s own
answer down. A state with D < 0.1 log-odds has nothing to revive: void for pruning. This covers the zero update and,
expected, the rank-1 states, which pushed it up.

| instrument | reading per state | refined how (from C1b) | high or low means "old" |
|---|---|---|---|
| **Pruning revival** | R = max over K ∈ {13, 40} of [(ℓ(top-K pruned) − ℓ(state)) − mean over 20 random K-unit removals of (ℓ(random pruned) − ℓ(state))] / D. Units are the 4 × 3072 MLP neurons (`steering.Net` site `mlp<l>`), zeroed at every position. That equals zeroing up_proj's row (GELU(0) = 0). Scored by Alg. 3's grad · w on up_proj's row j (and bias, if the module has one), the gradient of the rank-1 value's cross-entropy at the state, on the fit half. Read on the score half. The random draws are `default_rng([PRUNE_SEED, K])`, the same for every state | C1b's K 5 and 15 of 4,608 units scaled to 12,288 (0.11% and 0.33%). The reading is a share of D, not an accuracy (there is no accuracy to reach), with C1b's "minus random" | high |
| **Reverse fine-tuning** | s = the first checkpoint at which the held-out arithmetic-mean P(rank-1 value) (`finetune.eval_arm_a`'s `p_o`, all 75 windows) reaches W₀'s own value. Training is E54's recipe unchanged (`finetune.finetune_arm_a`: Adam, lr `anchors.FT_LR` 3e-5, batch 32, clip 1, KL on the other positions), started from W₀ + δW with the rank-1 table as target, seed = the state's seed. Checkpoints are `keydep.CIPHER_CHECKS` (every step to 30, then 100, 300). Before training it is read at step 0: already there → s = 0, void. Never there → 301 (the cap + 1) | One rate for every state (C1b), here the forward fine-tunes' own rate. No scratch divisor (§ 9: there is no 67M model that never had W₀'s behaviour, and p below is invariant to a common scale) | low |
| **Possession probe, reference only** | q per map = max over blocks l = 1-4 of the mean over the trained keys' held-out windows of [log P_lens,l(own value) − mean over the map's other distinct values of log P_lens,l(value)]. Here P_lens,l is W₀'s final norm and unembedding applied to the residual after block l (logit lens). Block 4 is W₀'s own output. It reads W₀ and the map only, never the fine-tune | Def 1's "linear readout trained on pretraining data" taken as W₀'s own unembedding. C1b's probe trained on the state's answers would read ≈ 1 on every map here: each state's trained-key answers are a lookup on the key token, which sits in the residual (§ 9) | high |

**Positions.** Each column is read only through where it puts the elicit map between two poles that VPD and the label
agree on. Copy is "old" to VPD and carries most, with the only "reuses" label. The cipher is "new" to both. With
seed means over non-void seeds:

p = (v̄_elicit − v̄_cipher) / (v̄_copy − v̄_cipher).

p near 1 puts elicit with copy, VPD's side. p near 0 puts it with the cipher, the label's side. Rank-1 is reported
beside (p with rank-1 as the pole), not scored: E58 found it a weak control on this split (18 distinct values).

## 4. Numbers and their expected values

| number | W₀ / zero update (do-nothing) | rank-1 states | elicit, copy, cipher | control |
|---|---|---|---|---|
| D, log-odds | exactly 0 → void | < 0 → void (pushed up by training) | > 0.1; size unknown, guess 1-6 (Amendment 1: at unseen keys the rank-1 word fell 0.3-0.6; trained keys should fall more) | — |
| pruning R | void | void | unknown, 0-0.5; direction filed copy > cipher | importance pruning on W₀ (sign −1, C1b's control): top-K lowers ℓ(W₀) below the random mean by ≥ 3 sd at some K |
| reverse s, steps | 0 → void | 0 → void | 1-30 expected (the forward fine-tunes needed 8-10 steps at this rate); direction filed copy < cipher | elicit_s0 first: reaches the criterion before the cap, and P at step 30 > P at step 0 |
| probe q, nats | W₀ only | rank-1 ≥ elicit | elicit ≥ 1 (values are W₀'s rank 2-20 picks, key-specific by selection); cipher within ±0.5 (all values under 1e-4); copy unknown | block 4 equals W₀'s own log-softmax (max abs ≤ 1e-5) |
| p, r (from files) | — | — | 0.880, single-seed range 0.854-0.923 | reproduces from `run.json`/`amend1.json` (CPU) |
| p, stand-in ratio | — | — | 0.879, range 0.828-0.919 | same |
| p, steering S | — | — | 0.684, range 0.586-0.759 | same |
| p, carry x (reported) | — | — | 0.378, range 0.310-0.440 | same |
| p, update size (baseline column) | 0 | — | void: copy minus cipher −0.04 against 3 × seed range 0.56 (matched by design: the cipher is read at the elicit update's norm) | same |
| p, D (functional distance, baseline column) | — | — | unknown; reported so that reverse fine-tuning reading distance alone shows | — |

The file values above were computed for this plan from `run.json` and `amend1.json` with jq and awk. They are the
rule's worked examples (§ 6).

## 5. Order, controls first, stop rules

One job script `experiments/open_pair_instruments/run.py`, stages in order. Each state's readings are written to
`results/run.json` as they finish, and a rerun resumes from them.

**Stage 0, alone. Exit 1 on any miss, before any scored state.**
1. sha256 of the 12 state files equal `run.json` / `amend1.json` `ft.sha256`. The windows files equal the recorded
   sha256. The four maps' trained-key windows have identical tokens.
2. Each state's raw gain G on its own map's trained-key held-out windows reproduces the recorded G within 1e-4
   (E58 Amendment 1's tolerance; it read within 1e-7).
3. The zero update: D = 0 exactly, pruning void, reverse s = 0.
4. Importance pruning on W₀ (sign −1, K 13 and 40): at some K, top-K lowers ℓ(W₀) by more than the random mean plus
   3 sd. A miss voids pruning (C1b's rule) and the job continues.
5. Lens check: block-4 logit lens equals W₀'s log-softmax at the answer position, max abs ≤ 1e-5. A miss voids the
   probe and the job continues.
6. **elicit_s0's reverse fine-tune, run first and alone.** Stop if P(rank-1 value) at step 30 is not above its step-0
   value (the loop does not revert), or if the run hits the cap. After the run the target weights equal W₀ exactly
   (`model_state` restored them). Stop on any difference.

**Stage 1.** Per state, in the order elicit, cipher, copy, rank-1, seeds 0-2: D, pruning R, reverse s, written after
each. The weights-equal-W₀ check runs after every reverse fine-tune. Then q per map.

**Score.** Pure arithmetic in the same job (§ 6), plus the file columns. Every scored number is also printed to the
log.

## 6. Claim, null, predictions

**Claim (one sentence).** Pruning revival and reverse fine-tuning, which need no decomposition, put E58's elicit map
with the copy map, where r and the stand-in ratio put it, and not with the cipher, where the carry label puts it. So the
granularity limit is shared by instruments that read mechanism, not VPD's alone.

**Null.** The cheap instruments do not place the elicit map with copy. Either they cannot read the copy-versus-cipher
pair on the 67M at all, or they put elicit between the poles or with the cipher.

**Rule (filed for every column; pure functions in `src/commute/revival_kv.py`, unit-tested on CPU before filing).**
1. **Reference gap, each instrument's positive control.** noise = the largest within-map seed range (max − min) among
   copy, cipher and elicit, floored at 0.05 for R (C1b's random-removal spread) and 1 step for s. gap = v̄_copy −
   v̄_cipher. "ok" iff d · gap ≥ 3 · noise, with d = +1 for R and q and −1 for s. "reversed" iff −d · gap ≥ 3 · noise.
   Else "no gap". Baseline columns with no filed direction (update size, D) use |gap|.
2. **Side**, only when the gap is "ok": "VPD" iff p ≥ 2/3, "label" iff p ≤ 1/3, else "between". Ties at 2/3 or
   1/3 go to the outer band. p > 1 is "VPD" and p < 0 is "label".
3. **Firm.** The side holds only if p from every single-seed triple (elicit_i, cipher_j, copy_k, non-void seeds only)
   falls in the same band. A triple with a zero denominator counts as outside. Otherwise the side reads "between".
4. **Verdict** over pruning and reverse fine-tuning: "shared" iff both read "VPD"; "VPD alone" iff both read "label";
   otherwise "not settled", with each instrument's reading named.

**Edge cases, run before filing (the code writer re-runs them as tests).**
- An empty pool: a map with fewer than 2 non-void seeds makes the column "void". The rank-1 map is void for R and s by
  design and is not a pole.
- A zero denominator: gap = 0 fails rule 1 ("no gap"). A triple with a zero denominator fails rule 3.
- Ties: at the band edges, rule 2. Discrete s with identical seeds: range 0, so the 1-step floor applies.
- Capped s: 301. A map capped on every seed has range 0, so the floor applies.
- Worked on the known columns: r → ok (gap 0.610 ≥ 0.087), p 0.880, firm, "VPD". Ratio → ok (0.700 ≥ 0.164),
  0.879, firm, "VPD". S → ok (0.461 ≥ 0.205), p 0.684, triples 0.586-0.759, so "between". Carry → ok (0.822 ≥ 0.397),
  0.378, triples 0.310-0.440, so "between": the graded carry is not itself on the label's side by this rule, which
  is why the label side is defined by the label's category (top of this plan). Update size → "no gap" (0.042 < 0.56).

**Predictions** (each with a lean. "Blind to", control and noise floor follow each).

| id | prediction | measurement and threshold | lean | blind to | positive control; noise floor |
|---|---|---|---|---|---|
| C0 | Stage 0 passes | § 5 items 1-6 | 85% | — | — |
| P1 | Each new instrument reads the copy-versus-cipher pair: gap "ok" | rule 1 | R 45%, s 55% | R: changes in attention (29-31% of each update's squared norm, E56) and changes spread over more than 40 neurons. s: anything finer than one step, and structure as against distance from W₀ (D is reported beside it for this) | the gap itself (copy and cipher are the agreed poles); R's floor 0.05 with 20 random draws; s's 1 step |
| P2 | Pruning puts the elicit map on VPD's side | rules 2-3 on R | VPD 25%, between 15%, label 5%, no gap or reversed 55% | as P1 | as P1 |
| P3 | Reverse fine-tuning puts the elicit map on VPD's side | rules 2-3 on s | VPD 35%, between 10%, label 10%, no gap or reversed 45% | as P1; a VPD reading with p_s close to p_D is read as distance, described | as P1 |
| V | Verdict "shared" | rule 4 | shared 15%, VPD alone 3%, not settled 82% | — | — |
| ref | Probe q: cipher within ±0.5, elicit ≥ 1, p_q reported | § 3 | 70% | the fine-tune entirely: q reads W₀ and the map | lens check; cipher as the null |

**Why these leans.** The elicit values are W₀'s rank 2-20 picks, so the elicit state is functionally closest to W₀.
Reverse fine-tuning should then revert it fastest, which reads "VPD" if copy also reverts faster than the cipher.
Copy pushes the key token, which W₀ rates about 6e-5, and could revert as slowly as the cipher, giving no gap. Pruning
on the 67M is untested. Ten Adam steps spread the update over every matrix, and a third of it sits in attention, so a
revival from 40 neurons may be near zero on every state. The likeliest outcome is "not settled". The run is still
worth its $0: P1 alone says whether C1b's instruments carry over to the 67M at all.

## 7. Code: what goes where

- **New `src/commute/revival_kv.py`** (about 150 lines): `old_logodds` (ℓ via `steering.read` and `logodds_at` with
  the rank-1 table), `unit_scores` (grad · w on each block's up_proj under `Net.at(dW)`), `prune_readings` (zeroing
  edits on site `mlp<l>`, the same structure as `neurons.neuron_edit`), `reverse_steps` (calls
  `finetune.finetune_arm_a` unchanged with `keydep.CIPHER_CHECKS`, reads `evals[*]["p_o"]`), `lens_prior` and the
  pure rules `reference_gap`, `position`, `side`, `verdict`. It is shared because the Qwen line and any later 67M set
  will call it. Tests: `tests/test_revival_kv.py`, CPU, rules only.
- **New `experiments/open_pair_instruments/run.py`** (about 200 lines: stages, resume, file columns) and `plot.py`.
  The run refuses to start unless `results/predictions.json` is committed unchanged (as `open_pair/run.py`).
- **Called, not changed:** `steering` (Net, Task, read, logodds_at), `heldout_keys.pool_of`, `keydep` (make_ctx,
  CIPHER_CHECKS, sha256), `finetune.finetune_arm_a`, `anchors.FT_LR`.
- **Nothing moves.** `src/commute/revival.py` stays as C1b ran. It is minGPT- and PCFG-specific (`pcfg.ANS`,
  `answer_loss`), and its callers are frozen scripts: `experiments/pcfg_anchors/c1.py`, `c1b.py`, `plot_c1.py`,
  `plot_c1b.py`, and `tests/test_revival.py`. The new module copies none of its code. Only the rule's shape (top minus
  random) is shared, and that is three lines of arithmetic.
- **Review.** One fresh-Opus review of the diff before the job, as C1b's (Codex is out of quota). The run is $0, so
  the review is not required, but the measurement code is new.

## 8. Compute, cost, outputs

- **Route.** MATS cluster `debug` QoS, L40, account c11, `cluster/open_pair_instruments.sbatch`, `--time=00:30:00`,
  through `/pod-runner` (the lead spawns it). Estimate: model load 1-2 min; per state, pruning (one gradient pass, 42
  forwards on 37 windows) under 30 s, reverse fine-tune (300 steps, 32 evaluations, about 3.6 GB of held deltas in host
  RAM, as E58's cipher fine-tunes) 1-2 min. That makes 15-30 min for 12 states. **One job, a second if the
  resume is needed. $0.** Not gated: $0, under 3 h, within the session cap.
- **Saved** (all small, committed): `experiments/open_pair_instruments/results/run.json` (per state D, per-K top and
  random ℓ, R, the reverse p_o curve and s; per map q per block; every column's gap, p, triples, side; the verdict),
  `results/predictions.json` (**DRAFT** in this commit, status field says so; filed only after the rule's CPU tests), `figures/positions.png` (each column's elicit position on its own copy-to-cipher axis,
  single-seed triples as a range, the 1/3 and 2/3 bands; one figure for the claim), log
  `cluster/logs/commute-open-pair-instruments-<job>.out`. No state files are written: the reverse deltas are discarded,
  and only their curves are kept.
- **After.** `/writeup` report `reports/analyses/<date>_analysis_e55-c3-open-pair.md`, a section in
  `notebooks/E55_pcfg_anchors.ipynb`, a `TRIED.md` row, a `PENDING.md` line.

## 9. Departures from the literature

- **Pruning target.** Jain et al. revive a pretraining capability with a labelled task. Here the "capability" is W₀'s
  own top-1 next token after marker and key, and revival is read as the share of its log-odds drop restored. K is
  scaled from C1b by unit count, and there is one number per state instead of a curve.
- **Pruning score.** Alg. 3's grad · w on the unit's input-weight row, as C1b. The zeroing is done as an activation
  edit at the GELU output, which is identical for GELU(0) = 0.
- **Reverse fine-tuning.** Jain et al. reverse at a rate below the fine-tune's, against a scratch model trained on the
  target. Here it runs at the forward rate (E54's recipe unchanged, so the recipe is not a new setting) with no scratch
  model. The 67M has no model that never had W₀'s behaviour, and p is invariant to a common scale. The KL term anchors
  the other positions to the state, not to W₀, because `finetune_arm_a`'s reference is its starting model. The
  target is W₀'s top-1, not its whole distribution.
- **Possession probe.** Def 1's readout is W₀'s unembedding at each block (logit lens, nostalgebraist 2020), not a
  probe trained here. The control is the map's other values, not Hewitt and Liang's control task. A tuned lens
  (Belrose et al. 2023) is not run.
- **Positions p, the bands, firmness and the verdict** are this project's. The papers plot curves.
- **Steering, r, the stand-in ratio and the carry label** are E58's, unchanged (their departures are in its report).

## 10. Not run

- The interchange-patch label (Stage B), the piece-level ground truth reading A asks for. Not built.
- The 96-key states. Only seed 0 of elicit, rank-1 and copy is saved, in `large/amend3/`, and there is no saved
  cipher at 96 keys, so no pole. Also not run: the shuffled map, the prior-matched cipher and Amendment 7's maps.
- VPD on the counting anchors (C2, gated, the user's call). A second base model or task family.
- Attention-head pruning; K above 40; reverse rates other than 3e-5; KL-to-W₀ reverse targets; a tuned lens.
- Scoring the graded carry, or the rank-1 map, as a target or a pole.
- Any edit to E58's or C1b's files or predictions.

## Hand-off

The next session reads this plan, then `src/commute/revival.py`, `steering.py`, `neurons.py`, `finetune.py`
(`finetune_arm_a`, `_run`), `experiments/open_pair/amend1.py` and the draft
`experiments/open_pair_instruments/results/predictions.json`. It writes `revival_kv.py` and its tests and runs the
rule's edge cases. It commits the predictions (status from DRAFT to filed), writes `run.py` and the sbatch, gets the
fresh-Opus review, then the lead spawns `/pod-runner` with stage 0 first.
