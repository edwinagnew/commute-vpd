# Plan (E55 C3): pruning revival, reverse fine-tuning and the probe on E58's open-pair states

**Reuse-lead, 2026-10-31. Plan only; nothing filed, nothing run. Revised after the independent review
`reports/reviews/2026-10-31_review_e55-c3-plan.md` (verdict "redesign"); every item is answered in "Changes after
review" at the end.** Adopted line: Fable's advice `reports/plans/2026-10-31_advice_e55-after-c1b-FABLE.md`, option C
(with option D as a `decisions.md` row, written with the first version of this plan). Inputs: C1b result
`reports/analyses/2026-10-31_analysis_e55-c1b.md`; C1b recipes `reports/plans/2026-10-30_plan_e55-c1b-refined-instruments.md`
§ 4 and `src/commute/revival.py`; E58 `reports/analyses/2026-10-30_analysis_e58-open-pair.md` (main run and
Amendment 1) and `experiments/open_pair/results/{run,amend1}.json`; the Stage B offer in
`reports/plans/2026-10-30_plan_regating-ground-truth.md` (note of 2026-10-30). Words as in `GOALS.md` → Terms and the
reuse-line rows of `decisions.md`.

**How the review changes the question (read this first).** Fable's option C asked whether the cheap instruments side
with VPD, which puts the elicit map with copy as "old pieces", or with the carry label, which calls it "new" like the
cipher. The review showed that framing cannot be settled. The one map that is old by construction is the rank-1 map,
which only strengthens W₀'s own answer. The label calls it "neither" on all three seeds, so the label disagrees with
construction there just as it does on elicit. A result "elicit sits with copy" therefore could not tell "the limit is
shared" from "the label reads rule generalisation, not reuse". This version reads every column on an axis whose two
ends are fixed by construction: the rank-1 map at the old end and the cipher at the new end. The question becomes:
**do instruments that need no decomposition place the elicit map near the old end, as r and the stand-in ratio do
(p 0.88 and 0.79)?** Copy's position and the carry label (graded x, and its category) are reported beside it, not
scored. One consequence should be said plainly: on this axis the graded carry itself places elicit at p 0.69. The
behaviour that disagrees with VPD is the label's cut at 0.5, which also puts rank-1 below it. So "the label reads
rule generalisation, not old versus new pieces" is now an outcome this run can show, named in § 2.

**Open question for the user: none.** The carry label stays a reported column (the re-gating plan's note of 2026-10-30
retired it as an agreement target). Making it a scored target would change the reuse line's definition. That is the
user's call, and this plan does not ask for it.

## In plain words

E58 found one map, the elicit map, on which VPD says the fine-tune runs through pieces the original model already had.
The held-out keys say nothing learnt carries over. This plan asks two cheap methods that need no decomposition where
they put the elicit map. Pruning a few neurons tries to bring the original model's answers back. Fine-tuning back
toward the original model times how fast it returns. Each is read between a map that is old by construction and one
that is new by construction, on eight seeds. If the cheap methods also put elicit near the old end, the blind spot
belongs to every method that reads mechanism, not to VPD alone.

## 1. The request, restated in the Terms' words

- **States.** Each state is a finished 67M update δW on the 24 decomposed matrices, E54's recipe on 24 of the same 32
  keys. There are four maps: elicit, rank-1, copy and the cipher. Seeds 0-2 are E58's and Amendment 1's, read from
  disk (`experiments/open_pair/results/large/{elicit,cipher}_s{0,1,2}.pt`, `.../large/amend1/{rank1,copy}_s{0,1,2}.pt`).
  Seeds 3-7 are new fine-tunes with the same recipe and code, 20 states. The cipher is read at the same seed's elicit
  update norm, as in E58. Each new state also gets E58's columns.
- **Instruments, ported from C1b to the 67M.** Pruning revival (Jain et al. 2024, Alg. 3) and reverse fine-tuning read
  each state. "The old behaviour" is W₀'s whole next-token distribution at the answer position: κ(m) = KL(p_W₀ ‖ p_m),
  averaged over windows. The possession probe (Def 1) reads each map, as a reference only. No decomposition is read by
  any instrument.
- **Beside them.** E58's re-gating share r, the stand-in ratio, steering's S, the carry x with its label, and the update
  size. New baseline columns: the distance D = κ(state); the prior, W₀'s mean log-probability of the map's value
  (read from W₀ alone); and the MLP share of ‖δW‖².
- **The question.** On the rank-1-to-cipher axis, where do the cheap instruments place the elicit map, compared with r
  and the ratio? Equivalently: is the granularity limit VPD's alone?

## 2. Design check (`/design`)

- **Goal served.** `GOALS.md` "If no simple Φ works": new capability or strengthening? The brief's Stage 1 asks that
  VPD and at least one non-VPD method be scored against a truth not set by construction of the scored question. C1b
  showed that a set whose labels come from construction cannot rank instruments. E58's elicit map is the 67M case where
  VPD and behaviour disagree.
- **Nearest tried.** E55 C1b (`TRIED.md`): the three instruments sorted a set built as P (reuses) against N (new), 21
  of 21. They tied an answer-only probe and update size within a recipe. E58 and Amendment 1 put r, the ratio and S on
  the twelve states. New here: whether the cheap methods carry over to the 67M, and where they place the disputed map
  on an axis fixed by construction.
- **Patch or step.** A step: the comparison the user's priority (VPD against pragmatic methods) turns on.
- **Decisions relied on.** The reuse-line definition row (Jain Def 1 and 2, plus the causal check: pruning and reverse
  fine-tuning); bounded g; "option 2 then 1"; the E58 reading rows (granularity limit; carry graded and reported); the
  2026-10-31 update-size row (update size is a baseline column). Departures in § 9: the old behaviour as W₀'s
  distribution, the logit lens as Def 1's readout, and no scratch model.
- **What changes next.**
  (a) Both instruments read firm "VPD" (or one does and the other cannot read the axis): the limit is shared by
  instruments that read mechanism. The line's claim says so, and C2 (VPD on the counting anchors, gated) goes to the
  user framed as "does VPD match the cheap methods".
  (b) Both read "label": on this pair VPD is the odd one out. This is flagged to the user at once, because it bears on
  the priority.
  (c) Either reads "distance": it tracks how far the state moved from W₀, not structure. It is reported as such.
  (d) Neither reads the rank-1-to-cipher gap: C1b's success does not carry over to the 67M. On this model only the
  decomposition reads the pair.
  (e) In every case copy's position and the carry are reported. If the instruments place elicit and rank-1 together
  while the label separates neither from the cipher at its cut, the reading "the carry label reads rule
  generalisation, not old versus new pieces" is written as the finding about the label.
  (f) Either instrument reads "prior" (rule 6, § 6): it tracks how likely W₀ already found each map's value, which is
  Def 1 possession, the same thing the probe q reads trivially. This axis is also an axis of the prior: rank-1's
  values are W₀'s top pick, the cipher's are under 1e-4, and elicit's sit between. So a "shared" verdict is claimed
  only for instruments that do not read "prior". r and the stand-in ratio are put through the same check. If r also
  sits within the prior's tolerance, the write-up says that on this pair VPD's readings are not separated from the
  value's prior. Copy is the map that separates the two readings, and its position is reported: its prior is low (about
  6e-5), while r puts it at the old end.
- **Control first.** Job 1, before the second filing, is measurement only: D and the prior on the twelve existing
  states, the MLP shares, the random-removal spread, and the planted wrapper built and checked. Job 3 opens with stage 0 alone (§ 5) and exits on a miss.
- **Numbers.** Two scored readings per state (R, t½), one reference per map (q), one position per column. Expected
  values are in § 4.
- **Class.** Class 4, the reuse line's named next step (Fable option C, adopted), and class 2: more seeds of runs that
  already happened. The 20 new fine-tunes repeat E58's recipe and code with new seeds. They are new states, so this
  goes beyond reading from disk; it is said here as the review asked. All jobs are $0 debug jobs and none is gated.
  No sign-off is needed. FYI to the user.

## 3. Instruments on the 67M

**Windows.** Every state is read on its trained keys' windows, the same tokens for all four maps (E54's windows, split
24/8 as E58; stage 0 checks that the tokens are identical). The rank-1 file `.../large/amend1/windows_rank1_trained.pt`
is used: `train` for reverse fine-tuning; `heldout` (75 windows) for reading. Its fit half (even indices) scores units
and its score half (odd) reads every number below. fp32, as E58.

**The old behaviour and distance.** κ(m) is the mean over score-half windows of KL(p_W₀ ‖ p_m) at the answer position,
in nats. D = κ(state) is the state's distance from W₀ in this unit. A state with D < 0.2 × the smallest pole-map mean D
(measured in job 1) is void. This covers the zero update; no pole state is expected under it.

| instrument | reading per state | refined how (from C1b) | "old" side |
|---|---|---|---|
| **Pruning revival** | Unnormalised revival Δ_K = (κ(state) − κ(top-K pruned)) − mean over 20 random K-unit removals of (κ(state) − κ(random pruned)). R = max over K ∈ {13, 40} of Δ_K / D. Units are the 4 × 3072 MLP neurons (`steering.Net` site `mlp<l>`), zeroed at every position; this equals zeroing up_proj's row, since GELU(0) = 0. Units are ranked by Alg. 3's grad · w on up_proj's row (and bias, if any), with the gradient of the fit-half mean κ at the state. The random draws are `default_rng([PRUNE_SEED, K])`, the same for every state. The top-K unit ids are saved per state | K scaled from C1b's 5 and 15 of 4,608 units (0.11%, 0.33%). C1b's "minus random". The target is W₀'s distribution, not one answer, so every map is readable. "Max over K" biases R up by the same rule for every state (review M2) | high |
| **Reverse fine-tuning** | t½ = the step at which κ on the score half first falls to D/2, interpolated linearly between the two steps that straddle it; κ is read after every step. Training: Adam, lr 3e-6, batch 32, clip 1, loss = KL(p_W₀ ‖ p_m) at every position of the trained keys' `train` windows, started from W₀ + δW, seed = the state's seed. The run stops at the crossing or at the cap of 1,000 steps. A state still above D/2 at the cap is "censored" | Jain et al. reverse below the forward rate, and C1 used a tenth of each state's own rate. Here that is one rate, a tenth of E54's 3e-5, for every state (all share the forward rate). Continuous, read on the same κ as D (review S7), no cap value enters any mean (S6). No scratch divisor (§ 9) | low |
| **Possession probe, reference only** | q per map = max over blocks l = 1-4 of the mean over the trained keys' held-out windows of [log P_lens,l(own value) − mean over the map's other distinct values of log P_lens,l(value)]. P_lens,l is W₀'s final norm and unembedding on the residual after block l (logit lens); block 4 is W₀'s own output. It reads W₀ and the map only | C1b's probe on the state's answers would read about 1 on every map, because each trained-key answer is a lookup on the key token in the residual. Not a third instrument: elicit reads high by selection, and copy at block 1 will likely decode the key as itself, so q puts elicit and copy together trivially (review M5) | high |

**Positions.** With seed means over non-void seeds (8 per map):

p = (v̄_elicit − v̄_cipher) / (v̄_rank-1 − v̄_cipher).

p near 1 puts elicit at the old end, where r and the ratio put it. p near 0 puts it at the new end. Copy's own position
p_copy = (v̄_copy − v̄_cipher) / (v̄_rank-1 − v̄_cipher) is reported beside every column. So is the carry label's
category. Copy's labels are neither / reuses / neither at seeds 0-2: it reads "reuses" on one seed of three (review S8).

## 4. Numbers and their expected values

D per map comes from job 1, before filing; the guesses below are replaced by the measured means then. The guess
assumes each state puts about 0.95 on its value and spreads the rest roughly as W₀ does, so that κ ≈ (1 − p₀(v)) log
((1 − p₀(v)) / 0.05) + p₀(v) log(p₀(v) / 0.95), with p₀(v) W₀'s probability of the value.

| number | W₀ / zero update | rank-1 (old end) | elicit | copy | cipher (new end) | control |
|---|---|---|---|---|---|---|
| D, nats | exactly 0, void | 1.5-2.5 (W₀ already gives the value 0.1-0.3) | 2.5-3 | about 3 | about 3 | — |
| p_D (baseline column) | — | — | expected about 0.2, near the new end: elicit's distance is close to the cipher's | reported | — | measured in job 1 |
| prior, mean log p_W₀(value) on the score half | W₀ only | about −2 to −3 (W₀'s top pick) | about −6.4 (E54: held-out geometric mean 1.70e-3) | about −9.7 (5.9e-5, E58 A6) | about −13.7 (E54: 1.08e-6) | measured in job 1 |
| p_prior (baseline column) | — | — | about 0.6-0.7, near the 2/3 edge; r's 0.876 is 0.2 above it | p_copy about 0.35 | — | measured in job 1 |
| MLP share of ‖δW‖² | — | measured on CPU in job 1 (E56: attention 0.29-0.31, so MLP about 0.7 on elicit and cipher) | | | | |
| pruning R | void | unknown | unknown | unknown | unknown; filed direction rank-1 > cipher | planted wrapper: ≥ 10 of 13 planted units in the top 13, and R ≥ 0.8 × the ceiling share (D − κ_zeroed) / D |
| reverse t½, steps at 3e-6 | void | unknown, guess 10-100 | unknown | unknown | unknown; filed direction rank-1 < cipher | planted wrapper reaches D/2 before the cap; its t½ reported |
| probe q, nats | W₀ only | ≥ elicit | ≥ 1 | likely high (lens decodes the key) | within ±0.5 | block-4 lens equals W₀'s log-softmax within 1e-5 |
| p, r (files, 3 seeds) | — | — | 0.876, bootstrap 90% 0.859-0.894 → VPD | p_copy 0.996 | — | recomputed from the files (CPU) |
| p, stand-in ratio | — | — | 0.787, 0.778-0.796 → VPD | 0.895 | — | same |
| p, steering S | — | — | no gap: rank-1 0.224 vs cipher 0.181, gap 0.043 under 3 sd 0.105 | — | — | same |
| p, carry x (reported) | — | — | 0.691, 0.650-0.728 → between (straddles 2/3) | 1.83 | — | same |
| p, update size | — | — | no gap (gap −0.03, 3 sd 0.28), matched by design | — | — | same |

The file rows were computed for this plan from `run.json` and `amend1.json` with jq and awk, using a 2,000-draw seed
bootstrap over seeds 0-2 only. They are the rule's worked examples (§ 6) and will be recomputed on eight seeds.

## 5. Jobs, controls first, stop rules

All on the MATS cluster `debug` QoS (L40, account c11, `--time=00:30:00`), through `/pod-runner` (the lead spawns it).
Each writes results as each state finishes and resumes from them. `experiments/open_pair_instruments/run.py --stage
<name>`, one sbatch `cluster/open_pair_instruments.sbatch`.

**Job 1, `measure`, after filing A and before filing B. Measurement only; no instrument reads a scored state.**
1. sha256 of the 12 state files equal `ft.sha256` in `run.json` / `amend1.json`. The windows files match their recorded
   sha256. The four maps' trained-key windows have identical tokens. Exit 1 on a miss.
2. Each state's raw gain G reproduces the recorded G within 1e-4 (E58 Amendment 1's tolerance). Exit 1 on a miss.
3. D on the 12 states and on the zero update (exactly 0). The prior per map: the mean of log p_W₀(value) on the score
   half, from W₀ alone. The MLP share of ‖δW‖² per state. The off-answer KL per state: the mean KL(W₀ ‖ state) at the
   positions other than the answer, which bounds the planted edit in item 4.
4. **Random-removal spread** (review m1): on each of the six pole states, the sd over the 20 random K-unit removals of
   (κ(state) − κ(random pruned)) / D, at both K. R's noise floor is the largest of these, filed after job 1. This
   replaces C1b's 0.05, which was in accuracy units.
5. **The planted wrapper** (review N2).
   - *Units.* Among block 3's units whose mean GELU activation at the answer position (fit half) is above the block's
     median, take the 13 whose zeroing on W₀ costs least: κ(W₀ with that one unit zeroed), measured unit by unit.
   - *Edit.* Set their down_proj columns to W₀'s plus −α · w / ā_j. Here w is the unit vector of the fit-half mean of
     (u_rank1(key) − ū), with u the unembedding rows and ū their mean, and ā_j is the unit's mean activation.
   - *Scale.* Choose α by bisection so that κ(planted) equals the median D of the six pole states, subject to the
     planted edit's off-answer KL staying at or below the largest off-answer KL among the 12 real states (item 3).
   - *Fallback.* If the median D cannot be reached within that bound, plant at the largest κ reachable within it and
     record it. The pass rules below are relative to the planted state's own D, so they still apply.
   - *Ceiling.* κ_zeroed = κ(W₀ with the 13 units zeroed) is recorded. R's ceiling on the planted state is
     (D − κ_zeroed) / D.
6. Pruning on the planted state: the overlap of its top 13 with the planted 13, and R against 0.8 × the ceiling.
   Reverse on the planted state: t½, reported.

**Filing.** Filing A is committed before job 1. It holds the rules (§ 6), the edge cases, every constant that does not
depend on job 1, C0a (job 1's stop rows) and the probe reference. Filing B is a dated entry appended after job 1 and
before job 2; it never edits A. It holds the measured D, prior and MLP-share means, p_D, p_prior, R's noise floor, the
planted units, α and ceiling, and every prediction that depends on them: C0b (job 2's reproduction control and job 3's
stage 0), P1, P2, P3 and V. Jobs 2 and 3 refuse to start without filing B committed.

**Job 2, `seeds`, after filing B.** First, alone, the reproduction control (review N4): retrain seed 0 of each map,
elicit first, then the cipher at elicit_s0's norm, then rank-1 and copy. Each must match the saved state's record:
- the scored step exactly;
- the norm within 1e-4 relative;
- G and x within 1e-3;
- r within 1e-3.
E58 Amendments 3 and 4 retrained states and reproduced them exactly (weight difference 0; the cipher's r 0.368 both
times), so these tolerances leave room for GPU nondeterminism only. Exit 1 on any miss, before any new seed.

Then seeds 3-7. Per seed, elicit runs before the cipher, because the cipher's target norm is the same seed's elicit
norm. The selection rules are E58's, unchanged. Elicit, rank-1 and copy are read at E12's scored step, the first of
10, 30, 100 and 300 with held-out P(target) ≥ 0.5 (`keydep.elicit_ft`). The cipher is read at the step in 1-30 whose
norm is nearest the elicit state's (`keydep.cipher_ft`). Each state then gets E58's columns
(`heldout_keys.behaviour` for carry and label, `heldout_keys.vpd_read` for r, ratio and S), all called unchanged. The
states are saved to `experiments/open_pair_instruments/results/large/` (113 MB each, 2.3 GB, gitignored; on the
cluster's shared filesystem, so the symlink rule for pod downloads does not apply), each with its sha256. Stop on
E58's block-0 rule.

**Job 3, `instruments`. Stage 0 alone first, exit 1 on a miss.**
1. The zero update: D = 0, pruning and reverse void.
2. Lens check: block 4 equals W₀'s log-softmax at the answer position within 1e-5. A miss voids the probe; the job
   continues.
3. The planted controls. Pruning must find at least 10 of 13 and read R ≥ 0.8 × the ceiling share. A miss voids
   pruning; the job continues. The planted reverse must reach D/2 before the cap. A miss stops the job: the loop does
   not revert. The planted t½ is reported and is not compared with the poles (review N3).
4. After the planted reverse, the target weights equal W₀ exactly (`model_state` restored them). The same check runs
   after every reverse; any difference stops the job.

Then D, R and t½ on the 32 states, in the order rank-1, cipher, elicit, copy and seeds 0-7, written after each, then q
per map. Then scores (§ 6). Every scored number is also printed to the log.

**Job 4, resume, planned** (review m3). Job 3 writes each state's readings as it finishes. A censored state costs
1,000 steps plus 1,000 reads, about 1-1.5 min on an L40, so a few censored poles push job 3 past 30 min. The code's
deadline is 25 min. On exit 3 the log prints the resubmit line, and the pod-runner submits `--stage instruments` again
(compute nodes cannot `sbatch`). It skips finished states, reruns stage 0, then scores.

## 6. Claim, null, predictions

**Claim (one sentence).** Pruning revival and reverse fine-tuning, which need no decomposition, place E58's elicit map
near the rank-1 map, the end that is old by construction, as r and the stand-in ratio do. They do it neither where
distance from W₀ alone would place it nor where W₀'s prior on the value alone would. So the granularity limit is
shared by instruments that read mechanism.

**Null.** The cheap instruments cannot read the rank-1-to-cipher axis on the 67M, or they place elicit between the
ends, at the new end, or where distance or the prior alone puts it.

**Rule (pure functions in `src/commute/revival_kv.py`, CPU-tested on the edge cases before filing).**
1. **Gap, each instrument's positive control on real states.** noise = the largest within-map seed standard
   deviation among rank-1, cipher and elicit. It is floored for R at the random-removal spread measured in job 1
   (filing B) and for t½ at 1 step.
   gap = v̄_rank-1 − v̄_cipher. "ok" iff d · gap ≥ 3 · noise, with d = +1 for R and q and −1 for t½. "reversed" iff
   −d · gap ≥ 3 · noise. Else "no gap". Undirected baseline columns (update size, D, MLP share) use |gap|.
2. **Censoring.** If any pole seed is censored, t½ reads "censored" and is void (review S6). A censored elicit or copy
   seed is reported and dropped.
3. **Side**, only with gap "ok": "VPD" iff p ≥ 2/3, "label" iff p ≤ 1/3, else "between". Ties go to the outer band;
   p > 1 counts as "VPD" and p < 0 as "label".
4. **Firm.** 2,000 bootstrap draws, resampling seeds with replacement within each map, with a fixed rng seed. A draw
   whose denominator is 0 counts as outside every band. The side holds only if the 5th and 95th percentiles of p lie in
   its band; otherwise it reads "between".
5. **Distance.** If p_D has gap "ok", an instrument whose firm side equals p_D's, or whose p lies within 0.1 of p_D,
   reads "distance" instead of its side.
6. **Prior.** The same with p_prior: if p_prior has gap "ok", an instrument whose firm side equals p_prior's, or whose
   p lies within 0.1 of p_prior, reads "prior". If both rule 5 and rule 6 fire, the reading is "distance and prior".
   r and the stand-in ratio are put through rules 5 and 6 too, and reported. The 0.1 tolerance in rules 5 and 6 has no
   source; like the bands, it is this project's choice and is stated as such (review m2).
7. **Per-instrument verdict** = its reading from rules 1-6: VPD, label, between, distance, prior, no gap, reversed,
   censored or void. **Joint verdict:** "shared" iff both read VPD; "shared (by X)" iff X reads VPD and the other reads no gap,
   void or censored; "VPD alone" or "VPD alone (by X)" likewise with "label"; "conflict" iff one reads VPD and the
   other label or reversed; otherwise "not settled", naming each.

**Edge cases, run on paper here and as tests before filing.**
- An empty pool: a map with fewer than 2 non-void seeds makes the column "void".
- A zero denominator: gap 0 gives "no gap" under rule 1. Bootstrap draws with a zero denominator count as outside.
  This matters for columns such as S, where rank-1 and the cipher nearly tie.
- Ties: at the band edges, rule 3. Discrete-looking t½ is continuous by interpolation, and the 1-step floor remains.
- Censoring: rule 2. A relative void floor on D means a seed just above an absolute 0.1 cannot blow R up (review M1).
  The unnormalised Δ is reported beside R.
- The distance rule when p_D has no gap: rule 5 does not fire. The same holds for rule 6 when p_prior has no gap.
- p_prior in the VPD band, firm: rule 6 then turns every "VPD" reading into "prior", r's included. That is intended,
  and is outcome (f). On the expected prior means, p_prior ≈ (−6.4 + 13.7) / (−2.5 + 13.7) ≈ 0.65, inside "between",
  so only the 0.1 clause can fire. r at 0.876 would not read "prior".
- The planted bisection cannot reach the target: the fallback in § 5 job 1 item 5 applies. The pass is relative to
  the planted state's own D and ceiling, so it stays defined.
- Worked on the known columns (3 seeds): r → gap 0.613 ≥ 0.050, p 0.876, bootstrap 0.859-0.894: "VPD". Ratio →
  0.782 ≥ 0.055, 0.787, 0.778-0.796: "VPD". S → gap 0.043 < 0.105: "no gap". Carry → 0.449 ≥ 0.110, p 0.691, 0.650-0.728
  straddles 2/3: "between". Update size → |gap| 0.032 < 0.284: "no gap".

**Predictions** (leans for the run as a whole; "blind to", control and noise floor in each row).

| id | prediction | rule | lean | blind to | positive control; noise floor |
|---|---|---|---|---|---|
| id | filing | prediction | rule | lean | blind to | positive control; noise floor |
|---|---|---|---|---|---|---|
| C0a | A | Job 1's stop rows pass: sha256, identical tokens, G within 1e-4, zero update exact | § 5 job 1 | 95% | — | — |
| C0b | B | Job 2's seed-0 reproduction and job 3's stage 0 pass, including both planted controls | § 5 | 75% (planted pruning 80%, planted reverse 90%, reproduction 90%) | — | — |
| P1 | B | Each new instrument reads the rank-1-to-cipher gap in its filed direction | rule 1 | R 55%, t½ 65% | R: attention changes (about a third of each update's squared norm, E56; MLP share reported) and changes spread over more than 40 units. t½: structure versus distance (rule 5). Both: the value's prior (rule 6) | planted wrapper; R floor measured in job 1, 20 random draws; t½ floor 1 step |
| P2 | B | Pruning places elicit at the old end | rules 3-6 on R | VPD 25%, between 10%, label 5%, distance or prior 10%, no gap/reversed/void 50% | as P1 | as P1 |
| P3 | B | Reverse fine-tuning places elicit at the old end | rules 2-6 on t½ | VPD 25%, between 10%, label 10%, distance or prior 20%, no gap/reversed/censored 35% | as P1 | as P1 |
| V | B | Joint verdict | rule 7 | shared or shared (by X) 25%; VPD alone (incl. by X) 10%; conflict 5%; not settled 60% | — | — |
| ref | A | Probe: cipher q within ±0.5, elicit ≥ 1, rank-1 ≥ elicit; p_q reported | § 3 | 65% | the fine-tune entirely | lens check; cipher as the null |

Filing B may revise the leans of P1-V in light of job 1's measurements, but only before any instrument reads a real
state. Filing A is never edited.

**Lean for a settled answer** (any joint verdict other than "not settled"): about 40%. A "prior" or "distance" reading
counts as not settled.

**Why these leans.** κ makes rank-1 readable: its fine-tune sharpens W₀'s own answer, so W₀'s distribution is
plausibly hidden behind a few units and reverts quickly, while the cipher overwrites it. That is the gap P1 needs.
Elicit reorders W₀'s top 20 and is expected to be about as far from W₀ as the cipher (p_D ≈ 0.2). A "VPD" reading
from an instrument is therefore unlikely to be distance in disguise, and rule 5 catches it if it is. Pruning on the
67M is still untested on real states. A third of each update sits in attention, so revival from 40 units may be small
everywhere. That is why the planted wrapper is the control, and why "no gap" stays the likeliest single pruning
outcome.

## 7. Code: what goes where

- **New `src/commute/revival_kv.py`** (about 180 lines): `kl_to_w0` (κ, with W₀'s answer-position log-probs computed
  once per window set), `unit_scores` (grad · w on up_proj under `Net.at(dW)`), `prune_readings` (edits at site
  `mlp<l>`, the same structure as `neurons.neuron_edit`; saves the unit ids), `planted_wrapper`, `reverse_half_life`,
  `lens_prior`, and the pure rules `gap`, `position`, `bootstrap_side`, `distance_check` and `verdict`. CPU tests in
  `tests/test_revival_kv.py` cover the rules and edge cases only.
- **Added to `src/commute/finetune.py`:** a public `finetune_to_reference(model, ref_logprobs, train, cfg, on_step)`.
  It runs the existing loop `_run` with the KL-to-W₀ loss and an `on_step` that reads κ every step. Nothing existing
  in `finetune.py` changes, so `finetune_arm_a` and every frozen caller run as before.
- **New `experiments/open_pair_instruments/run.py`** (about 250 lines: three stages, resume, file columns) and
  `plot.py`. The run refuses to start (except `--stage measure`) unless `results/predictions.json` is committed with
  status "filed".
- **Called, not changed:** `steering` (Net, Task, read), `heldout_keys` (behaviour, vpd_read, pool_of), `keydep`
  (make_ctx, elicit_ft, cipher_ft, sha256), `anchors.FT_LR`.
- **Nothing moves.** `src/commute/revival.py` stays as C1b ran. It is minGPT- and PCFG-specific (`pcfg.ANS`,
  `answer_loss`), and its callers are frozen: `experiments/pcfg_anchors/c1.py`, `c1b.py`, `plot_c1.py`, `plot_c1b.py`,
  `tests/test_revival.py`. No code is copied from it.
- **Review.** A fresh-Opus review of the code diff before job 3 (C1b's practice; Codex is out of quota).

## 8. Compute, cost, outputs

- **Estimate.** Job 1: model load 1-2 min, 12 κ reads and the planted build and check, about 8 min. Job 2: 20
  fine-tunes plus E58's readings. Amendment 1 ran 6 in about 2.5 min, so about 10-15 min (each cipher holds about
  3.6 GB of deltas in host RAM, as E58). Job 3: per state, pruning (one gradient pass, 42 forwards on 37 windows)
  under 30 s and the reverse (at most 1,000 steps of batch 32 plus a 37-window read each step) about 10-40 s, so 32
  states take about 15-25 min. **Three debug jobs, a fourth if job 3 resumes. $0.** None is gated: each job is $0,
  under 3 h and within the session cap (`decisions.md` 2026-10-30: a run is one Slurm job).
- **Saved.** In `experiments/open_pair_instruments/results/`, committed and small: `measure.json` (job 1), `seeds.json`
  (job 2: sha256, columns), `run.json` (per state D, MLP share, per-K κ and unit ids, R and Δ, the reverse κ curve and
  t½; per map q per block; per column gap, p, p_copy, bootstrap percentiles, side, distance flag; per-instrument and
  joint verdicts). Also `predictions.json` (**DRAFT** now, status field says so) and `figures/positions.png` (each
  column's elicit position on its rank-1-to-cipher axis with the bootstrap interval, copy marked, bands shaded; one
  figure for the claim). Gitignored: the 20 new states in `results/large/`. Logs go to
  `cluster/logs/commute-open-pair-instruments-<job>.out`.
- **After.** A `/writeup` report `reports/analyses/<date>_analysis_e55-c3-open-pair.md`, a section in
  `notebooks/E55_pcfg_anchors.ipynb`, a `TRIED.md` row and a `PENDING.md` line.

## 9. Departures from the literature

- **Old behaviour.** Jain et al. revive a labelled pretraining capability. Here it is W₀'s whole next-token
  distribution at the answer position, read as KL(W₀ ‖ m). No pretraining labels exist for these keys, and the
  distribution is readable on every map, including the old-by-construction rank-1 map.
- **Pruning.** Alg. 3's grad · w on the unit's input row, as C1b, with the gradient of κ. Zeroing is done as an
  activation edit, identical since GELU(0) = 0. K is scaled by unit count, and there is one number per state instead of
  a curve.
- **Reverse fine-tuning.** At a tenth of the forward rate, following Jain et al.'s lower reverse rate and C1's lr/10.
  The loss is KL to W₀, not cross-entropy on labels. The reading is a continuous half-life of κ, not steps to an
  accuracy. There is no scratch model: the 67M has no model that never had W₀'s behaviour, and p does not change if
  every reading is scaled by a common factor.
- **Possession probe.** Def 1's readout is W₀'s unembedding at each block (logit lens, nostalgebraist 2020), not a
  probe trained here. The control is the map's other values, not Hewitt and Liang's control task. A tuned lens
  (Belrose et al. 2023) is not run.
- **The axis, bands, bootstrap, distance rule and verdicts** are this project's; the papers plot curves.
- **Steering, r, the stand-in ratio and the carry label** are E58's, unchanged; their departures are in its report.

## 10. Not run

- The interchange-patch label (Stage B), the piece-level ground truth that reading A asks for. Not built.
- The 96-key states. Only seed 0 of elicit, rank-1 and copy is saved (`large/amend3/`), and there is no saved cipher
  at 96 keys, so no pole. Also not run: the shuffled map, the prior-matched cipher and Amendment 7's maps.
- VPD on the counting anchors (C2, gated, the user's call). A second base model or task family.
- Attention-head pruning; K above 40; other reverse rates; reverse targets other than KL to W₀; a tuned lens.
- Copy as a pole, and the graded carry or its label as a scored target.
- E58 Amendment 6's prior-matched cipher, retrained as a reported map beside copy. The review offered it as an
  option; it would separate the prior from kind at copy's prior, but not at elicit's.
- Any edit to E58's or C1b's files or predictions.

## Changes after review (2026-10-31)

| item | what was done |
|---|---|
| B1 target is not the poles' capability | The old behaviour is now W₀'s whole distribution (κ, KL to W₀), readable on every map. The poles are now construction-fixed (B2), so a "VPD" reading means "elicit sits with the map that only strengthens W₀"; the claim and § 2 were rewritten to say that, and the label's alternative reading is outcome (e) |
| B2 rank-1 void; poles unchecked | The axis is rank-1 (old by construction) to the cipher (new by construction). Copy and the carry label are reported beside it. The top paragraph says how this changes Fable's question. Graded carry reads p 0.69 on this axis, stated |
| B3 reverse resolution and rate | Rate 3e-6 (a tenth of the forward rate: Jain's lower rate, C1's lr/10). The reading is the continuous κ half-life, interpolated, read after every step, cap 1,000. `CIPHER_CHECKS` is no longer used |
| S1 worst-case firmness | A 2,000-draw seed bootstrap with a central 90% in one band. Eight seeds per map (20 new fine-tunes, job 2), said in the design check. The bootstrap guards zero denominators on the rank-1 axis |
| S2 one void decides the verdict | Per-instrument verdicts, and a joint verdict with "shared (by X)", "VPD alone (by X)" and "conflict" (rule 6) |
| S3 distance not scored | D is read on the 12 states in job 1, before filing, with guessed per-map values in § 4 that the measured means replace. p_D is a baseline column, and rule 5 makes "distance" a reading |
| S4 attention share, unit ids | MLP share of ‖δW‖² is a baseline column (job 1, all states). Top-K unit ids are saved per state |
| S5 pruning control cannot fail | A planted 13-unit wrapper on W₀ in block 3, scaled to the median pole D. Pruning must find ≥ 10 of 13 and read R ≥ 0.8; reverse must recover it faster than the median pole state. A miss voids that instrument |
| S6 arbitrary cap value | No cap value enters a mean. A censored pole seed voids t½ ("censored") |
| S7 statistics mismatch | D, R and t½ all use κ on the score half. The fit half only ranks units |
| S8 copy's label overstated | Copy reads "reuses" on one seed of three (neither / reuses / neither), stated in § 3 |
| M1 absolute D floor | The void floor is 0.2 × the smallest pole-map mean D, and the unnormalised Δ is reported |
| M2 max over K | Stated as an upward bias that is the same for every state |
| M3 time | Re-estimated: three debug jobs, plus a fourth if job 3 resumes, all $0 |
| M4 unsourced bands | Kept (stated as this project's), with the bootstrap so one seed cannot decide the edge |
| M5 probe trivially pairs elicit and copy | Stated in § 3; the probe stays a reference and is not counted as an instrument |
| N1 prior axis (re-check) | Prior column (mean log p_W₀ of the value per map, job 1, $0); rule 6 "prior" with the same treatment as distance, applied to r and the ratio too; outcome (f) in § 2; expected p_prior ≈ 0.65 from E54's geometric means. The prior-matched cipher of E58 Amendment 6 is not retrained (optional in the review; § 10) |
| N2 planted control passable | Units: above-median-active block-3 units with the lowest single-unit zeroing cost on W₀. Pass is relative: R ≥ 0.8 × (D − κ_zeroed) / D. α is bounded by the largest off-answer KL among the real states, with a fallback to the largest reachable κ, recorded |
| N3 planted t½ void rule | Dropped. "Reaches D/2 before the cap" stays as the stop; the planted t½ is reported |
| N4 retrain control | Job 2 retrains seed 0 of each map first and alone (elicit, cipher at its norm, rank-1, copy). Checks: step exact; norm within 1e-4; G, x and r within 1e-3. Exit on a miss. Elicit runs before the cipher per seed, and the scored-step rules are named |
| m1 R's floor units | R's noise floor is the random-removal sd in shares of D, measured on the pole states in job 1 and filed in filing B |
| m2 unsourced 0.1 | Stated as this project's choice in rule 6, as for the bands |
| m3 job 4 | Resume planned, with a 25-min deadline, exit 3 and resubmission by the pod-runner (§ 5) |
| m4 symlink | Noted: a shared-filesystem write on the cluster, not a pod download |
| Filing order (lead) | Filing A before job 1: rules, edge cases, C0a, probe reference. Filing B after job 1, before job 2: measured columns, R's floor, planted settings, and C0b, P1, P2, P3, V (§ 5 "Filing") |
| Rejected | None. One deviation from the lead's "one cluster run" wish: S1's seeds and S3's pre-filing D make it three or four $0 debug jobs. Each job is its own run under the gate rule, so nothing is gated |

## Hand-off

The next session reads this plan and the review, then `src/commute/revival.py`, `steering.py`, `neurons.py`,
`finetune.py` (`_run`), `keydep.py`, `heldout_keys.py`, `experiments/open_pair/amend1.py` and the draft
`experiments/open_pair_instruments/results/predictions.json`. It writes `revival_kv.py`, `finetune_to_reference` and
the CPU tests of the rules and edge cases, then commits filing A (status from DRAFT to "filing A"). The lead runs job 1
(`measure`) through `/pod-runner`. The measured D, prior, p_D, p_prior, MLP shares, R's floor and planted settings go
into § 4, and filing B is appended and committed. After that come the code review, job 2 and job 3.

## Amendment 1, 2026-10-31: planted control

**Why.** Job 1 (cluster job 46774) built the planted wrapper as § 5 item 5 specified, and it could not work. Block 3's
median activation at the answer position is negative, so 1,160 units qualified as "above the median", and the 13 with
the lowest zeroing cost were nearly silent (ā 0.002-0.097). The edit −α · w / ā_j scaled them up by as much as 400×,
at every position where they fire. The off-answer KL bound (0.577) was reached at κ 0.184, against a target of 7.0 and
below D_void (0.978). The pruning check on that state (4 of 13 found, R 0.39) is noise, not a failure of pruning
(`reports/reviews/2026-10-31_review_e55-c3-code.md`, findings 1-2). Filing A's text is unchanged; this amendment and
the `amendment_1` block in `results/predictions.json` replace its planted `units` and `edit` rules and add a floor.

**What changes (block 3 = site `mlp3`, 0-based, as filing A states).**
- *Units, by answer selectivity.* On W₀ and the fit half, for each block-3 unit j: a_ans(j) = the mean GELU activation
  at the answer position; a_else(j) = the mean |activation| over every other position. Candidates are the units with
  a_ans above the block's 90th percentile of a_ans (and above 0). The 13 planted units are the candidates with the
  largest selectivity a_ans / a_else (ties to the lower id). Single-unit zeroing costs are no longer measured.
- *Edit, without 1/ā.* Each planted unit's down_proj column += −α · w, the same α for every unit. w is unchanged.
- *Scale.* Unchanged: bisection to the median pole D (the target) within the off-answer KL bound, with the recorded
  fallback.
- *Planted D floor.* floor = max(D_void, 0.5 × the target), with D_void = D_void_frac × the smallest pole-map mean D, as
  filing B will file it. If the planted state's D is below the floor, the control is **void** ("control not built"),
  not failed: the planted pruning pass is not scored and reads "void", and the pruning instrument then has no positive
  control and is void in the same way as after a miss, but reported as "control not built". The planted reverse
  check (reaches D/2 before the cap, else stop) is unchanged and runs at any D. The factor 0.5 is this project's
  choice: it keeps the planted D within the range of the real pole states (D 4.4-10.6 in job 1).
- *Pass rules unchanged:* ≥ 10 of the 13 planted units in pruning's top 13, and R ≥ 0.8 × the ceiling
  (D − κ_zeroed) / D. The reverse rule is unchanged.
- *Filing B reads* `planted {units, alpha, floor}` (no `abar`); w is recomputed from W₀ and the rank-1 values.

**Rerun.** Job 1's `results/measure.json` (job 46774) moves to `~/commute-data/stale/reuse-lead/measure_job46774.json`
(it stays in git history at 2650657) and job 1 runs again. Every number other than the planted wrapper and its checks
must reproduce exactly (D, prior, off-answer KL, MLP share, R's floor, D_void); any difference is reported.

**Filing B, 2026-10-31 (after the Amendment 1 rerun, job 46808).** The planted control was built and passed (D 4.21
against a floor of 3.50; 12 of 13 found; R 0.973 against 0.793), and every other job-1 number reproduced exactly, so
filing B is appended to `results/predictions.json` from the new `filing_B_draft`, following Fable's advice
(`reports/plans/2026-10-31_advice_e55-c3-baselines-FABLE.md`, option A, adopted by the lead). Rules 5-6 stay as filed.
Both decomposition-free baselines read firm VPD (p_D 1.47, p_prior 0.92, the prior by construction), so any instrument
reading on the VPD side converts to "distance and prior": the leans for P2 and P3 "VPD" and for V "shared" and
"conflict" are 0. "Label", "between" and "VPD alone" (outcome b) stay reachable and keep their mass. A new reported-column
prediction, P_copy, leans 55% that an instrument reading the gap puts copy nearer r's position (1.00) than the prior's
(0.59), above the midpoint 0.791. An unfiled reading in which rules 5-6 fire only under the 0.1 clause will be reported
beside the filed verdict, labelled unfiled.
