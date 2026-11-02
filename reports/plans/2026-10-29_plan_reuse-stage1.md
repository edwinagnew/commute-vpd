# Plan (E55): does post-training reuse a capability the model already has, judged without any decomposition?

**Reuse-lead, 2026-10-29, from a draft by Fable (advisor). Reviewed (`reports/reviews/2026-10-29_review_e55-plan.md`:
4 blocking, 7 should-fix) and revised below. No predictions filed; nothing run.** Open questions for the user:
`reports/decisions/2026-10-29_decision_e55-plan.md` (card `decision-e55-plan`).
Sources: `~/commute-data/agents/reuse-lead/brief.md` (Stage 1), `~/commute-data/scratch/reuse-lead/jain/jain_notes.md`
(notes), `~/commute-data/scratch/reuse-lead/audit/task_audit.md` (audit rows), Jain et al. 2024 (L = line of
`~/commute-data/lit_review_pdfs/Jain2024_finetuning_wrappers_2311.12786.txt`; Table 6 = L3006-3075). Words as in
`GOALS.md` → Terms and `decisions.md` (reuse-line rows).

## Revision 1 (2026-10-29, after review)

1. **Ground truth replaced (B1, B2).** Was probe-direction ablation + pruning + reverse fine-tuning, 2 of 3. Now a
   probe-free behaviour check plus an interchange patch at `=`, per arm and named capability. Ablation dropped (shared
   number subspace, off-distribution residual, weak baseline); pruning and reverse fine-tuning move to Stage C as
   instruments with valid baselines (Jain's Fig. 76/77 and Fig. 9 predict they do not separate N1 from W).
2. **Labels name the function (B3).** W "goes through count-a"; R "goes through count-b, which HIGH possesses"; N1
   "does not go through count-a", new or not by Def 1 on LOW for count-b. Possession leaves the arm rule. Leans only.
3. **Bars from measured controls (S2, S3, S4).** A1 measures null, positive control and spread; A2's bars follow by a
   rule written here, filed before A2. A failed control voids a check; lo < hi is required. The 15-point
   reproduction gate becomes a prediction. Count-c probe replaced by NEVER-model and shuffled-label probes.
4. **Recipe (S5, S1).** Random contiguous 250-window; spurious b only in the counted window at fixed length; reverse
   fine-tuning sourced (L1249 = Tracr recipe; departure stated), one lr rule shared with scratch; reads at `=`.
5. **Stage B narrowed (S7; B4 open).** Lowercase only, f_C = context case; held-out keys and 67M pruning/reverse
   fine-tuning dropped; branches presuming lowercase reads reuse removed. Cipher/copy under Def 2: card item (ii).
6. **Stage C power (S6).** 16 LOW-based states of Table 6's grid (+ R), binomial chance per pass bar. C1 ($0):
   pruning, reverse fine-tuning, steering. C2 (gated): one VPD decomposition of LOW.

## In plain words

Our "existing vs new" labels for post-training have so far come from VPD itself, so a sceptic reads every result as a
fact about VPD. This plan builds a ground truth that owes nothing to any decomposition: a small model trained on Jain
et al.'s counting grammar, where we choose whether the model already counts a letter, then fine-tune it to count
another letter. For each fine-tune we ask whether the new behaviour runs through the old count, two ways: by what
the model outputs, and by swapping in the old count from another input. Other methods, and later VPD, are scored on
how often they give the same answer.

**Claim.** Where Table 6 shows the spurious solution (W1, W2) the fine-tuned output runs through count-a by both
checks; where count-b is learned for real (N1) it does not; at least one instrument agrees with this above chance.
**Null.** The two checks disagree on the anchors, or the patch moves every model alike: no instrument can be scored.

## Design check (`/design`)

- **Goal served.** `GOALS.md` "If no simple Φ works": new capabilities or strengthening. Gives that question a
  definition outside VPD (`decisions.md` reuse-line row) and a ground truth to score Φ against.
- **Nearest tried.** E12/E46/E52 (`TRIED.md`): copy vs cipher labelled existing/new by assumption; E52 B4: the
  original model does not do the copy task even prompted. E53 A0: lowercase is performed (+1.18 nats), no causal check.
  New here: possession set by pretraining prior, the reused function named per arm, a causal check that needs no
  probe, and instruments scored on many arms. Nothing in the repo patches f_C or revives a capability (audit). A
  step, not a patch: the reuse line's first apparatus.
- **Decisions.** Relies on the reuse-line definition row and the "GPU without explicit approval when audited" row.
  Departures: a new task family and model (PCFG minGPT, ~3M); a decomposition we train ourselves (C2) against `GOALS.md`
  "Not the goal: re-running the decomposition": C2 needs the user's go-ahead (card item iii). Def 2 read literally
  makes cipher and copy reuse (card item ii); Stage B reads lowercase only until that is decided.
- **What changes next.** Anchors read as leaned → the two checks are the line's ground truth and Stage C scores
  instruments. Checks disagree on the anchors → Def 2's practical form is wrong; no instrument is scored, card to the
  user. An instrument passes and VPD later fails (C2) → the sceptic's reading holds and the log says so.
- **Control first.** A0 rehearsal, then A1 on the pretrained models only; A1 fixes A2's bars, filed before any FT.
- **Numbers.** Per stage ≤ 6, each with its null, positive control and expected value (tables below).
- **Class.** A, B and C1 run under the user's standing permission for audited GPU work (`decisions.md` 2026-10-29),
  $0 on the cluster, not gated. C2 is gated and waits for the user.

## Stage A: Jain PCFG Counter anchors

**Reproduction target.** Table 6 (50K pretraining iterations). Model: minGPT, 6 blocks, 6 heads, d 192, 2-layer MLP
(L1160). Pretraining: AdamW, lr 1e-3, 20% linear warm-up then cosine, next-token loss on the whole sequence, 50K
iterations (L1164-1182), five task families, uniform family prior (L1170). Fine-tuning: answer-only loss, one (task,
operand) = (C, b), AdamW, batch 96, 10K iterations, 20% warm-up, cosine to lr/100 (L1186-1196).

| gap | filled with | source |
|---|---|---|
| pretraining batch | 96 | the paper's only stated batch (FT, L1189); Allen-Zhu & Li 2023 unchecked (A0 notes it) |
| context length | 262 = 7 prefix/suffix tokens + 250 text + answer; pad `#` after the answer | Eq. 1, Fig. 15 (L1029-1058) |
| 250-token window | a random contiguous 250-window of the expansion; an expansion < 250 is discarded and resampled | "randomly subsample" to remove start bias (L1025-1026) |
| dropout, weight decay, betas, clip | 0.1, 0.1, (0.9, 0.95), 1.0 | minGPT model/trainer defaults (the paper's named code, L716) |
| vocabulary | {a,b,c}, 5 family tokens, numbers 0-250, `$ < = #` | Fig. 15-16 |
| spurious strings (C_Tr) | in the counted window (last 40 text tokens) only, replace c by b until count(b) = count(a)+1; length and count(a) unchanged; reject and resample if count(b) > count(a)+1 or too few c; class histograms matched to C_Tr = 0 | "adding the target tokens" (L1086), histogram match (L1088-1116); replacement, not insertion, keeps the window and count(a) fixed (departure) |
| read position | `=` (ART), the position that predicts Ans, for every probe and patch | L1247 says "the answer token"; at Ans the input is the answer itself (review S1) |
| probe | linear, vocab-sized softmax, per block output, 4K AdamW iters lr 1e-3 ×0.1 at 2K/3K/3.5K, fresh C_Te = 0 samples per batch, 2K held out | L1238-1248 |
| data | pre-generated shards (CPU Slurm job); text generated once, labelled per prior | notes (c) |

**Models.** LOW (0.999, 0.001, 0), HIGH (0.5, 0.3, 0.2) (L1173-1178). NEVER (0, 1, 0): never trained to count a. The
token `a` trains as text, but `C a 40` is a prefix NEVER never saw; that is intended ("never had the task" exactly),
and it makes NEVER-based scratch training slower, a bias toward "existing" in the reverse fine-tuning instrument
(Stage C), stated there. Jain's Scr.+FT base is "pre-trained to count O_FT" (L457-460); its prior is not given.

**Fine-tunes (count b).** The LOW grid of Table 6 at lr ∈ {1e-4, 1e-5} × C_Tr ∈ {0, 0.5, 0.8, 1}, plus R; states
saved at 1K and 10K iterations, so 16 LOW states + R. Table 6 cells (count-b acc. on C_Te = 0, 1K / 10K):

| arm | lr | C_Tr | Table 6 | arm | lr | C_Tr | Table 6 |
|---|---|---|---|---|---|---|---|
| N1 | 1e-4 | 0 | 98.9 / 100 | M | 1e-5 | 0 | 32.6 / 99.8 |
| N2 | 1e-4 | 0.5 | 90.2 / 99.9 | M2 | 1e-5 | 0.5 | 12.7 / 99.1 |
| N3 | 1e-4 | 0.8 | 64.9 / 100 | M3 | 1e-5 | 0.8 | 1.4 / 93.9 |
| W2 | 1e-4 | 1 | 0 / 1.9 | W1 | 1e-5 | 1 | 0 / 0 |
| R (HIGH) | 1e-5 | 0 | 99.9 / 99.9 | | | | |

**The two checks (the ground truth).** For an arm and a named capability f_C (count-a or count-b), on C_Te = 0 inputs
where count(b) ≠ count(a)+1, prefix `C b 40` (the arm's own task):
- **Behaviour B.** Share of outputs equal to g(f_C(x)) for the g the fine-tuning data rewards: count_a(x)+1 for
  count-a, count_b(x) for count-b. No probe.
- **Patch P.** 1K pairs (x, donor x′) with the same value of the other count and a different f_C. Replace the residual
  at `=`, output of block l*, in the run on x with the run on x′. P = share of pairs where the patched output equals
  the model's own output on x′ and that differs from its output on x. P needs no g and no probe: a model that does not
  use f_C keeps its output, since the donor agrees on everything else it might use.
- **References (A1).** Positive control: the same patch on LOW's own count-a task (`C a 40`) and on HIGH's count-b
  task. Null: the count-a patch and B on NEVER with `C b 40` (counts b, never counted a); the count-b patch on LOW
  with `C a 40`. l* = the earliest block whose positive-control P is within 3 s of its maximum over blocks (l*_a on
  LOW, l*_b on HIGH). Behaviour's positive reference: the base's own accuracy on f_C.
- **Bars (filed rule, numbers from A1, filed in `predictions.json` before A2).** For each check, n₀ = null mean,
  p₊ = positive reference, s = max(sd over 3 independent sample sets, binomial sd of n₀ at n = 1K). lo = n₀ + 3 s;
  hi = n₀ + 0.5 (p₊ − n₀). If hi ≤ lo the check is void for the whole stage (apparatus; stop).
- **Rule.** "Goes through f_C" iff B ≥ hi_B and P ≥ hi_P; "does not" iff B ≤ lo_B and P ≤ lo_P; else "neither". A
  check whose reference failed is void for that arm, and an arm with a void check reads "neither". Edge cases filed
  with it: fewer than 500 qualifying pairs → void; ties at a bar count as "neither".

**Possession (Def 1, bases only).** Per block, held-out accuracy of the probe for f_C at `=` (count-a on `C a 40`,
count-b on `C b 40`). Controls: the shuffled-label probe on the same model and block; for count-a also NEVER's probe.
Margin m = acc − max(controls); s = sd over 3 probe seeds. Possessed iff at some block m ≥ 3 s and m ≥ 0.5 × m₊, with
m₊ = LOW's count-a margin at its best block (the positive control: LOW counts a at 99.9%). Void if m₊ < 6 s. The
majority-class rate is reported beside each accuracy.

**A0 rehearsal ($0, cluster debug, ~10 min).** 500 pretraining iterations, one 200-iteration FT, one probe, one
patch, one pruning, one reverse fine-tune, saves and a resume. Measures ms/step, shard rate, memory, disk. Stop: any
stage fails to run end to end.

**A1 (control stage; pretrained models only, before any fine-tune).**

| number | LOW | HIGH | NEVER | stop if |
|---|---|---|---|---|
| count-a / count-b behaviour, 4K fresh C_Te = 0 | ≥ 95 / ≤ 20 (Table 6: 99.9 / 5.2) | ≥ 95 / ≥ 95 (99.8 / 99.7) | — / ≥ 95 | a cell misses: recipe |
| count-a probe margin m, best block | m₊ ≥ 6 s | ≥ 0.5 m₊ | NEVER's accuracy is a control | m₊ < 6 s: probe |
| count-b probe margin, best block | **PA4** (open) | ≥ 0.5 m₊ | — | HIGH misses: probe |
| patch P, positive control at l* (count-a LOW, count-b HIGH) | expected ≥ 0.8 | expected ≥ 0.8 | — | hi ≤ lo |
| patch P null, B null (NEVER, LOW `C a 40`) | expected ≤ 0.05 | | expected ≤ 0.05 | hi ≤ lo |

**A2 readings and expected values (leans, not labels).** W1, W2 (both states): through count-a (B and P high).
N1, N2 (both), N3/M/M2/M3 at 10K: not through count-a. M2, M3 at 1K: through count-a (count-b at 1-13%). M, N3 at 1K:
likely "neither" (count-b 33-65%). R: through count-b, and not through count-a. N1 is **new** iff LOW does not
possess count-b (PA4); if it does, N1 reuses a count-b readout LOW had but did not output. The plan cannot exclude a
shared counter re-bound to a new operand: under the rule that is "not through count-a", and it is reported as such.

**Predictions (file in `experiments/pcfg_anchors/results/predictions.json`; A2 bars added, dated, after A1).**
- PA1: A1 reads as tabled. Lean high for behaviour (Table 6), medium for the patch rows (no prior patching on PCFG).
- PA2: count-b accuracy on C_Te = 0 at 10K within 15 points of Table 6 in ≥ 7 of the 9 10K cells. Reported, not a
  gate (labels come from the checks). Lean medium-high.
- PA3: W1, W2 read "through count-a" at both states; N1 reads "not through count-a" at both; R reads "through count-b".
  Lean medium: W's output on C_Te = 0 is unknown (Table 6 says only that it is not count_b); W could be "neither".
  Blind to: a W that computes count-a after l* from text positions (P low, B high → "neither").
- PA4: LOW possesses count-b. Lean 50%: count-b output is 5%, but every family counts tokens, and Jain's Fig. 8 shows
  readouts outliving behaviour. Noise: two FT seeds of W1 and N1, reported, not used for bars.

## Stage B: lowercase on the 67M, f_C = context case

One arm: E53's lowercase fine-tune (Adam 3e-5, 300 steps; E53 job 1's state; Stage B waits on it). The cipher, copy
and hand-write arms are out of this plan until card item (ii) is decided. Case positions as E53.

| number | how | null / control | positive control | expected |
|---|---|---|---|---|
| Def 1 probe at W₀: context case (lowercased vs cased text) at the query position, balanced accuracy | rows and words held out | shuffled labels | none on the 67M: bar = halfway from 0.5 to 1 (0.75) and ≥ control + 3 s | ≥ 0.75 |
| behaviour: E53's cased-context read (log-odds lowercase − capital, case positions, cased rows) | E53's bars: ≤ −1 through context case, ≥ 0 a bias | zero-update state | W₀ on cased rows (≤ −1 by E53 A0's design) | reported; no lean |
| patch: cased run, residual at the query position from the lowercased text, block l*; P = (FT gain, patched − FT gain, unpatched) ÷ FT gain on lowercase rows, gain = FT − W₀ log-odds | through iff P ≥ 0.5 (full = 1), not iff P ≤ 3 s (row bootstrap), else neither | a bias gives P ≈ 0 | the patch on W₀ moves its log-odds to the donor's (fixes l*: earliest block within 3 s of max) | reported; no lean |

Read only when the lowercase-row gain exceeds its measured noise floor (`decisions.md`, E53 open questions); else
the arm is "too small to read". Control first: zero-update state and W₀'s probe controls before the arm. VPD's r and
the steering share on this one arm are reported, not scored (n = 1).

## Stage C: score instruments against the ground truth

**Arms.** The labelled LOW states (up to 16; "neither" excluded) for every instrument; R for steering only (pruning
and reverse fine-tuning revive count-a, which R never lost; C2 decomposes LOW only). Each instrument labels an arm
"existing" or "new" by its own filed rule; truth "through count-a" (R: "through count-b") = existing.

**Pass bar.** a = the smallest agreement count with P(Binomial(n, 0.5) ≥ a) ≤ 0.05, raised to one above the
majority-class count if that is higher (a constant labeller then fails). At expected n: n = 8 → 7 (p = 0.035);
12 → 10 (0.019); 14 → 11 (0.029); 16 → 12 (0.038). Expected split from the leans: 6 through, 8 not, 2 "neither".
A failed instrument baseline voids that arm for that instrument (n shrinks), never counts as a vote.

- **C1 ($0, cluster).** Pruning revival (Jain Alg. 3): remove the top-K MLP hidden units by grad·weight on the count-a
  loss, K ∈ {5, 15}; "existing" iff count-a accuracy after pruning exceeds the mean over 20 random-K draws by ≥ 3 sd
  (floored at the binomial sd) at either K. Reverse fine-tuning: steps to 95% count-a on C_Te = 0, cap 10K; scratch =
  NEVER fine-tuned on count a at the same lr; "existing" iff steps ≤ 0.5 × scratch's (cap counts as scratch's value;
  void if scratch takes ≤ 1K). Recipe: L1249-1252 says reverse fine-tuning uses the Tracr FT recipe (SGD + momentum,
  2K warm-up) at an lr below the FT lr. Departure: we use the PCFG FT recipe (AdamW, batch 96, 20% warm-up, cosine) at
  lr = arm's FT lr / 10 (1e-5 after 1e-4; 1e-6 after 1e-5; Fig. 76/77 use these after a 1e-4 FT), because the Tracr
  SGD rates (1e-1 to 1e-4) have no AdamW equivalent on minGPT; scratch runs at both lrs. NEVER never saw the `C a`
  prefix, so scratch is slower than Jain's Scr.+FT: a bias toward "existing", reported beside the score. Steering
  (E52 B3): mean fine-tuned-minus-base residual at `=` per block added to the base; "existing" iff gain share ≥ 0.5.
- **C2 (gated; user go-ahead, card item iii).** One VPD decomposition of LOW with `param-decomp` (minGPT ported to its
  registry), then the route split's re-gating share r ("existing" iff r ≥ 0.5, E12's rule) on every labelled LOW
  state (VPD, not a crosscoder: it is the project's instrument). Gates: pre-mortem (hyperparameters with no
  published 3M values, memory, saves), every setting sourced (Goodfire's 67M recipe scaled, stated), $0 rehearsal (a
  1K-step decomposition on cluster debug; a 3M decomposition may fit a 2 h job), early checks (reconstruction loss at
  step 500 filed), two fresh-agent reviews, the user's go-ahead with the rehearsal's numbers.

**Predictions.** PC1 (pruning) passes: lean low (Fig. 76/77: revival at every C_Tr after a 1e-4 FT). PC2 (reverse
fine-tuning) passes: lean low (same figures; Fig. 9 puts LOW's revival at ~3K, near the scratch line). PC3 (steering)
passes: lean medium. PC4 (C2, VPD r) passes: lean low-medium (a 3M model's VPD is untested). Each is blind to
"neither" arms.

## Decision tree

1. A0 fails → fix, rerun ($0).
2. A1 stop row missed → stop; numbers to the user (Blocking card). Else compute bars by the rule, file them (dated)
   with PA3's thresholds, then A2.
3. A2: PA3 holds → the checks are the ground truth; Stage B and C1. W1 or W2 reads "not through", or N1 "through" →
   filed prediction reversed: stop and tell the user. B and P disagree on ≥ 3 states → the null; card to the user, no
   scoring. Fewer than 8 labelled states → C1 reported without pass bars; C2 not proposed.
4. Stage B: the reading is reported as it comes; no branch depends on it. Its control failing → stop that stage only.
5. C1: each instrument's pass/fail is reported. Then C2's gates; C2 runs only after the user's go-ahead.

## Cost and routing

| stage | compute | where | $ | wall |
|---|---|---|---|---|
| A0 | tiny sizes | cluster debug | 0 | 10 min |
| A1 | 3 pretrainings × 50K (notes: 15-25 ms/step H100, L40 ~1.5-2×), checkpoint + resume, probes, patches | cluster debug, 2-3 jobs | 0 | 1-2 h |
| A2 | 9 FTs × ~4 min, checks on 17 states | cluster debug, 1-2 jobs | 0 | 1 h |
| B | probe, patch, read on existing states | cluster debug, 1 job | 0 | 30 min |
| C1 | 16 reverse FTs + 2 scratch (≤ 4 min each), pruning, steering | cluster debug, 2 jobs | 0 | 1.5 h |
| C2 | port + 1 decomposition + 1 sweep, 2-4 H100-h, up to 2× | rehearsal on cluster, then RunPod H100 | 10-30 | 1-2 days agent |

Without C2: $0, ~3 days of agent time (`src/commute/pcfg.py` ~500 lines; `experiments/pcfg_anchors/run.py` < 300).
A runs on the cluster only; any RunPod stage gets the fresh-Opus code review first.

## Order of work

Read this plan, the notes, the audit, `src/commute/README.md`, E53's code. Write `src/commute/pcfg.py` (grammar,
labelers, window, spurious b, shards), `interchange.py` (behaviour, patch, probe), `revival.py` (pruning, reverse
FT); `experiments/pcfg_anchors/run.py` stages `a0 a1 a2 c1`; `experiments/reuse_arms/run.py` (B). File PA1, PA2, PA4
and the bar rule (edge cases run first); fresh-Opus code review; `/pod-runner` A0, then A1 alone; file the bars;
then A2. Each stage: `/writeup` report with "Not run", `TRIED.md` row, `PENDING.md` line.

## Not in this plan

Indexer and Tracr settings; TinyStories (no scalar f_C); Qwen arms (no anchor); the cipher, copy and hand-write arms
(card item ii); probe-direction ablation (reported at most, never in the rule); a crosscoder or LoRA-SVD instrument;
200K pretraining (Table 5); a second decomposition for R; editing E53's filed predictions (its cased-context read is
reused as a number, its bars untouched).
