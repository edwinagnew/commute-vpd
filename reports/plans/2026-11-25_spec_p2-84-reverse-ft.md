# P2.84 spec: Jain et al.'s reverse fine-tuning on Fig 4's 48 states

Cell P2.84 (due 11-26, 2 runs): one per-task panel for Fig 6, Jain et al.'s reverse fine-tune read on all 48 states,
with a passing positive control and a random-update null. E312. Checked by redteam
(`reports/reviews/2026-11-25_review_p2-84-spec.md`); **the Amendment at the end replaces § 3's positive control and
settles the review's points; where it and § 3 disagree, the Amendment holds.** Code `experiments/reverse_ft_fig4/`,
filed rules `experiments/reverse_ft_fig4/results/predictions.json`. Jain et al. is the 2023 preprint (ICLR 2024),
cited as 2023 as in `refs.bib`.

We take each fine-tuned state and train it back toward the original model's outputs on the fine-tune's own inputs, and
count how many steps it takes to undo half of the change. Jain et al. read a quick return as a sign that the old
ability was only wrapped, not removed. The panel reports that count for each task beside a random update of the same
size and beside gain-only fine-tunes, which are wrappers by construction. It gives no verdict on reuse.

**The method has already failed once on 9 of these 48 states** (E55 C3 and its step ladder, TRIED rows 120-121). The
same port, `revival_kv.reverse_curve`, ran on E58's rank-1, cipher and copy maps, the `c3/*` states here. It undid
the old-by-construction rank-1 map *more slowly* than the new cipher: t_half 13.3 against 10.1 steps, the wrong way, at
every update size. Its built control (a planted 13-neuron wrapper, 3.97 steps) passed. The ladder's untested
candidate was that the method reads retention of M's behaviour, which every map keeps, and not reuse. This panel puts
that reading on all 16 tasks. It is not a new test of whether the method separates reuse from new.

## 1. Jain et al.'s procedure (arXiv 2311.12786; line numbers in `~/commute-data/lit_review_pdfs/Jain2024_finetuning_wrappers_2311.12786.txt`)

- **What it is** (l. 270-272): "a methodology called reverse fine-tuning (reFT), wherein one takes a pretrained model,
  fine-tunes it on a downstream dataset, and then fine-tunes it again in the 'reverse' direction, i.e., on a dataset
  sampled from the original pretraining distribution."
- **Their reading** (l. 285-287): "if the behavior corresponding to a pretraining capability is retrieved in a few steps
  of reFT, fine-tuning did not meaningfully alter said capability". Revival (l. 194): "the model starts to perform well
  on the pretraining task again".
- **The PCFG run** (l. 401-402): "use a model fine-tuned to count OFT and reverse fine-tune it to re-learn the ability
  to count OPT". Measured as OPT accuracy against iterations, log scale, test spurious correlation CTe = 0 (Fig. 9,
  l. 442-456): "very quickly (0.1–1K iterations)" for strongly relevant capabilities, "slightly slower (3K iterations)"
  for weakly relevant ones.
- **Their baseline** (l. 457-462): "Scr. + FT, wherein the model is initialized with parameters pre-trained to count
  OFT and then fine-tuned to count OPT", which "only reaches perfect accuracy at 4.5K iterations" (l. 453-455). The
  randFT appendix adds Scr., a model trained from scratch (l. 1512-1516).
- **Model and settings.** minGPT on PCFG counting (and compiled Tracr models, and TinyStories). PCFG forward fine-tune
  (l. 1191-1195): "Large LR (ηL: 10−4), Medium LR (ηM: 10−5) and Small LR (ηS: 10−6). Fine-tuning is done for 10K
  iterations using AdamW optimizer with a batch size of 96 samples. ... cosine learning rate with an initial warmup of
  20% of the fine-tuning iterations." reFT (l. 1249-1252): "Same set of hyperparameters as used in the fine-tuning ...
  except for the learning rate, which we force to be smaller than the corresponding fine-tuning learning rate". The
  reFT rate is not swept (l. 1462-1463).
- **TinyStories** (l. 1619-1624): reFT "on a dataset of stories which all have Twists in", against "a model
  pre-trained on data with no Twist stories, which is then fine-tuned on Twist stories". It reads loss and the share
  of generations with a twist at iterations 0/30/300/3000 (Table 1, l. 557-573): fine-tuned models recover in "30–300
  iterations", and the baselines "never reach the same loss" (l. 555-556).
- **What it says about wrapper versus new.** A fast revival means the capability was kept under "a minimal
  transformation, which we call a 'wrapper'" (l. 22-23). The comparison is always against a model that never had the
  capability. There is no cut-off: the paper plots curves and reads them by eye.

## 2. Our mapping on the 67M model

M is Goodfire's 4-layer 67M Pile model (W0). F = W0 + δW is one of the 48 saved states: 16 tasks × seeds 0-2, in
`experiments/known_3seeds/results/known3.json` `.cells`. The file and sha256 of each state are already resolved in
`experiments/weightwatch/p2-80/predictions.json` `.settings.fig4` (E306, `p2-80/resolve.py`), and they are loaded
as WeightWatch loads them (`weightwatch/run.py:318-322`, `OT.load_dw`). The task windows come from E136's inventory
(`share_untrained/results/inventory.json` `.tasks`: kv, windows_pt and e53 formats; e58's from `refit_96key`).

| | Jain et al. | Here |
|---|---|---|
| the capability | the pretraining task (count OPT), by accuracy | M's whole next-token distribution at the task's read position |
| reFT data and loss | pretraining-task data, cross-entropy on its labels | the task's own training windows; loss KL(p_M ‖ p_F) at every position (C3's `finetune.finetune_to_reference`) |
| how far F is from it | accuracy drop | D = κ(F) = mean KL(p_M ‖ p_F) at the read position over the held-out score half |
| revival reading | iterations to regain accuracy (curves) | **t_half**: the interpolated step at which κ first falls to D/2 (`revival_kv.half_life`) |
| baseline | Scr. + FT | the state's own random-update null (§ 3), and gain-only wrappers |

**Per state we run** the reverse fine-tune from W0 + δW: Adam at 3e-6, batch 32, clip 1, no warm-up, the 24
decomposed matrices trainable, seed = the state's seed, cap 1000 steps (C3's `ReferenceConfig`, `finetune.py:238-246`),
with κ read after every step. **Per state we read** t_half, with D, the update norm and the whole κ curve saved
beside it. The code calls `revival_kv.reverse_curve` and changes only the loader: it reads the task's train and
held-out windows and its read position, which is the answer position for kv tasks, the read position h for e62, and
every position of the held text for e53. `commute.revival` itself (C1/C1b) is minGPT-only and frozen, and
`revival_kv` is its 67M port. In our terms this is a descriptive distance-recovery reading per state: no split and no
decomposition. It reads how quickly M's behaviour comes back, not which of M's parts F's behaviour runs on.

## 3. Controls, null and expected values (to file before running)

- **Zero update (do-nothing reference, W0).** D = 0 exactly, and t_half is void (`half_life` returns None at
  curve[0] ≤ 0). C3's `zero_D` was 0.
- **Identity.** After every reverse fine-tune the target weights equal W0 bit for bit (C3 restored them on 36/36,
  `w0_restored`). Any difference stops the job.
- **Reproduction.** c3/rank1_s0 and c3/cipher_s0 read again give C3's t_half: 13.91 and 10.69 steps
  (`open_pair_instruments/results/run.json` `.states.{rank1_s0,cipher_s0}.t_half`). The tolerance is set from a
  measured floor: two reads of the same state on the same card. They should be bitwise equal under a fixed seed and
  the same batch order. If they are not, the gap between them is the floor.
- **Positive control: wrappers by construction.** These are E290's FFN-gain states,
  `experiments/ia3_control/results/large/gains_ffn_{rank1,copy,prior_cipher,elicit}_s{0,1,2}.pt` (12 states). They
  change only a gain per MLP neuron, so every weight direction of M is kept. Each gain g_j is folded exactly into
  column j of `down_proj` as δW = W_down · diag(g − 1), because the gain sits after the GELU (to be checked against
  E290's `run.py` in Phase B). The cipher is excluded: its gain arm learned nothing (gain share 0.02, TRIED E290).
  The reference is the full fine-tune of the same task and seed (e58/*, and refit_96key's elicit). **Pass:** on at
  least 3 of the 4 tasks, mean t_half(gain) is below mean t_half(full) by at least 3 × the larger seed sd, floored at
  1 step. This is C3's ladder rule (`revival_kv.pole_gap`, floor 1 step), not a new cut-off. Expected: gain-only
  near the planted wrapper's 3.97 steps (`run.json` `.stage0.planted`), full fine-tunes 8-18 steps (C3, ladder). A
  known confound works against the pass: gain states do less of the task (E290 gain share 0.026-0.80), and on the
  ladder t_half *falls* as the update grows (corr −0.90 with norm). A smaller update therefore predicts a longer
  t_half, not a shorter one.
- **Random-update null, one per state.** δW_null,m = Q_m δW_m with a Haar-random orthogonal Q_m for each matrix m.
  The seed comes from the state and the matrix, as WeightWatch's sham does (`weightwatch/run.py:391-400`). This keeps
  each matrix's singular values, so its Frobenius norm and spectrum match the state's. It is the same kind of null as
  the Fig 6 share panel's rotated SVD. The null is reverse fine-tuned by the same recipe, and its D is recorded.
  Expected: D_null well below D (random directions mostly miss the read position). Its t_half is unknown: it may be
  censored at the cap or very short. Three states are read first, and the null's t_half there is reported before the
  rest run. A null with D_null below a measured κ floor gets a void t_half, and that count is reported. No band is
  guessed: the panel shows each state's t_half beside its null's.
- **Run order.** Zero update, identity and reproduction first, alone. Then the 12 gain-only states and their 4
  full-fine-tune references. If the control fails, the job stops before the 48 states. Then the 48 states, each
  followed by its null, each written as it lands.

## 4. Departures (summary; the full table is Appendix A)

The model and data are ours (67M Pile LM, the key-to-value, text and induction tasks), not PCFG, Tracr or
TinyStories. The capability is M's distribution and not a labelled task, so the loss is distillation and not label
cross-entropy. The reading is a half-life of KL and not steps to an accuracy. The baseline is a matched random update
and not a scratch model. There is one reverse rate, constant, at a tenth of the forward rate, capped at 1000 steps
rather than 10K. Only the 24 decomposed matrices train. There is no cut-off and no labels, unlike C1 and C1b.

## 5. Cost

Measured time per item, including pruning, which this run drops: **1.52 s** mean (n 32, 1.31-1.81 s), from
`experiments/open_pair_instruments/results/run.json` `.states.*.seconds` (C3, MATS debug, jobs 46808/46815/46840).
The ladder's mean was **2.98 s** (n 36, max 3.26 s), from `ladder.json` `.ladder.*.seconds` (job 46954, whose whole
job took ~5 min). Every C3 reverse halved within 9-19 steps (`run.json` curve lengths). That makes a step about
0.1-0.15 s with the per-step κ read.

- Items: 48 states + 48 nulls + 12 gain-only + 1 zero + 2 reproductions = 111.
- Typical (halving in < 20 steps): 111 × 3 s ≈ 6 min, plus model and window loads for 6 task families (unmeasured,
  ~1-2 min each by C3's loads) ≈ **15-20 min**.
- Worst case (every null censored at 1000 steps): 48 × 1000 × 0.15 s ≈ 2.0 h, ×2-3 for the longer e53 text and the
  96-key windows (unmeasured) ≈ **4-6 h**.
- **Route:** the MATS `debug` QoS, $0, if it starts at once, since the typical case is under 30 min. Otherwise one
  H100 pod at $4.40/h: typical ≈ 0.4 h ≈ **$2**. Worst case on 3 parallel pods (states split by task family, ≤ 4 pods
  per runner) ≈ 2 h each ≈ **$26**. Not gated (under $60 and under 6 h). A paid pod needs the $0 rehearsal on record
  first: the 3-state null probe in stage 0 is that rehearsal, run on debug.

## 6. What the reading is blind to

- **Reuse itself.** t_half reads how fast M's behaviour returns. It does not read whether F's new behaviour runs on
  M's parts (the frozen definition's needed variables and "kept"). On C3's maps every state keeps M recoverable, and
  the rank-1-versus-cipher gap came out wrong-way.
- **Update size.** t_half fell with norm on the ladder (corr −0.90). The norm and D are reported beside every value
  so that a task ordering which only follows size can be seen.
- **The prior.** C3's rule 6 found readings tracking how likely M already found F's answers.
- **Behaviour away from the read position**, other inputs, and other reverse rates (one rate, unswept, as Jain).
- **A ceiling at small updates.** Fine-tunes of 10-300 steps sit close to M. If most states halve in under 20 steps,
  the panel may have little spread to show.

## Open questions (they change the reading; not decided here)

1. **Known failure on 9 states.** C3 already found this port reads rank-1 versus cipher the wrong way. Is a
   descriptive 48-state panel what Fig 6 wants from it, or should the panel caption carry C3's result?
2. **Null matching.** The null here is matched on each matrix's spectrum (and so its norm), as the other Fig 6
   panels are. The alternative scales the rotated update until D_null = D (C3's planted bisection), which removes the
   tiny-D void but stops matching norm. The two can order tasks differently.
3. **The reverse target.** Here it is distillation to M on the task's inputs (C3). Jain's literal recipe is
   cross-entropy on data "sampled from the original pretraining distribution", which would be Pile text. That
   recipe reads general retention and might not touch the task's read position at all.
4. **Half-life or near-full recovery.** Jain read "reaches perfect accuracy". Steps to κ ≤ 0.1 D would sit closer to
   that, but censors more often.
5. **The positive control's partner.** A wrapper is compared here with the full fine-tune of the same task. If the
   full rank-1 fine-tune is itself a wrapper, a tie is the right answer, and the 3-of-4 rule would call it a fail.

## Not run

Nothing. No code exists for P2.84. The per-step time on the 96-key windows and the e53 text, the load times, and the
nulls' D and t_half are all unmeasured.

## Appendix A. Departures from Jain et al. 2023 (and from our C1, C1b and C3)

| # | Jain et al. (line) | Here | Reason |
|---|---|---|---|
| 1 | minGPT on PCFG / Tracr / TinyStories (l. 1164-1195, 1600-1603) | Goodfire's 67M Pile LM, Fig 4's 16 tasks | the paper's testbed and Fig 4's states |
| 2 | the capability is a labelled pretraining task (count OPT), read by accuracy (l. 401-402) | M's next-token distribution at the read position, read as KL(M ‖ F) | the 67M has no labels for these keys; C3 § 9 |
| 3 | question: did the fine-tune *delete* a capability? (l. 87-89, 1505-1510) | descriptive per state: F *taught* a task; how fast does M's behaviour come back? | the cell asks for the reading on Fig 4's states, which are not deletion fine-tunes |
| 4 | reFT data from the pretraining distribution, cross-entropy (l. 270-272) | the task's own train windows, KL(p_M ‖ p_F) at every position | C3's port; M's outputs are the only labels; open question 3 |
| 5 | iterations to target accuracy, curves read by eye (Fig. 9, Table 1) | t_half, interpolated; whole curve saved | continuous, never at a ceiling; C3's reading |
| 6 | baseline Scr. + FT / Scr. (l. 457-462, 1512-1516) | a per-state Haar-rotated null; no scratch model | a 67M with no trace of M does not exist, and a randomly initialised one would be censored at any cap, so it passes nothing |
| 7 | AdamW, 20% warm-up, cosine, batch 96, 10K iterations (l. 1191-1195) | Adam, constant 3e-6, no warm-up, batch 32, cap 1000 | C3's `ReferenceConfig`; every C3 reverse halved by step 19 |
| 8 | reFT lr below the forward lr (l. 1249-1252) | 3e-6 = a tenth of the forward Adam 3e-5 (E12, E53, E54, E58) | as C3 and C1 (lr/10); e62's forward lr to be checked in Phase B |
| 9 | all parameters train | the 24 decomposed matrices (embeddings and norms frozen) | the same matrices every Fig 4 state changed |
| 10 | no built positive control (Tracr is compiled, l. 1462-1465) | gain-only (FFN-gain) states as wrappers by construction (E290) | the cell asks for a passing positive control |
| 11 | C1: steps to 95% count-a accuracy against a NEVER-prior scratch, guessed 0.5 × scratch cut-off (`revival.py:261-268`) | no cut-off; per-task values beside a null | C1's guessed cut-off failed while its raw readings separated (TRIED E55 C1) |
| 12 | C1b: cut-off from calibration seeds; passed only where the label was the task's identity (TRIED E55 C1b) | no labels, no calibration split | the panel is descriptive; C1b's pass could not separate methods |
| 13 | C3: the same port on 32 states with pruning beside it, 8 seeds, placed on the rank-1-to-cipher axis | 48 states, 3 seeds, no axis, no pruning, random null and gain-only control added | the cell; C3's axis rules are not reused |
| 14 | code: the paper names no code release for its PCFG or reFT runs; its TinyStories code "is based on" karpathy/llama2.c (l. 1630) | our port, `revival_kv.reverse_curve` on `finetune.finetune_to_reference` | the project rule is the authors' pinned code (user, 2026-11-17), but there is none to pin, their testbed is minGPT on PCFG, and reFT is plain fine-tuning on the 67M's own loop; the port already ran on these states in C3 |
| 15 | reFT runs to the target accuracy | run on to kappa ≤ 0.1 D; t_half read from the same curve, t_tenth saved, never promoted | review SHOULD; the curve prefix to D/2 is the one C3 read |
| 16 | reFT batch as the fine-tune's (l. 1249) | batch = each task's forward batch: 32 (kv, E62), 8 (E53) | the same rule; § 2's "batch 32" holds for 14 of 16 tasks |
| 17 | reFT on pretraining data with cross-entropy (l. 270-272) | secondary only: the 3 probe states also reversed with next-token cross-entropy on Pile train rows 0-4095, read at the same position | review SHOULD; distillation on the task's windows stays the primary |

## Appendix B. Phase B outline (not started)

One `experiments/reverse_ft_fig4/run.py` (~200 lines). Config from `predictions.json` settings via
`config_from_settings`. It loads states through `.settings.fig4` (copied with sha256), windows through E136's Job,
the reverse through `revival_kv.reverse_curve` with `finetune.ReferenceConfig`, the null as the rotation in
`weightwatch/run.py:391-400`, and the gain-only fold from E290's gain files. Stages: 0 (zero, identity,
reproduction, 3-state null probe) → 1 (control; stop on a fail) → 2 (48 + nulls, resumable per state). There is one
plot, per task: three dots per state, its null beside it, and the control on the same axis.

(As built, Appendix B differs: windows load from the filed paths and `commute.windows.read_rows` for E53, not E136's
Job; the null uses `untrained.haar_orthogonal`; stages are 0, items (sharded), summary. See the Amendment.)

## Amendment, 2026-11-25, after the review (the lead's decisions; E312)

**MUST 1: the gate is C3's planted pair.** The gate is no longer gain-only against full fine-tunes. It is C3's
planted 13-neuron wrapper against a planted non-wrapper at the same D.
- The wrapper is rebuilt from `open_pair_instruments/results/measure.json` `.planted` (units, w, alpha). Those were
  saved, so nothing is retrained ($0, seconds).
- C3 never built a non-wrapper partner, so one is built here. Each of c3/cipher_s0-s2 has its δW scaled by one factor,
  found by C3's bisection (`revival_kv.fit_alpha`), until its D equals the wrapper's (4.2068 nats). The cipher is new
  by construction. Both read on C3's windows (the rank-1 file's; the tokens are identical across maps).
- Rule, `revival_kv.pair_gate` (C3 ladder's pole rule): gap = mean t_half(scaled ciphers) − t_half(wrapper), noise =
  max(sd, 1 step). Pass iff gap ≥ 3 × noise. Fail or void stops the job before any of the 48 states.
- Expected: wrapper 3.97 steps (C3), which also counts as a reproduction within 1 step; scaled ciphers 10-14 steps.
  The gain-only states (E290 FFN-gain, 15 states) are shown descriptively beside their e58 tasks, with no rule.

**MUST 2: void, censored and a zero sd**, as filed and as run through `tests/test_revival_kv.py`.
- Void (D ≤ 0) gives None.
- A wrapper that is void, or fewer than 2 non-void partners, gives "void": no call, and the job stops.
- Censored (never halves within 1000 steps) counts as 1000 in the mean: a lower bound, never missing. It is left out
  of the sd, because a cap carries no spread. The CPU test of the first draft caught one censored partner at 1000
  beside two at ~10. That made the sd ~570 and failed a clear pass.
- With under 2 uncensored partners, or an sd of 0, the 1-step floor is the noise.
- A censored wrapper fails, and fails also when all partners are censored (gap 0).
- One void partner leaves 2, which is enough.
- All partners censored with an uncensored wrapper passes (gap ~996 against 3).

**SHOULDs, all adopted.**
- **Null and D.** The null is primary, spectrum-kept: a per-matrix Haar rotation on the output side, from
  `untrained.haar_orthogonal`, with one seeded generator per state. D is plotted under every state/null pair, on the
  same x. Nulls under stage 0's D_void (10 × the batched-against-single κ floor) are counted.
- **Caption.** The figure carries C3's wrong-way result.
- **Code.** Appendix A row 14.
- **Targets and readings.** Distillation stays the primary target. The 3 probe states are also read under Pile
  cross-entropy in stage 0 (row 17). t_half is the reading, and t_tenth (0.1 D) is saved, never promoted (row 15).
- **Fold check.** Stage 0 compares the gain fold with E290's hooks (`gains.applied_ia3`) on 64 e58/copy windows. The
  bar is 10 × the change from moving each folded down_proj weight by one fp32 ulp (E229's factor, as E290).

**Stage 0** stops on any miss:
1. The zero update: D exactly 0 and t_half void.
2. The κ floor.
3. C3's reproductions: rank1_s0 13.91 and cipher_s0 10.69, each within 1 step.
4. The planted-pair gate.
5. The planted wrapper within 1 step of 3.97.
6. The fold.

Then two sizing steps, not stops:
- The 3-state null probe (c3/rank1_s0, e53/lowercase_s0, e62/copy_s0 and their nulls) gives the shard projection.
- The Pile cross-entropy secondary, reported only.

**Items.** The 111 items run as a 4-shard debug array and are summarised without a GPU.
