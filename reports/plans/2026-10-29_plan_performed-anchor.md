# Plan (E53): the stand-in ratio on an anchor the original model performs (lowercase Pile against a token cipher)

**Option A approved by the user 2026-10-29 (ask #8): "go with A, run the 2-minute check first". Stage A0 runs alone first; the fine-tunes wait on A0 and on the open questions below.** Written 2026-10-29
by a design subagent for the regating-lead. Builds on `reports/plans/2026-10-29_plan_restore-anchors.md` (its option C)
and the E50 card in `reports/analyses/2026-10-29_analysis_e50-restore-anchors.md`. Words as in `GOALS.md` → Terms.

## In plain words

Γ's stand-in ratio is the line's only number that has told an existing behaviour's fine-tune from a new one's. It has
only been read on toys the original model does not perform at all. This plan reads it, for free, on two fine-tunes of
the same rows: continuing lowercase text in lowercase, which the original model already does, and a fixed swap of 50
common words, which it cannot. If the ratio reads the
first high and the second low, it earns its place on Qwen; if it reads both alike, it ends as a number of kind.

## Design check (`/design`)

- **Goal served.** `GOALS.md`, "If no simple Φ works": is an update new capability or strengthening? This calibrates the
  re-gating line's one kind-separating number ("Reveal" against "Create") before any Qwen spend.
- **Nearest tried.** E36 Amendment 1 (`TRIED.md`): the same lowercase fine-tune, switching share 0.79 at step 300, but
  no "new" twin, no ratio, no relay number. E12/E46/E50: the ratio separates copies (0.87-0.97) from random maps
  (0.36-0.41), but the original model performs neither, even shown an example (E50 Amendment 1). New here: a pair on the
  same rows whose "existing" member the original model is checked to perform first.
- **Patch or step.** Both: it answers E50's anchor caveat, and it is the calibration step stage 1 on Qwen needs.
- **Decisions relied on.** Γ frozen; do-nothing placement and the least-squares split; fine-tunes of the 24 decomposed
  matrices; no update labelled before anchor readings. Departure: a new task family, so it needs the user's
  confirmation (ask #8). Same rows for both members, so the anchors' domain confound (a `backlog.md` item) is absent.
- **What changes next.** Separates → the ratio goes to Qwen stage 1 (ask #8's C) with the relay number beside it. Both
  high (E36's risk: about 0.8 for any small broad update) or both low → the ratio is withdrawn as a number of kind, and
  "No for the toys" (B) becomes the line's answer on the 67M. Reversed → to the user.
- **Control first.** Stage A0 runs alone, before any fine-tune: does the original model perform each member? Then the
  zero-update split must read 0 for both gains, and W₀'s held-out log-probability on the lowercase rows must reproduce
  E36's −3.146 nats per token within 0.01.
- **Numbers.** The table below; four numbers, one control row each.

## The question and its null

**Question.** On two fine-tunes of the 67M on the same Pile rows, one strengthening a behaviour the original model
already performs and one teaching a behaviour it cannot, does the stand-in ratio read the first at 0.7 or above and the
second at 0.5 or below, on both seeds? **Null.** The ratio reads both alike (E36's "about 0.8 for any small update"), so
its toy separation came from the toy tables, not from kind.

## The pair

- **Existing: lowercase Pile.** Pile rows decoded, lowercased and re-encoded (`row_copy.lowercase_rows`, E36/E37's 2,400
  training rows, 300 steps of 8, Adam 3e-5, all 24 matrices; `lowercase.finetune("adam")`). Scored at *case positions*:
  where the original text had a capital letter at the start of the next word and both forms are single tokens.
- **New: a token cipher.** The same 2,400 cased rows with a fixed derangement π of 50 word tokens of Pile frequency
  ranks 50-100 applied at every occurrence, in context and as targets (token level, no re-encoding; about 2-4% of
  positions). Same recipe. Scored at *cipher positions*: where the next token is π(x).
- **Does the original model perform them? Evidence so far, indirect.** On the lowercased held-out rows the original
  model reads −3.146 nats per token and the fine-tune gains only 0.13 (4%) by step 300 (E36 Amendment 1): it already
  models lowercase text nearly as well as the fine-tune does. The direct number is unmeasured; A0 measures it first:
  - *Lowercase*: the pairwise log-odds log p(lowercase form) − log p(capitalised form) at case positions, in W₀ on the
    lowercased rows. "Performs" = mean ≥ +1 nat. Beside it, the same positions in the cased rows, expected ≤ −1 nat:
    the preference follows the text's style, with no example and no instruction.
  - *Cipher*: log p(π(x)) − log p(x) at cipher positions in W₀. "Cannot" = mean ≤ −3 nats.
  - Stop rule: lowercase mean ≤ 0, or cipher mean > −1 → stop before any fine-tune; the card goes back to the user
    with the numbers and a fallback existing member (repeating an earlier name in the row, which the original model
    does by induction on Pile).

## Predictions (to file after the user's go)

Each fine-tune at two batch-order seeds. Per scored position, a window of the row with that position at the centre
(`half` = 63), so the existing key-position code reads it unchanged. T is the mean log-odds of the true next token
there; the gain G = T1 − T0. Up to 500 windows per task, from held-out rows (Pile val 0-255).

| number | W₀ / zero update | do-nothing | lowercase fine-tune | cipher fine-tune | random δW (matched on off-target KL) | hand write (block 1 MLP output, G-matched) |
|---|---|---|---|---|---|---|
| G, nats | 0 | = the fine-tune's | 1-3 (unmeasured; needs ≥ 1) | ≥ 3 | ≤ 0.6 (E46 Amendment 3: 0.04-0.56) | = the fine-tune's, by construction |
| **P1: stand-in ratio** (masked gain over subcomponent gain, least-squares split; ≈ g/U, the gate gain over the subcomponent gain) | both gains exactly 0 | undefined (subcomponent gain 0 by construction); g reported | ≥ 0.7 (lean, E36's switching share 0.79) | ≤ 0.5 (lean, E12's random maps 0.36-0.41); risk ≈ 0.8 | undefined (subcomponent gain < 1 nat) | near its task's fine-tune (E46 Amendment 3), reported |
| **P2: relay** (every-alive restore share, E50, under do-nothing) | 0 exactly | — | ≥ 0.5 | 0.6-0.8 (E50 random maps 0.66-0.73): lean "does not separate" | unscored | ≤ 0.2 (E50 Amendment 1: −0.16 to −0.32): its low control |
| **P3: hand write minus its fine-tune, on the ratio** | — | — | reported | reported | — | abs ≤ 0.15 means the ratio tracks the task, not the update |

**Separation rule (P1, scored; P2 the same rule, reported).** gap = min over seeds of the lowercase ratio − max over
seeds of the cipher ratio. Separates if gap ≥ 0.3; reversed if (max lowercase − min cipher) ≤ −0.3; unscored if any
fine-tune has G < 1 nat or subcomponent gain < 1 nat; else does not separate. If P1 separates and P3 says "tracks the
task", the reading is "separates these tasks", with the hand write named as the reason it may not be kind. The cipher
is saved at steps 10, 100 and 300; the step whose G is nearest the lowercase fine-tune's is read beside it as a
matched-gain row, reported only.

**Blind to.** *Ratio*: the real model (read in Γ's stand-in); whether the subcomponent changes themselves matter (the
part beyond g is 0.01-0.05 of the subcomponent gain on every toy row); a write that builds nothing can read like its
task. Never call it "the share Γ keeps". *Relay*: kind; it measures gain the original weights pass along against gain
the update delivers directly (E50 Amendment 1). *A0*: it shows the original model performs the token-level choice,
not that the fine-tune uses the same subcomponents to do so. Both readouts see only the scored positions.

**Noise floor.** The ratio's 90% interval over windows was ±0.024-0.034 at 97 windows (E46 Amendment 3), so about
±0.01 at 500; seed-to-seed spread up to 0.06 (E12); one restore in bf16 rounds by about 0.02 nats. The 0.3 bar is
five times the seed spread.

**The method's limit, in plain words** (`decisions.md`, "Limit of the elicitation reading"). VPD's smallest unit is one
rank-one subcomponent, not a circuit. A new mechanism assembled from subcomponents the original model already has, under
new gating, reads as "elicited" to every number here. The cipher's answers are common words, whose output pieces already
exist, so the cipher may read "elicited" for that reason alone. A high reading means "uses pieces the original model
has", never "uses a circuit it already runs"; only a Γ from a second decomposition seed narrows this.

## Compute and code

MATS cluster, `debug` QoS, L40, $0, via `/pod-runner`. Job 1 (about 10 min): A0, then the four fine-tunes and the
cipher's step 10/100 states, saved to `results/large/`. Job 2 (about 15-20 min): controls, the nulls (random δW, hand
writes), then the fine-tunes' ratio, gate gain and relay. Not gated (under $15 and 3 h). Agent time about a day: one new
`experiments/performed_anchor/run.py` of about 300 lines, then the fresh-Opus code review.

Reused unchanged: `row_copy.lowercase_rows`, `lowercase.task`, `finetune.finetune_arm_b`, `finetune.random_direction`,
`local_faith.route_split`, `route_cells`, `masked_lo_windows` (the parts E46's `cells` composes), the least-squares split,
`recruitment.trigger_io`, `restored_at`, `perturb.key_write`. Small additions to `src/commute/`: a seed argument to
`lowercase.finetune`; a per-window "true next token" option in `data.target_tokens`; a cipher-rows function; inputs for
the hand write grouped by next token rather than current token. `regating_table/run.py` and `restore_anchors/run.py`
are not imported or edited (E46's `code_unchanged` guard).

## Open for the user

1. Is a 50-word cipher in context and output the "new" member you want, or output only (input text left cased)?
2. Is "prefers the lowercase form by at least 1 nat at case positions" an acceptable test of "performs unprompted"?
3. If lowercase gains under 1 nat at case positions, read the ratio with a lower gain floor set by the measured noise,
   or train longer?

## Not run

Everything: A0 (the original model's pairwise log-odds on both members), the fine-tunes, every number in the table.

## Filed (E53 fine-tunes), 2026-10-29

A0 ran and passed (`results/a0.json`, GO). The user answered the three open questions (`decisions.md`, ask #9): the
cipher swaps words in context and outputs, lowercase counts as performed, and a fine-tune under 1 nat is read above a
noise floor rather than trained longer. Code: `experiments/performed_anchor/run.py` (`finetune`, then `read`),
`score.py` (the rules), `cluster/performed_anchor.sbatch <stage>`. Predictions: `results/predictions.json`, filed before
any run; every rule, threshold and tolerance is there. In plain words, what the code does beyond the table above:

- **Gain floor.** A row is read when its G and subcomponent gain both reach min(1 nat, F), F = max(0.1, 2 × the larger
  gain of the task's KL-matched random δW, 2 × the row's standard error over windows). So 1 nat or more is read as the
  plan filed; under 1 nat is read above the noise. If F cannot be measured, the 1-nat bar stands.
- **Batch order 1** is a fixed permutation of the same 2,400 rows (`lowercase.batch_order`); order 0 is E37's.
- **Cipher rows** are cased Pile train rows 0-2399 (the lowercase rows are the same documents, lowercased and joined);
  its held-out rows for the training log are ciphered val rows 0-63.
- **Windows.** 500 per task drawn (seed 0) from the positions whose 127-token window fits; 4,598 lowercase and 2,198
  cipher positions qualify (tokenizer-only check). Cells are read batch by batch: 500 windows' gate patterns would take
  ~10 GB of host memory each.
- **Hand write keys** are the distinct true next tokens of up to 2,000 windows of train rows 0-255 (764 for lowercase,
  50 for the cipher), each with its mean input to block 1's MLP output.
- **States** are saved bf16; every reading uses the bf16 copy, and a control checks its held-out gain against the fp32
  state's (within 0.005).
- **Controls, in order:** the recipe (lowercase order 0's held-out gain within 0.005 of 0.1301, E49's gate); bf16 copy;
  zero update (every gain within 0.001, relay within 0.01); do-nothing subcomponent gain within 0.001; each split
  state's G equal to its do-nothing G within 0.001. Any miss stops the job.
- **Edge cases run before filing** (`score.py` on synthetic rows, CPU): empty lists, None, NaN and inf ratios, zero and
  negative subcomponent gains, ties at ±0.3 (an exact tie in floating point can fall either side: 0.8 − 0.5 separates,
  0.7 − 0.4 does not), a capped or targetless write, an unmeasurable floor, every reading branch. All gave the filed
  answer.
- **Not sourced:** the bf16 tolerance (set to the recipe's 0.005, not measured); the 500-window draw seed; batch order
  1's permutation.

Compute (guessed, not measured): `finetune` 3-5 min (E49's fine-tune took 17 s per 300 steps on the L40), `read` 10-20
min (E46 Amendment 3 took 48 s per table at 97 windows of 41 tokens; here 16 times the tokens). Host memory guessed
under 16 GB; the job asks 32 GB, as A0's. Peak memory is logged at the end of each stage.

## Code review (2026-10-29, fresh Opus agent, diff 675a0af..b12f153)

No bug found that changes a scored number; rows, positions, batches and states line up across arms, and shared
`src/commute` changes leave other callers unchanged. Skipped, with reasons: (1) the read stage's zero, do-nothing and
same-G controls pass by construction (the split skips all-zero matrices; U and T do not depend on the components), so
the write-up says their passing carries no information; no in-run positive control is added, since the filed controls
already reproduced in E46/E50 on the same code path. (2) The lowercase hand write may be ill-conditioned (764 keys from
2,000 windows): `write_fit` is read before P3 is interpreted. (3) Lowercase G may fall under 1 nat (A0 already +1.18);
if the random null's |G| is large the filed floor stays at 1 nat and P1 reads "unscored", which is the filed outcome.
(4) The floor's standard error treats windows as independent (rows share context): noted under noise. (5)-(7) mixed
bf16 forward paths in matching, late `score` import, sbatch details: below tolerance or harmless; not changed.

## Amendment 1 (2026-10-30): the breadth-swapped pair

Approved by the user 2026-10-29 (`decisions.md`, row "E53 next (ask #10)"). Written by a code subagent for the
regating-lead. Rules: `experiments/performed_anchor/results/predictions_amend1.json`, filed before any run.

**In plain words.** E53's ratio read lowercase high and the 50-word cipher low, but lowercase was also the broader
update, so the ratio may read breadth rather than kind. This pair swaps breadth: a narrow lowercase (only 50 word
types lowercased; performed but targeted) against a broad cipher (500 words swapped; new but broad). If the ratio
still reads the performed member high, it follows kind; if it now reads the broad member high, it follows breadth.

**Design check (amendment, approved class 3 by the user's go).**
- *Goal*: `GOALS.md`'s fallback question (new capability or strengthening); it decides whether E53's separation is kind.
- *Nearest tried*: E53 (`TRIED.md`), where kind and breadth are confounded on every row (G per off-target KL 4.1-4.5
  lowercase vs 11-15 cipher). New: the two factors crossed.
- *Patch or step*: a patch on E53 that decides whether its result can be carried to Qwen.
- *Decisions*: as E53 (Γ frozen, least-squares split, all 24 matrices, the noise-floor rule of ask #9). No departure.
- *What changes next*: kind → the ratio goes forward as a number of kind (task-family caveat below); breadth → it is
  withdrawn as a number of kind; both or neither → to the user with the 2 × 2.
- *Control first*: in job 1 the W₀ check (A0's rule) before any fine-tune, with a stop rule; in job 2, E53's lowercase
  seed-0 state re-read by the extended code before any new state, against `read.json` (this one can fail, unlike
  E53's zero and do-nothing rows, which are dropped as true by construction).
- *Numbers*: per member, the step-300 stand-in ratio (P1) and G per off-target KL (the swap check); the KL-matched
  random δW for the floor. Expected values below.

**The pair** (both on E53's cipher rows: cased Pile train rows 0-2399, token-level maps by `performed.cipher_rows`):
- *Narrow lowercase*: tokens matching `Ġ[A-Z][a-z]+` whose lowercase form is one vocabulary entry, ranked by count on
  Pile train rows 0-4095, ranks 50-99 (mid frequency, as E53's cipher ranks) (`performed.title_lower_map`), each mapped
  to its lowercase form wherever it occurs; everything else stays cased, including capitals with no leading space.
  *(Superseded before the run: the type rule, list and counts below were revised; see "Revised before the run
  (review)".)* The first-filed 50 were With, License, National, Since, Cir, Suppose, May, Although, March, Ch, While, Section, Figure, County, City,
  String, Dr, World, My, Sh, South, God, First, De, Th, Cl, President, Now, From, Furthermore, Le, How, Here, Am, Ex,
  His, North, Un, Id, Health, Moreover, List, Department, Par, Some, Get, Also, Be, House, Theorem.
- *Broad cipher*: a seed-0 derangement of the 500 word tokens (`Ġ[a-z]{2,}`) of ranks 50-549, in context and targets.
- Tokenizer-only counts (CPU, 2026-10-30), val rows 0-255 / training tokens touched: narrow lowercase 354 positions in
  144 rows, 273 with a whole window (all read), 0.27% of training tokens; broad cipher 10,824 positions in 246 rows,
  8,137 with a whole window (500 drawn), 9.07%; E53's cipher 2,916, 2,198, 2.42% (reproduces `a0.json`); E53's
  lowercase 6,070 positions (`a0.json`), 4,598 windows.
- Recipe: E53's, unchanged (Adam 3e-5, 300 steps of 8, all 24 matrices, batch orders 0 and 1), saves at steps 10, 100
  and 300 for both, so gains can be bracketed.

**W₀ check** (job 1, first): A0's statistic on val rows 0-255. Narrow lowercase: log p(lowercase) − log p(capital) in
its rows, filed ≥ +1 nat; the same positions in the cased rows, ≤ −1 (reported); broad cipher: log p(π(x)) − log p(x),
≤ −3. Stop if narrow lowercase < +1, broad cipher > −3, or a mean is not finite. **Risk, said before the run:** the
narrow member is lowercased inside otherwise cased text, and many of the 50 are sentence starters (Since, Although,
While) or names (County, President), which the original model capitalises after a full stop; the only cue is an
earlier lowercased occurrence in the same row (145 of the 354 positions are such repeats). The check may well stop.
It reports repeats and first occurrences apart, so a stop comes back with the number a fallback (scoring repeats only,
or a different type set) would need; any fallback goes to the user.

**Rules, in plain words** (exact rules in the JSON).
1. *Swap check, gate*: breadth = G / off-target KL per step-300 row. Achieved iff the lower narrow-lowercase breadth
   is at least 1.5 times the higher broad-cipher breadth. Otherwise the pair is not read on kind or breadth.
2. *P1 on the new pair*, E53's rule: gap = min narrow lowercase − max broad cipher ratio; separates at ≥ 0.3, reversed
   at reverse gap ≤ −0.3, unscored if any ratio is unread (E53's noise-floor bar, its own KL-matched random δW).

| outcome | rule | reading |
|---|---|---|
| kind | swap achieved, P1 separates | the ratio follows performed vs new with breadth swapped; leans: narrow ≥ 0.7, broad ≤ 0.5 |
| breadth | swap achieved, P1 reversed | the ratio follows broad vs targeted; withdrawn as a number of kind |
| both | swap achieved, does not separate, not all four at one end | the pair converges: kind and breadth each carry part of E53's 0.65 gap |
| neither | swap achieved, does not separate, all four ≥ 0.7 or all ≤ 0.5 | only one E53 cell differs: an interaction or another task property |
| unscored | swap achieved, a ratio unread or not finite | no reading |
| not read | swap missed (or a breadth unread: G ≤ 0, KL ≤ 0, non-finite) | the pair did not swap breadth; ratios reported only |

The brief's wording "breadth predicts the reverse ordering or both converging" is filed as two outcomes: pure breadth
predicts "reversed"; convergence is "both", since with the swap achieved it means each factor moves the ratio. Leans
(not scored): kind 0.4, both 0.3, breadth 0.15, neither 0.15; swap achieved 0.8. Reported beside it: E53's step-300
and cipher rows from `read.json` (G, off-target KL, breadth, ratio), the 2 × 2 kind and breadth effects on seed means
(pure kind ≈ 0.66 and 0; pure breadth the reverse), the matched-gain rows in both directions, P1 at the plain 1-nat bar.

**Expected values.** Check: narrow lowercase uncertain (see the risk), cased ≈ −5 (A0 −5.58), broad cipher ≈ −5 (A0's
cipher −5.53). E53 repro: G 1.6708, ratio 0.9186. Swap: narrow lowercase breadth well above 15 (a small targeted map),
broad cipher below 5 (9% of tokens changed, most window positions off target). Random δW: |G| ≤ 0.6 (E53 0.24-0.25).

**Blind to.** As E53, plus: kind is still confounded with task family (both "kind" members are lowercase tasks, both
"new" ones ciphers), and no task-matched null runs; breadth is one definition (G per off-target KL at the windows).
**Dropped**: the hand write (both of E53's hit the α cap, and a one-matrix write would not settle breadth), the relay
(did not separate, not this question), the zero and do-nothing controls (true by construction).

**Edge cases run before filing** (`score.score_amend1` and parts on synthetic rows, CPU, 2026-10-30, script
`~/commute-data/scratch/regating-lead/e53a1_edges.py`, 28 cases, all gave the filed answer): every outcome branch;
swap exactly 1.5× (achieved) and just under; swap missed with P1 separating or reversed; G ≤ 0, KL = 0, NaN or inf in a
breadth; a NaN ratio; G under the floor (unscored); a non-finite random row (F = inf, 1-nat bar); ties at 0.3 and −0.3;
all four at exactly 0.7 and 0.5; a mixed 0.7/0.5 pair (both); empty lists; unread rows in the effects.

**Code** (diff on `experiments/performed_anchor/run.py`, `score.py`, `src/commute/performed.py`,
`cluster/performed_anchor.sbatch`). New: `run.py a1_finetune` (`a1_maps`, `a1_check`, `stage_a1_finetune`) and
`a1_read` (`reproduce_e53`, `stage_a1_read`); `score.score_amend1` with `breadth`, `swap`, `outcome`, `nearest`,
`effects`; `performed.title_lower_map`, `earlier_in_row`, `TITLE`. Shared code changed, callers unchanged:
`performed.word_ranks` gains a `pattern` argument defaulting to E53's `WORD` (callers `a0.py` and `run.py`; the
tokenizer check reproduced E53's 2,916 cipher positions); in `run.py`, E53's fine-tune loop moved into `finetune_all`
(same order, files, controls, outputs), the state check into `state_loader` (only its exit message reworded), `Rows`
takes an output name (default `read.json`), and `main` dispatches four stages with E53's a0 gate kept for its two.
No other experiment imports these.

**Compute and commands.** Cluster debug L40, $0, via `/pod-runner`, from the clone root:
`sbatch cluster/performed_anchor.sbatch a1_finetune` (model load, check ~30 s, four fine-tunes; E53's took 107 s; about
3-4 min), then `sbatch cluster/performed_anchor.sbatch a1_read` (E53 repro, 12 split reads at 273 and 500 windows, two
random matches; E53's read took 217 s with hand-write grids; about 4-8 min). RunPod 4090:
`PYTHONPATH=$PWD/src python experiments/performed_anchor/run.py a1_finetune`, then `... a1_read`; `a1_read` also needs
E53's `results/large/dW_lowercase_s0_step300.pt` staged (a hard link under `results/large/`) and the pod's own A1
states. Files to collect: `results/check_amend1.json`, `finetune_amend1.json`, `read_amend1.json`,
`results/large/{a1_finetune,a1_read}.log` and `.out`; the twelve `results/large/dW_{narrow_lower,broad_cipher}_s*_step*.pt`
(bf16, 55 MB each as E53's, 660 MB in all). Not gated (under $15 and 3 h). Review before running: one fresh Opus code review
(Codex out of quota).

**Not sourced**: the 1.5 swap factor (five times E53's 10% seed spread in breadth); ranks 50-99 and 50-549 (by analogy
with E53's 50-99); cipher seed 0; the repro tolerances (the plan's bf16 rounding and window noise, not measured across
cards); the outcome leans.

### Revised before the run (review), 2026-10-30

A fresh code review of 17a3a2d found two serious problems and four smaller ones. Nothing had run, so
`predictions_amend1.json` was revised in place (its "filed" line says so) and the code fixed. What changed and why:

1. **Breadth now uses the true off-target KL.** The off-target KL E53 reported (`kl_other`) averages all window
   positions but the scored one, including positions that predict another token of the task. About 12% of a broad
   cipher window's tokens are cipher tokens, against 1.8% for narrow lowercase, so the swap gate would have passed on
   target density, not on how far the update spreads. `run.kl_split` now splits each window's KL by whether the
   position's next token is a scored-type token (`performed.position_mask`, cut by the same window draw), without
   changing `local_faith.stats`. Breadth = G / `kl_off` for the swap gate; G / `kl_other` is reported beside it.
   E53's own breadth finding (4 vs 11-15) has the same problem (scored-type share 8.6% of lowercase window tokens,
   4.1% of cipher's), so `a1_read` re-reads E53's four step-300 states with the split KL, reported.
2. **Word pieces dropped from the narrow types.** About 12 of the first 50 (Cir, Ch, Sh, Th, Cl, De, Le, Un, Ex, Par,
   Am, Dr) are pieces of longer words; lowercasing them at token level makes tokenisations the model never sees, a
   cipher-like task. The rule now keeps a title-case type only if, on Pile train rows 0-4095, the next token does
   not start with a letter in at least 90% of its occurrences (`performed.whole_word_share`), and only if its
   lowercase form is the tokenizer's own one-token encoding; ranks 50-99 are taken after the filter. New 50: Figure,
   County, City, String, World, South, First, President, Now, From, Furthermore, How, Here, His, Id, Health,
   Moreover, Department, Some, Also, House, Theorem, Table, Note, District, Research, International, Code, Each, Of,
   Which, Because, Center, English, Two, Both, Institute, School, East, Calculate, Government, Day, Using, Your,
   Group, Supreme, Give, General, Even, Public. The check now records W₀'s log-odds per type.
3. **Repeats test the original token.** `earlier_in_row` now asks whether the capitalised token appeared earlier in
   the original cased row. The old test looked for the lowercase form, which occurs naturally (" from", " some").
   For a cipher the two tests agree.
4. **The repro control covers the cipher path.** E53's cipher seed-0 step-300 state is re-read too, with a0.json's
   π: G 3.8690 and ratio 0.2643, same tolerances. The seed-1 states are re-read as references, not stops.
5. **The maps are checked between jobs.** `a1_read` asserts that its recomputed maps equal `check_amend1.json`'s.
6. **Not changed:** the gain floor uses the random δW whatever its match status, as in E53 (inherited).
7. **Unused work removed.** The new members and the re-reads build only their scored windows (`Run.scored`). The
   hand-write windows, `trigger_io` and T0f stay in `Run.task`, which only E53's read uses.

**Revised counts** (tokenizer only, val rows 0-255): narrow lowercase 344 positions in 130 rows, 134 repeats, 272
whole windows, 0.21% of training tokens; broad cipher unchanged (10,824, 3,278, 8,137, 9.07%). The risk paragraph
above still holds, with 134 of 344 positions repeats. **Revised gate**: swap achieved iff min over seeds of narrow
lowercase's G / `kl_off` ≥ 1.5 × max over seeds of broad cipher's, all four read (G > 0, `kl_off` > 0, finite).
Edge cases rerun on the revised rules: 31 cases, all gave the filed answer, including the gate reading `kl_off` where
`kl_other` would give the opposite answer, in both directions. Shared code changed beyond the first commit:
`performed.earlier_in_row` (no callers outside Amendment 1), `run.cells` (adds `kl_off` and `kl_on_target`; E53's
numbers unchanged), `run.Task` and `Run.task` (E53's read builds the same windows; only the train windows' assert is
gone), `run.batches` (an optional tensor argument). Runtime: `a1_read` adds three E53 re-reads, about 1 min.

**Not run**: everything above except the tokenizer counts and the edge cases.

## Amendment 1b (2026-10-30): the broad cipher alone

**Stop result.** Amendment 1's W₀ check stopped (job 45948, `results/check_amend1.json`): the original model
prefers the capital for the 50 narrow types inside cased text, −4.72 nats (90% interval −5.56 to −3.95; repeats −4.41,
first occurrences −4.92), against the ≥ +1 bar, so narrow lowercase is not a performed member. The broad cipher passed
(−4.99, bar ≤ −3).

**The user's choice** (`decisions.md`, ask #11): run the broad cipher alone, and re-read E53's four step-300 states
with the true off-target KL. A new narrow performed task, and stopping, were not taken. Rules:
`results/predictions_amend1b.json`, filed before any run; `predictions_amend1.json` and `check_amend1.json` stand.

**What it can answer now.** With no performed member, the kind-against-breadth 2 × 2 is gone. The question is only
whether breadth alone explains E53's gap. A new update at least as broad as E53's lowercase is read: if its ratio
stays low like E53's cipher, breadth does not explain the gap; if it reads high like lowercase, it does.

**Rules, in plain words.**
1. *Gate*: breadth = G per true off-target KL (Amendment 1's revised `kl_off`). The broad cipher must be at least as
   broad as E53's lowercase: its higher seed's breadth no higher than lowercase's lower seed's, lowercase re-read in
   this run. Otherwise "not read (swap missed)".
2. *Outcome* on the broad cipher's two step-300 ratios (E53's noise-floor bar, from its own KL-matched random δW):
   both ≤ 0.5 → "breadth does not explain E53's gap"; both ≥ 0.7 → "breadth explains it"; otherwise "mixed"; a ratio
   unread or not finite → "unscored". Leans: does not explain 0.6, mixed 0.25, explains 0.15; gate achieved 0.6.
3. *Reported, unscored*: is E53's lowercase broader than its cipher on the true off-target KL (lowercase's higher seed
   below the cipher's lower seed)? If not ("not broader (density artefact)"), the independent check's confound came
   from target density in the pooled KL. The same order on the pooled KL should reproduce E53's 4.1-4.5 against
   14.5-15.0. Also: matched gain (the broad-cipher step nearest E53 lowercase's G per seed), the ratio at the plain
   1-nat bar, E53's four re-read rows.

**Blind to.** A "does not explain" outcome says breadth alone is not enough, not that kind is the cause: the two
low members are both ciphers and the high one lowercase (task family), and no broad performed member is read.

**Edge cases run before filing** (`score.score_amend1b` and parts on synthetic rows, CPU, 2026-10-30, script
`~/commute-data/scratch/regating-lead/e53a1b_edges.py`, 20 cases, all gave the filed answer): every outcome branch;
both ratios exactly 0.5 and exactly 0.7; gate exactly equal (achieved) and just over (not read); a None, zero-KL or
negative-G breadth on either side; empty lists; NaN, inf and under-the-bar ratios; all four E53-order classes.

**Code.** `run.py a1b_finetune`: `a1b_check` reads the broad cipher's check from `check_amend1.json` (its map must
equal the recomputed one, its mean ≤ −3) before the model loads, then `finetune_members` (Amendment 1's data and
fine-tune loop, factored out; `a1_finetune` calls it unchanged) fine-tunes the broad cipher at batch orders 0 and 1,
saving steps 10, 100, 300. `run.py a1b_read`: `stage_a1_read` with the broad cipher only, written to
`read_amend1b.json` and scored by `score.score_amend1b`. It runs the two E53 repro controls first, then E53's seed-1
re-reads, the six broad-cipher states and its random δW.

**Compute and commands.** Cluster debug L40, $0, via `/pod-runner`, from the clone root:
`sbatch cluster/performed_anchor.sbatch a1b_finetune` (about 2 min), then
`sbatch cluster/performed_anchor.sbatch a1b_read` (about 4-6 min). Files to collect: `results/finetune_amend1b.json`,
`results/read_amend1b.json`, `results/large/{a1b_finetune,a1b_read}.{log,out}`, and the six
`results/large/dW_broad_cipher_s{0,1}_step{10,100,300}.pt` (55 MB each). `a1b_read` needs E53's four step-300 states
in `results/large/` (present on the cluster clone).

**Not run**: the broad-cipher fine-tunes and every reading above.

## Amendment 2 (2026-10-30): kind or breadth, settled with saved states first

*Written 2026-10-30 by a design subagent for the regating-lead. Plan only: no code, no `predictions_amend2.json`; the
user sees this first. Step 1 of the adopted re-gating programme (`decisions.md`, ask #13;
`reports/plans/2026-10-30_plan_regating-development-FABLE-DRAFT.md`, option A). Kept in this file, not a new plan:
every stage reads saved states with this experiment's `run.py` stages and rules; the only new fine-tunes sit in a
branch, and they are Amendment 1b's recipe at another cipher size.*

### In plain words

E53's stand-in ratio reads lowercase (a behaviour the original model already does) at 0.92 and a 50-word cipher (one it
cannot do) at 0.26, but lowercase is also the broader update, so the gap may read breadth rather than kind. Two saved
results can settle this without a new fine-tune. First, lowercase's breadth is understated by a counting bias, and
fixing it may bring it level with the 500-word cipher that already read low. Second, E54's elicit map and cipher (the
other lead's pair: same keys, same windows, same update size, same gain) already carry every number the ratio needs;
the plan writer computed it from `stage2.json` today: elicit 0.956 and 0.951, cipher 0.336 and 0.289. If that pair is
matched in breadth, or the elicit map is the more targeted, a performed behaviour reads high and a new one low without
being broader, and breadth cannot be what the ratio reads. Only if both checks leave breadth open does a broader cipher
get fine-tuned.

**The request in `GOALS.md` → Terms' words.** On fine-tunes of all 24 decomposed matrices, each split by the
least-squares split, does the stand-in ratio (masked gain over subcomponent gain, Γ and Δ frozen) separate an update
that strengthens a behaviour the original model performs from one that builds a behaviour it cannot, when the two are
equally broad (equal gain per nat of true off-target KL)? "Performed" is judged by the original model's behaviour only;
a high ratio means "uses pieces the original model has", never "uses a circuit it already runs" (`decisions.md`,
limit of the elicitation reading).

### Design check (`/design`)

- **Goal.** `GOALS.md`'s fallback question (new capability or strengthening). It decides whether the line's one
  kind number reads kind or breadth, before the programme's step 2 (E55's labelled states) and any Qwen read.
- **Nearest tried.** E53 Amendment 1b (`TRIED.md`): the 500-word cipher missed its breadth gate (7.8 against
  lowercase's 4.5-4.9), so "not read"; provisionally its ratio sat at most ~0.07 above E53's cipher. Amendment 1: no
  narrow performed Pile task passes the W₀ check (narrow lowercase −4.72 nats). E54: r separates elicit from cipher
  (0.77 vs 0.24) at matched size; the stand-in ratio and breadth were not read there. New here: lowercase's breadth
  without the bias, and E54's pair placed on the breadth axis, which gives the performed-and-targeted cell Amendment 1
  could not build.
- **Patch or step.** A patch on E53 that is also the programme's step 1.
- **Decisions.** Relies on: Γ frozen, least-squares split, all 24 matrices, E53's windows and noise-floor bar, the
  breadth definition of Amendment 1 (G per true off-target KL). Departure: the true off-target KL for lowercase leaves
  out every position whose next token the lowercasing changed, not only case positions (a correction of the
  definition's intent, not a new quantity). Reads E54's saved states, owned by the baseline line (E56 also reads
  them); reads only, writes nothing there.
- **What changes next.** "Breadth does not explain" → the ratio goes to step 2 (scored on E55's labelled states) as a
  candidate number of kind at VPD's piece level. "Breadth open" → branch B (a broader cipher), then the same decision.
  A stop or an unread rule → to the user.
- **Control first.** Stage 0 alone: the code re-reads E53's two seed-0 step-300 states and E54's elicit and cipher
  seed-0 states and must reproduce their saved numbers; on every token-map state the corrected off-target KL must equal
  the old one exactly (the correction can only change lowercase). Any miss stops the job.
- **Approved class.** Stages 0-2: class 1 (analysis of what exists: GPU passes of the existing harness on saved states).
  Branch B: class 3 once the user confirms this plan (a branch of its decision tree), and class 4 (the programme's
  named step 1). The user asked to see the plan before predictions are filed.
- **Numbers.** Four, in the table below, plus inputs already measured.

### States

| state | file | made by | check before reading |
|---|---|---|---|
| E53 lowercase, cipher, seeds 0 and 1, step 300 | `results/large/dW_{lowercase,cipher}_s{0,1}_step300.pt` (bf16) | E53 `finetune` | sha256 in `results/finetune.json` |
| broad cipher, seeds 0 and 1, step 300 | `results/large/dW_broad_cipher_s{0,1}_step300.pt` (bf16) | Amendment 1b | sha256 in `results/finetune_amend1b.json` |
| E54 elicit (step 10) and matched cipher (step 9), seeds 0 and 1 | `experiments/keydependent_elicit/results/large/{elicit,cipher}_s{0,1}.pt`; windows `windows_{elicit,cipher}.pt` (`heldout`, 97 windows of 41 tokens, half 20) | E54 stage 2, job 45974 | sha256 in E54's `stage2.json` → `finetunes` and `stage1.json` |

No reference procedure is replayed: every state is read as saved.

### Stage 0: the control (runs first, alone; early stop)

| check | reads | must read | else |
|---|---|---|---|
| E53 repro | G, ratio and `kl_off` of E53 lowercase s0 and cipher s0 | `read_amend1b.json`: G within 0.02, ratio within 0.01, `kl_off` within 1e-4 (1b reproduced all three exactly) | stop, fix |
| E54 repro | G = T1 − T0 (log-odds at the key) of elicit s0 and cipher s0 on E54's held-out windows | E54 `stage2.json` cells: within 0.02 | stop, fix |
| token-map identity | corrected `kl_off` on E53 cipher s0 and broad cipher s0 | equal to the uncorrected `kl_off` bit for bit (for a token map the changed positions are the scored ones) | stop, fix |
| superset | the changed mask on lowercase | contains every case position (assert) | stop, fix |

These can fail: the repro rows exercise the new E54 path and the re-read, the identity row the new mask code.

### Stage 1 (option a): lowercase's breadth without the bias, $0

**The bias.** Breadth's off-target KL leaves out positions whose next token is a case position (a space-led capital).
The lowercasing also changed tokens that are not case positions: capitals with no leading space, word pieces the
tokenizer splits differently once lowered, end-of-row padding. Those count as off-target, so lowercase reads broader
than it is. **Fix.** A position is *changed* if the lowered row's token at that byte offset is not the cased row's token
at the same offset with the same string (a new `performed.changed_mask`; for a token map, `mapped != rows`). The
corrected true off-target KL leaves out every position whose next token is changed (`kl_split` with the changed mask
in place of the scored-type mask). Breadth b* = G / corrected `kl_off`.

**Measured today (tokenizer only, CPU, script `~/commute-data/scratch/regating-lead/plan-a2/changed_counts.py`).** In
E53's 500 lowercase windows, of 126 KL positions per window, 9.86 predict a case position and a further 13.38 (10.6%)
predict a changed token that is not one. The commonest are word pieces after a capital with no space ("the" after a
newline, "s", "p", "b"), and padding (124 of 6,690). 0 rows skipped.

**Estimate.** Removing 13.38 positions at mean KL k from the 116.1 off-target positions moves b* to the broad
cipher's 7.81 only if k ≥ 1.30 nats (seed 0) and k ≥ 1.56 (seed 1). If k equals the KL at case positions (0.81 and
0.85, `read_amend1b.json`), b* reads about 6.0 and 5.4. So the fix is unlikely to close the gap alone.

| number | how measured | W₀ / zero update | do-nothing | control | expected |
|---|---|---|---|---|---|
| b*, lowercase s0, s1 | G / corrected `kl_off`, 500 windows | undefined (0/0) | = the split state's (same full model), not read | identity row: ciphers unchanged | 5.4-6.0 (k ≈ 0.8); ≥ 7.81 needs k ≥ 1.3-1.6 |
| k, reported | mean KL at changed, not-scored positions | 0 | — | — | 0.8-1.6, above `kl_off` (0.34-0.36) |

### Stage 2 (option c, with saved states): E54's pair on the breadth axis, $0

E54's elicit map is a key → value table the original model already ranks 2nd-20th (performed, weakly); its cipher maps
the same keys to words the original model gives under 1e-4 (new). Same keys, same window recipe, matched update norm,
and matched gain (log-odds G 10.79 and 10.82 against 11.10 and 10.89). The stand-in ratio, from E54's least-squares
split cells (`(L11 − L00) / (U1 − U0)`, `stage2.json`): elicit 0.956, 0.951; cipher 0.336, 0.289; gap 0.615. Not
measured: breadth. This stage reads each of the four states' true off-target KL on its own 97 held-out windows: the
full model's KL to the original model at every window position except the key position (nothing else in an E54 window
is a task token), and breadth = G / that KL.

| number | how measured | W₀ / zero update | do-nothing | control | expected |
|---|---|---|---|---|---|
| E54 breadth, four states | G (T1 − T0 at the key) / off-target KL, 97 windows | undefined | = the split state's, not read | E54 repro row | unmeasured; both small KL (E12's recipe carries a KL term on other tokens) |
| x | min elicit breadth / max cipher breadth | — | — | — | lean ≥ 1/1.5 (see P2) |

Why "performed" here is fair: the original model gives each value probability 0.01-0.1 after marker and key (E54
stage 1), weak but real, and the top-ranked word is excluded. Why it can be broader: the elicit map strengthens
associations that also occur in plain Pile text (" life" → " span"), which may move the model at unmarked occurrences.

### Predictions (to file in `results/predictions_amend2.json` after the user's go)

- **P1, lowercase's gate re-read** (the 1b gate with the corrected KL; filed afresh, 1b's filed outcome stands as "not
  read"). Achieved iff min over seeds of lowercase b* ≥ max over seeds of the broad cipher's breadth, both re-read in
  this run (1b: 7.81). Equality counts as achieved, as in 1b. Unread if any G ≤ 0, `kl_off` ≤ 0 or non-finite. If
  achieved, 1b's filed outcome rule on its already-measured step-300 ratios (0.285, 0.273, both ≤ 0.5) gives "breadth
  does not explain E53's gap": known now, so P1's only open part is the gate. *Lean: missed, 0.7*, from the estimate above.
- **P2, E54's breadth order.** x = min elicit breadth / max cipher breadth (worst pair). *Reversed*: x ≥ 1 (the
  performed member is the more targeted on every pair). *Matched*: 1/1.5 ≤ x < 1. *Confounded*: x < 1/1.5 (elicit
  broader, as E53's lowercase). Unread if any breadth is non-finite, G ≤ 0 or KL ≤ 0. The factor 1.5 is Amendment 1's
  swap factor (five times E53's 10% seed spread in breadth), and 1b's provisional dose-response supports it: halving
  breadth within the cipher family moved the ratio by 0.01-0.07, so a 1.5-fold breadth difference cannot carry a gap
  of 0.6. *Lean: reversed 0.35, matched 0.35, confounded 0.3.*
- **Reading** (scored from P1 and P2):

| P2 | P1 | reading |
|---|---|---|
| reversed or matched | either | **breadth does not explain the separation**: in E54's pair a performed update reads high and a new one low at equal or reversed breadth; with E53, the ratio orders by kind in both breadth orders, across two task families |
| confounded | achieved | **breadth does not explain E53's gap**, within the Pile family (lowercase and the 500-word cipher equally broad, ratios 0.92 vs 0.28); E54 reported as confounded |
| confounded | missed | **breadth open** → branch B |
| unread | — | to the user |

E54's ratio gap (0.615 against E53's bar of 0.3) is an input already measured, not a prediction; the stage re-reads
only G and KL. **Edge cases**, run on synthetic rows before filing (CPU, as 1b's 20 cases): x exactly 1 and 1/1.5; P1
equality; a NaN, zero-KL or negative-G breadth on either side; empty lists; each row of the reading table.

**Blind to.** *Breadth* is one definition: gain per mean KL over window positions. E54's windows are 41 tokens around
an inserted marker and key, so its KL samples Pile text near a marker, not the whole Pile, and x compares the two
members of one pair only, never E54 against E53 (different window formats and gains). *k* sees only the next-token
change, not the changed context that follows a changed token (the same for every task). *The ratio*: as E53, plus
E54's cells come from its `route_split`, another implementation of the same cells than E53's `cells` (E46 Amendment 3's
toy ratios came from the same family). *Kind* is still read only up to VPD's pieces: a new circuit from old pieces reads
"elicited". Two task families now each have a performed and a new member, which narrows, not removes, the task-family
caveat; "performed" in E54 is weak (rank 2-20).

**Noise floor.** Breadth: seed spread 10% on lowercase (4.92 vs 4.46), under 1% on the broad cipher; each breadth gets
a 90% bootstrap interval over windows (reported; windows share rows, so somewhat narrow). E54's breadth sits on 97
windows; its seed spread is first measured here. Ratio: as E53 (±0.03-0.06 over windows at 500 windows; E46's
±0.024-0.034 at 97).

### Decision tree

1. Stage 0 misses → stop, fix if a plain bug, else to the user.
2. "Breadth does not explain" (either row) → `TRIED.md` row, log card; programme step 2 (ratio and r on E55's labelled
   states). Branch B not run.
3. "Breadth open" → **branch B** below, approved with this plan.
4. Branch B's ladder finds no size broad enough → stop; report the range read (never "no cipher works"); to the user,
   with a new performed task (option c below) as the next choice.

**Branch B: a cipher as broad as lowercase** (option b). Amendment 1b's recipe and code (`finetune_members`,
`stage_a1_read`), the token map widened. Training-token share from the tokenizer today (`plan-a2/cipher_counts.py`):
word ranks 50-1049 12.2%, 50-2049 15.7%, 50-4049 19.1%, 0-1999 31.1% (E53's 2.4%, 1b's 9.1%). Breadth fell as share^−0.5
from E53's cipher to 1b's (two points, provisional), so lowercase's 4.5-4.9 needs about 23% (ranks 0-1999 territory)
and a corrected 6 about 15%.
- *B1, the ladder (cheap first stage, one seed).* W₀ check first (A0's rule, mean log-odds ≤ −3 nats per cipher; stop
  if missed). Then ciphers of ranks 50-2049, 50-4049 and 0-1999, seed 0, 300 steps; each read for G and corrected
  `kl_off` only (no split). Pick the smallest whose step-300 breadth ≤ 0.9 × lowercase's min b* (0.9 covers the 10%
  seed spread). Stop rule: none reaches it → tree item 4.
- *B2, the scored read.* The chosen cipher at seeds 0 and 1, saved at steps 10, 100 and 300, the full split read and
  its KL-matched random δW, by 1b's filed rule with lowercase's b* as the bar: gate, then both step-300 ratios ≤ 0.5
  "breadth does not explain", both ≥ 0.7 "explains", else "mixed". Predictions filed before B2; same edge cases as 1b.
  *Lean: does not explain 0.6.* The matched-gain row (step nearest lowercase's G) is reported, as in 1b.

**Option c, a new performed-and-narrow Pile task: not planned.** Every candidate so far fails the W₀ check or is not a
distinct update: narrow lowercase (−4.72 nats), its repeats only (−4.41), and induction-repeated names (the original
model does them, but a fine-tune on natural text has nothing to strengthen). E54's elicit map is the performed-narrow
member this plan reads instead. A new one goes to the user only from tree item 4.

### Cost and routing

One job, cluster `debug`, L40, $0, via `/pod-runner`: model load ~30 s, stage 0 (two split reads at 500 windows, two
E54 reads) ~1 min, stage 1 (two lowercase full reads plus the cipher identity rows) ~1 min, stage 2 (four reads of 97
short windows) seconds; about 3-5 min (1b's read of 11 rows took 134 s). Host memory as 1b (6.2 GB). Branch B, if
reached: B1 about 3 min (three 300-step fine-tunes at ~23 s each, full reads), B2 as 1b (about 1 + 2.5 min). Not
gated: under $15 and 3 h.

### Order of work for the fleet agent

1. Read this section, Amendment 1b's code (`run.py`: `kl_split`, `cells`, `Run.scored`, `stage_a1_read`;
   `performed.py`), E54's `stage2.json` and `steering.load_task` (for E54's windows and tables; import from
   `src/commute`, never from `keydependent_elicit/run.py`).
2. Write `run.py a2_read`: `performed.changed_mask`; `Task` gains a changed mask; a full-model read of G and
   off-target KL for E54 windows (half and target table as arguments, not `CFG.half`); `score.score_amend2`. About
   100 lines; nothing else in `run.py` changes behaviour.
3. Run the edge cases (CPU), file `results/predictions_amend2.json`, commit both before the job.
4. Fresh-Opus review of the diff (Codex out of quota); fix what is real, record what was skipped.
5. `/pod-runner`: `sbatch cluster/performed_anchor.sbatch a2_read`, stage 0 first in the same job with `sys.exit(1)` on
   a miss. Collect `results/read_amend2.json` and `results/large/a2_read.{log,out}`.
6. Report as an "Amendment 2" part of `reports/analyses/2026-10-29_analysis_e53-performed-anchor.md` by `/writeup`;
   independent check; `TRIED.md` row; PENDING line; decision card if "breadth open" (branch B's run line) or to the
   user on any stop.

### Not in this plan

E54's ratio re-computed through E53's `cells` (its stored cells are read instead); E54's own-step cipher (step 30) and
E12's copy tables on the breadth axis; the relay, hand writes, zero-update and do-nothing rows (true by construction);
a second decomposition seed; Qwen; any edit to E54's files.

**Not run**: everything except the two tokenizer counts and the E54 ratio arithmetic from `stage2.json`.

### Filed (2026-10-30, code agent)

Rules filed in `experiments/performed_anchor/results/predictions_amend2.json` before any run; code `run.py a2_read`
(`pile_windows`, `e54_windows`, `kl_read`, `breadth_rec`, `a2_controls`, `stage_a2_read`), `score.score_amend2`,
`performed.changed_mask`. Two rule fixes found by the edge cases (43 cases, script
`~/commute-data/scratch/regating-lead/code-a2/edges.py`, all as filed after the fixes):
- **A case the reading table left out**: P2 confounded with P1 unread. Filed as "to the user (P1 unread)".
- **P2's ties**: compared as products (min elicit ≥ max cipher; 1.5 × min elicit ≥ max cipher), so x exactly 1 reads
  reversed and exactly 1/1.5 matched, as the table says, with no rounding in a division.
Also as built, not changes of rule: the "confounded and achieved" row returns 1b's filed outcome (`score.outcome_1b`)
on read_amend1b.json's ratios, which reads "breadth does not explain E53's gap"; E54's windows are read straight from
their file (sha256 against `stage1.json`), not through `steering.load_task`, which would load a second model;
`run.kl_split` takes `half` as an argument (default E53's). Tokenizer-only check on CPU: lowercase windows carry 9.86
scored and 13.38 changed-not-scored next tokens per window (the plan's counts); the corrected and uncorrected masks
agree on both token maps.
