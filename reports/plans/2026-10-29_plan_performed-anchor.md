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
