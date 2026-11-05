# Plan: E101, what job did E98's carrying neurons have before training? (67M)

*2026-11-05, baseline-lead E101 writer. Adopts option (b) of `reports/plans/2026-11-05_advice_e99-neuron-next-FABLE.md`,
with the lead's overrides below. **The advice file says E99; E99 and E100 were claimed by other leads, so this
experiment is E101.** Rules: `experiments/carrying_jobs/results/predictions.json`. Code: `experiments/carrying_jobs/`,
`src/commute/carrying_jobs.py`, `tests/test_carrying_jobs.py`, `cluster/carrying_jobs.sbatch`. Fast lane, not gated
($0, under 30 min per job): no pre-run review.*

## Claim, null, and the request in the Terms' words

**Claim.** A fine-tune built to reuse something the original model does is carried by neurons whose job in the
original model already relates to the task; one built to teach something new is carried by neurons whose job does not.
**Null.** Carrying neurons' original jobs relate to their task no more often than block-matched random neurons' do, on
built-to-reuse and built-to-be-new fine-tunes alike.

**Restatement** (`GOALS.md` Terms). No split, no component training, no decomposition is read. For each finished δW of
E98's scored states on the 67M, take E98's carrying set: the MLP neurons of W₀ ranked by how much of the gain their
re-firing carries (`experiments/neuron_r/results/carrying.json`). Read those neurons in W₀ alone, as the
"already clean" essay reads neurons (https://already-clean.goodfire.com/essay.md): top Pile windows, a one-line job
written blind, a test the job can lose. Score against the construction labels, marked assumed, on the method table's
rows (user, `decisions.md:154`: neuron-level baselines, 67M first, essay-style interpretation after the neuron share).

## Design check

- **Goal.** `GOALS.md`'s question is whether a parameter decomposition stays interpretable along post-training, and
  whether post-training reuses or adds. This is a decomposition-free reading of reuse, to stand beside VPD's r on the
  same rows: if neurons' original jobs tell reuse from new where VPD's r does, VPD's reading has a cheap rival.
- **Nearest tried.** E98 (`TRIED.md`, E98 row): the neuron share separated only lowercase from ciphers and copy from
  lookup; it gave the carrying sets but read no neuron. E66 (`experiments/TRIED.md:150`): the W₀ top-K neuron reading was
  dead at K 4096 (Fable cites line 156; it is 150). E63: counts, no jobs. None asked what the carrying neurons did.
- **Patch or step.** The user's named step 2 of the task, not a patch on E98's numbers.
- **Decisions.** Relies on `decisions.md:154` (the task, runs without sign-off up to the cap). No departure.
- **What changes next.** P1 fails: the 67M's neurons are not clean enough for the method; skip to Qwen3-1.7B (Fable,
  "What would change this"). P2 passes: neuron jobs are a reuse reader to put beside r. P3 reads either way on rank-1.
- **Control first.** Stage 0's C0 checks run before any reading and stop the job; stage 1's C1 (judge false
  positives) and C2 (maximiser self test) are read before P1-P3.
- **Saves.** All 12288 neurons' top-20 windows and engaged values on every task (`results/large/`), the key, every
  agent output (committed).

## Stage 0 (this commit; MATS `debug`, one L40, $0)

`sbatch cluster/carrying_jobs.sbatch` runs `experiments/carrying_jobs/run.py`, W₀ only, fp32:

1. sha256 of E98's task windows (as E98 recorded) and of `carrying.json`.
2. Pile pass over val rows 2048-4047 (`commute.data.pile_rows`, `src/commute/data.py:25`; cached `val_0_8192`, past the
   lowercase task's val 0-255, `src/commute/lowercase.py:39`): per neuron, the 20 rows with the highest peak firing,
   ±20 tokens around the peak, decoded, peak token marked `⟦ ⟧`. A sorted sample of every neuron's firing on all
   65,536 positions of rows 4096-4223 gives each neuron's quantile scale.
3. **C0_pile.** Engaged values on held-out rows 6144-6159: median over neurons in [0.45, 0.55], expected 0.5. Stop on
   a miss (the quantile code is wrong).
4. **Engaged value** of a neuron on a task: W₀'s firing at the answer position (`half`, where the target is read), as a
   quantile of its own Pile sample, mean over the task's read windows (the windows E98 ranked on). **Engaged share**
   of a set: the median over its neurons.
5. **C0_w0, the positive control.** Per task, the 16 neurons with the highest engaged value on even windows, read on
   odd windows, against 10 uniform random sets of 16. Expected about 0.9 against about 0.5. Stop if on any task the
   control is not above every random set by at least 0.2.
6. **C0_null, the do-nothing read.** P0's rule with random draw 0 in place of every carrying set (draws 1-9 as the
   band). Expected: no separation on any row. Stop if one separates: the rule is degenerate.
7. Per state (43 scored E98 states): carrying share, 10 block-matched random shares (outside the set, seed 101), the
   excess over their median, and above-band (over all 10). **P0** on the lowercase row; the other rows reported.
8. Stage 1's pool and batch files (below); **C0_files** checks they hold only ids and windows. Stop on any other key.

Carrying set = `carrying.json` `.rows[s].top[: min(k_half, 256)]`; the 256 cap cuts the sets with `k_half` 512 or 1024
(random-word b, cipher at its own step, elicit s1, both e62 maps) to their top 256, the most the file saved.

**Expected time: 3-6 min.** E98's job 52786 took about 4 minutes
(`reports/analyses/2026-11-05_analysis_e98-neuron-r.md:289`) for 56 states at 7 forwards per batch plus two 300-step
fine-tunes; this job is W₀ forwards only. Peak GPU about 16 GB (the sort of a 12288 × 65536 sample). A CPU dry run of
everything after the model, on fake firing with the real sets, rows and tokenizer, passed C0_null and C0_files and gave
the pool counts below (`~/commute-data/scratch/baseline-lead/e101-writer/dry_run.py`).

## Stage 1 (designed here; files and scorer in this commit; the lead runs it)

**Pool.** Eight groups, each a deterministic consensus sample of its states' carrying sets (neurons in most seeds'
sets first, `carrying_jobs.consensus`), each with a block-matched random partner of the same size from neurons in no
scored state's set:

| group | states | carrying | description the judge sees |
|---|---|---|---|
| lowercase | lowercase s0-1 (one set, 16) | 16 | lower |
| ciphers | e53 cipher, broad cipher, s0-1 | 24 | cipher |
| same_word | ident32, ident32b, s0-1 | 24 | copykey |
| random_word | rand32, rand32b, s0-1 | 24 | lookup |
| e62_copy / e62_lookup | e62 s0-2 each | 24 / 24 | induct / lookup |
| rank1 | C3 rank-1 s1-7, top 64 | 24 | rank1 |
| copy_map | C3 copy map s0-7 | 20 | copykey |

Plus 30 **W₀ controls**: the top 10 neurons by engaged value on lowercase, e62 copy and same-word windows, from neurons
in no carrying set. 180 carrying + 180 random + 30 controls, deduplicated (the dry run: 360 distinct, since some
carrying neurons serve two groups), shuffled into batches of 30: **12 batches, 36 agent calls** (at most 13 and 39). The
exact count is in `stage0.json` `.rows.pool`.

**Blinding.** Batch files (`results/large/batches/b<NN>.json`) hold a salted 12-hex id and 20 windows per neuron:
no state, task, label, block or role. The key (`results/large/key.json`) maps id to neuron, peak firing and roles; no
brief names it. Carrying, random and control neurons sit in the same batches in random order.

**Roles** (briefs in `experiments/carrying_jobs/briefs/`, Sonnet, one call per batch, lead-orchestrated Agent calls):
1. **Writer** (`writer.md`): reads one batch file, writes a one-line job per neuron →
   `results/stage1/writer/b<NN>.json`.
2. **Maximiser** (`maximiser.md`): reads only the jobs, writes 5 examples and 5 near misses per neuron →
   `results/stage1/maximiser/b<NN>.json`. Then one `debug` job, **`experiments/carrying_jobs/maximise.py`** (a later
   commit; `sbatch cluster/carrying_jobs.sbatch maximise`), scores each text's peak W₀ firing of its neuron →
   `results/stage1/max_scores.json`, after C2.
3. **Judge** (`judge.md`): `score.py judge-inputs` writes each batch's jobs with the six task descriptions in a fresh
   shuffled letter order; the judge marks which tasks each job would help → `results/stage1/judge/b<NN>.json`.

`score.py score` joins the outputs with the key and applies the rules → `results/score.json`.

**Win rule.** Strong = at least 30% of the neuron's strongest Pile window (the essay's caption). The essay does not say
how a neuron's win is decided (read 2026-11-05); here a job **wins** if more examples than near misses fire strongly,
ties if as many. **Explained** = wins. **Related share** of a group = share of its neurons that win *and* that the judge
ties to the group's description. **Floor** = the same share over all random neurons, per description.

## Predictions (filed; P1-P3 are Fable's, re-checked)

| | reads at W₀ | do-nothing (rule on random in place of carrying) | control | noise floor | blind to |
|---|---|---|---|---|---|
| P0 engaged excess, lowercase row, E51 rule (gap over larger spread, ≥ 0.2); lean pass | no carrying set | C0_null: none | C0_w0 ≈ 0.9 vs ≈ 0.5 | the 10 random draws' spread per state | whether firing at the answer is the job the fine-tune used; lowercase's two states are one set |
| C1 judge false positives on random ≤ 0.5 (stop: forced-choice redesign) | — | — | — | binomial, n ≈ 180 | — |
| C2 self test: a neuron's own Pile windows 2-6 beat other neurons' windows, win share ≥ 0.8 | — | — | is the control | — | — |
| P1 lowercase related share above its floor (gap ≥ 0.2 and ≥ 2 se); lean pass; if it fails P2, P3 are reported, not read | no carrying set | related = floor: fail | W₀ controls (reported, lean above) | binomial se of the gap | jobs not shown in 2,000 Pile rows; a judge prior for prose tasks (the per-description floor); relays |
| P2 reuse-higher on ≥ 2 of 3 rows (lowercase/ciphers, same/random word, e62 copy/lookup), each side minus its own floor; lean pass, open on same word | no carrying set | x = 0 both sides: fail | as P1 | se from four binomial variances | as P1; a copy job judged "induct" on a copykey group counts as unrelated |
| P3 rank-1 explained share above the random base rate: coherent / not coherent / no call; no lean | no carrying set | explained = base: not coherent | the random base rate (reported) | binomial se | coherence that is not the task's |

Every rule was run on its edge cases and on the do-nothing reading before filing (`tests/test_carrying_jobs.py`, 7
passed): an empty or one-state side, a non-finite value, a dead neuron (peak ≤ 0, unscorable), equal rates at 0 and 1
(read "level", not "no call", after the first draft's `gap < 2 se` left 0 against 0 as no call), a gap over 0.2 but
under 2 se (no call). A group under 8 complete neurons, or with over 10% missing outputs, is unreadable.

**Corrections to the advice** (its citations re-checked): the E66 row is `TRIED.md:150`, not 156; the k_half counts
are in the table at analysis lines 167-177 (lines 162-166 are its text); P2's "rank-1 and copy maps against their
ciphers" has no carrying set on the cipher side (C3's ciphers are unscored in E98, `cells.json`), so that row is
replaced by same word against random word, which has; the judge sees six descriptions, not seven (the two copy-key
tasks share one, the two lookups share one, elicit is stage 0 only); the essay's win rule is not stated; stage 1 is 36
calls, not about 4,000, by sampling. The rest held: `pile_rows` at `data.py:25`, the 0.43 floor at analysis line 88,
the separations at 110-112, the 3.4-fold overshoot at 182, the rank question at 189-190, `decisions.md:154`, and the
seed overlaps (lowercase 16 of 16; rank-1 top 64: 52 between seeds 1 and 2, 48 between 1 and 7, by `jq`).

## Cost and order

Stage 0 and the maximiser job: `debug`, $0, ≤ 30 min each. Stage 1: 36 Sonnet calls. No pod. Order: stage 0 → read
C0 and P0 → writers (12) → `judge-inputs` → maximisers (12) and judges (12) in parallel → `maximise` job (C2 first) →
`score.py score`. A stop in stage 0 or a C1/C2 miss goes to the lead before the next step.

## Departures from the literature

- The essay reads Qwen3-32B over about 10 billion tokens with an investigating agent; here a 67M, 1M Pile tokens
  (2,000 rows), and a writer that sees only 20 windows, no tools.
- The essay's win rule is unstated; ours counts strong examples against strong near misses.
- The judge and the engaged share are not in the essay; they tie a job to a task.
- The essay's distant-concept control is not run.

## Not run

- **Option (a), the relay-free neuron share.** Fable shows it reduces to the attention-only part of the update: a
  matrix-subset share, not a neuron reading, and lowercase would fall to about 0 (advice, option (a)).
- Option (c), a Qwen neuron share now: no VPD r beside it on Qwen and the anchors differ 37 times in size (advice).
- Elicit and its cipher in stage 1 (stage 0 only); C3's ciphers (no carrying set).
- Steering or ablation of the carrying neurons; a second writer per neuron; the essay's distant-concept control.
- Qwen3-1.7B: after the 67M reads, if P1 passes.

## Amendment 1 (2026-11-05)

The judge stage had no power (analysis): its brief excluded punctuation and breaks, and the W0 controls were the same
generic top firers on every task. Rules `results/predictions_amend1.json`, filed before judging. $0; first of two.
- **Judge** (`briefs/judge_amend1.md`): no excluded token kind; each description shows 4 of its tasks' windows with
  the answer marked; ids hashed, roles mixed, letters reshuffled. 376 neurons, 12 batches of 32.
- **Controls**: per task, the 10 neurons whose engaged value most exceeds their mean over tasks with other windows
  (`contrast_top`), outside the carrying sets and the random pool: lowercase, ident32, e62_lookup. e62_copy and
  e62_lookup read the same windows, so W0's firing cannot tell them apart. 24 are new: writer batch `c00`.
- **C1a**: each task's controls tied to their own description above its random floor (ties only, `gap_rule`; about
  3-4 of 10); not pass closes the route on the 67M, next Qwen3-1.7B. **C1b**: own above the most-tied other control
  description. **C1_fp** ≤ 0.5. **P1, P2** as filed; P1 read iff C1a's lowercase holds, P2 iff C1a, C1b and P1 pass.
- Order: `sbatch cluster/carrying_jobs.sbatch controls_amend1` (debug, < 5 min; the window loaders need the model) →
  writer on `c00` → `score.py amend1-inputs` → 12 judges → `score.py amend1-score`. No maximiser call.
