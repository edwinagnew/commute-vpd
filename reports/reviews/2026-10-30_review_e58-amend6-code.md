# Review: E58 Amendment 6 code (commit a946868), before the cluster run

A fresh agent reviewed the commit, which adds a prior-matched cipher and a first-letter rule to E58, before its one
debug-QoS job. The prior-matched cipher, the harness control, the shared-code change and the runtime look sound. The
first-letter rule's carry measure x has a bias the plan does not name: held-out keys mostly have letters that were
trained often, so a fine-tune that just leans toward often-trained words raises x with no letter rule at all. That needs
a null filed before the run, because predictions cannot change after it.

Scope: `git show a946868` plus what it calls (`amend1.load_split`, `amend3.read`/`dump`, `amend4.train`/`over_gate`/`stop`,
`heldout_keys.behaviour`/`specific`/`others`/`split_task`, `steering.load_task`, `kv.bijection`/`key_distributions`,
`keydep.vocab`/`finetune`/`geo_rate`). Checked against `amend3.json`, `amend4.json`, `amend5.json`, the job logs
46398/46426/46450, and the key letters from the tokenizer (one `fleet/cpu-run` job on the tokenizer only; script in
`~/commute-data/scratch/regating-lead/a6-review/letters.py`). Nothing was run on a GPU.

## Findings

### 1. BLOCKING for FL (PM unaffected): the first-letter x is biased upward by letter frequency, and has no null

- **Where.** `experiments/open_pair/amend6.py:77-88` (the table), read by `src/commute/heldout_keys.py:57-79`
  (`others` / `specific`) and gated by `src/commute/open_pair.py:539-542` (`carries`, x > 0.3).
- **Failure scenario.** H_spec is the held-out key's own value gain minus the *mean* gain of the other pool values. On
  the cipher every value is distinct, so a fine-tune that lifts the whole pool cancels out. On the first-letter map
  the 22 values are shared, and each word is trained on as many keys as share its letter. The trained-letter counts are
  very uneven: s 16, f 10, c 8, p/m/e/b/a 6, t/o 5, w 4, l/d/i 3, n/r 2, h/g/y/u/z 1 (tokenizer, amend3.json keys). The
  held-out keys fall on the frequent letters: their own letter has a geometric-mean 5.1 trained keys, against 3.3 for
  the other letters (mean log n_L 1.64 vs 1.19). So a fine-tune that only learns the marginal "after M, say the words
  in proportion to how often they were trained", with no letter rule, gives H_spec > 0 and x > 0. Its size is not
  measured anywhere. It can push a no-rule seed past the entry condition (0.3), or put one there that the rule alone
  would not reach, and that fills the `FL_cipher_band` branch ("the line's missing cell is filled") on a wrong reading.
  The same marginal effect does not exist in the PM map (distinct values), so PM stands.
- **Suggested fix (before filing; predictions are immutable after the run).** Add a per-read null, with no new
  fine-tune: recompute H_spec with each held-out key's target swapped for the word of a *different* letter that has
  the same (or the nearest) trained-key count, a seeded swap within count classes. Call it x_null. Then file the entry
  condition as x − x_null > 0.3 (or x > 0.3 and x − x_null > 3 s), and report x_null beside x. A heavier alternative is
  a letter-permuted control map (each letter group sends its keys to another group's word, held-out keys unchanged),
  at one more fine-tune of about 1.5 min. Also state x's "blind to" line: it is blind to a frequency-weighted marginal
  shift.
- **Severity.** Blocking for the FL prediction; the PM prediction and the controls can run as filed.

### 2. MAJOR: FL "no_carry" also covers seeds that were too small to label or missing

- **Where.** `src/commute/open_pair.py:555`. The test `tests/test_open_pair.py:538` asserts it.
- **Failure scenario.** `carries` is False for a None read, a `too_small` or `unreadable` label, or a non-finite x. If
  the first-letter fine-tune (22 shared values, a weaker specific gain) never reaches G_min by step 30 on two seeds,
  the class reads `no_carry`. The filed branch then says "the cell stays empty on the 67M; first thing to build on
  Qwen", a substantive claim from what was really an unlabellable map. PM treats the same cases as `unreadable`; FL
  should too.
- **Suggested fix.** "unreadable" when fewer than `min_entry` step-30 reads are labellable (label not in UNLABELLABLE,
  x finite). Give "no_carry" only when at least `min_entry` seeds are labellable and fewer than `min_entry` carry.
  Change the test at :538 to expect "unreadable".
- **Severity.** Major (a wrong branch is possible, and the expected FL gate step of 13-17 has not been measured).

### 3. MINOR: the prior is admitted on one set of windows and checked on another

- **Where.** `amend6.py:69-76` (the per-key geometric mean over the train and held-out row splits) against
  `amend6.py:120-122` / `steering.py:108`. The checked `w0_geo_p.trained` is taken over the trained keys' held-out-row
  windows only: 286 windows, about 3 per key (job 46398 log).
- **Failure scenario.** A value admitted at 2.1e-5 on all its windows can read under 2e-5 on its 3 held-out windows.
  The aggregate is an average over 96 keys, so a stop is unlikely, but not impossible if the draw piles up at the
  bottom of the band. Pool tokens are denser at lower priors, so the first admissible token in a random order sits
  low, likely about 3e-5, half a decade under copy's 5.9e-5.
- **Power check (no fix needed).** Interpolating r on log prior from shuffle (1.04e-5, 0.477) to copy (5.9e-5, 0.77)
  gives about 0.39 r per decade. A prior-only r would then read about 0.59 at 2e-5, 0.66 at 3e-5 and 0.77 at 5.9e-5.
  Anywhere in the band, a PM r under 0.5 separates from the prior-only null. Suggest reporting that interpolated value
  at the realised prior beside PM's r.
- **Suggested fix.** Admit values on the same windows as `lp0` (the held-out-row split), or keep it and note that the
  stop is possible.

### 4. MINOR: the fallback pool changes token kind without changing the reading

- **Where.** `amend6.py:55-62, 73-74`.
- **Failure scenario.** If the cipher pool runs out for PM (128 distinct values, each needing a prior in the band and
  a rank over 100 after its key; the greedy draw can fail where a matching exists), the draw moves to every word-like
  token from rank 1000. The PM branch then reads as "kind at equal prior" when kind is no longer E58's cipher pool,
  which is exactly the case the filed "blind to" line assumes away. The fallback is recorded in `tables.prior_cipher_pool`
  but no rule reads it.
- **Suggested fix.** File it: if `prior_cipher_pool != "cipher_pool"`, PM is reported, not scored (or the job stops).

### 5. MINOR: a letter's word can start with that letter

- **Where.** `amend6.py:82`.
- **Failure scenario.** Group "s" can draw " system". Then the rule is partly "say a word with the key's first letter",
  a cue W₀ may hold in its spelling features, which bears on the granularity-limit reading of an own-guess-band FL.
  The prior guard (< 1e-5) keeps it behaviourally new.
- **Suggested fix.** Exclude words whose first letter is the group's letter, or record the count in `tables`.

### 6. TRIVIAL: the noise floor is rounded down

- `predictions_amend6.json` thresholds.noise is 0.045, but the source is 0.0452. Only `near_cut`, a reported flag,
  uses it. Use the measured value, or say it is rounded.

## Checked and sound

- **`kv.key_distributions` frozen callers** (anchors, keydep.select, kv_task, new_vs_existing, keydependent_elicit,
  amend1-3). With `geometric=False` the ops are the same (`f = torch.softmax`, same arguments, same reduction), so
  the results are bitwise identical. The geometric path takes exp of the mean of the fp32 log_softmax in float64, as
  documented, and the new test checks it against a hand-computed alternation.
- **Harness control.** Same path as `amend4.cipher`: `amend4.train(ctx, d3, "cipher", 0, CIPHER_CHECKS)` on
  Amendment 3's sha256-checked cipher windows, then `amend3.read(vpd=True)` at `amend4.json control.0.step` = 16, with
  reference r 0.32412633847537903. The thresholds `read` uses (G_min, k_sd, n_boot, boot_seed, ratio_floor,
  block0_nats) equal Amendment 4's. The 1e-4 tolerance rests on a measured floor: Amendment 4's seed 0 reproduced
  Amendment 3's r to every digit at steps 10 and 30 (0.3683455521718003, 0.3038012406525232). The card cannot change,
  because the compute partition is one node of 8 L40s (`cluster/README.md:3`; 46426 and 46450 both ran on an L40).
  Amendment 5 did not re-run seed 0, so the floor comes from A3 to A4, not A5. The extra W₀ forwards before training
  did no harm in A3 either.
- **PM leakage.** `kv.bijection` gives 128 distinct values, so no held-out key's value is trained; the keys and elicit
  words are banned from both pools. The rank-over-100 filter (`topk(100).values[-1]`, strict <) excludes W₀'s top-100
  per key.
- **Key split** checked against `amend3.json keys.held`; the windows are sha256-checked; the split is by key
  (`split_task` asserts the counts).
- **FL "new under W₀":** every key of a letter gives the word a prior under 1e-5. **Learnable to carry:** 31 of 32
  held-out keys share a letter with a trained key; " video" (v) does not and dilutes x by about 3%. The letters are all
  alphabetic (the decoded keys are space-prefixed words).
- **References:** every key `references()` reads exists (amend3 `elicit/rank1/copy@30` and maps' `w0_geo_p`; amend4
  `cipher_s0-2@30`, `<map>_s1/s2@30`, `shuffle_s0@30`; amend5 `shuffle_s1/s2@30`).
- **Rule edge cases.** `band_class` ties (0.5 between, 0.6 upper), empty input, None and nan; PM too-small counts as
  missing; a gate step of None gives the key `@None` and so `unreadable` at the gate step (reported only). Spearman
  returns nan, never None, so the final log line cannot crash.
- **Runtime.** 46426 ran 10 fine-tunes from 17:07:34 to 17:18:58 (11.4 min after start-up); 46450 took about 80 s per
  shuffle seed. Seven fine-tunes plus the W₀ stage come to about 12-15 min, inside the 30-min cap with room to spare.
  `--mem=24G` covers the ~5 GB of checkpoints.

## Not run

- No GPU execution: the map builders, the geometric prior and the realised priors and letter counts in `tables` are
  unmeasured until the job runs.
- The size of the bias in finding 1 is not estimated beyond the count asymmetry; it needs the fine-tune.
- Whether the PM draw exhausts the cipher pool is not checked (it needs W₀'s per-key priors).
