# Review: E58 Amendment 7 code (commit af9384b, merged as 82e1d1a), before the cluster run

**In plain words.** I read the Amendment 7 diff and the code it calls. I also ran one CPU check on the embedding rows of
the 128 keys (no forward pass). Nothing in the code will crash or misread a filed rule, and the job fits the 30-min cap.
Two things should be fixed before the run, because each could push the class map's carry reading up without a rule.
First, the principal components are fit on all 128 keys, held-out keys included, and they move a lot when those keys are
left out. Second, the cross-class null cancels any lean toward a word that does not depend on the key. It does not
cancel a lean whose size differs by class, and the class map's classes track embedding norm and token id, which the
twin's do not. The rule can be learned from embeddings: a linear probe trained on the 96 trained keys gets 75-84% of the
held-out classes right, while the twin's labels read at chance. So the run is not doomed to read no_carry.

Scope: `git show af9384b` (`experiments/open_pair/amend7.py`, `src/commute/open_pair.py` § Amendment 7,
`results/predictions_amend7.json`, `cluster/open_pair.sbatch`, `tests/test_open_pair.py`, plan § Amendment 7), plus
`amend6.draw / save_map / harness_control`, `amend3.read`, `amend4.train / over_gate / stop`, `heldout_keys.specific /
pool_of / others / behaviour`, `kv.bijection`, `cluster/env.sh`.

Check script: `~/commute-data/scratch/regating-lead/a7-review/check.py`. It was run once through `fleet/cpu-run` with
the clone's `.venv/bin/python` and `PYTHONPATH=src`. It reads only the `wte` rows of
`artifacts/vpd/t-9d2b8f02/ckpt/model_step_99999.pt`. It reproduced the filed counts (39/36/33/20; PC shares 2.69%,
2.44%).

## Blocking

None. No crash path, no mis-scored edge case, and no environment fault found.

## Fix before the run

### F1. The principal components are fit on held-out keys too, and are unstable without them (Medium-High)

`experiments/open_pair/amend7.py:71` calls `OP.pc_classes(wte[keys])` on all 128 keys (`open_pair.py:575`).

Measured:
- PCs fit on the 96 trained keys only point at cos 0.61 and 0.63 to the all-key PCs.
- The two partitions agree on 69 of 128 keys, and on 17 of the 32 held-out keys, after the best relabelling.
- The held-out keys sit far from the boundary on the all-key PCs, because they helped place it: mean |PC1 score| is
  0.099 for held-out keys, against 0.076 for trained keys on the same PCs. On the trained-only PCs it is 0.042.

Failure scenario. The class of a held-out key is partly defined by that key's own embedding, so the rule is not one
"that carries to unseen keys". The held-out keys are also the easiest keys to classify. Both effects push x up for
reasons the twin does not share, since the twin's labels are random. The linear probe does not show inflation (0.75
held-out accuracy under either set of PCs, min-norm), so the size of the bias on x is unknown, not shown to be large.

Fix. Fit the PCs on the trained keys only: centre on the trained mean, take the SVD of the trained rows, project all 128
rows, and fix the signs on the trained rows. Recount the classes. The probe accuracy is unchanged (0.75), so power is not
lost. Refile the counts before the run (nothing has run, so this is a pre-run amendment, not a refiling). If the writer
keeps the all-key fit, add it to CL's `blind_to`.

### F2. The cross-class null does not cancel a lean whose size differs by class (Medium)

`open_pair.py:593` (`cross_class_null`), used at `amend7.py:108-116`.

What cancels. The specific gain is s[c][i] = Δ_c(i) − mean of the other three words' Δ at window i
(`heldout_keys.specific`, pool = the map's four words). It sums to 0 over the four words at every window. For any
Δ_c(i) = a_c + b_i, s[c][i] does not depend on i, so own = null exactly. This covers both degenerate cases asked about:
- "favours big-class words only" (a_c rising with n_c);
- "favours one word only" (a_0 > 0, others 0).

In both, x − x_null reads 0, at any class sizes. The writer's test covers the first.

What does not cancel. A multiplicative effect, Δ_c(i) = a_c · g_i: a word lean whose size at a key follows how strongly
the update moves that key at all. This gives

own − null = Σ_c (n_c/N) (a_c − ā_{−c}) (ḡ_c − ḡ_{¬c}),

where ḡ_c and ḡ_{¬c} are the mean g_i at class c's held-out windows and at the other windows. Worked case: held-out
counts 12/10/6/4, only word 0 leans (a_0 = 1), and class 0's keys respond 30% more (g = 1.3 against 1.0). Then
own = 0.279, null = 0.136, and own − null = 0.14 in units of a_0. With a_0 near G_spec, that is x − x_null ≈ 0.14, just
under the twin bar of 0.15 and half the entry bar of 0.3.

Why it is plausible here. The class map's classes are not neutral to features that could set g:
- η² of embedding norm on class is 0.070 for the class map and 0.009 for the twin;
- η² of token id (a frequency proxy) on class is 0.116 for the class map and 0.025 for the twin (chance ≈ 0.024).

So the twin cannot measure this term: its classes have no link to norm or frequency.

Fix (cheap, reported, no change to the filed rule's thresholds):
- record own_c and null_c per class in the read, so the analysis can see whether a "carry" sits in the one class whose
  word leans;
- add a reported amplitude-normalised version: divide each window's s[c][i] by that window's mean |s| over the four
  words (or by its raw pool lift), which removes g_i;
- add the multiplicative case to `test_cross_class_null` with its hand value, and add "a lean whose size follows the
  key's class" to CL's `blind_to`.

## Lower severity, one line each

- **F3 (Low).** `amend7.py:73-77` logs the class counts but never checks them against the filed 27/26/27/16 and
  26/31/27/12. Assert them in the W0 control: a wrong tensor (another embedding, a tied or rescaled weight) would then
  stop the job instead of silently changing the rule. PC2 has a key at 0.002 sd from the boundary, so the check also
  guards the fp32-on-GPU path.
- **F4 (Low).** The twin's smallest trained class is 12, exactly `min_class_trained` (`< 12` stops), so it passes with no
  margin. That is fine as filed. Note it, since any change to the split or PCs (F1) needs the twin recounted.
- **F5 (Low).** The twin is not chance-correlated with the PC signs or token features. Held-out agreement is 10 of 32
  (chance 8.6), η² of PC1 score on twin classes is 0.025 and of PC2 is 0.003 (chance ≈ 0.024). The probe reads the twin
  at 0.25-0.34 held-out accuracy (majority 0.41). The filed "41 of 128 keep their class" overstates the risk: on the
  held-out keys the excess is 1.4 keys.
- **F6 (Low).** The twin is not matched in trained class sizes (12-31 against 16-27) or held-out sizes (13/5/6/8
  against 12/10/6/4). Its word-frequency lean is therefore stronger. The null cancels that for x, not for r (D). This is
  covered by "differences the fine-tune creates" in D's `blind_to`, but name the class-size imbalance there.
- **F7 (Low).** No ‖δW‖ or G_spec matching between the twin and the class map, by design. G_spec at step 30 is reported.
  OK as filed.
- **F8 (Low).** `amend7.py:130` passes the class map's classes as `letter_of` to `save_map` for every map. So
  `maps.class_shuffled.n_held_letter_trained` and the EP's are computed on the wrong classes. This is diagnostic only,
  but misleading in amend7.json. Pass each map's own classes, or drop the field.
- **F9 (Low).** EP pool exhaustion (`amend7.py:84-85`, `kv.bijection`) can fail in two ways:
  - The bijection is greedy in key order, so it can fail where a matching exists.
  - A key whose 20th-largest prior is under 3e-4 has no admissible value at all, so both pools fail for that key alone.

  Either gives "skipped", which is handled. `draw` logs only the pool name, not the AssertionError's key. Log the key
  and each key's 20th prior, so a skip can be told apart from a narrow band. The elicit words, the most natural
  candidates near 1e-3, are banned, which makes exhaustion likelier. The fallback to `ranked_from_1000` is recorded and
  makes EP unscored, as filed.
- **F10 (Low).** The class words can also fall back to `ranked_from_1000` (`amend7.py:80`). That is recorded in
  `maps.<map>.pool`, but CL is not marked unscored on a fallback, unlike EP. It is unlikely at 4 words under 1e-5;
  state it in the plan.
- **F11 (Info).** `twin_max` 0.15 is Fable's bar, not a measured noise floor of x − x_null for a structure-free 4-word
  map. Amendment 6's first-letter rule read 0.04-0.08 at step 30 and up to 0.19 at the gate step. One seed over 0.15 by
  chance stops the job. That is a safe failure, but the 85% lean may be high.
- **F12 (Info).** Learnability. PC1 and PC2 are above a column-permuted noise edge (2.69% and 2.44% against 95th
  percentiles of 1.58% and 1.53%), so the classes are real structure in the keys, not noise. They are weaker than a
  random token set's leading direction (median 4.8%). A linear probe from the trained keys' `wte` rows reads 0.75
  (min-norm) to 0.84 (ridge) held-out accuracy, against 0.25-0.34 for the twin. So the embedding carries the rule
  linearly, and the run is not predictably no_carry. Whether 30 Adam steps at 3e-5 find it is not testable without the
  GPU run. The twin-first order is already the cheapest early check.
- **F13 (OK).** Environment: `cluster/env.sh` exports `PYTHONPATH=$COMMUTE_REPO/src` ahead of the shared venv's
  `commute.pth` (which names `commute-lead/src`), and the sbatch sources it from `$SLURM_SUBMIT_DIR`. So submit from
  this clone's root. Optional: record `commute.__file__` in amend7.json's header.
- **F14 (OK).** Runtime. The A6 measurement was 7 fine-tunes in 5.8 min plus about 1 min. A7 adds 3 fine-tunes (about
  1 min), 12 more VPD reads (twin 6, class map up to 6, EP 6, against A6's 6 plus harness; about 3 min at 15 s), the
  null at 3 steps × 6 seeds (about 2.5 min) and step-100 labels (about 0.5 min). The total is about 14-17 min, inside
  the 30-min cap with margin. Step 100 is in `KD.CIPHER_CHECKS`, so `ft.deltas[100]` exists.
- **F15 (OK).** Rule edge cases:
  - `twin_clean`, `contrast` and `score_amend7` handle a missing seed, too_small, nan x_null, nan r, the tie at 0.3,
    the tie at 0.15 and the tie at the D bound as filed.
  - `cross_class_null` returns nan for no windows and for a class with no outside windows.
  - A class with no held-out window drops out of both terms with consistent weights.
  - `rule_seed`'s VPD re-read replaces the record and re-applies the null, as in Amendment 6.
  - The own-term check (H_spec within 1e-6) validates the window order of the four constant-target copies on the GPU.

## Not run

- No GPU run, no forward pass: the 30-step learnability of the rule, g_i's actual dependence on class (F2), the EP pool
  sizes and each key's 20th prior (F9) are not measured.
- The F1 alternative (trained-only PCs) was only counted for agreement and probe accuracy, not refiled.
- No check of the class words' priors or of `kv.key_distributions`.
