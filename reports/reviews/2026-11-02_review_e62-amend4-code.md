# Review of E62 Amendment 4's code before its run

*2026-11-02. Fresh agent; wrote neither the code nor the plan. Scope: commit 56c6981, `experiments/induction_cue/amend4.py`, its rule functions in
`src/commute/copying.py`, `tests/test_copying.py`, `cluster/induction_cue_amend4.sbatch`, `results/predictions_amend4.json`, and the plan's "Amendment 4 (2026-11-02)". Read by hand, with the draft's option 1.
On the CPU worker, with no model loaded: `tests/test_copying.py` (208 pass), the real `windows.pt` and `alive.pt`, and `mask_test` on three cases. Nothing ran on a GPU.*

**In plain words.** I found no bug that makes a number wrong: positions, targets, the order of the subcomponents, the head switch and the re-read all do what the plan says, and the job should finish in its time limit.
The trouble is what the numbers can be read as. Nothing checks that the subcomponents called "W0's copying" carry W0's copying, the contrast that picks them mostly separates a key token from a plain token, and the filed `not_shared` fires on data that refutes the filed null. Since `not_shared` stops the 67M copying work, those three should change before the run, because they cannot change after it; each costs at most a few forwards at $0.

## Findings

### High

**H1. Nothing tests that `S_old` carries W0's copying, and this is the control the whole read is calibrated on.** `amend4.py:68-69, 80`; draft line 142-143.
Stage 1 tests that W0 copies at the second A and that head (2,4) passes through W0's mask. `S_old` enters stage 1 only as a count k. The masked fall can pass with `S_old` unrelated to copying, since it needs only (2,4)'s `o_proj` slot open at the second A.
- *Failing scenario.* The control passes, `S_old` is 60 subcomponents of no causal weight for B at the second A, and stage 3 reads closing share 0.02: `not_shared`, "the masked gain does not run through W0's copying subcomponents", when the set never held them. Or share 0.6 with a low lookup: `shared_pieces` on a set whose only claim to the name is "more open at A than at a plain token".
- *Fix.* One masked forward in stage 1, W0, read at `half - 1`, target B: `S_old`'s gates at `half - 1` set to their per-window values at `half - 2`, and the same for 5 random sets of k. Expected: the masked log-odds fall by at least 0.25 of the target's fall, the same quarter bar; random sets fall much less. A miss exits 1 as `unreadable`, or at the least is printed and blocks both `shared_pieces` and `not_shared`. Also print `S_old`'s overlap with `slot_reads` for (2,4) (no forward), the one direct link between the head label and the set.

**H2. `a` contrasts a key token with a plain token, so `S_old` may be mostly what reads keys, and the cheap contrast that cancels this is left to "a rerun's line".** `amend4.py:67-68`; plan line 688 and "Not run".
The real windows hold 32 keys at `half - 1` and 164 distinct plain tokens at `half - 2` (checked). The plan admits key-reading subcomponents stay in `a` and relies on the lookup to separate them. The lookup protects only `shared_pieces`: key-reading subcomponents hurt the lookup, which depends on the key, so a false `shared_pieces` is unlikely. It does not protect `not_shared`: if `S_old`'s half of the mass is taken up by key-reading subcomponents, copying subcomponents further down the ranking are left out, closing reads low on the copying map, and the verdict stops the line.
- *Fix.* One more Γ pass, on the pair-broken windows `CP.broken_pair(v["read"].tokens, v["p_read"], v["pools"]["keys"], half, S["control_seed"])`, as in `amend2.py:247`: same key at the same position, first A overwritten. `a_broken` = W0's mean gate at `half - 1` on the intact windows minus the same on the broken ones: what W0's mask adds because a copy is available. Its `S_old`, cosine and closing share go beside the filed ones, with one masked forward per state. Better, make it the primary contrast. The plan's first-A contrast costs no forward and should be printed too, but Γ reads B next to the first A there, so the pair-broken contrast is the cleaner one. Gates are not saved, so if it is not read now the job must rerun.
- *Also from the same pass.* The target and masked lean at the second A (intact minus broken). `p_b_min` compares P(B) with another token's position, and B's prior after A with no pair is unread. The lean is the direct read of "W0 copies here" and is what M1 needs.

**H3. `not_shared` can fire on data that refutes the null.** `copying.py:887`; predictions line 66.
The first arm reads "misses the absolute bars on every seed", not the null's "no more than chance, no more than random". I ran these through `mask_test`, the same on all three seeds, gain 13.8, k 50:

| Copying map: cosine, closing share, random median | Lookup: cosine, closing share | Verdict |
|---|---|---|
| 0.19, 0.49, 0.02 (chance 0.005) | 0.01, −0.05 | `not_shared` |
| 0.19, 0.30, 0.02 (chance 0.005) | 0.01, −0.05 | `not_shared` |
| 0.50, 0.01, 0.02, with k too small for any closing to matter | 0.01, 0 | `not_shared` |

The first row is 38 times chance, and closing costs half the gain against 2% for random, with the lookup untouched. That is departure 1's class of error, the opposite of the null, which the plan fixed only for the lookup bars. The third row is what the plan's departure 6 foresaw ("the report must read it beside any P2 miss", line 661), but the verdict string and the plan's next step ("the stop") do not read `largest_k_share`.
- *Fix.* The first arm: on every seed `share - random_median < random_margin` and (`cos < align_min` or `cos < chance_factor * chance`). The whole branch, both arms: also on every seed `largest_k_share >= close_min`; else `in_between`, printed as no power at this k. Pass `largest_k_share` into `mask_test`, add the table's rows to the tests, and file it as a further departure.

### Medium

**M1. The within-2 bar reads a level that every other number here nets out, and these windows already miss it at the marker.** `copying.py:852`; predictions line 40 ("no published source").
W0's masked read is 2.01 under its target read at the marker on these windows (−10.07 against −8.05, `amend2.json`, `amend3.json`), and the base-masked-error review measures W0's masked error at up to about 2 nats on this line (against every mask open, a related offset). Every reading in stages 2 and 3 subtracts L00, so the offset does not enter them.
- *Failing scenario.* At the second A the target reads −3.0 and the mask −5.1, with a masked fall of 2.0 against a target fall of 3.0. The mask carries the head fully, the job exits as `mask_off_target`, and the filed text says "no mask test can read shared mechanism on this model". The offset may well be smaller at the second A, where B's probability is high enough for the KL objective to care, so this is a risk and not an expectation. A stop costs a resubmission at $0, but also an amendment and a report to the user.
- *Fix.* Print the offset, and replace the bar by the masked lean over the pair-broken windows (H2) at least 0.25 of the target's lean: a difference, like the rest of the read, and a direct test of "the mask carries W0's copying".

**M2. Nothing checks the closing read on the real model, and non-alive subcomponents sit outside it.** `amend4.py:90, 93, 104, 115-118`.
`zero_closed_max_abs` cannot fail on its own. When the zero update's gates are W0's, which is what `zero_gain_max_abs` of 0 points to, closing to W0's values is a no-op whatever `closed` does. `closed` is tested on small tensors and a tiny model only. And 29,031 of the 38,912 subcomponents are not alive (`alive.pt`): their gates are in neither vector and are never closed.
- *Failing scenario.* Part of the 13.8 gain runs through subcomponents below the alive threshold that the state opens at the marker. No closing can reach it, so the 0.5 bar is partly out of reach with nothing printed to say so.
- *Fix.* One masked forward per state, reported: `close(torch.arange(n))[2]`, every alive subcomponent closed. Expected 1, up to what non-alive gates carry. It is the read's only positive control end to end, and it bounds what any `S_old` could reach. A value well under 1 goes beside any P2 miss.

**M3. The two references give little.** `amend4.py:119-120, 124`; `copying.py:804-811, 820-822`; plan line 693.
- `closing_pool` draws from any Δ > 0. The plan expects most of it barely open, so the random median is near 0 and `random_margin` rarely decides. A share of 0.01 against a median of 0.00 versus 0.02 flips `not_shared`'s second arm on noise from 5 draws.
- `largest` may include `S_old`, so "at or over the closing share" is close to automatic where they overlap. It says nothing about whether `S_old` matters more than other subcomponents opened as far.
- *Fix.* Report the k largest openings **outside** `S_old`, one more masked forward per state. A closing share above that one is the claim "these subcomponents rather than others"; no rule needed. Together with H3's change, the near-zero median can no longer decide the verdict alone.

### Low

**L1. `control_failed` names four causes and the filed reading fits two.** `amend4.py:76-77`; predictions line 61; plan line 602.
`no_copying` and `head_not_carrying` are about the target model on these windows, not the masks, yet the text says "the masks drop W0's copying", with the weighted slot read next. The status string is kept, so nothing is lost; the plan's next step should branch on it. The draft's fallback for `no_copying` (repeated sequences for `a`) has no code, which is fine if the plan says so. Also, `out["verdict"]` is a string on a stop and a dict on a finished run: anything reading it, such as a plot script, must handle both.

**L2. The cosine counts shared closings as agreement.** `amend4.py:122`; `copying.py:791`.
`a`'s negative entries (more open at a plain token than at A) times Δ's negative entries add to the cosine, while the claim says "opens". *Fix.* Print the cosine of the positive parts beside it, no forward. `old_opened` covers part of this already.

**L3. Alignment reads the marker and closing reads every position.** `amend4.py:113, 116`.
A gain from openings at earlier positions (B, the second A) moves the closing share and not the cosine, so the two can disagree for a reason no number names. It is partly in Blind to; a P1 and P2 split should be read with that in mind. No change needed.

## What I checked and found right

- **Positions and targets.** On the real `windows.pt`: `second_a_holds` is true, p runs 2 to 16, the first 41 columns are equal across maps, one marker token, the last column is B on the copying map and differs from B on all 200 lookup rows. `at = replace(R, half=half - 1)` moves only the read; gates do not depend on `half` (`route_cells`, and the test on a tiny `ComponentModel`).
- **Subcomponent order.** `gate_means` and `closed` both use sorted keys and index order, on CPU tensors. The alive count is 9,881 over 24 matrices, as filed, so the `n_alive` stop will not fire falsely.
- **Head off.** `ablated` wraps only the masked forward in `at.masked(w, g0, H)`; `g0` is read outside it. As `amend2.py`.
- **State handling.** `R.gates(w, dW)` restores the model on exit, and `R.masked` runs at W0's original subcomponents, Δ out. So closing's level and L01 are the same forward with different gates, and `share` matches the filed definition with the same L00 on both sides. `R.share`'s return order is unpacked correctly at `amend4.py:118`.
- **One Γ pass for both maps.** The target column is never fed to the model (`ci_len` 41) and batches are the same 64, so `zip(strict=True)` holds. Stage 2's use of `g0` on the lookup matches Amendment 3's L00.
- **The re-read will not stop falsely.** No shared code it uses has changed since Amendment 3's commit f92b3ec (`local_faith`, `vpd`, `keydep`, `steering`, `anchors`, `faithfulness`, `amend2`, `amend3`, the lock file: empty diff; `copying.py` only appends). `amend3.json` has both keys read, and same-card repeats read 0.
- **Rules as filed.** Thresholds match the predictions and every key read exists. `mask_test` agrees with the filed order, the seed pairing is right (`NAMES[:3]`, `NAMES[3:]`), P2 does imply P3, and the edge cases in the tests read as stated. 3.154e-4 is the geometric mean on these windows (`run.py:183-185`), and 1.1628 is `row4_w0_fall`.
- **Job.** Partition, QoS, GPU, memory and time are the same as Amendment 3's sbatch. `pipefail` and `|| rc=$?` carry the exit code through `tee`. Gates are about 1.3 GB per set at 200 × 41 × 38,912 in float32, so at most about four sets are live, inside 24 GB. About 16 forwards per state, the same as Amendment 3 (4.5 min); the H1, H2 and M changes add a handful, still far under 25 min.
- **Saves.** `amend4.json` after every stage and state, every number printed. The vectors are about 0.6 MB and saved only at the end, but they can be read again for $0.

## Skipped, and why

- The 0.2 cosine bar, the factor of 3 and the 0.25 are unsourced but filed openly as the draft's. No positive control can calibrate the cosine here, so I do not propose another number; H1 and H2 are what would give it meaning.
- Γ reading M when it writes the gates at `half - 1`: real, and already in Blind to. `a_broken` keeps M in both arms of the contrast, so H2 largely answers it.
- The prediction file's "random median 0.05 to 0.2" sits oddly beside the plan's "near 0". No rule reads it.

## Not run

The 67M was not loaded and no stage ran: no number at the second A, k or `S_old` is known, so whether H1, H2 or M1 would bite on this model is unread. A stub run of `amend4.py` was not repeated. Lint was not run. No fix above has been coded or tested.
