# E116: the 67M task table under the frozen definition (rows 81, 87, 164)

*definition-lead, 2026-11-06. Direction: the user's retarget brief, step 1
(`reports/decisions/2026-11-06_decision_definition-retarget.md`). Approved class 1: a re-reading of saved states with
existing readouts; no fine-tune, no new task pair. One MATS `debug` L40 job, $0. Apparatus detail, per-row sources and
edge cases: `reports/plans/2026-11-06_plan_e116-appendix.md`. Status: settled by the lead (questions below); awaiting the user's one reply before code.*

## The request, restated in the Terms' words

For each 67M fine-tune of the old table (a finished update δW on W₀), read the frozen reuse definition part by part:
name a capability C; train a linear readout of C on Pile text and check W₀ possesses it (row 81, Def 1, against a
Hewitt-Liang control); check the fine-tuned map is a small function g of that readout that carries to held-out values
(row 87); remove the readout's direction and check it breaks the fine-tune and spares a twin at the same trigger (row
164). Report the locator's part-level revert (E100 "own", E97) in its own column as row 81's "pruning a few neurons"
form, and say whether the earlier label agrees. A constant-answer task reads "does not apply" and stops. A part the
definition cannot read is "unreadable" with its reason, never a new rule.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s "new capabilities vs strengthening" needs labels set outside VPD. This is the label the other
  lines read against, on the states they already read (E100, E97, E12, E54-E58).
- **Nearest tried.** E90 (`TRIED.md:190-192`) did this table under rows 115/116: admissibility clauses, variance bands,
  a replaced control, no readout for token identity. Row 164 dropped all of that. E112/E113 read one new pair under the
  frozen rules. New here: the frozen rules on the old tasks, with held-out carrying read as the definition's own part
  and the locator column beside. See "What E116 learns that E90 could not".
- **Step or patch.** A step: the sanity check the user asked for before the Qwen anchors (step 2).
- **Working decisions.** Reads under rows 81, 87, 164 only; thresholds and verdict rule copied from E112's
  `experiments/case_label/results/predictions.json` lines 51-66. No departure. Where E112's rules were filed for a
  two-valued C and the task's C has m values, the m-valued statement is given below and anything that is not an exact
  restatement is a question for the lead, not a rule.
- **What changes next.** Any row that reads "reuses C" or "not through C" is a label the other lines can use at once.
  Rows that read "unreadable" say, per part, what shape a task must have for the definition to read it: that is the
  input to step 2 (which Qwen anchor can be read at all). Nothing here reopens the parked ask 7.
- **Control first.** Stage 0, alone: the moved probe code re-reads E112's saved task and reproduces its stage-0
  possession numbers (accuracy 0.683/0.994/0.885/0.763, control 0.476/0.524/0.464/0.479, `case_label/results/stage0.json`)
  within 1e-3, else stop. Then W₀'s readouts and the do-nothing control before any state is read.
- **Numbers.** P0-P3 per row as E112 defines them, the do-nothing share, the locator column copied from JSON. Each has
  an expected value in the table below. Nothing else is scored.
- **Saved for audit.** Per-C probes (`results/large/probes_<C>.pt`), per-state P1-P3 and do-nothing shares with the
  window counts (`results/stage{0,1,2}.json`), the readout classes and their Pile position counts.

## What E116 learns that E90 could not

E90 printed verdicts its own added rules gave: "no C listed" came from admissibility, "reuses C" on the copy maps from
a band and a control replaced after it failed, and token identity was never read by a trained readout against a
Hewitt-Liang control (`reports/analyses/2026-11-05_analysis_e90-old-labels.md`, "Weak control in Amendment 1"). E116
reads only the frozen parts and reports, per part, a number or the reason it cannot be read. Three things are new:
held-out carrying read as P1 on the 24-key tables and on E62's copy, whose read windows use B values never trained
(`src/commute/copying.py:311`, "no B is both trained and read"); the Hewitt-Liang control on every C (E90's "shuffled"
was a per-window shuffle, `experiments/new_vs_reuse/reading.py:118`); and E100's part-level revert beside each verdict.
What it cannot learn: anything on a constant-answer task, and whether "reuses token identity" means more than needing
the token (E95's open question).

## The table

Columns: C; P0 possession at the site (the locator's block where E97/E100 give one before the last block, else E112's
l*); P1 carrying; P2 small function; P3 removal with the twin spared; locator part-level revert (E100 own share of the
top part; E97 located part; the sums of single reverts and JSON paths in appendix §3); earlier claim; agrees/disagrees.
"n/a" = the part cannot be applied, reason in the appendix.

| Task | C named | P0 expected | P1 expected | P2, P3 expected | Locator revert (E100 own; E97) | Earlier claim | Expected reading |
|---|---|---|---|---|---|---|---|
| ident32 | token identity of the key | n/a: C is a function of the token type alone, so there are no held-out types to score on and the Hewitt-Liang control is the same task relabelled (beside: both ≈ 1) | n/a: no held-out keys exist (appendix §2) | no l* | mlp3 0.047/0.045; E97 not run | reuse, by construction | unreadable at P0; no agreement either way |
| copy-24 | same | n/a as above | read: 8 held-out keys; no lean (E58's x 0.42-0.55 is a different number) | no l* | mlp3 0.079/0.065/0.074 | reuse, by construction | unreadable at P0 |
| E62 copy | in-context copying: the token after the earlier occurrence of the token before the trigger | verdict-bearing; site block 2 (E97 located (2,3)+(2,4), 0.87/0.88/0.78, lookup 0.007); lean: fails 0.90 before the last block (W₀'s p(B) at the trigger 0.008, E97 `p_mean_w0`) | ≥ 0.90 on the 200 read windows (unseen Bs; E62 read 0.89 at step 10) | P2 ≥ 0.90 (g = identity on classes); P3: do-nothing at rank m−1 leans over 0.10 → stop (Q3) | (2,4) 0.008/0.007/0.005; E97 located | reuses, causal (head (2,4)) | unreadable at P0 or at the do-nothing; if both pass, "reuses C" needs P3 ≥ 0.50 (E90's head removal took 0.47-0.56) |
| E62 lookup (twin) | token identity of A | n/a (type-level) | n/a: a lookup defines no held-out target | spared under copy's C: P3 ≈ 0 (E90: −0.004 to 0.000) | mlp3 0.136/0.147/0.139 | new, causal | the spared control only |
| lowercase | — (no content function; the user's reset) | does not apply, stop | | | mlp3 0.156/0.158; E97 (1,1)+(1,2) 0.798/0.853, W₀ falls 1.10 nats | reuse | does not apply |
| rand32 | token identity of the key | n/a (type-level) | n/a: no held-out keys | no l* | mlp3 0.139/0.135 | new, by construction | unreadable at P0 |
| rand1 | — (constant answer) | does not apply, stop | | | not in E100's table | new | does not apply |
| text ciphers 50 / 500 | W₀'s next-word guess in the ciphered context; g = the table | verdict-bearing if the lead takes Q4's site; lean: fails 0.90 before the last block (E90's guess readout 0.68/0.78/0.88 at blocks 0-2, `new_vs_reuse/results/stage1.json .readout.acc`) | n/a: the table defines no held-out target | P2 as Q2; P3: no twin at a trigger, so "spares the twin" is n/a; the state's share printed | mlp3 0.479/0.494; E97 none, control not spared | new, by construction | unreadable at P1 at best |
| cipher-24 | token identity of the key (E58: no key-to-guess link) | n/a (type-level) | read: 8 held-out keys; lean under 0.90 (x −0.30 to −0.38, "new") | no l*; as rank-1's and elicit's twin under W₀'s guess: P3 printed (E90 Am2 0.08-0.12) | mlp3 0.135/0.128/0.129 | new, by construction | unreadable at P0 |
| emoticon | — (constant answer) | does not apply, stop | | | mlp3 0.197/0.186 | existing behaviour | does not apply |
| elicit | W₀'s next-word guess at the key (half the values are W₀'s own) | verdict-bearing; lean fails 0.90 as for the ciphers | read: 8 held-out keys; lean under 0.90 (x ≈ −0.03, "new") | P2 as Q2; P3 do-nothing leans stop | mlp3 0.125/0.128/0.124 | open | unreadable at P0, else "not through C" on P1 |
| rank-1 | W₀'s top-1 guess at the key; g = identity | same readout as elicit | read: 8 held-out keys; W₀ passes by construction; the state leans ≥ 0.90 | P2 ≥ 0.90 plausible; P3 do-nothing leans stop; if read, E90 Am2's 0.16-0.22 is under 0.50 → "not through C" | mlp3 0.193/0.182/0.181 | old, by construction | unreadable, else "not through C": disagrees |

The expected table is mostly "does not apply" and "unreadable", for three structural reasons: five rows have a
type-level C that E112's P0 cannot score; four rows have a many-valued C whose readout leaned under 0.90 before the
last block in E90; and a rank-(m−1) removal's random twin leans over the 0.10 do-nothing bar. Each is a measured
reading of the frozen rule, not a verdict added. That is the finding the user asked for: where the definition can read
and where it cannot, before the Qwen anchors.

## Thresholds and the m-valued statement of E112's rules

Numbers unchanged from E112 (`predictions.json:51-57`): P0 accuracy 0.90 and selectivity 0.30, l* the earliest passing
block before the last; P1 0.90; P2 0.90; P3 0.50; do-nothing ≤ 0.10 on every state read under a C, else that C's P2/P3
are not read; gain floor 0.1 nats. Verdict rule unchanged (`:66`): P0 fails → "unreadable"; P1, P2, P3 all meet →
"reuses C"; else "not through C"; a constant answer → "does not apply". For a C with m values:

- **P0.** One ridge probe per value (one-vs-rest, `removal.fit_ridge`, λ = 1e-3 as E112), read as the argmax over the
  values. For m = 2 the two probes are complements, so argmax is E112's "score > 0.5" exactly. Accuracy = top-1 on the
  task windows of held-out types; control = the same probes with one seeded random value per type of the token before
  the position (E112's `case_label.control_y` with values for bits). The probe's Pile positions exclude the task's own
  key types, as E112's excluded held-out pairs. An exact restatement.
- **P1.** The state's top-1 among the task's value set on held-out windows. E112's label accuracy is the argmax between
  its two labels (`case_label/run.py:125` `label_diff`); the m-valued argmax is the same rule. Full-vocab top-1 printed
  beside.
- **P2.** E112's g is a threshold and sign on a scalar score, fitted on the fit windows. There is no filed m-valued g.
  The closest existing code is E70's table from readout class to answer, `removal.lookup_accuracy` (`removal.py:132`,
  most common state answer per class on the fit windows, read on held-out windows, an unseen class wrong). For m = 2
  it is E112's g with the threshold fixed at the probes' crossing, not fitted: not exact. **Q2.**
- **P3.** The removal basis is the orthonormalised span of the m probe weights at l*, at the trigger only (for one-hot
  labels the ridge weights sum to zero, so the rank is m−1; for m = 2 it is E112's unit direction exactly). The
  do-nothing is `removal.random_basis` of the same rank and seed 5 in the same place: "in place of the probe's",
  restated for a basis. **Q3** on the rank.

## Predictions, W₀ and the do-nothing state

Filed per row in `experiments/definition_table/results/predictions.json` before the job, as the table's "expected"
columns with these controls. W₀: the E112 reproduction within 1e-3; the type-level readouts and their controls both
≈ 1 (beside); the copying and guess readouts under 0.90 at blocks 0-2 (lean, from E90's 0.876 at block 2) with controls
≈ 1/m; W₀'s P1 ≈ its own prior except on rank-1, where it passes by construction. Do-nothing state (δW = 0): P1 = W₀'s,
gain 0 → P3 nan, verdict "not through C" (E112's "no gain" edge); it cannot pass. Do-nothing direction: rank-1 ≤ 0.10
on every state (E112 read ≤ 0.0006); rank m−1 leans over 0.10 on the copy states (E90's rank-31 draws took 0.30-0.42 of
ident32's gain) and near it on the guess states (band 0.163). Edge cases run before filing (appendix §5): a class
with no Pile position, a state with one answer, a zero gain, ties at a bar, an empty held-out set, the do-nothing state.

## Run order, code, time

One job, `cluster/definition_table.sbatch`, stages in one script with E112's 25-minute check and exit-3 resubmit.

0. **W₀ alone.** (a) The E112 reproduction, stop on a miss. (b) Per C: Pile positions, probes at blocks 0-3, controls,
   accuracy and selectivity on the task windows, l* and the locator's site. Type-level readouts printed beside, no l*.
1. **P1 per state** (needs no readout): 26 states, appendix §1.
2. **Per C with an l*:** do-nothing basis on every state first (stop that C on a miss, as E112), then P2, P3 and the
   twin's P3; verdicts.

Code from `src/commute`: `removal.fit_ridge/remove/random_basis/lookup_accuracy`, `case_label.first_passing_block/
removed_share/verdict/do_nothing_ok/meets`, `readout.next_token_windows/resid`, `heldout_keys` windows,
`copying.cue_windows`'s layout, `locator_table.sites.TASKS/Files` (loaded with `commute.jobs.load_experiment`) for the
state files and their sha256. E112's `probes` loop (`case_label/run.py:192`) moves to `src/commute/case_label.py` as a
function over a value set; E112's run keeps calling it. Nothing is copied from a finished experiment. Time: model load
about 1 min; stage 0 about 30 s per E112's stage 0 (`analysis_e112-case-label.md:406`, three jobs in 10 min) plus about
1 min per C for the Pile scan (E90's three readouts and bases took 5 min, `experiments/SAVED.md:419`); stages 1-2 about
4 min for 26 states (E90 read 16 states in 2.4 min, `SAVED.md:422`). About 12-15 min in all.

## Questions settled by the lead (2026-11-06; Fable's draft recommendations, one overridden)

1. **Type-level C (ident32, copy-24, rand32, cipher-24, lookup): measured, not declared n/a.** Overrides the draft.
   P0 is run as filed: the probes are fitted on Pile positions of the key types and scored on the task windows
   (positions the fit never saw); the Hewitt-Liang control gives each type one seeded random value. A relabelling of
   types is learnable as well as the types themselves, so selectivity is expected near 0 and P0 to fail its 0.30 bar,
   giving "unreadable" by the filed rule with the number shown. The rows' expected readings above stand; the reason
   becomes a measured one.
2. **P2's g for m values: `removal.lookup_accuracy`'s table** (most common state answer per readout class on the fit
   windows, read on held-out windows), filed before the run as the measurement of Def 2's g under row 87, which bounds
   g by carrying, not form. Listed under Departures. Not a new clause: no bar or verdict branch is added.
3. **Do-nothing: rank-matched random basis** (`random_basis(d, m-1, [5])`), the rank-1 share printed beside.
4. **Text ciphers' site:** W₀'s guess read at positions whose target is a table value on E53's 64 held rows.
5. **Site: the locator's block when it passes P0**, E112's l* printed beside.
6. **The writer's questions (lead, 2026-11-06, after the code writer found the draft could not be built as written;
   filed in `experiments/definition_table/results/predictions.json` "lead_decisions_2026-11-06").** (a) P2's g: the
   identity on classes, no fitting, where the answer is the readout's own value (ident32, copy-24, E62 copy, rank-1);
   the table of item 2 only where it is not (elicit, cipher-24, text ciphers, rand32, lookup), an unseen class wrong. A
   table fitted on the train windows could not carry to E62's never-trained Bs or elicit's new values, so item 2 alone
   would have given P2 ≈ 0 there by construction; on the table rows P2 is predicted low, which is P1's failure seen again,
   not a new rule. At m = 2 the identity is a threshold at the probes' crossing with fixed sign. (b) P3's margin: the
   target's log-odds within the task's value set, log p(y|S) − log(1 − p(y|S)), E112's margin at m = 2;
   full-vocabulary log-odds beside; a twin uses its own value set. (c) The exclusion and the Hewitt-Liang control are
   keyed on the slot where the key sits: t for the key tables and token identity there (the appendix's "t−1" was E62's
   and E112's layout), t−1 for E62 and E112, the token at t for the text ciphers. (d) The writer's smaller choices
   (fit and held windows, the state's answer as its top-1 within S, the cap per value and key-slot type) accepted.

## Departures from the literature

- Jain et al. 2024 train the readout on PCFG pretraining data with a PCFG probe; here a ridge probe on Pile residuals
  at one position (E112's form).
- Jain put no bound on g; row 87's held-out carrying is the user's dated departure, read as P1.
- For a C with more than two values, g is measured as a table from readout class to answer (`removal.lookup_accuracy`), fitted on the fit windows; Jain define g only as a function, and E112 measured it as a threshold for two values.
- Direction removal is our addition (row 81's sentence); Jain's causal evidence is pruning and reverse fine-tuning.
- The "pruning a few neurons" form is reported as the locator's part-level revert (E100 own, E97), the user's
  instruction, not Jain's App. D Alg. 3; E112 ran Alg. 3 and E116 does not.
- The Hewitt-Liang control is one random value per token type; Hewitt and Liang's control tasks are per word type too,
  but their probes are classifiers, not ridge regressions.
- Jain's possession is "for all x"; here bars of 0.90 and 0.30 filed by E112.

## Not run

- Any fine-tune, new task pair, new windows, or the subject-number pair.
- Pruning (Alg. 3), reverse fine-tuning and INLP on these states; E113's patching form.
- ident32b/rand32b, kv_task's E10 originals, seeds 3-7 of the 24-key maps, 32-key elicit/cipher states.
- A held-out target for the text ciphers or the lookup (none is defined; no rule is added).
- Readouts at the word position (E112 Amendment 1's form) or at more than one position.
- Step 2 (the Qwen anchors): its own plan after this table is read.
