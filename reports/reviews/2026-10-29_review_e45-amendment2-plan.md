# Review: E45 Amendment 2 draft plan and predictions (2026-10-29)

*Independent review by a fresh agent that wrote neither file. Scope: `reports/plans/2026-10-29_plan_e45-amendment2.md`
and `experiments/qwen_updates/results/predictions_amend2.json` (draft), read against the parent plan, the earlier
predictions files, `RUN_BRIEF.md`, Fable's three reviews, the saved results JSONs, `run.py`/`job.py`/`qwen_updates.py`,
and `.claude/CLAUDE.md`. The user approved the three changes in principle; this checks whether the draft implements
them soundly.*

**In plain words.** The draft carries out the three changes the user approved, and most of its rules are complete,
but one rule as written picks the wrong saved state for anchor 2's second seed: that seed's step 300 sits closer to
the target than its step 100, so the general "nearest state" rule contradicts the approved "first learned state"
rule. Two smaller gaps would let a learned, in-band state be thrown away, and let the new format rule reject a reply
that the old rule accepted, against the draft's own claim. All of this can be fixed in the text before filing, for
$0, and none of it changes what any update trains on.

## Findings, most severe first

### 1. High: anchor 2 seed 1 is matched at step 300 under the draft's general rule, not step 100

`decisive_point.rule` reads *every* run, "a2long_s0/_s1 themselves" included, at the saved state with the smallest
|log(KL_maths / target)|. a2long has a saved state at step 300. From `E_a2long_s1.json`: step 100 maths KL 0.60740
(|log| 0.0176), step 300 maths KL 0.62514 (|log(0.62514/0.61818)| = 0.0112). Step 300 is nearer, and it is learned
(invented 1.0, known 0.99 against 0.97). So the general rule chooses step 300 for seed 1, while `decisive_point.anchor2`,
`expected_matched` and the decision row (decisions.md row 63: "anchor 2 is read at its first learned state (step
100)") all say step 100. Seed 0 is unaffected (step 300 at |log| 0.066). The two fields disagree and the code writer
will implement one of them; a Job 2 retrain to step 300 would also cost more.

**Fix.** Say in `rule` that anchor 2 is read at the state `anchor2` defines, and that the nearest-state rule applies
to the other runs. Also restrict every run's pool to saved steps ≤ 100 (the other arms have no later states, and the
target is a step-100 quantity). Add a test: a2long_s1's real states must give step 100.

### 2. Medium: "nearest, then must be learned" can discard a learned, in-band state

The rule takes the nearest state first and then requires it to be learned; otherwise the seed is missing (anchor 1:
a card) or the arm is not read. If, for example, anchor 1's maths KL reaches 0.62 at step 4 (a format shift on the
maths rows can come fast, as anchor 2's did) before its gap closes 0.25, while step 100 sits at 0.85 and is learned,
the rule throws anchor 1 away although a learned state lies in the band. The parent rule had a fallback ("the smallest
grid value above 0.1 where they all are"); this one does not. Anchor 2 uses the opposite order (learned first).

**Fix.** One order for all runs: pool = saved states (steps ≤ 100) that are learned and have positive finite maths
KL; choose the nearest to the target; matched if within the band. Report the nearest unlearned state beside it. For
anchor 2 this reproduces step 100 on both seeds (steps 2-12 are not learned) and makes finding 1's special case
unnecessary. Note that `learned` for anchor 2 needs a generation reading, which exists only at E's chosen steps and
the last steps; a state without one counts as not learned (say so).

### 3. Medium: the new format rule is not a superset of the filed one, contrary to the edge-case list

The filed rule is `FORMAT_RE.fullmatch(response.strip())`, which forgives leading whitespace. The new rule only
`rstrip`s each line, so a reply starting with spaces before "Given:" passed the filed rule and misses the new one. The
JSON's edge cases claim both "leading spaces: miss" and "every response the filed regex accepts is accepted
(superset)"; those two contradict for leading whitespace on the first line. No saved P reply starts with whitespace
(checked: 0 of 512 in all five files), so the 474/0 counts are unaffected, but `format_unprompted` on the trained
anchor-1 states and anchor 1's E readings are not yet seen.

**Fix.** Apply `.strip()` to the joined text as well (or `lstrip` the whole response first), so that the new rule is
a true superset, and keep "leading spaces before '- ' on a step line: miss". The count stays 474/512 (no reply has
leading whitespace). I re-ran the draft's jq expression on `large/P/*.jsonl`: format 474/512, the filed-style count
17/512, none/fewshot/recheck/givens 0/512, as the plan says.

### 4. Medium: the reproduce check must name its measurement, and it needs new code

`early.reproduce_step2` (0.005660122 and 0.006690659, 1% tolerance) are the 128-row `kl_maths_all` values of the
"step 2 KL on the maths rows" check in `T_own_s0.json`/`T_null_s0.json`, not the 16-row tracker `rec["kl"]`.
`early()` today reads a key `reproduce` with a per-step, per-axis structure against the tracker; the new key has a
different shape. Appendix C lists the change in one line.

**Fix.** Write in the JSON: "compared with the step-2 reading on all 128 maths rows (`kl_maths_all`), not the
tracker". In the code, one branch inside the existing `t == 2` block. The fresh T start deletes the old
`T_own_s0.json`; the references are now only in this file, which is fine because they are copied at full precision.

### 5. Low-medium: the extra save is triggered by the 16-row tracker and on "either axis"

Stage T's tracker reads maths KL on `track_rows = 16` rows; the target is a 128-row value. The extra save brackets the
16-row crossing, and E then picks among saved states on 128 rows. Gold's tracker and E readings at step 2 agree to
0.3%, so this is likely fine, but the plan does not state it. "On either axis" also saves at the own-rows crossing of
0.618 (gold: own KL runs ~1.2-1.6 x maths), adding states that matching never uses (2.8 GB each).

**Fix.** Save at the target on the maths axis only, and say the bracket comes from the 16-row tracker. Optionally add
a check: the chosen state's 128-row KL is within the band (it is the matching rule anyway).

### 6. Low: anchor 2's positive control is circular; only gold tests the matching

The target is the mean of the two anchor-2 step-100 values, so both seeds match by construction. The plan lists them
as positive controls; only gold's step 100 (0.697, |log| 0.119) is an independent check. The factor 1.5 was also
chosen with gold's 0.697 already known (a tighter tolerance of 1.1 would exclude gold). Both are within what the user
approved, but the plan should say so in the "does not" column.

### 7. Low: Fable's "hard problems" concern is moot for training, but worth one line

I ran the check the plan lists as not run. Of the 38 tolerant misses, W₀ without the prompt got 26 right (0.68); of the
474 follows, 401 (0.85). So the misses do lean to harder problems, by about two standard errors. This does not bias
anchor 1's training data or own rows: S a1 keeps every finished sample (`keep_finished`) and `a1_rows.json` keeps
every greedy answer, neither filtered by format. Replace the "Not run" item with this result and that sentence. The
other open item (overlap of E40's 128 maths rows with the 2,000 own-sample problems) is ruled out by construction:
`maths_rows.json` comes from E40's 256 held-out problems, and `job.pool` samples only `train_idx`, which excludes them
(`job.py` lines 131-133). Say so in the plan.

### 8. Low: the step-2 check's "blind to" line is too pessimistic, and the anchor-1 range is not a coded stop

KL grows about as the step squared, so a step twice too large gives about 4 x KL: own samples 0.023, gold 0.015, both
over 0.0107. The check is blind only to steps up to about 1.4 x too large for own samples and the null, 1.7 x for
gold, and about 3.5 x for anchor 2. State that. Separately, `expected.step2_kl_maths.a1` ("if outside 0.0037-0.0067,
the recipe is looked at before more arms run") is not a code stop; put it in `RUN_BRIEF_amend2.md` as a manual stop
between T a1 s0 and T a1 s1.

### 9. Low: pre-mortem for the amendment itself is thin

Gate 1 formally applies only if the run is gated; the plan asks the user to confirm it is not (each job ≤ 30 min, $0,
sum ~4 h). Under the headings asked: recipe, data, precision, memory, downloads (none, cluster) are unchanged and need
no new guard. Missing lines: **code paths never run on a GPU**: P's `a1_rows.json` write (P stopped before it), the
target-save path in `stage_T`, the `reproduce_step2` branch, and E's generation at the new chosen steps; the first is
guarded only by P finishing, the others by the T own job that runs first. **Saves**: about two extra states per run
(target brackets), ~110 GB more beside 98 GB already in `commute-data/results-large/qwen_updates`, on a shared volume
at 90% (3.9 TB free): fine, but say it. **Loss and resume**: unchanged (P, S, E resume; T for 100 steps never
pauses). **Time**: E gains a generation per new chosen step (~2 min each), within its budget and resubmit. Add these as
one short table.

## What holds

- The three changes match the user's approval (decisions.md row 63) and Fable's recommended options (anchor-2 review
  option (d) with factor 1.5; anchor-1 option (A); step-2 option (B)).
- The format count reproduces exactly: 474/512 with the prompt, 0/512 without, 17/512 under the filed regex, 0/512 on
  the other three prompts (jq, rerun here). The P re-read as the first GPU check, stopping unless 474/0, is sound.
- Matching numbers check out: target 0.6181848, band [0.4121, 0.9273]; gold step 100 at |log| 0.119, matched and
  learned (drop 1.98 nats); anchor 2 step 100 learned on both seeds (invented 0.91 and 0.905, known 1.0 and 0.99).
  The empty-pool, tie, band-edge, KL-0, NaN and overshoot edge cases are all specified, and the tie rule matches
  `nearest()` in the code.
- For own samples and the null, the own rows are the maths rows (`job.own_rows` returns None), so matching on maths KL
  is also matching on their own rows.
- Step-2 limit: 0.1 x 0.10679 = 0.010679; own2's scaled 0.042716 (4 x) leaves room for own at 2e-5 (~0.023
  expected). The positive control (one step at peak, 0.107) is ten times the limit. The reproduce check (1%) against a
  measured repeat noise of 2e-5 is a real control-first test.
- Control first: CPU tests, then P's re-read, then T own reproducing its stopped step 2, before any new arm.
- Nothing any update trains on changes; the only reading made after seeing data beyond the approval is the tolerance's
  relation to gold (finding 6), which is disclosed rather than hidden.
- Cost and time: ~13-17 debug jobs, ~4 h, $0; Job 2 stays gated with its gates, plus the large-update pre-mortem item.

## Not run

- The Python `follows_format` change and the `match` function (no code exists yet).
- No check of how fast anchor 1's maths KL rises; whether it reaches 0.412 by step 100 is unknown, and a miss on both
  seeds ends in a card.
- No check of the 16-row tracker against the 128-row reading beyond gold's step 2.
