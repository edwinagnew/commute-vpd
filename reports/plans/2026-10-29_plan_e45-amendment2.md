# E45 Amendment 2: anchor 2 read where it has learned, anchor 1 as the format template, one warm-up check (2026-10-29)

*Draft for the qwen-decomp lead, to show the user before anything is filed or run. Plan:
`reports/plans/2026-10-28_plan_qwen-q2.md`. This is the plan's last allowed amendment. The user took Fable's three
recommendations on 2026-10-29 (`decisions.md`, last row). Draft predictions:
`experiments/qwen_updates/results/predictions_amend2.json`.*

**In plain words.** Three of E45's filed rules turned out not to fit the updates as they actually behave, and the
user has agreed to change all three in one amendment. The invented-facts anchor is now read where it has learned its
facts, and every other update is read at the point where it has moved the model's maths answers by the same amount.
The prompt-copying anchor becomes "answer in a fixed template", scored by a rule that forgives trailing spaces. The
first-step check becomes a test for a missing warm-up, the failure it was written to catch. Nothing any update trains
on changes. The remaining training work is about four hours of free cluster jobs.

## Design check (`/design`)

- *Goal.* D3 of the re-plan: can VPD's readings separate "already there" from "new content"? Without this amendment
  neither anchor can be read, so no decisive row can be validated.
- *Nearest tried.* Amendment 1 (`a2long`, 300 steps) made anchor 2 learn. Plan §3 already names the format fallback.
  Nothing new is tried. This amendment only changes how the runs are read.
- *Patch or step.* A patch: three reading and check rules, forced by data the filed rules did not foresee.
- *Decisions.* It relies on rows 51 and 55-57 and departs from none. It changes the plan's §5 decisive point, the §3
  format regex and the "early" step-2 limit. The user approved this on 2026-10-29.
- *What changes next.* If anchor 1 is matched and learned on both seeds, Job 2 can validate rows. If it is not, the
  rows cannot be validated and a card goes to the user (appendix C).
- *Control first.* Stage P re-reads its saved answers and must reproduce 474/512 and 0/512 exactly before anything
  else runs. Then the first T job (own samples) must reproduce its stopped step-2 KL, 0.00566, within 1%.
- *Numbers.* (1) Format follow with and without the prompt: expected 0.926 and 0. (2) Step-2 maths KL per arm:
  expected 0.0057 for own samples and 0.0067 for the null; anchor 1 between gold's 0.0037 and the null's. (3) The
  matched maths KL per run: target 0.618. Gold is expected at 0.697 at step 100, anchor 2 at 0.629 and 0.607. At W₀
  all three readings are 0. The positive control for the warm-up check is the one-step-at-peak control: it reads
  0.107, ten times the limit.
- *Class.* This is a confirmed plan's amendment, which the user decided in chat. It needs the user to see this file
  before anything is filed (`.claude/CLAUDE.md`: a plan restates the request in the Terms' words, and that is what the
  user confirms).

## The three changes, in the Terms' words

All five updates are full-weight SFT δW of the 196 decomposed matrices, made with one recipe. KL to start means the
mean per-token KL(W₀ ‖ W₀ + δW).

1. **Anchor 2's decisive state.** Anchor 2 is the invented-facts δW, run `a2long`. It is read at its first saved state
   where it has learned the facts. That is step 100 on both seeds: held-out recall 0.91 and 0.905. One rule serves
   every run, anchor 2 included. Take the saved states up to step 100 that have learned. Read the one whose KL to
   start on E40's 128 shared maths rows is nearest 0.618, the two anchor-2 seeds' mean. A run counts as matched within
   a factor of 1.5, and its matched KL is stated with every reading. The step-100 limit matters: without it, anchor 2's
   seed 1 would be read at step 300, which sits slightly nearer the target. The filed point,
   own-rows KL 0.1, is still reported. For anchor 2 it falls at step 2, which recalls 1% of the facts.
2. **Anchor 1.** Anchor 1 is SFT on the model's own sampled answers to the answer-template prompt, trained without
   the prompt. "Follows the template" now also counts an answer that matches once whitespace at the start, trailing
   spaces and blank lines are removed. Anything the filed rule accepted is still accepted. The filed
   regex's count, 17/512, is reported beside the new one, 474/512. Every anchor-1 reading also states that the prompt
   lowers W₀'s accuracy from 0.834 to 0.680 and shortens answers from 190 to 130 tokens.
3. **Step-2 check, for every arm.** At the first step with a nonzero rate, KL to start on the maths rows must be at
   most 0.0107, which is a tenth of the one-step-at-peak control (0.107). The per-matrix |δW| band stays as it was.

## What each change does and does not let us conclude

| change | lets us conclude | does not |
|---|---|---|
| 1 | Anchor 2 is read where its content is present, and every arm is read at equal movement on one shared set of rows | The states are not matched on each arm's own rows: anchor 2 sits at own-KL 4.5 and gold at 0.86. Most of anchor 2's movement on the maths rows is a format shift to one-line answers, not its content (`check_format.json`: with a 4-shot prompt its maths accuracy at step 100 is within 0.016 of W₀'s). Anchor 2's δW is also much larger, so a row that differs by update size alone would look like a separation. That is why the size of δW goes beside every reading. Only gold's step 100 tests the matching independently. The two anchor-2 seeds match by construction, since the target is their mean. The factor 1.5 was chosen with gold's 0.697 already known: a factor of 1.1 would exclude gold |
| 2 | Anchor 1 fixes behaviour novelty for a surface format that the model already produces on request | It says nothing about a better way to solve problems. Anchor 1 lowers maths accuracy, the opposite sign to the GSM8K arms. Both anchors share a format component on the maths rows. The rule was loosened after the data were seen, so it is a filed rule change, not a re-read |
| 3 | Own samples, the null and anchor 1 can run under the one recipe, and no reading changes | KL grows about as the step squared. The check therefore still fires on a step about 1.4 times too large for own samples and the null, 1.7 times for gold and 3.5 times for anchor 2. Below that, only the |δW| band guards |

The decomposition-row predictions in `predictions.json` stand as filed. They were written for an accuracy-raising
anchor 1 read at own-KL 0.1. They are now scored on a format anchor at maths-KL 0.62. A miss on anchor 1's
thresholds (behaviour kept ≥ 0.8, re-gating share ≥ 0.5) may come from that change, and the write-up has to say so.

## Run order, cost, gate

All of it runs on the MATS cluster under `debug`, one L40 per job, for $0. Items marked † are new.

| # | job | why | ~time |
|---|---|---|---|
| 0 | † CPU via `cpu-run`: the new tests, plus the format rule on the saved P answers (474 and 0) | the format rule must reproduce Fable's count before filing | minutes |
| 1 | **P resubmitted** | check: re-read under the new rule, stop unless 474/0. Then `chosen = format` and the format prompt's greedy answers to test problems 256-511, which become anchor 1's own rows (P stopped before writing them) | ~6 min |
| 2 | T own, then E own | † check: step-2 KL on all 128 maths rows reproduces 0.00566 ±1%. This is the control: nothing changed since the stop | 15 + 20 min |
| 3 | T null, then E null | † reproduces 0.00669 ±1% | 15 + 20 min |
| 4 | S a1 (2,000 problems × 8 samples, format prompt), resubmitted until done | | 3 jobs, ~65 min |
| 5 | T/E a1 s0, then T/E a1 s1 | step-2 KL expected between 0.0037 and 0.0067. Outside that range is a manual stop in the run brief before a1 s1 | 2 × 35 min |
| 6 | score | `amend2_decisive` in `score.json` | 1 min |

Already done and not rerun: the control, gold, `a2long` on both seeds, S own, and the format check. `a2` seed 1 is not
run, because Amendment 1 replaced it. The approved branch "own samples not learned at step 100 → own2 at 2e-5" stays,
and `own2` reads this file. In total this is about 12-14 jobs, ~4 h, $0, and ~110 GB of δW on `commute-data`.

**Job 2 (E44's table) changes its state list but not its gates.** The decisive state per run becomes the matched
state: step 100 for gold and for both anchor-2 seeds. Anchor 2's 0.03 states are dropped because they are not learned.
The pod's retrain stage must reproduce `a2long`'s schedule to step 100, not `a2`'s. Cost stays about $15.50 on one
H100, so Job 2 stays gated with its gates unchanged. One new pre-mortem item for it: E44's pass rule was set on small
updates, and anchor 2's state is a large one.

**Is this amendment gated?** No, if the plan's reading of Job 1 holds. Each job is a separate ≤ 30-min $0 unit whose
work is saved, it needs no budget, and the approved plan kept gates for Job 2 alone. The sum of job time does pass 3 h
(~4 h), so the user should confirm that reading.

## Pre-mortem for this amendment

Gate 1 formally applies only if the run is gated. The lines are filled here anyway.

| heading | what could go wrong | guard |
|---|---|---|
| recipe, data, precision, memory, downloads | nothing changes: no arm's data or recipe is touched, and nothing is downloaded on the cluster | none needed |
| code never run on a GPU | P's `a1_rows.json` write: P stopped before it | P must finish with exit 0 |
| code never run on a GPU | the target save in `stage_T`, the `reproduce_step2` branch, E's generation at the new chosen steps | T own and E own run first; a miss stops before any new arm |
| saves | about two extra δW states per run, ~110 GB in all, beside the 98 GB already in `commute-data`; the shared volume is at 90%, with 3.9 TB free | enough room; size logged after each T |
| loss and resume | unchanged: P, S and E resume after exit 3; a 100-step T never pauses | as RUN_BRIEF |
| time | E adds ~2 min of generation per new chosen step | within E's 25-min budget and resubmit |
| controls | the matching has only one independent positive control, gold | stated in the reading; no other exists without a new run |
| predictions | a degenerate matching rule | appendix B's cases, tested before filing, including a2long_s1's real states → step 100 |

## Not run

- The Python `follows_format` change has not run. The 474/0 counts come from a jq expression of the same rule,
  appendix A.
- The independent review checked whether the 38 tolerant misses cluster on hard problems. They lean that way: W₀ got
  0.68 of them right against 0.85 of the 474, about two standard errors apart. This does not bias anchor 1. S a1 keeps
  every finished sample, and its own rows keep every greedy answer; neither is filtered by format.
- E40's 128 maths rows cannot overlap the 2,000 own-sample problems. They come from E40's 256 held-out problems, and
  the sample pool excludes those by construction (`job.py`, `train_idx`).
- There are no anchor-1 numbers of any kind yet. Nobody knows whether anchor 1's maths KL reaches 0.412 by step 100.
  If it misses on both seeds, a card goes to the user.
- The 16-row tracker has been checked against the 128-row reading only for gold at step 2, where they agreed to 0.3%.

---

## Appendix A: the rules, exactly

- **Format follow.** A response follows the template if the filed rule accepts it, or if the normalised text matches.
  The filed rule is `FORMAT_RE.fullmatch(response.strip())`. To normalise, apply `lstrip()` to the whole response,
  split it on `\n`, apply `rstrip()` to every line, drop the lines that are now empty and join the rest with `\n`.
  Then apply `FORMAT_RE.fullmatch`; the regex itself is unchanged. The "or" makes the new rule a superset of the
  filed one. Without it, a line of only spaces, which the filed regex's `[^\n]+` accepts, would be emptied. The jq
  version, run on `large/P/*.jsonl`, gave format 474/512, the filed rule 17/512, and the union 474/512, so the 17 lie
  inside the 474. None, fewshot, recheck and givens each gave 0/512. The normalising half in jq: `.response |
  sub("^\\s+";"") | split("\n") | map(sub("\\s+$";"")) | map(select(length>0)) | join("\n") | test("\\AGiven:
  [^\\n]+\\nSteps:\\n(?:- [^\\n]+\\n)+The answer is [^\\n]+\\z")`. Absolute thresholds, unchanged: ≥ 0.9 with the prompt, ≤ 0.05 without.
  Blind to: whether the steps are right. The template is surface form only. Noise floor: none for the decision, because
  greedy decoding re-read from disk gives the same answer every time. The binomial s.e. of 0.926 on 512 problems is
  0.012, so the margin over 0.9 is about 2 s.e.
- **Decisive point.** The target, 0.6181848, is the mean of the two `a2long` seeds' step-100 maths KL. The pool is the
  run's saved states up to step 100 that are learned (`learned`, unchanged) and have a KL above 0. Anchor 2's facts
  are read only at states with a generation reading; any other state counts as not learned. From the pool, take the
  state with the least |log(KL_maths / target)|; a tie goes to the later step. The run is matched if that distance is
  at most log 1.5, inclusive after rounding to 9 places. That means KL between 0.412 and 0.927.
  The nearest unlearned state is reported beside it. The rule is absolute.

  The extra save fires on the 16-row maths tracker, on the maths axis only. Matching then uses only the 128-row
  values that E reads.

  Blind to: movement on the arm's own rows, and the difference between format movement and content movement.

  Positive control: gold's step 100 is the only independent one. It must match (distance 0.12) and be learned. The
  anchor-2 seeds match by construction. The step-100 state of `a2_s0` is not in the target and reads 0.609. It checks
  only that the target can be reproduced, since it uses the same data with another schedule.

  Noise floor: anchor 2's seed gap is 0.035 in log, against a tolerance of 0.405.
- **Step-2 check.** KL ≤ 0.010679 × (peak / 1e-5)², so 0.0427 for own2. Absolute. Blind to a step under about 1.4 × too large (own samples, null), 1.7 × (gold), 3.5 × (anchor 2).
  Positive control: the stage-control step at peak, 0.107. Noise floor: about 2e-5 relative (the a2 against a2long
  repeat). The reproduction check compares the step-2 reading on all 128 maths rows (`kl_maths_all`), not the 16-row
  tracker.

## Appendix B: edge cases (all in the draft JSON; tests before filing)

| case | outcome |
|---|---|
| Empty pool (no learned state up to step 100 with KL > 0, or E not run) | unmatched |
| Tie in log distance | later step |
| Exactly at log 1.5 | matched |
| A state with KL 0 or NaN | never chosen |
| Anchor-2 seed never learned | seed missing, no row validated |
| Arm's KL never reaches 0.412 | unmatched: GSM8K arms and the null not read on the decisive rows; an anchor-1 seed missing → card |
| Overshoot with both brackets outside the band | unmatched |
| Nearer unlearned state beside a learned in-band one | the learned one is read |
| A state at step 300 | never in the pool |
| a2long_s1's real states | step 100 (step 300 would be nearer without the limit) |
| Format: empty or whitespace-only response | miss |
| Format: CRLF line endings | pass |
| Format: whitespace before "Given:" | pass |
| Format: spaces before "- " on a step line | miss |
| Format: text after the last line | miss |
| Format: anything the filed regex passed | pass |
| Step-2 limit: exactly the limit | pass |
| Step-2 limit: 0.010679006 | miss |
| Step-2 limit: NaN | miss |

## Appendix C: code changes (~70 lines plus ~45 of tests; review by a fresh agent before job 1)

| file | change | ~lines |
|---|---|---|
| `src/commute/qwen_anchors.py` | `follows_format`: normalise as above; regex unchanged | 3 |
| `src/commute/qwen_updates.py` | new pure `match(pool: dict[int, float], target, tol_ratio) -> {step, kl, log_dist, matched}`, the pool already limited to learned states up to step 100 | 15 |
| `experiments/qwen_updates/job.py` | `ARMS` own, own2, null, a1 → `pred="predictions_amend2.json"` | 4 |
| `experiments/qwen_updates/run.py` | `main`: P loads this file. `stage_P`: recompute `format_follow` for format and none from `large/P/*.jsonl`, keep `with_filed_regex`, check 474/0. `stage_T`: a maths-axis-only save at `target_kl_maths` when the file names it (a small change to `qwen_train.train`, or a second grid passed for the maths axis). `early`: the `reproduce_step2` branch in the `t == 2` block, reading `kl_maths_all`. `score`: `amend2_decisive` block | 35 |
| `experiments/qwen_updates/RUN_BRIEF_amend2.md` | the table above, plus the manual stop: anchor 1 s0's step-2 KL outside 0.0037-0.0067 → report before T a1 s1 | 25 |
| `tests/test_qwen_updates.py` | `test_amend2_*`: appendix B's cases, the values copied from the earlier files | 45 |

`step2_kl_max` needs no code change: run.py already reads the number from the predictions file. `qwen_anchors` and
`qwen_updates` are imported only by `experiments/qwen_updates/` (`run.py`, `job.py`, `check_format.py`) and its tests,
so no frozen experiment can break. If `qwen_train.train` takes a per-axis grid for the maths-only save, its only callers are `experiments/qwen_updates/run.py` and the checkpoint test; the finished runs (gold, a2, a2long) are not rerun. P's stored `format_follow` (17/512) is recomputed,
not trusted. The fix stands as a dated record because `P.json` keeps the filed count beside the new one.
