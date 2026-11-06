# Plan: E115, does the carrying set read a genuinely new fact when it is taught in the known arm's style?

*2026-11-06, Fable's advice for baseline-lead, written as the plan for ask 24 option (c) (`decisions.md` line 170).
Nothing run; the only computation is `jq` over stored sets. Apparatus, sources and the file list are in
`reports/plans/2026-11-06_plan_e115-invented-facts_appendix.md` (sections A-E). E115 is unused in `experiments/TRIED.md`.
Follows E107, E109 and the table (`reports/analyses/2026-11-06_analysis_{e107-label-vs-style,e109-neurons-larger-size,neurons-vs-vpd-table}.md`).*

## 1. The request in the Terms' words

Fine-tune W0 (Qwen3-1.7B, full weights; no decomposition is read, this is the non-VPD baseline) with E103 Amendment 1's
recipe on the known arm's 100 real-author rows, their five wordings and their writer's sentences, with the quoted book
title in the question replaced by an invented title and the stated author by an invented author's name, both drawn
from TOFU's fictional authors and never paired as TOFU pairs them. Every question then asks for a fact W0 cannot hold,
in the exact words and answer form of the known arm. Read each update's carrying set (the 1,024 MLP neurons with the
largest first-order share of the update's gain on a probe) at the matched step, |δW| 0.49, and at the end of the known
arm's schedule, step 335, and ask whether it sits with the known arm's set on the same 100 rows (E107's AUTH arms),
which reads as style and data source, or as far from it as the fictional-author arm's, which reads as novelty. It is
the comparison E107 and E109 ran for the wrong-answer arm, with "contradicted" replaced by "new".

## 2. Design check (`/design`)

- **Goal.** `GOALS.md`'s fallback question: does post-training teach new things or strengthen existing ones, and can
  any reading tell them apart? The table's weak link is the anchors: the "new" arm differs from the "known" arm in
  topic, entity type and answer style at once. This builds a "new" arm that differs from the known arm in nothing but
  the facts.
- **Nearest tried.** E107 (`TRIED.md` line 217): a contradicted fact in the known arm's words sat with the known arm
  at matched size, "topic or style" on both probes. E109 (line 219): the same at the learned size had no power. E106's
  option (6), an invented name in a known question, was named and not run. New here: whether a fact that is new
  rather than contradicted moves the set, every other difference removed.
- **Patch or step.** A step: the anchor the table's "Where neither reads" names.
- **Decisions.** Row 152: the matched step is a size control only; a "new" claim rests on the end state, size stated.
  Row 121: a new circuit built from old parts reads as reuse, stated in every reading. No new rule, clause or band:
  the verdict is E109's `score.wrong_rule` with other state names (appendix C). Not a reuse verdict: rows 81 and 87
  (Jain et al. 2024 Defs 1-2 with held-out carrying) do not apply and are not touched. Departure from E106's option
  (6): the title changes too, so the fact is new, not a new name on a known question; option (6) runs beside it,
  reported.
- **What changes next.** Section 8.
- **Control first, alone.** Stage 0: C2n (a stored matched state re-read reproduces its gain and set; fails if the
  read code or a state file changed), W0's true-sentence level, then the novelty check: W0's log P on each new
  sentence below the true one in at least 0.75 of rows, and W0's greedy answers to the new questions recalling the
  invented author in at most 0.2 of words. It fails if TOFU's names or titles reached W0 in pretraining.
- **Class.** The user's ask 24 (c), ordered 2026-11-06. The user's one decision is section 3's option.

## 3. Options, and Fable's recommendation

- **(1) Substitution, recommended.** Title and author both invented on the 100 author rows: $0, no clerks, no judge;
  the arm differs from E107's AUTH arms by two string replacements per text (appendix B). Two seeds at matched size
  in one 12-min job, two seeds at the end of the known schedule in two more; option (6) beside, matched size only.
- **(2) Real long-tail facts W0 lacks** (my E108 advice, section 9): a PopQA-style file screened by E81's greedy
  reader into about 110 known and 110 unknown rows, sentences by Haiku clerks under
  `experiments/known_answer_pair/CLERK_BRIEF_AMEND1.md`. The gold test of the label, but 2-3 h of agent time, both
  halves new rows, and "unknown" for mixed reasons (rare, ambiguous, misspelt). Second, not first.
- **(3) Fictional rows rewritten in the known style** (E106's option 4): clerks, a judge, and TOFU's topic. Not now.

What would change it: the user wanting a real-world unknown rather than an invented one ((2), with (1) as its $0
companion); stage 0's novelty check failing (the invented names are in W0; (2) replaces (1)).

## 4. States

| state | what | file |
|---|---|---|
| AUTH42, AUTH142 | E107's known arm on rows 0-99, matched step 31, |δW| 0.487; sets exist on both probes | `label_vs_style/results/large/dW/AUTH{42,142}_matched.safetensors` (bf16) |
| FICT_matched, FICT_learned, two seeds each | E102 / E91's fictional states; sets exist | E109's `PATHS` |
| NEW42, NEW142 | **trained**: the substituted rows, E107's 64-step arm, matched step | fp32, for VPD |
| NAME42, NAME142 | **trained**, reported: true title, invented author (option 6), matched | bf16 |
| AUTH_end42/142 | **trained**: E107's AUTH data on E103 Amendment 1's 335-step schedule; the reference | fp32 |
| NEW_end42/142 | **trained**: the NEW data on the same schedule | fp32 |

Each trained arm differs from E107's AUTH arm in the data strings only (NEW, NAME) or in the step count only (the
end arms: 335 steps of 500 sequences is 10.7 epochs against E103's 5 of 1,085; warm-up and cosine unchanged, so the
end norm should land near the known arm's 8.3). AUTH_end exists for no other reason than that no 100-row known
state exists at the end; it is the reference, read before the candidate.

## 5. Stages and numbers

**Stage 0, job A, first and alone** (stop on any miss): C2n as E107 (gain 2.2331 within 1e-3, set overlap ≥ 0.95);
W0's content level on the 217 true sentences −2.01584 within 1e-5; the novelty check on the 100 rows for NEW and
for NAME (E107's `label_check`, min share 0.75; greedy recall of the invented author ≤ 0.2). Then W0's tokens on the
new sentences are saved for C1.

**Stage 1, job A:** NEW42, NEW142, NAME42, NAME142 through `known_facts.to_matched` as E107 (first batches differ,
step 1-2 checks, 44 GiB, C1 ≥ 1.2365 on the arm's own new sentences), read on `tofu` (scored) and `real`
(reported), E98's signed ranking at K 1,024; then the score.

**Stage 2, jobs B then C:** AUTH_end42/142 (B), then NEW_end42/142 (C, refuses until B wrote the gate):
`to_matched(..., keep="end")` as E109, the end δW saved first; on the 100 rows the new-against-true sentence read and
greedy recall of the invented and of the true author; reads on `tofu`, `real` and `T` under the ranking section 7
fixes. B also re-reads FICT_learned42/142 under that ranking and writes w, x and the power gate before any NEW_end
arm trains.

| number | at W0 / do-nothing | control or reference | expected |
|---|---|---|---|
| novelty check: rows with the new sentence below the true; mean gap | W0 reads; the true sentence is the reference | E107's wrong sentences 0.857, 0.65 nats a token | NEW 0.95-1.00, 1.5-4 nats; NAME 0.90-1.00 |
| W0 greedy recall of the invented author | 0.0-0.1 (E109's do-nothing read 0.10) | — | ≤ 0.1; a miss stops |
| C1 at the matched step, own new sentences | 0, fails | bar 1.2365; E107 WRONG 1.92, AUTH 1.37 | NEW 2-4, NAME 1.5-3 |
| w_known = AUTH seeds; x = max AUTH-in-FICT_matched, tofu [real] | NaN | stored (appendix A) | 0.879, 0.609 [0.799, 0.473], exact |
| power at matched size, w − x > 1 − w | unread | — | tofu 0.270 > 0.121, real 0.326 > 0.201: power |
| w_new = NEW seeds; cross = four NEW-in-AUTH, tofu | NaN, unread | AUTH 0.879; E107's wrong arm in AUTH 0.737-0.770 | 0.84-0.92; 0.72-0.84 (NAME 0.76-0.86) |
| learned gate, NEW_end: prefers new ≥ 0.75, recall of the invented author ≥ 0.585 | 0.0, 0.0-0.1: fails | AUTH_end's recall of the true author (reported) ≥ 0.9 | 0.9-1.0, 0.8-1.0 |
| end: w_known = AUTH_end seeds, tofu; cross = NEW_end-in-AUTH_end | NaN | E109's 217-row known end 0.652; its wrong end 0.658-0.687 | 0.55-0.80; 0.5-0.8 |

Reported, no stop (appendix C): matched step and norm (expected 26-34, within 0.02 of 0.4865), end norms (AUTH_end
and NEW_end within 10% of each other, 7-10). Blind to: rank inside a set, neurons past the cut, anything inside a
neuron (row 121); whether an invented title is a real work's (a few of TOFU's quoted titles are real books named as
influences; the pairing is new either way, and the novelty check reads the pairing); 100 rows against the known
arm's 217 (the AUTH reference is on the same 100, so it cancels in every overlap).

## 6. Rules to file (`experiments/invented_facts/results/predictions.json`; `score.py` imports E109's)

Per probe, E109's `score.wrong_rule` unchanged, with the states AUTH, FICT_matched, NEW at matched size (gates
`[True, True]`, as E109 scores its matched rule) and AUTH_end, FICT_learned, NEW_end at the end (gates from E109's
`learned`). In words: w = min(known seeds, new seeds), band = 1 − w, x = the highest known-in-fictional overlap;
**power** iff w_known − x > 1 − w_known, read before a candidate trains; **familiarity** iff max(cross) ≤ x + band
and ≤ w − band; **topic or style** iff min(cross) ≥ w − band and min(cross) − x > band; else **no call**; unread
with a set not of 1,024 or a value not finite. With the stored numbers, on `tofu` familiarity needs max(cross) ≤
0.730 and topic or style min(cross) ≥ 0.758 (if w_new ≥ 0.879); on `real` 0.598 and 0.674. Scored: `tofu` at both
sizes. Reported under the same rule: `real` at both sizes, NAME at matched size, `T` at the end.

Edge cases, run on CPU before filing with the stored sets (E109's list with E115's names, appendix C): the empty set
(the do-nothing state: unread, never a pass), a set of 1,000, three cross values, NaN seeds, one neuron either side
of each line, identical seeds. Stand-ins: E107's WRONG matched sets in NEW's place against AUTH read "no call" on
`tofu` (their cross 0.737-0.770 straddles 0.730 and 0.758); the fictional matched seeds in NEW's place read
"familiarity". The novelty check's edge cases are E107's `label_check`'s, the learned gate's E109's.

**Leans.** Matched, `tofu`: NEW **topic or style** (medium: at step 31 the update is mostly the sentence skill, as
E107's and E109's matched reads showed); NAME topic or style (medium). End, `tofu`, if power: **no call** likelier,
familiarity the lean (low: E109's wrong end sat level with its seeds, but a new title and name are what the fictional
arm has and the wrong arm lacked). Power at the end: no lean; E109's 217-row known end had none at FO-1024 and a
100-row arm's seed level is unmeasured.

## 7. How the plan reads under each outcome of E109 Amendment 1

E109 Amendment 1 (`reports/plans/2026-11-06_plan_e109-amend1-stable-ranking.md`) selects, per probe, a ranking (FO
or IG8) and a set size whose seed overlap clears the power line at the learned size, written to
`experiments/larger_size/results/amend1/`. Stage 2 reads that folder when job B is submitted; no decision is needed:

- **A candidate clears on `tofu`:** stage 2 uses it on `tofu` for AUTH_end, NEW_end and the FICT_learned re-reads,
  FO-1024 reported beside; on `T` likewise if it cleared there, else FO-1024 reported.
- **Nothing clears, or Amendment 1 has not run:** FO-1024, as E109. Stage 2 still runs: its power gate is the
  AUTH_end seeds against x, measured in job B, so the 100-row arm gets its own reading of power, and the states are
  saved for VPD either way; with no power, C still trains NEW_end and the rule is reported "no power at this size".
- **IG8 selected:** job B takes about 6 min longer; nothing else changes. Per-neuron FO scores are saved for every
  end state, so another K is a CPU re-cut, and an IG8 selection after stage 2 is one 15-min read job, not a retrain.

## 8. What each outcome means

- **Topic or style at both sizes, with power at the end.** A fact new in title and name, in the known words, acts
  through the known arm's neurons: set identity is closed as a novelty reading on Qwen, labelled with row 121, and
  the table's known-against-new rows read "neurons: style" at both sizes.
- **Matched topic or style; end familiarity.** The set reads new content once it is learned and E103-E109's matched
  reads were too early: a novelty reading at the learned size, provisional until the independent check and the VPD
  column on the same states.
- **NEW familiarity, NAME topic or style.** The new question, not the new name, moves the set; reported as such.
- **No call:** reported, no amendment. **No power at the end:** reported; the second amendment slot is for a
  ranking, not more seeds. **Novelty check fails:** TOFU's names or titles are in W0; stop, option (2).

VPD beside: the fp32 states (NEW matched and end, AUTH_end, two seeds each) go to qwen-decomp-lead by a note once
saved, for E94 Amendment 1's readings and E107 Amendment 1's switched-on-set Jaccard against the AUTH seed null.
Their decision, not this plan's cost.

## 9. Cost and routing

Cluster `debug`, one L40 a job, $0, three jobs, none over 20 min, under the gate (sources, appendix D): job A about
12 min (stage 0 about 1.5 min; four 64-step arms at about 2 min each, E107's AUTH arm took 119 s with reads and save);
job B about 15 min (two 335-step arms at about 6 min each with reads; FICT re-reads 1 min at FO, 7 min at IG8); job
C about 14 min. `--mem 24G`, GPU peak 41 GB as E107. Disk: six fp32 states, 34 GB, in `results/large/`, kept for VPD.

## 10. Code-writer hand-off

Write `experiments/invented_facts/{run,score,plot}.py`, `cluster/invented_facts.sbatch` (one script, an argument
`matched` / `end-known` / `end-new`, appendix D's timings in its header) and `results/predictions.json`. Shared code:
`src/commute/known_facts.py` gains `substitute(row, text, title_to, author_to)` beside `swap` and a `subjects()` map
(appendix B), tested in `tests/test_known_facts.py`; `train_seqs` gains a `subst` argument that generalises `wrong`
(callers `experiments/label_vs_style/run.py` and `experiments/larger_size/run.py` keep `wrong=True` working; nothing
else calls it). `score.py` imports E109's `score.wrong_rule`, `learned`, `sentences` and E107's `label_check`; no
new rule. `run.py` follows E109's shape. CPU dry run: `results/substitution.json` (every replaced string counted
per row), the edge cases, the stand-ins. Then `/pod-runner` for job A alone; B and C after A's score is read.
Report by `/writeup`, `TRIED.md` row, `PENDING.md` line, the note to qwen-decomp-lead.

## Not in this plan

World-fact rows (places and years, no invented pool); option (2)'s long-tail facts; an own-rows answer-token probe
for the new sentences; NAME at the end; more than two seeds; a fictional retrain to |δW| 8.3; any change of K or
ranking outside what E109 Amendment 1 selects; VPD reads.
