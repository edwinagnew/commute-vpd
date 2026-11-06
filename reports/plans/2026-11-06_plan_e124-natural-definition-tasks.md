# E124: natural tasks that both the frozen reuse definition and re-gating can read (67M)

*definition-lead, 2026-11-06. Plan only; nothing run. A new task family, so it waits for the user's confirmation after
the design check below. Not gated (about one hour of MATS `debug` L40, $0). Apparatus detail:
`reports/plans/2026-11-06_plan_e124-natural-definition-tasks_appendix.md` (cited as App. §n).*

**In plain words.** We will teach the small model two things it already half does in ordinary text: "is" or "are" to
agree with a subject a few words back, and "he" or "she" to match an earlier first name. Each task also has a version
that must say a new word for the same property, and a twin that follows an arbitrary per-word list. The run asks whether
the frozen reuse definition gives these tasks a verdict, and whether re-gating separates them as on the old tasks.

## The request, restated in the Terms' words

Find natural 67M tasks with the shape the frozen definition reads (`decisions.md` rows 81, 87, 164): a two-valued
property C of the input, read by a Pile readout before the last block, and an answer defined for held-out values. On
each, fine-tune three members (E112's arm-A recipe): the **natural member**, whose answer the original model already
gives; the **new-word member**, the same C mapped to two words the original model does not give there; and the **twin**,
the natural answer words assigned by a shuffled per-word list. Read every state twice on the same windows: (a) the
definition's verdict by E112's code and bars unchanged; (b) the re-gating share r and the stand-in ratio (`GOALS.md` →
Terms) of the least-squares split. The tasks "replicate re-gating" when the natural member reads r above 0.5 and the
new-word member below it, as E60 and the synthesis found (below). Qwen only later, and only if these read (user, today).
New tasks only: regating-lead is reading the old 67M tasks against rows 81/87 (`~/commute-data/agents/regating-lead/handoff.md`,
"Next") and drafting E123, "build one task both read" (same file, "Now"; its draft was not found on disk, so overlap
is unchecked: the lead should compare before coding).

## The definition this reads under, quoted, nothing added

- Row 81: "M possesses capability C if, on a sub-domain X_C with x_i = i_C, some layer has a linear readout (trained on
  pretraining data) equal to f_C(x_d) ... Def 2: fine-tuning reuses C if the post-trained map is g∘f_C".
- Row 87: "a fine-tune reuses capability C only if g, learnt on the fine-tuning examples, also works on held-out values".
- Row 164: "Possession = a linear readout trained on Pile text at some block, beating a Hewitt-Liang control task by a
  bar filed before any run ... the one added check = removing the readout's direction breaks the fine-tuned behaviour
  and spares a twin fine-tune at the same trigger, with pruning and reverse fine-tuning reported beside it. Verdicts:
  'reuses C', 'not through C', 'unreadable', 'definition does not apply'."
- Bars, verdict rule and edge cases are E112's, read from `experiments/case_label/results/predictions.json` (`.rules`:
  possession accuracy 0.90 and selectivity 0.30 before the last block; carrying 0.90; small function 0.90; removal 0.50;
  random direction at most 0.10; gain floor 0.1) and `src/commute/case_label.py` (`first_passing_block`, `verdict`,
  `removed_share`, `fit_threshold`, `do_nothing_ok`). Nothing new decides a verdict.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s "new capabilities vs strengthening existing ones" needs labels set outside VPD; E116 found the
  frozen definition labels no old 67M task. This gives it natural tasks it can read, on states VPD also reads.
- **Nearest tried.** E112/E113 case label (TRIED:222, :225): possession, carrying and small function passed; removal
  0.08-0.21 and patching missed, so the real member read "not through C". Its answers were built label words. New here:
  natural text and natural answers, and r read on the same states (E112 never read r, plan line 93).
- **Patch or step.** A step: the first tasks chosen for both readings at once.
- **Decisions relied on.** Rows 81, 87, 164; row 173 paused case label and declined "subject number anyway" in its
  label-word form; this is the verb-answer form the review recommends instead (review §4.1). The user's rule "only if
  case label reads cleanly" (row 164) was about that label-word version: flagged for the user to confirm.
- **Reads unchanged?** Yes. The window builders are new; verdicts come from E112's functions and bars. An arm the rules
  cannot read is reported "unreadable"; a miss is a result.
- **What changes next.** Both tasks read and re-gating replicates: the joint table (verdict against r) is the deliverable
  and the Qwen question opens. Re-gating does not replicate: the tasks still give definition verdicts, but the two
  readings cannot be compared on them. Possession fails on both: natural two-valued properties are unreadable at the
  67M's readout, and we write that up.
- **Control first.** Stage 0 alone, W0 only: window counts, the possession readout and its control, W0's random-direction
  removal, the zero update's route cells, and E60's re-read of E54 elicit seed 0 through the per-window target path.
- **Numbers.** Table below; each has its value at W0, at the do-nothing (zero) update and for its control.
- **Early guards.** Stage 0's stops (App. §4); stage 1 logs train accuracy and the full-vocabulary log-odds gain at
  every checkpoint, so a member with no gain shows in the first minute.
- **Saved.** Windows and pools (`results/large/task_<task>.pt`), probes, every scored update in bf16, per-window
  margins and per-window masked log-odds as JSON beside (App. §5).

## The tasks (two of the shortlist's three; App. §1-2)

1. **Agreement.** Pile windows ending "the N PREP (the) N2", where the text's next word is " is" or " are". C = the
   number of N. Pool: noun pairs (" key", " keys") that are both single tokens, top 400 by count in the pattern, 300
   train and 100 held-out pairs. Half the windows have an attractor (N2 of the other number), after Linzen et al. 2016's
   attractor construction, mined from the Pile (no parser); windows whose verb disagrees with N are dropped, and their
   share is reported.
2. **Pronoun.** Pile windows ending at a position whose next word is " he" or " she", with exactly one pool name 3-15
   tokens back. C = the name's gender, from the US Social Security Administration's national first-name counts (share
   of one gender ≥ 0.95), single-token names only; names split 3:1 within gender. Windows whose pronoun disagrees are
   dropped and counted.

Rejected: case-style matching (shortlist 3): C may be read off the current token, and its per-type control has no key
slot, which would need a new filed choice. Qwen abstention and the class map: out of scope (review §4.4-5).

**Members (same windows, same positions, per task).** Natural: the text's word. New-word: E112's label words (" carpet",
" camel", `label_seed` 1) by C, the "answer W0 does not give" cell. Twin: the natural words by a shuffled per-word list
(`CL.twin_y`), the definition's twin. Fine-tune: E62's arm-A recipe as E112 ran it (Adam 3e-5, batch 32, KL 1, clip 1,
seeds 0 and 1, checkpoints 10-1000, scored at the first checkpoint with train accuracy ≥ 0.90).

**Headroom (stage 0).** The natural member must leave learning to do and still be "already given": W0's train-window
accuracy (natural word over the other, by margin) between 0.65 and 0.85, else that task stops. The upper bound keeps the
0.90 learning bar from passing at step 10 with no gain; the lower bound is where "W0 already gives it" stops being true.
Both are chosen here, not sourced (they gate entry, not verdicts).

## Numbers and expected values

| # | Number (per state) | W0 / zero update | Natural | New-word | Twin | Control |
|---|---|---|---|---|---|---|
| P0 | Possession, E112's (per task) | passes at block 1 or 2 (lean 0.6) | - | - | - | per-type control ≈ 0.5 |
| P1 | Held-out carrying | natural ≈ 0.75, others ≈ 0.5 | ≥ 0.90 | ≥ 0.90 | ≤ 0.65 | - |
| P2 | Small function | - | ≥ 0.90 | ≥ 0.90 | ≤ 0.65 | - |
| P3 | Removal share of the label gain | zero gain: "no gain" | lean < 0.50 | lean < 0.50 | ≤ 0.20 | random direction ≤ 0.10 |
| R1 | r, full-vocab log-odds, fit windows | nan (gain 0) | 0.72-0.92 | 0.12-0.29 | flipped windows 0.3-0.6 | E54 elicit s0 0.7726 ± 0.03 |
| R2 | Stand-in ratio | nan | 0.87-0.97 | 0.26-0.41 | reported | zero update exact 0 |

Expected bands for R1-R2 are E60's (TRIED:124) and the synthesis's (`reports/analyses/2026-10-30_analysis_regating-synthesis.md:84-100`).

## Run order and cost (MATS `debug` L40, $0; App. §3)

| Stage | What | E112's measured time | E124 estimate |
|---|---|---|---|
| 0, alone | data, counts, headroom, P0, W0 removals, route-split controls | 27 s after load (`experiments/case_label/results/stage0.out:5-9`) | one job, ≤ 10 min (two tasks, name-list fetch) |
| 1 | 12 fine-tunes, each saved at once | 4 states in 38 s (`stage1.out:8-98`) | one job, ≤ 10 min |
| 2 | P1-P3 and pruning, reverse fine-tune beside; then R1-R2 | under 7.5 min, stage 1's end to stage 2's (`stage1.out:98`, `stage2.out:28`); E62's route split, 6 states in ~3.5 min (TRIED:136) | one or two jobs, ≤ 25 min |

Three to four jobs, about an hour of card time, through `/pod-runner`. Fast lane: no pre-run review.

## Predictions in words (filed by the code writer in `experiments/natural_tasks/results/predictions.json`)

- **Definition.** Natural: "not through C" on both seeds (lean 0.55, from E112's removal at 0.08-0.21), else "reuses C".
  New-word: "not through C" (lean 0.6, E112's own shape). Twin: "not through C" (0.85). Possession fails on a task: that
  task reads "unreadable" and stops (E112's code).
- **Re-gating.** Natural r ≥ 0.5 and new-word r < 0.5 on both seeds, separated by E54's rule (`elicit.separation`: gap
  above the larger seed range and ≥ 0.2): "replicates" (lean 0.7). Twin on its flipped windows: under 0.5 by the own-guess
  account, above it by the prior account (the wrong form is a likely word there); reported, no rule.
- **Joint.** New-word reading "reuses C" with r under 0.5 is the informative disagreement (same C, new word); natural
  "reuses C" with r above 0.5 agrees; natural "not through C" with r above 0.5 says r follows the answer's prior, not C's use.
- **Edge-case pass** (App. §6). Empty or short pools stop stage 0. Under a gain of 1 log-odds r is nan and the state
  reads "no call", never a new band. A tie at 0.5 counts as above, for both members. The zero update reads "no gain"
  and r nan, so the do-nothing state passes neither bar. W0 already passes carrying on the natural member, so its
  "reuses C" rests on the removal check, as the frozen rule says.

## Decision tree

- Stage 0 count short, or a control missed: stop and report; the code is fixed before anything else runs.
- Headroom outside 0.65-0.85: that task stops; Amendment 1 rebuilds it from attractor windows only (or both-sides only,
  if over 0.85 means too easy).
- P0 fails: that task is "unreadable", reported as a result, no fine-tune.
- A state with no gain or r nan: reported as read, "no call", no amendment.
- Re-gating does not separate: written up as "not replicated on this task"; the verdicts stand alone.
- Two amendments per E-code; a third is a new plan.

## Not run, and risks

Not run: Qwen; E113's patch form (removal is row 164's check); a random update of matched size (on the 67M its gain is
near zero, so r is nan); case-style matching; shortlist 4-5. Risks: few pattern windows per held-out pair; regex subjects
mislabel some windows (dropped when the verb disagrees, which also drops natural errors); the name list (not in the repo;
URL to verify) may give few single-token names; r may read the prior of the twin's word, as the synthesis found.

## Go-ahead (Fable, 2026-11-06 evening; the user delegated go-aheads to Fable)

"Go with changes", advice `~/commute-data/scratch/definition-lead/e124-plan/fable_advice.md`. Changes, none of which
adds a verdict rule, band or score:
1. Scored state = the first checkpoint with train accuracy ≥ 0.90 (E112's learning bar) and full-vocabulary log-odds
   gain ≥ 1 (`local_faith.MIN_GAIN`). A recipe departure from E112, so the natural member is scored after it has learned.
2. Agreement probe windows balanced half attractor and half not per train pair; readout accuracy printed per kind.
3. The 0.65-0.85 headroom band is replaced by two gates: natural W0 train accuracy < 0.90 (E112's bar) and above the
   new-word member's W0 accuracy on the same windows. W0 accuracy printed per kind.
4. Names: `https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv` (SSA extract, top 1000 per
   sex per year, 1880-2008), percent summed 1950-2000, one-gender share ≥ 0.95. Fetched on dev into
   `artifacts/sets/ssa_names/`, sha256 in stage0.json. Caveat: top-1000 names only.
5. Twin P1/P2: filed as "≈ W0's agreement with the twin labels, reported", not ≤ 0.65.
6. Each seed gets its own verdict; a member-level prediction passes only if both seeds match.
7. Stage 0 prints drop share per kind and the flipped-twin fit-window count.
8. Row 164's subject-number clause covered the label-word version in the E99-reset sequence. The user's evening
   retarget and ask 9 supersede it (decisions row, 2026-11-06). E124 reads r and the stand-in ratio itself with the
   existing readers; teacher training and E38's switched-on set are left to regating-lead, who gets the states.

## Revision before any run (Fable, 2026-11-06, later): pronoun only, smaller sizes, more Pile

The writer's count on Pile rows 0-32767 found agreement at 155 windows in total and pronoun at ~1,000 kept training
windows, 10-100 times short (`~/commute-data/scratch/definition-lead/e124-writer/explore*.out`). Fable's revision
(same advice file, "2026-11-06, later"). Nothing was filed or run, so this is a plan revision, not an amendment.
1. **Agreement dropped.** Only the attractor form keeps W0 under 0.90, and its cells hold 2-82 windows per 32k rows.
   A TRIED row records it. If a second pair is wanted later: his/her after a name (same C), as Amendment 1.
2. **Pronoun sizes.** Train 3,200, fit 512, held-out 800, balanced by gender. The pool is the top 200 names per gender.
   E112's sizes were a choice: carrying needs ~400+ held-out windows for a 0.90 bar, and P3 and r are means that E54
   and E53 read on 97 and 500 windows.
3. **Pool filter, a counted rule.** Exclude a name whose lowercase token is at least as frequent in the Pile as its
   capitalised one (months, places, common words). The excluded list is printed, and the drop rule runs after it with
   the drop share per gender. Never exclude by a name's own drop share.
4. **More Pile.** `data.pile_rows("train", 65536, 524288)` in a CPU Slurm cache job, after
   `cluster/jain_readout_cache.sbatch` (65k rows in under 10 min, job 48321). Probe rows stay 32768-65535.
