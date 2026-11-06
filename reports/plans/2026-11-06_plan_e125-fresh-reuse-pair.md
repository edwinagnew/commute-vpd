# E125: a fresh 67M pair that the frozen reuse definition can read as "reuses C" (the repeat label)

*regating-lead's writer, 2026-11-06. Plan, filed predictions and code; nothing run. Not gated (three MATS `debug` L40
jobs, about 30 min in all, $0). Apparatus detail and the time sources: `reports/plans/2026-11-06_plan_e125-fresh-reuse-pair_appendix.md`
(App. §n). Predictions `experiments/fresh_reuse_pair/results/predictions.json`; code `experiments/fresh_reuse_pair/{run,stage2}.py`,
`src/commute/{repeat_label,label_reads,decomp_reads}.py`; job `cluster/fresh_reuse_pair.sbatch`.*

**In plain words.** We teach the small model to say one word when the word just before a marker already appeared
earlier in the passage, and another word when it did not, beside a twin that learns the two words from a shuffled
per-word list. The model's own induction machinery already tracks repeats, so this is a fine-tune that could reuse
an existing capability, and the reuse definition can read it on words the fine-tune never saw. On the same four
fine-tunes we then read the decomposition side, so that for the first time both readings could say "reuse" together.

## The request, restated in the Terms' words

Build one 67M task pair of E112's shape on which the frozen definition (`decisions.md` rows 81, 87, 164, frozen by row
165) can reach "reuses C", and read the same four fine-tunes by the decomposition side: the route split's re-switching
share r and masked-off share e_gate under E119's filed lines (fine-tune and teacher-trained copy) and E38 Amendment 2's
trigger-only removal of the switched-on set S. E112's case-label pair cannot do this: its real arm reads "not through
C" three times (E112 stage 2, Amendment 1; E113). The candidate is Fable's (`~/commute-data/scratch/regating-lead/e123-fable/plan_draft.md`
§4): C = "the word before the trigger already occurred earlier in the window". Stage 0 measures possession before
anything else; a miss stops the job and is the result "possession fails".

## The definition this reads under, quoted, nothing added

- Row 81: "M possesses capability C if, on a sub-domain X_C with x_i = i_C, some layer has a linear readout (trained on
  pretraining data) equal to f_C(x_d) ... Def 2: fine-tuning reuses C if the post-trained map is g∘f_C".
- Row 87: "a fine-tune reuses capability C only if g, learnt on the fine-tuning examples, also works on held-out values".
- Row 164: "Possession = a linear readout trained on Pile text at some block, beating a Hewitt-Liang control task by a
  bar filed before any run ... the one added check = removing the readout's direction breaks the fine-tuned behaviour
  and spares a twin fine-tune at the same trigger, with pruning and reverse fine-tuning reported beside it."
- Bars, verdict rule, edge cases: E112's, copied unchanged (`experiments/case_label/results/predictions.json` → rules:
  possession accuracy 0.90 and selectivity 0.30 before the last block; carrying, small function 0.90; removal 0.50;
  random direction at most 0.10; gain floor 0.1), applied by E112's functions in `src/commute/case_label.py`.

## Design check (`/design`)

- **Goal.** `GOALS.md` asks whether post-training teaches new capabilities or strengthens existing ones, judged by a
  label set outside the decomposition. No 67M task has the label "reuses C" (E116's table; E112). This pair is built to
  get one, so the decomposition side can be checked in the positive direction, not only against "not through C".
- **Nearest tried.** E112/E113 (TRIED.md E112-E114 rows): same shape, C = the previous word's case; possession,
  carrying and small function passed, removal took 0.08-0.21 against 0.50. E116: every other old task fails
  possession. E123 (Fable's draft, not yet on origin): the decomposition side on E112's states. E124
  (definition-lead's plan, awaiting the user): natural agreement and pronoun tasks. New here: a C that is not part of
  any token's identity. Case is: its own-word readout reads 1.00 at block 0 (E112 Amendment 1), and its full erasure
  took 148 directions (Amendment 2). So a fine-tune can read case from many identity directions that a one-direction
  removal leaves alone. "Repeated" exists only in context: the model must compute it by attention. That makes a
  low-rank readout more plausible, though it is not shown.
- **Patch or step.** A step: the first pair chosen so that the definition can say "reuses" and the decomposition can be
  read on the same states. Not a third E112 amendment: a new task, with E112's rules unchanged.
- **Decisions relied on.** Rows 81, 87, 164, 165 (frozen); row 166 (reverse fine-tuning stays beside); E119's lines
  (E12, E12 Amendment 1); E38 Amendment 2's rule. Departure to flag: row 175 said "No new task pairs" for the
  definition line on 2026-11-06. This pair is the re-gating lead's, from the user's later "build one task both read".
  Row 178 allows new 67M tasks for the definition. **A new task family: needs the user's go-ahead after this check.**
- **Leads kept apart.** E124's tasks (agreement, pronoun) are natural-answer tasks owned by definition-lead. E125 uses
  E112's built-label form and neither of E124's properties. If both run, the write-ups cite each other; neither
  repeats the other.
- **Reads unchanged?** Yes. Every verdict comes from E112's functions and bars. Every decomposition label comes from
  E119's `label` and E38 A2's `d_seed_class`, moved into `src/commute/decomp_reads.py`, same calls. One case the old
  rule cannot read is reported, not ruled: an empty S reads "unscored: S is empty" (as E38 reported lowercase).
- **What changes next.** (i) The definition reads "reuses C" on the real arm and the decomposition agrees: the first
  positive anchor, and the paper's reuse sentence gets a 67M case. (ii) Reuse, but the decomposition gives no label or
  the opposite: the decomposition is shown blind or wrong on a case the definition reads. (iii) The real arm reads "not
  through C" again: a second shape fails the causal check, and the frozen check is the limit at 67M. Write that up with
  E112. (iv) Possession fails: the candidate is out, and a two-valued contextual property is not linearly readable at
  the trigger in this model.
- **Control first.** Stage 0 is W0 alone and stops on a possession miss. Its positive control is the induction gap:
  W0 predicts the token after the earlier occurrence only in repeated windows, so the model reads our written repeats.
  Stage 1 reads the random direction on all four states before any verdict (E112's order). Stage 2 reads E119's
  do-nothing identities and E38's empty-set removal before any label, and reads E12 Amendment 1's students as the
  route rule's positive control.
- **Numbers.** Below and in the predictions' `expected`. Each has its W0 value, its do-nothing value and its control.
- **Saved.** Windows, labels and the probe's Pile windows (`results/large/task.pt`); probes; the four δW in bf16 with
  sha256; the four students as bf16 offsets; every read in `results/stage<N>.json`.

## The pair (App. §1)

- **Pool.** Single-token words (a space and letters) with at least 32 Pile positions in each class: with and without
  an earlier occurrence in the 19 tokens before. Words that repeat more than half the time are out, and so are the
  marker and the label candidates. From those that qualify, 400 are drawn by seed: 300 train, 100 held out. They are
  drawn, not ranked, because a ranking by count gave a pool led by " of", " to", " and" in the CPU dry run (App. §1).
- **Windows.** E112's layout and rows: the word at 19, E112's marker " \+" at 20, label read at 20. The span holds the
  word nowhere in positions 0-18. One position p in 1-17 is overwritten with the word itself (repeated, y = 1) or with
  another pool word (not repeated, y = 0). Both classes hold one written pool word before the word, and each word
  appears in both classes. The builder asserts the class from the tokens.
- **Real.** Label A (" carpet"-type, drawn by E112's seed from its list) if repeated, else B.
- **Twin.** Same windows, A or B by a seeded shuffle per word, as E112's twin. Only memory can learn it, and it is
  at chance on held-out words by construction.
- **Possession probe.** E112's ridge, at the trigger position of Pile windows whose previous token is a train word.
  The label is the earlier-occurrence flag, balanced per word (at most 16 positions per class). It is scored on the
  held-out-word task windows. Control: one random value per token type (E116's per-type Hewitt-Liang form). Every word
  has both classes, so a type readout scores 0.5.

## Stages, cheapest first, with what each should read

| stage | job | reads | expected (lean) | stop |
|---|---|---|---|---|
| 0 | W0 only, ~3 min | W0 label accuracy; induction gap; P0 per block and control; W0 under the removals | accuracy 0.40-0.60; gap > 0, about 1 nat or more; P0 passes at block 1 or 2 (about even odds), control 0.45-0.55 | P0 misses at every block before the last: exit 1, "possession fails" |
| 1 | ~8 min | four fine-tunes (E112's recipe, 2 seeds); random direction on all four; P1-P3, verdicts; pruning, reverse beside | random direction ≤ 0.002; real: P1, P2 ≥ 0.90, P3 ≥ 0.50, "reuses C" (low, about one in three); twin: P1, P2 0.45-0.55, P3 ≤ 0.05 | random direction > 0.10 on a state: exit 1 |
| 2 | ~17 min | route split of each δW (do-nothing, least squares); E38 A2 removal of S; E34's student per state | e_gate = −r and LS r = do-nothing r to 1e-6; r 0.45-0.65, real and twin within 0.15, "neither" on all; copy r within 0.06 of its fine-tune; \|S\| > 0; removal "not"/"unclear"; E12 A1 control 4 re-switching, 4 masked-off | an identity over 1e-6 or the empty-set removal over 1e-5: exit 1 |

**Why the real P3 might pass this time, and why it might not.** For: "repeated" is not in any token's embedding, so the
fine-tune cannot read it from many token-identity directions, as E112's could. If it reads the flag at all, it reads a
computed one, possibly the probe's direction. Against: E112's one-site, one-direction removal took 0.08. The fine-tune
could read the flag at the word's position (19) through attention at a later block, missing the trigger site, or build
its own detector. The lean is low.

## What each joint outcome means (no rule joins them; App. §3)

| definition, real / twin | decomposition, real / twin | meaning |
|---|---|---|
| reuses C / not through C | re-switching (route, fine-tune or copy) or "through" (removal) / otherwise | **agree, positive**: the first case where both readings say reuse, and the decomposition tells it from the twin |
| reuses C / not through C | neither or unclear / the same | the decomposition gives no label where the definition reads reuse; its "neither" band is blind here |
| reuses C / not through C | masked-off or "not" / any | **disagree**: the decomposition calls a reuse case new |
| reuses C / not through C | twin re-switching, real not | reversed: the reading follows something other than reuse (update size, trigger) |
| not through C / not through C | any | E112's outcome again on a second shape; the decomposition labels are reported as E123 reports E112's |
| unreadable (possession fails) | not run | the candidate fails; stop and report |

The twin's update was about 3x the real's in E112, and we expect the same here (predictions → expected). Any
real-against-twin difference in the decomposition must carry that caveat in the same sentence.

## Blind spots, noise, controls

- The probe reads one direction at one site; carrying and small function are near-tautologies once readout and
  labels are both near 1 (E112's caveat). P3 is the deciding test, and it is linear and rank-1 at one position. It is
  blind to the flag read at the word's position or recomputed later. Noise floor: the random direction (E112: 0.0006).
- The route split reads one position, full-vocabulary log-odds of the label token. It is blind to which components move
  and to the A-against-B choice: the twin's gain is mostly both labels rising. Noise: E12 Amendment 1's batch-order
  floor, |Δe_gate| 0.002. Positive control: E12 Amendment 1's students (copy re-switching, maps masked-off).
- The removal reads S at the threshold 0.3 only and the trigger only. Noise: its 20 random sets. Its control is the
  empty-set change.
- Two seeds per arm; one marker; one pair of label words; 256 route windows of training words.

## Time (App. §4)

Stage 0 about 3 min, stage 1 about 8, stage 2 about 17: one `debug` job each, sources in App. §4. Stages 1 and 2
exit 3 and resume if the 25-min time check would be passed.

## Departures from the literature

- E112's departures all stand: a binary ridge readout in place of Jain et al.'s grammar probe; held-out carrying (row
  87); the direction-removal check (row 81); E62's fine-tune recipe.
- The control is E116's per-type form of Hewitt and Liang's control task (one random value per word type), not
  E112's per-pair form, since this pool has no pairs.
- The task windows are built by writing a word into Pile text, as E112 wrote a word and a marker. The probe is trained
  on natural repeats only.

## Not run (by design)

- Amendments of E112's removal form (every block, every position, INLP): this pair is read by the filed one-site form
  first. A stronger removal would be the pair's own amendment, the user's call.
- Teacher training on Pile text (arm B), a third seed, other markers or label words.
- Qwen: only if the 67M pair reads (row 178).
- The definition at the word's position: the probe there is reported beside, with no rule.
