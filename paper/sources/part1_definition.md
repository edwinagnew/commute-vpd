# Part 1 summary: a working definition of reused against new (cells P1.1-P1.25)

*Read-only summary for the writeup-lead, 2026-11-17. All paths are relative to the repo
`/mnt/nw/home/e.agnew/commute-agents/writeup-lead` unless they start with `~`. Every reading below is provisional
(no independent check of any number on any page cited).*

## 1. Wordings tried, and what each read on the built controls

**Jain-style (frozen definition, decisions.md rows of 2026-10-29 and 2026-11-06).** A model possesses a capability
when a linear readout trained on pretraining text recovers it. A fine-tune reuses it when the new map is g after that
capability, plus a causal check. On the old 67M tasks no task got a verdict: possession failed everywhere because no
readout recovered its named capability both well and clearly above a control
(`reports/analyses/2026-11-06_analysis_e116-definition-table.md`, figure
`experiments/definition_table/figures/possession.png`). Under the refined rules only the in-context copy pair kept a
verdict (`reports/analyses/2026-11-05_analysis_e99-definition-verify.md`).

**First wording, "reuse at a variable" v1** (`~/commute-data/scratch/definition-session/2026-11-09_reuse-at-a-variable.md`).
It failed adversarial review before any run, and was then read only as a comparison arm. On the four hand-made pairs
it calls the built XOR "reuse", and every random direction counts as "acted on", so "built" almost never applies
(cell P1.3 state; `reports/analyses/2026-11-10_analysis_reuse-variable-first-test.md`).

**Second wording, v2: re-weighted / rewired / built** (`...2026-11-09_reuse-at-a-variable_v2.md`). This is the
**tentatively adopted** working definition. It has two shares. The re-weighting share G is the part of the gap closed
by an affine map of a four-direction variable from its own value. The rewiring share H is the part closed by a map
from everything computed before that layer. Cut-offs are G 0.8 and H 0.5. All numbers below are from
`reports/analyses/2026-11-10_analysis_gate-page-working-definition.md`.
- Hand-made changes: Always on G 0.92, Cue 0.67, Table -0.03, XOR -0.03 with H -0.01. Labels follow construction, as
  they must, since these set the cut-offs.
- Old tasks, 2 seeds each: in all five matched pairs every reuse seed has a higher G than every twin seed. The
  narrowest margin is 0.17, on repeat. Example: copy 0.70/0.69 against lookup 0.33/0.29. As absolute labels, only
  the real case label (0.83) reads re-weighted. The other reuse sides read rewired, like their twins.
- Built was never read. Four built-control attempts on 11-11 (two-key XOR, parity of a sum, strongly taught XOR,
  last-number-odd) gave no clean contrast. On the last-number-odd positive control H read 0.466 and 0.459, and the
  search reads only 0.84 on an exact edit.
- Memorised fact against its parity twin: G 0.10/0.12 against 0.39/0.40, and both read "no map found" (gate page,
  "Update 2026-11-12").

**Third wording, soft intervention** (`...2026-11-12_reuse-as-soft-intervention_v3.md`; tested, not adopted). The
fine-tune is the original with one named variable's rule changed, checked by interchange both ways (carry and
reversion). Every built edit read its constructed grade at bars 0.8, 0.9 and 0.95. On all twenty fine-tunes (fact,
twin, eight old matched pairs, two seeds each), the smaller of carry and reversion was at most 0.33, so no fine-tune
had a carrying variable among the named ones. Twin against fact was no call: the one-variable fit read 0.75 against a
bar of 0.8 (`reports/analyses/2026-11-13_analysis_third-wording-measured.md`, figure
`experiments/soft_intervention/figures/carry.png`).

**Fourth wording, simplest intervention, read as a ranking of rungs 0-4** (`...v4.md`; E184, tested, not adopted).
- The four controls read exactly their built rungs.
- Seven of nine other built items read above their filed rung. The planted constant read rung 3 against a filed 1;
  16 names closed at most 0.72 of its gap.
- Twin against fact was no call: twin rung 3 and fact rung 4 on both seeds.
- The old-pairs rule (R2) failed: it held in 7 of 8 pair-seeds, and case seed 0 read 2 against 1.

Sources: `reports/analyses/2026-11-16_analysis_e184-simplest-intervention.md`, figure
`experiments/simplest_intervention/figures/rungs.png`. Search miss against wording fault is unresolved; P1.21 was
cancelled with nothing read.

**Curve reading of E184 (E191, descriptive; E192, filed on fresh seeds 2 and 3).** The curve is closure against the
number of the original's variables used. On the fresh seeds, the filed rule (the reuse side exceeds the new side by
more than 0.0295 at 16 names, on all ten pair-seeds) failed. It held on 7 of 10 on the full distribution and 8 of 10
on answer tokens.
- Copy/cipher gaps were 0.36-0.40.
- Identity/random-map gaps were 0.43-0.44.
- Parity twin/fact gaps were 0.17-0.20.
- Repeat gaps were 0.020 and -0.015, and case gaps 0.035 and 0.023.

The seed 0/1 gaps show the same pattern (`reports/analyses/2026-11-17_analysis_e192-curve-test.md`, figures
`experiments/curve_test/figures/{curves,gaps,controls}.png`; seeds 0/1 in
`reports/analyses/2026-11-16_analysis_e184-curves.md`, `experiments/e184_curves/figures/curves.png`).

**Causal-rule retention** (`~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`; tested, not
adopted). The model is described as units with rules. A fine-tune retains a mechanism to the degree that the matching
unit keeps its rule. The reading is the mean retention over a smallest set of changed units that carries 0.9 of
KL(F‖M).
- P1.24 (E193/E194): do-nothing read 1 on all 12,288 neurons and 28 heads/blocks. A circuit doubled read 0.80 at
  neuron level and 0.906 at heads/blocks. The "replaced circuit" item could not be built in two runs: every fit
  either copied a circuit neuron (correlation up to 0.999) or failed to fit. Call: no call
  (`reports/analyses/2026-11-17_analysis_p1-24-rule-retention.md`, figures
  `experiments/rule_retention/figures/{calibration,replaced_fits}.png`).
- P1.25 run 1 (E195): this used the fixed choices (1 - cos², re-pairing inside a layer). Do-nothing, the moved copy,
  ×2 and ×10 all read 1. A circuit switched off and rebuilt by fine-tuning layers 2-3 read 0.97 (0.97154-0.97183 over 5
  orders) against 1. At heads/blocks it read 0.909 against 0.995/0.973. The knockout itself is invisible at neuron
  level. Lower as filed, but only by about 0.03, on a partial rebuild
  (`reports/analyses/2026-11-17_analysis_rule-retention-fixed-run1.md`, figure
  `experiments/rule_retention_fixed/figures/retention.png`).
- P1.25 run 2 (E196, five trained pairs): submitted 2026-11-17 19:10. `experiments/rule_retention_pairs/results/`
  holds only `predictions.json`, so there is no result yet.

## 2. Has the user adopted a definition?

Only the second wording, and only tentatively. `decisions.md`, row dated 2026-11-10, quoted exactly:

> | Working definition of reuse (ask 29, card `reports/decisions/2026-11-10_decision_gate-working-definition.md`; user, 2026-11-10, task file and dialog): "yes you can tentatively adopt the 2nd wording as the working definition of re-use. if it cant detect built, then it will have to be revised however." Cut-offs rounded by the user: G 0.794 -> 0.8 ("round the 0.794 to 0.8 for simplicity") and H 0.476 -> 0.5 ("i would prefer 0.5 so maybe the old task stuff could be negative results? if i made 0.476 the official cut off in the paper, taht would look extremely suspicious"). The built control (plan `reports/plans/2026-11-10_plan_built-control-design.md`, candidate 1) runs as a new cell on L1.4. | 2026-11-10 | gate page P1.1; ladder L1.4 | user, 2026-11-10 |

It was kept after the first built control failed (row 2026-11-11: "...keep the old wording..."). Since then:
- Third wording: testing only (row 2026-11-12, E173).
- Fourth wording: "tested, not adopted ... the wording is not adopted until a later dated row" (row 2026-11-13).
- Fourth wording failure recorded (row 2026-11-16).
- Retention: the draft's fixed choices are used "as written" for P1.25 (row 2026-11-17).

The user's own condition, "if it cant detect built", has not been met: built has never been read
(gate page, "The built control, as run"). The roadmap's gate was "reuse on the built-to-reuse side and not on its twin,
on three pairs, three seeds each" (`reports/plans/2026-11-09_plan_paper-roadmap.md`). It has not passed under any
wording.

**Current standing candidate:** causal-rule retention, under test in P1.25. Open items:
- E196's reading of the five pairs.
- The replaced-circuit calibration, never built.
- The question from the premise check: "if a fine-tune's new units reproduce an old circuit's outputs, what should
  the definition call that, and what observation would decide it?"
  (`reports/digests/2026-11-17_digest_definition-lead.md`)
- Whether the case and repeat pairs really differ in reuse (P1.23 premise check).

## 3. Built controls and tasks

- **Hand-made edits** (E137): one each, written into the plain 67M. These were Always on (a fixed push on a variable),
  Cue (a push set by one rare token), Table (set by 32 keys) and XOR (set by an XOR of random bits, so new by
  construction). Ordinary fine-tunes imitating them were trained on 3 seeds (E139).
- **Planted edits** (E168, E173, E184): a constant push, a push gated by last-number parity, a random-bit-per-pair
  "floor", and a planted fact.
- **A matched pair** means two fine-tunes on the same inputs, push size and prior. One side is built to reuse
  something the original already does. The other side is built to be new, either by a random target or by shuffling
  labels per word, so it can only memorise. The five trained pairs (E192, card section "What we were trying to test")
  are:
  1. copy against cipher (24 keys);
  2. identity against random map (32 keys);
  3. real case label against its per-word-shuffled twin;
  4. real repeat label against its twin;
  5. parity twin (push when the last number is odd) against memorised fact (push on a random bit per number token).
- **Seeds:** 2 seeds (0, 1) in E142/E184. Seeds 2 and 3 were added in E192, so the curve reading has 4 seeds per pair.
  The in-context copy against lookup was a sixth pair under the second wording only.
- **Unpredictable controls:** one shifted neuron, one triggered neuron, and a random bit per window.

## 4. Claims that could go in a paper today

1. **Jain et al.'s definition gives no verdict on the 67M tasks.** Possession fails on every task. *Solid as a
   negative within its scope; provisional* (one job, no independent check). Figure
   `experiments/definition_table/figures/possession.png`.
2. **The re-weighting share G ranks reuse above new in all five matched pairs on 2 seeds, by at least 0.17.**
   *Provisional.* Figure `experiments/old_task_labels/figures/gate_within_pair.png`.
3. **As absolute labels, the second wording does not follow construction.** Only the case label reads re-weighted,
   and built is never read. *Solid* (as a failure). Same source.
4. **Closure curves separate 3 of 5 pairs on all four seeds:** copy/cipher, identity/random map, and parity twin/fact.
   *Solid within the 67M* (filed before the fresh seeds). Case and repeat do not separate, so the filed "all pairs"
   rule is *contradicted*. Figures `experiments/curve_test/figures/curves.png`, `gaps.png`.
5. **Several wordings read hand-built edits correctly but give no verdict on real fine-tunes.** For the third wording,
   min(carry, reversion) was at most 0.33. For the fourth, 7 of 9 built items read above their rung. *Provisional*
   (search against wording unresolved). Figures `experiments/soft_intervention/figures/carry.png`,
   `experiments/simplest_intervention/figures/rungs.png`.
6. **Neuron-level rule retention separates a rebuilt circuit from a turned-up one only narrowly (0.97 against 1),
   and does not see a knockout.** *Provisional* (one circuit, one fit seed, partial rebuild). Figure
   `experiments/rule_retention_fixed/figures/retention.png`.
7. **Fresh neurons fitted to a circuit's outputs copy one of its neurons** (corr ≥ 0.995 in every fit that fits).
   *Provisional.* Figure `experiments/rule_retention/figures/replaced_fits.png`.
8. **Related-work argument:** no diffing method has been validated against a known reuse/new label
   (`reports/paper_notes.md`). *Provisional:* it depends on a built control reading "new", which none has done
   cleanly.

## 5. What a paper section still needs

- **A definition the user adopts** by a dated row, or the decision to frame part 1 as "why existing definitions
  fail" (roadmap gate). The user did not recognise that fallback (row 2026-11-09, swap calibration).
- **E196 results** (retention on the five pairs, seeds 0 and 1), and the replaced-circuit calibration, never built.
- **A clean built control** where the reuse arm reads high and the built arm low (gate page, "Not run").
- **A third seed** for the second-wording pair ranking. The roadmap gate asks for three seeds; E142 has two.
- **An account of case and repeat** that does not come from the measure itself (E192 "Not run"). Also an answer
  reading on seeds 0/1, a noise floor for the answer reading, and the other two unpredictable controls rerun.
- **A search that recovers planted edits at their rung,** to tell a search miss from a wording fault (P1.21
  cancelled).
- **Independent checks.** No headline number has had one. At least claims 2-4 would need one if they enter the answer
  paragraph (`.claude/CLAUDE.md`, Reviews).
- **Qwen or a fresh held-out matched pair,** listed in the gate page as what would make us drop the wording.
- **Literature departures** written per use. Each page has a departures list. The gate page compares Jain (probe for a
  named function) and Prakash (patching named heads, scored on task accuracy).

Friction: the 2026-11-16 gate-page addendum says "the old matched pairs held in seven of eight cases" while E184's
scorecard calls R2 "does not hold". Both are right (7 of 8 held, the rule needed 8), but read out of context they look
contradictory. No file lists which wording is current in one place, so this had to be pieced together from about eight
decisions.md rows.
