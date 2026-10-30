# Advice (Fable, 2026-10-31): what follows E55 C1b

Advice only; the lead decides. Read: `reports/analyses/2026-10-31_analysis_e55-c1b.md`, the C1b plan §8 (C2),
`reports/plans/2026-10-30_plan_regating-ground-truth.md` (Stage B offer, E58 note), `GOALS.md`, `TRIED.md` E55 rows,
`decisions.md` rows 103-106, `/design` approved classes.

## Where we stand
C1b gave the pragmatic instruments their fair chance and all three passed, 21 of 21, with one cut-off each that held
across both recipes. The set is degenerate because its label is fixed by construction (which task was trained), so any
reader of the construction (the answer-only probe; update size within a recipe) ties. Note that this is structural:
**every** by-construction label is some function of the construction, so no by-construction set can make VPD "beat"
a labeller that knows the construction. Only states whose label is **open**, decided by the behaviour and patch checks
rather than by how they were built, can rank instruments. One advantage is already on record: the three instruments
needed one cut-off across recipes; update size needed one per recipe (a factor 30 apart).

## Options
| | cost | can show | cannot show |
|---|---|---|---|
| **A. C2 as planned** (VPD on the 21 states) | minGPT port to `param-decomp` 1-2 agent days; decomposition $10-30 H100 (gated) or $0 on cluster `normal` if the rehearsal fits, which needs the "Qwen only" rule relaxed | VPD **fails** where cheap methods pass: the sceptic's result, high value. Tie: VPD not worse, and the brief's Stage 1 item "score VPD on the anchors" is met | a tie cannot rank VPD above the probe or update size; a tie is the likely outcome (every instrument separated these states with wide margins) |
| **B. Harder set as the report proposes** (P from NEVER vs P from LOW; or N from HIGH vs from LOW) | $0 cluster, ~5 h GPU, 1-2 agent days; C2 on it needs a **second** decomposition (NEVER or HIGH), doubling A | matched answers, so the probe is blind | swaps task-degeneracy for base-degeneracy: "which base" is a 100% labeller, and update size (NEVER must build a counter), lr and steps all re-confound. Not harder in the sense that matters |
| **C. E58's 12 open-pair 67M states** (elicit, rank-1, copy, cipher × 3 seeds) as the open set | $0 cluster, one to two debug jobs; port pruning revival and reverse fine-tuning to the 67M (revival.py is `nn.Linear`-generic; the "old behaviour" to revive = W₀'s own next-token distribution on the keys), ~1 agent day; steering S, r and the stand-in ratio **already read** there, 67M decomposition exists | the one question VPD's standing now turns on: on a map the held-out label calls "new" (elicit, carry ≈ 0) while VPD reads "old pieces" (r 0.76-0.79, ratio 0.93-0.95), do the pragmatic instruments side with VPD or with the label? Same → the granularity limit is a property of all instruments, not VPD. Split → VPD is the odd one out. n = 12, open labels, labels graded | the label is graded, not binary (carry 0.55 → −0.3); the interchange-patch label (Stage B) is not built; one base, one task family; the brief's Stage 1 anchor item for VPD stays unmet |
| **D. Decide the update-size question first** | $0, a decision row | in deployment the recipe is known, so update size within a recipe is a legitimate cheap baseline column, not a nuisance; every later table carries it | nothing new on its own |

## Recommendation
1. **Do C now and D now; defer A; drop B as stated.** C is the only $0 step whose result is unknown and bears on the
   user's priority (VPD vs pragmatic) on labels not fixed by construction; its VPD side is already read, so one job settles
   the comparison. Report the three instruments beside S, r and the ratio against the graded carry column.
2. **A after C, not instead of it.** A's value is the fail case; its tie case is cheap to describe in words. Propose it
   with its gates once C has read, framed "does VPD match the cheap methods on the by-construction anchors", so the
   brief's anchor item is met at the lowest spend (prefer the $0 `normal` route if the rehearsal fits).
3. **Harder counting set only with open labels**: same base LOW, same task, label by checks (the WN spurious-row states
   are the existing instance; C1 already read them, n small, drift to "neither" by 2K). Not P from NEVER.

**What would change this.** If the user wants the by-construction VPD result on record now (a write-up deadline, or
the Qwen line needs the port), run A first. If porting pruning and reverse fine-tuning to word maps proves ill-posed
(no clean "old behaviour" to revive) or over a day, fall back to A. If E58's Amendment 2 line overturns reading A, C's
question changes and should be re-asked.

## Who decides
- **Lead may act (inside the adopted line):** C (the regating plan offers the states "note only"; the brief's Stage 1
  says score VPD and at least one non-VPD method against the ground truth; class 1/4), with FYI and a `/design` check
  written; D as a dated `decisions.md` row; the option-3 note on steps labellers in the next scoring table.
- **User:** any spend or `normal`-QoS relaxation for A (gated, $10-30 or rule change); B as a new task family or second
  base model; any change to the reuse line's definition (the carry label as a scored target would be a departure from
  the 2026-10-30 E58 reading row).
