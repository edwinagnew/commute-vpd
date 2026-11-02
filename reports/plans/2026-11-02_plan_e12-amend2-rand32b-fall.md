# Plan: E12 Amendment 2, why the re-switching term falls on rand32b's students

*2026-11-02, coding agent, for the lead. Coded, not run. Predictions `experiments/new_vs_existing/results/predictions_amend2.json`; code
`experiments/new_vs_existing/amend2.py`; job `cluster/new_vs_existing_amend2.sbatch`. Class 1 and 2 of `/design` (analysis, replication, a
control); no sign-off needed. Amendment 1's filing said "nothing further runs": the lead departs from that for this anomaly only (review M5), FYI to the user.*

We would retrain six students of Amendment 1's and ask three plain things about the fall. Does it survive another batch order? Does Γ switch on the
same subcomponents on a student's activations as on its fine-tune's? And when we write the fine-tune's values back into the student's at the
trigger, which values give the fall back? One free debug job, about 14 minutes on an L40.

## The request in the Terms' words

On the two rand32b fine-tunes and the two rand32 fine-tunes, run teacher training as in Amendment 1 (component training, loss KL(teacher ‖ student), teacher =
the fine-tuned model, Γ and Δ frozen, Adam 1e-5, 1000 steps). Read the student's own state, Δ fixed, from its fp32 weights. On rand32b the
re-switching term L01 − L00 falls by 1.02 and 1.22 nats against the do-nothing read. Why, and do students switch the same subcomponents?

## Design check

**Goal.** `GOALS.md`'s second branch asks if the class difference survives a simple map. Amendment 1 says Γ on a student's activations gives about the same
term as on the fine-tune's, except here, and it never read sets (review H2). **Nearest tried.** Amendment 1 (shares, not sets; no second batch order on this table) and
E12 (sets on fine-tunes, none on students). New: a student's set, and a batch-order reading on rand32b. **Patch or step.** A patch: an anomaly run down, as the
user asks (a standing rule, 2026-11-02). **Decisions.** Relies on Γ frozen and reading A; no departure from `decisions.md`. **What would change what we do.** "Gone" makes
the fall noise. "Few" names subcomponents to read. "Diffuse" closes it as a spread lowering of values. "Same" allows a claim Amendment 1's report cannot now make. No outcome changes Amendment 1's scored
verdicts. **Control first.** Stage 0 without training: W₀ cells and each fine-tune's L01 against Amendment 1. Then rand32b s0 and rand32 s0 must reproduce their L01, or the job stops, before anything else runs.
The recipe is Amendment 1's own code and config: same card (a stop), batch order, teacher and fresh Adam. The one difference from Amendment 1 is what ran earlier in the job. Training seeds its own generator and starts from zero offsets, so that is what the control tests.

## What the brief asked and what was cut

| reading | kept as | reason |
|---|---|---|
| reproduce the falls from fp32 | control, stops | filing's rule; retrained, so the bf16 question does not arise and is not measured |
| second batch order | P1, both rand32b seeds | review M1. Both seeds cost 3.5 min and also cross batch sharing with the teacher, which order 0 could not |
| sets at the trigger | P2 | review H2 |
| how much each change accounts for | P3 (blocks), P4 (a ladder) | P3 answers lost members or lower values, P4 the report's "few or diffuse" |
| weight distance | P5, small | zero extra compute. It is the one direct test of the report's reach reading |

Cut: every copy table, Amendment 1's state (b), C1, least-squares reads, E42, per-layer counts, per-key sets, a rank by effect, and restoring by position past the one trigger-or-not number.

## What runs, in order

1. **Stage 0, no training.** Load, W₀ cells (`Job.task`). For rand32b and rand32, s0 and s1: Γ's values on the fine-tune's activations, its L01, its switched-on set. The set anchors. The ladder's control. About 1 min.
2. **The control.** rand32b s0, rand32 s0, rand32b s1, rand32 s1 at batch order 0. Each stops the job if its L01 misses Amendment 1's by more than 0.05 nats. About 115 s each.
3. **Batch order.** rand32b s0 and s1 at batch order 1. About 115 s each.
4. **Score**, each number in the log.

Six students at about 115 s (Amendment 1's log: training plus read), plus load and reads of about 2 min: **about 14 min, `debug` QoS, one L40, $0**, limit 30 min, `--mem=32G` as Amendment 1, whose peak was not recorded. The route is the cluster, as under 30 min.
Writes as it goes: `results/new_vs_existing_amend2.json` after each state; `results/large/amend2/` holds offsets (60 MB each, unread here, since `Job.train` writes them) and Γ's values at the trigger, per window, fine-tune and student, for any later reading without a rerun.

## The numbers

Fall, J and restore are defined in the filing's `measurements`. Fall = L01(fine-tune's values) − L01(student's values). Set = alive subcomponents with mean trigger value up by over 0.3 from W₀'s (E12's rule). J = Jaccard of two sets. Restore = nats given back by writing the fine-tune's values at the trigger into the student's pattern.

| number | at W₀ | do-nothing | control | student, filed |
|---|---|---|---|---|
| fall, nats | 0 by code | 0 by code | rand32b s0 1.225, s1 1.023; rand32 s0 0.076, s1 0.084, within 0.05 | order 1: at least 0.5, by the lean |
| J to the fine-tune's set | nan (both empty) | 1 by code | J seeds near E12's 0.655; J between tables below it | rand32 at least 0.7; rand32b 0.5-1, a guess |
| set size | 0 | the fine-tune's | rand32b 27, 21; rand32 41, 51 (E12), within a few | |
| nats back at the trigger | 0 | 0 | restoring everything everywhere = the fall, within 1e-6 | at least half of the fall, by the lean |
| ladder on the term, W₀ to fine-tune | not read | not read | label 'few' or 'middling', low confidence | |
| distance | nan, 0/0 | 1 by code | 0 for a state against itself, by code | 0.3-1.5, a guess |

## Predictions, in one line each, with what each is blind to

- **P1, batch order.** Both order-1 falls ≥ 0.5 nats: holds; both |fall| ≤ 0.15: gone; else varies. Lean holds. *Blind to* more than one extra order. The noise here fills M1's gap.
- **P2, sets.** J ≥ J between seeds on both: same; J ≤ J between tables on both: different; else partly; anchors inverted: no power. Beside it, rand32b against rand32 with the batch-order change as noise: lower, higher or no different. Lean partly. *Blind to* values that fall without crossing 0.3, and to per-key subcomponents. rand32's teacher is half-learned, so the reference differs in that too.
- **P3, which block.** Dropped, kept, added or outside, the one giving back ≥ half of the fall on both states; else spread or mixed; at the trigger under half: not at the trigger. Lean kept. *Blind to* restore order, and to values at other positions beyond that one label.
- **P4, how many.** The smallest k of 1, 3, 10, 30, 100, 300 giving back half: ≤ 10 few, 30 middling, ≥ 100 or none diffuse. Lean diffuse. *Blind to* effect per unit of value: it ranks by change, so a small change with a large effect arrives late. Its control is the same ladder on the term itself.
- **P5, distance.** E12's separation rule between tables on student-to-teacher distance over teacher size: further, nearer or no separation. Lean no separation. *Blind to* direction and to what a student cannot express with Δ fixed.

Lines are sourced in the filing's `sources_of_lines`, from Amendment 1's measured noise and E12's set sizes. The one line not measured is "half". Edge cases ran before filing: `~/commute-data/scratch/commute-lead/e12a2-writer/edge.py`, 59 cases.

## Risks

- **Amendment 1's file.** `amend1.dump` writes to its module's `OUT`, so importing it and running `Job.task` would overwrite Amendment 1's scored results. `amend2.py` sets `amend1.OUT` at import; `stub.py` beside the edge cases asserted it. A reviewer should look at this first.
- **A stop on rand32b.** If a same-order retrain misses by over 0.05 nats, this table's term is more fragile than any measured, and that itself answers part of the question. It stops at about 3 min, $0.
- **Not run on a GPU.** Bookkeeping ran on made-up tensors with a stub read (`stub.py`); no model was loaded. The scored run is its own smoke, and the first student comes 3 min in.

## Not run

As the filing's `not_run`: the copy tables, a rank by effect, restoring by position, per-key values, a third batch order, other steps or learning rates, the bf16 rounding, any harness of faithfulness.
