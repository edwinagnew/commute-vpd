# Plan: E72, a second test for D1 after E68's ceiling

*2026-11-03, D1 blocked, day 1 of 8. Fable's advice, taken except where marked: `reports/plans/2026-11-03_advice_e72-d1-test-FABLE.md`. Under the gate: $0 on a cluster L40, no pre-run review. Measures only: it does not change D1's test, which is the user's call (log card `decision-d1-test`).*

**In plain words.** Under the original model's masks even the whole update barely works, so that test cannot pass anything. This job asks whether a different ruler has room: masks taken once from the fine-tuned model and held fixed for every state, with a floor and a ceiling both measured. It also reads the update's parts in the clean model, with no masks at all.

## Design check

- **Goal.** D1: a map with a test that can fail a bad placement and pass a good one. E68 has the first half only.
- **Nearest tried.** E68's reading A let each state make its own masks, and read 0.24 of the gain whatever was placed. Holding one set fixed is a different test and has not been read. E44 Amendments 1 and 1b showed Γ's masks follow the activations they are given.
- **Patch or step.** A step: new E-code, E68's amendments are spent.
- **Decisions.** `decisions.md` of 2026-10-29: Δ's share is a column, never an automatic fail.
- **What changes next.** See the tree.
- **Control first.** Do-nothing under fixed masks must reproduce E68's reading A on u1 (K 0.1044, gain 0.240), then the whole update, with a stop if the ceiling has no room.

**Override of Fable.** Its fallback sentence says the fine-tune "acts through which subcomponents Γ switches on, not through what they compute". That is not shown: masks made from the fine-tuned model's activations carry its behaviour whether or not the model re-gates (E44 Amendments 1, 1b), and the elicitation limit applies. No write-up uses that sentence unless a read separates the two. T3 is not run here.

## What runs

Updates in order u1 (E40 step 10), u2 (gold step 100), u3 (template step 100), rows and files as E68. Stage S as E68, its files on the node.

**T1, fixed masks.** g from Γ on the fine-tuned model's clean forward with the original components, per row, the same g for every state. Masked forward with the state's U′, V′, Δ off. States: do-nothing (floor), whole update unmasked (ceiling, `forward_plus`), often-on half, rarely-on half, matched random at E68's scale, least squares, adjoint.

**T2, no masks, clean model.** W0 plus each part: the least-squares split's in-span part, its remainder, the often-on half's part, the rarely-on half's part, the random change at E68's scale. FT and W0 are the ends by definition.

| number | W0 | do-nothing | positive control | blind to |
|---|---|---|---|---|
| gain kept, T1 | not read | 0.240 on u1 | ceiling, to measure | Γ's response to the state; Δ, off |
| K, T1 | not read | 0.1044 on u1 | ceiling, to measure, below 0.1044 expected | whether the difference is confidence or behaviour |
| top-token agreement with FT, all positions | read | to measure | ceiling to measure; FT = 1 | anything but the top token |
| mean entropy | read | not read | FT read | where it falls |
| gain kept, T2 | 0 | not read | FT = 1 by definition | Γ entirely: a weight-space test, not faithfulness |

## Rules (filed in `experiments/qwen_fixed_masks/results/predictions.json` before any run)

Gain is E68's `gain_kept` formula, with masked W0 at W0's masks as its zero, so that the floor is the 0.24 read. w is the 95% row bootstrap half-width (seed 0, 2000 draws) of gain(upper) − gain(lower), per update.

- `control_reproduce`: u1 do-nothing K within 1e-3 relative of E68's A_do_nothing, gain within 1e-3 absolute; else stop. Same forward and rows as E68's; only where g is stored differs.
- `room`: per update, gain(ceiling) − gain(do-nothing) > 2w and > 0.1. Else "no room": on u1 the job stops, and the fallback goes to the user. Expected: room, ceiling gain ≥ 0.5, medium lean.
- `teeth`: gain(upper) − max(gain(lower), gain(random)) > 2w. Expected: holds, low lean, E68's own-mask halves did not separate.
- `label`, per split, where room and teeth hold: "passes" if gain ≥ gain(ceiling) − w; "fails" if gain ≤ max(gain(do-nothing), gain(lower), gain(random)) + w; if both, "fails"; else "partial". Expected: least squares partial.
- `T2`: per update, with s the in-span share of the update (stage S, 1 − Δ's share). "carried" if gain(in-span) − s > 2w and gain(remainder) < gain(in-span); "by norm" if |gain(in-span) − s| ≤ 2w; else "not carried". Clean states are always exact: gains can pass 1 or fall below 0, read as they come. Expected: no lean.
- Edges: gain's denominator below E68's 0.05 nats: unscored. NaN: unreadable.

## Tree (for the lead's reading; D1's test stays the user's call)

- No room on u1: a fixed-mask test cannot pass anything either. Say so at once, with the fallback statement restated without Fable's cause.
- Room, teeth, least squares passes: least squares is the candidate; adversary three seeds on it, Δ's 0.36 to the user.
- Room, teeth, least squares partial: teacher training under these masks has a target. Its masked student is new code, a gated run, its own plan.
- Room, no teeth: test cannot fail; report, no further test on this ruler.

## Cost and saves

$0, one L40, `normal` QoS if the queue is empty at submission. Per update S ~20 min, 7 reads + 5 clean reads ~30 min; ~2.5 h. Results JSON after every read, every number in `results/run.log`, as E68. Stage S files stay on the node.

## Not run

u4; t sweep; K(W0 ‖ masked at t); maths accuracy; teacher training; the adversary; teacher-forced inputs (Fable's optional read separating routing from input starvation).
