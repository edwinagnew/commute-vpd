# Review: is E76's "B works" right, and what is it evidence for?

*2026-11-04, independent check. I read the code, recomputed from the result files and recomputed the saved offsets on the CPU worker. No model forward and no GPU job.*

**Plain words.** "B works" and arm L's "works" are what the filed rules give, every quoted number recomputes, and I found no bug. The label says much less than its name, because the rule only asks that the masked term end below the clean-only arm's, and B's arm would have earned it at step 100. What the run does show is that on unseen maths rows, with the original model's masks frozen, training can bring the masked forward from 1.33 to 0.28 nats of the fine-tuned model. That is still 6.6 times the 0.042 the original decomposition reads, so it is progress on `GOALS.md`'s D1 and not yet a faithful decomposition. I would not write "D1 has its map" from this.

Suggested sentence: "On u1, with Γ's masks frozen on W0, adding a masked term to teacher training lowers KL(FT clean ‖ student masked) on eight unseen maths rows from 1.33 to 0.28 nats, against 1.13 for the clean term alone and 0.042 for W0's own masked KL. The change's squared size is 3.4 times δW's and still growing, and it sits mostly late. Top-token agreement, FineWeb rows, Γ on the student and the adversary are not read."

## Answers to the three questions

| number | rows | trained on? |
|---|---|---|
| (i), (ii), (iv), and the amendments' reads | maths index 0-7, first 256 tokens, one fixed mask draw per row (`common.py:104`, `arms.py:85-97`, `amend1.py:64`) | No. Training is maths index 32-63 and FineWeb index 0-31 (`common.py:104,110-111`). No row is equal; see finding 7 for shared text |
| (iii), all size numbers | none: weights only | not applicable |
| `train[].clean`, `train[].masked` in the arm files | training rows, fresh masks, TF32 | Yes, and these are the only training-row numbers. No quoted number uses them |

So a held-out read exists and it is the one quoted, on maths rows only. **Rule (iii) cannot see size**: scaling a change by any factor leaves it unchanged. **The code's rule matches the filed rule** on every clause and edge case I ran (finding 1). No filed rule reads weight size, though the filed question says "of its size" (finding 3).

## Findings

1. **Confirmed: the verdicts and the arithmetic.** Re-running `rules.score` on the arm files gives `score.json` exactly: B works, L works, A "(i)-(iii) hold". Margins:
   - B reads (i) 0.0243 against 0.09, (ii) 0.624 against 0.94, (iii) 2.723 against 3.
   - B's (iv) is below A's by 0.847, and L's by 0.738, against the band of 0.00704.

   All your quoted numbers match to their printed digits: finals, sizes, and the amendment shares. One nit: the last block is 23.6 times δW's, not the plan's 23.5 (plan line 98). The rule tests pass, 18 of 18, including zero denominators, NaN, ties and stopped arms. One mismatch never fired: `rules.py:99` lets "A short" outrank "B works", where the plan reads B against A in that case.

2. **Caveat, the largest: the (iv) clause is relative to arm A and has no bar on the value (`rules.py:75`, plan line 67).** It moves what "works" means, not any number.
   - Applying the same rule at every reading, both arms read "works" from step 100 onward. B's (iv) there is 0.97, which is 23 times W0's own masked KL.
   - The yardstick in `GOALS.md` is that D′ scores on W + δW about as D scores on W. W0's masked KL on these rows at seed 74 is 0.0423 (`stage0.json`). B reads 6.6 times that and L 9.2 times.
   - (i) and (ii) cannot fail once training runs. The 0.09 is the worse half of δW itself (E72), and A passes it by about step 100. (ii) is pinned by (i): when the student nears FT, it reads δW's own value, 0.619 on these same 8 rows (sum of `K_clean` in E77's `route_stochastic_k196.json`), and B reads 0.624.
   - So the label rests on (iii) and on a relative clause that any fall passes.

3. **Caveat: size is outside every rule, and the end-of-run size depends on where the run stopped.** The filed label does not move. The constant named `SIZE` in `rules.py:10` is a KL.
   - I recomputed the sizes from the saved offsets and they match: A 1.795, B 11.92, L 8.21, in both fp32 and bf16.
   - Size has no plateau: B's squared size grows about 10 per 1000 steps to the end, while (iv) still falls about 0.012 per 100 steps. The plan doubled the run from 600 to 1200 steps.
   - At step 300 B's size is 2.71, under δW's, and 73% of the fall is already there. But its last block is already 6 times δW's and blocks 21-27 are 2.1 times.
   - If "of its size" had meant squared weight size within 1.5 times δW's, B and L would miss at step 1200. They would pass up to step 500 (size 4.84, (iv) 0.419) and step 750 (size 5.12, (iv) 0.476). So most of the excess bought little: after step 500 B gains 0.14 nats for 7 more of squared size.

4. **Caveat: the size numbers overstate how far the columns moved, and squared size is a weak ruler here.**
   - In norm, B is 1.85 times δW's overall and 4.9 times in the last block.
   - Against W0's norm, B's change is 0.26% (δW 0.14%), and 0.64% in the last block (δW 0.13%).
   - Columns keep their direction: the 5th-percentile cosine of V′ with V is 0.99995 in block 27, 0.99998 overall.
   - Control C's teacher has squared size 262, 75 times δW's, and moves clean KL by only 0.01. δW at 3.49 moves it by 1.84. So the same squared size can mean very different behaviour.
   - None of this shows the components still mean the same thing: a very small rotation is enough to produce a 1-nat fall.

5. **Artefact: much of the size is Adam's walk, not a target.** This bears on A's "0.52 times" and on all early-block sizes.
   - Under pure noise, Adam's offsets reach about lr √t in rms: 3.46e-5 at 1200 steps, 1.73e-5 at 300.
   - A reads 3.71e-5 and C reads 1.76e-5, which is what a walk gives.
   - B reads 9.28e-5 overall: 4.08e-5 in blocks 0-14, about the walk, and 1.85e-4 in the last block, 5.3 times it. So B's late excess is real signal, while its early size may be mostly walk.
   - δW has the same character: rms about 5e-5 at lr 1e-5 over 10 steps, and flat across blocks (largest over smallest 1.15).
   - A and B share a batch order. B's change contains about 0.96 of A's (dot product over A's squared size), plus an extra part orthogonal to it of about 10 in squared size. The shared batches may explain part of that overlap.

6. **Artefact: rule (iii)'s reading near 2.7 does not come from u1.**
   - C's teacher has late share 0.256 (ratio 1.03), and C's student still reads (iii) 2.75. At step 300 B reads 2.74 and C 2.75. Filed for C: "near arm A's", which is 1.88, so that expected value was missed. It had no stop rule and I found no report of it.
   - So the masked term with Adam pushes size late whatever the teacher.
   - δW's share of 0.248 is simply 7 of 28 blocks, because Adam moved every block alike, so the ratio's denominator says nothing about where the update matters.
   - The threshold 3 has no stated source. The ratio's ceiling is 4.03, so 3 means at most 74% late. B's margin is 0.28 and L's 0.58. A threshold of 2.5 would have flipped the outcome to "B fails, L works".
   - The rule is filed and stands, but it gives weak support to "placed like the update".

7. **Caveat: "unseen" holds by row, not fully by text, and the 8 rows are uneven.**
   - 14.6% of reading positions start a 16-token run found verbatim in the training rows, which is the prompt template; for 32-token runs it is 8.9%. The column cut also used g on these 8 rows (`stage0.py:206`), which is a small leak.
   - Row 5 is the only row with no chat-template token and no shared 16-token run. It reads W0 (i) 0.186 against 1.76-2.52 for the rest. Its (iv) goes 0.213 to 0.091, a 57% fall, against 77-81% on the others, and that is 3.9 times its own masked KL of 0.023.
   - B is below A on 8 of 8 rows, so the direction holds there too. Row 5 is a single row, but it is the cleanest held-out read available.
   - A row-level fit on user-turn positions gives 60% of W0's (iv) in user turns, 28% of positions, but R² is only about 0.5 with 8 rows, so that is a hint. Per-position KL was not saved. E73 already put u1's KL mostly on positions where W0 is confident, and on user-turn text the fine-tune's loss never targeted.
   - No FineWeb row is read at all, though half of each training batch is FineWeb.
   - Training terms read lower than the held-out reads: B clean 0.0069 against 0.0243, masked 0.177 against 0.280. They mix FineWeb, TF32 and fresh masks, so this is not a clean measure of a generalisation gap.

8. **Caveat: KL(FT ‖ ·) rewards softness, and top-token agreement was not read.** FT is softer than W0, which is why masked W0 (1.33) already sits nearer FT than clean W0 (1.84). E72's best single temperature leaves 0.82 on u1, on its own 128 rows. So B's 0.28 goes below what one temperature gives, but the earlier part of the fall could be softening. E72 named top-token agreement as the read a temperature leaves alone, and E76 does not have it. The entropy of the student's masked forward is unread too.

9. **Caveat: what the student is.** Its weights are W0 + change, and they are not close to W0 + δW.
   - Cosine with δW: A 0.106, L 0.062, B 0.045.
   - Squared distance to FT's weights: A 4.75, L 11.04, B 14.83, against 3.49 for W0 itself. Every arm ends further from FT in weights than W0 started.
   - This is teacher training as defined, and the (i) blind spot "a re-implementation" is filed. But `GOALS.md` asks for a decomposition "of W + δW", so the sentence should say the student matches FT's outputs, not its weights.
   - Γ, Δ and the masks are frozen on W0, so (iv) reads a mix that no deployed state would run. The plan cut the student under its own masks because E72 read 0.103-0.105 for every state, but B and L are new states, so that reason no longer applies to them.
   - Only 39,105 columns move; they hold 0.835 of Σg.

10. **Artefact, in how Amendments 1 and 2 are quoted: the shares are not a split of the fall.** Each is "this part alone", so they overlap: blocks 0-14, 15-20 and 21-27 sum to 1.44 for B, 1.23 for L and 1.03 for A.
    - Taking parts away gives a different picture for B: removing blocks 21-27 loses 39% of the fall, removing blocks 0-20 loses 24%, and the remaining 37% either part can supply.
    - "The fall sits late" is therefore too strong. It sits mostly in 15-27, with real redundancy.
    - Clean KL shows the same: B's blocks 0-20 alone read (i) 0.12, blocks 21-27 alone 0.73.

11. **Caveat: Amendment 2's comparison with E77 is not like for like.** E77 reads how much KL a change passes, which is second order in the shift. A fall in (iv) is first order, against an existing gap of 1.33 nats. As a rough order of size, by treating KL as a squared distance, the 0.18 fall from blocks 0-14 needs a masked shift of only about 0.0065 nats. That is about 0.03 of that part's clean KL (0.21) against E77's τ of 0.001-0.004, and it is a trained direction, not the update's own output. So the lean's miss (0.17 against 0.1) does not contradict "small changes barely pass", and the narrowing of E77's wording need only be that a trained direction passes several times more.

12. **Caveat: the controls test the gradient, not the size of the fall.** C's fall of 0.007 (from 0.055) shows the masked term trains. It says nothing about a 1-nat fall, or whether that fall is specific to u1. C ended at 0.048, still above W0's own 0.042. The band is a fall on another teacher and not a noise floor, but the 120-fold margin makes that harmless. No second training seed or second reading draw exists. At that margin the sign is safe, while the exact 0.28 has no error bar.

13. **Caveat: u1 is one large update.** It is 10 steps of SFT at 1e-5 with no warm-up, reading 1.64 nats from W0, and E73's advice calls gold fine-tunes on these rows a format change read on text FT would not write. "B works" is on this update alone, as the plan says.

14. **Confirmed: no bug in the parts that could produce the fall.**
    - The masked hook (`qwen_masked.py:60-66`): I expanded ((x V′)(m − d))U′ + d x W′ᵀ by hand and it equals the code's terms. Stage 0 reads an exact 0 logit gap at zero offsets.
    - Orientation of `change` and `F.linear`; column indices through `halves` and `on`; row-to-g alignment (`run.py:74`, `amend1.py:65`).
    - Reading masks are reseeded per reading (`arms.py:85`), and all arms read identical step-0 rows.
    - Readings are fp32. Dtype floor 2.3e-6; TF32 against fp32 on the masked term differs by 6e-5 of 1.26.
    - δW's sha256 matches. It holds exactly the 196 sites and no other tensor.
    - L's resume at step 1050 restores step, Adam and generators, and its curve is smooth across it.
    - Amendments: W0 reproduces step 0 bitwise; the bf16 "all" differs from the fp32 finals by about 1e-6; part sizes sum to the whole.
    - L's (iv) is all-masked at reading, so routed-out sites cannot carry the update around the masks in that number, which was the plan's concern. It was trained under routing, though, so the number is not read on its own training route.

## What this does to the next step

The plan sends "B works" to RL's step 100 on the paid card. Before that, one $0 read on an L40, of about the same length as the amendments, using the saved offsets:
- top-token agreement and entropy per position for A, B, L, masked and clean, against FT;
- the same on FineWeb evaluation rows and on response positions alone;
- Γ on the student's forward (the student under its own masks).

It would say whether 0.28 is agreement or softness, and whether it survives Γ. It would be a new E-code, since Amendment 2 was E76's last. A null is also missing: B's recipe with a teacher that is a random change of equal clean KL.

## Not checked

- Anything needing a forward: per-position KL, top-token agreement, entropy, Γ on the student, adversary, FineWeb reads, accuracy. No GPU job was allowed, and these numbers were not saved.
- What the late change does, or whether components still mean what they did. Cosines of columns are the only read.
- Other seeds, other mask draws on the reading rows, or a lower rate.
- Whether Adam's walk truly accounts for A's and C's size. I compared rms with lr √t only, with no direct separation of signal from walk.
- The stage-0 hook values against E77 beyond the digits in `stage0.json`.
- E72's temperature floor on these 8 rows; it is from its 128.
- The rough 0.0065 figure in finding 11 is an order of size, not a measurement.
- The random-change size was recomputed with a CPU generator, so it is the same distribution as C's teacher but not the same draw.
