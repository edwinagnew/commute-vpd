# E34 Amendment 9: does a student's change point along what the pieces could express of the update?

Class 1 (analysis of what exists): no sign-off needed. $0, one `cpu-run` job, about 15 min, no model and no forward. Option D of
`reports/plans/2026-11-02_plan_commute-next-draft.md`, whose claims were checked here; two of them do not hold as written (below).
Rules and expected values: `experiments/teacher_training/results/predictions_amend9.json`. Code: `experiments/teacher_training/amend9.py`.

## The request, in the Terms' words

For each saved student of a teacher training on the 67M, take its weight change s = (U + dU)(V + dV)ᵀ − UVᵀ from W₀'s alive (U, V).
Take its fine-tune's finished update δW, and the inside part I(δW) = δW − O(δW), O(δW) being the least-squares split's remainder,
(I − P_U) δW (I − P_V) per matrix. Read cos(s, I(δW)), cos(s, δW) and the inside share ‖I(δW)‖² / ‖δW‖², with inner products over the
24 matrices together, in fp64.

## Goal

`GOALS.md` asks whether a small update can be carried past the decomposition so that it stays interpretable along training. Teacher
training is the commute line's map, and how we describe what it does decides what a pass means. This run decides between two
wordings: "copies the fine-tune into the subcomponents" and "reproduces its behaviour by a change the pieces can express".

## Design check

- **Goal served.** The wording of the commute line's headline claim. It also bears on the size card (`decision-chain-size`): a size
  read against what the pieces can express. It gives the Qwen lead a reference; nothing runs on Qwen.
- **Nearest tried.** E34 Amendment 8c, `experiments/TRIED.md`: cos(s, δW) only, 0.27 to 0.28. E12 Amendment 2's P5: the same cosine on
  map tables, 0.30 to 0.41. Neither divides out what the pieces cannot write. A grep of `TRIED.md`, every report, and
  baseline-lead's and redteam-lead's trees found the reference only in the Option D draft. No reading exists.
- **Patch or step.** A patch: an open reading on existing states, named unrun in Amendment 8's "What next" (item 3).
- **Decisions relied on.** `split.py`'s tolerance 1e-6, the least-squares remainder as defined in the Terms. No departure.
- **What changes next.** P1 'copies' or 'expressible change' picks the wording. P3 'outside counts' goes to the user: the premise is
  wrong by more than noise. P2 'no power': the question goes back unanswered.
- **Control, first and alone.** On each of the six fine-tunes, before any student: the least-squares split's first-order change reads
  cosine 1 to the inside part, and the do-nothing's change reads √share. A miss stops the job. No reference procedure is reproduced.
- **Numbers and what each should read.** Table below; `predictions_amend9.json` holds the rest.

## What the algebra already says

O is an orthogonal projector, so

cos(s, δW) = cos(s, I(δW)) · √share + out_term, where out_term = ⟨O(s), O(δW)⟩ / (‖s‖ ‖δW‖).

Three things follow. They change how the result will read, so they are stated before it.

1. **The share bounds the second number, not the first.** Option D says the inside share "bounds the first", cos(s, I(δW)). It bounds
   cos(s, δW), at √share, for a student wholly inside. The bound on cos(s, I(δW)) is 1. So "near the bound" has one honest form:
   cos(s, I(δW)) against 1. A rule on cos(s, δW)/√share would be the same number when out_term is 0, and is not filed.
2. **The student is not only in the span.** Option D says the change "can lie only in the subcomponents' span". dU Vᵀ + U dVᵀ does, and
   dU dVᵀ does not: (I − P_U) dU dVᵀ (I − P_V) is non-zero in general. A test on random matrices shows it. So the student's own inside
   share is read, and P3 tests whether the identity's last term matters. The same term means the least-squares state reads cosine 1
   only to first order: the control is the first-order change, and the whole state is reported with no stop.
3. **The wording is close to decided by arithmetic.** With out_term 0, 'mostly along' (cos² ≥ 0.5) needs share ≤ 2 cos²(s, δW). The filed
   cosines give share ≤ 0.15 on Amendment 8's real student and ≤ 0.34 on the highest map-table student. The isotropic update's inside
   share from the alive ranks is 0.61, and E2's fine-tunes read about 0.45 inside on the 20 matrices not saturated, with the 4 saturated
   ones wholly inside. So P1 reads 'copies' only if these fine-tunes have far less inside than any earlier reading. The run's open
   numbers are the share, out_term and the student's own inside share.

## Numbers

Share in the table is the lean, 0.5 to 0.8, medium-low; it has no earlier reading on these fine-tunes in this form.

| Number | W₀ | Do-nothing | Control (least-squares, first order) | Null student on the real δW | Real students, own δW |
|---|---|---|---|---|---|
| cos(s, I(δW)) | nan, refused | √share | 1 within 1e-6, stop | 0.05 to 0.07 | real 0.31 to 0.40; rand32 0.33 to 0.45; rand32b 0.44 to 0.58 |
| cos(s, δW) | nan, refused | 1 | √share | 0.047 to 0.048, filed | real 0.274, 0.281; rand32 0.298, 0.320; rand32b 0.391 to 0.412, filed |
| share of δW inside | as δW | as δW | as δW | as δW | 0.5 to 0.8 |
| student's own inside share | nan | share | 1 | 0.85 to 1, low | 0.85 to 1, low |
| out_term | nan | 1 − share | 0 | under 1e-3 | under 1e-3, medium-low |

The size against the inside part, ‖s‖ / ‖I(δW)‖, is reported for the size card: real about 0.8 to 1.0, rand32 2.6 to 3.2, rand32b 1.7
to 2.2. Multiplied by cos(s, I(δW)), it says how far the student reaches along the inside part, where 1 is all of it.

## States

| Family | Students | Fine-tunes |
|---|---|---|
| E12 Amendment 2 | `new_vs_existing/results/large/amend2/{rand32_s0, rand32_s1, rand32b_s0, rand32b_s0_o1, rand32b_s1, rand32b_s1_o1}.pt` (bf16 offsets) | `regating_table/results/large/{rand32,rand32b}_{s0,s1}.pt` (fp32) |
| E34 Amendment 8c | `teacher_training/results/large/amend8/amend8c/{real_o0, real_o1, null_o0, null_o1}.pt` (bf16 offsets) | same folder, `{real,null}_teacher_dW.pt` (bf16) |

Each reads its own fine-tune and one more: E12's the other table at the same seed, Amendment 8c's the other of its two.

**Not saved, left out.** E42's fine-tunes. Its students exist (`e42/{rand32,ident32,rand1}/{A,B}_step1000.pt`), but each step-1000
fine-tune was regenerated in-job. `kv_task/results/large/rand32/` holds step 10 only, and `regating_table`'s `rand32_s0` is a
first-learned-step fine-tune. Option D names E42; it cannot be read without training. Amendment 8's n1, n2, n4, n5 and moved students
have a moved start or no fine-tune to be a reference.

**Layout and alive set, checked on keys and shapes.** All 24 matrices in every file. Every δW is (d_out, d_in) of `uv0.pt`, every offset
is `uv0.pt`'s column count, in the layout `offsets_state` asserts. Column counts equal `transport_test/alive.pt`, the file E12's job used.
E12 saved no U, V of its own, so `uv0.pt` (Amendment 8c's) stands in. The re-read control tests that: a wrong U, V would not reproduce
E12's filed fp32 cosine, and the job stops at 1e-3. Inner products use E12 Amendment 2's `change_stats`, imported, over all matrices
together in fp64.

## Noise

Offsets are bf16 for all ten students; Amendment 8c's δWs are bf16, E12's are fp32. Measured effect on a cosine: 6.4e-7 on 8c's
`rebuilt_cos_real`, and here each student's re-read against its filed fp32 cosine. The arithmetic worst case, every rounding aligned with
the other vector, is 2 × 2⁻⁸ = 0.008, and rounding errors are near-orthogonal to a fixed direction. Amendment 3's reload difference
(0.004 nats at most, on the term) is not a cosine and is not used. Cosine differences under 1e-3 are not read. Batch order moves these
cosines by 0.007 to 0.02 (the order pairs' filed values), which is the larger noise.

## Blind spots

Frobenius: every weight direction counts alike, so a student matching the inside part on all directions inputs excite and differing on
the rest reads 'mostly elsewhere'. That is the case the second wording would need to rule out and this run cannot; an input-weighted
form needs moments from a forward and is not run. Size is not in any cosine. One null fine-tune, one label draw.

## Cost

Tensor arithmetic, so CPU, no sbatch needed: `cpu-run -c 4 --mem 16G`. SVDs of 48 matrices, then projections for six fine-tunes and ten
students: about 15 min, up to 30. No money. The dev node runs nothing.

## Departures from the literature

None: no cited procedure is reproduced. The cosine is the project's own, as in E12 Amendment 2 and Amendment 8c. The one departure from
the brief is in point 2 above (control read to first order).

## Not run

E42's students; per-matrix cosines ($0 later); input-weighted cosines and shares; the other seed's fine-tune as a reference; anything on
Qwen; any comparison with cheaper methods (baseline-lead's line).
