# Review: what "pull-back", "lift" and "LoRA pull-back" have meant, where they drifted, and what to call things

*2026-10-20, coding agent, at the user's request after the E13 hand-back ("we have a major terminology issue").
Four audits fed this: every report, note and backlog entry; every notebook and the research-log page; every code
path, arm name and result key; and the session history for the user's own words. Nothing here was run.*

The project's two names for its maps, "pull-back" and "lift", have each carried at least three meanings since
2026-10-14, and the user's meaning of "LoRA pull-back" was never built. Every number in the results stands for
what the code computed, because the code has only ever had three unambiguous operations, but five "LoRA
pull-back" runs answered a question the user did not ask, and about a dozen written claims are true under one
meaning and false under another. The fix is to retire both words, name the operations by what they consume and do,
and put the user's wording in `GOALS.md` with the code symbol beside each term.

## 1. The operations that exist, and the ones that do not

W = U Vᵀ + Δ per matrix; U and V hold the alive subcomponents' write and read vectors as columns. Letters are used
below and in the audits.

| Letter | Operation | Consumes | Code | Run? |
|---|---|---|---|---|
| A | Train (U, V) on the fine-tune loss, Γ and Δ frozen; Adam, many steps | loss and data | `pullback.train`, free `Offsets` | E2–E6, E10, E11, E13 |
| F | A with rank-r offsets dU = P Sᵀ, dV = Q Tᵀ | loss and data | `Offsets(rank=r)` | E7 (one matrix), E13 (24 matrices) |
| C | Write a finished δW as the least-norm (dU, dV) with dU Vᵀ + U dVᵀ = δW − O(δW); O(δW) = (I − P_U) δW (I − P_V) stays in Δ | a finished δW | `lift.euclidean_lift`, `lift.lift_state` | E2, E7–E10, E12 |
| B | Write a finished δW as dU = δW V, dV = δWᵀ U (the adjoint of do); weight change K(δW) = UUᵀδW + δWVVᵀ | a finished δW | none | never |
| D | Rescale the fixed rank-1 pieces, δW ≈ Σ α_c u_c v_cᵀ | a finished δW | `transport.py` | E1 |
| E | The geometric words: strict lift, Q-horizontal lift, path lifting, pull-back of a covector | — | — | theory only |

B and C are the two ways to split a finished update over the subcomponents. They are related: C applies the adjoint
after preconditioning by K⁺, so C = B ∘ K⁺ on the expressible part. They give different per-subcomponent splits and
different weight changes; B is one line of code and has never been computed on any δW. The code audit found no path
that applies B to a fine-tune difference or a LoRA's BA.

## 2. What the user meant, in the user's words

From the session history (dates are the sessions'):

- 2026-10-14. The user asked for "a simple operation that maps a post-training update (lets say LORA) to an update
  on parameter components". "Pull back" entered as the assistant's word in the reply: "pull the update back through
  W = Σ U_c V_cᵀ. That means applying the same training step to the components' own parameters". The same day the
  user asked for "the pull-back equation (or whatever you want to call distributing a lora update over a
  decomposition)". The artifact put distributing BA under "lift" and training the pieces under "pull-back".
- 2026-10-15. "you never ran lora through pull-back? that always struck me as the simplest thing to try." The
  assistant offered three definitions, all of them training the subcomponents with the LoRA's matrix or rank; the
  user picked two of them. That choice became the backlog's definition and the source of every later F run.
- 2026-10-19. "decompose the lora rank r matrix into a projection onto the relevant subcomponents"; "isnt that just a
  projection onto existing pieces?"; "wtf since when is that how pullback worked? i have been assuming the pullback
  as defined in the artifact"; "take the final output AB and decompose it?"; "ok fair enough i was mistaked to have
  approved that previous interpretation of lora". The plan file written that day records "the user's intended
  question is the plain one" and still built F beside C.
- 2026-10-20. "train a lora of rank r, yielding AB. pullback AB into subcomponents and update accordingly"; and in this
  session: "train a rank r lora matrix AB, possibly to completion. Then split it into a sum of updates on
  subcomponents. Study how much lies outside the span of the subcomponents."

So the user's "pull-back" has meant, consistently since 2026-10-14, "distribute a finished update over the
subcomponents and see what is left outside". The repo's "pull-back" has meant A since the theory note of
2026-10-14 (section 5.3, "what 'pull back' meant in chat", which records the assistant's chain-rule sense, not a
user sentence) and was fixed as A in `GOALS.md` Terms on 2026-10-19. The user's "lift" on 2026-10-14 was path
lifting (theory section 6.1); the repo's "lift" became the one-shot least-norm solve C.

## 3. The "LoRA pull-back" runs against the user's definition

The user's definition has three parts: train BA; split it into per-subcomponent updates; measure what lies outside
the span.

| Run | Trained a LoRA? | Split BA over subcomponents? | Outside share measured? | Called in the write-up |
|---|---|---|---|---|
| E2 lift test, LoRA stage | yes, one matrix | no | yes, Frobenius and input-weighted | "LoRA obstruction" |
| E7 lora_commute | yes, ranks 1–32, one matrix | yes, by C; per-piece split not saved | yes, vs rank | "lift"; the "pull-back" arms were F and A |
| E8 lora_matched | yes, rank 8, 24 matrices | yes, by C | yes | "lifted LoRA" |
| E9 lora_plateau | yes, to 1000 steps | yes, by C | yes, vs step, with a random-update null | "lift" |
| E13 lora_pullback | **no** | no | no (zero by construction) | "LoRA pull-back" |

The user's question was therefore partly answered under the name "lift" (E7, E8, E9), with the outside share
reported as a side quantity under adversarial ratios and route splits, and not answered at all by the run that
carried the user's name for it. The per-subcomponent split itself was never saved or studied. The B split was never
computed. The LoRA factors for every run are on disk (`experiments/lora_*/results/large/*_factors.pt`), so the
user's study needs no training.

The outside-share numbers that exist, pooled over the non-saturated matrices, input-weighted (1 = all outside):

| Update | Outside share | Random update of the same size |
|---|---|---|
| Fine-tune, step 10 (E8) | 0.27 | 0.53 |
| Rank-8 LoRA, steps 1 to 1000 (E9) | 0.48 to 0.71 | 0.55 to 0.76 |

Under the first-order span, a LoRA sits outside the subcomponents about as much as noise does; a fine-tune sits
well inside.

## 4. Written claims that depend on the meaning

Each is true under one letter and false or unproven under another.

1. "For a LoRA handed over as BA, only the lift applies" (`GOALS.md` Terms; `reports/decisions/2026-10-19_decision_success-criterion.md`:47;
   quoted in `reports/reviews/2026-10-19_review_lora-failure-and-fixes.md`:35; research log, Lift card). False: B applies too.
2. "The pull-back does not map a given δW (e.g. a LoRA's BA) to a component update" (`theory/commuting_updates.md`:235).
   True for A, false for the user's pull-back.
3. "The pull-back was never run on a LoRA" / "LoRA pull-back on all 24 matrices (never run)" (`README.md`:39;
   `reports/reviews/2026-10-15_review_faithfulness-scope.md`:46; `reports/analyses/2026-10-19_analysis_recruit-test.md`:10;
   research log Q3 and Record). Stale under F (E13 ran it), and never says which meaning.
4. "Projection is dead (0.06%)" (research log Q2; `notebooks/E01_transport_test.ipynb`). True for D only; the
   user has used "projection" for B and C.
5. "The pull-back is not a projection; letting Δ take the leftover each step is a lift with a worse preimage"
   (`reports/reviews/2026-10-19_review_update-map-variants.md`; decisions log). Variant (b) there is B applied per step to
   the fine-tune's gradient. It was dismissed as "not worth a run" and is the nearest the repo came to the user's map.
6. "A fine-tune cannot be lifted; a LoRA can" (research log, Result 2). C is defined for every δW; the claim is about
   adversarial fragility, and the LoRA-vs-fine-tune comparison is confounded (different steps and cards).
7. "The pull-back works on a LoRA-shaped update" / "The LoRA pull-back works" (E13's report and notebook, deleted 2026-10-20 at the
   user's request together with its directory; the E13 hand-back). F, not the user's operation.
8. Theory section 5.3 characterises the pull-back as one SGD step, −η K(∇ℓ); every run used Adam for 10 to 300 steps.
   The "weight-faithful by construction" and "kernel = obstruction" statements hold for the SGD step only.
9. Code: `pullback.Offsets.lifted()` returns the A-trained components and lifts nothing; the lift-test stage
   `projected` scores W₀ + (δW − O) and neither projects nor lifts; `lift.alive_uv` and `pullback.alive_uv` share a
   name with different signatures.

Not at risk: every result's number. The code audit found each arm computes exactly one of A, C, D or F, and the
arm labels in results JSON and notebooks match the computation. What is at risk is which question each number was
read as answering.

## 5. Proposed names

Rules: a name says what the operation consumes and what it does; the prose name is the code symbol; "pull-back",
"lift" and "projection" are retired from prose and from new code, and appear only in the Terms table as the old
names each new name replaces.

| New name | Replaces | Definition |
|---|---|---|
| **component training** | pull-back (A), pull-back replay | train the alive (U, V) on the fine-tune loss with Γ and Δ frozen |
| **rank-r component training** | LoRA-shaped pull-back, LoRA pull-back (F), pb_r | component training with rank-r offsets |
| **split** (of a finished update) | the user's pull-back; "decompose BA"; "distribute over the subcomponents" | write δW as per-subcomponent updates (dU_c, dV_c) plus a remainder outside the span |
| **least-squares split** | lift (C), Euclidean lift | the split with least-norm (dU, dV); remainder O(δW) |
| **adjoint split** | B; variant (b); "pull-back of δW" | the split dU = δW V, dV = δWᵀ U; never run |
| **rescaling split** | projection (D), T0 | δW ≈ Σ α_c u_c v_cᵀ, directions fixed |
| **do-nothing** | do-nothing | δW into Δ, subcomponents untouched |
| **LoRA split** | "LoRA pull-back" as the user meant it | train BA, split it, report the outside share and the per-subcomponent updates |

A measurement, not a method: the **outside share** replaces "obstruction fraction ω" and "inexpressible share": the
energy of δW outside the subcomponents, input-weighted, with its random-update null, reported as four blocks (inside on
both sides, write side only, read side only, outside both). Accepted by the user 2026-10-20 ("rescaling split"; "outside
share is not a decomposition method").

The user's sentence in the new words: "LoRA split: train a rank-r LoRA to completion, take the least-squares and
adjoint splits of BA over the subcomponents, and report the outside share and the per-subcomponent updates."

## 6. Fixes beyond names

- `GOALS.md` Terms is rewritten by the user, or from the user's sentences above, with four columns per term: what it
  consumes, the formula, the code symbol, the old names. The agent proposes; the user approves.
- A rule for `.claude/CLAUDE.md`: when the user's words and a file's definition of a term disagree, flag it to the user
  at once and insist on explicit acknowledgement of which reading holds, never pick a side silently (user, 2026-10-20);
  the user's words win and the file is fixed first; a plan quotes the user's request verbatim and restates it in the Terms' words
  before any arm is designed. The 2026-10-19 plan quoted the user and still built the file's definition; the rule
  has to make the restatement the thing the user confirms.
- `.claude/skills/writeup/check.py` flags "pull-back", "pullback", "lift", "lifted" and "projection" in any new
  report or notebook, the way it flags long sentences.
- Code: rename `Offsets.lifted` to `Offsets.components`; rename the lift-test stage `projected` in its docstring only
  (finished experiments stay as they ran); fix the `alive_uv` name clash on the next edit of either module. New code
  uses the new names (`component_training.py`, `split.py`); the old modules get a one-line header naming their new
  name, not a rename.
- Documents: fix the nine claims in section 4 where they live; `README.md`:39 and the research log's Q3, Record and
  backlog entries lose "never run" and gain the letter; the research log's glossary gets the table in section 5 and
  a Mistakes entry for this review; E13 is relabelled rank-8 component training.
- Theory: `theory/commuting_updates.md` section 5.3 gets a note that the runs are Adam, not the SGD step it analyses,
  and section 6.1's "the 'lift' in the user's sense" is marked as the path-lifting sense the repo did not adopt.

## 7. The user's study, in one paragraph, pending one decision

Train nothing. Load the saved LoRA factors (E7: ranks 1, 4, 8, 32 on one matrix at step 100; E8: rank 8 on 24
matrices at step 2; E9: rank 8 on 24 matrices at nine steps to 1000) and the fine-tune δW for comparison. For each,
compute both splits, save the per-subcomponent updates, and report the outside share with its random-update null,
by matrix and pooled. The decision is which span "outside" refers to: the first-order span dU Vᵀ + U dVᵀ (the
recommendation, and what the table in section 3 already uses), the span of the rank-1 pieces themselves (the
rescaling fit's span, 0.06% inside for a fine-tune), or the block inside both the write span and the read span.
Through `/design` once the span is chosen.

## Not run

The adjoint split on any δW. The least-squares split's per-subcomponent updates saved for any LoRA. The outside
share under the two alternative spans. Any check that the nine claims in section 4 change a verdict (none does by
inspection, since each verdict was scored under the letter its code ran).
