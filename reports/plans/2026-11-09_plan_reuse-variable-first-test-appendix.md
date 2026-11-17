# Appendix: apparatus and departures for the first test of the second wording

*Companion to `2026-11-09_plan_reuse-variable-first-test.md`. Quotes and page numbers were taken by two reading agents
from the full texts in `~/commute-data/lit_review_pdfs/survey/` (2301.04709, 2311.12786, 2402.14811); their notes are
in `~/commute-data/scratch/definition-lead/v2-papers/`. Figures were not rendered, so Prakash Fig. 4 and Jain Fig. 8 are
known from captions and text only.*

## A. Departures, Geiger et al. 2025

| Draft's claim | Verdict | Departure |
|---|---|---|
| v2: F = M[φ←s(v_M)] is "an intervention on one of the original's own variables, in the sense of Geiger" | Accurate with a gap | The operation is an interventional (Def 11). F is not the intervened model: the equation is equality of outputs on D. A subspace is a variable only of a translated model (Defs 28, 46). Any same-signature map is already an interventional (Remark 13), so the phrase adds no restriction. The affine class is not an intervention algebra (Def 16(b), Remark 22), so Def 25's exact transformation does not apply to it as a map on mechanisms. An off-distribution s is steering, not interchange (section 3.7). |
| v2 credit: causal model, intervention, interventional are Geiger | Accurate with a gap | Causal model and intervention are credited by the paper to Spirtes, Woodward, Pearl; only "interventional" is its own. The definition does not use the terms outside the credit line. |
| v1: replacing φ's value "is the interchange intervention of Geiger ... with the alignment restricted" | Accurate with a gap | For a subspace the object is the distributed interchange (Def 46). v1 has no high-level model, so no alignment (Def 31) to restrict. "Changes the output" is not Geiger's test, which is agreement with a high-level model (Def 48). |
| v1 credit: approximate abstraction is Geiger | Accurate with a gap | Def 41 is credited to Beckers et al., Rischel and Weichwald, Zennaro et al. v1 never instantiates it. |
| v2: stating a fine-tune as an intervention on its base is in no paper found | **Wrong as written** | Section 3.7, p. 44: "parameter-efficient fine-tuning methods such as Adapter-tuning (Houlsby et al., 2019) can be viewed as interventions on model parameters". One sentence, on parameters, for one family of methods. |
| v1: applying abstraction to a base and fine-tune is in no paper found | Accurate with a gap | Geiger et al. do not do this, and the sentence above is its nearest neighbour. |

Other points. A value computed from other variables is a soft intervention (Def 10), so v2's s(z_M(x)) is soft. The
split conditions take values from another model, which Defs 44-46 do not cover. Their subspace search (DAS) is scored
against a named high-level model, and φ has none. The paper never says what a failed search shows, does not discuss
the illusion problem (Makelov et al. is not cited), and places no limit on the alignment map, so linear φ and affine s
get neither support nor a guard from it.

## B. Departures, Jain et al. 2024 Def 1 against the stipulation

Verdict: accurate in form, with these gaps. v2 records only the second.
1. Def 1 reads a named surjective function fixed in advance; v2 reads any quantity, in use F's own value.
2. That function takes only the data part of the input; v2's depends on the whole input and on F's weights.
3. Def 1 holds on a sub-domain with one capability identifier; v2 on a chosen mixed D.
4. Def 1's read-out is trained on pretraining data, then held to the domain; v2 needs only that an affine map exists on D.
5. Def 1 outputs a token, scored as accuracy; v2 equates a real vector.
6. Def 1 says "linear layer", with no bias in App. D; v2 says affine.
7. Def 1 uses one layer; v2 uses one map over φ's layer and all earlier layers.
8. Def 1 ranges up to the output; v2's z_M stops at φ's layer.
9. Jain probe one position or an average of ten; v2 states no position.
10. Def 1 is exact and declared idealised, with no bar in the experiments; v2 keeps equality for labels and has no cut-off.
11. Jain mostly probe the fine-tuned model for the pretraining function, the reverse of v2's question.
12. Def 1 leads to Def 2's wrapper g ∘ f; v2 draws no such consequence and says rewired certifies presence, not use.

Departures 4 and 7 are where caveat 1 comes from. The stated substitution (F's variable for a named function) is real
and is the smaller change.

Other points. Neither paper has these categories or measures a setter. The nearest match to re-weighted is Jain's
wrapper on a weakly relevant capability, and Prakash for Vicuna-7B. Goat-7B and FLoat-7B are open (175 heads against 72,
"inserting additional components"). Jain know by construction that a capability is present (compiled models, grammar
prior, TinyStories pretrained without Twists), but not what the fine-tune does. Prakash know only each prompt's correct
object. Jain's probe bounds are "Present" and "Not in Pretraining", with no random-label control.

## C. Departures, Prakash et al. 2024 against the split

Verdict on the first condition: accurate with a gap. Patching is fine-tune into base, same input (CMAP), which is
M[φ←v_F(x)](x)'s direction. The second condition is not in the paper.
1. They patch named attention-head outputs at fixed positions, not a subspace of a layer.
2. Heads come from circuit discovery and DCM.
3. They score aggregate task accuracy, not equality on each input.
4. They use only the task's own prompts, not both sides of a cue and far inputs.
5. CMAP covers Goat-7B and FLoat-7B, not Vicuna-7B.
6. Its purpose is finding which step improved, not testing a split or who computes the value.

For CMAP the text gives base and full fine-tuned reference lines and no null patch.

## D. Apparatus

Model. The 67M model, 4 blocks, residual width 768, through `commute.steering.Net`, whose `run` edits and captures
`in<l>` and `out3`. fp32 throughout. Windows of 8 tokens cut from Pile rows.

Pieces. All sit at `in2`, last position. Always on: learned orthonormal 4-column Q and vector b, trained on the
next-token loss on lowercased windows (the task of `commute.lowercase`, re-trained at this length). Cue and XOR reuse
that Q and b. Cue: f = w·h + a, with (w, a) a ridge read-out of `in1`, last position, fitted to "the cue token is in
the window", then frozen. XOR: a bit per vocabulary token from a fixed seed, XOR over the last 4 tokens, f in {0, 1}.
Table: h ← h + C·z + c, with z the `in0` residual at the last position, C and c trained on 32 keys to arbitrary
answers, using the bijection of `commute.kv`, at the last position. Rank of C is read and must exceed k.

Sides of D. Always on: lowercased windows, cased windows, far windows. Cue: with and without the cue. XOR: every window
has a value, so both sides come for free. Table: key and non-key at the last position. Each set has a far quarter,
taken from Pile rows of a distant part of the shuffled set, with no key or cue. Equal weights inside D.

Stage 0, alone, stops the job on failure.
- Identity: M′ with every piece at its start gives the same output as M, within the float floor between two runs.
- Each pair's gap above the floor.
- Rank of z_M over D, counted as singular values above 1e-6 of the largest.

Stage 1: exact readings at the witnesses (G, H, reversion, setter mismatch, the first wording's equalities).

Stage 2: search.
- G: Q (QR-parametrised, k columns) and affine (A, b) at `in1`, `in2`, `in3`, `out3`, last position; Adam, KL(F‖N) as
  the objective, three starts (two random, one from the top principal directions of the F-minus-M activation
  difference), read on the held-out half with d.
- H: φ = the whole residual at each site, last position; s by ridge from z_M to v_F on the fitting half, with the
  ridge value from a small grid on a validation split, then 200 gradient steps on KL. One further site, `final`, the normed input of the unembedding: there KL is convex in
  the affine map, so this is the fit the page's convexity claim refers to. Corrected 2026-11-09: an earlier version
  said this of `out3`, but a final norm follows `out3`, so the fit there is convex only in v_F.
- Reversion and setter mismatch for each found pair of φ and s. Ties go to the highest reversion and lowest mismatch.

Capacity: ridge from z_M to the XOR, and to XOR labels shuffled across windows, in-sample and held-out.

First wording's exact checks: is φ a variable of M on E; F[φ←v](x) = M[φ←v](x) at values from other inputs and random
values; the two consequences; "does not count". Tolerance: the largest difference between two identical runs of M.

Acted-on count: at each of `in1`, `in2`, `in3`, `out3`, 100 random unit directions, and the span of the activations on E.

Nothing above has been coded. Code would follow `src/commute/README.md` and live in one new experiment folder once a
cell exists.
