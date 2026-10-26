# Review: which E14–E20 claims hold only for the large Adam update (2026-10-22)

*Review agent, read-only. Source of the comparison readings: `reports/analyses/2026-10-22_analysis_sgd-vs-adam.md` (E21), its
tables and its result JSONs. Claims as worded in `docs/research_log/parts/sections/07_claims.html` and the E14–E20 rows
of `experiments/TRIED.md`. Two numbers not in the E21 report were read from JSONs: the size of change of E16's states
(`experiments/split_comparison/results/split_comparison.json`, `rel_norm_max`) and the share of the least-squares
split's change left in Δ for E21's small fine-tunes (`experiments/sgd_finetune/results/{split,adam_matched}.json`,
`delta_change_frac`). Vocabulary: `GOALS.md` → Terms. No analysis report was edited and no GPU job was run.*

**In plain words.** E14 to E20 measured almost everything on two kinds of update: a full Adam fine-tune at rate 3e-5, and
Adam component training at rate 1e-5 read at step 10. E21 showed that both of these fail the adversary only while the
update is large or still damaging the model's other predictions. Smaller or longer-trained versions pass. So the claims
built on a failure mostly describe that one update. The claims about the splits' trade-offs and about the adversary
itself still stand. For `GOALS.md`, the case against maps with Γ frozen now rests on fewer, narrower failures than the
log states.

> **Every verdict below is a 20-step-adversary reading,** both the E14–E20 readings and E21's. Amendment 2 of
> `reports/analyses/2026-10-21_analysis_cancellation-complement.md` found that search far from its maximum.
> **Pending: E20 Amendment 3** (budget sweep at 40, 80 and 160 steps with step size 2/steps,
> `experiments/cancellation_complement/results/complement_amend3*.json`) is running in another session and was not
> read for this review. It re-reads W₀, Adam 1e-5 component training at step 10 on r0 t352, r1 t534 and r2 t275, the
> emoticon least-squares split of the 3e-5 fine-tune, and the adjoint split as the passing control. Rows 1, 2, 8, 9, 10 and
> 12 and the headline row depend on it. It reads only large-update states, so it can confirm or dissolve the failures,
> but it cannot show whether E21's small-update passes survive a longer search. Re-judge those rows when its JSONs land.

## The two updates in question

- **Large fine-tune:** emoticon Adam 3e-5, seed 0, read at steps 10, 100 and 1000 (E14, E15, E16, E18, E19, and E17's
  splits). Size of change at step 10 is 0.0112. E21 matched it with an SGD fine-tune at 0.0024 and an Adam 3e-6
  fine-tune at 0.0022.
- **Early component training:** Adam 1e-5, read at step 10 (E11's states, reused by E17 and E20). E21 trained the same
  arm on r0 and r1 to steps 100 and 1000. There it passes at 1.00 to 1.07, with damage elsewhere down from 0.20 to 0.03.

## Verdicts

*Only large:* the claim rests on a failure that E21 shows goes away at a smaller or later update. *Untested beyond:*
measured only on these updates, with no E21 reading to compare. *Stands:* E21 agrees, or the claim doesn't depend on
the update.

| # | claim (log wording, short) | states it read | E21 reading | verdict |
|---|---|---|---|---|
| 1 | The least-squares split of a finished fine-tune is adversarially fragile (E14, E15, E16, E19) | large fine-tune: 5.24 to 7.3 at step 10 across cards; 1.48 and 2.11 at 100 and 1000 (E16) | step 10: SGD 1.16, matched Adam 1.12, both passes; the large fine-tune 5.41 | **only large** at step 10. Steps 100 and 1000 for a small fine-tune not run |
| 2 | A split LoRA survives the adversary where a split fine-tune does not (E14) | large fine-tune vs E9's rank-8 LoRA | fine-tune side passes when small. The LoRA's size of change is 0.044 at step 10, four times the large fine-tune's and about twenty times the small ones' | **only large** on the fine-tune side. What makes the LoRA pass is not a small size of change |
| 3 | A fine-tune's expressibility is on the read side only; its outside share rises along training (E14) | large fine-tune | outside share at step 10: SGD 0.034, matched Adam 0.157, large Adam 0.218 | **untested beyond.** The share depends on the optimiser, so "a fine-tune" should read "an Adam fine-tune". "A LoRA is easier to commute than a fine-tune, refuted on expressibility" **stands**: SGD sits further inside, which makes the refutation stronger |
| 4 | The do-nothing reference fails for full-size Adam fine-tunes (E14 2.29) | large fine-tune | small fine-tunes pass at 1.08 and 1.12. The large one passes at step 100 (1.12) and is inconclusive at 1000 (1.39) | **only large, at step 10** |
| 5 | The rescaling share grows with training; the rescaling split as a state keeps none of the behaviour (E14, E16) | large fine-tune | none | **untested beyond.** Not an adversarial claim |
| 6 | A warm-started refit of Γ drifts the unchanged model; the refit's attacker is weaker than the harness's (E15) | W₀ control | — | **stands.** About the recipe, not the update. The refit was motivated by row 1's failure, which a small update doesn't have |
| 7 | Frozen Γ's stale importances are not the split's fragility (E15 stage 0) | large fine-tune, least-squares split | — | **untested beyond.** At the small size there is no fragility to explain |
| 8 | On emoticon no split keeps the behaviour, the floor and Δ fixed together; component training does (E16) | large fine-tune, steps 10/100/1000 | step 10, small fine-tunes: least squares passes but leaves 0.25 (SGD) and 0.32 (matched Adam) of the change in Δ. Adjoint passes with Δ untouched but target probability 0.71 and 0.69 | **stands at step 10 for every size measured.** The least-squares leg now fails on Δ, not on the adversary. Small updates at steps 100 and 1000 not run |
| 9 | The adjoint split keeps the floor and Δ; its behaviour fades (E14, E16) | large fine-tune | adjoint 1.02 to 1.03; target probability 0.69 to 0.71 at step 10, against the large fine-tune's 0.93 | **stands**, and the behaviour fades sooner at the small size |
| 10 | The adversary's excess is not cancellation across the changed pieces (E17) | Adam 1e-5 component training at step 10, six tasks; large fine-tune splits | Adam 1e-5 passes r0 and r1 from step 100 on | **stands as a description of those states.** The failures it explains are now the early, high-damage Adam ones. Not run on the component-training fails that persist (SGD at step 1000, r2) |
| 11 | Re-switching is a mechanism, not a leak (hypothesis H, E18) | do-nothing of the large fine-tune at step 10, three tasks | SGD's emoticon state also needs Γ's switching: share 0.01 with Γ on the original activations | **untested beyond.** Not an adversarial claim; nothing says it depends on size. The unscored adversary turned-down share is a large-only, 20-step reading |
| 12 | The excess sits in the interaction of the alive and dead removals, not in the originals (E20) | four failing states: Adam 1e-5 at step 10 on r0, r1, r2; least-squares split of the large fine-tune | three of the four stop failing at a smaller or later update. r2 fails under every arm | **only large** for r0, r1 and the least-squares split. r2 stands as a failure not tied to the large update. Amendment 3 re-reads all four |
| 13 | A step-by-step split keeps less behaviour in the pieces than least squares; moving the bases changes nothing; the per-step scale refit sends more into Δ (E19) | large fine-tune path, 1000 steps | none | **untested beyond.** Its adversarial 2.67 at step 1000 is the large path's |
| 14 | VPD's 20-step adversary is far from a near-worst mask (E20 Amendments 1–2) | W₀ and one failing state | — | **stands**, independent of the update |

**Headline row: "no map with Γ frozen keeps the decomposition adversarially faithful"**
(`reports/decisions/2026-10-21_decision_regating-main-line.md`). Its fine-tune and component-training failures are rows 1, 4
and 12. At the 20-step budget, what remains is narrower: the r2 failures under every arm, SGD component training at
step 1000 on r0 and r1, SGD on the 32-pair and copy ciphers, and the least-squares split's share in Δ at any size. The
decision is the user's, as E21 says.

## What E21 can't settle for these claims

- Each comparison is one task. Only emoticon has fine-tunes at both sizes, and only steps 10 to 1000 exist for
  component training.
- The size comparison is across optimisers. Within SGD, size predicts the verdict. Across update kinds it doesn't: the
  LoRA passes at four times the large fine-tune's size (row 2).
- The E16 and E14 readings come from other cards (H100, RTX A5000). Their floors differ, so ratios are compared through
  verdicts, not values.

## Not run

- The least-squares, adjoint and do-nothing placements of a small fine-tune at steps 100 and 1000.
- E17's and E20's decompositions of the excess on the failures that persist: SGD at step 1000 on r0 and r1, and the SGD
  cipher states. Their runs save states to `results/large/` on the cluster. No local copy exists
  (item 6 of `reports/reviews/2026-10-21_review_sgd-finding-influence.md`).
- E18's shared-pattern test on an SGD or small-Adam update.
- E19's step split along an SGD or small-Adam path.
- The E14 outside-share and rescaling series along an SGD fine-tune.
- Any reading of Amendment 3's results. They were not on disk when this was written.
