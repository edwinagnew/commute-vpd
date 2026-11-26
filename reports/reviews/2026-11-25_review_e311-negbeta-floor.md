# E311 (P2.83): the noise-floored negative-beta filter, checked before run 1

P2.83, due 11-26. Asked by whip-lead (~16:10): check the rule baseline-lead is filing after the gate-5 code review's
MUST (`2026-11-25_review_e311-code.md`). That MUST: a latent is dropped when a base β is below −(noise floor) instead
of below 0; the floor is measured in the same job from a half-split spread; literal and floored filters are reported
side by side, and the run-2 kill rule reads the floored one. At 16:10 the rule was not yet in
`experiments/crosscoder_fig4/results/predictions.json` (main 1bf5aa9b, no uncommitted change in baseline-lead's
clone or worktrees). So below are the conditions the filed text must meet. I re-check it against them once it lands.

**Plain words.** The authors drop a latent whose base strength comes out negative, but for a latent that exists only
in the fine-tune that strength is zero plus noise, so the drop is a coin toss. Dropping it only when it is clearly
below zero is the right repair. But the noise has to be measured so that a one-sample estimate does not make the bar
meaningless, and the result is then no longer "the authors' published rule", which is what the user's cell asks for.

## MUST

1. **The cell's done_when names the published rule.** `fleet/cell show P2.83`: "The planted control passes under
   Minder et al.'s published rule (nu cut-off 0.2, our 0.5 specific bar unchanged) ... or the cause of the miss is
   shown". The negative-β exclusion is part of that rule (App. E.2), and the floor relaxes it. A pass under the
   floored filter alone meets the second branch, not the first: "missed under the published rule because the sign
   filter acts on noise; passes with it floored". The paper's sentence and the cell fill must say it that way. If the
   intent is for the floored rule to count as the first branch, that is a change to the user's done_when, so it goes
   to the user, not whip-lead. It is the same kind of call as the placement answer.
2. **The sd from a half split needs pooling.** One split gives each latent's β two estimates, β_A and β_B. The full
   estimate's sd is then |β_A − β_B| / 2: var(half) = E(β_A − β_B)² / 2, and the full value is the mean of two halves.
   That is one degree of freedom per latent. Under that estimate, P(β < −m·σ̂ | true β = 0) is a t(1) tail: 0.15 at
   m = 2 and 0.10 at m = 3, against 0.023 and 0.0013 for a known σ. So the floor would barely protect the plant.
   File one of these:
   (a) A pooled σ: the RMS of (β_A − β_B)/2 over latents of similar firing count, e.g. the alive latents of the same
   side binned by token count, or pod a's self-control latents at the same count.
   (b) K-fold (K ≥ 10) per latent, so σ̂ = sd of the fold means / √K, with K − 1 df.
   Then state m and the false-exclusion rate it gives under (a) or (b).
3. **Split by window, not by token.** The 100k latent-scaling tokens come in windows, and tokens of one window are
   correlated. An interleaved token split underestimates σ, and the floor then drops clean latents again. Split by
   whole sequence.

## SHOULD

4. **Edge cases to file:**
   - σ̂ = 0, or a latent that fires in only one half or fold: then −m·σ̂ = 0, which is the literal rule. Say so, and
     log it.
   - A latent with fewer than `alive_min` firing tokens in a half: undefined. Name the branch (excluded, kept, or
     reported as missing). It must not decide the gate silently.
   - The two betas (β^ε_base, β^r_base): one rule for each, both must clear.
   - Ties at exactly −m·σ̂: inclusive or not.
5. **The same filter for the plant and the 48 readings.** The floored filter also moves change from "artefact" back
   to "specific" on the 48 states (code review MUST 1, last bullet). Report d_authors under both filters for every
   side, not only the plant, so the plotted reading and its gate use one rule.
6. **Log the lead's four betas and their σ̂** in every `stage0b_*.json` (the code review asked for the betas too), so
   a miss can be read as a sign flip or as a real base direction.
7. **Restate b/c's expected value** against E212's rescored miss (0.766 → 0 under the literal filter), as a dated
   note beside the filed file: literal filter, lean miss at one or both seeds; floored, lean pass.

## Departure row (for `appG_departures.tex` and the plan appendix A)

| authors | ours | why |
|---|---|---|
| App. E.2: latents with a negative base β are dropped; in BatchTopK they found "None" (p. 26) | dropped only when a base β is below −m·σ̂, σ̂ the [pooled / K-fold] sampling sd of that β in the same job; the literal rule reported beside | a latent present only in the fine-tune has base β ≈ 0 with a sign set by sampling noise; on our 67M the literal filter dropped E212's isolated plant (part 0.766 → 0) |

## Not run

No rescore of E212 or E203 with a floor (the betas aren't saved). The filed rule itself wasn't checked: it wasn't
filed at the time of writing.
