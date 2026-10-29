# Review: what anchor 1 becomes now that no prompt raises accuracy (E45, advice, 2026-10-29)

*Fable, for the qwen-decomp lead. Read: the Q2 plan §1, §3-§9, the re-plan §6-§7, Amendment 1, `results/P.json`,
`results/large/P/*.jsonl` (the 5 × 512 greedy responses), `run.log`, `predictions.json`, `src/commute/qwen_anchors.py`,
my anchor-2 grid review, `GOALS.md` → Terms. Advice only; the user decides.*

**Plain words.** No prompt raised the model's maths accuracy by more than noise, so anchor 1 cannot be "the model
answers better because it was prompted"; the plan's fallback, a fixed answer template, was reported as failing too,
but the responses on disk show the model followed the template on 93 of 100 problems and the filed regex rejected
them over trailing spaces and blank lines. I recommend taking the format fallback as the plan already provides, with
the regex made tolerant of whitespace and that change filed in Amendment 2 together with anchor 2's grid fix, and
stating beside every anchor-1 reading that the prompt also makes the model shorter and less accurate at maths. The
anchor then fixes behaviour novelty for a surface format the model already produces on request, which is weaker than
the accuracy version but still the one reference on the "already there" side that the elicitation question needs.

## 1. What stage P found, re-read from the responses

| prompt | strict accuracy (512 train problems) | mean response, tokens | follows the template, filed regex | follows it, whitespace-tolerant |
|---|---|---|---|---|
| none | 0.834 | 190 | 0 | 0 of 512 |
| fewshot | 0.801 (−0.033, se 0.017) | — | — | — |
| recheck | 0.850 (+0.016, se 0.016) | 200 | — | — |
| givens | 0.846 (+0.012, se 0.016) | — | — | — |
| format | 0.680 | 130 | 17 of 512 (0.033) | **474 of 512 (0.926)** |

The filed rule needs ≥ 0.9 with the prompt and ≤ 0.05 without. The filed regex (`FORMAT_RE`) requires each line to
end in exactly one newline; the model writes markdown line breaks ("  \n") and a blank line before "The answer is".
Every one of the 512 responses begins "Given:", 503 contain "Steps:", every one ends "The answer is N." (the baseline
instruction asks for that last line, so it does not count as template-following; the "Given:"/"Steps:" scaffold does,
and appears in 0 of 512 unprompted responses). The 38 tolerant misses are mostly missing "Steps:" or a step line
without a leading dash. **So the fallback qualifies on the plan's own criterion; what failed is a whitespace bug in
the regex.** Loosening it after seeing the data is a rule change and must be filed as such (Amendment 2), with the
strict count (0.033) reported beside the tolerant one.

Two facts about the format prompt that matter for what the anchor means:
- **It lowers accuracy from 0.83 to 0.68 and shortens answers from 190 to 130 tokens.** The wrong answers are
  arithmetic slips in terser steps, not grader misses (their last lines are well formed, e.g. "The answer is 18.").
  Anchor 1 therefore distils three things at once: the scaffold, shorter reasoning, and a maths accuracy ~0.15 lower.
  All three are behaviours the frozen model produces under the prompt, so behaviour novelty is still fixed, but the
  anchor's effect on GSM8K has the opposite sign to the GSM8K arms'.
- **Anchor 2 also shortens GSM8K answers** (198 → 18-47 tokens, accuracy 0.78 → 0.13-0.20; my anchor-2 review §2).
  Both anchors now move the maths rows partly by a format shift. Rows read on the shared maths axis will see that
  common component; rows read on each anchor's own rows (the filed decisive point) will not.

## 2. Options

| option | compute | amendments | what it lets us conclude | risks and confounds |
|---|---|---|---|---|
| **(A) Format fallback, regex made whitespace-tolerant** | $0: P's format responses exist; one ~3-min resubmit of P for the chosen prompt's greedy answers to test 256-511 (anchor 1's own rows); S A1 sampling 2,000 × 8 at ~130 tokens (2-3 `debug` jobs); T/E × 2 seeds (~70 min). Job 2: +4-6 states, ~$3-4 on the pod | Shares Amendment 2 with anchor 2's grid fix: one note, two items (the regex, and anchor 2's decisive state) | The plan's reading survives: a GSM8K arm is read between an "already producible" update and a "new content" update on the decisive rows. Every anchor-1 sentence carries "a surface format the model already follows on request, not a better way to solve problems" (plan §3) plus "the prompt lowers accuracy 0.83 → 0.68" | The template is learned in a few steps; anchor 1's decisive state (own-KL 0.1) will be an early state where the δW is mostly scaffold tokens. Behaviour-kept and placement may then read "high" for a trivial reason (few tokens, one direction). The common format component with anchor 2 narrows separation on maths-row readings only. Post-hoc rule change, filed |
| (B) New accuracy-raising prompt family (8-shot, verify-by-substitution, program-style reasoning) | $0, one P job (~20 min) per 3-4 candidates | One amendment (new candidates were "fixed now" in §3); none left for anchor 2 unless combined | Same as the original design if one qualifies | Headroom is ~0.17 above 0.834 and the best filed candidate gave +0.016 ± 0.016; +0.03 is unlikely, and a prompt that only just qualifies gives a small distillation gap, so anchor 1 would be a weak, noisy update anyway. Program-style reasoning risks a think-like change. Not worth the last amendment |
| (C) Promote the self-distillation null to the "nothing new" reference, in place of anchor 1 | $0: S null seed 1 (a second draw of 4 samples × 2,000; the sampling budget freed by anchor 1) + T/E null s1 (~35 min); Job 2 states as planned for anchor 1 | One amendment: the null is filed as carrying no prediction and no label (§1); it would need the anchor-1 predictions and the two-seed rule | Reads GSM8K arms between "SFT on the model's own unprompted distribution" and "new facts". Novelty is zero by construction, which is cleaner than a format | Own samples shares its data with the null (the filter is the only difference), so "own samples reads like the null" is near-certain by construction and says little; gold and GRPO are the informative arms. The null moves slowly (sharpening toward T = 0.7) and may not reach own-KL 0.1 by step 100: no reference at the decisive point. Relabelling an unlabelled arm after the fact is the kind of move the re-plan §7 was written to forbid |
| (D) Drop anchor 1; read against anchor 2 alone | $0 saved (no S A1, no anchor-1 T/E; Job 2 ~$3-4 cheaper) | The validation and reading rules ("both anchors", t between two means) must be refiled: one amendment, shared with anchor 2's | One-sided: "unlike new content on rows X" or "like it". Cannot say "like elicitation": D3's "distinguish elicitation versus new mechanisms" has no elicitation side | Loses the deliverable's shape; a GSM8K arm far from anchor 2 could be far in any direction |
| (E) Distil think-mode answers into no-think | — | — | — | Forbidden by the re-plan §7 (makes the think-overlap row true by construction). Stays out |
| (F) One-line-answer format ("Reply with one line") | as (A) | as (A) | Isolates the format component of anchor 2's maths-row shift: anchor 2 − anchor 1 on maths rows ≈ content | Makes the anchors collinear on maths rows by construction; own rows of 20 tokens give tiny per-row KL. A reading, not an anchor; keep for Q3 if the maths-row confound needs separating |

## 3. Recommendation, and what would change it

Take **(A)**: anchor 1 = the format template, regex tolerant of trailing whitespace and blank lines, filed in
**Amendment 2 alongside anchor 2's decisive state** (my anchor-2 review, option (d)). Both items are reading rules,
not recipe changes, and both are forced by the same cause: the filed rules were written for arms that behave
differently from how these two behave. Keep the null beside anchor 1 as the plan already says (§1): with a format
anchor, the null is the free check that "reads like anchor 1" is not just "reads like any SFT on the model's own
samples". Add the sentence "the format prompt lowers W₀'s accuracy 0.83 → 0.68 and shortens answers 190 → 130
tokens" to the anchor-1 limits in §1, next to the domain sentence. Expected values worth writing into the amendment
before the run: own-KL crosses 0.1 within the first ~10 steps (the scaffold is a few fixed tokens per answer, as
anchor 2's template was: own-KL 0.064 at step 2, 0.24 at step 3); gap closed ≥ 0.5 well before step 100; GSM8K
accuracy at step 100 falls toward the prompted 0.68 and length toward 130. If accuracy falls far below 0.68, the
update did more than the prompt did, and that is reported.

**Would change it.**
- If the user rules that loosening the regex after the data is a refiling (memory: never refile predictions), then
  (A) is out as a scored anchor. Second choice is (C) for the reference plus the format arm run and reported as an
  observation, unscored: $0 and the same jobs.
- If, at the decisive point, anchor 1's state is so early (own-KL 0.1 at step 2-3, in the warm-up ramp at rates of
  10⁻⁶) that the behaviour rows read near W₀'s noise floor, the row is "not separated", not a finding about
  elicitation. Say so in the amendment's expected values, so it is not read as a result.
- If the lead wants anchor 1 to also train 300 steps (Amendment 1 left this open): no. The anchors are matched on KL,
  not steps, and the format anchor saturates early; 300 steps adds states nobody reads.

## Not run
- The tolerant count (474 of 512) is from a shell regex over `large/P/format.jsonl`, not from `follows_format`; the
  fixed function must reproduce it before Amendment 2 is filed.
- No check of whether the 38 misses cluster on hard problems (would bias the own rows toward easy ones).
- The prompted greedy answers to test 256-511 (anchor 1's own rows) do not exist yet; P stopped before writing them.
