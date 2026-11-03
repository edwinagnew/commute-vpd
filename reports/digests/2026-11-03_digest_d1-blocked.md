# Digest, qwen-decomp-lead, 2026-11-03 (afternoon)

**D1 blocked, day 1 of 8: the pass test can fail a bad placement, but on KL it cannot yet pass anything; the fix goes to the user today (ask 22), and teacher training is held.**

**What ran, $0 on cluster L40s.** E68 scored four saved updates under the original model's masks (job 50047). Its built controls separate on all four, neither split is faithful, and least squares and adjoint both read "partial". Amendment 1 (jobs 50356, 50364) added the whole update unmasked: it closes 0.10, 0.12, 0.03, 0.14 of the KL gap, against a filed lean of 0.85 or more on the three maths-row updates, with gain kept 0.26, 0.44, 0.02, 0.48. Amendment 2 (job 50371) opened the masks on the first two updates: with every subcomponent fully on and Δ off, KL is about 0.013. Report: `reports/analyses/2026-11-03_analysis_e68-d1-pass-test.md`; notebook `notebooks/E68_qwen_pass_test.ipynb`.

**Independent check: holds with caveats, no bug** (`reports/reviews/2026-11-03_review_e68-ceiling.md`). Corrected in chat: "not the missing Δ" overreached, since Δ on under the masks is unread. Nothing yet separates "the masks drop any change this size" from "the update needs parts the masks switch off".

**Fix proposed.** E72 (`reports/plans/2026-11-03_plan_e72-d1-second-test.md`, code being written, about 3 h, $0) measures only: a random change and Δ on under the original masks, then masks from the fine-tuned model held fixed with a floor and a ceiling, then the update's parts in the clean model. Fable's advice is `reports/plans/2026-11-03_advice_e72-d1-test-FABLE.md`, with its causal fallback sentence overridden.

**Needs the user.** Ask 22, log card `decision-d1-test`: what D1's test becomes. Their answer can wait for E72's numbers.

**Unexplained, no cause named.** Half-open masks with no update lower the next-token loss below the clean model's, half of E40's own drop on those rows. The masked unchanged model is closer to the fine-tune than the clean one on three updates. Least squares nearly equals the often-on half on all four. The gain ruler reads 0.88-0.93 for the fine-tuned model itself, because it starts from the masked forward, so the plan's "1 by definition" was wrong.

**Spend:** $0. **Friction:** the debug job's 30-minute limit left no margin and took one resubmit; runner wake-ups while waiting on their own jobs.
