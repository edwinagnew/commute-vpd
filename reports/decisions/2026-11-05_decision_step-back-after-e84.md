# Advice: step back on D3's anchors, and the cheap checks on E84 (2026-11-05, for qwen-rl-lead)

*Read: GOALS, the reorient and replan plans, the qwen-rl plan, E57/E61 decisions, E69, E75, E78, E81, E84, E85, E86 reports, the D1 advice, both digests, the decomp-lead handoff, backlog, decisions.md, E84's `train.json`. "Verified" = read in a file; "judgement" = mine.*

## Q1. What serves the aim now

**The problem is not the anchors; it is that no decomposition reading on Qwen has yet separated anything but direction and size.** Verified: the re-gating share reads 0.99 for every change, random included (E57); the direct share tracks size (E61, "size explains the direct share"); masked forwards pass 0.0004-0.013 of a column change (E72/E74); own-mask reads cannot credit a placement (E79); the adversary fails only changes 250-740 times a real update's size (E85). What does separate updates is non-VPD: size, and how the gain builds (E86: a tenth of the first fine-tune keeps 0.93, of gold 0.83, of TOFU 0.28).

**The two anchors cannot be compared at matched size, and that is itself the finding.** Verified: anchor 1 step 100 has ‖δW‖ 0.49; E84's 5-epoch update 18.1, 37×. E84 saved anchor 1's step grid, and its step 20 reads 0.31 against anchor 1's 0.30, but at that size TOFU has learned nothing (rephrased recall 0.24 at step 200, floor 0.18). The replan's rule "compare on a common KL grid, read only where learned" (§7, user 2026-10-28) cannot be met: content the model lacks needs tens of times the update that behaviour it already produces does. Judgement: write that sentence into D3 rather than build around it. One claim in the files looks wrong and bears here: the replan's prediction table says anchor 2's "direct share larger"; E57 already read invented facts *below* random at equal size (0.012 vs 0.022). Do not reuse that table as leans.

**Options.**

1. **Stop training; make the six arms readable on one footing.** Run: second seeds of anchor 1 (27 min L40, $0) and of E84's 5-epoch update (12 min H100, ~$2, or L40 $0), because the user's own rule validates a row only when the anchor gap exceeds the seed spread; plus a one-page inventory (arm, seed, steps, ‖δW‖ and gain per step) for qwen-decomp-lead. Half a day. Tells: nothing new itself; removes the single-seed caveat from every D3 row by 11-11. Next: decomp-lead's rows.
2. **A facts-domain anchor 1′: facts W0 answers only under a few-shot prompt (PopQA/TriviaQA, filtered at $0 on an L40), context-distilled bare→prompted with E84's recipe.** One day, $0-3, new dataset so it needs the user's yes. Tells: whether "37× smaller and saturating" tracks novelty or domain; this is the backlog's domain-matched pair. Next: if it reads like anchor 1, D3's separating rows are domain-free.
3. **A maths-domain anchor 2: a made-up operator learned to generalise to new numbers** (the 67M cipher's analogue, beside anchor 1 and the GSM8K arms). One day, $0-3, needs the user's yes. Tells: new content with a true generalisation test, in the arms' own domain. Next: GSM8K arms read against two maths anchors.
4. A third RL seed ($20): does not change E75's table, where RL, null and own-samples read alike. Not recommended.

**Recommendation: option 1 now, with Q2's checks in the same jobs; options 2-3 proposed to the user as the next anchor pair, after 11-11 unless decomp-lead's first rows separate the anchors on something other than size.** Judgement: with six days left and no D3 row yet read on any anchor, a new training day buys less than making the existing ones defensible.

**What changes it:** decomp-lead reads a row that separates anchor 1 from TOFU and is not size (then option 2 first, since only a domain-matched pair tells novelty from domain); or ask 29 resolves to C (D1 cannot be met), in which case the lead's time goes to the write-up.

## Q2. Cheap checks on the 5-epoch update

Do, as one $0 L40 `debug` job (new E-code, class 1): the update and W0 on one card, (a) rephrased/trained/floor recall, (b) bare questions without the suffix, (c) real facts (E81's 217-row read). Rule: same-card within 0.02; bare rephrased recall under floor + 0.1 adds "tied to the suffix" to the label; real-fact greedy accuracy more than 10 points under W0's 69% adds "at a cost to known facts". (c) matters most: a damage reading would mean a "new" row on this anchor could be harm.

Do: a blind hand count, new E-code: Opus reader, 100 rephrased rows each of W0, trained wording and step 1250, rubric fixed to TOFU's key facts, W0 ≤ 0.10 filed as the control. $0, 2-3 h. The only check that can move the label "carrying only partly".

Skip: a third Haiku judging (rule), more epochs (10 added 0.02), synthetic biographies (decided).
