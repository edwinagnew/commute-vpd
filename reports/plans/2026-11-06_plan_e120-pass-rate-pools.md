# E120 plan: anchor 2 as pass-rate pools of one maths task

*2026-11-06, qwen-rl-lead. Task (user, 2026-11-06, `decisions.md` row 177): option 1 of
`reports/decisions/2026-11-06_decision_anchor2-organic-options.md`, option 3 as backup. Design advice: Fable,
`reports/reviews/2026-11-06_review_fable-e120-design.md` (adopted except where "Overrides" says). Not gated
(under $60 and 6 h per run); fast lane: no pre-run review, the lead reads and labels the result.*

**In plain words.** We sort real maths problems by whether the unchanged Qwen3-1.7B can already solve them, sampling
it many times per problem. One pool holds problems it solves often ("known"); the other holds problems it never solved
in 64 samples ("new"), matched on source, level and subject. We fine-tune on each pool's human-written solutions with
the recipe the GSM8K gold arm used, two seeds each, and save the updates for `qwen-decomp-lead`'s three D3 readings.

## Request restated in the Terms' words
Two SFT updates δW of Qwen3-1.7B, one per pool, two seeds each, by the D2 gold-arm recipe (E61 `gold_1e-5`), saved in
fp32 at the steps where their weight norm matches anchor 1 (0.49) and the RL arm (0.35), at step 10 and at the end
(step 100). "New" is reported as "never solved in 64 samples", never "absent". Readings are qwen-decomp-lead's.

## Design check
- **Goal.** `GOALS.md`: does post-training teach new capabilities or strengthen existing ones, and can a VPD reading
  tell which? D3 calibrates the readings on two updates whose kind is known; this gives both from one real task, so
  the GSM8K GRPO, own-samples, shuffled-reward and template states are read against anchors from their own task.
- **Nearest tried.** E84/E88 (TOFU facts, anchor 2 until now: new facts, memorised, carrying to rephrasings for most;
  rows 147-151) and E78 (template, anchor 1). TOFU is a different task and format; this keeps the format and task and
  varies only whether the model could already solve the problem. E61 `gold_1e-5` is the same recipe on all of GSM8K.
- **Patch or step.** A step: the user replaced the synthetic anchors with a real-task design.
- **Decisions relied on.** Rows 152 (both the full update and the matched step are read; claims of "new" rest on the
  full update with its size stated), 177 (this design). Departures below.
- **Definition.** No verdict rule is added. The D3 readings are qwen-decomp-lead's, under their filed rules; reuse
  readings, if any, read under rows 81, 87 unchanged.
- **What would change what we do.** A new pool under ~100 problems after the guards, or a failed novelty check, sends
  us to option 3 (or to the user, see check N1). Readings that do not separate the two pools tell D3 that the readings
  cannot tell elicitation from new content on real data.
- **Control first.** Stage 0 alone: vLLM on the unchanged model reproduces the repo's GSM8K numbers (below), and both
  graders score gold-vs-gold 1.0. Nothing else runs until it passes.

## Stages and what each number should read
**Stage 0 (control, H100 pod, minutes).** vLLM 0.11.0 in its own venv (pins torch 2.8.0). (a) Greedy, GSM8K test
0-255, strict: inside E61's band [0.72, 0.84] (measured 0.779 bf16, 0.784 fp32). (b) Sampling at `qwen_eval.SAMPLE`
on the baseline's 224 prompts: mean accuracy within 0.05 of 0.753. (c) `gsm8k.grade` and `math-verify` 0.9.0 on gold
answers: 1.000. Any miss stops the job.

**Stage 1 (sampling, same pod, ~2-4.5 h, $8-18).** GSM8K train (7,473) and MATH train levels 3-5 (`EleutherAI/
hendrycks_math`, sha 21a5633…, ~5,500 problems; verify on load). Chat template, thinking off. GSM8K keeps
`qwen_eval`'s instruction; MATH gets the Qwen3 card's "Please reason step by step, and put your final answer within
\boxed{}." Caps: GSM8K `MAX_NEW`, MATH 1,024 tokens. Adaptive: 16 samples for all; 48 fresh samples for every
0-of-16 problem and for up to 3 known candidates per new candidate in its stratum. Labels read on the 48 fresh samples
(so a problem is not labelled on the samples that picked it): new = 0/16 and 0/48 (never solved in 64); known =
≥ 36/48. If the first batch reads over 20k tokens/s, run the plain 64 for every problem instead. The write-up says
which problems stopped at 16. Early checks: tokens/s printed; truncation GSM8K ≤ 0.02, MATH ≤ 0.10 (else raise MATH's
cap to 2,048); stop if level 5's 0-of-16 share is under 5% (no new pool to be had, go to option 3).

**Stage 2 (pools, CPU, $0).** Strata = (source, level, subject); GSM8K is one stratum. Out of the new side: gold
unparsable; ≥ 32 of 64 samples agreeing on one non-gold answer (a gold-label error guard); ≥ 16 of 64 truncated (also
out of the known side: unread); prompt + gold over 512 tokens (the cut would remove the answer). Known and new matched
1:1 per stratum by seeded draws; unmatched cut and counted; both pools the same size n; 20% of each held out,
stratified. The lead reads 20 new-pool survivors with their gold solutions. Expected: n a few hundred, nearly all
MATH level 4-5, GSM8K contributing almost nothing to the new side. n under ~100 → option 3.

**Novelty checks (stage 1's pod, after the pools).** N1: 4 samples with thinking on per new-pool problem (card's
thinking settings, 8k cap). Fails if over half the pool is solved with thinking: the skill would then be in the weights
and would confound D3's think-overlap reading. On a fail the lead stops and asks the user (changing "new" to
"unsolved without and with thinking" changes their words); option 3 only below 100 problems. Expected: likely tight,
about 0.3-0.6 solved. N2 (known side, $0, CPU): the longest exact word run shared between correct samples and the
gold text, against a cross-problem reference; fails at +10 points (memorised gold, not solved).

**Stage 3 (SFT, cluster L40, ~29 min per arm, $0).** The gold recipe unchanged (peak 1e-5, warm-up 10, cosine over
100 steps, clip 1, 64 sequences a step, micro 4, fp32 forward), the pool's training part cycled in a seeded order
(6,400/n passes, the same for both pools); seeds 0 and 1 change the order only. Four arms: known s0, s1, new s0, s1.
fp32 δW saved at the steps bracketing norm 0.35 and 0.49 (the gold arm crossed them at steps 5-7 and 7-8), step 10,
step 100. Early checks on the gold arm's path: step-1 δW exactly 0; step-2 \|δW\|/(lr·√numel) in [0.85, 1.05]; peak ≤
40 GiB; loss finite and falling by step 10; saved-file norms within 1e-6 of the record. Readings at step 100: new
pool held-out pass@1 over 8 samples (≥ 0.10 = learned; trained-part greedy ≥ 0.5 with held-out under 0.10 reads
"memorised, carrying little", E84's label); known pool held-out loss drop beyond the seed spread and no damage; GSM8K
test 0-255 strict beside W0's 0.784 for every arm.

## Predictions (filed in `experiments/qwen_pass_pools/results/predictions.json` before stage 1)
P1 the new pool has ≥ 100 problems after the guards (lean: yes, 200-600). P2 N1 passes (lean: uncertain, 0.5).
P3 the new arm learns by the held-out rule (lean: no, 0.35: memorised, carrying little). P4 the known arm leaves GSM8K
test within 0.05 of W0. P5 at step 100 the new arm's norm exceeds the known arm's (lean: yes). No D3 reading is
predicted here.

## Overrides of Fable's advice, and departures
- Known bar: ≥ 36 of the 48 fresh samples, fallback ≥ 24/48 only if level-5 strata cannot match (Fable gave both
  48/64 and 36/48); one rule, read on fresh samples.
- **Departure (FYI):** adaptive sampling, not 64 samples for every problem, unless throughput allows it. Every pool
  member has 64 samples; only problems in neither pool stop at 16.
- **Departure (FYI):** the decision report's "~6 min per fine-tune" was unmeasured; E61's gold arm took 1,732 s on an
  L40, so the four arms are ~2 h of card, still $0.
- GSM8K enters as one stratum; matching on level and subject applies to MATH.

## Schedule
11-06/07 code and stages 0-2; 11-07 novelty checks; 11-08 SFT; paths to qwen-decomp-lead 11-08/09; readings by 11-10.

## Not run
Everything above. Option 3 is not built unless P1 or N1 sends us there.

## Amendment 1 (2026-11-06, lead; first of two)
**What happened.** Stage 0 passed (`results/stage0.json`: greedy 0.785, sample mean 0.756, graders 1.0). Stage 1's
first shard stopped the job: MATH truncation 0.294 at cap 1,024, then 0.115 at cap 2,048, over the 0.10 bar, which
was guessed, not measured (`results/stage1.json` → `stopped`). In that shard the truncation sits on the hard problems:
25 of 210 MATH problems have at least 8 of 16 samples cut, and 17 of the 25 never-solved-in-16 problems have 4 or more
cut; the cut texts read as loops or long floundering, with some genuine long solutions.
**Change.** The cap stays 2,048 (a longer cap would push the run past 6 h on the cards available). The first-shard
MATH bar becomes 0.20 at the last cap, about twice the 0.115 measured on this path; the GSM8K bar is unchanged. The
per-problem guard is unchanged: 16 or more truncated of 64 keeps a problem out of both pools. "New" is reported as
"never solved in 64 samples of at most 2,048 tokens". Stage 2 also reports how many never-solved problems the
truncation guard removed, since it bears on P1. No filed prediction changes.
**Also.** On a pod with no git history the filed-file check compares sha256 with the value the runner passes, in place
of a commit (friction 599). The rerun resumes from the cap-2,048 shard 0 already home.

## Amendment 2 (2026-11-06, lead; the last allowed)
**What happened.** Stages 1-2 and N1 ran (pod 2, $15.72 session total; `results/stage1.json`, `pools.json`, `n1.json`).
The pools came out at 78 a side, under P1's 100: **P1 misses and stands.** N1 passed (new pool solved with thinking
0.256 against the 0.5 line; known pool 0.974, control), N2 passed. The gold-error guard cut 260 never-solved problems
(GSM8K 204, MATH 56). A $0 check (`reports/reviews/2026-11-06_review_e120-gold-error-guard.md`) read them: about two
thirds are problems the model gets wrong the same way each time; MATH's removals are almost all grader misses on
presentation (choice letters, `\text{}` units, trailing punctuation, a percent sign on one side, unordered lists),
16 of 56, not gold errors; one of the 78 kept (`math/number_theory/654`, gold = choice D) is also a grader miss.
**Change (a bug fix, not a new rule).** The MATH grader normalises presentation before math-verify: a choice letter
matches the gold's choice, `\text{...}` units and trailing punctuation are dropped, percent signs are matched, and a
comma-separated list compares as an unordered set. Every MATH sample is regraded on CPU from the stored answers (stage
1's pass counts, labels and guards recomputed), and stage 2 rebuilt with every guard unchanged, GSM8K included. The
check's rerun projects n ≈ 115 (72 MATH, 43 GSM8K). The known side is regraded the same way. Stage 0's grader controls
(gold-vs-gold 1.0, next-gold ≤ 0.05) are rerun on the new grader before the rebuild.
**Branches.** n ≥ 100 → SFT as planned (four arms), the report stating P1's filed miss and the post-fix count side by
side. n < 100 → the backup (option 3), with a decision card first because n sits near the line. GSM8K supplies over a
third of the new pool against the plan's "almost nothing"; about a fifth of GSM8K never-solved problems carry a wrong
gold label (the check's 9 of 50), which the guard may not catch: the write-up states it, no further rule is added.
