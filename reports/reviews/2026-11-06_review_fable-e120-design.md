# Fable's advice on E120's design: known and never-solved maths pools for the D3 anchors

*2026-11-06, Fable for qwen-rl-lead. Brief: the user's option 1 (`decisions.md` row 177;
`reports/decisions/2026-11-06_decision_anchor2-organic-options.md`). Read: `.claude/CLAUDE.md`, `GOALS.md` Terms,
the D2 gold arm (`experiments/qwen_recipe/train.py`), E78, E94, E104, the prefix-test analysis, and the measured numbers
quoted below. Not committed; the lead decides. Each question: options, a recommendation, what would change it.*

**Numbers everything below rests on.** HF `generate` on a cluster L40 in bf16 made 344 samples per 115-130 s batch
(`experiments/qwen_updates/results/run.log`, stage `[S own]`, 86 prompts x 4) at a mean of 185 response tokens
(`results/large/S/own/batch_0000.jsonl`), about 550 new tokens/s; greedy batches of 128 took 37 s
(`reports/analyses/2026-10-27_analysis_qwen-gsm8k-baseline.md:254`). GSM8K train has 7,473 problems
(`src/commute/qwen_sets.py:14`); MATH train levels 3-5 hold about 5,500 (published split 1,592 / 1,690 / 2,234; the HF
datasets-server confirmed level-5 counts of 436 algebra, 421 geometry, 276 counting, the other calls timed out). So the
full job is ~13,000 problems x 64 = 830,000 samples, roughly 240M response tokens (200 per GSM8K sample as measured,
~400 assumed for MATH): **90-120 h with the repo's HF path**, which no card and no gate allows. The D2 gold arm is peak
1e-5, 10 warm-up steps, cosine over 100 steps, clip 1.0, 64 sequences per step, micro 4, fp32 forward, sequences of 512
(`experiments/qwen_recipe/results/train_gold_1e-5.json` → recipe; `experiments/qwen_updates/job.py:59-65`,
`train.py:56`), one pass over 6,400 distinct GSM8K problems (`job.py:248`), and it took 1,732 s on an L40 with its reads
(`train_gold_1e-5.json` → seconds; E78's retrain of the a1 arm: 27 min, `reports/analyses/2026-11-03_analysis_e78-anchor1.md:114`).
The decision's "fine-tunes ~6 min each" is wrong by about five times; four arms are ~2 h of L40.

## 1. Sampling engine and cost

Options. (a) HF `generate` as in the repo: ruled out by the number above. (b) vLLM on one H100 pod. (c) vLLM on a cluster
L40 under the `normal` QoS (allowed for the Qwen line when the queue is empty, `.claude/CLAUDE.md` → Rules).
vLLM 0.11.0 pins `torch==2.8.0` and `transformers>=4.55.2` (PyPI, 2025-10-04), the repo's torch (`pyproject.toml:13`);
later versions move to torch 2.9 and 2.14. Still install it in a separate venv (`uv venv` under `/ephemeral/$USER` on the
node, or on the pod), never into the project's: a broken shared venv has already cost the cluster's clones once
(`pyproject.toml:53`). The repo has no measured vLLM throughput; for a 1.7B bf16 model at a few hundred concurrent
sequences an H100 is plausibly 10-20k output tokens/s, an L40 about half. That puts the full 64 at 3.5-7 h on the H100:
on the edge of the 6-h gate, and the number is a guess until the first batch prints it.

**Adaptive scheme.** 16 samples for every problem first (~60M tokens, 1-1.5 h); then 48 more for (i) every problem at 0
of 16 and (ii) the known candidates the matching will draw from, up to three per never-solved problem in the same
stratum, chosen at random with a filed seed. Every problem that enters a pool has 64 samples; a problem in neither pool
stops at 16 and the report says so. Read the labels on the 48 fresh samples where a selection was made on the first 16
(new: 0 of 16 and 0 of 48, so "never solved in 64" holds literally; known: at least 36 of 48, so the first 16 do not
flatter the rate). That is a faithful reading of the brief as long as the write-up says which problems got 16. Cost
estimate 2-4.5 h and $8-18 at the H100's $3.99/h (`experiments/qwen_rl/results/S.json` → usd_per_h): under the gate.

Recommendation: (b) with the adaptive scheme, through `/pod-runner`, sampling settings `qwen_eval.SAMPLE`
(`src/commute/qwen_eval.py:37`: T 0.7, top-p 0.8, top-k 20, the Qwen3 card's non-thinking setting, README lines 151 and
276), not the baseline's top-p 1 (`experiments/qwen_gsm8k_baseline/results/baseline.json` → sampling). Model revision as
`qwen_eval.py:31`. Write every batch's samples to disk as they finish and download them while the next runs (CLAUDE.md's
pod rules). What changes it: a first-batch throughput above 20k tokens/s means the plain 64-for-all pass fits in 3.5 h,
and then run that, it is the simpler reading; below 5k tokens/s, cut MATH to level 5 plus GSM8K (about 9,700 problems)
rather than cut samples. Take (c) only if no pod is up within 10 min; the venv build on a compute node is unrehearsed.

## 2. MATH: source, prompt, cap, grader, SFT text, and where GSM8K goes

Source: `EleutherAI/hendrycks_math`, seven subject configs with `problem`, `level`, `type`, `solution`; not gated, last
modified 2025-01-12, sha `21a5633873b6a120296cce3e2df9d5550074f4a3` (pin it, as `qwen_sets.py` pins GSM8K);
`hendrycks/competition_math` is disabled on the Hub. Drop the ~70 rows whose level is "Level ?". Verify the per-level
counts on load and write them into the plan.

Prompt: the same chat template with `enable_thinking=False` (`qwen_eval.prompt`, `qwen_eval.py:53-55`). Two
instructions: GSM8K keeps `INSTRUCTION` (`qwen_eval.py:32`, "The answer is N."), so its pass rates and SFT rows stay
those of E45, E61 and E69; MATH gets the Qwen3 card's own line, "Please reason step by step, and put your final answer
within \boxed{}." (README line 282). One instruction for both was the alternative; rejected because a boxed GSM8K prompt
would move every GSM8K number away from the existing readings and the known arm's rows away from the D2 gold arm. Say
plainly that the task has one template and two answer formats.

Response cap: GSM8K keeps 512 (`qwen_eval.py:34`); MATH 1,024 new tokens for sampling, because truncated answers are
never right (baseline analysis, line 103) and a sample cut at the cap is "unread", not "unsolved". Rule: a truncated
sample counts as not solved, and a problem with 16 or more of its 64 samples truncated is unread and enters no pool
(reported). If the first batch shows over 10% truncation on level 5, raise the cap to 2,048; vLLM's cost is per token
generated, so a higher cap costs only on the tail.

Grader: GSM8K through `commute.gsm8k.grade` strict, unchanged; MATH through `math-verify` 0.9.0 (PyPI, 2026-01-10; it pins
`latex2sympy2_extended==1.11.0`): `parse` the response's last `\boxed{}`, `verify` against the gold's. Control before any
sampling: every gold solution graded against its own answer reads 1.0 (the GSM8K baseline's control did,
`baseline.json` → control); MATH problems whose gold `math-verify` cannot parse are dropped before sampling and counted.

SFT response: MATH, the gold solution text as it is (it ends in `\boxed{}`) plus the end token, built by
`qwen_sft.sft_seq` (`src/commute/qwen_sft.py:136`) with the MATH prompt; GSM8K, `gold_as_response` as E61. `sft_seq` cuts
at 512 tokens, which on a long level-5 solution would cut the boxed answer the pass rate reads: drop problems whose
prompt plus gold exceeds 512 before matching, and report the share per level.

GSM8K and matching: a stratum is (source, level, subject); GSM8K is one stratum of its own. It contributes its
never-solved problems that pass the label guard (question 3) and the same number of known GSM8K problems; expect few
(one GSM8K problem in eleven was unsolved in 8 samples, `reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md:95`;
at 64 samples far fewer) and expect most of them to be gold errors.

## 3. Pool rules

Known: at least 48 of 64 (or 36 of the 48 fresh under the adaptive scheme). New: 0 of 64. Between: neither pool.
Matching: within each stratum the known count equals the new count, known drawn at random with a filed seed from the
eligible; a stratum with fewer known than new is cut on the new side (random, seeded, reported); a stratum with no known
is dropped from both and reported. File the known threshold as an ordered pair before the build, 48 then 32: if 48
leaves level-5 strata unmatchable, 32 (a majority) holds and the report says which did. Hold out 20% of each pool,
stratified, never trained; it is what "learned" reads. Cut both pools to the same size after the hold-out, so the arms
see the same number of epochs (question 5).

Guards on the never-solved side, both counted in the report. (i) A gold the grader cannot parse is out. (ii) Where 32 or
more of 64 samples agree on one final answer different from gold, the problem is out: a confident, consistent "wrong"
answer is more often a gold error or an ambiguous problem than a missing skill, and both sets carry known gold errors;
it would otherwise put the model's correct answers in the "new" pool. The lead reads 20 random survivors with their
majority answer (20 min) and reports what they look like. Under ~100 problems after the guards, even with the known
threshold at 32, is the brief's trigger for option 3.

## 4. The novelty check

It has to be able to fail, and the repo's prefix test cannot read on this model: its copy control read 0.539 against a
bar of 0.90 (`reports/analyses/2026-10-28_analysis_qwen-prefix-test.md`, scorecard P0), so a Wu et al.-style
whole-remainder match is out. Two checks that can fail:

(a) **Think mode, the one that matters.** The same unchanged model with `enable_thinking=True`, the card's thinking
settings (T 0.6, top-p 0.95, top-k 20, README line 275), 4 samples per new-pool problem, cap 8,192 tokens (the card
asks for 32k; 8k keeps it near 25M tokens, 30-60 min on the pod). Reading: the share of the new pool solved at least once
in think mode, with the known pool's share beside it (expected near 1). Bar: the pool fails as "new" if more than half of
it is solved in think mode, since the skill is then in the weights and the no-think label reads a mode, not a missing
capability; it would also confound D3's think-overlap reading, which is defined on think-mode subcomponents
(`src/commute/qwen_d3.py:427-431`). The bar has no measured reference; it is the point past which "never solved"
describes the prompt rather than the model. Expect this to be tight: Qwen3's small models gain a lot from thinking on
MATH (Qwen3 report; not re-checked here). If it fails, my recommendation is not option 3 but a definition the user must
acknowledge, because it changes the words of the brief: new = never solved in 64 no-think samples and in 4 think samples,
labelled exactly so, if at least 100 problems remain; option 3 only below that.

(b) **Memorised gold text on the known side.** On known-pool problems, the longest run of exact words shared between a
correct sample and the gold solution (exact word matches, not difflib, friction 488; `qwen_sets.grams` has the 8-gram
machinery). Reference: the same measure against the gold of another problem in the same stratum. Bar: the known pool
reads "memorised training text" rather than "known" if the share of problems with a 20-word shared run exceeds the
reference by more than 10 points. Run it today for $0 on E45's own samples against GSM8K gold
(`experiments/qwen_updates/results/large/S/own/`) to see where this model sits before the pool exists.

Not recommended: a train-against-test likelihood gap for MATH; both splits may be in the pretraining data.

## 5. Recipe, "learned", matched size, saves

Recipe: the D2 gold arm's, unchanged (peak 1e-5, warm-up 10, `hf_cosine`, 100 steps, clip 1.0, 64 per step, micro 4,
fp32 forward, AdamW betas 0.9/0.999 and no decay, `qwen_sft.py:187-189`). The only free choice is data order on a pool
of a few hundred. Options: (a) 100 steps x 64 with the pool cycled in a seeded order (`job.py:262`'s `cycle`), 6,400/n
epochs, both pools at the same n; (b) one or two epochs, which will not teach the new pool anything (E84's TOFU arm
needed five epochs to pass its recall bar, `decisions.md` row 147); (c) epochs matched to E84, which leaves the gold
recipe. Recommend (a): same step count and schedule as the gold arm and the 100-step GRPO arms, so every size reading
compares like with like. Seeds 0 and 1 change the order only; the hold-out split is fixed across seeds. Four arms at
~29 min each: four `debug` jobs at $0, or ~1.2 h on the pod after the sampling, whichever starts first.

"Learned". New pool: at step 100, pass@1 over 8 samples on the held-out part at or above 0.10 (W0's is at most 1/64 per
problem by construction) and greedy accuracy on the trained part at or above 0.5. Trained part passing and held-out not
reads "trained solutions memorised, carrying little to held-out problems", the label E84's anchor 2 already carries
(row 147); that is a usable anchor, not a failure. Known pool: held-out and trained greedy accuracy at or above W0's own
rates on the same problems (no damage), and held-out gold loss lower than W0's by more than the two seeds' spread; E61's
gold bar of 1.58 nats (`train_gold_1e-5.json` → verdict) does not carry to a mixed GSM8K/MATH pool. Both arms: GSM8K test
0-255 strict accuracy (`qwen_eval.score`) at the end beside W0's 0.784 (fp32, `results/train_W.json`), the damage reading
every Qwen arm has.

Matched size. qwen-decomp-lead matched anchor 2 to anchor 1 on weight norm, 0.48 and 0.50 against 0.49 (E94 analysis,
"What we were trying to test"), matched the randoms on output size (KL on 16 maths rows, band 1.25,
`experiments/qwen_d3_rows/run.py:50`), and read the maths arms at steps 10 and 100 (E104); output size on the 128 maths
rows is read on any saved state afterwards, so nothing has to be matched on it in advance. The GRPO arms end at norm
0.351 and 0.347 (`experiments/qwen_rl/results/lr1e-6w10{,_s70}/R.json`, step 100) and sit at 0.069 at step 10. The gold
arm at 1e-5 crossed 0.35 between steps 5 and 7 (0.267 at 5) and 0.49 between 7 and 8 (0.579 at 8), inside warm-up
(`train_gold_1e-5.json` → steps). Save fp32 δW (5.6 GB each: E78's eight files are 45.1 GB) at the first step whose norm
reaches 0.35 and the step before, the first reaching 0.49 and the step before, step 10, and step 100: at most seven files,
~40 GB per arm, 160 GB for four (3.3 TB free on `/mnt/nw`). Which of each bracket is "matched" is qwen-decomp-lead's
choice afterwards, as `qwen_updates.choose` does by KL. State in the plan that the norm-matched states are warm-up states
that have not learned, as E94 found for TOFU; it is the same comparison, not a defect to fix.

## 6. The control that runs first, and the early checks

Stage 0, alone, with a stop. vLLM on W0: (i) greedy on GSM8K test 0-255 with the repo's prompt, strict accuracy on fitting
answers inside E61's filed W0 band [0.72, 0.84] (`experiments/qwen_recipe/results/predictions_train.json` → stage_W
→ w0.unprompted); the measured HF readings on those problems are 0.779 in bf16 (E45) and 0.784 in fp32 (E61), so the band
is the measured spread plus binomial noise, not a guess; (ii) sampling at `SAMPLE` on the baseline's 224 prompts, mean
sample accuracy within 0.05 of 0.753 (`baseline.json` → samples; the baseline used top-p 1, so a larger move is a
finding about settings, stop and look); (iii) the two graders at 1.0 on gold. Then the first 16-sample batch prints
tokens/s and the truncation share: GSM8K at or under 0.02 (baseline 0.012-0.016), MATH at or under 0.10; a running
never-solved share per level every batch, and a stop if level 5 is under 5% unsolved after 16 (under 110 problems: the
pool cannot reach 100, go to option 3 before spending more).

Pool build on the CPU, $0: counts per stratum, the guards' exclusions, then the think-mode check (question 4a) before
any training. W0 greedy on both held-out parts: new at or under 0.02 (a check of the build), known at or above the
threshold's rate.

SFT, each arm, bars from the gold arm's same path (`predictions_train.json` → first_minutes → gold_1e-5): step 1 at rate
0 with norm exactly 0 (E78: the step-1 file is all zeros); step 2 norm within |dW| / (lr sqrt(numel)) in [0.85, 1.05]
(E69's band, `experiments/qwen_rl/results/predictions.json:32`; the gold arm read 0.0333 at 1e-6); peak GPU at or under
40 GiB; loss finite and lower at step 10 than at step 1 (gold: 2.41 → 0.54). The gold arm's loss bands themselves do not
carry: the pools' loss scale differs, so read step 1's loss against the same state's held-out loss and stop on a gap
above 0.1 nats, as E61's `held_gap` did. Each saved file's norm within 1e-6 of the training record (E78's check).

Friction: the HF datasets-server timed out on most of the 35 per-level count calls, so the MATH level counts above are
the published split checked at three cells, not measured in full; the lead should print them on load. The decision's
"~6 min" per fine-tune did not come from any measured run.
