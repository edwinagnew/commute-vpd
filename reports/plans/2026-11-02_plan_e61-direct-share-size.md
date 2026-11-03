# E61: fix the fine-tune recipe, then test the direct share against size (2026-11-02)

*Task: the user chose option A of `reports/decisions/2026-10-30_decision_e57-no-teeth.md` and added: "make sure you have
fine-tuned the synthetic tasks properly - e.g. they actually learn and youre using the right learning rates and so on".
Sources: `reports/reviews/2026-11-02_review_qwen-finetune-recipes.md`, `experiments/qwen_recipe/results/A.json` (job 48367),
`reports/plans/2026-11-02_advice_e61-recipe-FABLE.md`.*

**Restated request.** Scale random weight changes up to the size of the biggest fine-tune, read a smaller state of that
same fine-tune, and read them all through the frozen decomposition by the direct share (with the original masks kept, the
fraction of the masked model's error removed by changing only the subcomponents). Before that, make sure the synthetic
fine-tunes (template, invented facts) learn, at rates that suit this model.

## Design check
- **Goal.** `GOALS.md`: whether a simple map fails because training teaches new capabilities. This decides whether the one
  reading left on Qwen3-1.7B measures anything beyond size. Indirect: it decides whether the reference comparison is worth
  running.
- **Nearest tried.** E57 Stages A-C (direct share, size confounded), E45 (the recipes), E61 stage A (the rounding). None
  varied size with the kind of change held fixed, and none trained with the reader's forward.
- **Patch or step.** A patch on E45's recipe, needed before any Qwen update can be read at small size, then the step the
  user chose.
- **Decisions departed from.** Decision row on the warm-up recipe (fp32 master weights, bf16 forward): the forward becomes
  fp32. Requires the user's sight of this plan. Everything else as E45.
- **What changes next.** Part 1: which recipe B would use. Part 2: B or C, by the line below.
- **Controls first.** Part 1 arm 0; Part 2's reproduction of C0.

## Part 1: the recipe (Job R)
Train with a true fp32 forward, no autocast, no TF32: the same forward every reader uses. Under bf16 autocast, weights
round to a grid and W0 sits exactly on it, so training is blind to updates under about 6e-5 per entry. **Departure from the
literature:** standard mixed-precision training uses the autocast forward, from memory and not checked here; it suits runs
of thousands of steps, where a few blind steps do not matter. Here those steps are what we read.

Settings: batch 64, sequence 512, warm-up 10 steps, cosine, clip 1.0, AdamW (0.9, 0.999), no decay, all as E45. Clip 1.0 is
the Hugging Face default, checked in the installed `training_args.py`. Rate: the cited 1e-5 has no copy in the repo, so the
rate is now measured by the grid, not cited. Micro 4 (measured 20.9 GiB + Adam 10.5).

| arm | task, peak, steps | why | min |
|---|---|---|---|
| 0, control, alone first | own answers, 1e-5, 10 | same recipe as E45 except the forward: training must now see its update | 9 |
| 1 | template, 3e-6, 100 | does a synthetic task need 1e-5? | 39 |
| 2 | template, 1e-6, 100 | floor of the grid | 39 |
| 3 | gold, 1e-5, 100, states saved 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100 | the fixed-recipe fine-tune A reads, with the smaller states | 36 |
| 4, the user's choice | invented facts, 3e-6, 300 | the other synthetic task, under the fix | 85 |

**Stops, filed:** arm 0 stops if training loss at step 4 is under 0.08 (training still cannot see its update; expected
at least 0.12; the rounded run read 0.055). Arm 1 stops if loss is over 0.9 at step 10; arm 3 if over 2.2 at step 4; any arm
if first-step allocation passes 40 GiB. Arm 0's other first-minute values are in Fable's advice, section 2.

**"Learned" rules, by behaviour, on full-precision weights at the arm's last state (256 problems, se about 0.03):**
- Template: unprompted template rate at least 0.74 (0.8 of the teacher's 0.926), and strict accuracy at least 0.60. The
  lowest passing peak is the recipe. Neither passes: run 1e-5 under the fix.
- Gold: held-out loss drop at least 1.58 (0.8 of 1.98), answer length at most 0.6 of W0's.
- Invented facts: recall at least 0.8, 4-shot maths accuracy at least 0.65.
These replace E45's filed rules for this experiment only; E45's predictions stay as filed.

## Part 2: the direct share against size (Job S)
Same 128 maths rows, same code path as E57 (`read_state`), about 35 min per state, in this order:
1. Control: the Gaussian change at C0's scale reproduces C0's K00 on 16 rows, within 1e-3, as C0 used. Stop on a miss.
2. Gold (fp32-forward run), step 100. Beside the saved gold's 0.092, this gives a spread between two runs of the recipe.
3. The gold state whose size is within 15% of a random change's (0.63 or 0.86), expected at steps 7-15, in place of the
   card's step 9, which came from training blind and which no fixed recipe repeats. It must pass the gold rule's
   loss part scaled to where it is read, or the report says it is a state on the way.
4. Gaussian change scaled to gold step 100's size, by search on 16 rows. 5. Alive-subcomponent random change, same.

Size is K00 − K0 (the KL from the changed model to the CI-masked original, minus the same at the original model). At W0 the
size is 0 and the share has no denominator. Expected: nothing filed on which side it falls; a share proportional to size
would put random near 0.04-0.05 at 1.44, gold near 0.09.

**Direct share here (user, 2026-11-02):** K10 uses masks truly from the original subcomponents, the split applied only in the masked forward; E57's version, whose masks read the moved vectors, is reported beside it. Both size axes (K00 − K0, KL from W0) are reported; B needs the line met on both.

**Filed line, at matched size, against the higher random (toward C), both sizes:**
- gap at least 0.03 at both sizes: training differs beyond size → B.
- gap at most 0.01 at either size: size explains the direct share → C.
- Anything else: C, with a note that the remainder is under three floors and B's own limit would bite.
The floor 0.01 is the 0.007 gap between the two random kinds plus the bootstrap half-widths. Edge cases as in Fable's
section 3 (no state in band, a difference between the two gold runs over 0.01 replaces the floor, NaN: no reading, a point
estimate decides at the line and the report says so). Blind to: anything but the direct share; one seed each; maths rows
only. It cannot itself separate "already there" from "new".

## Gates (treated as gated: Job R with arm 4 is about 3.5 h; with Job S about 6 h in all, $0)
1. **Pre-mortem** (each with its guard). Recipe: the fix fails → arm 0's stop. Data: same rows and formatting as E45, code
   reused, E45 row hashes checked. Precision: fp32 throughout, checked by arm 0. Memory: 37 GiB expected at micro 4, stop at
   40. Time: normal QoS, 2 h jobs, resume per arm at saved Adam state (16 GB). Saves: states in `results/large/` on the
   cluster's disk, 42 GB + facts; survive job loss. Downloads: none. Controls: same rows and card as E57. Predictions:
   walked over edge cases at filing. Code paths never on a GPU: the fp32 training loop, run first in arm 0.
2. **Settings sourced:** above; the peaks are measured, not cited.
3. **Rehearsal:** arm 0 is the rehearsal, stage A has measured memory and time. Resume is rehearsed by killing arm 0 at step 5.
4. **Early checks:** the stops above.
5. **Two reviews:** plan and code, each by a fresh agent, before Job R.
6. **The user's go-ahead:** after arm 0's numbers, one line.

## Not in this plan
Own-answer retraining and the null at new rates (nothing in A reads them; deferred to D's decision). A review note on earlier
claims comes separately: E40 steps 1 and 10, E44 and E57's readings stand as readings; E45's early "learned" flags and
"step 12, recall 2%" need a note; the 67M looks unaffected, since its checkpoint is fp32 (inferred, not measured).

## Amendment, 2026-11-02, after the five checks (`experiments/qwen_recipe/results/checks_*.json`)
No filed check missed. Γ's bf16 cast has no floor at a tiny change (K01 − K0 −3e-5, switched on 133 against 5,730 at E40). Row format does not change K0 (0.056 unpacked, 0.070 packed). The direct share with truly original masks is within 0.002 of E57's on four states, same order, on 16 rows (gold 0.089, E40 0.062, invented facts 0.014, Gaussian 0.022): E57's numbers stand as a description, and truly original masks decide here (user).
- **Job S reads the direct share with truly original masks, E57's beside it** (`experiments/qwen_recipe/checks.py` `true_kl`), and both size axes plus the reverse KL on every state. On 128 rows, the two random changes need their true-mask reading at their existing size too, since K10 and K11 true were read on 16 rows only: add 2 × ~13 min (split saved for the Gaussian).
- **fp32 W0 sits beside every learned rule**: read at the job's start on the same problems. The template's accuracy line 0.60 is the prompted W0's 0.680 less 0.08; Fable's source "W0's 0.70" was wrong. Its filed number stands; its source is this.
- **Not yet checked by these checks:** fp32 Γ against bf16 Γ directly at a large change (the tiny floor passed, and the rule sent nothing further), and whether the null gains on fp32 weights. Own and null remain out of this plan.
- **Fine-tune code.** `sft_step`'s autocast becomes a named choice, default unchanged, as frozen E40 and E45 scripts call it. No frozen script changes.

## Amendment 2, 2026-11-02, after the plan review (`reports/reviews/2026-11-02_review_e61-plan.md`; these override the text above)
All nine must-fixes and the should-fixes are taken, as the review words them, except where stated.
**Job R.**
- **Stage W, new, right after arm 0, before any 100-step arm** (about 25 min, fp32, no training): the rule prompts at W0 on fp32 (unprompted, prompted template, facts, 4-shot; test 0-255 for the template), then the saved references read in fp32: `a1_s0` step 100 (template, accuracy, 256), `a2long_s0` step 300 (recall, 4-shot), `gold_s0` step 100 (256). Rehearses generation. 4-shot's batch halved for fp32. Generation runs after training in a later pass, read from saved files, never with Adam's moments held.
- **Template line.** `strict_dropped` at least the prompted W0's fp32 reading on test 0-255, less 0.08, computed in Stage W and written before any arm reads it. Rate line stays 0.74. Each arm reports sum of rates, |δW|, maths KL beside the reference at 1e-5, because a passing lower rate has value only if it gets there with less movement.
- **Invented facts: 6e-6, 300 steps, not 3e-6** (FYI to the user). 3e-6's summed rate, 4.5e-4, is under the 5e-4 that reached recall 0.32 at 100 steps; 6e-6 gives 9e-4, near the 8.8e-4 that reached 0.91. Filed: the sum table, an expected loss at the pause (from `T_a2long_s0` at the same sum), and if the arm fails, it reports "this sum is too low" and 1e-5 for 300 steps (about 100 min) runs only if the user asks. Warm-up 10 of 300 is not the cited ratio 0.1: a departure, as in E45.
- **Gold's rate is not tested**: one rate, 1e-5. The report says so. The length line 0.6 has no source: E45 filed 0.8; keep 0.6 as measured from `gold_s0`'s 0.48, stated as chosen from the reference. Saves add 8, 9, 11, 12, 13.
- **Differences from E45**: fp32 forward; micro 4 (same sum). Both listed.
- **First-minute values** come per step from that step's own batch in the E45 record, tolerance from the fp32-autocast gap at W0, at steps 2 and 3 (a1 step 2 is 1.108, not Fable's 0.96-0.98). Arm 0's 0.08-0.12 continues with a note. Memory stop reads step 2, per arm, at 40 GiB. Arm 1's stop catches no movement only; arm 0 catches blindness.
- **Rehearsal.** Arm 0 with `t_chunk` 5: exit 3 at step 5, resubmit, steps 6-10 run, asserts on restored norms, rate and Adam count. Then Stage W. A 100-step arm's fallback when `normal` is not free: `t_chunk` 50.
**Job S.** The size verdict takes these, replacing "Filed line" where they conflict:
- True-mask pass computes each row's masks in turn, one mask pass at W0, then the applied forward for that row, never holding masks for all rows (154 GB otherwise). Per-row sums written for the bootstrap. Budget written per row. 128G checked on the rehearsal.
- **Rehearsal: Job S on 2 rows and two ladder states end to end first**: selection, search, both masks, both KLs, reverse KL, bootstrap, write, resume.
- **Scale search on K00 − K0**, against gold's own 16-row value, match 1.05; then a 128-row check within 15%; on a miss, one more search step, then report unmatched.
- **Which gold state:** the saved state with 16-row K00 − K0 nearest a random change's existing size, within 15%; report its drop and length and call it a state on the way, with no "learned" check.
- **Both axes.** Each random kind has two reads, so interpolate its share linearly at gold's value on each axis, K00 − K0 and KL from W0; below its lowest read use that read (toward C). Compute the gap on each axis; "the higher random" per axis. B needs both axes and both sizes at gap at least 3f; C on any gap at most f; else C with the note.
- **Floor f** = max(0.01, the difference between `gold_s0` and the fp32 gold at step 100, on E57's masks, if their sizes are within 15%); otherwise 0.01. Gold's size floor note: a read-off difference over f is reported.
- **Branches.** fp32 gold fails its rule: stop, report before Job S. Step-100 size at or under 0.99: read `gold_s0` (true masks added) as the big state and scale the randoms to it. Between 0.99 and 1.44: scale to the fp32 size. NaN: no reading. Negative gap: C.
- The expected value under proportionality is 0.037-0.067 for random at 1.44, not 0.04-0.05; 0.067 would read under B's line.
**Times.** Job R about 4.5 h, Job S about 3 h: about 7-7.5 h in all, 82 GB, $0.

## Amendment 3, 2026-11-02, after the verdict: three more saved states, exploration
Line C stands and is closed (`results/size_verdict.json`, checked by `reports/reviews/2026-11-02_review_e61-size-check.md`). Before the user decides between stopping and the reference comparison, read gold's steps 10, 20 and 50 through Job S's `state` path: `sbatch cluster/qwen_recipe_size.sbatch more --budget-s 10000` (about 26 min a state, 80 min, $0). Rule and expected values: `results/predictions_size_more.json`.
- **Goal.** `GOALS.md`: whether the one reading left on Qwen measures anything beyond size. Says what the words of line C are worth; it cannot change the line.
- **Nearest tried.** Job S itself, which read steps 8 and 100 only. Fable's advice (`reports/plans/2026-11-02_advice_e61-after-size-FABLE.md`) asked for 20 and 50; step 10 is added here because the independent check found step 8 won its pick narrowly, with no reading just past it.
- **Patch or step.** Patch: approved class 1, a GPU pass of the existing harness on saved states. One new stage of three lines; nothing else changes.
- **Decisions departed from.** None. **Control first:** each state's size reproduces `select`, or the job stops. Not gated: under 3 h, $0.
