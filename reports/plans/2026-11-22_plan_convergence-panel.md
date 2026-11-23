# Plan: the realistic convergence panel (cell P2.66, E272)

**In plain words.** We fine-tune the released 67M model on nine ordinary tasks, three seeds each, with one recipe, and
pick checkpoints where the tasks have moved the model's outputs by the same amount (and, separately, where the
weights have moved by the same amount). On those checkpoints we read the four measures of reuse that worked on built
tasks and the four baselines, and ask two things: do the measures rank the tasks the way blind raters expect, and do
they agree with each other more than the baselines do?

Code `experiments/convergence_panel/` (`run.py` stages 0-2, `eb.py` stage 3, `score.py`, `plot.py`, `tasks.py`),
shared `src/commute/panel.py`; rules `results/predictions.json`; card `card.md`; Slurm `cluster/convergence_panel.sbatch`.

## Design check (/design)
- **Goal.** Claim 2 (whip-lead 2026-11-22: "we can sort of measure re-use on realistic tasks"): the user's hopes are
  that measured reuse follows intuition, and that the toy-validated methods converge on realistic tasks.
- **Nearest tried.** E211 (P2.38: every scorecard method on three real-text pairs, size-matched, no KL record, French
  under the gain floor); E233 (British style vs twin); E265 (ADL column); E268 (E_b over K on built and RL items).
  New here: many tasks at once, matched on KL as well as norm, rated blind before any reading, and the four readings
  compared with each other rather than with a built answer.
- **Patch or step.** A step: no known answer exists for realistic fine-tunes, so agreement is the evidence left.
- **Decisions relied on.** decisions.md 2026-11-18 "two-label contrast" (topic read in the contrast); MIN_GAIN 1.
  Departure: label and prompt tasks use the LM trainer (plain SFT on the answer, batch 8) rather than E211's arm A.
- **Definition.** No rule is added to the definition. E_b is read over the interchange-needed K only (P1.54); a
  sampled interchange is labelled a sampled estimate. No verdict on reuse is given; the panel reports agreement.
- **What changes next.** H1 positive and H2 above: claim 2 is written with the panel as its realistic evidence. Both
  fail with readable readings: a clean negative for claim 2 on the 67M. Readings unreadable: say so, no claim.
- **Apparatus control first.** Stage 0 runs alone and stops the job on a miss: the zero update, E211's re-read, and
  E187's copy against prior-matched cipher split.
- **Saved for audit.** Every checkpoint on the rungs (bf16), every record JSON, the selection file, before reading.

## What runs, and what each number should read
| Stage | Reads | Expected at W0 / do-nothing | Expected at the control |
|---|---|---|---|
| 0 | zero update on E53's lowercase rows | size, KL, gain, share numerator, held gain, ADL numerators exactly 0; shares nan | - |
| 0 | E211 `T1_s0_0.0175` re-read | - | share 0.7698 (tol 0.01), steering 0.7514 (tol 0.001; the denominator here is the share's gain, 1.6273 against E211's 1.6286, so about 0.0006 off), erank 597.714 (1e-6) |
| 0 | E187's split, E254's states, E228 reader | - | copy_s0 at c 1: 0.7707; prior_cipher_s0 at c 0.7798: 0.3669 (tol 0.01 each) |
| train | three Gaussians at 0.0175 per task | held gain within ~0.02 of 0 | floor = 3 x the largest |gain| |
| train | squad, conll: real against twin | - | learned when real >= floor and real - twin >= floor; else dropped with a note |
| select | KL rungs 0.001 x 2^(i/2), size rungs 0.002 x 2^(i/2) | - | "hi" = the rung with the most tasks whose 3 seeds each have a state within 10% with gain over the floor (ties to the larger); "lo" = two rungs below (half) |
| read | share, steering, erank, ADL D, size, KL, prior per state; one Gaussian per seed-0 state over its paired gain | random within ~0.02 of 0 | - |
| eb | E_b over K (sampled estimate, 4 derangements x 64) | do-nothing exactly 1 | x2 0.7614 (E268, tol 0.04 = its card spread); random ~0.011 |

## Filed rules (results/predictions.json "rules")
- Readable: between-task sd of task means above the pooled seed sd; else "no information", out of W.
- H1 (KL "hi" set): Kendall tau-b of tier code against the oriented task mean, per rater and the consensus; "positive"
  when tau > 0 and one-sided task-permutation p <= 0.05; beside it, partial tau on the prior only where the prior has
  common support across the top and bottom tiers, leave-one-task-out range, 90% task-bootstrap interval.
- H2 (KL "hi" set): Kendall W across readable readings against W across readable baselines, the matched variable left
  out; "above" when W_readings > W_baselines and above the 95th percentile of W for random rankings.
- Fewer than 5 tasks: "too few tasks". Missing scores and failed controls are reported as such, never 0.
- Predictions (Fable): H1 "+", H2 "above", both low confidence. The other three matched sets are descriptive.
- **Amendment 3 (2026-11-22, redteam-lead's plan check, before any read).** H1 "positive" only when the partial tau
  on the prior is > 0 at permutation p <= 0.05, or the raw call is positive and beats the prior's own tau with the
  tier; with no common support on the prior (its ranges over the reuse-side and new-side tasks do not overlap) a raw
  positive is "confounded with the prior". H2 compares each side's W against its own null at its own m (percentiles),
  with mean pairwise tau and the readings' W after partialling prior and size beside it. Drops are written to
  `results/dropped.json` by `select`; `score.py` refuses unless it predates every read. Critical tau-b (exact, n = 9,
  consensus tiers) 0.526; power at a true tau of 0.3 is 0.23 (0.20 at n = 7, where the critical tau-b is 0.62).
  **If H1 is "no call"** (the likely case), the paper says: on nine realistic tasks the readings' agreement with the
  raters' tiers is reported with its interval and is too weak to call at this panel size, and the panel does not show
  that the readings track intuition beyond the base prior. On the wide panel (Amendment 5, 18 tasks, the joint
  consensus tiers) the critical tau-b is 0.317 and power at a true tau of 0.3 is 0.50 (0.92 at 0.5), so a "no call"
  there is reported the same way, as agreement too weak to call and unlikely to be as strong as 0.5.
- **Amendment 4 (2026-11-22, whip-lead's 22:55 note, user "yes to all four"; before any read).** Two descriptive
  extras, outside H1/H2 (`rules.amendment4`). *Mixture:* lower (consensus reuse, code 3) mixed with yoruba (new,
  code 1) at 0/25/50/75/100% of yoruba's rows in every part of the task (`tasks.mix_task`); 0 and 100% are the
  panel's own runs, so only mix25/50/75 train. Pair: yoruba is the only code-1 task; of the four code-3 tasks only
  lower's saves have held gain over the floor (pile, code and british read below 0 at every save), and lower and
  yoruba share row shape, mask and read layout, so a mix is a row mix. Mix states are picked at the panel's KL "hi"
  target without entering its selection rule. Per measure, Spearman rho of seed means over the 5 fractions,
  "monotone" when |rho| >= 0.9 (exact one-sided 0.05, n = 5), sign against -orientation. *Trajectory:* no new
  saving (train already saves at every rung; lower's seed 0 has 14 saves over its floor): select picks 5 of seed
  0's saves over the floor, evenly spaced, for lower and yoruba; "stable" / "monotone" / "jump" against 3 x the
  measure's seed sd on the KL "hi" set. At W0 every mix reads numerators 0 and shares nan: "no call", never
  "monotone". New jobs: 3 train (5-10 min each, as a panel LM task, plus the second task's build) + 3 read (3 states
  at <= 58 s, ~4 min + set-up each) on debug QoS, and up to 10 trajectory states (<= 58 s each) in lower's and
  yoruba's reads: ~0.5-0.8 GPU-h, $0; exact lines in the sbatch header; select runs
  after the mixes' train jobs and before any read.
- **Amendment 5: wide panel (2026-11-22, whip-lead's 23:10 GO with redteam-lead's agreement, user away).** Nine tasks
  join the filed nine: English, German, Swahili, Tagalog and Spanish Wikipedia (as Yoruba), arXiv abstracts (as the
  Pile rows), sentiment (E211's SST-2 label task), animal-or-plant article category (DBpedia-14, in the topic task's
  form) and news summary (CNN/DailyMail: the article's opening, then its first highlight; its twin shuffles the
  highlights, and it joins SQuAD and CoNLL under the twin learnability rule). The guards: (a) fresh blind Opus and GPT
  raters rated all 18 together in one batch (`results/wide/rater_{opus,gpt}.md`, filed as `raters_wide`), and the wide
  panel is scored on that rating; (b) the added tasks were picked by the filed rule and listed before any of their
  training (`results/wide/tasklist.md`, `order.txt`, commit 48e10783), and a task that misses its KL or floor is
  listed in `results/wide/dropped.json`, never swapped out; (c) the critical tau-b and power are refiled at the new n.
  Two sets are reported side by side and neither replaces the other: the nine, scored exactly as Amendment 3 from
  `results/` on the filed raters, and the wide set, from `results/wide/` on the joint rating, with its own selection
  (the filed rule over all 18, so its matched rungs may differ) and drops. The critical tau-b stays exact up to n = 9
  and uses 200,000 random orderings from n = 10; score.py records the critical tau-b and power at the n kept. The
  nine are scored first, then the wide set. The joint raters again rank by pretraining prevalence, so Amendment 3's
  prior-aware H1 call is the scored call for the wide panel too. Wide consensus tiers: 5 at 3, 3 at 2.5, 5 at 2, 2 at 1.5, 3 at 1. At
  n = 18 the critical tau-b is 0.317 (Opus 0.339, GPT 0.333), power 0.50 at a true tau of 0.3 and 0.92 at 0.5; with
  drops, n = 17: 0.327-0.334 (power 0.48 without summary), n = 16: 0.336-0.354 (0.43 without summary and
  sentiment), n = 15: 0.354-0.375 (0.43 without the three twin-checked tasks). Held masks follow Amendment 7's rule:
  the Wikipedia and arXiv tasks go through `lm_task`, so their held gain is scored at their 500 read positions;
  sentiment and category (label tasks) and summary (a prompt task) score the answer only; no wide builder scores
  anything else. The wide set's E_b (stage 3) is not
  wired: it needs its own go-ahead. Jobs, after the amendment is on main and from the main clone:
  ```
  for T in wiki_en arxiv german swahili tagalog spanish sentiment category summary; do
    sbatch --comment="E272 stage: train $T" cluster/convergence_panel.sbatch train $T; done
  ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/run.py select wide    # after all 18 train files
  for T in <each task in a kept list of results/wide/dropped.json>; do
    sbatch --comment="E272 stage: read $T" cluster/convergence_panel.sbatch read $T; done
  ```
  `read T` reads every set whose dropped.json exists and holds T, the nine first; a state the nine set already read is
  copied into `results/wide/read/`, so a filed task's wide read costs only the states the wide rule picks anew. The
  nine set's `select` and reads go ahead now and never wait for the wide jobs. Then `score.py` scores both. The wide
  drops live in their own file, `results/wide/dropped.json`, checked by `score.py` against the wide reads only
  (`results/wide/read/`, `results/wide/eb/`); `select wide` never touches the nine's `results/dropped.json` or its
  timestamp, and `select` refuses to run again for a set whose `read/` already holds a file.
- **Recipe probe, no recipe change (2026-11-23, probe job 63813, commit 64a6ae60).** Matched-KL SFT on in-distribution
  text only loses held likelihood at 67M: on held-out Pile the best gain over 3 recipes (filed 3e-5/batch 8; 1e-5 and
  3e-6 at batch 32 with 6% warm-up) was +0.0005 nats/token, below a 0.014 floor, turning negative by KL 0.003. The
  recipe stays as filed, so pile, code and british go to `dropped.json` under the filed rule, and wiki_en and arxiv
  are expected to drop the same way in the wide set. The reuse tier is thin (pile, code and british drop out of the
  nine), and this weakens H1.
- Edge cases run on a stub before filing (score.py --stub; since Amendment 3, `--results DIR` on run.py's read shape): a dropped task, a zero-spread reading, a nan seed, ties
  (tier codes), n = 4, the prior with no common support. All gave the calls the rules name.

## Cost and time per stage (measured item x count)
- Stage 0, cluster debug L40, $0: three reads at <= 58 s (one full E211 read, a superset of these readings,
  `cluster/logs/commute-realistic-r2-61448.out` lines 12-13) + E254's world and two reads at 39-47 s
  (`prior_matched/results/a/items/*.json` .seconds): ~8 min.
- Train, $0: a 300-step LM fine-tune 10-16 s (61448 line 36: 10 s for 187 steps), x 3 seeds, + up to 27 saves a seed
  at ~2 s (not measured; the job prints it) and one 16-row forward a step: ~5-10 min a task, nine jobs.
- Read, $0: up to 12 states + 4 Gaussians a task at <= 58 s: <= 16 min a task, nine jobs (~2.3 GPU-h).
- E_b (gated): 738 s per state for 200 windows of 64 tokens on an H100 (E268, definition-lead
  `e268_dl/run_rl_p0.log:8`); the scan scales with tokens, so SQuAD (256-token prompts) ~4x, CoNLL (96) ~1.5x, topic
  (34) ~0.5x. KL-matched only (up to 54 states + 18 Gaussians): 7 LM tasks x 8 x 738 s + squad 8 x 2,952 s + conll
  8 x 1,107 s + topic 8 x 369 s = 21.8 h; with the Gaussians ~30 h = ~$80 at ~$2.7/h on four pods (~7.5 h wall).
  Norm-matched adds about as much again: run it only if the KL-matched readings are readable.

## Gates for stage 3 (gated: over $60)
1. **Pre-mortem.** Recipe: E268's code and settings unchanged (derangements 4, examples 64, seed 268) - guard: x2's
   fresh E_b over K against E268's value in eb stage 0. Data: D = the first 200 read windows cut at the read position;
   SQuAD's 256 tokens may OOM at rows 1024 - guard: the rehearsal measures peak memory on squad. Precision: fp32, tf32
   off, as E268. Memory: 11.3 GB peak for x2 (E268 items/x2.json) - fits an H100. Time: per state above; the job logs
   seconds per state and the runner checks the first against 738 s x tokens/64. Saves: per-task JSON written per
   state (KB); nothing large. Downloads: per state as written. Pod loss: rerun skips written states. Controls first:
   eb stage 0 alone (do-nothing exactly 1, identity 0 flips, x2 K* to 1e-11, x2 K within 0.04). Predictions: H1/H2
   read E_b like any reading; a nan is missing. Never-run path: `eb.py` on a panel state.
2. **Settings sourced.** All E268's (kstar_needed predictions); n_windows 200 = E268's RL items.
3. **Rehearsal ($0).** `E272 rehearse: eb` on a cluster L40: eb stage 0 and one squad state at 200 windows;
   record seconds per state and peak memory, `fleet/rehearsed` it.
4. **Early checks.** eb stage 0 stops the job; the first state's seconds are logged before the second.
5. **One review** (fresh Opus) of this plan and `eb.py`.
6. **Go-ahead** from whip-lead (in-plan gated run).

## Open choices (for the lead)
- The raters' tiers go in `raters` before `read` (both refuse while it is empty); the user's sort when it arrives.
- E272: no E-number is claimed yet; every file says E272.
- Extra rows (E256/E260 procedural, E218/E222 RL) are not wired: they need their own read rows (rb_states'
  `p256_windows`, `rl_windows` files) as CueReads. Left for a follow-up job; they sit outside the statistics.

## Not run
Everything: no GPU job was submitted. CPU only: the stub score, the card sketch, the selection rule and prompt-row
builders on toy inputs, and an import check of eb.py.
