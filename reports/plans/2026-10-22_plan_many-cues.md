# Plan: does the fixed change follow the cue or the answer? (E33, `experiments/many_cues`)

*2026-10-22, coding agent for the lead session. Follows E28 (`reports/analyses/2026-10-22_analysis_regating-tasks.md`) and runs
beside E32 (`reports/plans/2026-10-22_plan_shift-anatomy.md`, running). E30 to E32 are taken, so this is E33. Words as in
`GOALS.md` → Terms.*

## The design check

**Approved class.** Class 4: the re-gating line's "What's next" card names "a new word task in which many different
cues lead to one answer, approved by the user as the line's next task" (research log, changelog v33; the user,
2026-10-22). Induction and spread-out fine-tunes stay on the backlog. No sign-off needed.

- *Goal.* `GOALS.md` → the fixed-shift hypothesis, and the pending scope decision on it. E28 found that the hypothesis held
  on every word task and failed on the 32-key tasks. Every word task so far had one trigger and one target. So "the model
  already does the task" and "one cue, one answer" are confounded. This experiment separates them.
- *Nearest tried.* E28 (one fixed change per task, single-pair word tasks) and E32 (the same change computed per key on
  copy, where key and answer coincide). Neither has many cues sharing one answer on a behaviour the model already has.
- *Patch or step.* A step: it decides which reading of E28's scope holds before the hypothesis is restated.
- *Working decisions.* Γ frozen, do-nothing placement, Δ out of the CI-masked model, E28's 10-step Adam fine-tune at
  3e-5. The cues are mined with the task ladder's rows and rules (`src/commute/ladder.py`). No departure.
- *What changes next.* The decision tree below: each reading names a different scope for the hypothesis.
- *Control first.* G1, the emoticon state, runs first and alone and must reproduce E28. Then one of E28's word tasks
  through every arm, where the per-cue change must equal the shared one (C_one). The job stops if either fails.
- *Numbers.* Five, in the table below, each with its expected reading.

## Restatement

Each state is a do-nothing state: a 10-step Adam fine-tune of all 24 matrices puts δW in Δ, the released subcomponents
stay, Γ is frozen and reads the updated model, and Δ is left out of the CI-masked model. Every arm writes some
importance values at the trigger and keeps the original model's values elsewhere. It is scored as E28 scores its arms:
recovery ρ = (lo − L₀₀) ÷ gap, and R = ρ ÷ ρ_trig, a share of what each input's own updated values give back.

The question: when K different cue words already lead the model toward one answer, and a fine-tune strengthens all of
them at once, is the fixed change one change for all cues, or one change per cue?

**The task.** From the task ladder's Pile rows, the 600 most frequent word-like tokens form a pool. A pair of trigger and
target *qualifies* when the target is word-like, the original model's mean probability of it after the trigger lies in
rungs 0 to 2 (from 0.001 up), and that probability is at least five times the target's probability over all positions.
The last rule keeps cues and drops words after which the target is merely common. The target is the one with the most
qualifying triggers. The cues are its qualifying triggers in order of frequency, K = 32 of them. That matches the copy
task's 32 keys, so the comparison with E28 and E32 is like for like. The job stops after mining if fewer than 16 qualify.

**Windows.** Up to 60 windows per cue, split by source row as the ladder splits. That gives each cue about 48 training
windows, at least 30, against E32's 9 to 15 per key. Each cue has 4 to 6 held-out windows, about 190 in all, twice
E28's count.

**The three states.**

- *One answer* ("one"): the K cues, all fine-tuned toward the target. The question.
- *Many answers* ("many"): the same K cues and windows, each fine-tuned toward its own other qualifying target, matched
  in baseline. This is the null that separates "follows the answer" from "any mix of existing pairs works".
- *Control* (r1_t588_v452, ` will` → ` have`): one cue, one answer, E28's state.

## The numbers (trigger position, held-out windows, W₀'s values elsewhere)

- **R_shared_net**: E28's net fixed change, (ρ(m₁) − ρ(m₀)) ÷ ρ_trig, means over all training windows.
- **R_cue_net**: the same with m₁ₖ and m₀ₖ averaged over cue k's training windows and given to cue k's held-out windows
  (`shift_anatomy.group_fixed`, E32's per-key code).
- **R_group_net**: the same with K random groups, five draws: the size-matched null for R_cue_net.
- **overlap**, reported: the per-cue change sₖ = m₁ₖ − m₀ₖ over alive subcomponents, split-half. The mean cosine
  between different cues' changes, divided by the mean cosine between the two halves of one cue's windows. It reads 1
  when all cues move the same way, 0 when their changes are unrelated.
- **R_other_within_net, R_other_across_net**, reported: each held-out window given another window's updated values,
  from the same cue or from a different cue, minus the same with that window's original values.

| number | W₀, or a zero change | control (one cue) | many answers, predicted | one answer, "answer" | one answer, "cue" | one answer, "kind of task" |
|---|---|---|---|---|---|---|
| R_shared_net | gap 0, undefined; W₀'s own values give back 0 | 0.916 (E28) | < 0.5 | ≥ 0.7 | ≤ R_cue_net − 0.2 | ≥ 0.7 |
| R_cue_net | 0 | = R_shared_net within 1e-3 | ≥ 0.7 | ≥ 0.7 | ≥ 0.7 | ≥ 0.7 |
| R_group_net | 0 | = R_shared_net | within 0.1 of R_shared_net | same | same | same |
| overlap | undefined | undefined, one cue | < 0.5 | ≥ 0.7 | < 0.5 | ≥ 0.7 on both states |
| R_other_within / across | 0 | ≈ 0.93 (E28: 0.955 − 0.028) / undefined | ≥ 0.7 / < 0.5 | ≥ 0.7 / ≥ 0.7 | ≥ 0.7 / < 0.5 | ≥ 0.7 / ≥ 0.7 on both |

Under "kind of task" the many-answers column also reads R_shared_net ≥ 0.7.

**What each number is blind to.**

- The shared and per-cue changes are means. They are blind to anything that varies within a cue, such as its context.
  E28's word tasks showed that such variation matters little there: another window's values gave back 0.96 to 1.00.
- R_cue_net is blind to the size of the averages; R_group_net is that control.
- The overlap weighs every alive subcomponent equally, including ones that do not matter for the target. It is a
  mechanism readout and decides nothing.
- Every number sees the trigger position only, one fine-tune seed, one 67M model, and frequent cue words.

**Noise floor.** E28's readings are deterministic per state. Its control matched E18 within 0.002, and E18's two seeds
agreed within 0.02. The rules' margins are 0.1 and 0.2. The random-group draws give the per-cue arm's sampling spread,
and the within-cue cosine is the overlap's own ceiling.

**Gates.** G1 (emoticon, first, alone): r within 0.05 and R_shared_net within 0.02 of E28 (0.5965, 0.8147). The
per-window read must give W₀'s own values a recovery of 0, the state's own values ρ_trig, and E28's mean change, each
within 1e-4. G2 (control): r and R_shared_net within the same tolerances of E28, reported. C_one (control): R_cue_net
equals R_shared_net within 1e-3. A G1 or C_one fail stops the job. A many-cue state is scorable as in E28 (gain ≥ 1 nat,
gap ≥ 0.5, ρ_trig ≥ 0.5) and is read only if C_size holds on it.

## Predictions (`experiments/many_cues/results/predictions.json`, filed before the run)

Absolute rules throughout; the full text is in the file. A state's reading is "neither" if R_cue_net < 0.5; "per cue
only" if R_cue_net ≥ 0.7 and leads R_shared_net by at least 0.2; "shared" if R_shared_net ≥ 0.7 and R_cue_net leads by
less than 0.2; else "partial".

- **C_one**: the control's per-cue change equals its shared change. High.
- **C_size**: random groups land within 0.1 of the shared change on both many-cue states. Medium.
- **A_many**: "per cue only". Medium-low. Each pair is an existing behaviour, which worked alone in E28. An average over
  32 different answers should carry little of each.
- **A_one**: "shared". Low. E32's working story is a fixed "when" part and a "what" part that follows the answer. With
  one answer, the "what" part is shared.
- **Q**, the combined reading: "follows the answer". Low.

## Decision tree

The pending scope decision (E28) is whether to restate the hypothesis as "on tasks the model already does".

- **G1 or C_one fails:** stop and report; nothing is scored.
- **Follows the answer** (one: shared; many: per cue only): the fixed change is one per answer. Copy fails because it has
  32 answers. The scope becomes "one fixed change per answer", and "tasks the model already does" is not the right line.
  E32's per-key reading on copy, where key and answer coincide, should then hold; on the 32-pair cipher a change per
  answer is the next cheap test.
- **Follows the cue** (per cue only on both): the fixed change is one per cue. The one-cue confound is resolved toward
  the cue. E32's per-key reading on copy then says whether existing behaviour matters as well.
- **Follows the kind of task** (shared on both): one change carries any mix of existing pairs. E28's scope, "tasks the
  model already does", stands, and copy fails because its behaviour is new.
- **The per-cue change fails on a joint fine-tune** (either state "neither"): even per cue, a fixed change does not carry
  the switching when many pairs are trained at once. The hypothesis covers single-pair fine-tunes only. Stop and tell
  the user.
- **Mixed or partial:** report; the user decides. If C_size fails, that state is reported, not read.

## Cost and platform

E28's eight states took 142 s on a RunPod 4090, model load included. This job runs four states, three of them through
about 35 masked forwards each, plus one mining pass over 32768 Pile rows with a 600-token pool. Expected 5 to 8 min on
an L40, under 12 GB of GPU memory and about 10 GB of CPU memory. Cluster `debug` first (`cluster/many_cues.sbatch`,
`--time=00:25:00`), where E5 cached the rows. On a RunPod 4090 the rows stream from HuggingFace first, a few minutes
more, under $0.40.

## Not run (by design)

Seeds beyond 0; positions other than the trigger; E28's shuffle and wrong-target nulls; E32's strength, matrix-type and
steering arms on these states; a per-answer change on the 32-pair cipher (next, if "follows the answer");
faithfulness of the states under the adversary; induction and spread-out fine-tunes (backlog).

## Code

`src/commute/many_cues.py`: the mining rules (`qualifying`, `pick_target`, `alternative`, `table`) and the per-cue
bookkeeping (`overlap`, `donors`, `reading`, `combined`). `ladder.trigger_stats` now keeps its sums on the GPU and
returns the background distribution; its results are unchanged and its CPU test passes as before. CPU tests in
`tests/test_many_cues.py`. `experiments/many_cues/run.py` reuses `regating.prepare`, `finetune`, `measure`,
`shift_anatomy.trigger_read`, `group_fixed` and `rho_of`. `plot.py` waits for the results.

## Hand-off (pod-runner brief lines)

- Command: `cd /mnt/nw/home/e.agnew/commute && sbatch cluster/many_cues.sbatch`, after `cluster/sync.sh up`.
  On a pod: `bash cluster/pod_setup.sh`, then `.venv/bin/python experiments/many_cues/run.py`.
- Inputs not in git: `experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt}` and
  `experiments/task_ladder/results/large/windows_r1_t588_v452.pt` (on the laptop and NFS). Pile train rows 0..32768
  are cached on NFS (`artifacts/sets/train_0_32768.npy`). The run checks the files before loading the model.
- Early stop: the job exits by itself on G1, C_one or too few cues (`status` says which); report that log line.
- Outputs: `experiments/many_cues/results/many_cues.json`, and in `results/large/`: `full.log`, `windows.pt` (the mined
  task), `{r1_t588_v452,many,one}_shifts.pt` (the shared and per-cue changes, a few MB).

## Review before the run

Codex is out of usage until 2026-11-18, and this agent could not spawn a reviewer. A self-review re-read every
measurement function against this plan; what a reviewer should still check is listed in the hand-off message to the
lead.

Predictions filed 2026-10-22 18:04:03, sha256 `1d778e7ae7d3d6d1cc0491fb41708e7fbe652b78fc0b1741390c184f318c092d`.

## Hand-off (2026-10-22, lead)

Ran clean on the cluster (debug job 29625, L40, 197 s). G1, G2, C_one passed; 32 cues mined toward ` of`; both K-cue
states scorable. Results: `experiments/many_cues/results/many_cues.json` (sha256 1ac2e9cd…2c1e), log
`results/large/full.log`, shifts in `results/large/`. Filed reading Q = "follows the cue": per-cue change works on both
states (0.90 many answers, 1.06 one answer); the shared change fails on many answers (0.04, random groups 0.04) and is
partial on one answer (0.55, cross-cue cosine 0.41 vs 0.14). The filed lean on the one-answer state ("shared") and the
combined lean ("follows the answer") read as misses by the filed rules, but the one-answer state's shared change is
well above the many-answers null; the analysis must report it as a partial shared part, not a null. Next: a fresh
session writes `reports/analyses/2026-10-22_analysis_many-cues.md` by `/writeup` together with E32's analysis (one story), adds
TRIED.md rows, queues log cards, and rewrites the pending scope-decision card with the per-cue option.
