# Plan: what is the fixed change of Γ's importance values made of? (E32, `experiments/shift_anatomy`)

*2026-10-22, coding agent for the lead session. Follows E28 (`reports/2026-10-22_analysis_regating-tasks.md`). The
code is E32: E30 (`gamma_shift_map`) and E31 (`adjoint_sgd`) went to other sessions. Words as in `GOALS.md` → Terms.*

## The design check

**Approved classes.** Class 1 (GPU passes of the existing harness on regenerated states) and class 4: the re-gating
line's named next step on the log's "What's next" card is "a fixed change computed per key on the copy task and the
32-pair cipher". The other arms read the same states with the same harness. No sign-off needed.

- *Goal.* `GOALS.md` → the fixed-shift hypothesis and the log's Q5 (mechanism or leak). E28 left one open question: does
  the hypothesis fail on the key-value tasks because the change depends on the input, or because it depends on the
  answer? The pending scope decision turns on it.
- *Nearest tried.* E28 (one fixed change per task, eight states) and E18's amendments (the filed pattern, the shuffle,
  another window's values). Neither split the change by key, by strength, by matrix type, or in Γ's input space.
- *Patch or step.* A step: it tests the working story that the change has a fixed "when" part (spot the cue) and a
  "what" part that follows the answer, constant on the word tasks.
- *Working decisions.* Γ frozen, do-nothing placement, Δ out of the CI-masked model, as E28. No departure.
- *What changes next.* The decision tree below.
- *Control first.* G1, emoticon, first and alone; the job stops if it fails.
- *Numbers.* The table below; each has an expected reading.

## Restatement

E28's eight do-nothing states are regenerated in the job exactly as E28 made them. A 10-step Adam fine-tune of all 24
matrices puts δW in Δ, the released subcomponents stay, Γ is frozen, and Δ is left out of the CI-masked model. Every
arm writes some importance values at the trigger and keeps the original model's values elsewhere. It is scored as E28
scores its arms: recovery ρ = (lo − L₀₀) ÷ gap, and R = ρ ÷ ρ_trig, a share of what each input's own updated values
give back. The arms ask four things of the fixed change s = m₁ − m₀:

1. Does it work once it is computed per key?
2. Does each input move along s, only by a different amount?
3. Is the fixed part in the attention matrices and the input-dependent part in the MLPs (guess A)?
4. Is the change fixed in Γ's input space instead, like a steering vector (guess B)?

A fifth, on copy only: does each key's change follow that word's own subcomponents on ordinary text (guess C)?

## The numbers (trigger position, held-out windows, W₀'s values elsewhere)

At W₀, with no update, the gap is 0 and every share is undefined. Each builder fed a zero change gives back exactly 0;
that is the sanity line, checked on the control.

| number | definition | sanity / control | expected |
|---|---|---|---|
| R_mean_net | E28's (ρ(m₁) − ρ(m₀)) ÷ ρ_trig | G1: 0.8147 ± 0.02; G2: E28's per state | E28 |
| R_key_net | per key k (the token at the trigger): m₁ₖ, m₀ₖ over training windows with key k, given to held-out windows of key k; (ρ(m₁ₖ) − ρ(m₀ₖ)) ÷ ρ_trig | one key: equals R_mean_net (≤ 1e-3; word tasks, rand1) | ident32 ≥ 0.7; rand32 two-sided |
| R_group_net | the same with 32 random groups, 5 draws | word tasks: ≥ R_mean_net − 0.1 (12 windows suffice) | ident32 ≈ its R_mean_net (0.07) |
| R_strength | pre-activations p, g = clip(p) (released `leaky_hard`). d(x) = p₁ − p₀; s_p its training mean; a(x) = ⟨d, s_p⟩ ÷ ⟨s_p, s_p⟩ over alive subcomponents; clip(p₀ + a s_p) | a ≡ 0 gives back 0 | word ≥ 0.7; ident32 < 0.5 |
| R_preshift | the same with a ≡ 1, reported | — | near E28's filed pattern |
| var_frac | share of d's variance across held-out windows, about its mean, along s_p | — | larger on word tasks than on ident32 |
| corr_a_logp | corr of a(x) with the original model's log-prob of the target | — | negative, reported |
| corr_rem_logp | corr of lo(own) − lo(fixed m₁), per window, with that log-prob | — | word tasks: \|corr\| ≥ 0.3, negative (low) |
| R_attn_fixed / _ref | attention matrices at m₁ (ref m₀), MLP at each input's own g₁(x) | — | word ≥ 0.7; ident32 ≥ 0.7 and ref at least 0.2 lower |
| R_mlp_fixed / _ref | MLP matrices at m₁ (ref m₀), attention at g₁(x) | — | word ≥ 0.7; ident32 < 0.5 |
| R_steer | Γ re-run on W₀'s activations, each matrix's input at the trigger moved by the mean (updated − original) input over the training windows | zero move gives back 0 | word ≥ 0.5 on 3 of 4; ident32 < 0.5 |
| R_steer_own | each input's own updated input at the trigger only: this arm's ceiling | — | word ≥ 0.5 |
| Pile match | 32-key tasks: mean over keys of corr(sₖ − mean sₖ, cₖ − mean cₖ), cₖ = W₀'s importance contrast at Pile val rows 0..600 holding key k (keys with ≥ 50 occurrences), against 20 key derangements | control: the same with W₀'s m₀ₖ in place of sₖ passes | ident32 supported (low) |
| prev-token | s and "switched on" (s > 0.3) at `h.1.attn.q_proj:316`, `h.1.attn.k_proj:329`, VPD's previous-token subcomponents; on the 32-key tasks the spread of sₖ over keys | reported only | — |

**Why each arm, and what it is blind to.**

- *Per key* separates "the change depends on the key" from "on the whole input". It is blind to anything a key
  shares with its context. Per-key means rest on 9 to 15 training windows and are given to 1 to 7 held-out windows
  per key. The random-group arm of the same size is the null: it measures what averaging fewer windows does alone.
- *Strength* asks whether the inputs differ only in how far they move along s. Working before the clip removes
  the clip's distortion, so a(x) s_p = d(x) would give back each input's own values exactly. It is blind to direction
  changes that cancel in the mean.
- *Matrix type* tests guess A, that the "when" is in attention and the "what" in the MLPs. It is blind to a split
  inside a matrix type. The reference arm, the block at the original mean, shows whether that block's values matter
  at all.
- *Steering* tests guess B, that the change is fixed in Γ's input space. Γ is a bidirectional transformer over the
  window, and the update also moves the inputs at the other positions. So R_steer_own, the trigger-only ceiling, is
  needed to read R_steer. Blind to steering at other positions.
- *Pile match* tests guess C, that copy turns up each word's own subcomponents. Blind to subcomponents that matter
  but do not respond to the word on ordinary text.
- *Previous-token subcomponents*, reported only. If they are switched on at the key with a near-constant sₖ, the
  fixed "when" part includes the model's previous-token machinery reading the marker.

**Noise floor.** E28's readings are deterministic per state; its control matched E18 within 0.002, and E18's two seeds
agreed within 0.02. The random-group draws give the per-key arm's own sampling spread.

**Gates.** G1 (emoticon, first, alone): r within 0.05 and R_mean_net within 0.02 of E28 (0.5965, 0.8147). The
per-window read must match E28's original values at the trigger and E28's fixed change within 1e-4. A zero change
through the strength and steering builders must give back 0 within 1e-4. A fail stops the job. G2, per state: r and
R_mean_net reproduce E28's within the same tolerances; a fail makes the state "reported, not scored".
Scorable as E28: gain ≥ 1 nat, gap ≥ 0.5 nats, ρ_trig ≥ 0.5.

## Predictions (`experiments/shift_anatomy/results/predictions.json`, filed before the run)

Absolute rules throughout; the full text is in the file.

- **C_key, C_size** (controls): one-key states read R_key_net = R_mean_net; random groups lose at most 0.1 on word tasks.
- **K_ident32**: holds per key (≥ 0.7), and beats random groups by ≥ 0.2. Low-medium confidence. **K_rand32**: two-sided.
- **S_word**: R_strength ≥ 0.7 on every word task. **S_ident32**: < 0.5, "not one direction". Low-medium.
- **Rem_word**: the word tasks' remainder tracks the original log-prob, |corr| ≥ 0.3 on 3 of 4. Low.
- **M_word**: both mixed arms ≥ 0.7 on the word tasks. **M_ident32**: guess A holds. Low.
- **B_word**: steering ≥ 0.5 on 3 of 4 word tasks, else guess B is unreadable. **B_ident32**: guess B fails. Low.
- **P_control, P_ident32**: the control passes; guess C supported. Low.

## Decision tree

- **G1 fails:** stop and report; nothing is scored.
- **Per key holds on copy and the key matters:** the fixed-shift hypothesis is restated as one fixed change per
  cue-and-answer pair. The scope question becomes "per answer", not "tasks the model already does". Matrix type and
  the Pile match then say where the per-key part lives. Next: the new word task with many cues and one answer, already
  approved.
- **Per key fails, strength or steering holds on copy:** the input dependence is a strength effect or a coordinate
  effect. The hypothesis survives in that form; the user decides whether to restate it.
- **Per key, strength and steering all fail on copy:** the switching on copy depends on the input beyond its key.
  E28's scope branch stands: the hypothesis covers tasks the model already does. The user decides.
- **A word-task control fails** (C_key, C_size, S_word, M_word or B_word): that arm is not read on copy. If the
  failure bears on the hypothesis itself, as S_word or M_word would, stop and tell the user.

## Cost and platform

E28's eight states took 142 s on a RunPod 4090, model load included. E32 drops E28's wrong-target fine-tune and its
shuffles. It adds four per-window reads per state, about 25 masked forwards and three Γ re-runs, plus one Pile pass
over 600 rows. Expected 3 to 5 min on a 4090, 5 to 8 min on an L40, under 12 GB of GPU memory, under 2 GB of CPU
memory. Cluster `debug` first (`cluster/shift_anatomy.sbatch`, `--time=00:20:00`); RunPod 4090 if it has not started
in 10 min, under $0.50.

## Not run (by design)

Seeds beyond 0; the wrong-target and shuffle nulls (E28 ran them on these states); steering at positions other than
the trigger; the per-key arm on SGD or small-update states; Pile matches for the word tasks (one key each: no
derangement null); per-key strength or mixed arms.

## Code

`src/commute/shift_anatomy.py`: the per-window read `trigger_read` (Γ's values, pre-activations and each matrix's
input at the trigger, by `vpd.compute_ci`'s path); the input edits; the pure bookkeeping (`group_means`, `per_window`,
`along`, `strength`, `mixed`, `key_match`); and the arms `group_fixed` and `steer`. CPU tests in
`tests/test_shift_anatomy.py` run on the toy ComponentModel. `experiments/shift_anatomy/run.py` reuses
`regating.prepare`, `finetune`, `measure` and `load_*`, and `turned_down.contrasts` for the Pile pass. `plot.py` waits
for the results.

## Hand-off (pod-runner brief lines)

- Command: `cd /mnt/nw/home/e.agnew/commute && sbatch cluster/shift_anatomy.sbatch`, after `cluster/sync.sh up`.
  On a pod: `bash cluster/pod_setup.sh`, then `.venv/bin/python experiments/shift_anatomy/run.py`.
- Inputs not in git: as E28 (`experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt}`,
  `experiments/task_ladder/results/large/windows_<4 tasks>.pt`, `experiments/kv_task/results/large/windows_{ident32,
  rand32,rand1}.pt`; all on the laptop too). Pile val rows 0..600 download on first use (cached on NFS). The run checks
  the files before loading the model.
- Early stop: the job exits by itself if G1 fails (`status: "stopped at G1"`); report the G1 line of the log.
- Outputs: `experiments/shift_anatomy/results/shift_anatomy.json` (the results) and `results/large/full.log`.

## Review before the run

The Codex adversarial review could not run: two attempts returned no review, the second with "usage limit, try again
2026-11-18". An independent reviewer agent read the four files and the shared code instead. It found no bug that
would silently change a number. Fixed from its list:

- the control rules gave a vacuous pass when no word task was scorable; they now read "none";
- the Pile predictions are scored only when ident32 is scorable;
- the remainder rule now records its sign;
- the read check is named for what it compares, this job's own E28-path reads;
- the largest change on dead subcomponents at the trigger is logged, since the strength arm keeps W₀'s values there;
- repeated means were moved out of loops.

Skipped: rounding of the steering move to the cache's dtype. Γ casts its input to bf16 anyway.

Predictions filed 2026-10-22 17:43:55, sha256 `1dc6755ebd8b5198e9988f004166005ef5c5904ba2b1bba6e24aeca3ebf1537c`.

## Hand-off (2026-10-22, lead)

Ran clean on a RunPod 4090 (118 s, $0.21; cluster debug job 29618 did not start in 10 min and was cancelled). G1 and
G2 passed on all eight states. Results: `experiments/shift_anatomy/results/shift_anatomy.json`, log
`results/large/full.log` (sha256 in the runner's report: json 0afad63a…a4b5, log 8c4ba160…c19c). Next: a fresh
session writes `reports/2026-10-22_analysis_shift-anatomy.md` by `/writeup`, scores each filed prediction from the JSON,
adds the E32 row to `TRIED.md`, queues the log card in `docs/research_log/PENDING.md`, and updates the pending scope
decision card (re-gating line) with the per-key result. The lead's first read, to be checked against the JSON: the
fixed change holds per key on copy and the 32-pair cipher (0.84, 0.81, against 0.07 for random groups); the steering
arm, the strength arm and the Pile-signature test all fail on copy while working (or having power) on word tasks; in
the attention/MLP split, the attention matrices' change barely matters on any task and the MLP matrices carry the
switching, so the filed ident32 attention prediction (reference ≥ 0.2 lower) reads as a miss.
