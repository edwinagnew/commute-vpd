# Plan: re-run E14–E20's measurements on small updates (2026-10-22)

*Lead session, after `reports/reviews/2026-10-22_review_large-update-scope.md` (the scope review) and
`reports/analyses/2026-10-22_analysis_sgd-vs-adam.md` (E21). User's instruction 2026-10-22: re-run the relevant experiments with
one Opus subagent per experiment, RunPod allowed, be thorough, then update the research log. Vocabulary: `GOALS.md` →
Terms.*

**In plain words.** The scope review found that most of E14–E20's failures were read on one update, the large Adam
fine-tune at step 10, or on Adam component training at its first learned step, and that E21's small updates pass the same
checks. Its "Not run" list names the measurements that were never taken on a small update. This plan runs them: the
placements of a small fine-tune at later steps, the outside-share series along it, the cancellation and complement
decompositions on the failures that persist after E21, the re-gating pattern test on a small update, and the step
split along a small path. Each is a replication of an existing experiment with the update changed, so each falls in
the approved classes; the user's instruction is the go.

## Design check (one for the set; each experiment restates the parts that differ)

- **Goal served.** Q1 (the weak version: does some map with Γ frozen keep the decomposition faithful) and the
  2026-10-21 decision's move-back condition. The review says the case against frozen-Γ maps rests on narrower failures
  than the log states; these runs decide whether the E14–E20 mechanism claims (where the update sits, what the
  adversary's excess is, what re-gating is, what the path does) hold on the updates that pass.
- **Nearest tried.** Each experiment below names its parent (E14, E16, E17, E18, E19, E20). What is new is only the
  update: SGD 0.01 and Adam 3e-6 emoticon fine-tunes (E21's matched pair, size 0.0024 and 0.0022 at step 10) and
  E21's saved SGD component-training states.
- **Patch or step.** Controls and replications (approved class 2) and GPU passes of existing measurements on saved
  states (class 1). None opens a new sub-question.
- **Decisions relied on.** Γ and Δ frozen; reading A as the pass criterion; the harness's same-job W₀ floor; component
  training is optimiser-agnostic. No departure.
- **What would change what we do next.** If the small fine-tune's least-squares and do-nothing states pass at steps 100
  and 1000, claim rows 1 and 4 of the review are "only large" at every step and the headline claim's fine-tune leg
  goes. If the SGD step-1000 failures have the same excess structure as the Adam step-10 ones (E20's interaction term),
  the account generalises; if their ratios converge to W₀'s under a longer adversary, the remaining component-training
  failures are budget artefacts. If re-gating on a small update needs the per-window values (recovery of the fixed
  pattern low), H is scoped to the large update.
- **The control runs first, alone, with early stop.** Every job starts with W₀ through the same measurement (the Pile
  floor over two or three adversary seeds, `faith_score.w0_check`; for the revert tests W₀'s own revert costs; for the
  budget sweep gate C3's 20-step reproduction) and stops if it fails. The second stage is the reproduction gate: the
  reading that already exists for that state (E21's step-10 numbers; sgd_finetune's split.json) within the floor's
  spread.
- **Numbers and expected readings.** Listed per experiment below. Each experiment reports only those, plus its positive
  control.

## The shared recipe

The small fine-tunes are E21's (`experiments/sgd_finetune`): arm A on emoticon, seed 0, all 24 decomposed matrices,
`finetune.finetune_arm_a` with `ArmAConfig(optimizer="sgd", lr=0.01)` and `ArmAConfig(optimizer="adam", lr=3e-6)`; the
large reference is `ArmAConfig(optimizer="adam", lr=3e-5)` (E14/E16). Trained to 1000 steps with checkpoints at E9's
steps (1, 2, 3, 5, 10, 30, 100, 300, 1000) where a series is read, or at (10, 100, 1000) where only states are read.
The step-10 δW of the SGD fine-tune must reproduce `experiments/sgd_finetune/results/large/sgd_ft_lr0.01_step10.pt`
(cluster NFS; the Adam 3e-6 one is regenerated the same way `adam_matched` did). Harness: `harness.measure` with 256
Pile rows, 20 PGD steps, two adversary seeds for the floor; pass ≤ 1 + f, fail ≥ 1 + 4f with f the same-job floor
spread (`faith_score.verdict`); the ratio's noise is W₀'s seed-to-seed spread (0.21 on the L40, read fresh per card).

Inputs not in git: `experiments/transport_test/results/large/alive.pt` (local) and `windows.pt` (cluster only,
`/mnt/nw/home/e.agnew/commute/experiments/transport_test/results/large/windows.pt`, 155 KB: `scp` it to the same local
path before a pod sync), `experiments/transport_test/results/windows.json` (local), `experiments/task_ladder/results/
large/windows_<task>.pt` (local). E21's saved states are on the cluster only (60 MB each, bf16 offsets written by
`arms.save_offsets`); the experiment that reads them runs on the cluster.

## The experiments

| code | directory | parent | question | where |
|---|---|---|---|---|
| E22 | `experiments/small_finetune_splits` | E14 part 1, E16 | the three placements of the small fine-tunes at steps 10, 100 and 1000 on one card beside the large one, and the four blocks and rescaling share along the SGD and matched-Adam paths | RunPod |
| E23 | `experiments/cancellation_sgd` | E17, E20 (+ Amendment 3's budget sweep) | the excess decomposition and the adversary's budget curve on the component-training failures that persist after E21, beside their passing twins | cluster (states there) |
| E24 | `experiments/regating_small` | E18 | does the fixed shared pattern recover the re-gating effect for a small update | RunPod |
| E25 | `experiments/step_split_small` | E19 | the step split along the SGD and matched-Adam paths | RunPod |

### E22 small_finetune_splits (rows 1, 3, 4, 5, 8, 9 of the review)

Run order: W₀ floor (seeds 0, 1; stop if `w0_check` fails) → SGD 0.01 fine-tune to 1000 with the nine checkpoints,
gate G1: step-10 δW matches the saved one (`rel_norm_max` within 1%) and its do-nothing ratio within f of sgd_finetune's
1.12 → Adam 3e-6 the same (gate against adam_matched.json's 1.08) → Adam 3e-5 (gate against E16's step-10 do-nothing
2.29 ± card difference: read as fail) → per fine-tune and step in (10, 100, 1000): do-nothing, least-squares split,
scaled adjoint split through `harness.measure` (Pile ratios, route split, Δ's share of ‖δW‖², held-out P(target)) →
per fine-tune and all nine checkpoints: `split.block_table` (four blocks with null, algebraic block) and
`split.rescaling_share` with its random null, as E14 part 1. Results dumped after every state.

Numbers and expected readings. Adversarial ratio: W₀ 1; small do-nothing at step 10 ≈ 1.1 (pass, E21); large do-nothing
at step 10 ≥ 1.8 (fail, E16 2.29 and E21 2.18) and at 100 ≈ 1.1. Δ change fraction of the least-squares state: 0 for the
adjoint state (positive control: adjoint keeps Δ), 0.25–0.32 for the small fine-tunes at step 10 (E21). P(target) under
the adjoint split: 0.69–0.71 at step 10 (E21). Outside-both ÷ null: large fine-tune 0.18 at step 10 rising to 0.41 at
1000 (E14). Rescaling ÷ random: large 2.0 at step 10, 9.6 at 1000 (E14).

Predictions to file (two-sided where the review says untested): P1 the small fine-tunes' do-nothing and least-squares
states pass at steps 100 and 1000 (the review's "only large" reading; the SGD fine-tune's size at 1000 is unknown and
may cross E21's SGD size cut of 0.009, so a fail there is the size story, a pass the step story); P2 the adjoint states
pass at every step with Δ unchanged and P(target) falling with step; P3 the SGD fine-tune's outside-both ÷ null stays
below the large one's at every checkpoint (0.034 vs 0.218 at step 10); P4 its rescaling share's climb along training
(E14's 2× → 9.6×) is read two-sided.

### E23 cancellation_sgd (rows 10, 12; E21's "what next" item 1)

States, all from the cluster's `results/large/`: W₀; SGD 0.03 at step 1000 on r0_t352_v310 and r1_t534_v369 (fail,
2.82 and 3.00); Adam 1e-5 at step 1000 on the same tasks (pass, 1.07 and 0.95, the same size of change); SGD 0.01
rand32 and SGD 0.003 ident32 from `sgd_kv` (fail, 4.16 and 2.37); SGD 0.1 at step 100 on r2_t275_v2667 (fail, 5.20);
SGD 0.03 at step 100 on r0 (pass, 0.92, the positive control: a passing state's excess is ≈ 0). The states are bf16
offsets (`arms.save_offsets`: `{"offsets": {k: (dU, dV)}, ...}`), not E11's (δW, U2, V2): add one loader to
`src/commute/saved_states.py` that rebuilds (δW, (U2, V2)) from the offsets and the alive (U, V) at W₀, with a CPU test
that the rebuilt δW equals U2 V2ᵀ − U Vᵀ.

Stage A (two debug shards): W₀ and the gate first: each state's 20-step adversarial KL ratio reproduces E21's verdict
(same-job floor; stop a shard whose W₀ floor is off). Then E17's three tests (coherence against its null, half-revert,
adversary-revert) and E20 Amendment 1's split of the excess (alive removal alone, dead removal alone, both, Δ′ alone,
interaction = both − alive − dead), each saved as E20 did. Stage B (one or two shards): the Amendment 3 budget sweep
(`gamma.pgd_sources`, 20/40/80/160 steps at step size 2/n, seed 0; W₀ seeds 0 and 1) on W₀ and the four step-1000
states. Do not touch `experiments/cancellation_complement/` or its results: another session is running its Amendment 3
there now.

Expected readings. Positive control SGD r0 step 100: excess ≈ 0 (within W₀'s revert noise). Failing states: the
20-step excess as E21's ratios. E20 Amendment 1 on the Adam step-10 fails put the excess in the alive–dead interaction
with Δ′ negligible; two-sided here. Budget sweep: ratio of SGD step-1000 ÷ W₀ at 20 steps ≈ 2.8–3.0; P: stays ≥ 1.5 at
every budget (as Amendment 3's P9) against the null that it converges to Adam's; Adam step-1000 stays ≤ 1.24 (P10's
twin).

### E24 regating_small (row 11)

E18's scored run on the do-nothing states of the small fine-tunes at step 10: emoticon with SGD 0.01 and Adam 3e-6,
seed 0, beside the large Adam 3e-5 as the gate (reproduce E18's r 0.60 and its recovery readings for emoticon seed 0,
`experiments/regating_leak/results/regating_leak.json`). Patterns trig, shared, mean, cross (' http' null fine-tune by
the same optimiser and rate), random (20 draws); recovery; S_on size; r. If a 10-step SGD rate for r1_t534_v369 passes
E4's rule on a five-rate grid (`sgd_finetune`'s SGD_LRS) within the job, add that task; otherwise emoticon only and
say so. The pattern functions live only in `experiments/regating_leak/run.py`, which another session is editing: move
what E24 needs into a new `src/commute/regating.py` (copied logic, with a CPU test on a toy that the five patterns
reproduce the run's definitions), and do not edit `experiments/regating_leak/` at all.

Expected readings. r: E18 read 0.60 on emoticon seed 0 for the large fine-tune; the SGD state's trigger share with Γ on
the original activations was 0.01 (E21: re-gating carries it too). Recovery of the shared fixed pattern: E18 read it
high on emoticon (H: a fixed switch pattern); two-sided for the small updates. Cross pattern ≈ 0 (positive control
that the test separates patterns).

### E25 step_split_small (row 13)

E19's run with the fine-tune recipe swapped: the SGD 0.01 and Adam 3e-6 paths, `split.StepSplit` carrying the bases
(and `carry=False` for the consistency numbers), states at steps 10, 100, 1000 through `harness.measure` beside
do-nothing and the least-squares split; W₀ floor first, gate: the step-10 do-nothing ratios reproduce sgd_finetune's
(1.12, 1.08 within f). Numbers: the behaviour kept in the pieces (P(target) of the step-split state) against the
least-squares state's; Δ's share of ‖δW‖²; the piece change; the adversarial ratio. E19 on the large path read the
step-split state at 2.67 at step 1000 and less behaviour in the pieces than least squares. Two-sided here; the
prediction with a direction: Δ's share under the step split is lower for SGD than for Adam 3e-6 (SGD component training
≈ its adjoint split, cosine 0.957, E21).

## Rules for the agents

- One Opus agent per experiment, end to end: restate this check for its experiment at the top of its own plan
  section (append to this file under "Agent notes"), write `run.py` (+ `plot.py`), file `results/predictions.json`
  before running, `make lint` clean, Codex adversarial review on the diff before a paid run, hand the GPU run to a
  `pod-runner` agent per `.claude/skills/pod-runner/SKILL.md`, then the analysis report
  `reports/analyses/2026-10-22_analysis_<topic>.md` per `/writeup` with `check.py`, the `TRIED.md` row with its E-code, one
  line in `docs/research_log/PENDING.md`, and the `README.md` codes table row.
- Shared files (`experiments/TRIED.md`, `docs/research_log/PENDING.md`, `README.md`, this file) are edited by several
  agents and by another session: re-read right before each edit, append only, never rewrite others' lines.
- Do not edit `experiments/regating_leak/`, `experiments/cancellation_complement/` or their results (another session is
  running amendments there). Finished experiments' `run.py` stay as they ran; shared steps go into `src/commute/` as
  new modules or new functions, never by changing an existing function's behaviour.
- Budget: RunPod runs share the session's $40 `REMOTE_KERNELS_BUDGET`; caps per run: E22 $8, E24 $6, E25 $6 (E23 is on
  the cluster, $0; if the cluster does not start its jobs within 10 min it may regenerate the states on a pod under a
  $6 cap, with a gate that each regenerated state reproduces E21's reading). Terminate pods as soon as the results are
  downloaded and verified.
- Never load the model on the laptop. The scored run is its own smoke; W₀ first.

## Not run (by this plan)

- E15's stage 0 on a small update (no fragility to explain) and row 2's LoRA-vs-fine-tune contrast at a matched size.
- A second training seed or a second adversary seed per updated state.
- Any reading of E20 Amendment 3 or E18 Amendment 4 (the other session's).

## Agent notes

### E22 small_finetune_splits (agent, 2026-10-22)

Design check as above; restated for E22. **Goal**: rows 1, 3, 4, 5, 8, 9 of the scope review: is the small fine-tunes'
pass at step 10 a property of their size or of step 10. **Nearest tried**: E16 (placements, large only), E14 part 1
(series, large only), E21 (small at step 10 only). **Class** 1–2, no sign-off.

Arms: SGD 0.01, Adam 3e-6 and Adam 3e-5 emoticon fine-tunes (seed 0, 1000 steps) on one card. Order: W₀ floor (seeds 0,
1; stop on `w0_check` fail) → per fine-tune: train, its step-10 do-nothing / least-squares / adjoint states and gate G1,
then the series at all nine checkpoints → steps 100 and 1000 of all three. The cheap series moved ahead of the late
states (the plan has it last) so every code path runs in the first ~10 min; nothing else is reordered.

Numbers and expected readings (all in `results/predictions.json`): adversarial ratio (W₀ 1; small do-nothing at 10 ≈ 1.1;
large ≥ 1.8 at 10, ≈ 1.1 at 100, 1.4–1.5 at 1000); Δ's share (adjoint 0 = positive control; least squares 0.25–0.32
small at 10); adjoint P(target) (0.69–0.71 small at 10; large 0.93 → 0.60 → 0.10); outside-both ÷ null (large 0.18 →
0.41; SGD 0.034 at 10); rescaling ÷ random (large 2.1 → 11.4 on the non-saturated pooling, 2.0 → 9.6 over all 24).

What I changed and why:
- Verdicts use `faith_score.verdict` (pass ≤ 1.5/(1+f), fail ≥ 1.5(1+f)), as E21 did; the plan's "≤ 1 + f / ≥ 1 + 4f"
  wording does not match that function, and E21's readings that the gates reproduce were made with the function.
- G1's do-nothing tolerance is relative max(f, 0.209): the reference was read under the L40's 0.209 floor, and the run
  goes to a RunPod card whose floor differs. G1 compares to E21's recorded rel_norm_max (1%), not the saved δW (cluster
  only, not uploaded).
- A failed G1 marks the predictions reading that fine-tune unscored but the run continues (the measurements are cheap);
  P3 reads only G1's size check, since the blocks depend on δW alone.
- Added G2 (the large series reproduces E14's within 0.03 / 25%) and C1 (random draws' outside-both at its analytic
  null) as the series' controls. New shared function `split.rescaling_table` (E14's `matrix_record` without the
  pieces), with a CPU test.
- P3's "below" = SGD ≤ 0.8 × large at every checkpoint (E21's "lower"). P4 is two-sided with four named outcomes.
  Series scored on the non-saturated pooling, as the brief asked; the all-24 pooling is reported.

### E25 step_split_small (agent, 2026-10-22)

Design check as above; restated for E25. **Goal**: row 13 of the scope review: do E19's step-split readings (less
behaviour in the pieces than least squares, Δ above the update's energy by step 1000, fail at 2.67) hold on a small path.
**Nearest tried**: E19 (the large Adam 3e-5 path only), E21 (the same small fine-tunes at step 10 only). **Class** 2, no
sign-off.

Arms: the SGD 0.01 and Adam 3e-6 emoticon fine-tunes (seed 0, 1000 steps), each with `split.StepSplit` carried and
fixed-bases stepping after every optimizer step (float32, as E19); at steps 10, 100, 1000: do-nothing, the step-split
state and the least-squares split through `harness.measure` with the Pile. Order: W₀ (seeds 0, 1, `w0_check`, stop) →
SGD path (gate on rel_norm_max first, then do-nothing at step 10 first) → Adam path. Card: RunPod.

Numbers and expected readings (all in `experiments/step_split_small/results/predictions.json`): adversarial ratio (W₀ 1;
do-nothing at 10 ≈ 1.12 SGD, 1.08 Adam); Δ's share of ‖δW‖² under the step split (P1, directional: SGD below Adam at
every step; expected SGD 0.4–0.9, Adam 0.9–1.2; E19 large 1.03–1.21); verdict at 1000 (P2, two-sided, E19 fail 2.67);
behaviour in the pieces (P3, two-sided). Positive control: carried and fixed bases give identical pieces after step 1
(gap ≤ 1e-6, expected 0).

What I changed and why:
- **P3's "P(target) of the step-split state"** is read as P(target) with every component on and Δ out (the route split's
  unmasked cell `p_cells["U1"]`). The step-split state's own P(target) is the fine-tune's by construction (E19's G2), so
  it cannot separate anything. U1 is already in every `harness.measure` record and E19 has it for the large path (0.97 /
  0.94 / 0.81 against least squares' 1.00 / 0.99 / 0.98), so the comparison is on the same axis. Rule: |diff| < 0.05 is
  "same". E19's e_Δ is reported beside it.
- Verdicts use `faith_score.verdict`, and the do-nothing gate's tolerance is max(f, 0.209), as E22 did and for its
  reasons. A failed rel_norm_max gate skips that path, not the job.
- E19's outside-both bookkeeping (C1) and the fixed-vs-one-shot piece gap were dropped; the energy shares, piece
  sizes and second-order shares moved to a new `src/commute/step_split.py` (`numbers`, `carry_fixed_gap`), with CPU
  tests in `tests/test_step_split_readouts.py`. `split.py` and `finetune.py` unchanged.

### E24 regating_small (agent, 2026-10-22)

Design check as above; restated for E24. **Goal**: row 11 of the scope review: does one fixed pattern of Γ's values
(E18's shared change) recover the switching effect of a small update, as it did (0.73 of the trigger's own values) for
the large one? **Nearest tried**: E18 (large Adam 3e-5 only, three tasks), E21 (the small states' r, never the
patterns). **Class** 2, no sign-off.

Arms, in run order (one job, ~5 min of GPU): G1 the large emoticon state (Adam 3e-5, seed 0, regenerated) through every
pattern, gated against E18's seed-0 readings (r within 0.05; ρ_trig, ρ_shared, ρ_mean within 0.02, amendment 1's repeat
tolerance; W₀'s gates recover 0); stop on fail → SGD 0.01 and Adam 3e-6 emoticon states, each gated on E21 (held-out P
within 0.01, r within 0.05 of sgd_finetune's 0.617 and 0.594), each with its own ' http' null by the same optimiser and
rate → r1_t534_v369: the five-rate SGD grid at step 10 with E4's rule (P ≥ 0.9, KL elsewhere ≤ 0.243, sgd_ladder's
limit); if a rate passes, the task's Adam 3e-5 state (gate against E18, r within 0.1) and the SGD state → J_nat on 600
Pile rows (reported).

Numbers and expected readings (`experiments/regating_small/results/predictions.json`): r (large 0.60; small ≈ 0.6,
already read by sgd_finetune, so P1 is near a reproduction); R_shared = ρ_shared ÷ ρ_trig (large 0.73; small two-sided,
≥ 0.7 "fixed pattern", ≤ 0.3 "per-window"); ρ_cross ≤ 0.1 (positive control, E18 −0.08 to −0.30); random null ≈ 0;
W₀'s gates 0 exactly.

What I changed and why:
- Platform: the job is ~5 min, so it goes to the cluster's debug QoS (RunPod after 10 min without a start), not RunPod
  as the table says; `windows.pt` is already local, and the cluster has every input.
- The large state is regenerated and fully re-measured in-job (not read from E18's file) so the small states are read
  beside it on the same card; E18's numbers gate it.
- Shared steps in a new `src/commute/regating.py` (E18's logic, copied), with `tests/test_regating.py`;
  `recruit.py` and `experiments/regating_leak/` untouched.

### E23 cancellation_sgd (agent, 2026-10-22)

Design check as above; restated for E23. **Goal**: rows 10 and 12 of the scope review and E21's "what next" item 1:
does E17/E20's account of the adversary's excess (no cancellation; the excess in the joint removal of the alive and
dead blocks, Δ′ negligible) hold on the failures that survive E21, and do the step-1000 SGD-vs-Adam verdicts hold at
larger budgets? **Nearest tried**: E17, E20 + amendments 1 and 3 (Adam step-10 states only). **Class** 1, no sign-off.

States (cluster NFS, bf16 offsets rebuilt by the new `saved_states.offsets_state`): control SGD 0.03 r0 @100 (E21 0.92,
pass); SGD 0.03 r0/r1 @1000 (2.82, 3.00); Adam 1e-5 r0/r1 @1000 (1.07, 0.95, passing twins); SGD rand32 (4.16), ident32
(2.37), SGD 0.1 r2 @100 (5.20). Stage A (two debug shards, ~26 and ~20 min): W₀ Pile harness, 3 seeds (`w0_check`,
floor f) → W₀ through the new `excess.measure` (128 rows, seed 0) gated G0 against E20 amendment 1's W₀ (F_all 0.6604,
F_orig 0.4868, F_dead 1.070, F_dense 0.0068, F_rest 1.075, within 1e-3 rel) → W₀ seed 1 (the null decomposition) →
per state: C1 (harness ratio gives E21's verdict and is within 25%), then E17's coherence/half/adversary reverts and
E20's parts X = O + F_adv + I + D + K + P + J. Stage B (two debug shards, ~25 min each, after A shard 0): amendment 3's
budget sweep (20/40/80/160 steps of 2/n) on W₀ (seeds 0, 1) and the four step-1000 states; C3: 20 steps reproduce
stage A's F_all within 1e-3 rel.

Numbers and expected readings: P0 control |X|, |R|, |J| ≤ f × F_all(W₀) ≈ 0.14 nats (expected X ≈ −0.05); P1 E20's
account (X > band, R ≥ 0.5 X, reading '4', |D| ≤ 0.05 X) two-sided over the five failures, decision at ≥ 4 / ≤ 1; P2
E17's readings (κ ≥ 1, h ≤ 0.5, F_adv ≤ 0.2 X) two-sided likewise; P3 SGD @1000 ratios ≥ 1.5 at 40/80/160 (P9 twin);
P4 Adam @1000 ≤ 1.241 (P10 twin, stage B's positive control); P5 max/min ≤ 1.5 on ≥ 3 of 4 (P11 twin).

What I changed and why:
- E20's per-model arms moved into a new `src/commute/excess.py` (E17's and E20's logic, copied; G0 checks it
  reproduces E20 at W₀) with a CPU test of the exact split; `cancel.py` and both parents' run.py untouched.
- Dropped E20's importance/layer bins and permuted-originals arm (no prediction reads them); kept E17's F_perm (P2
  reads it). The harness's C1 is a 256-row reading (E21's own measurement), the parts are 128-row (E20's); the band
  converts the 256-row floor to nats at 128 rows.
- Stage B's W₀ seed 0 runs in shard 0 only; shard 1 re-reads it at 20 steps as its control (C3), so its ratios use
  shard 0's W₀ (deterministic across jobs: amendment 3's C3 read rel 0.0).
