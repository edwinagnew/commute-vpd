# Plan: reorient the fleet to the Qwen deliverables by 2026-11-11

*2026-11-03, written for the user after they asked why nothing had been achieved against the two-week plan of
2026-10-28 (`reports/plans/2026-10-28_plan_qwen-line-replan.md` §7). The user approved the direction in chat
("reorient ourselves around getting this qwen stuff working ... fewer checks for quicker runs ... we're under budget").
This plan is approved work for the roles it names. Words as in `GOALS.md` → Terms.*

**In plain words.** On 2026-10-28 the goal was set: sharp preliminary results on Qwen3-1.7B in two weeks. Six days in,
none of the three deliverables has moved, while seven of eight fleet roles worked the 67M lines. This plan stops the
67M lines, puts two leads on Qwen with one deliverable each, keeps the baseline line as a secondary comparator for the
Qwen readings, and cuts the per-run process to what a $0 or small run needs. Eight days remain.

## The deliverables (user, 2026-10-28, unchanged)

- **D1. Cleanly decompose a post-training update at scale.** A map Φ on Qwen3-1.7B that keeps the decomposition
  faithful for a real update, judged by a test that can fail a bad placement. Owner: `qwen-decomp-lead`.
- **D2. The same for SFT and for RL.** The same table for an SFT update and an RL (GRPO) update of the same model.
  Owner: `qwen-rl-lead` (new role) for the training; `qwen-decomp-lead` for the readings.
- **D3. Elicitation against new mechanisms, read from the decomposition.** Readings that separate two anchor updates
  whose novelty is fixed by design, then applied to the GSM8K arms. Owner: `qwen-decomp-lead`, with
  `baseline-lead` putting a cheap non-VPD comparator beside every row.

## What went wrong, so it is not repeated

1. The plan's first risk fired on day 1 (the pass test had no teeth: E44, Amendment 2) and was never fixed. The line
   was moved to "descriptive rows" on 2026-10-30 and D1 quietly dropped. **Rule now:** when a named risk fires, the
   lead's digest says "D<n> is blocked, day <k>" in its first line and proposes the fix or the fallback the same day.
2. The GRPO loop, named the schedule risk on day 0, was never assigned. **Rule now:** D2 has its own role.
3. The condition "if the line runs only this table" was broken: 67M work took most of the fleet and most of the advice.
   **Rule now:** 67M work runs only when the user asks for it by name.
4. A 90-second $0 job carried a plan, filed predictions, a code review, an independent check, an advice file, a digest
   and a card. **Rule now:** the fast lane below.

## Eight days (2026-11-03 to 2026-11-11)

| day | D1 (`qwen-decomp-lead`) | D2 (`qwen-rl-lead`) | baseline (`baseline-lead`, secondary) |
|---|---|---|---|
| 11-03 | Read this plan. Score the saved updates (E40 δW at steps 10 and 100; E61's gold and template states) under the pass test of §D1 below, as one cluster job | Read this plan and `2026-10-28_plan_qwen-rl.md` §3-5. Write the GRPO loop (`experiments/qwen_rl/`), rehearse at tiny size on the cluster | Write the 67M method table as one section of the write-up, no new runs |
| 11-04 to 11-05 | Pass test read. If the controls separate: the test stands, every saved update is scored and D1 has its first table. If not: file that as D1's finding and start teacher training (Q4, `experiments/qwen_sft/amend1.py`, rehearsed already) as the map | GRPO run (~$20-40 on an H100), null arm (shuffled rewards), SFT arm on the same prompts, δW saved at steps 1, 2, 5, 10, 20, 50, 100 | Build the cheap comparators for Qwen states: update size, the original model's rank of the trained answer, one steering vector |
| 11-06 to 11-07 | Map and test on the RL, null and SFT δWs as they arrive | Anchor 1 retrained at the learning rate E61 found; anchor 2 after the user's discussion of what it should be (see Open) | Comparators run on every Qwen state the other leads save |
| 11-08 | D3 rows (route split, where δW lands against the random null, behaviour kept under masks) on every arm, comparator beside each | Second seed where a row's verdict rests on one | Comparator table beside the D3 rows |
| 11-09 to 11-10 | `writeup-lead` restarted: one write-up, three tables, plain words | Spare for reruns | Spare |
| 11-11 | Preliminary results to the user | | |

### D1: the pass test, stated so it can fail

E44 showed the paper's adversary cannot fail do-nothing when Γ's masks come from the fine-tuned model's activations,
and under masks from the original model (reading B) no placement of a step-10 update keeps enough behaviour to pass
(13% per unit carried). So the test is built from the triple the 67M line settled on, not from the adversary alone:

- *(Corrected 2026-11-03, see `reports/plans/2026-11-03_plan_e68-d1-pass-test.md`: least squares is exact because 0.36 of the update goes to Δ, and the user's rule of 2026-10-29 allows a small change to Δ.)* Δ untouched is the criterion, not a reading: a map that puts anything in Δ is out. On Qwen the least-squares split is
  exact with Δ untouched, so it is the first candidate, with the adjoint split beside it.
- Behaviour kept under the state's own masks (reading A) on the task rows, as a ratio to the fine-tuned model's gain.
- The adversary once per experiment, as a ratio to the unchanged model's floor on the same rows, three seeds.
- Two built controls that must separate before any split is scored: the update placed on subcomponents Γ often
  switches on against the same update placed on ones it rarely does, and a random change of matched effect. The
  first pass of this on E44 Amendment 2 read 0.13 against 0.001 and 0.005 under reading B; under reading A the gap is
  what the first job measures. If after two jobs no reading separates the controls, D1's finding is "VPD's masked
  tests are blind to a step-10 update on this model by construction", stated with the numbers, and teacher training
  becomes the map: it is then judged on behaviour kept, the adversary ratio and Δ untouched, as on the 67M.

### D2: the GRPO loop

`reports/plans/2026-10-28_plan_qwen-rl.md` §3 is the recipe (Dr. GRPO form, 64 problems × 8 samples, AdamW 1e-6, the
196 decomposed matrices trainable, fp32 master weights, δW saved in fp32). It was held on 2026-10-28 for SFT first;
the hold is lifted. Changes: the SFT arm uses the model's own correct samples (the canonical update chosen 2026-10-28),
gold SFT is the off-policy contrast; the readout set H and stage G stay. Rehearse on a cluster L40 at tiny size
(`normal` QoS allowed for the Qwen line), then one H100 pod. This run is gated (over $60 or 6 h would be; it is not, so
it needs one review and the user's go-ahead on the cost line only).

### D3: anchors

Anchor 1 (behaviour the model already produces under a prompt; context distillation or the format template) stands.
Anchor 2 (content the model lacks) was invented facts in the 2026-10-28 plan; the user has asked why and wants to
discuss it in another session. Until then `qwen-rl-lead` builds no anchor-2 data; it is the one open item that gates
D3's second column.

## Roles

| role | state | task |
|---|---|---|
| `qwen-decomp-lead` | restart | D1, then D3 readings. Task text in §Tasks |
| `qwen-rl-lead` | new clone | D2 training and anchor arms |
| `baseline-lead` | restart, secondary | 67M method table write-up; cheap comparators on Qwen states; one iteration a day; no new 67M runs |
| `writeup-lead` | stopped until 11-09 | one write-up from the three tables |
| `commute-lead`, `regating-lead`, `redteam-lead`, `cards-lead`, `reuse-lead` | stopped, finished | no work. The 67M commute line is closed (2026-11-02); E62 is closed as the limit of the copying-mask work (user, 2026-11-03); the benchmark question on the 67M stays unanswered (option (a) of `decision-benchmark-next`). `fleet/delete` when convenient |

## The fast lane (process changes, user 2026-11-03: "fewer checks for quicker runs")

Written into `.claude/CLAUDE.md` and `fleet/roles/lead.md` in the same commit as this plan.

1. **Gate raised.** A run is gated at over $60 or over 6 h (was $15 and 3 h). Session budget `REMOTE_KERNELS_BUDGET`
   $100 (was $40). The user said the project is under budget; these two numbers are the lead's to spend without asking
   and the user can change either.
2. **Under the gate: no pre-run review, no independent check.** The lead reads the result itself, labels it, and moves
   on. Controls still run first with a stop rule; predictions are still filed.
3. **Gated: one review**, plan and code together, by one fresh Opus agent; the rehearsal for $0 stays where a cluster
   card fits the stage. "Two independent reviews" is gone.
4. **Independent check only for a headline claim**, one that goes into the write-up's answer paragraph.
5. **Two amendments per E-code, on every line** (the Qwen plan's rule 5, now general). A third is a new plan or the
   experiment is dropped with its reading as it stands.
6. **Thresholds come from measurement.** A filed rule's band is set from the control and the null read in the same
   job's first stage, not guessed. A "no call" verdict is reported as what was read and does not open an amendment.
7. **Defaults when the user is away.** An ask whose recommended option is under the gate and inside this plan is
   taken after two hours without an answer, recorded as "taken by default", with an FYI. Only gated spend, a change
   to D1-D3 or to `GOALS.md`, and anchor 2 wait for the user.
8. **Advice once per experiment**, at plan time; not per amendment.
9. **One log flush per lead per day**, plus one when a deliverable's status changes. Every digest's first line names
   the deliverable and the day of this table.

## Tasks (the text given to `fleet/start`)

**qwen-decomp-lead.** Read `reports/plans/2026-11-03_plan_qwen-reorient.md`. You own D1 and the D3 readings. Day 1:
one cluster job scoring the saved updates (E40 δW steps 10 and 100 from `experiments/SAVED.md`; E61's gold and
template states) under §D1's test, controls first with a stop. Decide by 11-05 whether the test stands or teacher
training is the map, and say which in the digest's first line. Then map and test every δW `qwen-rl-lead` saves, and
run the D3 rows on every arm. The fast lane applies. Your earlier whole-story write-up and `decision-e61-next` are
superseded by this plan.

**qwen-rl-lead.** Read `reports/plans/2026-11-03_plan_qwen-reorient.md` and `reports/plans/2026-10-28_plan_qwen-rl.md`
§3-5. You own D2: write the GRPO loop in `experiments/qwen_rl/` with shared pieces in `src/commute/`, rehearse it at
tiny size on the cluster for $0, then run RL, the shuffled-reward null and own-samples SFT on one H100, saving δW at
steps 1, 2, 5, 10, 20, 50, 100 to the volume and to `~/commute-data/results-large/`. Tell `qwen-decomp-lead` (a note in
its handoff folder and an FYI) when each δW lands. Then anchor 1 at E61's learning rate. Do not build anchor 2 until the
user has answered on it. The fast lane applies; the H100 run needs one review and the user's go-ahead on its cost line.

**baseline-lead.** Read `reports/plans/2026-11-03_plan_qwen-reorient.md`. Your line is secondary: one iteration a day.
First write the 67M method table as one write-up section (`reports/analyses/`), no new 67M runs; E65 and the key-swap
patch are not run. Then build cheap comparators that run on any saved Qwen state (update size, the original model's
rank of the trained answer, one steering vector) and run them on each state the Qwen leads save, so every D3 row has a
non-VPD reading beside it.

## Open

- Anchor 2: what fixes "content the model lacks" on Qwen. Invented facts is the current candidate; the user wants to
  discuss it. Decision needed by 11-06 for D3's second column to exist by 11-11.
- The two budget numbers above are the assistant's proposal, not the user's figures.

## Not in this plan

Any 67M run; RL on a benchmark other than GSM8K; thinking mode; a second decomposition; a Γ refit on Qwen.
