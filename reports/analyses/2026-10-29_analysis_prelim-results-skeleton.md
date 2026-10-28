# Skeleton: the preliminary-results write-up, about 2026-11-11

*2026-10-29, writeup-lead. A skeleton, not the write-up: it fixes what each line can claim today, what evidence sits under
each claim, which figure would carry it, and what the write-up cannot yet say. Sources: the research log's parts after
v88 (`docs/research_log/parts/sections/{claims,questions,overview,gaps,techniques}.html` and the cards), `GOALS.md`, the
Qwen re-plan's §7 (`reports/plans/2026-10-28_plan_qwen-line-replan.md`) and the technique review's §2
(`reports/reviews/2026-10-28_review_technique-comparison.md`). Three readers extracted the claims, one per line; their
notes are summarised here and every row cites the log. Vocabulary: `GOALS.md` → Terms.*

We read the research log's claims, questions, techniques and gaps against the project's aim and sorted every claim into
three lines of work. On the small model, one map, teacher training, copies narrow fine-tunes into the decomposition while
keeping behaviour and faithfulness, and the masked stand-in reads most narrow gains as changed importance values; on the
larger model nothing about an update has been scored yet. A write-up in two weeks can report a partial yes on the small
model, a measuring instrument rather than a mechanism for the second line, and at most a first table on the larger model.

## 1. The question, in the user's words

> Given a (VPD) parameter decomposition, can one commute small gradient updates from post-training past the parameter
> components. If so, this gives "continuous interpretability" (within some limit). If not, is it because post-training
> teaches new capabilities (vs strengthens existing ones), or maybe just parameter decompositions are too brittle.

Made precise by the user on 2026-10-14: is there a simple operation that maps a post-training update to an update on the
subcomponents, such that the resulting decomposition is faithful for the new model, with Δ not absorbing the update? The
write-up answers the weak version only; the strong version, matching a re-decomposition, is postponed in `GOALS.md`.

The write-up's goal, from the re-plan's §7, is three deliverables: **D1** cleanly decompose a post-training update at
scale, **D2** for both SFT and RL, **D3** readings from the decomposition that tell elicitation from a new mechanism,
calibrated on two anchor updates whose novelty is fixed by design.

**Proposed answer paragraph, to be rewritten once Qwen numbers exist.** On a 67-million-parameter model, training the
subcomponents to copy a finished fine-tune keeps its behaviour, keeps the decomposition faithful and leaves the residual
untouched for narrow fine-tunes, including one teaching a cipher the model lacked. It fails for a broad fine-tune and
needs Adam. No simple split of a finished update does all three. On the 1.7-billion-parameter model the decomposition is
already strained on maths text before any update, and the first maps are being scored now.

## 2. The three lines and what each can claim now

### Commute line: log question Q1 with Q2, Q4, Q8 and Q11, 67M model

The log's verdict on Q1 is "Partly". Teacher training is "the first map of a finished update that keeps the behaviour,
faithfulness and Δ together" (`overview.html:159`). That holds on three trigger tasks, from task text or plain Pile text,
at two batch orders and under a longer adversary, and on three cipher tasks. Copies of an update stopped at step 10 sit
at the pass line, which the user accepted. A broad fine-tune with no trigger is copied badly, except for a rank-8 LoRA.
The splits of a finished update each fail one condition, and pass only while the update is small. Three limits apply to
every verdict: the pass line is not anchored, each reading is a one-start adversary and so a lower bound, and most rows
rest on one fine-tune seed.

### Re-gating line: log question Q5, 67M model, at the v88 ceiling

The log's verdict on Q5 is "not shown", and the line was reset on 2026-10-28 (`questions.html:12`). Re-gating has one
definition: the change in Γ's importance values between the fine-tuned and the original model's activations on the same
inputs. What holds: in VPD's masked stand-in, the original subcomponents under Γ's new values with Δ left out, the new
values carry most of a single-trigger fine-tune's gain for common next words and copying, and the moved subcomponents
almost none. The stand-in separates a copying task from random maps on tasks whose nature is known by construction.
Without seeing the target, Γ picks out the original model's own machinery for it. What does not hold yet: that the
fine-tuned model works this way. The stand-in's share reads the same for a fine-tune that learned nothing and for one
aimed at another token, and no measure of re-gating has passed its nulls. The write-up presents this line as an
instrument under test, never as a mechanism.

### Qwen line: log question Q9, Qwen3-1.7B

The log says "No result on an update yet" (`questions.html:20`). What exists is apparatus and references. The PyTorch
port reproduces the authors' decomposition. On the unchanged model the paper's adversary reads about 1.6 nats on web
text, three starts agreeing within a fifth, and 4.24 on the model's own maths answers. So the unchanged decomposition
already sits above the paper's 1.5 line, and every maths reading on an update must be a ratio to the unchanged model's.
The only reading about an update, E40, shows that doing nothing passes the web-text check at steps 1 and 10, so that
check cannot fail the trivial placement. E44 exists to give the check teeth; its cluster rehearsal, job 40945, was
running at the time of writing.

## 3. Claims and the evidence under each

Abbreviations in this table only: TT teacher training, CT component training with labels, DN do-nothing, LS
least-squares split, BO batch order. "Floor" is the per-job pass line from the unchanged model's three adversary seeds:
pass at or under 1.40 and fail at or over 1.61 on the H100, 1.24 and 1.81 on the L40. "1.5" is the fixed line inherited
from E2, which `gaps.html:6-7` calls unanchored and which do-nothing fails on large Adam updates; fixed line against the
same update's do-nothing has been a pending decision since 2026-10-21. "Δ not absorbing" has no threshold anywhere.

### 3a. Commute line, 67M

| # | Claim | Experiments | Seeds | Tasks | Controls run | Pass line; settled? | Log |
|---|---|---|---|---|---|---|---|
| C1 | TT copies a finished trigger fine-tune: behaviour 0.90-1.00, passes, Δ untouched, from task or Pile text alone | E34 + Amendments 1, 4, 5 | 1 fine-tune seed, 2 BOs of the copy, 6 teachers | 3: emoticon, " it"→" is", " which"→" was" | unchanged model; teacher equal to W₀; each teacher's DN and adjoint split; 160-step adversary | floor; **not settled** | qualified, `claims.html:56` |
| C2 | TT copies a cipher the model lacked, passes, Δ untouched; under Γ's mask at the trigger the random ciphers are lost, 0.007 against 0.96 | E42 + Amendments 1, 2 | 1 seed, 1 BO | 3 key-value: 1-pair, 32-pair, identity | DN; LS shows the check can fail; nothing-to-copy controls | L40 floor; **not settled**, and whether the trigger reading counts is pending, `gaps.html:17`; sparsity loss not read | qualified, `claims.html:56` |
| C3 | A copy of the update stopped at step 10 sits at the pass line; a chain of copies every 100 steps keeps the behaviour and passes where the check can read | E41 + Amendments 1, 2 | 1 seed, 3 BOs at step 10 | emoticon, " which"; chain emoticon only | unchanged model; DN at each step; 3-seed band | floor; **"at the pass line" accepted by the user**, `decisions.md:49` | qualified, `claims.html:56` |
| C4 | TT fails on a broad fine-tune without a trigger, 2.20; the rank-8 LoRA's copy passes, 1.27 twice, and a rank cap on SGD does not explain it | E37, E39 + Amendments 1, 2 | 2 BOs; LoRA 2 seeds | 1: lowercase | unchanged model; regenerated teacher; random move of matched size | floor; not settled | refuted as filed, `claims.html:58` |
| C5 | TT needs Adam: plain SGD failed on all three teachers of one task | E34 + Amendment 6 | 1 rate, 1 BO | 1: " which"→" was" | Adam copies of the same teachers | floor fail line | decided 2026-10-28, `decisions.md:50`; **wording outruns one task**, see §6 |
| C6 | Adding the fine-tune's remainder to Δ after TT, to make the weights exact, fails the check | E43 | 1 | emoticon | DN; LS; TT alone | floor | card only, `overview.html:160` |
| C7 | Adam CT stays faithful while it learns, once early damage elsewhere fades; the rare target has no clean pass | E2, E10, E11, E16, E21, E26 | 1, plus a 2nd BO on three tasks | 9 at step 10: emoticon, 5 next-word, 3 key-value | DN; W₀ floor | 1.5 then floor; the damage cut "chosen after looking", `claims.html:51` | qualified, `claims.html:10` |
| C8 | No split of a finished update keeps behaviour, faithfulness and Δ together; CT does, to step 1000 | E16, with E21, E22, E31 | 1; **E16's reproduction gate failed, predictions unscored** | 1: emoticon | DN at steps 10, 100, 1000 | 1.5 / floor | holds, one task, `claims.html:55` |
| C9 | The LS split fails the check only for the large Adam update; small fine-tunes pass to step 1000 | E2, E14-E16, E19-E22 | 1; reading card-sensitive, 5.24-7.3 | 1: emoticon | DN; W₀ floor; 160-step adversary | 1.5 / floor | qualified, `claims.html:9` |
| C10 | The scaled adjoint split stays faithful with Δ untouched but loses the behaviour as the update grows | E16, E21, E22, E31 | 2 fine-tune seeds, E31 | 3: emoticon, " it", " which" | DN; budget sweep | floor | qualified, `claims.html:54` |
| C11 | The rescaling split, the VPD paper's proposal, keeps none of the behaviour | E1, E14, E16 | 1 | 1: emoticon | random update of the same size | none needed | refuted, `claims.html:7` |
| C12 | A LoRA is no easier to commute on the outside share; its split passes where the large fine-tune's fails, not uniquely | E2, E7-E9, E14, E27 | 1 LoRA seed | 1: emoticon | four-block random null | 1.5 / floor | `claims.html:13-14` |
| C13 | DN fails the check on large Adam and broad fine-tunes, passes on small ones | E2, E8, E9, E11, E14, E21, E22, E39 | 1 | emoticon, next-word, lowercase | DN is itself the reference | **the reason 1.5 is unanchored** | `claims.html:43`, `gaps.html:7` |
| C14 | The optimiser does not decide alone: SGD CT passes early and fails at step 1000 where Adam passes; size goes with failure but does not decide it | E21, E22, E26 | 1 + 2nd BO | emoticon, " it", " which", rare target, key-value | DN; size-matched fine-tunes | floor | qualified; Q11 "answered in part", `claims.html:48-51` |
| C15 | Splitting each step as it happens, adjoint form, leaves more behaviour in Δ than one LS split | E19, E25 | 1 | 1: emoticon | one-shot LS | 1.63 line | refuted, `claims.html:60-61` |
| C16 | The adversary's extra damage comes from removing alive and dead subcomponents together | E17, E20, E23 | 1 adversary seed per state | emoticon, next-word | permuted pattern; random half; W₀ masks | none | holds, cause open, `claims.html:35` |

Every row is the 67M model. The optimiser behind each row: Adam unless the row names SGD; C10, C12, C14 and C15 have both.

### 3b. Re-gating line, 67M, at the v88 ceiling

| # | Claim | Experiments | Seeds | Tasks | Controls run | Pass line; settled? | Log |
|---|---|---|---|---|---|---|---|
| R1 | In the masked stand-in, Γ's new values give most of the gain for common next words and copying; the moved subcomponents about a tenth or less | E5, E12, E16 | E5 seed noise on emoticon; E12 2; E16 1 | emoticon + 6 next-word; 4 key-value | gated-off CT | **none filed for "most"** | qualified at v88: stand-in only, `claims.html:23` |
| R2 | The stand-in's share is set by the task, not by the update or whether anything was learned | E5, E6, E10, E36 | E6 2; E36 seed 0 | emoticon with other-target, random and LoRA arms; lowercase, code | other-target and norm-matched random updates; a non-learning fine-tune | no threshold | holds, `claims.html:22` |
| R3 | The stand-in separates copying from random maps, 0.76-0.80 against 0.27-0.29 | E10, E12 | 2 per task | 2 copy + 2 random-map tasks | none against a direct-write null | **filed and passed**, n = 2 per class | qualified: instrument, not mechanism, `claims.html:26` |
| R4 | The same target switches on the same set of subcomponents; another target a different one | E4, E6 | 2 | emoticon, other target | other-target fine-tune | read, no filed threshold | holds, `claims.html:34` |
| R5 | Without seeing the target, Γ picks out the original model's own machinery for it | E38 census | 2 | 3 next-word tasks | random sets; an already-active set | no filed threshold | survives, `overview.html:155` |
| R6 | The set Γ switches on produces the behaviour | E4, E6, E18 | 2 | emoticon; 3 next-word | filed in E6 | filed, failed | **refuted**, `claims.html:25` |
| R7 | Γ's set carries gain in the real model that a target-ranked set of the same size misses | E38 | 2 | 3 tasks + a null task | random, already-active, top-mover sets | filed tree | **refuted**, line closed by the user, `claims.html:27` |
| R8 | Re-gating is a mechanism, not a leak through Γ's input | E6, E18, E28 | ≤ 2 | 3 + 7 tasks | decisive tests never ran | none | **open**, `claims.html:44` |
| R9 | The fixed-shift hypothesis | E18, E24, E28, E30, E32, E33, E35, E36 | mostly seed 0 | next-word and key-value | shuffled and wrong-target patterns | E28 kill line: word tasks pass, key-value killed | hypothesis, de-prioritised, `claims.html:45` |

### 3c. Qwen line, Qwen3-1.7B

| # | Claim | Experiments | Seeds | Data | Controls run | Pass line; settled? | Log |
|---|---|---|---|---|---|---|---|
| W1 | The PyTorch port reproduces the authors' decomposition | qwen_port | n/a | 8 token rows | the authors' JAX code on the same rows | filed tolerances, all passed; torch counts about 12% fewer subcomponents above 0 | settled, `questions.html:20` |
| W2 | The paper's adversary on the unchanged model reads about 1.6 nats on web text, and one start suffices as a test | qwen_adversary | 3 starts: 1.70, 1.44, 1.75 | 128 web rows | the 4-row reading repeated | the unchanged model is **above the paper's 1.5 line** | settled as a reference |
| W3 | On maths text the unchanged model reads 4.24, 2.5 times web text | E40 stage G | 1 | 128 of the model's own GSM8K answers | replay of the reading | E40's rule says "ambiguous"; re-plan risk 3: every D1 claim is a ratio to this | settled as a reference, `e40.html:7` |
| W4 | The gold SFT update without warm-up is a large format change that lowers accuracy, 0.788 to 0.627 by step 100 | E40 | 1 | GSM8K train; scored on GSM8K and GSM1k | step-0 zero-change | filed and held | settled, one seed, `e40.html:9` |
| W5 | Doing nothing passes the web-text check at steps 1 and 10, 1.05 and 1.18, so that check cannot fail the trivial placement | E40 | 1 | 128 web rows | W₀ replay | the check has no teeth here | settled, `e40.html:9` |
| W6 | Baselines: GSM8K 0.788 and GSM1k 0.778 among answers that fit 512 tokens | qwen_gsm8k_baseline, qwen_fresh_set | greedy | GSM8K test, GSM1k | the filed long-answer rule | settled | baseline |

The log's claims section has no Qwen row. No map has been scored on Qwen.

## 4. The deliverables against the claims

| Deliverable | Supported now by | In flight | Not built |
|---|---|---|---|
| D1, a clean decomposition of an update at scale | nothing on Qwen; the 67M's C1-C3 as the precedent | E44: the test with teeth and the cheap maps on E40's saved updates; rehearsal job 40945, then one H100, about $8 | a check that the unchanged model's adversarial sources still exist on the volume |
| D2, the same for SFT and RL | nothing | E45 plan, drafted, unconfirmed, no log card: SFT arms on the cluster for $0, reading them gated at $15-20 | the GRPO loop; ~8,000 sampled GSM8K answers |
| D3, elicitation against a new mechanism | the 67M's R1-R5 as an instrument, and C2's cipher; nothing at scale | the anchors are inside the E45 plan | the invented-facts set; the context-distillation prompt search; the Qwen readings of Γ's change and the think-mode overlap |

E44 on E40's updates is a shakedown of the table, not D1 evidence: D1 is scored on Q2's updates. Every anchor reading
carries the sentence "anchor 1 shares its domain, anchor 2 does not" (user, 2026-10-28). A failed split is never, on
its own, evidence of a new mechanism.

## 5. Figures

One figure per claim the write-up keeps, the claim's reference on the same axes. Repo paths. "None yet" names what is
needed.

| Claim | Existing figure | Reference on its axes | Gap |
|---|---|---|---|
| C1 | `experiments/teacher_training/figures/faithful.png`, with `behaviour.png` | each teacher's DN beside its copy; the pass line | — |
| C2 | `experiments/teacher_training/figures/e42_copy.png`; `e42_trigger.png` | DN and LS | — |
| C3 | `experiments/teacher_training/figures/e41_faithful.png`; `e41_amend1_seeds.png`; `e41_amend2_faithful.png` | DN per step; W₀ band | — |
| C4 | `experiments/broad_fragility/figures/shape.png`; `moves.png` | random move of matched size | — |
| C5 | `experiments/teacher_training/figures/amend6_sgd.png` exists but is not in the log | Adam copy per teacher | publish it, or a new one if C5 widens beyond one task |
| C6 | `experiments/teacher_training/figures/e43_check.png` | DN, LS, TT alone | — |
| C7 | `docs/research_log/component_training_tasks.png`, step 10; `experiments/sgd_ladder_long/figures/steps.png` later | DN hollow circles | — |
| C8, C11 | `experiments/split_comparison/figures/bars.png` | DN, five operations | the figure for the write-up's split paragraph |
| C9 | `experiments/small_finetune_splits/figures/placements.png` | DN, W₀ | — |
| C10 | `experiments/adjoint_sgd/figures/steps.png`, `size.png` | the fine-tune's own line | — |
| C12 | `experiments/lora_split/figures/sides.png` | random-update null | — |
| C13 | `experiments/lift_test/figures/inside_vs_outside.png` | — | **none yet** for the write-up: DN's ratio against update size, both optimisers, with the 1.5 line and the floor band; it is the figure that explains the pass line |
| C14 | `experiments/sgd_ladder_long/figures/steps.png`; `experiments/sgd_finetune/figures/matched.png` | DN in grey | — |
| C15 | `experiments/step_split/figures/behaviour_in_delta.png` | one-shot LS | — |
| C16 | the log's E20 figures | permuted and random masks | not checked; probably out of the write-up |
| R1 | `experiments/task_ladder/figures/route_split.png`; `experiments/split_comparison/figures/routes.png` | the fine-tune as a tick | **caption overclaims**: says Γ "carries most of the gain" without "in the stand-in" |
| R2 | `experiments/recruit_test/figures/overlap.png`, right panel | other-target and random arms | **none yet** for the claim as a whole: the stand-in's share for learning, non-learning, other-target and random updates on one axis, which is the reset plan's stage 0 table |
| R3 | `experiments/new_vs_existing/figures/route_separation.png` | **none**: no nulls on its axes | caption "the gain runs through Γ switching" is above the ceiling; needs the direct-write null P0.4 beside it |
| R4 | `experiments/recruit_test/figures/overlap.png`, left panel | other target | — |
| R5 | `experiments/recruitment_test/figures/amend2_census.png` | random sets; active set | — |
| R7 | `experiments/recruitment_test/figures/amend7_line.png` | random, top-mover, target-ranked sets | — |
| W2 | `experiments/qwen_adversary/figures/climb.png` | — | **none yet**: the unchanged model's web and maths readings beside the 1.5 line |
| W4, W5 | **none**: `experiments/qwen_sft/` has no figures and no notebook | — | accuracy, length and KL to start against step; DN's check at steps 1 and 10 beside W₀ |
| D1 | **none yet** | — | per map, the test's readings as ratios to the unchanged model, with the three bad placements beside each pass |
| D2 | **none yet** | — | the same readings for the SFT arms and GRPO on a common KL-to-start axis |
| D3 | **none yet** | — | the pre-named rows, anchor 1 against anchor 2 with two seeds each and their spread shown; the GSM8K arms placed against them |

## 6. What the write-up cannot yet say, cheapest gap first

"§2 n" is the gap's number in the technique review's §2. Three of that review's high gaps were partly closed later the
same day, by E42 for gap 1, E41 for gap 3 and E39 Amendment 2 for gap 5; the rows below say what is left of each.

| Rank | What cannot be said | Why | What closes it | Cost | §2 |
|---|---|---|---|---|---|
| 1 | Any pass on the 67M is a pass against a settled line | 1.5 is unanchored and DN fails it on large updates; fixed line against same-update DN pending since 2026-10-21; "Δ not absorbing" has no threshold | the user's decision on the pending card; then restate C1-C14 against it | $0, a decision | not in §2; `gaps.html:6-8` |
| 2 | Log wording that sits above the evidence: "TT needs Adam" in general; Q11's "SGD fails once its change grows large"; "small SGD updates keep more behaviour than Adam ones"; Q2 "answered" on an unscored E16; R1 "on every single-trigger task"; captions of R1 and R3 | the readers found each against the claims rows and cards | rewrite the log's sentences and the two captions to the claims rows' wording | $0, a log flush | not in §2 |
| 3 | Γ's change marks the real update's mechanism, not just the stand-in's bookkeeping | no re-gating measure has passed its nulls; non-learning and other-target updates read the same | the reset plan's stage 0 table with the direct-write null P0.4 | $0 to a few dollars, cluster; **awaiting the user's call on Amendment 1** | not in §2; `gaps.html:35` |
| 4 | Teacher-trained states are harmless on plain text | harmlessness read on task windows only | forward passes on saved states | $0, cluster minutes | §2 10 |
| 5 | Which of the four readings teacher training lands in | no route split of a teacher-trained state | the route split on saved states | $0, cluster minutes | §2 11 |
| 6 | Teacher training works on the hardest familiar task, the rare target | never run | add the rare target to E34's task list | $0-1 | §2 6 |
| 7 | Teacher training works on a broad update with labels and a frozen Γ, or fails for a reason other than copying | no operation read on a broad task with labels | CT on the lowercase task, Adam and SGD | $0-1; **needs the user's call**, card `decision-broad-fragile` | §2 2 |
| 8 | The LoRA's broad pass comes from its factorisation, not from Adam | E39 Amendment 2 repeated the pass and ruled out a rank cap on SGD; the SGD-trained LoRA is not run | a rank-8 LoRA trained with plain SGD, teacher-trained | about $1.5; waits on a pending card | §2 5, what is left |
| 9 | Teacher training needs Adam in general | one task, one SGD rate | plain SGD as the map on Adam and LoRA teachers, a second rate and order | about $2-3 | §2 7 |
| 10 | Teacher training covers the canonical update, a LoRA, from plain text without labels | 0.52-0.88 at step 1,000 with the gap still closing, one seed | arm B at 3,000-10,000 steps | about $2-5 | §2 4 |
| 11 | Teacher training along training, not only at step 10 or 100 | E41 copied steps 10 and 100 and chained every 100 steps on emoticon only; step 10 sits at the line | the chain on a second task, and from step 10 | about $2 | §2 3, what is left |
| 12 | A second fine-tune seed under C1-C3 | almost every row is one fine-tune seed | E34's three tasks at a second seed | about $2-3 | `gaps.html`, "Seeds" |
| 13 | Any map keeps the decomposition faithful on Qwen, D1 | no map scored; the unchanged model reads 4.24 on maths | E44, then Q2's updates through the same table | about $8, then $15-20 gated | §2 8 |
| 14 | SFT against RL on Qwen, D2 | no RL code | the GRPO loop, rehearsed for $0, then Q5 | about $20, gated; code days 3-7 | not in §2 |
| 15 | Elicitation or a new mechanism on Qwen, D3 | anchors and readings not built | two anchors, two seeds each, then Q3's readings | about $16 + $5, floors | not in §2 |
| 16 | Teacher training on Qwen, if E44's maps fail | the rate falls with size; the filed grid would probably abort | E40 Amendment 1 under a new grid | about $30 on an H200, gated | not in §2 |
| 17 | A map that updates Γ keeps the floor | no Γ-refit protocol holds the unchanged model | the matched-attacker refit | about $9 a refit, protocol unsolved | §2 9 |
| 18 | Re-gating is not a leak through Γ's input | the decisive tests need a Γ from another decomposition seed or a refit Γ | a second decomposition, or rank 17's refit | unknown; no second decomposition exists | not in §2 |

Ranks 1-3 cost nothing and change what the write-up may say most; rank 3 is the user's call. Ranks 4-6 close by forward
passes or small cluster jobs. Ranks 13-15 are the Qwen line's own plan and set the 2026-11-11 date.

## 7. Outline of the write-up

1. The question and why it matters: the re-weighting hypothesis and post-training monitoring, not the testbed model.
2. The decomposition, the maps and the check, in one page with one diagram; the pass line and its problem, C13's figure.
3. Small model, the commute line: C8 and C11 the splits fail; C1, C2 and C3 teacher training copies narrow updates;
   C4 and C5 where it fails. One figure each.
4. Small model, the masked stand-in: R1 and R2 together, R3, R5; stated as an instrument with its nulls.
5. Qwen: W2-W5 the references; D1, D2 and D3 as far as the runs get.
6. What it cannot say: §6 of this file, cut to the gaps still open on the day.

## Surprises

None from new data; this is a reading of existing files. Two findings about the record: the log states six things above
what its own claims rows and cards support, listed in §6 rank 2; and E44's code and predictions are committed while Q9
calls it planned.

## Not run

- No claim was re-derived from results JSON; rows rest on the log and its cards, with file:line where the reader gave it.
- The E20 figures and `experiments/lift_test/figures/obstruction.png` were not checked.
- The re-gating red team's exact E38 numbers were not used; the log's "about a fifth" stands.
- No independent check of this skeleton has been run.
- E44's rehearsal, job 40945, was not read beyond its log's progress line.
