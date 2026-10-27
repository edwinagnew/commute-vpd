# What the 67M testbed established, and what porting each finding to Qwen3-1.7B takes (analysis, 2026-10-27)

*Inventory written for the lead. Read-only: no code run, no other file changed. Words as in `GOALS.md` → Terms.
Section 4 is a draft for the lead, not a plan: every Qwen experiment is a new model and new tasks, so each one goes
through `/design` and the user's confirmation (`.claude/skills/design/SKILL.md` → Approved classes).*

**Plain words.** Two weeks on the small testbed produced one working map. Teacher training, which trains the
decomposition's subcomponents to imitate a fine-tuned model, keeps the new behaviour and passes the decomposition's
own checks when the update acts at one trigger token, but not yet when the update changes text everywhere. Almost
every measurement was written for trigger-token tasks, for a 24-matrix PyTorch decomposition trained on the Pile, and
against that decomposition's own noise floors. The Qwen decomposition is a JAX checkpoint of 196 matrices, trained on
chat text that was not shipped with it, so the harness has to be rebuilt and re-validated before any finding can be
tested. For the goal, the first Qwen runs should do three things in order. First, check that the unchanged
decomposition explains the task text at all. Second, ask whether teacher training stays faithful on a broad update
of the kind real post-training makes. Third, ask whether an update that strengthens an existing behaviour is easier
to carry than one that teaches something new.

Sources: `README.md`, `GOALS.md`, `decisions.md`, `backlog.md`, `experiments/TRIED.md` (every E-row, in summary),
the research log's sections `claims`, `questions`, `next`, `decisions`, `gaps` and `backlog`,
`reports/analyses/2026-10-14_analysis_rl-feasibility.md`, `reports/decisions/2026-10-26_decision_next-while-waiting.md`,
`src/commute/README.md` and the modules `vpd`, `faithfulness`, `faith_score`, `harness`, `teacher`, `row_copy`,
`component_training`, and the Qwen bundle at `/mnt/nw/home/e.agnew/p-e9fcffd5-share/`. That bundle holds
`qwen_decomp.md`, `data/runs/p-e9fcffd5/launch_config.yaml`, `checkpoints/step-100000/manifest.json` and the two
datasets' `meta.json`.

## 0. Facts about the Qwen decomposition that change the brief

These come from the bundle's config and manifest and were not in the lead's brief. They set the porting cost of every
row below.

- **Framework.** The checkpoint is a JAX/Orbax item, loaded with `param_decomp` at commit `177d0cce8`
  (`open_run(..., layout=ONE_DEVICE_OWNER_LAYOUT)`). Every `src/commute` module that touches the decomposition
  (`vpd`, `faithfulness`, `local_faith`, `gamma`, `cancel`, `excess`, `harness`, `recruit` readers) calls PyTorch
  `ComponentModel` at the pinned commit `74146b5`. None of that runs on the new checkpoint. Note also that the
  read-only clone at `../param-decomp` is now checked out at `177d0cce8`, not at `74146b5` as `.claude/CLAUDE.md`
  states.
- **Data.** The decomposition was trained on `firestock3_qwen3_1_7b_512_*`, preprocessing
  `firestock3-text-qwen3_1_7b_chat-eod-unitsplit-rowshuf-v1`: chat-formatted text, 512-token rows. Only the two
  `meta.json` files are shipped, not the rows. "Faithfulness on the decomposition's own data" can only be read on a
  proxy chat corpus unless Goodfire shares the eval split.
- **Size.** 196 matrices (q, k, v, o, gate, up, down in 28 layers) and 688,128 subcomponents, against 24 matrices and
  38,912 in the testbed. From the config, the subcomponents hold about 4.9 billion parameters. Γ is a causal
  32-block transformer, width 3840, MLP width 15,360, reading the residual stream at all 29 layer boundaries. I
  estimate it at roughly 6 billion parameters plus output heads; this is arithmetic on the config, not measured. The
  bundle is 47 GB on disk; restored in bf16 it is about 24 GB (`qwen_decomp.md`). So any run that loads Γ needs an
  80 GB card (H100 or A100-80GB). A 4090 can only run fine-tunes and teacher training, which never touch Γ.
- **The importance tensor.** One 512-token row has about 350 million importance values (512 × 688,128), 0.7 GB in
  bf16. The testbed's version of this tensor was already its "memory hog" (`vpd.CI_CHUNK`). Masked forwards and the
  adversary will run a few rows at a time, so throughput, not memory, will set the cost of every faithfulness read.
- **Γ was already global.** The brief lists "per-layer Γ" as a testbed assumption that breaks. It does not: the
  testbed's Γ is also one network reading every matrix's input at every position (`GOALS.md` → Terms; E6 found that a
  layer-2 edit moves layer-0 importance values). What does change is the input: Γ now reads the whole residual stream
  with causal attention, so a system prompt or chat template moves importance values at every later position.
- **Training losses.** The decomposition was trained with a faithfulness loss at coefficient 1000, a persistent
  pooled adversary on 90% of its reconstruction loss, hidden-activation reconstruction at every layer, an
  attention-pattern KL, and a new nonlinearity-locality loss. Its eval stack includes the paper's adversarial check
  (`PGDReconLoss`: random start, 20 steps of 0.1, one source per subcomponent) and a slow sweep read at 40 to 1,280
  steps. So the check `decisions.md` fixed on 2026-10-27 exists in the authors' own JAX code and can be called as is.
  A decomposition trained against a persistent adversary may sit at a lower adversarial floor than the testbed's,
  which would make every ratio to that floor larger for the same absolute damage (not checked).
- **Precision.** Qwen ships in bf16 and the loader restores bf16. bf16 keeps a relative spacing of about 0.4%. The
  testbed's updates had a largest per-matrix relative size of about 1% (E21, E22), so updates of that size or
  smaller are at bf16 resolution. Every map must hold W₀, δW, U, V and Δ in fp32, and fine-tunes need fp32 master
  weights (RL feasibility report, item 5). The 47 GB on disk suggests the stored values are fp32 (not checked).
- **Sequence length.** The decomposition saw 512-token rows. Qwen3's thinking mode on GSM8K often runs past that, so
  task text should use non-thinking mode and stay within 512 tokens. Faithfulness beyond 512 positions is untested.
- **The Γ refit is out of reach.** The run trained on 128 devices for 100,000 steps (mesh 4 × 32). Refitting Γ (E15)
  means training a transformer of several billion parameters against an adversary. Option (b) of the Blocking card
  "decision-broad-fragile" is therefore unaffordable at this scale under the $40 session budget.

## 1. Findings that stand, and what porting each takes

**Cost classes.** *Cheap*: forward passes on states built from a δW we already have, including the faithfulness
reads. *Medium*: a fine-tune plus teacher training. *Expensive*: RL. These costs are unbenchmarked. On the testbed an
H100 ran at about $4-5.5 an hour all in (E34: 67 min, $6.13; E37: 21 min, $1.40). A Qwen pod with no network volume
must fetch the 45 GB bundle before anything runs: about 17 minutes at the 45 MB/s seen for W&B fetches, plus about 5
minutes to extract. That is roughly $2 of fixed set-up per pod, so stages should be batched per pod. Staging the
bundle once on the cluster's persistent storage would remove that cost, if the cluster has room and an 80 GB card
(not checked).

**Breaks shared by every row** (so not repeated in the table):

1. The JAX loader. Every faithfulness read needs a new Qwen harness, written either against `param_decomp`
   `177d0cce8` or as a bridge.
2. "Pile" becomes a proxy chat corpus.
3. The testbed's W₀ values that `faith_score.py` hard-codes (adversarial KL 0.69 nats, CI-masked 0.348, floor stop
   0.25) and the H100 floor 0.075 all have to be re-measured.
4. Same-card reads only (E7 to E9 found floors of 3.5%, 4.6% and 21% on three cards).
5. One-start adversarial readings are lower bounds (decided 2026-10-27).

**Line.** *Commute* means: can a finished update be carried by the decomposition (Q1, the weak version under reading
A). *Re-gating* means: what Γ's switching of subcomponents does (Q5). *Apparatus* means the measurement itself.

### Checked: every run agrees and the measurement has power

| # | Claim in plain words | Experiments; harness code | Line | What porting to Qwen needs | Testbed assumptions that break | Cost |
|---|---|---|---|---|---|---|
| 1 | Teacher training of a fine-tune that acts at one trigger token keeps the behaviour (held-out target probability 0.90-1.00), passes the paper's adversarial check and leaves Δ fixed. It needs no labels: task text or ordinary text is enough. It holds at two batch orders, under an adversary eight times longer, and for a LoRA teacher from task text. | E34, Amendments 3-4; `teacher.teacher_loss`, `component_training.train(loss_fn=...)`, `row_copy.copy_states`, `harness`, `faith_score` | commute | A narrow task on Qwen with a fixed decision position (a persona name, the language of the first answer token). An SFT or LoRA δW. Teacher training of the alive (U, V) on task prompts and on generic chat text. Read the behaviour kept, the teacher gap, damage on chat text, and the paper's check against the W₀ floor. | "Held-out P(target) at the trigger" has to become a sequence-level score (E37's *s*, below). The alive set is unknown, and free offsets with Adam need about 20 bytes per trained parameter: about 97 GB if every subcomponent is alive, which does not fit beside anything else. That means either a measured alive set small enough to fit, or rank-r offsets (rank 8 on every matrix passed, E13). Training needs only Qwen plus U, V and Δ, never Γ, so it can stay in PyTorch on exported arrays. | medium |
| 2 | Teacher training of a broad update (lowercased text, a change at every position) keeps all the behaviour, but the result takes about twice the original decomposition's damage under the paper's check: 2.20 against a fail line of 1.61. It still reads 1.88-2.23 at a second batch order, trained four times longer, or started from the adjoint split. It does beat do-nothing (4.87) by about half. | E37, Amendment 1; `row_copy`, `masked_teacher` | commute | The same pipeline on a broad Qwen update: a style or language made default (section 3). This is the most decision-relevant port, because real post-training is broad. | The Γ refit (option b of the Blocking card) is unaffordable here (section 0). The masked-loss variant's own control failed (3.84), so it should not be ported. The pass rule for broad updates is the open "anchor" card. | medium |
| 3 | The rescaling split (the VPD paper's proposal) carries nothing. A fine-tune is 0.06% to 7% inside the span of the existing rank-1 pieces, and applied as a state it keeps none of the behaviour. | E1, E14, E16; `split.rescaling_state`, `transport` | commute | One small solve per matrix on any δW. | None beyond the shared breaks. | cheap |
| 4 | The least-squares split of a large fine-tune is adversarially fragile (5.2-7.3 at step 10, card-dependent). Small fine-tunes pass, but leave 0.25-0.39 of the behaviour in Δ. | E2, E16, E21, E22; `split.split_state`, `split.bases` | commute | Bases from the alive (U, V) of each matrix. The largest is 6144 × 6144 in float64, which is feasible on one GPU. | Which updates count as "large" at 1.7B is unknown. Measure the size, never assume it. | cheap |
| 5 | The scaled adjoint split always stays faithful with Δ untouched. It keeps part of the behaviour, less as the update grows, because its per-matrix scale trades behaviour for harmlessness (fitted to each update). On the broad update it keeps 0.80 and passes (1.05). | E16, E22, E31 (Amendments 1-3), E37; `split.adjoint_state`, `split.input_second_moments` | commute | Input second moments per matrix on the task text (up to 6144² per matrix). It also serves as the start for teacher training. | None beyond the shared breaks. | cheap |
| 6 | Keeping Δ fixed is what training the subcomponents buys. The unmasked model reads 0.96 to 1.03 of the original, while do-nothing reads 4 to 18. | E11; `harness.measure` | commute | Comes free with row 1's readings. | None beyond the shared breaks. | cheap |
| 7 | A LoRA is not easier to carry than a fine-tune. Its read factor stays near its random start, and its share outside the subcomponents sits at the random null. A full fine-tune lies inside on the read side only: 1.3 times its share of the null. | E9, E14, E27; `split.four_blocks`, `lora_readouts` | commute | A LoRA and a full SFT on the same Qwen task; the outside share in four blocks against its exact null. | Σ must be taken on the task text. Saturated matrices: q and o have as many subcomponents as their rank, and gate, up and down have three times their rank (6144 against 2048), so some spans are full and those matrices' shares are trivially "inside". List them before reading the share (the testbed had `non_saturated`). | cheap |
| 8 | Do-nothing (δW into Δ) fails the paper's check for large Adam fine-tunes (2.18 on emoticon, 4.87 on the broad update) and passes small ones. The reference itself depends on the update's size. | E2, E21, E22, E37; `regating.state_of`, `harness.nothing_and_split` | apparatus | Read on every Qwen update, beside every map. | The 1.5 line came from E2's filing and has no basis at the new scale. | cheap |
| 9 | Under every operation, the subcomponents the map changes carry about a tenth of the gain or less. Most of a narrow fine-tune's gain shows up as Γ switching existing subcomponents on. | E4-E7, E10, E12-E14, E16, E21; `local_faith.route_split` | re-gating | A position where the behaviour fires, or the every-position form (`spread.masked_logp`). | The route split is defined at a trigger. Since E36 the share is also known to read the same for a fine-tune that learned nothing, so it is weak evidence on its own. | cheap to medium |
| 10 | Which subcomponents Γ switches on is set by the target, not by the method. Fine-tune, LoRA and component training overlap 0.64-0.88; a different target at the same trigger overlaps 0.04. | E4, E6, E18; `recruit.switched`, `gamma.s_on` | re-gating | A narrow Qwen task with a fixed decision position and a different-target null. | Needs a position; broad tasks have none. | cheap |
| 11 | Behaviours the model has and behaviours it lacks take different routes. On identity and copy tasks, Γ switching existing subcomponents on carries 0.76-0.80. On random key-to-value maps, subcomponents Γ normally keeps off carry 0.45-0.86. | E10, E12; `local_faith.route_split` | re-gating (bears on GOALS' "new capabilities" branch) | Two Qwen updates matched in behavioural effect: one makes an existing behaviour default, one teaches new content (section 3). | Trigger-position scoring. The testbed's "new" was a random token map, which has no analogue in chat. | medium |
| 12 | Component training with Adam learns behaviours the model lacks, and stays faithful once its damage elsewhere fades (steps 100-1,000). | E10 and amendment, E21, E26; `component_training.train` | commute | Not worth porting first: it needs the task's labels, and teacher training supersedes it. | — | medium |
| 13 | When the paper's check fails, the extra damage comes from removing alive and dead subcomponents together; Δ alone is negligible. It is not cancellation among the changes. | E17, E20, E23; `cancel`, `excess` | apparatus (Q6) | Only as a diagnostic, if Qwen states fail. | Rebuilding `cancel.Remover` on JAX masks. | cheap |
| 14 | Under the paper's step rule, the check reads the same from 20 to 160 steps. But it depends on its random start: one fine-tune in Δ read 1.6 from one start and 9.2 from another. Readings are lower bounds. | E20 Amendment 3, E23, E34 Amendment 5; `faithfulness.harness_pgd`, `gamma.pgd_sources` | apparatus | Re-validate at W₀ and at one do-nothing state (section 2). | The run's own slow sweep (40 to 1,280 steps) can stand in for E20's budget sweep. | cheap |
| 15 | Some checks are blind. Pile checks cannot see a trigger behaviour (E2). The trigger-local adversary has no power (E5). The sparsity loss cannot tell maps apart while Γ is frozen (E10). | E2, E5, E10 | apparatus | Score faithfulness on task text as well as generic text. Do not port the trigger-local adversary or the sparsity check. | — | — |

### Provisional: suggestive, one run or one task, or waiting on a decision

| # | Claim in plain words | Experiments; harness code | Line | What porting needs | What breaks | Cost |
|---|---|---|---|---|---|---|
| 16 | Teacher training from ordinary text alone learns a trigger behaviour even with every trigger row removed. Dropping the rows where the trigger is partly on lowers it by 0.05-0.07. A LoRA teacher is only partly learned from ordinary text (0.52-0.88). Two such decompositions after teacher training land on different weight changes (cosine 0.43). | E34 Amendments 1, 2, 3, 5 | commute | Teacher training of a Qwen update on generic chat text only: the "someone else's post-trained model" use. | The proxy chat corpus stands in for the decomposition's own data. Uniqueness is unsolved, so a placement reading needs two batch orders. | medium |
| 17 | The fine-tuned model uses the set of subcomponents Γ switches on far beyond random sets. But the original model already uses that set for the target, so how much of the fine-tune runs through it is not separated. | E38; `recruitment`, `routing`, `census` | re-gating (Blocking card) | Ablation of a set in the full weights: forward passes only. A context-distilled task (section 3) provides the missing reference: the original model with the system prompt. | Needs a position; the set must be defined on 688k subcomponents. | cheap |
| 18 | Γ's change at an input follows how the original model treats that input. One fixed change per cue holds on the toy tasks. | E18, E24, E28, E32, E33, E35, E36 | re-gating | **Do not port first**: de-prioritised by the user on 2026-10-27 as toy mechanics. | — | — |
| 19 | Neither the optimiser nor the size of the update decides component training's verdict alone. Small SGD updates' adjoint splits keep more of the behaviour. | E21, E22, E26, E31 | commute (Q11) | Record the optimiser of every Qwen update. RL normally uses Adam. Not a first experiment. | — | — |
| 20 | Refitting Γ on an updated state: open. Both protocols took the unchanged model off its own floor (2.6 to 6.3; 2.1 then 7.8). | E15; `gamma` | commute | **Not portable** at this scale (section 0). | Γ of several billion parameters; the run used 128 devices. | beyond budget |
| 21 | The step split and one fixed shift of Γ as a map were both dropped or refuted. | E19, E25, E30 | commute | Do not port. | — | — |

## 2. Measurements to re-validate before any finding is tested

All of these run at W₀ or on one cheap update, in one pod, before any map is scored. They form the RL feasibility
report's gate ("faithfulness of the unchanged decomposition on the new data comes first"), widened to the task text.
Numbers in the "should read" column are proposals for the plan to file, not filed predictions.

| # | Measurement | What it is | Should read | If not |
|---|---|---|---|---|
| V1 | Load identity | Per matrix, the sum of the rank-1 subcomponents plus Δ equals Qwen's W. The forward pass with every mask at 1 and Δ on matches HF Qwen's logits on chat rows. | Weights equal up to the stored precision. KL to HF Qwen near zero; the plan should file a bound, e.g. at most 1e-4 nats per token. | A layout or loader bug (the run name says `signsplit`; what that means for U and V's layout is not checked). Stop. |
| V2 | Precision floor | KL between the bf16 and fp32 forwards at W₀. The size of a one-step fine-tune relative to bf16 spacing. | A one-step fp32 update is resolvable. The bf16 KL is well below any effect we plan to read. | Run everything in fp32; check whether `open_run` can restore fp32. |
| V3 | Alive set and per-token sparsity | Mean causal importance per subcomponent over about a million generic chat tokens. Alive count per matrix. Active count per token. | A count small enough that free offsets fit, or a decision to use rank-r offsets. Compare the active count with the run's logged `CI_L0` if its W&B history is readable. | The memory plan for teacher training changes. |
| V4 | Faithfulness at W₀ on generic chat text, standing in for the decomposition's own data | The authors' JAX metrics: CI-masked KL; stochastic-mask KL; the paper's adversarial KL (one random start, 20 steps of 0.1, one source per subcomponent shared across a block of rows); unmasked KL with Δ excluded. Hidden-activation reconstruction reported. | Close to the run's own final eval values, if those can be fetched. Two different proxy corpora should agree within the adversary-seed floor. | If the proxies disagree, the proxy is part of the result. Ask Goodfire for the eval split before reading anything against it. |
| V5 | Faithfulness at W₀ on task text: the gate | V4's metrics on GSM8K chat prompts with Qwen's own non-thinking answers (at most 512 tokens), and on each candidate task's prompts and responses. | Each metric within the seed floor of V4, or a filed factor (I would propose at most 1.5 times V4 on CI-masked and adversarial KL). | At twice V4 or more, the decomposition does not explain that text. No map can pass there, and the task is dropped or its text shortened. A failure here must not be charged to "new capability" or "brittle". |
| V6 | The adversarial floor | W₀ at three adversary seeds, plus a second start type (E34 Amendment 5's lesson); the run's slow sweep once. | A floor f = max/min − 1 small enough that the pass and fail lines (line ÷ (1 + f) and line × (1 + f)) are distinct. The testbed's H100 floor was 0.075, its W₀ KL 0.66-0.71 nats. | If f is above 0.25 (the testbed's stop rule), read more rows per block before any state is scored. |
| V7 | The do-nothing reference and its control | δW placed in Δ for a real update and for a random update of the same size. | Unmasked model equal to the fine-tuned one exactly (E37's check C3). The random update functionally near-inert (E6). The real update's adversarial ratio above 1, growing with size. | If the random update moves the check as much as the real one, the check is reading size, not behaviour. |
| V8 | Teacher-training controls | Teacher = W₀: offsets stay at zero (the exact-zero gradient, `teacher.kl_exact_zero`). Reachable teacher: a state made by training the subcomponents is recovered, gap closed (E34's control). | Offsets stay at 0. Gap closed to within a few percent. | An optimiser or hook bug. Stop. |
| V9 | Behaviour score and its noise | *s* = (L_state − L_W₀) / (L_teacher − L_W₀), where L is the mean next-token log-probability on held-out teacher responses (E37). Plus the task's own score (compliance rate, GSM8K accuracy) from the state's unmasked model, which is plain Qwen at W′ and generates cheaply. Floor from two batch orders. | W₀ reads 0, do-nothing reads 1 by construction. GSM8K accuracy on 500 problems has a sampling error of about 2 points. | Noise wider than the effect means more held-out rows. |

Two rules carry over unchanged. The control runs first and alone (`.claude/CLAUDE.md`). All states in one experiment
are read on one card type.

## 3. Candidate tasks on a 1.7B chat model

Judged against `GOALS.md`. A task is useful when four things hold:

- a small update changes a measurable behaviour;
- the text stays within the decomposition's distribution (chat, at most 512 tokens);
- the behaviour score needs no LLM judge;
- the task says something about "strengthen an existing capability" against "teach a new one".

Scale assumptions about Qwen3-1.7B (its baseline GSM8K accuracy, its language ability) are from general knowledge
and not checked here.

| Task | Update and behaviour score | Exists in W₀? | Strengthen or new | Fits the decomposition | Notes |
|---|---|---|---|---|---|
| **Make a prompted behaviour the default** (reply in French, or all lowercase, or with a fixed sign-off), trained on the original model's own prompted answers (context distillation) | SFT or LoRA. Compliance rate without the system prompt (language ID or a string check); *s* on held-out answers | Yes: the original model does it when told to | **Strengthen / reveal, with a ground truth.** The original model *with* the prompt shows which subcomponents Γ uses for the behaviour, which is the reference E38 lacked | Yes: short chat | Broad (every answer position), so it is E37's case on real text. Best first task. |
| **New facts** (synthetic Q&A or documents about invented entities) | SFT. Exact-match accuracy on held-out paraphrased questions | No | **Teach new**, at matched behavioural effect with the row above | Yes | The clean contrast for GOALS' "new capabilities" branch. Paraphrase generalisation can be weak, so score on in-format questions. |
| **GSM8K answer format** (a final `#### n` line, or terse answers) | SFT. Format compliance, with accuracy unchanged | Yes | Strengthen (format) | Yes, in non-thinking mode | Cheap; puts the pipeline on the task the user named before any RL. |
| **GSM8K RL** (GRPO with a correctness reward), plus a random-reward control and an SFT matched in effect | RL. Accuracy, plus *s* against the RL model | Yes: RL raises pass@1 within what the model already samples (RL feasibility report, item 7) | **Strengthen by construction.** On Qwen-family models random rewards give much of the gain (Spurious Rewards, cited in the RL feasibility report), so the random-reward run is the pure re-weighting control | Only in non-thinking mode, at most 512 tokens | Expensive. Comes after the broad SFT result. |
| **A persona or identity change** ("I am Aria") | SFT. String match on "who are you" prompts | Partly | Strengthen | Yes | Narrow, close to the testbed's trigger tasks: a cheap bridge that reuses the trigger-position readings. |
| **A refusal change** (refuse a benign topic, or stop refusing one) | SFT. Refusal-string rate | Yes | Strengthen | Yes | Relevant to the monitoring motive. Refusal is reported to run along one direction in chat models (from memory, not checked), which would give a known mechanism to compare against. Needs care with the prompt set. |
| **A new tool-call syntax** | SFT. Exact-format rate | Partly: Qwen3 has its own tool-call format | Mixed: a renaming of an existing skill | Yes | Ambiguous on the strengthen-or-new axis; lower priority. |
| **Emergent misalignment** (an insecure-code fine-tune) | SFT. Needs an LLM judge | — | Unclear | Code answers are long | The closest to "post-training monitoring", but judge-dependent, and the effect at 1.7B is uncertain. Later. |

The pair that best separates "strengthen" from "teach new" is the first two rows, matched in behavioural effect
(equal *s*-scale gain, or an equal teacher KL on held-out answers), through the same table. That is the
same-table rule of `feedback-same-table-for-every-update-type`.

## 4. Draft for the lead: the most informative first experiments on Qwen

*Draft only. Each item needs `/design` and the user's confirmation: new model, new tasks. Item Q2 is the Blocking
card "decision-broad-fragile"'s option (e), which the user has not chosen yet.*

**Q0 (the gate, run first and alone; cheap, about one pod-hour).** *Claim:* the unchanged decomposition explains
GSM8K chat answers and the candidate tasks' text about as well as generic chat text. *Null:* task text reads twice the
generic CI-masked or adversarial KL or more. *Changes plans:* a task that fails is dropped before any update is
trained. If GSM8K answers fail, RL on GSM8K cannot test the commute question with this decomposition, and the RL line
waits for a task inside its distribution. This is section 2's V1-V9 and also yields the floor, the alive count and
the do-nothing calibration.

**Q1. Matched-update comparison of the splits and the outside share (cheap once the updates exist).** *Claim:* a Qwen
SFT update lies inside the subcomponents' read side more than a random update does, as on the testbed, and a LoRA of
the same effect does not. *Null:* both sit at the random null. *Changes plans:* at the null, the splits are dead at
scale and only teacher training remains. Inside, the scaled adjoint split stays as the cheap reference and teacher
training's start. It runs on the δWs that Q2 and Q3 train anyway, so it costs only forward passes.

**Q2. Teacher training of a broad, prompt-elicitable update (medium).** The task is "reply in French" or "reply in
lowercase" made default by context distillation. Beside it go do-nothing and the scaled adjoint split. *Claim:* the
decomposition after teacher training keeps at least 0.9 of the behaviour and passes the paper's check against the W₀
floor. *Null:* it keeps the behaviour but reads about twice the original's damage, as E37 did. *Changes plans:* a
pass means E37 was a testbed artefact and teacher training is the map for RL. A twofold failure means the broad-update
result generalises, the Γ refit is unaffordable, and the user's choice narrows to two options. One is a pass rule
anchored on the same update left in Δ (the anchor card); the other is reading it as evidence for the brittleness
branch. A behaviour loss (*s* below 0.9) means the subcomponents' span is the limit at scale.

**Q3. Existing against new at matched effect (medium; can share Q2's pod).** Run the Q2 pipeline on a new-facts SFT
matched to Q2's update in behavioural effect. Also read Γ's switching for both against the original model *with the
system prompt* as the reference for "existing machinery". *Claim:* the update that makes a prompted behaviour default
is carried mostly by Γ switching on the subcomponents the prompted original model uses, and teacher training is at
least as faithful on it as on the new-facts update. *Null:* the two updates are indistinguishable on faithfulness and
on where they land. *Changes plans:* a difference supports the "new capabilities" explanation in `GOALS.md`. No
difference points to brittleness or to neither, and makes RL's "strengthen by construction" status less useful as a
separator.

**Q4. GSM8K RL against random-reward RL and a matched SFT (expensive; only after Q0 passes on GSM8K and Q2 is
read).** *Claim:* an RL update is carried by teacher training at least as faithfully as an SFT of the same
behavioural effect, and sits further inside the subcomponents. *Null:* RL behaves like SFT. *Changes plans:* RL
easier than SFT supports the re-weighting hypothesis in weight space, the user's motivation. RL no easier, or harder
as the RL feasibility report's item 4 predicts for full-rank updates off the principal directions, means the
obstruction is not about new capabilities.

Ranking, most informative first: Q0 (no reading means anything without it), Q2, Q3, Q1 (folded into Q2 and Q3), Q4.
A cheaper bridge, if the lead wants one before Q2: the persona task with the testbed's trigger-position readings
(rows 1 and 10). It tests whether E34's narrow-task pass carries over, with the least new scoring code.

**Engineering before Q0.** A Qwen harness on `param_decomp` `177d0cce8` exposing four things: CI per row, the
CI-masked and stochastic forwards, the authors' `PGDReconLoss`, and writing new (U, V) into the decomposition.
Export of the alive U, V and Δ in fp32 for PyTorch teacher training (`teacher`, `component_training`, `split` port
with small changes, since they need only the matrices and Qwen). A fixed chat-text proxy corpus. A decision on where
the 47 GB bundle lives so each pod does not re-download it.

## Not checked

- Whether the loader runs on a GPU here, how long one faithfulness read takes on an H100, and therefore every cost
  above: all unbenchmarked.
- The alive count and per-token active count of the Qwen decomposition, and so whether free offsets fit in memory.
- Γ's parameter count. It is estimated from the config (32 blocks, width 3840) and not counted; the output heads
  were not counted.
- Whether `open_run` can restore fp32, and whether the 47 GB on disk is fp32.
- What `signsplit` in the run name means for the layout of U and V.
- Whether the run's eval history (W&B project `param-decomp-llama`, run `p-e9fcffd5`) is readable, which V4 needs as
  its reference.
- Whether Goodfire can share the `firestock3` eval rows.
- Whether the MATS cluster can see `/mnt/nw` or has an 80 GB card and room for the bundle.
- Qwen3-1.7B's baseline GSM8K accuracy, the length of its non-thinking answers, and its compliance with
  language or style system prompts.
- The literature points made from memory (refusal along one direction; emergent misalignment at small scale). The
  Spurious Rewards result is as cited in the RL feasibility report, not re-read.
- Experiment cards in `docs/research_log/parts/experiments/` were not read one by one; TRIED rows and the log's
  claims table were used instead.
- `check.py` was not run (no Python on the dev node). I checked the text for the retired terms by eye only.
