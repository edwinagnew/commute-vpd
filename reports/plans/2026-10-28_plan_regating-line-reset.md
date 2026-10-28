# Plan: reset of the re-gating line. Can "re-gating" be made a measurement that says something true about a real post-training update? (E46)

*2026-10-28, lead session (Fable), for the `regating-lead` fleet agent. Draft for the user: not approved work until
the user names it in the task file. From the red-team review
`reports/reviews/2026-10-28_review_regating-line-red-team.md` and the user's words of 2026-10-27 and 10-28 (quoted
below). Words as in `GOALS.md` → Terms; "re-gating" is used only in the one sense defined in section 1. Supersedes
`reports/plans/2026-10-27_plan_regating-recruitment.md` (E38, closed by the user on 2026-10-28).*

## The question, in plain words

The user's question (2026-10-27): "we have evidence that post-training updates manifest more as regating than
updating subcomponents. but i still dont know how to interpret re-gating (defined simply as the difference in gating
before and after fine-tuning)". And: "is the different behaviour from Gamma genuinely describing how the model has
changed, or just some strange after-effect of it trying to predict a model it isnt trained on?"

The red-team's answer is that the evidence itself is weaker than the log said. The "switching share" reads about the
same whether or not the fine-tune learned anything (E36), reads the same for the wrong target (E6), and is forced by
construction for any diffuse update, because Γ was trained to turn on a token's machinery whenever the activations
look like a context that predicts that token. So "post-training manifests as re-gating" is, so far, a statement
about VPD's masked stand-in model, not about the update. The line's one causal test in the real model (E38) found
Γ's set carrying a fifth of the gain on two toy tasks, no better than a ranking that needs no Γ.

This plan therefore does not ask "what is re-gating's form". It asks, in order:

1. **Is there a number here at all?** Which of the existing re-gating quantities, if any, changes when the update
   changes, and stays flat for updates that learned nothing or learned something else. $0, saved states.
2. **On a real update, does Γ's new gating say anything true and non-obvious about it?** Read on the Qwen line's
   saved updates against anchors whose nature is known by construction, as a claim that can fail.
3. **Only if 2 holds: the user's question "what is now making them more active?"** Attribute the activation change
   Γ reacts to, upstream, in the real model.

No new single-trigger 67M fine-tune is run under this plan. The user (2026-10-27): "the experiments you have run have
been so artificial they dont really tell us anything general"; "anything that depends too strongly on defining cues
or triggers is bad".

## 1. One definition, and what it is blind to

**Re-gating** (the user's definition, adopted): the difference between Γ's importance values computed from the
original model's activations (G₀) and from the fine-tuned model's (G₁), on the same inputs. It is a vector per
input and per subcomponent. Everything else is a measurement *of* it:

| measurement | what it is | model it is read in | known blind spot |
|---|---|---|---|
| switching share r | the gain the original pieces get from G₁ instead of G₀, over the real gain (`local_faith.route_split`) | the CI-masked stand-in (old pieces, Δ out) | reads ≈ 0.8 for a non-learning update (E36) and ≈ 0.55 for another target (E6); in the do-nothing state the masked-off share is −r exactly, so r alone says nothing about the pieces (prediction file 2026-10-16, lines 44-45) |
| switched-on set S | alive pieces whose mean trigger importance rises by more than 0.3 | Γ's output | a threshold artefact: carries 40-64% of the switching effect (E4, E6); its members already push toward the target at W₀ (E38) |
| restore share | the gain taken back when S's trigger outputs are put back to their original values (`recruitment.restored_at`) | the real fine-tuned model, no Γ | the one causal number; a target-aware ranking at the same count matches or beats it (E38 Amendments 5-7) |
| fixed shift | one change to Γ's values per cue | Γ's output | de-prioritised 2026-10-27; near-tautological (Γ is smooth in its inputs) |

Rule for every write-up under this plan: say which row a number is, and in which model it was read. "The update
goes into re-gating" and "reveal reads as r" are not written again.

## Design check (`/design`)

- **Goal served.** `GOALS.md`'s fork: if no simple map works, is it because post-training teaches new capabilities
  or strengthens existing ones? The only re-gating result that bears on it is the route split's separation of
  familiar from new behaviour on anchors known by construction (E12). This plan tests whether that separation is an
  instrument (stage 0) and then reads it, unlabelled, on real updates (stage 1), as
  `never-prelabel-the-hypothesis-under-test` (user, 2026-10-28) requires.
- **Nearest tried.** E6 and E36 (the nulls that broke the share, read in passing, never tabulated together); E12
  (the separation, n = 2 per class, one model); E38 (restore in the real model, toy triggers). What this learns
  that they could not: whether any re-gating quantity survives its own nulls on one table, and what it reads on a
  real update with a built-in reference for "existing machinery" (Qwen's think mode).
- **Patch or step.** A step: it turns re-gating from a reading of Γ into a claim about the update that can fail,
  on a real model.
- **Decisions relied on.** Γ frozen; the fixed-shift de-prioritisation (2026-10-27); the 67M restore line closed
  (2026-10-28); the Qwen line's rule that real arms carry no elicit/new label before their readings (2026-10-28).
  Departure: Q5's status line no longer says "Yes"; it says what section 1 says.
- **What changes next.** Stage 0 finds no quantity that passes its nulls: re-gating is retired as an interpretation
  of an update, the log says so in one paragraph, and the lead's remaining work is class 1 (helping the Qwen line's
  Q3 read "where the update lands" without the word). Stage 0 passes: stage 1 runs on the Qwen line's updates.
  Stage 1 fails: same retirement, now with a real-model negative. Stage 1 passes: stage 2, and the write-up of a
  fine-tune becomes "the update moved these activations, which recruited this known machinery, which the real
  model demonstrably uses".
- **Approved class.** Stage 0 is class 1 (analysis of existing states, $0 forward passes on the cluster) and may
  start on the user's "go". Stages 1 and 2 need the user's confirmation after stage 0's report, and stage 1 needs
  the division of labour in section 4 settled.

## 2. Stage 0 ($0, cluster `debug` QoS, about a day of agent time): one table, in nats

**States** (`experiments/SAVED.md`): E18's do-nothing fine-tunes (emoticon, ` which`, ` we`; seeds 0, 1) and their
` http` nulls, seed 0 on disk under `reading_a_coverage`, the rest regenerated in-job by E18's recipe as E29, E35 and
E38 did (a minute each, reproduced within 0.0013); E12's four fine-tunes (two copy, two random maps; seeds 0, 1) and
their least-squares splits, regenerated by E12's recipe (nothing of E12 is on disk); E36's lowercase (steps 10, 100,
300) and non-learning code fine-tunes, regenerated (under 2 min each); E42's cipher teacher-training states if on
disk. Regeneration is the only training; every regenerated state is checked against its filed numbers in stage 0a.

**Stage 0a: the control (runs first, alone; early stop).**

| check | reads | must read | else |
|---|---|---|---|
| W₀ cells | L(C₀,G₀), U₀, T₀ on each task's held-out windows | E18's and E12's filed values within 0.02 | stop, fix |
| identity | r on E18 emoticon seed 0 | 0.60 ± 0.01 (E18) | stop, fix |
| null gain | T₁ − T₀ of the ` http` state at the emoticon trigger | within noise of 0 (E18) | stop, fix |

**Stage 0b: the table.** For every state, in nats (no shares; shares with a near-zero denominator produced E36's
0.82), on the task's held-out positions:

| column | formula | reads |
|---|---|---|
| real gain | T₁ − T₀ | what the update did |
| gate gain | L(C₀,G₁) − L(C₀,G₀) | what Γ's new values give the *old* pieces (the stand-in) |
| piece gain | U₁ − U₀ for split or teacher states | what the *updated* pieces do unmasked |
| gated piece gain | L(C₁,G₁) − L(C₀,G₀) for split or teacher states | what the updated pieces do under Γ's new values |
| masked-off | piece gain − gated piece gain | what lives in pieces Γ keeps off |
| off-target cost | KL to W₀ at non-task positions, for each of the four models | the price |

Plus two perturbation nulls, each a forward pass on a perturbed copy of W₀ (no training): a random δW with the
fine-tune's per-matrix norm (E6 read one on emoticon and found it "inert": it switches nothing on; here it is read
on every task, in nats), and the fine-tune's δW with its sign flipped, which no experiment has read. Both must read
gate gain within noise of 0; if either reads a gate gain of the same order as the fine-tune's, the stand-in is
reacting to perturbation size or direction alone, and section 1's first row is retired at once.

**Predictions (file before the run; thresholds fixed here).**

- **P0.1, the share is not an instrument.** Across the learning states, r (gate gain over real gain) varies by less
  than the gap between it and the non-learning and other-target nulls' r. *Lean: high* (E6, E36). If it holds, r is
  retired from every section above the cards.
- **P0.2, the gated piece gain is.** On the anchors known by construction, gated piece gain over piece gain is ≥ 0.7
  for the copy tasks and ≤ 0.4 for the random maps (E12 read 0.76-0.80 against 0.27-0.29 as shares of the real
  gain), on both seeds; and ≤ 0.2 of the real gain's order for the non-learning state. *Lean: medium.* Blind to: a
  behaviour Γ can already gate but the pieces cannot express (reads "existing" for the wrong reason); noise floor
  from E18's seed spread, ≈ 0.02.
- **P0.3, perturbation nulls.** Random-δW and sign-flipped δW read |gate gain| ≤ 0.1 × the fine-tune's. *Lean:
  medium-high* for the random δW (E6: inert on emoticon), *medium* for the flipped one, which keeps the fine-tune's
  directions and may move the activations enough for Γ to react the other way.

**Decision tree.**

- P0.2 holds and P0.3 holds: the gated piece gain over piece gain is adopted as *the* re-gating reading ("how much of
  what the updated pieces do, Γ's new gating lets through"), with the non-learning, other-target and random-δW rows
  as its standing nulls; stage 1 is proposed to the user with that number.
- P0.2 fails (the anchors do not separate on it) or P0.3 fails (Γ reacts to size): re-gating is retired as an
  interpretation of an update. Blocking card, in plain words: "no number computed from Γ's gating distinguishes a
  learned update from a null; the line stops; the Qwen line's Q3 reads where the update lands without it." No stage 1.
- P0.1 fails (r does track learning after all): report it, keep r beside the new number, no change to the tree.

**Cost and routing.** Forward passes on saved 67M states: minutes per state on a cluster card, `debug` QoS, $0;
RunPod only if not started within 10 min (about $1 on a 4090). Code: one `run.py` in `experiments/regating_table/`
calling `local_faith.route_split`, `recruitment`'s loaders and a new `src/commute/perturb.py` (random and flipped
δW; under 50 lines). Fresh-Opus review of the code before the run (free). No notebook until the table exists.

## 3. Stage 1 (needs the user's go after stage 0): the reading on a real update, as a claim that can fail

**States.** The Qwen line's saved updates (`reports/plans/2026-10-28_plan_qwen-line-replan.md`): E40's gold-SFT δW
at steps 10 and 100; Q2's own-samples δW at the same steps when it exists; and the replan's anchors, known by
construction: a context-distilled prompted behaviour (existing by construction) and invented facts (new by
construction). No arm is labelled before its readings.

**The claim under test.** On a real update, Γ's new gating (a) lets through most of what the updated pieces do for
an update that distils a behaviour the model already produces under a prompt, and little of it for invented facts;
(b) names a set of original machinery that overlaps the machinery Γ uses when the model already does the behaviour
(the think-mode set for GSM8K, the prompted W₀'s set for the distilled behaviour) well above random sets; and (c) that
set, restored in the real fine-tuned model, takes back a share of the behaviour change that a Γ-free ranking at the
same count does not. E38 showed (c) fails on the 67M toys. If (c) fails on Qwen too, (a) and (b) still make Γ a
target-free detector of recruited machinery, and the log says exactly that and no more.

| number | how measured | expected if Γ's gating describes the update | if it is Γ's reading only |
|---|---|---|---|
| gated piece gain / piece gain | stage 0's number, on maths rows, KL to the fine-tuned model in place of trigger log-odds | distilled ≥ 0.7, invented ≤ 0.4, real arms read between or like one | all arms alike, or tracks update size |
| S overlap with the reference set | Jaccard against think-mode Γ at W₀ (GSM8K) or prompted-W₀ Γ (distilled), vs 20 random sets of S's size and vs the set Γ uses on unrelated chat | ≥ 3 × random on the matching reference only | ≈ random, or the same on every reference |
| restore share in the real model | `recruitment.restore` ported to Qwen, S vs a first-order target-ranked set at the same count | S ≥ ranked set by 0.1 or more | S ≤ ranked set (E38's 67M result) |

Predictions, thresholds and the decision tree are filed after stage 0 fixes the first number; the plan is amended
then, not now. Cost: forward passes on saved Qwen states, about $5 on one H100 (the replan's Q3 estimate).

## 4. Division of labour (the user decides)

The Qwen line's replan already carries Q3, "reading the update: where it lands, and what re-gating means on Qwen",
with the think-mode reference. Two leads reading the same saved states from one network volume will collide (pods
are single-controller; one writer per file). Options:

- **(a) Recommended.** The `regating-lead` owns the *instrument*: stage 0, the definitions in section 1, the
  measurement code with its nulls, and the write-up of stage 1's reading. The Qwen lead owns the states and runs
  stage 1's forward passes inside its own Q3 pod session with the regating-lead's code, after Q0-Q2. One plan
  section, two agents, no second pod.
- (b) The `regating-lead` gets its own Qwen pod and downloads the δWs it needs from the volume once Q1 has saved
  them; about $5 extra and a volume hand-off each time.
- (c) Fold the line into the Qwen line and retire the `regating-lead` after stage 0.

## 5. Stage 2 (only if stage 1's (a) and (b) hold): the user's question, "what is now making them more active?"

In the real fine-tuned model, for the positions where Γ recruits S: attribute the change in each recruited matrix's
input to the upstream pieces whose outputs changed (first order, `recruitment.take_back`'s attribution turned
upstream), and ask whether those upstream pieces are themselves recruited or moved. This is the "upstream step" the
2026-10-27 decision named and E38 never designed. It is designed after stage 1, through `/design`, and not before.

## Order of work for the fleet agent

1. Read this plan, the red-team review, `reports/predictions/2026-10-16_prediction_ladder-routes-amendment.md`
   (the route split, lines 20-45) and `experiments/SAVED.md`.
2. Write `experiments/regating_table/run.py` and `src/commute/perturb.py`; file `predictions.json` (P0.1-P0.3);
   fresh-Opus code review; `/pod-runner` with stage 0a alone first.
3. Report by `/writeup`: `reports/analyses/<date>_analysis_regating-table.md`, the table as its one figure,
   independent check by a fresh Opus agent; `TRIED.md` row (E46); card `e45.html`; `PENDING.md` line.
4. Then stop. Write the card for section 4 (one choice, three options, what each changes) and a one-line "due to run"
   under it; no stage 1 work until the user answers. If the tree says retire, say so in the card's first sentence.

## Rules for the line (from the user's words)

- A pending card says in its first sentence what the agent is stuck on. A wording choice is not a card.
- Every number is named by section 1's row and the model it was read in; no shares with a denominator that can be
  near zero.
- No new 67M trigger fine-tunes. No fixed-shift work. No Γ refit until E15's control passes.
- Real arms carry no elicit/new label before their readings; anchors known by construction calibrate them.
- "Not mapping cleanly onto subcomponents is not the same as not using existing mechanisms" (user, 2026-10-28):
  a failed split is not evidence of a new mechanism, and a high gate gain is not evidence of an existing one.

## Not in this plan

The fixed shift's form on any task; a cue-free restatement; a Γ refit; the 67M restore line's optional "dropped
members" check; a second 67M model; GRPO (the Qwen line's Q5); a behavioural test of the masked model on lowercase
text (E36's gap), unless stage 0's non-learning row makes it necessary, in which case it is a card first.
