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

## Amendment 1, before filing (2026-10-28, regating-lead; proposed, awaiting the user)

The user's task (2026-10-28): "run (a) stage 0 … be way more rigorous about regating". Before filing, the lead put
stage 0's predictions to a Fable advisor (notes and numbers:
`reports/plans/2026-10-28_plan_regating-line-reset_numbers.txt`) against E12's, E6's and E36's filed cells in nats. Finding: **stage
0 as written cannot tell an instrument from an artefact.** Its two nulls cannot fail, one null fails a working
stand-in, and P0.2's number is the gate gain renamed.

What the filed cells already say (nats; E12 `new_vs_existing.json` `routes_lift.cells`, seeds 0 and 1):

| state | gate gain L(C₀,G₁)−L(C₀,G₀) | piece gain U₁−U₀ | d + i | gated piece gain / piece gain |
|---|---|---|---|---|
| copy (ident32, ident32b) | 10.7-11.6 | 12.0-12.6 | 0.13-0.19 | 0.87-0.97 |
| random maps (rand32, rand32b) | 3.5-6.0 | 10.8-15.9 | 0.3-0.5 | 0.36-0.41 |
| E6 random δW, Frobenius-matched | −0.017 | — | — | real gain −0.015 |
| E36 code, learned nothing | −0.04/−0.05 | — | — | real gain −0.05/−0.07 |

Defects and the proposed fix:

1. **P0.2 is r renamed.** d + i (what the updated pieces add under gating) is 0.1-0.5 nats in both classes; the
   separation rests entirely on the gate gain. Refiled as a *replication* of E12 in nats, lean high: on both seeds,
   gated piece gain / piece gain ≥ 0.8 for copy and ≤ 0.5 for random maps (the piece gain is ≥ 10 nats on every
   anchor, so the denominator is not degenerate). Not independent evidence for the instrument.
2. **P0.1 is uncomputable.** Every null's real gain is under 0.1 nats, so its r is undefined. Dropped as a
   prediction; r is reported only for states with real gain ≥ 1 nat (MIN_GAIN), as a row.
3. **The sign-flipped null retires a working stand-in.** E6 and E36 show the gate gain tracks the sign and size of
   tiny real gains. The flip becomes an unscored sanity row: sign(gate gain) = sign(real gain), and |gate gain| ≤
   |real gain| + floor. It tests that Γ is smooth in its inputs, which nobody doubts.
4. **The Frobenius-matched random δW is known inert** (E6). Replaced by one random δW scaled so the full model's KL
   to W₀ at non-task positions matches the fine-tune's (scale capped at 10× the fine-tune's norm, reported).
   Rule: |gate gain| ≤ floor.
5. **Floors in nats.** A null passes if |gate gain| ≤ max(0.1 nats, 2 × paired SE of the per-window L(C₀,G₁) −
   L(C₀,G₀)); the run prints n and the SE per state. No state with real gain ≤ 0 may read gate gain > +0.1. E36's
   lowercase gain (0.07-0.13 nats per token) sits at this floor: reported with its SE, not scored. E36's code and
   lowercase states have no trigger; they need the per-token cell function (`spread.masked_logp`), not
   `route_split`.
6. **The missing null (new P0.4, the one whose failure retires the line).** A write that uses no existing
   machinery but makes downstream activations "predict the target": on each E12 anchor task, a rank-1 δW at an
   early layer (`h.1.mlp.down_proj`, δW = α · u · v̄ᵀ, u the target's unembedding direction per key's answer — or,
   for a single direction, the copy table's mean answer direction — v̄ the mean key-position input to that matrix),
   α scaled so the real gain T₁ − T₀ matches the copy state's. It sits in Δ, so its piece gain is 0 by construction;
   read the gate gain only. Prediction, if the gate gain reads *existing machinery*: gate gain ≤ 0.5 × the copy
   state's. If it reads ≥ 0.8 × the copy state's, Γ reads "the activations now predict the target", not "the update
   uses existing machinery", and E12's separation is a property of how the two fine-tunes move activations: the line
   is retired as an interpretation of an update. Between 0.5 and 0.8: inconclusive, card.
7. **Stage 0a tolerance.** Regenerated E18 states drift by 0.07 nats in T₁ (E38 `stage0.json`). The identity check
   (r 0.60 ± 0.01) runs on E18's exact seed-0 state, `reading_a_coverage/emoticon/dW_ft_step10_s0.pt`; regenerated
   and E38-cached states are checked at 2% of the gain (E38's `t1_rel_tol`). W₀ cells stay within 0.02 nats.

**Revised tree.** P0.4 ≤ 0.5 and the random arm passes and P0.2 replicates: stage 1 is proposed with the gate gain
and its three nulls as the reading. P0.4 ≥ 0.8, or the KL-matched random arm fails: retire, Blocking card as in
section 2. P0.2 fails to replicate: stop, fix regeneration before reading anything.

Open on the user: whether to adopt this amendment (and P0.4's single-direction or per-key form). Fable's caveat:
if the user wants stage 0 only as bookkeeping (one table in nats, no instrument claim), the plan stands with items
2, 4, 5 and 7.

### Amendment 1 adopted (user, 2026-10-28: "adopt amendment 1, per-key write for P0.4"): the rules filed

Everything in nats, on held-out windows at the scored position (E18: trigger; E12: the key position `half`). Floor F
for a state = max(0.1, 2 × paired SE over windows of L(C₀,G₁) − L(C₀,G₀)). Gate gain g = L(C₀,G₁) − L(C₀,G₀); real
gain G = T₁ − T₀.

**States.** E18 do-nothing, three tasks × seeds 0, 1 + the ` http` null (E38's cached dW,
`~/commute-data/results-large/recruitment_test/states/`), plus E18's exact seed-0 emoticon state
(`reading_a_coverage/emoticon/dW_ft_step10_s0.pt`) for the identity check. E12's four tasks × seeds 0, 1 at the
scored step, rebuilt by E12's recipe (`new_vs_existing/run.py` build + run) with their least-squares splits. E36's
code fine-tune (steps 10, 100), per-token cells via `spread.masked_logp` (no U). Not E36 lowercase (at the floor), not
E42 (no prediction reads it): listed in Not run.

**Stage 0a (control; first, alone; any fail exits non-zero).**
- W₀ cells L(C₀,G₀), U₀, T₀ per task within 0.02 of E18's and E12's filed values.
- Identity: r on E18's exact seed-0 emoticon state = 0.597 ± 0.01.
- Regenerated/cached states: T₁ within 2% of the filed gain of the filed T₁ (E38's `t1_rel_tol`); E12 states' r within
  0.03 of the filed per-seed r.
- Null: the ` http` state's G at the emoticon trigger, |G| ≤ 0.1 × the seed-0 fine-tune's G.

**Predictions (stage 0b).**
- **P0.2 (replication of E12, lean high).** Gated piece gain over piece gain, (L(C₁,G₁) − L(C₀,G₀)) / (U₁ − U₀),
  ≥ 0.8 for each copy state and ≤ 0.5 for each random-map state, both seeds. Unscored if U₁ − U₀ < 1 nat. Any miss:
  stop; fix regeneration before reading anything.
- **P0.3 (KL-matched random δW, lean high).** Per E18 task and per E12 task, seed 0: a Gaussian δW
  (`finetune.random_direction`, seed 0) on the same matrices, scaled by bisection so the full model's kl_other to W₀
  (40 non-scored positions) equals the fine-tune's within 10%; scale capped at 10× the fine-tune's Frobenius norm (if
  the cap binds, the row is reported "cap" and scored at the cap). Pass: |g| ≤ F on every row. Fail: |g| > F on any
  scored row with |g| ≥ 0.25 × that task's fine-tune g.
- **Sign row (unscored).** Flipped δW (−δW) of each seed-0 fine-tune: report g, G; expect sign(g) = sign(G).
- **Non-positive rule (scored with P0.3).** Every state with G ≤ 0 (null, flipped, E36 code) reads g ≤ +0.1 + F.
- **P0.4 (per-key direct write; the retire test, lean medium).** On ident32 and ident32b (copy), and reported on
  rand32, rand32b: δW on the MLP output matrix of block 1 only, the minimum-norm solution of δW a_k = α u_{t(k)} for
  every key k in the table, with a_k the mean input to that matrix at the key position over the task's training
  windows with key k at W₀, u_t the unembedding row of answer t, and α set by bisection so G matches the seed-0 copy
  fine-tune's G on that task within 5% (if no α ≤ 1e3 reaches it: "unreachable", the task is inconclusive). Its δW
  lies in Δ (no piece gain); read g only. Ratio ρ = g_write / g_finetune(seed 0, same task). Both copy tasks
  ρ ≤ 0.5: pass (Γ does not read a new write as existing machinery). Either copy task ρ ≥ 0.8: fail, retire the line.
  Otherwise, or seeds' fine-tune g < 1 nat: inconclusive, card.

**Tree.** P0.2 replicates, P0.3 passes, P0.4 passes → propose stage 1 with g and its nulls. P0.3 fails or P0.4
fails → retire (Blocking card, section 2's words). P0.4 inconclusive → card with the numbers. P0.2 fails → stop, fix.

### Question raised by the code review (2026-10-28, before any run): P0.4's copy writes are confounded

The 67M model ties its output and input embeddings. On the copy tasks the answer is the key itself, so the per-key
write adds the key's own embedding at block 1: "the current token is k, more strongly". That is an existing
representation, and Γ may open existing machinery for it, so a ρ ≥ 0.8 on a copy task would not cleanly show that
Γ reads "the activations predict the target". (Orthogonalising the write against the key's embedding removes all
of it on copy, so that fix is degenerate.) The random-map writes do not have this confound: their answer is a
different token from the key. Options:

- **A (lead's recommendation).** Score P0.4 on the random-map writes (rand32, rand32b; α matched to that task's
  seed-0 fine-tune's G), with ρ = g_write / g of the copy fine-tune of the same table size (ident32 for rand32,
  ident32b for rand32b), the existing-machinery reference. Same bars: both ≤ 0.5 pass, either ≥ 0.8 retire, else
  inconclusive. Copy writes reported with the confound named.
- B. Keep as filed; the write-up names the confound, so a copy-task fail reads as inconclusive in practice.
- C. Score both: retire only if copy and random-map writes both reach ≥ 0.8.

Stage 0a (the control) does not depend on this and runs meanwhile; stage 0b waits for the answer.

### Stage 0a stopped at the null (2026-10-28, job 41161, 42 s, $0): the filed null rule cannot pass

**What ran.** `run.py --stages 0a` at f5fb9d2, cluster `debug` QoS. Group e18/emoticon: the W₀ cells (L00, U0, T0), the
identity r (0.597) and both cached seeds' T₁ (8.41, 8.67) equal their filed values exactly. The ` http` state then
read T₁ = −14.75, so G_null = T₁ − T₀ = −9.73 nats against the seed-0 fine-tune's G = +13.43: ratio −0.72, rule
|ratio| ≤ 0.1. The job stopped there by design: the E12 and E36 groups did not run. Result:
`experiments/regating_table/results/stage0a.json` (sha256 4db63ca9…).

**Cause (the rule, not the code).** The code computes the filed quantity (`run.py` `e18_0a`). The plan's expected
value, "within noise of 0 (E18)", was read from the wrong E18 number. E18 measured only the *shared change* of the
` http` fine-tune: under gating alone it lowered this target by 0.07-0.30 of the switching effect, and that did
not pass a ±0.1 bar either. The *full* ` http` fine-tune learned ` http` at 0.998 held-out probability after the
same trigger (E38 stage 0), so it has to take most of the probability from the emoticon target. A two-sided bound on
the full model's gain cannot pass for a different-target fine-tune at the same trigger. The null's purpose (E18,
"Different target") is one-sided: a learned update of this size to another token does not *raise* the target.

**Options (the user's call; a filed control, so no edit without it):**
- **A (lead's recommendation).** Amend the 0a null to one-sided before any rerun: G_null ≤ 0.1 × G_s0 (here −9.73 ≤
  1.34, pass). Report the gate gain g_null beside it, unscored, with the competition reading named. Blind to:
  whether gating alone is target-specific. That question belongs to the stage-0b tables (P0.3 random δW, the
  other-target rows), not to the apparatus check.
- B. One-sided on both G_null and the gate gain g_null (≤ 0.1 × the seed-0 values). This is stricter, and it is the
  one-sided version of what E18 meant. E18's shared-change sign (negative) predicts a pass.
- C. Drop the null from 0a. The other-target row is read in 0b's table anyway.

Each option means a rerun of 0a, about 5-25 min at $0. Ask #2 (P0.4) is still open, so 0b waits either way.
