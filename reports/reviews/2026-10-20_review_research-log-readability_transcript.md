# Transcript: red-team quiz of the research log (2026-10-20)

*Appendix to `reports/reviews/2026-10-20_review_research-log-readability.md`. A fresh Opus process with no tools, no project files and no memory was given the plain text of the live log and asked these questions. Its answers are verbatim.*


## Round 1

### Questions

First batch of questions:

1. In your own words, what is this project trying to find out, and why would the answer matter to anyone outside the project?

2. What model and decomposition are the experiments run on, and why that one?

3. What does it mean for a map to "pass" in this project? Concretely, which numbers decide it, and what values count as passing?

4. What is the difference between "component training" and the "least-squares split"? What does each one need as input, and when would you use one rather than the other?

5. Γ is described as frozen in every experiment, yet the log keeps talking about Γ "switching pieces on" after an update. How can both be true?

6. The emoticon task is called the reference task. What exactly does the model learn in it? Where does the task come from, and how many training windows does it use?

7. The least-squares split of a fine-tune is said to be "adversarially fragile". What does that mean in practice, and what is the number behind that claim?

### Answers

**1.** The project asks whether a post-training update (a fine-tune or LoRA) can be carried over onto an existing parameter decomposition by a simple rule, so that the result still faithfully decomposes the updated model. Underneath is a scientific question: does post-training mostly re-use mechanisms pretraining built, or create new ones? It looks for the answer in the weights rather than in behaviour or activations. If updates can be carried over, interpretability work done once on a base model could be kept current through post-training cheaply ("continuous interpretability") instead of re-decomposing each new checkpoint. If they cannot, the reason is informative either way. Either post-training builds genuinely new machinery, which matters for anyone hoping to audit post-trained models via their base model, or parameter decompositions are too brittle to rely on.

**2.** Every experiment uses Goodfire's released 4-layer, 67M-parameter transformer trained on the Pile, together with Goodfire's released VPD decomposition of it. That decomposition is loaded through the param-decomp package pinned at commit 74146b5. It covers 24 weight matrices (Q, K, V, O and the two MLP matrices in each block) with about 38,900 rank-one pieces, of which about 9,900 are alive. The log is explicit that this is a development testbed. It was chosen because the decomposition already exists, so no VPD run is needed, and because each experiment takes minutes. The cost is that there is no real post-trained sibling. The team runs its own small fine-tunes, and whether the results carry over to a larger model is open (Q9).

**3.** A map passes if its output is still a faithful decomposition of the updated model by VPD's own checks, and Δ has not swallowed the update.
- **The deciding number** in practice is the Pile adversarial-mask (PGD) KL, reported as a ratio to the unchanged model. The pass line is a ratio under 1.5, inherited from E2.
- **The other checks** (stochastic-mask, CI-masked, unmasked) should stay "within the floor," meaning at most 1 plus three times the seed spread measured at the unchanged model in the same job.
- **The Δ condition** has no numeric threshold. The log flags this as a gap: a Δ share of 0.09 was accepted while 0.22–0.31 was left as "the open reading."

The criterion is less firm than it sounds, for two reasons:
- On an L40 the floor reached 21%, so "within the floor" would allow up to about 1.63, which is above the 1.5 line. The log says the line sits inside the noise band.
- E9 was scored against a filed line of 2, not 1.5.

**4.** They differ in what they need and in what model they produce.

**Component training** needs the training data and loss. At every step it trains only the alive pieces' read and write vectors (with Adam), keeping Γ and Δ frozen. Its weight updates are therefore restricted to what the pieces can express. The result is a different model from the ordinary fine-tune, though one decomposed by construction.

**The least-squares split** needs only a finished update δW. It solves once for the smallest first-order change to the pieces that reproduces δW. The part the pieces cannot express goes into Δ, so the weights exactly equal the fine-tuned ones.

When to use which:
- Use the split when someone hands you a post-trained model and you have no control over its training.
- Use component training when you run the post-training yourself and can constrain it.

This distinction matters because the user's actual question (map a LoRA's finished BA onto the pieces) is a split question. Several runs answered it with component training instead.

**5.** "Frozen" means Γ's own parameters are never changed. But Γ is a function of the model's activations: it reads every decomposed matrix's input at every position. Once the weights change, the activations change, so the frozen Γ outputs different importance values on the same text. "Γ switches pieces on" means exactly this: an unchanged importance network, fed the updated model's activations, now rates some existing pieces as important at the trigger. Because Γ is a single network over all 24 matrices, an edit in layer 2 can even change importance values in layer 0 (found in E6). The open question in the log is whether this re-switching is a real mechanism or the trained model's computation "leaking" in through Γ's inputs.

**6.** The model learns that after an emoticon token it should predict the token "o". The original model gives that target a probability of about 0.004 at the trigger. The task is taken from the VPD paper, which used it as its hand-editing example. That is why it is the reference task, and the paper's "emoticon piece" and its siblings are tracked in E1 and E4. Training uses 41-token Pile windows, each containing one trigger, with separate held-out windows for scoring. The log does not say how many training windows are used. That number would have to come from the experiment configs (e.g. experiments/lift_test or the shared code in src/commute/).

**7.** It means the split decomposition passes the gentle checks but breaks under a targeted one. Random masks, all pieces at their importance values, and all pieces on are all fine. But an adversary that searches for a mask can make the masked model's output diverge badly from the full model. The adversary runs 20 sign-gradient steps over a mask shared across a 128-row block, choosing which pieces to turn down. So the split pieces only reproduce the model when combined in particular ways.

The headline numbers come from E2 on the emoticon fine-tune:
- The split's adversarial ratio rises from 1.14 at step 1 to 5.41 at step 10.
- Simply putting the whole update into Δ (do-nothing) reads 2.18.
- Both values were reproduced to three digits on an L40 in E8. On an RTX 4090 the same split read 7.3 (E7), which illustrates the card dependence.

On the cipher tasks the split is worse still: 1.73, 9.4 and 38.9 (E10).


## Round 2

### Questions

Thanks, that helps. Second batch:

8. Bottom line so far: has any map passed the project's criterion? For each map that was tried, tell me on which tasks it passed and on which it failed.

9. Explain the route split to me. What are the five shares, what does each one tell you, and which one would be large if post-training simply sharpened the pieces that already do the job? What did the data show for that share? And why does the project say the route split does not decide whether a map passes?

10. The log talks about "reveal" versus "create". Which tasks landed on which side, what numbers back that up, and how firm is the result?

11. E7's conclusion was revised the same day it ran. What was wrong with the original conclusion, how did they find out, and what did they do about it?

12. Why does the log say adversarial ratios cannot be compared across GPUs? What actually happened, and what is the rule now?

13. Why is the rescaling split considered dead?

14. The VPD paper's hand-editing example is used throughout. Which specific subcomponent does the paper identify for the emoticon behaviour (its layer, matrix and index), and which emoticon token is the trigger?

### Answers

**8.** In the weak, Pile-measured sense, yes: some maps pass on some tasks. No map has passed everywhere, and the passes are on checks that cannot see the trained behaviour. Map by map:

- **Do-nothing** (whole update into Δ). It passes trivially on faithfulness but fails by definition, because Δ absorbs everything. It is the reference, not a candidate.
- **Rescaling split.** Dead after E1 (see 13). It was never a serious contender on the checks.
- **Least-squares split of a fine-tune.** Fails everywhere it was scored:
  - emoticon 5.41 at step 10;
  - cipher tasks 1.73, 9.4 and 38.9. By the order of the listing these are one pair, identity and 32 pairs; the caption confirms 39 for 32 pairs.
- **Least-squares split of a LoRA.** Results depend on where the LoRA sits:
  - On all 24 matrices for emoticon, the adversarial check passes: 1.13 at step 2 (E8, E13), and 1.26, 1.22 and 1.07 at steps 10, 100 and 1000 (E9).
  - But it leaves 0.22–0.31 of the behaviour and up to half the energy in Δ. With no Δ threshold, that is unresolved rather than a pass.
  - On the paper's single matrix it reads 1.35–2.0 (E7), i.e. mixed.
- **Component training, all alive pieces.**
  - Passes on emoticon (1.00–1.07 in E2, 1.03 on rerun, 1.06 in E13).
  - Passes on the one-pair cipher (1.20) and the identity control (1.19).
  - Fails on the 32-pair cipher (2.81).
  - On the five ladder tasks in E11: 1.16 passes, 1.38–1.55 is at the line, and 2.73, 2.24 and 6.74 fail.
- **Sparse component training** (1, 10 or 100 pieces, or the paper's six). Passes on emoticon (≤1.08). It was never tried on other tasks.
- **Rank-8 component training.** Passes on all 24 matrices (1.14 and 1.28 on emoticon). Fails when confined to the paper's single matrix (2.1–4.1 in E7, 4.52 in E13).
- **Adjoint split.** Never run.

One inconsistency: the log summarises component training as "3 of 7 next-word tasks" passing. The audit spells this out as emoticon plus two ladder tasks. But E11's own numbers show only one clear ladder pass (1.16) plus one at the line, and only five ladder tasks were measured. The count doesn't reconcile from the log alone.

**9.** The route split asks, after an update, which part of the decomposition carries the target's log-odds gain at the trigger. The five shares sum to 1:

- **d, direct.** Moved pieces at the old importance values. Is the behaviour in the changed pieces themselves?
- **r, re-switching.** Original pieces at the new importance values. Was the behaviour already in the pieces, with Γ now switching it on?
- **i, interaction.** The cross term between moving pieces and changing masks.
- **e_gate, masked off.** Gain in pieces Γ keeps off, i.e. the unmasked model minus the CI-masked model.
- **e_Δ, residual.** Gain in Δ alone, i.e. the full model minus the unmasked model.

If post-training simply sharpened the pieces already doing the job, **d** would be large; the log calls this "strengthen." It never is: the log reports at most 0.08 for every operation and task (0.03–0.08 on the ladder, 0.03–0.05 in E13). There is a small discrepancy: E7's card reports 0.05–0.11, which exceeds the "at most 0.08" stated elsewhere.

The route split does not decide a pass because of the decision of 2026-10-19:
- The criterion is "reading A": the decomposition stays faithful with Δ not absorbing the update.
- Demanding a large direct share would be "reading B," a claim about mechanism that no map has met.

So the route split describes where the behaviour lives; it is not a requirement.

**10.** "Reveal" means the gain runs through Γ re-switching existing pieces (large r). "Create" means it lives outside the CI-masked model, in masked-off pieces or Δ.

**Reveal side:**
- emoticon (r about 0.55–0.60 in E6, rising to most of the gain in E5);
- ladder rungs 0 and 1 (r 0.75–0.92);
- the identity and copy cipher tasks (r 0.76–0.80 in E10 and E12).

**Create side:**
- the random-mapping cipher. In E10 the masked-off share is 0.75–0.86. In E12, r is 0.27–0.29 and masked-off is 0.45–0.53.

**In between:** the rare-target ladder tasks at rung 2 (r 0.3–0.6, with the masked-off share taking over).

The result is suggestive, not firm:
- The audit rates it "holds, one format, one seed." E12 added two tasks × two seeds per class, but it is still one format, 32 pairs and full fine-tunes only.
- There is an internal inconsistency. E12 claims to reproduce E10 "to three digits," yet its masked-off share (0.45–0.53) is well below E10's (0.75–0.86). Worth checking in the reports.
- Whether re-switching is a real mechanism or a leak of the trained model's activations through Γ's input is untested.
- The switched-on set carries only 40–64% of the effect.
- "Re-weighting" itself is explicitly not yet defined.

**11.** E7 originally concluded that a LoRA on the paper's emoticon matrix is no easier to commute than a full fine-tune, and harder by the weak version. The same-day "LoRA failure and fixes" review found a site confound:
- The decomposition reads only 21% of that matrix's input energy. Even the fine-tune's own update there is 0.36 outside the pieces.
- So a LoRA on a badly-read matrix had been compared with a fine-tune spread over all 24 matrices, most of them well read.

In response they:
- designed E8, with the LoRA on all 24 matrices through a shared harness beside the fine-tune;
- reran the fine-tune split on an L40 to check a separate cross-card discrepancy (see 12);
- dropped a planned read-side LoRA arm.

E13 later confirmed the diagnosis: rank-8 component training fails confined to one matrix (4.52) but passes on all 24, so the single matrix, not the rank, was the problem. E7 also has a provenance caveat: the pod was terminated before the final results file was downloaded, and three arms were rebuilt from the log.

**12.** In E7, on an RTX 4090, the fine-tune's least-squares split read 7.3 where E2, on an L40, had read 5.41: a 35% miss with the same code. E8 reran it on an L40 and reproduced E2's 2.18 and 5.41 to three digits, so the difference came from the card's adversarial search, not the model.

The seed spread of the adversary at the unchanged model also differs by card. The overview gives about 4% on an RTX 4090 or L40S and up to 21% on an L40; the audit lists 3.5%, 4.6% and 21%. The log doesn't explain why the card matters.

The current rule:
- Adversarial ratios are compared only within one card.
- Every job measures its own floor over three adversary seeds.

**13.** The rescaling split only rescales existing pieces. It cannot change their directions, so it can only express the part of an update that lies in the span of the fixed rank-one pieces. E1 measured that span directly:
- Only 0.06% of a real fine-tune update's energy lies in it, against 0.035% for a random direction.
- The filed prediction was ten times random, so it was reversed.

Almost none of the update is a rescaling, so the map cannot carry it regardless of the checks. One unexplained loose end remains: the emoticon update's rescaling share grows from 1.7 to 10 times random between steps 10 and 300. The planned LoRA-split run reports the rescaling share again as the strictest span.

**14.** The log doesn't say. It refers to "the paper's emoticon piece," its "six siblings" and "the paper's emoticon matrix," and notes that piece ranks second in its matrix at step 10 in E1. But it never gives the layer, the matrix type, the subcomponent index, or which emoticon token is the trigger. Those would have to come from the VPD paper itself or the project's code (e.g. the task definition in src/commute/ or experiments/lift_test). I'd rather not guess.


## Round 3

### Questions

Third batch:

15. The "outside share" and the "residual share" both seem to be about Δ. What is the difference between them? Could one be large while the other is small, and did that ever happen?

16. E13 was deleted at the user's request. What was E13, why was it deleted, and what is "the LoRA split" that the user actually wanted? How does it differ from what was run?

17. The log says the Pile checks are "blind to the trained behaviour". Why is that, what number supports it, and what was done about it?

18. What is a "saturated matrix", which ones are they, and why do they matter for the LoRA results?

19. Walk me through the task ladder (E5): how were the tasks chosen, what is a "rung", what is "coverage", what was the hypothesis, and why was it refuted?

20. The log says the LoRA-versus-fine-tune comparison in E8 was "never like for like". Explain why, say what would fix it, and tell me whether the fix has been run.

21. What are the six ladder tasks? Give me the trigger and target words of each, and the rung each one sits on.

### Answers

**15.** They measure different things.

**Outside share** is a property of the update δW itself, before any map is applied. It is the fraction of δW's effect on real inputs that lies in (I − P_U) δW (I − P_V). That is the part no change to the alive pieces can produce, so any split must leave it in Δ. It is reported against the same number for a random update (its isotropic null).

**Residual share (e_Δ)** is behavioural and depends on the map. It is the fraction of the target's log-odds gain at the trigger that Δ alone carries after the map. It is 1 for do-nothing by construction and 0 for component training, which never touches Δ.

So one can be large while the other is small, and it happened repeatedly:
- **E2:** about 30% of the fine-tune's update was outside the pieces, yet dropping that part kept the behaviour. The least-squares split left only 0.09 of the behaviour in Δ (E8).
- **E9, LoRA plateau:** 62% of the update was outside the pieces and Δ held 0.54 of the energy, but only 0.31 of the behaviour.
- **E12:** the outside share was about 0.6 of its null in both task classes, while the Δ share of behaviour was 0.12–0.24.

The log also uses a third quantity: Δ's share of the update's energy after a split, e.g. 0.40 in E13. That is neither of the two above, and it is easy to confuse with both.

**16.** E13 is rank-8 component training on all 24 matrices.
- **Setup:** offsets dU = P Sᵀ, dV = Q Tᵀ, with Γ and Δ frozen, on emoticon for 30 steps with two seeds. Controls were the same restricted to the paper's single matrix, plus free component training.
- **Results:** adversarial ratios of 1.14 and 1.28 on all 24 matrices, but 4.52 on one matrix. This showed that the single matrix, not the rank, caused E7's failure.

It was deleted because the same-day terminology review found it was the fifth run to answer the user's LoRA question with component training instead. At the user's request the directory, notebook, report and Slurm script were removed; the log card and TRIED's row are the only record. The log doesn't quote the user's own reason beyond that context.

**The LoRA split** the user wanted works like this:
- train an actual rank-r LoRA;
- take its finished BA;
- split it over the subcomponents by both the least-squares and the adjoint split;
- save the per-subcomponent updates and count how many carry half the energy;
- report the outside share in four blocks (inside both spans, write side only, read side only, outside both) against its null.

E13 instead constrained the pieces' own training to a LoRA's rank. It never mapped a trained LoRA's update onto the pieces.

There is an inconsistency here. The card says "No LoRA is trained in it," yet its Run line lists "a rank-8 LoRA split and placed do-nothing," and Found reports that arm (1.13, 0.22 of behaviour in Δ). Even if that arm existed, it was only the least-squares split: no adjoint split, no saved per-piece updates, no four-block share. The full LoRA split is listed as "Running" in section 6 with no results yet.

**17.** The Pile checks score Pile validation rows 0–255. Those rows contain about three trigger positions in 131,000 tokens. So a ratio near 1 shows that the decomposition's account of the old model's behaviour on ordinary text survived. It says almost nothing about the newly trained behaviour. The 2026-10-15 faithfulness-scope review found this and re-scoped E2's conclusions accordingly.

The response was:
- **a task-local harness:** the target's log-odds gain on held-out trigger windows, the route split, and off-target KL at the other 40 positions;
- **a protocol rule:** every plan must state what each metric is blind to and name a control that must fail.

The fix is incomplete. The task-local adversary has no power: its seed spread was 55–88 nats against gains of 4–15 (E5). The log concludes that nothing measures faithfulness at the trigger with power. Reading the Pile adversary at the trigger is in the backlog.

**18.** A saturated matrix is one whose alive pieces span every direction. Any update to it is then expressible by construction: its outside share is zero no matter what the update looks like. There are four, the two MLP matrices in each of layers 0 and 3.

The wording is slightly inconsistent: section 3 says the pieces span every direction, while the glossary says "on one side." One side is enough to make the obstruction zero.

They matter for the LoRA results because a LoRA whose energy sits on saturated matrices will look expressible regardless of its random read side. E8 tried to explain its Δ share this way: seed 0 put 60% of its energy on saturated matrices. But seed 1 put only 25% there and left the same share of behaviour in Δ. So the explanation fits one seed only, and the 2026-10-20 review flagged it as unsupported.

**19.** Each part of the design:
- **Tasks:** six next-word tasks mined from the Pile (an example is " will" followed by " be"). There are two per rung, one high-coverage and one low-coverage. Emoticon was added as a seventh.
- **Rung:** the decade of the target's baseline probability. Rung 0 is 0.1–1, rung 1 is 0.01–0.1 and rung 2 is 0.001–0.01.
- **Coverage:** the share of the first component-training step's squared update that lands on pieces Γ rates highly important at the trigger.
- **Hypothesis:** high coverage would predict retention. If the gradient lands on pieces Γ already uses, the CI-masked model should keep more of the learned gain.

It was refuted for these reasons:
- Both filed predictions came out null.
- The direct share, the channel coverage could plausibly move, carried only 0.03–0.08 on every task.
- Retention was driven by re-switching, which ordered by rung rather than coverage: 0.75–0.92 at rungs 0 and 1, and 0.3–0.6 at rung 2.
- Re-switching agreed within 0.16 across the fine-tune, component training and gated-off component training.
- Where re-switching was small, the masked-off share took over, not the direct share.

Further caveats:
- The gated-on positive control did not learn in 10 steps.
- The task-local adversary had no power.
- Optimiser-seed noise was measured on emoticon only.
- States were saved only for emoticon.

**20.** E8 scored the split LoRA at step 2, where it had learned, and the split fine-tune at step 10. E2 had already shown that the fine-tune's split is near the floor early (1.14 at step 1) and only becomes fragile later (5.41 at step 10). So "LoRA 1.13 vs fine-tune 5.41" compares different points in training.

E9 then followed the LoRA to step 1000 on an L40S but ran no fine-tune arm, and compared against the L40 reference. Adversarial ratios are card-dependent, so that comparison doesn't hold either.

The fix is both update types at matched steps, in one job, on one card. Section 6's first item is designed to do this. It reruns the emoticon fine-tune with updates saved at the same nine steps as E9's LoRA, then puts do-nothing, least-squares split and adjoint split of both through the Pile checks at steps 10, 100 and 1000. It is marked "Running" with no results in the log, so the fix has not been completed.

Matching steps still won't match learning rate (1e-3 vs 3e-5), update size or behavioural effect. It removes the training-point and card confounds, not all of them.

**21.** The log doesn't list them. It gives only one example, " will" followed by " be", without saying which rung it is on. Beyond that it says only that there are two tasks per rung and baselines span 0.25 down to 0.003. E5's figure is labelled by trigger and target words, but the extracted text doesn't include those labels. The full list should be in experiments/task_ladder (its results file or src/commute/ladder.py).


## Round 4

### Questions

Fourth batch:

22. If you joined this project tomorrow with one GPU-hour and one afternoon, what would you run first, and why that rather than the alternatives?

23. What are the three biggest weaknesses in the evidence so far, in your judgement? Rank them.

24. What does the "re-weighting hypothesis" mean in this project, and how is it operationalised? Is the project's evidence for or against it so far?

25. What loss is the fine-tune trained on? Give me the KL penalty coefficient, the batch size and the learning rate.

26. Roughly how much did the experiments cost, and how many GPU-hours were used in total?

27. Describe Γ's architecture: how many layers and how wide is it, what exactly does it take as input, and how does it produce an importance value per piece?

28. What is the "adjoint split", how is it related to the least-squares split, and what is K⁺? Would you expect the adjoint split to do better or worse than the least-squares split on the Pile checks, and why?

29. What learning rate does component training use, and is it the same in every experiment that used it?

30. Which GPU was each of E5, E12 and E13 run on?

### Answers

**22.** First I'd check whether the LoRA split from section 6 has finished, and read its output. It is marked "Running," costs about 40 GPU-minutes, and is the user's actual question, asked at least eight times. If it hasn't run, it gets the GPU-hour before anything else. Duplicating or pre-empting it would repeat the project's main process failure.

Assuming it is done or in flight, I'd split the time this way:
- **GPU: the cipher matched-size check** (about 10 minutes, saved E10 states, no training). It is the cheapest test that turns an ambiguous result ("breaks somewhere between 1 and 32 associations") into a clear one about update size versus novelty. Novelty is the "new capabilities" branch of the user's question. The user parked this work as "trying to do too much at once," so I'd confirm with them before running it.
- **Afternoon: the CPU cancellation tests on saved E11 states.** These are the coherence statistic and the random half-revert. They are the most direct test of why component training fails the adversary, which is the project's main open mechanism question.
- **Leftover GPU time: more adversary seeds** on the verdicts sitting near 1.5. Two of seven ladder verdicts depend on which side of an uncalibrated line they fall.

I'd defer the larger-model and RL phases until the measurement problems are fixed. I'd also defer the per-step split, which needs a tolerance rule that doesn't exist yet.

**23.** My ranking:

1. **Nothing reliably measures faithfulness on the trained behaviour, and the pass criterion is unanchored.** The checks that decide passes run on Pile rows with about three triggers in 131,000 tokens. The task-local adversary has no power. The 1.5 line sits inside a floor that reaches 21% on the L40, and the Δ condition has no threshold. Every "pass" therefore means "the old model's account survived," and borderline verdicts may be noise.
2. **The central mechanism finding rests on an untested interpretation.** Most gain runs through re-switching, but whether that is a mechanism or the updated model's computation leaking in through Γ's inputs is open. "Re-weighting," "recruit" and "create" are undefined, and the switched-on set carries only 40–64% of the effect. Some points help it: random updates are inert, and different targets give different sets. Still, "reveal versus create" is unproven as a statement about mechanisms.
3. **Narrow, thinly replicated evidence on the comparisons that matter.** There is one 67M model with self-made 10-step fine-tunes and no real post-training. Single seeds or bijections carry the 32-pair failure and the Δ share at the plateau. The reveal/create split is one format. The LoRA-versus-fine-tune comparison is confounded by step and card, and the LoRA split itself has never been run. Lost files and uncorrected reports add to the weakness.

**24.** Informally, the hypothesis is that post-training re-weights mechanisms pretraining built rather than creating new ones. Its operationalisation was fixed on 2026-10-20 and is explicitly provisional. Re-weighting means the update leaves the subcomponents as they were while Γ switches them on differently, measured by the re-switching share r.

On that definition the evidence is mixed and limited:
- **For it,** for behaviours the model already has. Emoticon, ladder rungs 0–1 and the copy tasks run mostly through r, and the direct "strengthen" share is at most about 0.08 everywhere.
- **Against it,** for the one genuinely new behaviour, the random-mapping cipher. Its gain sits mainly in pieces Γ keeps off.
- **Partly circular.** It is close to expected that behaviours the model already has get re-routed. The hypothesis is about real post-training, which hasn't been tested.

So the evidence so far shows that the decomposition can tell existing behaviours from new ones, rather than settling the hypothesis. It also leans on the untested assumption in my point 23.2.

**25.** The log gives Adam, learning rate 3e-5 and batch size 32, on the 24 decomposed matrices, with a KL penalty to the original model at the non-trigger positions. It doesn't give the KL coefficient. It also doesn't state the task loss explicitly; cross-entropy on the target at the trigger is the natural reading, but that is my inference. Both should be in the experiment configs or src/commute/.

**26.** The log doesn't say. The v2 changelog says "costs removed." The only figures are per-run: "each takes minutes," about 40 GPU-minutes for the LoRA-split job, about 10 minutes for the cipher check. No total cost or GPU-hour count can be derived from it.

**27.** The log gives only an outline:
- one network shared across all 24 matrices;
- it reads every decomposed matrix's input at every position;
- it runs "a small transformer over the window";
- it outputs an importance value in [0, 1] per piece per position.

The number of layers, the width, how the 24 inputs are combined, and the output head or squashing are not given. These would be in the VPD paper or the param-decomp package at commit 74146b5.

**28.** The adjoint split maps a finished update with no solve: dU = δW V and dV = δWᵀ U. To first order the weights change by UUᵀδW + δWVVᵀ, which is K(δW). K is the operator from the component-training section, K(G) = UUᵀG + GVVᵀ. Δ stays fixed, so the result is not the fine-tuned weights.

The log says the least-squares split is the adjoint split "after preconditioning by K⁺." The log doesn't define K⁺; I read it as the pseudo-inverse of K. On that reading, least squares applies the adjoint to K⁺(δW) instead of δW. That is what makes its weight change equal the expressible part of δW exactly, rather than K's distorted version of it.

Neither has been tested, so my expectation is reasoned, not measured:
- **Likely better on the adversarial check.** K is ill-conditioned when pieces overlap, so K⁺ can amplify small-eigenvalue directions into large, opposing changes on near-collinear pieces. That kind of cancellation is exactly what a selective adversarial mask would expose. The adjoint changes each piece in proportion to its overlap with δW, with no amplification.
- **Likely worse at being the fine-tuned model.** K(δW) over-weights directions covered by both sides and under-weights others. Its scale has to be chosen, which is why the plan says "scaled adjoint." It may not reproduce the behaviour, and the 2026-10-19 review dismissed it as "a worse preimage" for that reason.

So my guess is more robust but less faithful to W + δW; the LoRA-split job should test both.

**29.** The nominal rate is Adam at 1e-5 on the alive pieces' vectors. It was not the same everywhere:
- E2's first grid used steps 3 to 100 times the fine-tune's and gave an adversarial ratio of 41. It was then redone at matched update size.
- E4 tried plain-SGD rates, none of which learned without large off-target damage.

The log doesn't give per-experiment rates for the sparse, rank-r or gated-off variants. It notes that component training runs "at another learning rate" from the fine-tune, so I can't confirm that 1e-5 was used throughout.

**30.** E13 ran on the MATS cluster L40. E5 says only "RunPod" without naming the card. E12's card gives no compute line at all. Since adversarial ratios are card-dependent, the missing card for E5 and E12 is worth recovering from the reports or job logs.


## Round 5

### Questions

Last batch, a few follow-ups on things you said.

31. You said component training "passes" on the one-pair cipher at 1.20 and on the identity control at 1.19. Is that the same 1.5 criterion you described earlier, or the "within the floor" criterion? Would those two numbers pass the "within the floor" rule on the card they were run on?

32. Do-nothing puts the whole update into Δ and leaves the pieces untouched. So why does do-nothing read 2.18 on the adversarial check on emoticon, and 4 to 18 on the unmasked check in E11, rather than 1? Walk me through the mechanism.

33. The log refers throughout to "the user". Who is the user, who did the work, and who wrote the log?

34. The glossary defines "reveal" as Γ switching existing pieces on, but the grid in section 4 uses "re-weight" for that and adds "re-wire". Are "reveal" and "re-weight" the same thing? And what is "re-wire", and did any data land there?

35. Of the 13 experiments, which ones actually moved the main question (Q1) forward, and which were detours? Be blunt.

36. "Round" numbers and E-codes are both used. Which experiments have a round number, and which don't? Why do some not have one?

### Answers

**31.** I applied the 1.5 line, not the floor rule. Under "within the floor" (at most 1 plus three floors), the answer depends on E10's own measured floor, and the log doesn't report it. It also notes E10 used only one adversary seed per state.

E10 ran on the MATS cluster L40, whose floor the log puts at 21%. If E10's job saw that, the band is about 1.63, and 1.20 and 1.19 pass comfortably. If the floor were nearer the 4% seen on other cards, the band would be about 1.12, and both would fail.

So likely yes on that card, but unverified. On the L40 the floor rule is also looser than the 1.5 line, which is part of the "pass line has no anchor" problem.

**32.** The checks score the masked model against the full updated model. Under do-nothing, the update lives entirely in the residual (Δ + δW), so any check that removes or masks the residual removes the update. That is my reading; the log doesn't spell out the mechanism.

- **Unmasked check** (all pieces on, residual out). This is essentially the original pieces' model with none of δW. Its KL to the updated model therefore includes the whole update's effect on Pile text. The baseline is the unchanged model's unmasked KL, which is small because the pieces reproduce the original well. A modest absolute KL thus becomes a large ratio, 4–18 in E11.
- **Adversarial check.** The residual gets its own mask, so the adversary can turn down Δ + δW and strip out the update, on top of choosing damaging piece masks. The ratio is smaller (2.18 on emoticon), mainly because its denominator, the unchanged model's adversarial KL, is much larger than the unmasked baseline.

Why the update shows up on Pile at all: although Pile rows contain almost no triggers, the fine-tune still shifts predictions elsewhere. The KL penalty limits this off-target change but doesn't remove it.

This is exactly why E1's original check, which recomputed Δ from the weights, let do-nothing pass trivially. It is also why E11's caption says that keeping Δ fixed is "what component training buys."

**33.** The log doesn't name the user. They are the person who posed the question on 2026-10-14, owns GOALS.md, and makes the decisions quoted in it (parking the cipher work, deleting E13). The byline reads "Edwin Agnew, with coding agents," at MATS, and the cluster paths are under e.agnew.

My inference is that Edwin is the user, directing coding agents that did most of the implementation. The "Words" card refers to "the assistant" introducing terms and sessions overriding the user. The log itself appears agent-written for the user to read: the v5 changelog entry reads "From the user's reading of v4." The log never states either point outright.

**34.** They name the same share. The glossary's "reveal" and the grid's "re-weight" are both the re-switching share r (original pieces, new masks). The route-split table also labels r "reveal." It is a naming inconsistency, not two concepts.

The grid's cells map to shares as follows:

| Update goes into | Masks unchanged | Masks change |
|---|---|---|
| Subcomponents | Strengthen: d | Re-wire: moved pieces that Γ also switches differently |
| Δ (pieces as before) | Create: e_gate + e_Δ | Re-weight: r |

The log doesn't say which share measures re-wire. Structurally it should be the interaction share i. The log never reports i separately, so I can't say whether data landed there. The grid marks only re-weight (green) and create (red) as occupied.

There is one more wrinkle. The cipher's "create" result is mostly the masked-off share: pieces Γ keeps off, not Δ. Yet the grid places "create" in the Δ row.

**35.** Bluntly:

**Moved Q1 forward:**
- **E2:** the core result. Component training passes on emoticon, and the least-squares split is fragile.
- **E11:** showed component training's pass doesn't generalise.
- **E10:** the cipher passes and failures.
- **E8:** matched-site LoRA, and it closed the cross-card issue.
- **E9:** the LoRA's read side never moves, and Δ takes a third of the behaviour.
- **E13 and E7:** relevant via the single-matrix diagnosis. But both answered component-training questions instead of the user's LoRA split, and E7's original conclusion was confounded.
- **E1:** killed the rescaling split, but its checks were flawed.

**Detours for Q1:**
- **E3 and E4:** mechanism work. E4's sparse-arm Pile numbers are a Q1-relevant side result.
- **E5 and E6:** route-split mechanism.
- **E12:** explicitly "a step back from commuting the update."

The mechanism work produced the project's most interesting findings, but it is Q5 rather than Q1. Meanwhile the experiment that most directly answers the user's form of Q1, the LoRA split, still hasn't run after a week.

**36.** Seven experiments have a round number: E1 (round 1), E2 (2), E5 (3), E7 (4), E8 (5), E10 (6) and E12 (7). Six don't: E3, E4, E6, E9, E11 and E13. The log says only that round numbers are the reports' and E-codes are the document's; it doesn't explain the gaps.

My inference is that rounds mark the main planned designs, and the unnumbered ones are follow-ups:
- E3 and E4 follow E2;
- E6 follows E5;
- E9 follows E8;
- E11 follows the ladder;
- E13 was an extra.

Filed predictions don't explain it, since E4 had them too.
