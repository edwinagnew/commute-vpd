---
slug: jain2023-finetuning-procedural
title: Mechanistically Analyzing the Effects of Finetuning on Procedurally Defined Tasks
authors: Jain et al. (Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktäschel, Krueger)
year: 2023
id: 2311.12786
tags: [EH, DEFINITION, FT-INTERP, BASELINE, KNOWN-ANSWER, SUPPORTS]
primary_tag: DEFINITION
in_our_bib: jain2023
cited_in_draft: yes
themes: [elicitation, interpreting-finetunes, reuse-definitions, baselines]
read_from: "local text of the ICLR 2024 version (arXiv v2): abstract, Sec. 1-3 (definition of capability), Sec. 4-5 (Tracr, PCFG pruning/probing/reFT, TinyStories), Table 1, Fig. 7-11 captions"
digest_buckets: [1-missing-method, 2-known-answer-case, 3-definition, 4-supports]
importance: 8
---

# Mechanistically Analyzing the Effects of Finetuning on Procedurally Defined Tasks (Jain 2023)

**What it does.** Asks whether fine-tuning creates new capabilities or modulates existing ones, in controlled settings:
Tracr-compiled transformers (counting, max-identifier) and small transformers pretrained on PCFG data (count or index a
token), with a TinyStories-Instruct check. The model is fine-tuned at a medium or small learning rate (pretraining LR
over 10 or over 100) to count a different token, sometimes with a spurious correlation that makes the pretraining
capability "weakly relevant". What changed is read with pruning, linear probes, attention maps and reverse fine-tuning
(reFT).

**Key result.** Fine-tuning "rarely alters the underlying model capabilities". A localized "wrapper" is learned on top.
Pruning only 5 to 15 neurons restores the pretraining task and removes the fine-tuned one (Fig. 7). Probes still read
the pretraining count after fine-tuning. reFT revives the capability in a few steps. With a larger LR and no spurious
correlation (C_Tr = 0), Tracr attention does change, so real alteration does happen. On TinyStories, a model fine-tuned
not to write twists relearns them faster than a model never pretrained on twists (e.g. 81% against 31% at 30 iterations,
Table 1).

**Quote.** "a minimal transformation, which we call a 'wrapper', is typically learned on top of the underlying model
capabilities, creating the illusion that they have been modified" (abstract)

**Relevance to our paper.** This is the closest prior definition. Its Definition 1 says a model "possesses a capability
C" if a linear readout at some layer computes f_C on the sub-domain. That is a readout definition, not a mechanism
definition: our claim that two fine-tunes with one input-output behaviour can differ under our definition holds against
it only partly, since the probe looks inside too, but it reads values, not mechanisms. The draft already contrasts the
two and adapts Jain's pruning, reFT and readout tests as baselines (they fail their controls on our setting). Its Tracr
and PCFG settings are known-answer cases by construction: a compiled capability, a weakly vs strongly relevant regime,
and an LR switch under which the computation does change (larger LR, C_Tr = 0). They are close in spirit to our built
edits and key tasks, and the draft should say how ours differ (realistic 67M LM, weight-level mechanism). It supports
EH in toy settings, and it gives a concrete case where fine-tuning does build something new: a known "new" arm we could
cite.

**Why the project used, then dropped, this definition (project history, 2026-10-29 to 11-18).** From 2026-10-29 the
project's working definition of reuse was Jain et al.'s section 3 (decisions.md, "Reuse line ... Jain, Kirk, Lubana").
*Possession*: a linear readout trained on Pile text at some block reads the capability, beating a Hewitt-Liang control
task. *Reuse*: the fine-tuned behaviour is a small function of that readout. *Causal check*: removing the readout's
direction breaks the fine-tune and spares a twin fine-tuned on shuffled labels. On 10-30 we added that the function
must also work on held-out values, a stated departure: without it, any fine-tune keyed on something the model
represents counted as reuse, so the label separated no 67M pair. On 11-06 the definition was reset to Jain's alone. It
was dropped for four reasons.
1. *It never found reuse on the 67M.* The clearest case, the case-label fine-tune, passed possession, carrying and the
   small-function test, then failed the causal check three times: E112's removal took only 8-21% of the gain, a
   stronger removal at every later block still read "not through C", and E113's patch flipped the fine-tune's label on
   0.2-1.4% of inputs. The log's claims page: "Under the earlier Jain-based definition, no 67M fine-tune was shown to
   reuse an old capability" (E123, E125).
2. *It often said nothing.* Many cases read "unreadable" or "does not apply". The user's criteria (2026-11-09): a
   working definition must give a verdict on most fine-tunes, agree with fine-tunes built to be one or the other, and
   look inside the model; not tied to Jain. On 11-13: "no reading" was the Jain failure we set out to fix.
3. *Its core cannot see what was learned.* The user rejected the causal-tracing core itself on 11-09, not only its
   coverage: a swap from base to fine-tune cannot see what the fine-tune learned, and possession needs a hand-written
   algorithm (the capability C named in advance) to probe for. So it cannot give a verdict on every fine-tune, and it
   does not look at the parts the fine-tune changed.
4. *It reads values, not mechanisms.* Possession is a linear probe, so a built control's label simply followed from
   whether its feature was linearly readable. That is why rung L1.4 asks for a built control whose label "does not
   follow from linear unreadability".
The replacement since 2026-11-18 is the mechanistic sentence: the parts F needs still compute what M's parts compute,
up to a gain (kept AND needed). Sources: decisions.md rows of 2026-10-29, 10-30, 11-06 and the 2026-11-09 paper plan;
research-log claims and timeline; definition-lead, 2026-11-23.

**For the digest.**
- 1-missing-method: pruning-to-revive, probing and reFT are already our adapted baselines; nothing new to add, but the large-LR/C_Tr=0 regime is a "new" control we could mirror.
- 2-known-answer-case: Tracr/PCFG counting fine-tunes with compiled ground truth (wrapper at small LR; altered attention at large LR, no spurious cue); TinyStories twist deletion shown persistent by probe + reFT. Toy or tiny, not realistic.
- 3-definition: capability = a linear readout of f_C at some layer on a sub-domain; contrast with our mechanism-level "kept up to a gain".
- 4-supports: fine-tuning mostly wraps existing capabilities (EH), with an explicit exception at larger LR.
