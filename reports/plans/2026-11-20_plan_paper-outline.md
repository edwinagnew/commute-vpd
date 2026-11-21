# Paper outline: A Mechanistic Foundation for the Elicitation Hypothesis (draft plan, 2026-11-20)

For the user to cut or approve before more prose is written. Built on Nanda's "Highly Opinionated Advice on How to
Write ML Papers" (lesswrong.com/posts/eJGptPbbFPZGLpjsp): one to three claims, every main-body section must say what
breaks if it is cut, strongest evidence in the body and everything else in the appendix, claims hedged to the evidence.
Main body aimed at ~9 pages, five figures.

## The three claims (what a reader should leave with)
1. **The elicitation hypothesis needs a definition, and any good one is relative to a featurization.** We give one:
   a fine-tune reuses the base model, for a behaviour, to the extent that the parts the behaviour runs through still
   compute what the base model's parts computed (causal abstraction, Geiger vocabulary). It is total: every fine-tune
   gets a reading. *Evidence:* the definition and its properties (§3). Strength: conceptual.
2. **At the granularity of neurons and heads, "only re-weighting" is too weak a claim to test.** Training only one
   gain per neuron and per head (12,288 + 24 numbers) on a frozen 67M reaches 0.72-0.91 of the gain on tasks built to
   be new, about as far as on tasks built to reuse. So the hypothesis is only meaningful at a coarser unit, which is
   what claim 1's featurization says. *Evidence:* gain-only training on the five synthetic key tasks (one figure).
   Strength: systematic on one small model; learning rate not swept.
3. **Fine-tunes with a known answer, plus size, prior and random controls, are the bar any reuse measure must pass,
   and nothing we tried yet passes it.** On built fine-tunes the cheap explanations sort most cases (update size, base
   prior, steering); our definition's readings follow update size at heads/layers/blocks and saturate or reverse at
   neurons and VPD subcomponents; the re-gating share separates every built pair but separates pure-gain updates by
   task just as well, so it reads the task, not the update, and its KL form does not beat a matched random change.
   *Evidence:* the five-task graph per measure (pending P2.50), size-matched R, scorecard. Strength: negative result
   on one small model; framed as a test bed and a standard of evidence, not a verdict on the hypothesis.

What we do **not** claim: that the hypothesis is true or false; that VPD is or is not the right featurization; that
the re-gating share works.

## Section plan (main body)
| § | content | pages | what breaks if cut |
|---|---|---|---|
| Abstract | context, gap (EH stated, never defined), 3 claims, "preliminary, one 67M model" | – | – |
| 1 Intro | EH and why it matters for monitoring post-training; why mechanisms; the 3 claims; contributions list; Fig 1 (definition on a toy graph) | 1.5 | everything |
| 2 Related work | user's; short | 0.75 | positioning |
| 3 Definition | 3.1 formal definition and properties; one-paragraph mention of the featurizations (neurons, heads/layers, VPD, SAEs) | 2 | claim 1 |
| 4 Set-up | 4.1 model + VPD (3 sentences); 4.2 featurizations read (user's text, one table); 4.3 fine-tunes with a known answer: five key tasks at one update size + three controls (twins, hand-made edits, random change); 4.4 measures: re-gating share, gain-only training; baselines in one paragraph | 1.5 | reproducibility of 2-3 |
| 5 Results | 5.1 strength alone learns new tasks (Fig 2, claim 2); 5.2 the definition's readings follow update size (Fig 3: R gap before/after size match; VPD reversal in one sentence); 5.3 the share and the cheap baselines on the five tasks, with the random band and the pure-gain reading (Fig 4); 5.4 scorecard of every measure against the controls (Fig 5 or table) | 2.5 | claims 2-3 |
| 6 Discussion | what a passing measure would need; what this says about the EH; recommendation that diffing methods be scored on built fine-tunes; limitations; departures | 1 | honesty |

## Moved to the appendix (one sentence and a pointer in the body)
- Worked example of the copying head (illustration only; its own caveats are long). *Was the Results opener on the
  11-20 skeleton: user to confirm the move.*
- Held model (needed for the Qwen appendix only), gate variants and untrained-SVD share, top-neuron and neuron-level
  baselines, crosscoder details, grafts, structured control, key-set variants, copy-at-new-marker/lookup.
- Realistic fine-tunes (three real-text pairs, British-spelling twin: mostly no call) and Qwen3-1.7B (no call):
  one paragraph in Discussion "how far this gets beyond toy tasks".
- Carrying an update onto the decomposition (the project's original question).

## Figures (main body)
1. Definition on a toy graph (exists; panels c, d to redraw).
2. Fraction of gain reached by gains alone, five tasks on the x-axis, reuse tasks vs new tasks (exists for nine
   pairs, E231; redraw on the five-task set when P2.50 lands).
3. R gap per pair, saved vs size-matched, with the size-only prediction (exists, E232; relabel).
4. One panel per measure (share, R_VPD, prior, steering): five tasks on the x-axis, random band (P2.50, due 11-22).
5. Scorecard (exists as table + figure; trim to the measures the body names).

## Pending inputs
P2.50 five-task graph (11-22) feeds Figs 2-4; R_SAE (P1.40, 11-23) is one row of Fig 4/5 or a sentence; crosscoder
replication (P2.49) is one row of the scorecard. Definition wording waits on definition-lead (may change by 11-21).

## Questions for the user
1. Are these the three claims? (Claim 2 is the one result that is crisp and pre-registered; claim 3 is the bulk.)
2. Worked example to the appendix?
3. Real text and Qwen as one Discussion paragraph plus appendix, rather than Results subsections?
