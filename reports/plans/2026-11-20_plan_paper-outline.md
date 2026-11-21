# Paper outline: A Mechanistic Foundation for the Elicitation Hypothesis (draft plan v2, 2026-11-20)

v2 after the user's comments (11-20 evening): claim 1 must say mechanistic evidence as well as featurization; drop the
"re-weighting is too weak" claim; proxies are scored against the definition, not the known answer (size, prior and
random controls stay); the hoped-for claim is "a couple of proxy measurements that sort of work, with caveats", made
only if the results bear it out; the copying head stays; real text and Qwen are Results subsections, as negative
results. No page target. Writing advice followed: Nanda, "Highly Opinionated Advice on How to Write ML Papers"
(lesswrong.com/posts/eJGptPbbFPZGLpjsp).

## Claims
1. **Whether post-training elicits or creates is a question about mechanisms, and needs a mechanistic definition
   relative to a featurization.** Behavioural evidence (a fine-tune is fast, cheap, low rank) cannot settle it; we
   define reuse as the parts a behaviour runs through still computing what the base model's parts computed, at a
   featurization fixed from the base model. Evidence: §3, and the copying head as the worked case.
2. **(Hoped for) A couple of cheap proxies track the definition, with stated caveats, and do better than the baselines (crosscoder included).** (User 11-20 ~20:55: strong yes to the first part; only weak to framing it as "scored against R with size, prior and random controlled", which become checks, not the frame; baselines stay in the main body.) Candidates: the re-gating
   share, and whichever baselines survive. Scored against the definition's own reading, with update size, base prior
   and a matched random change controlled. Caveats stated with it: one 67M model, built tasks, the share's random
   control.
3. **Beyond built tasks the proxies do not yet give a call:** real-text fine-tunes and Qwen3-1.7B (negative results,
   unless real text turns around).

## What claim 2 needs (the gap to close)
- A reading of the definition that is not just update size. Coarse R follows size (E232: signs kept on 5-6 of 9 once
  sizes match). Since 11-20 two $0 reanalyses help: R with size removed (residual against the size-only rule) orders
  all six hand-made edits (margin 0.57 against a band of 0.25) and puts reuse higher on 7 of 9 pairs, ranking with
  size at -0.37 (P1.42); a size-normalised R orders the edits but only 5 of 9 pairs, and rises with size (P1.43). Both
  post-hoc; definition-lead may still change them; R_SAE lands 11-23.
- **The proxies scored against that reading**, not against how the pairs were built. Not yet computed. It is a $0
  reanalysis of saved readings (share, prior, steering, rank, size against size-removed R on the nine pairs and the
  edits). Recommend asking definition-lead or compare-lead for it as a cell.
- The five-task graph at one update size (P2.50, 11-22), with the random band.

## Section plan
| § | content | what breaks if cut |
|---|---|---|
| Abstract | EH stated, never defined mechanistically; definition; proxies scored against it; how far that gets | – |
| 1 Intro | EH and monitoring post-training; why behaviour is not enough; claims; contributions | everything |
| 2 Related work | user's | positioning |
| 3 Definition | formal definition, properties; one-line mention of featurizations (3.2 deleted) | claim 1 |
| 4 Set-up | model + VPD; featurizations (user's 4.2); fine-tunes with a known answer: five key tasks at one update size + twins, hand-made edits, random change; proxies: re-gating share, baselines in one paragraph | reproducibility |
| 5.1 Worked example: copying head | the copy gain runs through a base copying head (head off removes 0.47-0.56 of it, none of the lookup's); the share reads 1.09 on copy and depends on the head through its gates. This is the mechanistic evidence the definition asks for, on one behaviour, and the share agreeing with it. Gap: R not read on this pair (a small cell would close it) | claim 1 made concrete |
| 5.2 The definition on built fine-tunes | R at each featurization; follows size until size is removed; size-removed R orders the edits | claim 2's reference |
| 5.3 Proxies against the definition | five-task graph per proxy with random band; share vs size-removed R; scorecard | claim 2 |
| 5.4 Realistic fine-tunes | real-text pairs and twin: no call so far | claim 3 |
| 5.5 Scale: Qwen3-1.7B | no call | claim 3 |
| 6 Discussion | what the evidence supports; what a passing proxy needs; limitations; departures | honesty |

## Appendix (one sentence and a pointer in the body)
Gain-only training (one line in 5.3 as a baseline: re-weighting units alone reaches new tasks' gains), held model
(Qwen needs it), gate variants and SVD share, top-neuron and neuron-level baselines, crosscoder, grafts, structured
control, lookup and key-set detail, carrying an update onto the decomposition.

## Questions for the user
1. Ask for the $0 "proxies against size-removed R" reanalysis as a cell? (Recommend yes: claim 2 cannot be written
   without it.)
2. Ask for R on the copy/lookup pair so the worked example reads the definition too? (Recommend yes, if cheap.)
