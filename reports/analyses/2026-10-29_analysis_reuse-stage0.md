# Can our tasks say whether post-training reuses a capability the model already had?

Reuse line, Stage 0, 2026-10-29: reading and audit only, no runs.
Inputs: the Jain, Kirk and Lubana paper, arXiv 2311.12786, cited by line number of
`~/commute-data/lit_review_pdfs/Jain2024_finetuning_wrappers_2311.12786.txt` as L<line>;
the task audit `~/commute-data/scratch/reuse-lead/audit/task_audit.md`; the definition row in `decisions.md`.

We read the paper whose definition of reusing an existing capability the project adopted today, and checked every
task and post-training update in the repository against it. Most of our tasks cannot state what the capability
computes or on which inputs, including the two that carry most of the commute line's findings. Nothing measured so
far can be scored against this definition, so the next stage has to build tasks where possession and use are known
before any decomposition is read.

## The answer

Not yet. Six of our fifteen tasks and updates state the capability and its inputs clearly. None fixes during
pretraining whether the model has it; we only know absence by construction, for the random ciphers and invented
facts. The emoticon and trigger-ladder tasks, which carry most of the commute line's results, fail the definition.
The paper released no code for its grammar tasks. The grammar is fully written down, though, so rebuilding one
counting setting is about 600 lines and a few GPU hours. No probe or revival test exists in the repository yet.

## The definition, and what Jain left to us

An input is split into a part that says which capability to use and a part that carries the data. The capability is
a function of the data, $f_C(x_d)$, defined on a set of inputs $X_C$ that all carry the same selector $i_C$.

- **Possession**, Def 1 at L146-148: the model has the capability if, for every input in $X_C$, some layer has a linear
  readout, trained on pretraining data, that equals $f_C(x_d)$.
- **Reuse and relevance**, Def 2 at L167-170: fine-tuning reuses the capability if the post-trained map is
  $g \circ f_C$. The capability is strongly relevant if $g \circ f_C$ is right on the whole fine-tuning distribution,
  weakly relevant if only on the fine-tuning sample, through a spurious correlation.
- **The causal requirement**, added by the user: decodability is not enough. Ablating or patching the readout
  direction must break the fine-tuned behaviour. Or a few pruned neurons, or a short reverse fine-tune, must bring
  back the pretraining behaviour faster than a scratch model learns it.

Jain turned these into practice loosely, so every threshold we use will be ours:

- "For all inputs" has no threshold. Possession is judged by eye: a probe's per-layer accuracy curve looks "similar"
  to the pretrained model's, L427.
- The probes have no controls: no shuffled labels, no randomly initialised model, no selectivity test, no held-out
  accuracy table.
- No probe direction is ever ablated or patched. The causal weight rests on pruning and reverse fine-tuning.
- Pruning has no random baseline. It picks neurons by the gradient of the loss it then scores, so any model should
  recover somewhat. The only guard is a class-histogram check.
- Relevance is never measured. It is set by construction, through how often the fine-tuning operand appeared in
  pretraining and the rate of the spurious correlation in the fine-tuning data.

## Jain's three settings and the numbers we could compare against

| Setting | Model | Recipe | Numbers to compare against |
|---|---|---|---|
| Grammar counting and indexing: count `a` in pretraining, fine-tune to count `b` (L225-257, App. B.2, C.2) | minGPT, about 3M parameters, 6 blocks, width 192, trained from scratch | Pretrain AdamW 1e-3, 200K or 50K steps. Fine-tune on the answer only, batch 96, 10K steps, rates 1e-4, 1e-5, 1e-6. Spurious rate 0, 0.5, 0.8, 1 | Table 5, L2916: before fine-tuning, counting `b` is 27% right at the low prior and 99.9% at the higher ones. Low prior, rate 1e-5, full spurious correlation: counting `b` 0% without the correlation and 100% with it. Pruning 5-15 neurons revives counting `a`, Fig. 7. Reverse fine-tuning revives it in 0.1K-3K steps against about 4.5K for a scratch-then-fine-tuned model, Fig. 9 |
| Tracr counter and max-identifier (L211-224, App. B.1, C.1) | compiled, 3 blocks, no LayerNorm | SGD with momentum, 10K steps, batch 96, noise added to the weights first | Pruning one weight raises counting `a` by over 60 points, L1868. Reverse fine-tuning revives after a medium-rate fine-tune but not after a large one, Table 4 |
| TinyStories "Twist" deletion (Sec. 5.2, App. F) | llama2.c, 91M, 12 layers | 30K pretraining steps; deletion by filtering, filtering plus relabelling, filtering plus randomised labels | Share of Twist stories during reverse fine-tuning at 0, 30, 300, 3000 steps: 44, 81, 81, 82% after filtering at the medium rate, against 12, 31, 44, 81% for a model never shown Twist, Table 1 |

**Code.** The grammar and Tracr settings are not released. The only release is
`github.com/RobertKirk/tinystories-wrappers`, MIT licence, TinyStories only, with checkpoints. Searches of the authors'
GitHub profiles and the paper pages found nothing else; Lubana's `ConceptPercolation` is a different grammar without
task prefixes. The grammar itself is written out in full in Fig. 14, six levels of two rules each, so the generator
is small.

**Cost of rebuilding one grammar counting setting**, one low and one high prior, 200K pretraining steps:

- Code: about 500-700 lines. The grammar sampler and labellers take about 200, the training loop 150. Fine-tuning
  data with the spurious rate, the probe and pruning take about 200 together; a small GPT is reused.
- Size: $6 \times 12 \times 192^2 \approx 2.65$M parameters plus about 0.1M of embeddings, so about 2.8M.
- Tokens: the pretraining batch is not stated. Assuming 96 sequences of about 260 tokens, 200K steps is about
  $5 \times 10^9$ tokens.
- Compute: $6 \times 2.8\text{M} \times 5 \times 10^9 \approx 8.4 \times 10^{16}$ FLOPs, plus about 20% for
  attention, so about $10^{17}$ per pretraining run.
- Time: a model this small is bound by kernel launches. Assuming 15-25 ms per step on an H100, a pretraining run
  takes 50-85 min. Three of them, the two priors and the scratch baseline, take 3-4 hours. The 16 fine-tunes and as
  many reverse fine-tunes take about 2 more. The total is 5-6 H100 hours, or 2-3 with 50K pretraining steps.
  That is roughly $10-15 on RunPod. Python grammar sampling is the likely bottleneck; pre-generate it on CPU.

**Recipe gaps to settle before a reproduction:**

- pretraining batch size, context length, dropout, weight decay and vocabulary size are not given;
- the spurious-correlation strings are built by "artificially adding" `b` tokens, without saying whether the strings
  stay grammatical;
- the rule for samples shorter than the 250 tokens kept is not given, nor which matrices may be pruned;
- the learning-rate labels disagree by one decade. The main text, L308-310, puts the medium and small rates at about
  1e-4 and 1e-5. The appendix, L1191-1192, calls those large and medium and adds 1e-6. Run all three.

## Our tasks against the definition

"By construction" means fixed by how the data or weights were made; "inferred" means read off the model's behaviour.
The 67M is Goodfire's released Pile model, so we never built its pretraining data.

| Task (model) | Selector, inputs, function stated? | Possession | Relevance in the fine-tune | Verdict |
|---|---|---|---|---|
| Emoticon, `:` to `o` (67M) | Inputs are where VPD subcomponent `h.2.mlp.down_proj:2359` fires, `src/commute/data.py:76`; target constant | Inferred | Weak: no non-emoticon colons in the sample | Usable after a change: text rule for the inputs, add plain colons |
| Trigger ladder, six trigger-target pairs (67M) | Function is a constant target, `src/commute/ladder.py:118` | Inferred from baseline probability | None: a bias suffices | Cannot state a non-trivial function |
| Many cues (67M) | Constant, or a 32-entry lookup | Inferred | None | Cannot state a non-trivial function |
| Copy, marker then key gives key (67M) | Yes, 32 keys | Assumed; the original model does not do it, even in context | Strong if the capability is "repeat"; train keys equal test keys | Usable after a change: held-out keys |
| Random cipher, 32 keys (67M) | Yes | Absent by construction, but a probe reads any 32-key map at layer 0 | No prior function | Usable as the "new" partner once a held-out-value rule exists; the one-key cipher cannot |
| Lowercase Pile (67M) | Yes: context case, scored at case positions | Measured: +1.18 nats on lowercased rows, −5.58 on cased | Strong, but "always lowercase" fits equally | Best 67M candidate after a change: read cased contexts too, add a probe |
| Token cipher on Pile (67M, planned) | Yes, 50 words | Absent by construction, −5.53 nats | New table | Usable as the lowercase partner, with held-out words |
| Code-like Pile text (67M) | No | Cannot say | Learned nothing | Cannot state the function or inputs |
| Nulls: continued pretraining, Gaussian and sign-flip updates (67M) | No capability | n/a | n/a | Nulls, not anchors |
| Per-key hand write into one matrix (67M) | As copy and cipher | n/a: the update is planted, a new mechanism by construction | Not trained | Usable as "same behaviour, new mechanism" control |
| GSM8K gold fine-tune (Qwen) | No: many capabilities | Inferred, 0.756 strict at the start | Cannot say | Cannot state the function or inputs |
| GSM8K on own correct samples (Qwen) | No | Behaviour by construction, function not | Cannot say | Cannot state the function |
| Self-distillation null (Qwen) | No | Behaviour by construction | Stopped at step 2 | Cannot state the function; a null |
| Context-distillation anchor (Qwen) | — | — | Never built: no prompt qualified | Does not exist |
| Invented facts (Qwen) | Yes: relation template, subject, attribute | Content absent by construction; known-fact recall 0.97 | Known facts not relevant to invented answers | Usable after a change, as new content; mechanism needs a probe on known facts |

Plainly:

- **Tasks that cannot state a non-trivial function or its inputs:** the ladder, many cues, code text and the three
  GSM8K arms. The emoticon task states its inputs only through VPD's own choice of positions.
- **These carry most of the commute line's findings.** The ladder holds the task-ladder runs, the second ladder seed,
  the adjoint split of plain-SGD fine-tunes and teacher training: TRIED.md rows E5, E26, E31 and E34. The emoticon
  task holds the runs E1 to E4 and E14 to E16. Under this definition those findings say nothing about reuse.
- **No task has possession controlled by construction in pretraining.** Absence is known by construction for the
  random ciphers and invented facts. One mechanism was planted: the hand write.
- **Def 1 is vacuous on small token domains.** A linear probe reads any map of 32 tokens from the first layer's
  embedding. So copy and cipher would both count as possessed. A held-out-value rule fixes it: the probe, and the fine-tune, must
  work on values never seen in training. A copy mechanism can generalise to new keys; a random table cannot.
- **The repository has no linear probe of a capability and no revival test.** The two functions named `probe` are a
  timing check and an in-context demonstration. Nothing implements reverse fine-tuning or pruning revival.

## Which causal check fits which kind of task

| Kind of task | Fitting check |
|---|---|
| Scalar function with known labels, such as grammar counting | Ablate or patch the probe's direction at the answer position: in a model that reuses counting `a`, ablating it breaks counting `b` on correlated data; patching in a donor's count shifts the answer by the donor's count. Pruning and reverse fine-tuning against a scratch baseline, as Jain did, plus a random-neuron pruning baseline |
| Grammar indexing | The same, as a second instance; positional answers make probes weaker |
| Compiled Tracr models | No probe needed: the count lives in known dimensions; zero or patch them. A positive control, not evidence about learned mechanisms |
| A binary text feature, such as TinyStories Twist or our lowercase context | Reverse fine-tuning against a model never shown the feature; probe-direction ablation is possible but noisy. For lowercase, the cheapest separating read: a fine-tune that reuses context-following keeps capitals in cased contexts, a bias does not |
| Key-value maps, copy against cipher | Held-out keys: reuse of a copy mechanism generalises to unseen keys, a learned table does not. Ablating "the identity direction" is the embedding itself, so unusable |
| Content added on a known template, invented facts | Probe the relation-attribute direction on known facts in the original model; test whether invented facts are read out through it; patch |
| Constant-target tasks and GSM8K | None tractable until the function is stated |

## Does the objection apply equally to ours?

The objection to our earlier definition was that "the same component" was whatever VPD returned, so the instrument
was the definition and could not fail. Jain's version partly escapes it. The function and its inputs are fixed from
the text before any model is seen, and a probe can fail against those labels. Two loopholes remain. A trained probe
is too strong on small domains and decodes anything, so possession needs a selectivity guard such as held-out values
or control tasks. And if the causal check ablates the probe's own direction, the probe is again both definition and
instrument. Three guards close them: a held-out-value rule, controls for every probe, and a reuse verdict that also needs
a check not using the probe, such as held-out generalisation or revival. With them the objection does not carry
over; without them it does, in a weaker form.

## What this implies for Stage 1

These are implications, not a plan; the plan goes through `/design`.

- A reproduction of Jain's grammar counting setting is the only anchor with possession set in pretraining by
  construction. It brings published numbers to compare against and is cheap enough to run.
- On the 67M, the best existing pair is lowercase Pile against the 50-word token cipher on the same rows. It needs
  cased contexts read and a probe of context case.
- Copy against the random cipher becomes a real existing-against-new pair only with held-out keys.
- The per-key hand write is new by construction yet behaves like the copy fine-tune. Any instrument should call it
  new.
- Qwen has no reuse anchor at all yet; known-fact recall with a probe on the subject token is the nearest candidate.

## Side findings for other leads

- TRIED.md's E53 row says "Not run", but `experiments/performed_anchor/results/a0.json` holds a finished A0 with
  decision GO.
- E40, the Qwen GSM8K fine-tune, has no row of its own in `experiments/TRIED.md`; it appears only inside E44's row.
- E45's own-samples arm has a training file `T_own_s0.json` but no evaluation file in `experiments/qwen_updates/results/`.

## Not run

- No probe, ablation, pruning or reverse fine-tune on any model. Every possession verdict above comes from code and
  saved results.
- No check that the paper's figures match its text: figure values were not read, only captions and tables.
- The grammar reproduction's cost is an estimate from parameter counts and an assumed step time, not a timing.
- No contact with the authors about unreleased code; only public GitHub profiles and paper pages were searched.
- The Tracr and TinyStories settings were not costed.
