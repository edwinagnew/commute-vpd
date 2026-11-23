# P2.69: Wang et al.'s overlap reading, run with their pinned code on our items (method spec, revision 1)

*definition-lead, 2026-11-23. Cell P2.69 (open, 0 of 3 runs used, due 2026-11-25). Step 1 of the cell: the spec only.
No code was written and nothing was run. Revision 1 answers redteam-lead's and the GPT plan-reviewer's reviews and the
coordinator's ruling on the x2 bar (section "Review record"). Every step is either the published one, quoted in the
appendix (`2026-11-23_plan_p2-69-eap-ig-overlap_appendix.md`, section A), or a named departure below.*

**Plain words.** Wang et al. 2025 compare the circuit a model uses for a task before fine-tuning with the circuit it
uses after. They find the circuits with Hanna et al. 2024's edge attribution method with integrated gradients. They count
which attention heads, MLPs and connections stay, appear or vanish. We run that count with their pinned code on our
known-answer edits, the five key tasks and E62's copy and lookup pair. This is an adaptation of their section 4.2
endpoint comparison to our items, not a reproduction of their results.

## Sources (all read; clones in `artifacts/p269_src/`, gitignored, read only)

| Source | Identity | Pinned at |
|---|---|---|
| Wang et al. 2025 | Xu Wang, Yan Hu, Wenyu Du, Reynold Cheng, Benyou Wang, Difan Zou, "Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis", ICML 2025, arXiv 2502.11812v2 | paper v2, 2025-06-13 |
| Wang's code | github.com/Xu0615/FinetuneCircuits, its vendored `eap/` tree and its circuit notebook | `fdaeeaac63d5` (2025-05-30) |
| Hanna et al. 2024 | Michael Hanna, Sandro Pezzelle, Yonatan Belinkov, "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms", COLM 2024, arXiv 2403.17806v2 | paper v2 |
| EAP-IG library | github.com/hannamw/EAP-IG, package `eap` | HEAD `e24bafd3d22a`; Wang's tree derives from `aa8cdc8` (2024-07-22) |
| Hanna's paper code | github.com/hannamw/eap-ig-faithfulness | `d42fea6aa2c7` (2024-07-11) |

**What runs.** Wang's vendored `eap/` at `fdaeeaac63d5`, with the KL callback and faithfulness function copied verbatim
from his notebook. Its `attribute.py` matches EAP-IG `aa8cdc8` apart from comments. Its `graph.py` only treats a missing
score as 0, and its `evaluate.py` only adds a helper. We do not pip-install `aa8cdc8` itself: it has no `eap/__init__.py`, so its
`find_packages` installs nothing, and it has no `eap/metrics.py`. The tree is a namespace package, imported from a staged
copy as Wang's notebook imports it.

## Claim, null, controls

**Claim tested.** On the built pairs, the edge set of the original model's circuit is unchanged by edits that strengthen
an existing part, and changed by edits that add a new lookup.

**Null.** The overlap does not tell the built new edits from the built strengthening.

**Control 1 (cell, verbatim: "a built x2 edit should show full overlap").** Filed prediction: M's and x2's circuits have
the same edge set, exactly. A miss is reported with the counts and the edges that moved. It is not "passed" under a
looser bar.

**Control 2 (redteam c3).** new_few and new_thin each read an edge IoU below the range of the M-against-M resampling
diagnostic on their task. A miss is reported the same way.

**Reading rule for both.** If either control misses, every later stage is still run and reported, but labelled as values
of a method that failed its control. Neither control stops the job. Stops are only for adapter failures (stage 0).

**Why x2 may miss.** x2 doubles 8 of MLP1's 3,072 output columns. That changes the scores of edges out of MLP1 by however
much those columns carry, and shifts gradients downstream. A small change can move an edge across the 46-edge cut with no
new wiring. One moved edge changes IoU by about 0.02-0.04, so counts go beside every IoU.

| Arm | Expected edge IoU against M | Why |
|---|---|---|
| Do-nothing (M against M, same windows) | exactly 1 | determinism; the pipeline only |
| Resampling diagnostic (section C of the appendix) | measured; Wang's Fig. 2 right reads about 0.98 to 0.94 for Pythia, 0.55 for a random model | data noise at full size, reported beside every value, not a bar |
| x2 | exactly 1 (control 1) | the cell |
| x10 | no lean | magnitude can move edges with no new wiring |
| rebuilt | no lean | it zeroes MLP1's 8 columns and replaces MLP2's and MLP3's weights, so three MLP nodes change |
| new_few, new_thin | below the resampling range (control 2) | the edit's MLP-to-output edge should enter F's circuit only |
| Matched randoms | inside the resampling range | a random update hardly changes behaviour, so it reads "reused" |
| E62 copy, lookup | no lean | E97 Amendment 1: both fine-tunes keep both of the original model's circuits |
| Key grid | no lean filed; E237's squared-cosine order (copy high, cipher low) beside | descriptive |

## What a value means, and when there is none

Faithfulness gates the headline. An item enters the headline only when both circuits keep at least 70% faithfulness by
Wang's formula. Below that the item reads "circuit not faithful", and its raw overlap goes to the record only. Filed cases:

- **No gap.** F(M), the full model's KL from its corrupted run, at or under the stage-0 fp32 floor or not finite: the
  faithfulness ratio is undefined, and the item reads "no gap".
- **Empty circuit.** After pruning the output node is not in the circuit: the item reads "empty". Two empty circuits do
  not count as equal, so x2 cannot pass that way.
- **Sizes.** Each circuit's edge and node count after pruning goes beside every IoU and every ΔS.

## Departures from the published method

- **D1, code.** Wang's vendored tree (above), not EAP-IG HEAD. HEAD integrates at k = 0..m−1 (`13c0033`, 2025-02-23);
  both papers and Wang's tree use k = 1..m.
- **D2, model.** Our 67M as a TransformerLens model from a config and a full native state per item, no weight
  processing. Wang loaded Pythia-1.4B through `from_pretrained`. Stage 0 checks it.
- **D3, tokenisation.** `tokenize_plus` in `eap.attribute` and `eap.evaluate` is replaced to return our token windows,
  without BOS, with a full mask. The released function re-tokenises strings and prepends BOS. Nothing else changes.
- **D4, read position.** Windows are cut after the read position, since the code reads the last token. Causal attention
  leaves the read unchanged.
- **D5, tasks and corruptions.** Ours, per item in appendix B. Wang's arithmetic is beyond the 67M, and the cell fixes the
  items. Each corruption swaps the one token that decides the answer, with a fixed pairing seed.
- **D6, two checkpoints.** Before and after, as Wang's Figure 3 left. ΔS has n = 1.
- **D7, headline circuit.** The 5% circuit, as Wang's section 4.2 counts. Section 4.3's 95th-percentile re-threshold is
  reported beside as a sensitivity; on 46 edges it keeps two or three.
- **D8, a code bug corrected.** Node IoU excludes input and output nodes, as Hanna section 5 intends. Her code subtracts
  "inputs" where the node is named "input". The code's value goes beside ours.
- **D9, a code bug corrected.** The hypergeometric p is the paper's chance of an overlap of at least k. The code computes
  more than k. The code's value goes beside ours.
- **D10, states.** Saved states, not Wang's LoRA fine-tunes. The controls are additions; Wang's only control is a randomly
  initialised model.
- **D11, key-grid rows.** E237's read rows hold the task's rows, rows without the behaviour and far rows. We use the task
  rows only, because a corruption needs a key to swap.
- **D12, second selector.** Hanna's greedy search, smallest circuit at normalised faithfulness of at least 0.85, with KL to
  the clean run as the metric. It is an adaptation: Hanna's tasks scored that bar with task metrics.

## What it is blind to

It sees heads and MLPs, so a change inside one MLP's neurons reads as reuse. It ranks edges by score size, so
strengthening can move edges across the cut. It reads which parts matter for the task, not what they compute, Hanna et
al.'s own warning. A random update reads as reuse because nothing changes. The circuit depends on the chosen corruption.

## Stages, in order (cluster debug, $0; each writes its JSON as it ends)

0. **Adapter, the only stops.** Run on M and on x2 before any attribution. Every native parameter maps to the converted
   model exactly once. The converted logits on clean and corrupted windows match the native forward within the
   batched-against-single floor of the native forward, measured first. Evaluating the full circuit gives the clean
   output, and the empty circuit gives the corrupted output, within the same floor. State hashes are checked against their
   records. Not stops: do-nothing IoU, the fp32 KL floor for "no gap", seconds per item.
1. Control 1: x2. Then x10. The resampling diagnostic for M and x2 on copy24.
2. Control 2: new_few, new_thin, with their resampling. Then rebuilt and the two matched randoms.
3. E62: copy and lookup, seeds 0-2, and one random at copy_s0's per-matrix norms. Head (2,4)'s place in M's copy circuit
   is reported as a result.
4. Key grid: 15 scaled states and their 15 randoms.

## Cost and time

E97 Amendment 1 (job 52785) scored and exactly patched all 927 edges on six states in about 4 minutes on a debug L40.
Here there are about 108 attributions: 44 states, M on 9 task sets, and 55 resamples. Each item also needs one native
export of about 270 MB, written to node-local scratch, converted and deleted. Scaled from E96's forward times, this is
15-30 minutes with the venv build, to be measured on one item in stage 0. One or two debug jobs, $0. If debug does not
start within 5 minutes, an H100 pod: about 1.5 GB of states upload in 150 s at 10 MB/s, well under the $20 cap.

## Review record

| Point | Done |
|---|---|
| RT (a): report Wang's 95th-percentile re-threshold beside | D7, as a sensitivity |
| RT (a), GPT P2-7: code values beside the D8, D9 corrections, labelled code bugs | D8, D9 |
| RT (b), GPT P1-1: x2 bar | Coordinator's ruling: exact equality filed, a miss reported with counts and moved edges, never passed under a looser bar. The half-split stop bar is dropped. The overstated "rescales every outgoing score" is corrected |
| RT c1: halves bias the floor | Replaced by full-size bootstrap resamples, 5 per task, M against M and F against F for x2 and new_thin, as a diagnostic beside the values |
| RT c2, GPT P1-5: faithfulness gates; empty and degenerate cases | Section "What a value means" |
| RT c3: a control for "new" | Control 2, same reading rule as control 1 (coordinator), not a stop |
| RT c4: sizes after pruning | Beside every IoU and ΔS |
| RT label, GPT P2-7: claim no more than what runs | Title and plain words: an adaptation of section 4.2's endpoint |
| GPT P1-2: E62 windows and corruptions; (2,4) a result, not a stop | Checked against `experiments/locator_table/pairs.py` and `src/commute/copying.py`: windows of 41 tokens, read at 20, prefix 0-20; copy B to B′, lookup the second A swapped. My first draft's "8 tokens, marker last" was E242's generator, not the saved rows. Appendix B |
| GPT P1-3: E242 random_kl reads new_few's cue windows; E237's read rows mixed | Checked in `kstar_vpd_wording_a/run.py` `random_kl` and `old_tasks.task_windows`: both true. Appendix B corrected; D11 |
| GPT P1-4: `aa8cdc8` not installable; incomplete weights; Python pin; adapter checks | Checked: no `__init__.py` or `metrics.py` at `aa8cdc8`; `checkpoint_weights` returns the decomposed matrices only. Wang's tree, a full native state per item, Python 3.10, stage 0 |
| GPT P1-5: secondary selector's metric | D12, named an adaptation with a KL reference |
| GPT P2-6: rebuilt reads a 37 MB payload and changes MLP2, MLP3 | Checked in `rho_saved_items/run.py` `rebuilt`: true, and it takes a config. Payload registered and hashed (appendix B); expected reading changed to no lean |

## Not run

Nothing. Not checked: TransformerLens 2.0.0's support for this config, which stage 0 checks; window counts per task.
