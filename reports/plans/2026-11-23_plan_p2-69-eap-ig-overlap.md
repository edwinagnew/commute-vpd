# P2.69: EAP-IG circuit overlap before and after fine-tuning, as a published baseline (method spec)

*definition-lead, 2026-11-23. Cell P2.69 (open, 0 of 3 runs used, due 2026-11-25). Step 1 of the cell: the spec only.
No code was written and nothing was run. Every step below is either the published one, quoted in the appendix
(`2026-11-23_plan_p2-69-eap-ig-overlap_appendix.md`, section A), or a named departure in the list below. Redteam-lead
and an independent GPT reviewer check this file before any run.*

**Plain words.** Wang et al. 2025 compare the circuit a model uses for a task before fine-tuning with the circuit it
uses after. The circuits come from Hanna et al. 2024's edge attribution method with integrated gradients. They count
which attention heads, MLPs and connections between them stay, appear or vanish. We will run their method with the
released code on our known-answer edits, the five key tasks and E62's copy and lookup pair. A circuit that stays the same
should mean "reused" and one that changes should mean "new". The run asks whether that holds where the answer is known.

## Sources (all read; clones in `artifacts/p269_src/`, gitignored, read only)

| Source | Identity | Pinned at |
|---|---|---|
| Wang et al. 2025 | Xu Wang, Yan Hu, Wenyu Du, Reynold Cheng, Benyou Wang, Difan Zou, "Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis", ICML 2025, arXiv 2502.11812v2 | paper v2, 2025-06-13 |
| Wang's code | github.com/Xu0615/FinetuneCircuits | `fdaeeaac63d5` (2025-05-30) |
| Hanna et al. 2024 | Michael Hanna, Sandro Pezzelle, Yonatan Belinkov, "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms", COLM 2024, arXiv 2403.17806v2 | paper v2 |
| EAP-IG library | github.com/hannamw/EAP-IG, package `eap` | HEAD `e24bafd3d22a` (2026-05-23); we run `aa8cdc8` (2024-07-22) |
| Hanna's paper code | github.com/hannamw/eap-ig-faithfulness | `d42fea6aa2c7` (2024-07-11), submodule EAP-IG `0edbdd7` |

Wang's repository vendors `eap/`. Its `attribute.py` is EAP-IG commit `aa8cdc8` with comments added. Its `graph.py`
differs only in treating a missing score as 0, and its `evaluate.py` only adds a helper. So `aa8cdc8` is the code that
produced Wang's numbers, and it is the version we run (departure D1 explains why not HEAD).

## Claim, null, expected readings

**Claim tested.** At the published settings, the overlap between the 67M's circuit and the fine-tuned model's circuit
is high for edits that strengthen an existing part (x2, x10, rebuilt) and low for edits that add a new lookup
(new_few, new_thin), with matched random updates read beside.

**Null.** The overlap does not separate built reuse from built new beyond the floor set by resampling the task windows.
If it fails on the built pairs, its readings on trained items are reported as values of a method that failed its known
answer.

| Arm | Edge IoU, expected | Why |
|---|---|---|
| Do-nothing (M against M, same windows) | exactly 1, no edge added or removed | the code is deterministic; tests the pipeline only |
| Floor (M on two random halves of the windows, 5 splits per task) | measured in stage 0; Wang's edge Jaccard under data perturbation reads about 0.98 to 0.94 for Pythia, 0.55 for a random model (Fig. 2 right, read off the plot) | data noise alone |
| x2 (eight copy neurons' output doubled) | at or above the floor's minimum; node IoU near 1 | positive control; see the note below |
| x10 | below x2 is possible | magnitude, not new wiring, can move edges into the top 5% |
| rebuilt | near the floor | the copy neurons are replaced inside the same MLP, which the head and MLP grain cannot see |
| new_few, new_thin | below x2 and below the floor | the edit's MLP-to-output edge should enter F's circuit and not M's |
| Matched randoms (size, KL) | near the floor | a random update hardly changes behaviour, so it reads "reused" |
| E62 copy against lookup | no lean filed | E97 Amendment 1 found both fine-tunes keep both of the original model's circuits |
| Key grid | copy highest, cipher lowest if it tracks the squared-cosine panels | E237's order; randoms near the floor |

**Note on the cell's "full overlap" for x2.** Exactly 1.0 is not guaranteed by construction. Doubling one MLP's output
scales the scores of every edge out of that MLP, so the top 5% can shift without any new wiring. The filed control bar
is therefore "edge IoU at or above the stage-0 floor's minimum", set from stage 0 in the same job. Stage 1 stops below it.
Redteam-lead should confirm this reading of the cell.

## Design check (`.claude/skills/design/SKILL.md`)

- **Goal.** Part 2 of the paper plan, claim 2: the cheap proxies track the definition and beat the baselines. This is a
  published baseline, scored on the same items as the proxies, run as decisions.md's 2026-11-17 row requires: "the
  strongest published recipe with the authors' code at a pinned version".
- **Nearest tried.** E97 Amendment 1 ran EAP-IG in our own code (`src/commute/edges.py`) with cross-faithfulness, not
  overlap. The top edge was block 3's MLP to the output on copy and lookup alike. E198 reimplemented a crosscoder and
  failed its control. New here: the published overlap reading, the published code, the built pairs and the key grid.
- **Patch or step.** A step: a row of the baseline table. **Decisions.** Follows the 2026-11-17 row. No departure.
- **Definition.** No verdict under the working definition; overlap is a baseline scored against the built labels. No
  rule is added to the definition.
- **What changes next.** Passing x2 and separating new from reuse makes EAP-IG a baseline the proxies must beat on the
  trained items. Failing x2 gives the paper a published baseline that fails a known answer.
- **Control first.** Stage 0, then x2 alone, each with a stop.
- **Saved.** Per state: the full 927-edge score vector, both circuits and both faithfulness numbers (about 10 KB per
  state, as JSON in `results/`).

## Departures from the published method

- **D1, code version.** `aa8cdc8`, not HEAD. HEAD sums its gradients at k/m for k = 0 to m−1 (changed in `13c0033`, 2025-02-23).
  Both papers' formula and Wang's code use k = 1 to m.
- **D2, model.** Our 67M is built as a TransformerLens `HookedTransformer` from a config and our weights, with no weight
  processing. Wang loaded Pythia-1.4B through `from_pretrained`. Stage 0 checks that the logits match our forward.
- **D3, tokenisation.** The package re-tokenises strings and prepends a BOS token. Our items are token windows read
  without BOS. We replace `tokenize_plus` in `eap.attribute` and `eap.evaluate` by a function that returns our tokens with
  a full mask. Nothing else in the package changes.
- **D4, read position.** The package reads the last token. Windows are cut after the read position, which leaves the read
  unchanged because attention is causal.
- **D5, tasks and corrupted inputs.** Wang's arithmetic tasks are out of reach for the 67M, and the cell fixes the items.
  The corruption follows both papers' rule: swap the one token that decides the answer for another of the same pool, so
  the input "fits the same task schema but elicits distinct output" (Hanna, section 2). Per item in appendix B.
- **D6, two checkpoints.** Before and after only, as in Wang's Figure 3, left. Their change rate then has n = 1.
- **D7, no second threshold.** Wang's section 4.3 keeps only edges above the 95th percentile of the circuit's scores. On
  46 edges that leaves two or three, so the headline uses the 5% circuit, as their change counts in section 4.2 do.
- **D8, node set for IoU.** Input and output nodes excluded, as Hanna's overlap code intends. That code subtracts the name
  "inputs" where the node is called "input", so it keeps the input node.
- **D9, significance.** The hypergeometric p as the paper states it, the chance of an overlap of at least k. The code
  computes the chance of more than k.
- **D10, fine-tuning.** Wang fine-tuned with LoRA. We read saved states (built edits, E58 and E62 full fine-tunes, E237's
  scaling). Our controls (do-nothing, floor, built edits, randoms) are additions; Wang's only control is a randomly
  initialised model.

## What it is blind to

It sees heads and MLPs, so a change inside one MLP's neurons reads as reuse, which makes the rebuilt edit invisible. It
ranks edges by score size, so strengthening a part can move edges in or out of the circuit with no new wiring. It reads
which parts matter for the task, not what they compute, Hanna et al.'s own warning. A random update reads as reuse,
because nothing changes. The circuit depends on the chosen corruption.

## Stages, in order (cluster debug, $0; each writes its JSON as it ends)

0. Alone, stops on any miss: logits of the converted M and of copy_s0 against our forward, bar the batched-against-single
   floor of our own forward measured first; do-nothing IoU exactly 1; the floor on every task's windows; head (2,4) in
   M's E62 copy circuit, as E97 Amendment 1 found; seconds per item.
1. x2, then x10, on copy24. Stops when x2's edge IoU is under the floor's minimum.
2. rebuilt, new_few, new_thin and the two matched randoms.
3. E62: copy and lookup, seeds 0-2, and one random at copy_s0's per-matrix norms.
4. Key grid: 15 scaled states and their 15 saved randoms.

Hanna's own selection (greedy search, smallest circuit with normalised faithfulness of at least 0.85) is computed beside
the primary in stages 1-4, from the same scores.

## Cost and time

E97 Amendment 1 (job 52785) scored and exactly patched all 927 edges on six states in about 4 minutes on a debug L40.
Our count is about 132 attributions: 44 states plus M on 8 tasks plus 80 half-window floors. At E96's measured forward
time on E62 windows, this is about 6 s per state, so 15-30 minutes with the venv build. This is an estimate, to be
measured on one item in stage 0. One or two debug jobs, $0. If debug does not start within 5 minutes, an H100 pod: about
1.5 GB of states at 10 MB/s is 150 s up, plus under an hour of compute, well under the $20 cap.

## Not run

Nothing. Not checked: TransformerLens 2.0.0's handling of this architecture, which stage 0 checks; the window counts
per task; the exact read windows of the randoms, taken from E242's and E237's code at build time.
