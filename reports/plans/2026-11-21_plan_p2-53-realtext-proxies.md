# P2.53: do the surviving proxies order the real-text pairs?

*Compare-lead, 2026-11-21. Cell P2.53 (1 run, due 2026-11-23). Code `experiments/realtext_proxies/run.py`, filed
settings and rules `experiments/realtext_proxies/results/predictions.json`, card
`~/commute-data/scratch/compare-lead/p253/card.md` (passes `fleet/card check`). E-number not yet claimed.*

**Restatement.** We read three cheap proxies (the log-odds re-gating share r, P1.45's S, and the original model's prior)
on the saved fine-tunes of the four real-text pairs. Nothing is trained. The do-nothing control runs first and alone,
and a random change at the fine-tune's per-matrix norms is read before each fine-tune. The output is one table saying
whether any proxy puts the side built as reuse above the other on every pair. This fills cell P2.53. P2.51
(`reports/analyses/2026-11-21_analysis_p2-51-proxies-vs-r.md`) found that only the prior tracked size-removed R on the
built items.

## Design check
- **Goal.** Part 2 of the paper plan: whether a cheap reading tracks the definition beyond built items. P2.51 left that open on real text.
- **Nearest tried.** E211 (P2.38 run 2) and E233 (P2.48) read r and W0's log-odds prior on these same states. This run adds S, E133's form of the prior, and random controls on the real-text rows. r is re-read here as the control that the rows are right.
- **Patch or step.** A step: it is the cell's own reading. It adds no new method.
- **Decisions.** None departed from. Label tasks: r is read in E228's log-odds unit, not the two-label contrast that decisions.md 2026-11-18 scores label pairs on. The contrast values are in `run2_score.json`.
- **Definition.** No verdict of reused or new is given. Calls are made against the built labels only.
- **What would change next.** If S hits on every readable pair and passes its random control, it joins the prior as a proxy that carries to real text. If no proxy hits on every pair, part 2's proxy claim stays on built items (card `stop`).
- **Control first.** The do-nothing on each pair's first rows: r's cells exactly 0, S's KL(F‖M) ≤ 1e-6 and K01 − f ≤ 1e-6, a stop on a miss. A random change then runs before each fine-tune: a gain of 1 or more stops the pod. Each fine-tune's r must also reproduce its saved value within 0.01, a stop on a miss.
- **Numbers.** Per state: r, S (and S_c beside it) and size, plus each random change's r, gain and S. Per task: the prior. The expected values are in `predictions.json` → `expected`. At the do-nothing, r's numerator and gain read exactly 0 and S's KL is 0. The random change should read r nan, with |gain| < 1.
- **Saved.** Per-pod JSONs (`results/pod<I>.json`) holding every reading, the halves of S and the random seeds. No states are written: each random change is rebuilt from its seed.

## The pairs, their saved states and the reference order
Each pair lists the side built as needing what the original model holds first. All 24 states exist, 3 seeds a side,
at the scored (larger) size, bf16, 56.6 MB each:
- label: L1 topic vs L3 coin, size 0.01, `experiments/realistic_tasks/results/large/run2/{L1,L3}_s{0,1,2}_0.01.pt`
- lowercase: T1 vs T4 reversed order, 0.0175, `.../run2/{T1,T4}_s{0,1,2}_0.0175.pt`
- language: T2 French vs T3 Yoruba, 0.0175, `.../run2/{T2,T3}_s{0,1,2}_0.0175.pt`. French gains under 1, so r cannot be read on it.
- British vs scrambled twin, 0.0175, `experiments/british_style/results/large/scored/states/{real,twin}_s{0,1,2}_0.0175.pt`.

The smaller saved sizes (0.005 and 0.00875) are not read. Two further families are out of scope: E140's shuffled-label
twins, which are built rules rather than real text, and P2.41's RL arms, which are RL against a shuffled reward and not
a text twin.

**Reference order.** Within each pair, the reference is the built label. Run 1 (`experiments/realistic_tasks/results/stage1.json`)
set the label pair by a linear probe and the text pairs by the model's loss, and E233 set its pair by the prior itself,
so the prior's British call is circular. Definition-lead has read no R on these states (grep: no reader outside
`realistic_tasks`, `british_style`, `rl_67m`). So no reference orders the pairs against each other, and cross-pair gaps
are descriptive.

## S: question for the lead
P1.45's code (`experiments/gating_vpd/run.py`, E241) can be run and is imported here (`cells`, `score`, `halves`). It
has no results of its own yet: no `items/`, no `stage0_*.json` and no `summary.json` on origin/main. Until it passes its
built edits and its random control, an S read on real text cannot be told apart from noise. There are three options:
(a) run now, with S labelled "not yet validated"; (b) hold the pods until E241's summary lands; (c) drop S. With (c),
r's real-text values only reproduce E211 and E233, so the new readings would be the random controls and the
geometric-mean prior.

## Pods, inputs, cost
Two H100 SXM pods, one per line of `settings.pods`:
- pod 0 reads label + lowercase, 12 states.
- pod 1 reads language + British, 12 states.

Gitignored inputs, each staged by hard link into a flat directory per pod:

| file | size | pod |
|---|---|---|
| the 24 state files above | 56.6 MB each | 12 per pod |
| `experiments/realistic_tasks/results/large/run3/texts.pt` | 21 MB | both |
| `experiments/british_style/results/large/data.pt` | 21 MB | 1 |
| `experiments/transport_test/results/large/alive.pt` | 0.36 MB | both |

Tracked inputs come by `sync()`: run2/E233 JSONs, `predictions*.json`, E228, E241, E211 and E233 code. The released
decomposition comes by W&B on the pod. Pod 0 also downloads AG News and SST-2 from HF (E211's label loader, pinned
revisions).

Transfer, at 10 MB/s:

| pod | upload | time |
|---|---|---|
| 0 | 679 + 21 + 0.4 = 700 MB | 70 s |
| 1 | 679 + 42 + 0.4 = 721 MB | 72 s |

Downloads are a few hundred KB of JSON.

Time is estimated from one measured item: E233 read share plus k_half on 500 × 127 rows in 24 s per state on an L40
(`british_style/results/scored/states/*.json` `read_seconds`). Here each state reads r and S on the fine-tune and on its
random change, so the estimate is ≤ 2 × 24 = 48 s. That gives ≤ 10 min of reading for 12 states, plus about 10 min of
set-up (venv, the 65 s W&B fetch, sync), so about 25 min a pod.

Cost:
- rehearsal: about 15 min, $1.10
- pods: 2 × ~0.45 h × $4.40 ≈ $4.00
- total ≈ $5. Not gated.

**Rehearsal** (card): on a pod `commute-E###-rehearse-p253`, run `run.py pod 0 --rehearse` then `pod 1 --rehearse`.
That reads seed 0 of each pair's first side at full read sizes. Each line is then run again, which must skip what is
written. The inputs are 4 states plus texts, data and alive, 270 MB, 27 s. Record the result with `fleet/rehearsed`.
The rehearsal pod may then continue as pod 0 with the scored line, which resumes from what is written.

    PRED_SHA=<sha of HEAD:experiments/realtext_proxies/results/predictions.json> python experiments/realtext_proxies/run.py pod 0 2>&1 | tee -a experiments/realtext_proxies/results/pod0.log
    python experiments/realtext_proxies/run.py score     # CPU: results/table.json

Download `results/pod<I>.json` and its log after each pair's `PAIR ... DONE` line.

## Not run
- The smaller matched sizes.
- The label pair in its two-label contrast unit for r.
- S read over every position.
- KL-matched random changes (the cell asks for per-matrix norms).
- More than one random draw per fine-tune.
- R on the real-text pairs (definition-lead's).
- E140's twins and P2.41's RL arms.
