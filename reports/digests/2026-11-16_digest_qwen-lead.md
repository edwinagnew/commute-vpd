# qwen-lead digest, 2026-11-16

Part 3 has no open cells left, so qwen-lead is finishing. P3.10, the reference for the re-gating share on the
frozen-neuron decomposition, is filled. Its filed rule made no call, and its premise check came back doubtful, which
bears on what the 11-18 gate specifies for the readings carried onto Qwen.

## Cells

| cell | status | runs | reading |
|---|---|---|---|
| P3.1, P3.2 | filled | 1/1 each | share on the new decomposition for the saved updates |
| P3.3 | filled | 1/2 | placement and RL noise floor |
| P3.5 | filled | 3/3 | pair library |
| P3.8 | filled | 2/2 | share on the run-3 pairs and gold twin |
| P3.9 | filled | 2/2 | frozen-neuron decomposition p-015e0710 |
| P3.10 | filled today | 1/1 | see below |

Rung L3.2 waits on L2.3 (to be specified at the 11-18 gate).

## P3.10 (E181)

Page: `reports/analyses/2026-11-13_analysis_e181-qwen-share-reference.md`.

- **Controls:** all held.
- **Filed rule, no call:** only seven of the fifteen changes built to switch neurons off could be scored, and the
  rule needs nine. Of the seven, five read above the random band and two inside it.
- **Fine-tunes:** 25 of 36 read above the band.
- **Why so few scored:** a few very large neurons on every row set kept most built changes off their size target.
- **Premise check, doubtful** (`~/commute-data/premise/P3.10.md`): fine-tunes may simply be more structured than
  random changes, so being above the random curve may not single out re-gating. This is a question for the gate.

## Money

E181 cost about $26.6 on two H100 pods, against the cell's $20 (the plan said about $12). The brief undercounted the
random draws per pod.

Today's summary pod cost $0.05.

## Asks

None open.

## Friction

- **Compute node:** it was down from 11-13 16:20. The lead waited for it instead of moving to a pod, which lost the
  weekend; it moved once the user said to.
- **Stale task paragraph:** the task file still carries P3.8's paragraph, which was taken for new work. The classifier
  refused the duplicate `cell add`.
- **Pod labels:** `start()` rejects spaces and colons, so the "E### stage: why" form has to be written with
  underscores.
- **`sync()` and lint:** `sync()` skips `.claude/`, so `check.py` cannot run on a pod. Pyright does not install on
  pods.

## Not run

- `check.py` on the P3.10 page.
- Pyright on `run.py` and `plot.py`.

Both wait on the compute node.
