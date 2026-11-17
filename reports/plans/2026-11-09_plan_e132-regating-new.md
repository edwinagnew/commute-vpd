# E132 (cell P3.1, due 2026-11-11): the re-gating share on the new decomposition

**What runs.** One job on the cluster's `debug` queue, one L40, about 80 minutes, $0. It reads E57's re-gating share,
(K00 − K01) / (K00 − K11) over the 128 maths rows, on the new decomposition p-9707d6ff for E57's four saved updates
(E40 step 10, gold step 100, anchor 2 seeds 0 and 1 at step 100) and E57's two random changes. Code:
`experiments/qwen_regating_new/run.py`, which runs E57's `lab.py` with its paths, its alive count and its statistics
pointed at the new decomposition; `cluster/qwen_regating_new.sbatch`.

**Goal check (`/design`).** Goal: part 3 of `reports/plans/2026-11-09_plan_paper-roadmap.md`, the key findings repeated
on Qwen3-1.7B. Fills P3.1. Nearest row tried: E57, read on old only, which read 0.993 to 0.9995 for every update and
every random change ("no teeth at this scale"). New here: whether the same holds on a decomposition trained on
chat-format text with Qwen's own samples. A step towards the goal, not a patch. No `decisions.md` row is departed from.
No verdict under either wording of reuse (P1.1 is open): this reports numbers and one separation rule. What changes next:
if some update separates beyond seed spread, part 3's table has a re-gating row with a reading; if not, re-gating has no
teeth on Qwen on either decomposition and the row reads "does not separate".

**Which term.** "Re-gating share" here is E57's KL-based share, which the cell cites by its old range, not `GOALS.md`'s
log-odds r at the trigger. Numbers from E57 sit in the same file under `old`, computed by the same function.

**Controls first and alone, each a stop.** W0 as the fine-tuned model: K00, K01 and the zero split's K10, K11 equal K0
within 1e-6. E40's unmasked KL on 16 rows equals E57's 0.6184 within 1e-3 (no decomposition enters it). The do-nothing
state through the split path reads a share of exactly 1 (within 1e-6). The alive set counts 500,244 (checked on CPU
before submission: matches). Then the Gaussian and in-span random changes, each matched to E40's KL on 16 rows, alone,
before any update.

**Expected.** Do-nothing 1.000. At W0, K0 about 0.08 mean nats per position (E131's masked KL on 512-token rows was 0.0728 on new; E57's old
ratio of 256 to 512 tokens was 1.18). Random changes at least 0.8 (old 0.993 and 0.994). Updates at least 0.98 (old
0.9927 to 0.9995).

**Filed rule** (`results/predictions.json`). Seed spread is the share difference of the two anchor 2 seeds, the one
saved seed pair with an old reading. A pair of states is separated when the share difference exceeds the seed spread and
its paired row-resample interval excludes 0. Applied to old and new alike. On old, already checked on CPU: E40 and gold
separate from the anchor 2 seeds and from both random changes; E40 and gold do not separate from each other; anchor 2
seeds do not separate from the random changes.

**Blind to.** The share reads 1 whenever K01 and K11 are both small against K00; a high share does not show an update
re-gated. One seed pair is a thin spread.

**Departures from E57.** Gate-count cells (S), placement nulls, the own-row cells and overlaps are dropped: the cell reads the share
only, and without the nulls the split takes about 6 minutes instead of 16. The in-span random change's scale is searched to E40's KL as the Gaussian's
already is, because E44 Amendment 2's per-site scales were set on the old decomposition's live columns. No stored W0 gate
comparison (E57's Stage A counts are old).

**Cost and limits.** ~80 min against the 2 h debug limit; GPU ~42.6 GiB (E57, same column count 688,128); host ~93 GB;
`--mem=128G`. The cell has one run: a job that dies on a code error before reading anything may be fixed and resubmitted
under E132; anything else closes the cell with what was written.

**Not run.** The RL updates, E45 Amendment 2's anchor 1 and own-samples states (no old reading to set beside); a second
seed pair; S, placement and overlap on new.
