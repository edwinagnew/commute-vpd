# Review: E62 Amendment 6 code, before its first GPU run

Commit 539c815, by a fresh agent that wrote neither the code nor the plan. No model was loaded and nothing ran on a GPU. **I found no bug that would crash the run or read the wrong tokens.** The findings below concern what a verdict would mean, and the first two should be settled before the job, because `mask_holds_answer` closes a line of work.

## Findings

### 1. High: `mask_holds_answer` can fire on a forward that does copy

`src/commute/copying.py:1065`; `results/predictions_amend6.json:27,29,43`; plan line 919, 932, 934.

**Scenario.** The forward reads on -7.30, b_over -8.82, new_intact -11.60, new_over -9.91, lean 4.34. Then lost is 0.35, gained 0.39 and the gap 1.09, so the verdict is `mask_holds_answer`. The new token rose 1.69 log-odds, 5.4 times in odds, on being written into the text, and B fell 1.52 on leaving it. The plan's next step is then to close the 67M mask work on the user's reading, "the forward computes no copying".

**Why the bar does not protect against this.** `band_source` sets 0.4 at 2.2 sds, using the sd of a share unpaired with its denominator (0.043). The real rows are paired per window. The plan's own table gives 0.007 for the nearest paired share (line 934), and I reproduce 0.0069 from `amend5_windows.pt`. So 0.39 would sit far outside the noise (about 50 of those sds, though that row is a stand-in; the run prints its own `gained_sd`). `tests/test_copying.py:1056` pins a lost of 0.38 as `mask_holds_answer`, so the code and rule agree: the problem is that the rule's name and next step claim more than its bar tests.

**Smallest fix, before the run, as a dated amendment (Amendment 4's rules were amended this way).** Add a `no_copy_max` threshold for this verdict, set against the printed paired sds and Amendment 5's token part (0.063), for example 0.15, and read `mask_holds_answer` only when lost and gained are both under it. Between that and 0.4 reads `mixed` and goes to the user. If the band is kept instead, rename the claim in the plan and predictions to "mostly holds" and take "computes no copying" out of it.

### 2. Medium: `mask_holds_answer` has no positive control, and `mixed` cannot be told apart, when one cheap read would do both

`experiments/induction_cue/amend6.py:100-104`; plan lines 921, 942-943.

**Scenario.** The plan admits the masks at p + 1 were read for B and may pass B and not the new token. Take a real route through those masks: B falls, the new token does not rise, and the verdict is `mixed`. The plan then sends both numbers to the user and proposes a further job to separate "stale mask" from "B-only route". Separately, the plan says "No forward is known to hold an answer", so a holds verdict rests on stub tests alone.

**Smallest fix.** Add one Γ pass on the overwritten tokens and two report-only reads, with no verdict and no stop, after the verdict line:

```python
g_over = R.gates(J.W["b_over"])  # Γ's own mask with B overwritten
own = {k: R.masked(J.W[k], g_over, []) for k in ("b_over", "new_over")}  # saved in rows, printed
```

File the expected values first:

| Case | new_over under the mask read on the overwritten tokens | B under that mask |
|---|---|---|
| The mask carries whatever the text held (the positive control for `mask_holds_answer`) | near -7.3 | near -11.6 |
| `mixed` with lost high and gained low, stale mask at p + 1 | near -7.3 | near -11.6, so held-mask gain was low because the mask was stale |
| Own mask does not follow the token | near new_intact | anything; a holds verdict is then suspect and goes to the user |

It adds under a second and may save the round trip the `mixed` branch would cost. The lead may decline it under "fewer measurements", but then the report must state that the holds verdict has no positive control.

### 3. Medium: "says B" is a filed reading of the user's words, and the user has not acknowledged it

Plan line 946; `results/predictions_amend6.json:43`; `.claude/CLAUDE.md`, Terms.

**Scenario.** The user said "if the model still says B". The rule reads B about 1 log-odds over the new token, in a forward whose B level is about -7.3 on the mean, which is roughly 0.07% probability. Neither token need be near the top, and the job does not read what the masked forward's top token is. A `mask_holds_answer` read as "the mask holds the answer" can therefore mean only that B sits at about 0.07%, above a token that sits lower still. The plan's Scale bullet discloses this, but the user reads the log and chat, not plans, and the Terms rule asks for explicit acknowledgement when the user's words and a file's definition differ.

**Smallest fix.** Give the two readings in one line each in chat and get the answer before the job. The reading should also state B's level in plain probability. If it is cheap to add, also print each forward's top-1 rate for B and for the new token.

### 4. Low: the next step closes more than the verdict reads

Plan lines 877, 919; `results/predictions_amend6.json:8`.

**Scenario.** The read is W₀, the second A, one position. The design check generalises: "no masked copying read on this model can score that", and the Qwen caption says "on the 67M the mask carried the copied token". E62's masked gain on fine-tuned states is read at the marker, on other masks, which this job does not touch; the filed `granularity_limit` itself says no verdict reads a fine-tuned state. The closure is the user's instruction, so it stands, but the log line can overstate it.

**Smallest fix.** Word the limit and the caption "at W₀, at the second A, under W₀'s mask", and name the marker and the states as unread.

### 5. Low: `unreadable` exits 0

`experiments/induction_cue/amend6.py:102-104`; `cluster/induction_cue_amend6.sbatch:6`; plan line 922.

**Scenario.** A non-finite cell under the mask gives verdict `unreadable`. The job continues to the fp32 row and exits 0, while the plan treats it like the exit-1 stops and the sbatch header does not list it. A runner keying on the exit code reports a finished read. It is unlikely, since `identity.log_odds` works in float32 and the filed lean is 4.34.

**Smallest fix.** In `run`, after the re-read, `if mask["verdict"] == "unreadable": J.stop("unreadable", ...)`.

### 6. Low: the fp32 row's first cell is not checked against Amendment 5

`experiments/induction_cue/amend6.py:106`; `results/predictions_amend6.json:45`.

**Scenario.** If the fp32 path were still under autocast, this row would read bf16 numbers, `on_diff_a5` would be about 0.005 rather than 0, and `fp32_verdict` would agree with the bf16 verdict for the wrong reason. Amendment 5's plan named this failure ("else fp32 autocast was still on"). Here the number is printed with no flag, and the filing states no outcome for a miss.

**Smallest fix.** One LOUD log line when `fp32["on_diff_a5"] > J.th["reread_tol"]`; it stops nothing.

## Checked and found fine

Run through `cpu-run`, no model (`~/commute-data/scratch/regating-lead/review-e62a6/dry.py`, `comp.py`):

- **`load`'s file checks on the real files:** the windows.pt sha256 matches; `amend5_windows.pt` means equal `amend5.json` exactly under the `!=` compare; all rows `cells` needs exist (`ones_*`, `target_*`, `mask_*`, `gamma32_on`).
- **`overwrite` on the real windows:** holds, pool 178, 117 distinct new tokens, 1 window with the new token elsewhere, as filed. `W["on"]` equals Amendment 5's `w` exactly, so gates and rows align per batch and window. The pool is disjoint from the keys, no new token equals A or the marker, and 4 windows draw a B of the same A elsewhere (negligible).
- **The draw:** new tokens' levels on Amendment 5's rows sit +0.31 (target), +0.34 (every mask at 1) and +0.59 (mask) over the Bs' for the filed seed, within about 1.4 sds over 600 seeds, and mostly window noise: within-token sd is 3.1 to 3.8 of a window sd of 3.4 to 4.2. No effect on lost or gained, which difference one token, and well under the control's 2.0.
- **Rule against filing:** code and filed text agree on all branches, ties and the control's order. The filed derived numbers (-9.04, +1.74, +2.60) are right.
- **`cells`:** `lost_sd`, `gained_sd` and `gap_sd` reduce to the right shares by algebra through `share_sd`. The denominators are each forward's own, with bf16's for fp32, as filed.
- **Reads:** only the first n columns reach Γ and the forward, the last column is the target (`LAST`), `Reads5`'s fields are in order, and the empty head list is a no-op, as Amendment 5.
- **Stops:** `sys.exit` with `pipefail` and `tee` gives exit 1; results are dumped before each stop.
- **Re-read tolerance:** the only `compute` nodes are L40s, the same card as Amendment 5, so 1e-3 will not stop on card differences.
- **Sbatch:** the diff against Amendment 5's touches only names, comments and paths; `alive.pt` exists.
- **Tests and lint:** `tests/test_copying.py` 5788 passed; ruff and pyright clean on the three files.

Not checked: anything needing the model, including `absent_b`'s tokens through a masked forward, and the plan's stub run in its scratch folder.
