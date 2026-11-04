# Plan: does any other saved update's gain build gradually, and does the clean table still work on it? (E86)

## The question, in plain words
On Qwen3-1.7B's first fine-tune, a tenth of the update already keeps 93% of the gain, so "behaviour kept" cannot grade anything there: a placement could lose most of the update and still read as keeping the behaviour. We read two other saved updates, the TOFU fine-tune at step 1250 and the gold fine-tune at step 100, to see whether one of them gains gradually, which would give that column something to grade. On each we also read the clean table, to see whether it still passes good splits and fails wrong changes. D1 is a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement.

In the Terms' words: W0 + t δW with no masks at t = 0.1, 0.25, 0.5, 1, on one gain per update; K = KL(updated model ‖ state) as a share of K at W0, for the least-squares split's in-span part and remainder, the adjoint split, the often-on and rarely-on halves, a random change and a sign-flipped change at δW's size per matrix; and the plain share of δW's squared size left in Δ, least squares and adjoint.

## Design check (`/design`)
- **Goal:** the weak version in `GOALS.md` at scale; it tells which update D1's test can grade on.
- **Nearest tried:** E80 Amendment 2 (fractions on the first fine-tune only, saturated); E75 (the clean table on five updates, gold among them, none of TOFU). New: fractions on any other update, the table on TOFU.
- **Patch or step:** a step; a second kind of real update through the same table.
- **Decisions:** relies on the clean table as D1's table (`decisions.md`, ask 24). No departure.
- **What changes next:** the update that "builds" is the one teacher training is aimed at; if both saturate, "behaviour kept" leaves the table and D1 is judged by K, left in Δ and size.
- **Class:** approved class 1, the existing harness on saved states, $0, under the gate.
- **Saved:** per-read JSON only; no new weights.

## States and what differs from the references
| update | file | dtype | gain, one per update | table rows |
|---|---|---|---|---|
| first fine-tune (u1), control | `qwen_sft/nowarmup_regen/dW/step_10` | bf16 | held-out GSM8K response tokens, 256 problems | 128 maths rows of 256 |
| gold fine-tune (u2), E61 step 100 | `qwen_recipe/train/gold_1e-5/step_100` | **bf16; no fp32 copy exists** | same rows | same rows |
| TOFU at 2e-5, E81 step 1250 | `qwen_tofu/dW/a2_2e-5/step_1250` | fp32 | response tokens of the 800 rows, trained wording | 128 rows of 256, evenly from the 213 those 800 pack into |

E81 trained on 4000 rows; the 800 are the ones with paraphrases, each checked by the job to be among the 4000. Choosing TOFU's own rows for the table follows E75, where every update read its own task's rows. Σ, for the adjoint's scale, stays the maths rows' on every update, as in E75, so it is off-task on TOFU.

**Against E80 Amendment 2's fractions:** card L40 here, H200 NVL there; no components on the card here; same batch 8, same fp32 forward, same hooks, same bf16 file, with t multiplied in bf16 as there. **Against E75's table:** same card model, batch 1 per row, same reader. One difference: only the X and c of stage S are computed, by the same calls, because the rest of stage S takes most of its 23 min and nothing here reads it. Gap 0 on u1 and gold is what checks this. Random is E75 Amendment 1's Gaussian at δW's norm, not the size-searched one, which sat 18 to 100 times the update's size. E75 already holds gold's table, so re-reading it is a second control and adds only left in Δ and the fractions.

## Control, first and alone (u1); any miss stops the job
| check | must read | band |
|---|---|---|
| kept at t = 0.1, 0.25, 1 against 0.9332, 1.0010, 1.0000 | same | 10 floors over the gain. Floor: the largest of W0 read twice, FT read twice, W0's and FT's gaps to the H200 reads |
| K of W0, FT, each part, random, signs against E75's stored reads | gap 0 | 10 floors. Floor: K of W0 read twice; at floor 0 this is gap 0 to 9 places |
| left in Δ | least squares 0.0900, adjoint under 1e-6 | 1% (E80 Amendment 2's band); E80's adjoint used a c within 0.95 to 1.11 of this one |
| t = 0.1 with t multiplied in fp32 | logged, no rule | the double rounding gold shares |

Others: gold's W0 and FT within 1e-3 of E61's 2.4952, 0.5096, and its table against E75 as above. TOFU's first training batch at W0 within 1e-3 of E81's 4.7826, its file's norm within 1e-6 of the record 17.9984. Every update: t = 1 keeps 1 and W0 read again keeps 0 within 1e-3, K of the updated model under 1e-6.

## What each number should read
| number | W0 | do-nothing | updated model | u1 (control) | gold | TOFU |
|---|---|---|---|---|---|---|
| K over the ruler | 1 | 0, by construction, not read | 0 | E75's: in-span 0.0056, often-on 0.018, rarely-on 0.054, adjoint 0.10, remainder 0.32, random 1.00, signs 1.52 | E75's: 0.0066, 0.021, 0.060, 0.135, 0.38, 1.00, 1.56 | in-span under 0.02, random about 1, less sure: δW is 0.014 of W0 |
| kept share | 0 | 1 | 1 | 0.93 at 0.1 | unknown | lean under 0.9, weak |
| left in Δ, least squares | none | 1, by construction | none | 0.090 | near 0.09, no rule | near 0.09, no rule |

**Rules, filed in `experiments/qwen_d1_fractions/results/predictions.json`.** Kept at t = 0.1 at or above 0.9 is "saturated", else "builds", with the older file's "graded" at or below 0.5 and "partly" beside it. Gain under 0.05 nats, an unread loss, or t = 1 off by over 1e-3 is "no call"; above 1 still counts as at or above 0.9; a dip along t is flagged and changes nothing. The table "works" when in-span is under 0.02 and both wrong changes sit above the remainder by more than the row bootstrap's half-width. These ran on edge cases before filing (zero, negative and tiny gain; kept above 1; exactly 0.9; a dip; unread reads; zero ruler; empty record): all label or say "no call".

**Blind spots.** The kept share reads one direction, one gain; on TOFU it reads memory of trained rows, not generalisation. t = 1 repeats FT's read, so its teeth come from u1 against E80, not from itself. K is not a faithfulness check: no masks, no Γ. Left in Δ for the adjoint reads near 0 whatever it loses, and K reads that loss instead. Positive controls: u1 throughout. Noise floors: measured in each job, as above.

## Time, memory, what is saved
| stage | minutes | source |
|---|---|---|
| model, files, rows, checks | 1.5 | E75 log: load 41 s, each file about 4 s |
| 7 loss reads, 8 on u1 | 2 | E61's L40 read 16 s; TOFU has 21% more tokens |
| W0 twice, FT, random, signs | 6 | E75 log: 72 s a read |
| Σ, then X | 6 | E75 log: 173 s. X: E80 42-58 s for least squares on an H200; the halves unmeasured, about the same each |
| five parts, left in Δ | 10.5 | E75 log: 110-123 s a read with a split; E80: 29 s |

About 26 min per update, so a job can pass `--wall-min 26` and exit 0: resubmit if `results/<update>/left_in_delta.json` is absent; it resumes, with X kept on `/ephemeral` (about 17 GB, removed at the end). `--time 30 min`. GPU: E75's 45.6 GB included Γ (17.1 GB), which is not loaded here; the job refuses a card with under 45.6 GB free, the one held outside Slurm, before any model load. Host: `--mem=72G`, E75's 61.5 GB through stage S and a margin; E75's later 92 GB came with Γ reloaded and is not reached here, which the first job's peak will confirm. E81's 42.1 GB is training with no components, and is not this job's. Saved: `results/<update>/<read>.json`, `results/score_<update>.json`, `results/run.log`, under 1 MB; nothing in `results/large`.

## Run
From the clone root, $0, QoS `debug`:
```
sbatch cluster/qwen_d1_fractions.sbatch --update u1     # first and alone; the others refuse until it has finished
sbatch cluster/qwen_d1_fractions.sbatch --update tofu
sbatch cluster/qwen_d1_fractions.sbatch --update u2     # the gold fine-tune
```
The last two may run together. Exit 1 with `results/stopped.json`: tell the lead, do not resubmit. Exit 1 without it: the card was short, resubmit.

## Decision tree
- One update "builds" and its table "works": it is the update teacher training is aimed at; same recipe, same job size.
- Both "saturated": "behaviour kept" leaves D1's table; judged by K, left in Δ and size.
- A table "does not" on TOFU: report which state; the table stands on the smaller updates, and it goes to the lead before TOFU is used.
- A control's miss: stop, numbers to the lead.
- Not run: recall or accuracy as a second gain, fractions of any split's change, more seeds of the wrong changes.
