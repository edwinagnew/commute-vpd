# Plan: a locator that can be read across the 67M tasks, for baseline-lead's reuse table (E97)

*mechanism-detection-lead, 2026-11-05. Status: drafted; code waits on the independent check of E96 (`reports/reviews/2026-11-05_review_e96-independent-check.md`). Advice: `reports/reviews/2026-11-05_review_e97-locator-fable-advice.md` (Fable's), adopted with the overrides below. Cost: `debug` QoS, $0, under the gate, so no review or independent check unless a result reaches a headline.*

**Limit on every claim.** A new circuit built from old parts still reads as reuse. Finding a part says the behaviour runs through it, not that one old circuit is unchanged.

## The request, in the Terms' words

For each fine-tuned 67M state on a task, name the part of the model its gain runs through, or say "not found" at a stated grain, with a control on the same windows, so baseline-lead can read its frozen definition (`decisions.md` lines 81, 115, 121) against the labels the old tasks assumed. The definition does not change here.

## Design check

**Goal.** `GOALS.md`'s second question: does post-training teach new capabilities or strengthen existing ones? E90 found that only the copying pair has a located mechanism, so the other rows have no causal reading. The link to the map itself is indirect.

**Nearest tried.** E96 mean-ablated heads and failed on lowercase, where about half the best share is W₀ rising; salience and probes failed. E66 patched block inputs and neurons (context patch, not a W₀-to-state patch) and stopped at its control. E38 and E50 restored changed outputs, but only for VPD's switched-on set. New here: W₀-to-state patching at part grain with no VPD set, reverting the update one matrix or head at a time, and known-answer plants that do not rest on E62.

**Patch or step?** A step: E96 itself says a net share misleads when W₀'s target probability is near 0.001.

**Decisions relied on.** Lines 115, 116, 121: a label needs a located mechanism with a spared control; constant-answer tasks stay "does not apply", and any reading on them goes beside. No departure.

**What would change what we do next.** See the tree.

## The reads

Grain: 24 head outputs and 4 block MLP outputs. All positions. Gain G is the state's mean log-odds of the target minus W₀'s.

- **Carry (A).** The state, with part P's output replaced by W₀'s output on the same input. Share = fraction of G lost. W₀ is not moved, so W₀ rising cannot enter. Blind to whether P did anything in W₀; credits downstream parts for an upstream write.
- **Write (B).** The state, with one matrix's update reverted (24 matrices), or one head's slices of q, k, v and o. Share = fraction of G lost. All reverted at once must give G = 0. The sum of single shares is the non-additivity number. Blind to what carries the write onward.
- **Removal (the label).** E96's mean ablation in the state's own terms, on the parts A and B nominate plus the greedy pair: the state's fall as a fraction of G, on the task and on its control. W₀'s own move is printed beside and never subtracted.
- **Capability.** W₀'s reading of C with the part off, only where C has a read: copy's second-copy top-1 (exists, E96). Lowercase's capability map is deferred until the 0.25 against 1.18 gap is explained.

Sufficiency (W₀ with the state's P, Prakash et al.'s cross-model patch) is saved beside, no rule.

**From the E96 check (`reports/reviews/2026-11-05_review_e96-independent-check.md`, first version).** The removal's means are W₀'s, taken on each task's own windows, not on cased natural text (F3): a mean from another kind of text puts a foreign vector in, which may be part of E96's "W₀ rising". On lowercase both units are saved: the filed one is E96's log-odds of the target against all tokens, and the lowercase-against-capitalised form goes beside (F1). Standard errors are by row, not by window, since windows overlap (F4). The probability printed is the mean probability, not the probability at the mean log-odds (F5).

**After the check's GPU job (52604, final verdict: holds with caveat).** The 0.25 against 1.18 gap is context length: on the same positions it reads 0.35 at 64 tokens and 1.85 on whole rows. So stage 2 reads lowercase and cipher on whole rows (256-token windows if time forbids), where W₀ performs; the 64-token read goes beside. Means from lowercased text barely change W₀'s rise (0.395 against 0.39), so the task-window means are kept for consistency, not as a fix, and E96's natural-text means are saved beside (the choice moves shares by up to 0.05). The lowercase capability map no longer waits on the gap, but stays out of this job.

**Filed located rule.** On every seed: task fall ≥ a·G_task, control fall ≤ 0.2·G_control, and task fall ≥ 2 × control fall in nats. a is the weakest copy seed's own-terms fall at the sweep's top head, read in stage 1 of this job and truncated to two decimals, written to the results before stage 2 runs. E96's table suggests about 0.56. Edge cases to run in `tests/` before filing: G ≤ 0 on either state ("unreadable: no gain"), a part that raises the state (fall below 0, not located), a tie at the top of A (no part named), a share past 1 (counts as located), a seed that disagrees (not located).

## Stages, control first

**Stage 0, plants, alone; stop on a miss.** No fine-tuned state and no E62 in the answer. Two one-matrix edits fitted to the 24-key copy table's windows: `ctxpatch.write_state` in `h.3.mlp.down_proj`, and `perturb.key_write` in `h.1.mlp.down_proj`, each scaled to about the state's gain.

| Number | Expected | Source of the bar |
|---|---|---|
| B at the edited matrix | 1.0 ± 1e-4 | exact by construction; E96's `reread_tol` |
| B at the other 23 | \|share\| ≤ 1e-4 | same |
| A on every part upstream of the edit | 0 ± 1e-4 | exact by construction |
| A at the edited MLP | 1.0 ± 1e-4 | exact by construction |
| A at blocks 2-3 for the block-1 plant | measured, no bar | this is the inheritance floor |
| All 24 reverted on any state | G ≤ 0.001 | E96's `zero_tol` |
| Gains re-read E62's and E53's | within 0.02 | E96's `g_tol` |

**Stage 1, control: copy against lookup, E62's six states.** A must rank (2,4) first on all three copy seeds, else the tree's first branch.

| Number | Expected on copy | Expected on lookup |
|---|---|---|
| Removal's own-terms fall at (2,4), of G | 0.56-0.65 (E96) | at or below 0 (E96) |
| A's top part | (2,4), lean; share unfiled | unknown, lean an MLP at block 0-2 |
| B's top matrix | unknown, no lean | unknown, no lean |

Then a is written. **Hand write against copy:** with (2,4) off, the block-3 plant's fall must stay within the lookup's fall (it was fitted to its own answers, downstream of the head): a by-construction new mechanism must not read as sharing copy's.

**Stage 2, lowercase against cipher, E53's step-300 states, seeds 0-1.** Lean: "not found" by the removal rule at head grain (E96: at most 0.39 of 1.67 for any head). New information: whether A and B put the gain in block MLPs, and whether the top A part clears the inheritance floor.

**Stage 3, the other saved pairs, only if stages 0-1 pass.** Same reads, all saved seeds. Rank-1, half-known and cipher at 24 keys; copy-24; ident32 against rand32. Beside only, with the same-windows null as control: emoticon and two ladder rungs (`L/recruitment_test/states/`). These stay "does not apply", with the two behavioural numbers beside. One independent agreement to look at: VPD's paper puts emoticon in `h.2.mlp.down_proj`; if B names another matrix, say so.

## Deliverable

`experiments/locator_table/results/table.json`: one row per (task, seed), E90's fields (`task`, `C`, `share`, `control_share`, `band`, `verdict`, `reason`, grepped in `experiments/new_vs_reuse/results/stage2.json` before coding), plus `part`, `a_share`, `b_share`, `c_share`, `w0_move`, `located`. `verdict` holds only the definition's words, filled by baseline-lead; this experiment supplies the part. A note to baseline-lead in its folder when it lands.

## Tree

- **A misses (2,4) on copy:** A reads propagation, not mechanism. Fall back to B with a direct-effect read in the one amendment; say so.
- **A plant reads wrong in stage 0:** a code bug. Fix and rerun; no reading.
- **Stages 0-1 pass, lowercase has no located part:** report "not found at head or MLP grain" with where A and B put the gain, and run stage 3.
- **Block-1 plant's inheritance at blocks 2-3 exceeds the smallest nominated share on lowercase:** this grain cannot nominate there; report it and stop that row.
- **Stage 3 locates parts on few tasks:** the table says so; rows stay unreadable with the reason. Whether a finer grain is worth it is a question for the user then.
- **The user wants labels on constant-answer tasks:** that reverses line 121, a decision card, not an amendment.

## Overrides of the advice

Stage 3 is added: the task asks for "across the 67M experiments", which three pairs do not reach, and it costs a few minutes. The emoticon agreement with VPD is added as a beside-reading only, since VPD must not choose the part.

## Saved

Every per-part vector for both reads, per state, both units; `results/` small, no states written. Time: E96's sweep ran 3-4 s per E62 state and 2.5 min for W₀ plus four E53 states at 500 × 127 tokens (`experiments/locate_mechanism/results/run.out` lines 16-21, `stage2.out` lines 12-16). A and B at 60 parts are about 2.5 times that; stages 0-3 fit under 25 min on an L40, split into two `debug` jobs if not.

## Not run

Direction-level units (E90's value form already reads degree only). Finer MLP units. Position-restricted patches. Rebuilt states (class map, many cues). Anything on Qwen.
