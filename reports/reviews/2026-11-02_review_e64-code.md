# Review: E64 code and filed rules, before its first GPU run (2026-11-02)

Scope: commit b0a3274. `experiments/readout64/run.py`, the E64 section of `src/commute/readout.py`,
`cluster/readout64.sbatch`, `experiments/readout64/results/predictions.json`, against
`reports/plans/2026-11-02_plan_baseline-next3-draft.md` (option 1 and its amendment). Reviewer wrote none of it. No saved
state and no real model were loaded and no outcome was read: only the JSON records' key structure, and a random
stand-in network.

**Verdict.** The code does what the filing says and every path executed on the stand-in. No arithmetic bug found. The
findings are rules: F1 and F2 both lean towards the claim, F3 lets an under-fitted readout's miss read as a real result.
Amend before running; all of them are cheap, and nothing has run, so no rule is chosen knowing an outcome.

## Findings, by severity

### F1 (high). The cipher bar grows with |M| and is not a statement about the cipher
`src/commute/readout.py:441-445`, called at `experiments/readout64/run.py:163`; reaches P1 and P3 (`readout.py:472`).
- **Defect.** "Cipher not possessed" means fewer cipher keys than 3/4 of the elicit |M|. That bar depends on how many
  elicit keys W0 holds, and it has no gap against the elicit count. Chance is 2.5 keys and W0's own cipher set is at
  most 2.
- **Input and wrong result.** |M| = 16, so the bar is 12. Block 1 possesses 12 of M and 11 of 32 cipher keys. P1 reads
  pass, reading "claim": elicit held, cipher not, on 12 against 11. The same 11 at the last block pass P3 with W0's
  cipher set at 0, so the "construction check" passes on a readout holding 11 cipher keys the model does not. At |M| =
  24 the bar is 18.
- **Smallest fix.** A fixed bar: possessed iff count >= 6, the filing's own "at least 6 keys" (3/4 of `min_M`). That is
  `cipher_possession(own, th["min_M"], num, den)` at `run.py:163`. It equals today's rule at |M| = 8.

### F2 (high). P2's cipher arm has no positive control, and its rows are trained on far fewer positions
`experiments/readout64/run.py:204` (the control passes on the elicit arm alone) and `:227`; `settings.cap` in
`predictions.json`.
- **Defect.** Removal at l* is validated only on elicit's M. Cipher rows see about 350 positions, common elicit rows
  2000. A worse-fitted row is a noisier direction, so its removal costs less, and that alone pushes A towards elicit >
  cipher: P2's pass direction. P1's `blind_to` states the prior tilt; P2's says nothing about it. The last-block removal
  is the only reference on cipher rows and it is at another block.
- **Input and wrong result.** Cipher rows at l* are mostly noise, so A(cipher) is about 0 at every seed. Elicit A is 0.3
  with range 0.1. P2 passes and the branch reads "the pair's label is causal", from sample size alone. Nothing in the
  run tells this apart from the claim.
- **Smallest fix.** Train a second readout with `cap` at the cipher's minimum (241) for every value, in seconds on the
  cached residuals, and report A at the same block beside the scored one. File before running that a P2 pass which
  vanishes there reads "not separable from row quality". Add the sample-size exposure to P2's `blind_to`.

### F3 (medium). The fit bar compares the readout with W0 at a different prior
`experiments/readout64/run.py:142-145`.
- **Defect.** W0 keeps its natural prior on a sample capped at 2000 per value, so its loss there is inflated by the
  values over the cap (fit counts have a median of 3207). The readout was trained on the capped prior and gets that
  margin free. The bar is one-sided, so the margin all goes to passing. `fit_pass` matters: `branch64`
  (`readout.py:494`) turns a P1 elicit or P2 miss into "unreadable" only when the bar fails.
- **Input and wrong result.** W0 natural 2.60, W0 at the sample's prior 2.20, readout 2.65 (illustrative numbers). The bar
  passes although the readout is 0.45 nats worse than a fair W0. A P2 fail then reads "no causal footing; the line
  pauses" instead of "unreadable".
- **Smallest fix.** Read the bar against W0 plus the shift `w0_at_training_prior` already computes (`run.py:118`), added
  to `wh`. Log both.

### F4 (medium). l* has no floor on |M_fit|
`experiments/readout64/run.py:172`; `min_M` is applied only to the all-windows pool at `:252-254`; `unfit_stop` likewise.
- **Defect.** The fit half has about 1.5 windows per key, so M_fit is single-window readings, and its size is unchecked.
  `edge_cases.empty_M_fit` covers only size 0.
- **Input and wrong result.** |M| = 8, M_fit = {one key}. Need is 1. Block 0 ranks that value <= 5 on its one window, so
  l* = 0 and all of P2 is scored at a block chosen by one window.
- **Smallest fix.** l* is None unless |M_fit| >= 6 (3/4 of `min_M`), with a dated edge case. P2 then voids with its reason.

### F5 (medium). W0's positive control can pass without the row being specific
`experiments/readout64/run.py:199-204`.
- **Defect.** The bar is an absolute 0.3 nats and nothing compares it with another direction in the same space. The
  random-direction drop is read over all keys (`:202`, `t.keys`), not M's windows, so it is not even on the scored
  windows. `cos_mean` measures only the share common to the rows.
- **Input and wrong result.** Rows lie in a shared high-variance subspace at l*, so removing any value's row costs W0 0.4
  nats. The control passes and P2 goes on to read "the gain runs through that direction".
- **Smallest fix.** Report, on M's windows, W0's drop under a wrong value's row (next index in the same task, same
  windows) and under the random direction. File the reading: a pass needs the target row's drop above the wrong row's.

### F6 (low; costs a rerun, not a number). `cache64` keeps every batch's whole stack on the GPU
`src/commute/readout.py:340`.
- **Defect.** `h[:, :, half]` is a view, so each of the ~223 batches holds its full (4, 256, 41, 768) stack, 129 MB, until
  the `torch.cat`. Confirmed on the stand-in: the view's `_base` is the stack.
- **Input and wrong result.** 57,000 windows peak at about 29 GB plus model, 12 states (`dws` load to their saved device)
  and 2 GB of per-batch logits, on a 46 GB L40. It probably fits; the plan measured no such number. A wider window set
  or a smaller card OOMs in stage 0.
- **Smallest fix.** `hs.append(h[:, :, half].clone())`.

### F7 (low). The step-250 early check cannot catch "still at chance"
`predictions.json` `thresholds.early_checks.bars["250"]`; `readout.py:365`.
- **Defect.** The note says a readout still at chance stops, reading chance as ln 64 = 4.16. The sample's own prior
  entropy from the filed counts is about 3.8, under the 4.0 bar, and the rule subtracts 3 standard errors besides.
- **Input and wrong result.** A readout that learned only the value frequencies has loss 3.8 at step 250 and passes.
  Step 2000 and the fit bar remain as guards.
- **Smallest fix.** Bar = the entropy of `y` minus 0.2, computed in the run; or correct the note.

### F8 (low). `branch64` drops an earlier cipher possession from the null reading
`src/commute/readout.py:496`.
- **Input and wrong result.** Elicit blocks [3], cipher blocks [1], P2 void. It returns "P1 holds only at the last block …
  a reading of the prior", yet the cipher is held earlier than the elicit map, which is not the filed null. Filed
  `edge_cases.cipher_possessed_somewhere` expects the cipher reading.
- **Smallest fix.** Add `and not a["cipher_blocks"]` to that condition.

## Checked and sound
- **`readout.ablation`, `w0_drops`, `remove_at`.** Ran at all four sites on the real `Net.run` hooks, on a stand-in
  network on CPU, so still unrun on a GPU. After removal h.u is about 3e-7 (from about 2) at the answer position, other
  positions are unchanged, and earlier sites are unchanged. Device moves read correct; `U[sc]` aligns with `t.toks[sc]`.
- **`run.run` end to end** on the stand-in with `check_inputs`, `pile_rows`, and the stop rules patched to reach each
  path: l* at 0, 1, 2 and None, W0 fail, P3 fail. Every `dump`, `say` and `json.dumps` worked and every stop wrote
  run.json. Repo results untouched. Tests: 29 passed (`test_readout64`, `test_readout`).
- **Windows.** `next_token_windows`: index arithmetic correct, residual at `half`, next token at `half + 1`, asserted in
  the run; fit and held rows disjoint and away from the task rows.
- **Training.** Loss per block, schedule, seeds and `Probes` shape match `train_readouts`' optimiser. Batch sd check as
  filed. `idx_of` preserves window order.
- **Rules.** Rank, specificity, M, possession, `unfit_stop`, `model_stop` and P2 match the filing at their bounds. `shares`
  is E59's amended rule. Stage 0's order matches: no fine-tuned removal is scored before P3, do-nothing and W0 pass.
- **Records.** `saved_states`' key paths exist in `stage1.json`, `stage2.json` and `job2.json`. sbatch differs from E59's
  only in name and script. `results/large` is a symlink, and readouts.pt writes through it from the job.

## Not run
- Anything on a GPU, the real model, the saved states or Pile rows.
- The sizes in F3 and F6: F3's numbers are illustrative, F6's come from the filed window count.
- Whether the residual scale makes step 250 stop falsely at lr 1e-3 (E59's log was not used to set anything).
