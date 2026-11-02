# Adversarial review of the E63 code, before it runs
*Review, 2026-11-02, of commits 8182c27 and 3aaa18c against `reports/plans/2026-11-02_plan_e63-neuron-sparsity.md`. By an agent that wrote neither. No GPU, no model loaded. Scripts and output: `~/commute-data/scratch/baseline-lead/e63-review/` (`check_states.py`, `toy.py`, `states.out`).*

I read the code and ran small checks on the saved weight changes to see whether the five numbers mean what the plan says. The arithmetic, the indexing, the file loading and the first batches are right; I found no bug that makes a number wrong. The problems are in how the rules read the numbers: the basis guard can pass for reasons that have nothing to do with neurons, family B's test can fail because of training length rather than kind, and C1 can pass on one grid step, so a yes or a no from `score.json` could say more than it means.

**Disclosure.** W1 needs no model, so `check_states.py` computed W1 and its rotated value on all 26 saved states. W1's outcome is therefore already visible to anyone reading this: it does not answer yes, as P1 leans (elicit is the denser side in every family). The fixes below were chosen on what the rules can do, and they do not change W1's outcome. No other number was read. Any rule change goes in as a dated amendment before the run; nothing has run, so none of this is a refiling after a result.

## Findings, ranked

### 1. The basis guard passes on the wrong side, and for A1 on any update (high)
`src/commute/neuron_sparsity.py:154-163`, `experiments/neuron_sparsity/run.py:342-345`, `predictions.json` `rules.guard`.

The guard exists so that a number with no neuron information cannot answer yes. It is two-sided and reads only the level on the elicit side, so it does not do that job.

- **Wrong side, verified on real states.** On family A, W1 averages 0.976 and its rotated value 0.923, a gap of 0.053 against a seed range of 0.005. The guard passes, yet the neuron basis is *denser* than the rotated one. This is what Adam's flattening predicts. Rotating low-rank, flattened rows brings back variation, so the rotated value is not a dense ceiling. `below_reference` records the side, but `answers_yes` ignores it. It is moot for W1, which fails T1; nothing stops the same pass on W1g or A1.
- **A1 passes whatever the update, verified on a toy only.** The GELU makes the neuron basis special for every update, because inactive neurons barely move. A Gaussian change of the pre-activations through a GELU reads 0.02 to 0.27 of a block in the neuron basis and 0.33 rotated (`toy.py`, pre-activation mean -2 to +1). The 0.33 is the kurtosis limit of a single rotated vector, N/3. So A1 sits below its rotated value for a Gaussian update too, and a pass shows nothing about the fine-tune. The real model's size is unmeasured.
- **W1g, suspicion, same mechanism.** A row of the c_fc gradient carries GELU'(pre-activation) as a factor, so W1g is likely sparser than rotated on any task. Not computed.

**Fix.** Make the guard one-sided, on the sparser side. For A1, read the per-state Gaussian too, and use it as the guard's reference in place of the rotated value. `parts` returns only eight matrices and `Net.at` asserts all 24, so zero-fill the other 16 for this read. For W1g, take the reference from the same first batch with its targets shuffled.

### 2. T3_B pools step-10 and step-30 fine-tunes, so it can fail on training length (high)
`predictions.json` `tests.T3_B`, `src/commute/neuron_sparsity.py:147`.

The new side holds `rand32` at step 10 (norm 0.63) and `rand32b` at step 30 (norm 1.12). `ident32` and `ident32b` are both at step 10. The within-kind range therefore measures progress, not seed noise. Verified on W1: `rand32` averages 0.970 and `rand32b` 0.926, so the spread is 0.047 and the kinds' gap of 0.025 falls under it. Family A shows the same pull, with W1 at 0.971 for the matched cipher and 0.923 at its own step 30. Any number that moves with progress fails T3_B through that spread, whatever the kinds do, and because every yes needs T3_B, the tree's "none" branch then writes "neurons do not map eliciting more cleanly". The plan's "not size-matched" names this difference but does not carry it into the rule. A1 and C1 were not read, so whether they are hit is unknown.

**Fix.** Compare per task: take each task's mean, compare the elicit tasks with the new tasks at step 10 only (`ident32`, `ident32b` against `rand32`), and report `rand32b` beside them. Or keep all four and measure spread within task, with the two task pairs required to agree in direction. The first is simpler and removes the confound.

### 3. C1 passes on one grid step, against the file's own noise floor, and its guard cannot fail in practice (high for P5, medium overall)
`src/commute/neuron_sparsity.py:147-151,161`, `run.py:344`, `predictions.json` `numbers.C1.noise_floor`, `edge_cases.ties_rule`.

- **Verified.** `separation([11]*5, [12]*5, True)` passes: a gap of exactly one step with zero range. The filed `noise_floor` says one grid step is noise, but the code has no such floor. `ties_rule` files the pass as intended, so the file contradicts itself. The plan itself calls E56's gap "one coarse grid step" and treats it as weak.
- **Suspicion, from E56's mean-ranked curves.** Half of each state's own full share falls between 1024 and 4096 for all three kinds: elicit 0.47, matched cipher about 0.34, own-step cipher about 0.48. Elicit and own-step cipher then likely read 2048, and matched cipher 2048 or 4096. Per-window ranking may shift this, but a T1 pass is likely to be a single step, which is exactly what P5 would score as "right". Dividing by the state's own full share also pulls the cipher, whose full share is near 0.69, towards elicit (near 0.94).
- **Verified.** The guard passes at C1 = 13 on every seed, because 0.585 is over range 0. It fails only if all five seeds read full width, and a rule that nearly always passes is no guard. Nothing says the ranking does better than chance.

**Fix.** Apply the filed floor: for C1, spread = max(range, one grid step). Better, read the crossing by log-linear interpolation of the share curve, so that C1 is not a grid number at all. For the guard, use per-window *random* ranks with a fixed seed on the same grid and compare with that curve's crossing. This doubles stage 3's forwards, still minutes.

### 4. The kinds' gap is not tested in the neuron basis; only its level is (medium)
`run.py:342-347`, `src/commute/neuron_sparsity.py:56-58,61-69`.

Rotation is within a block, so it keeps how the update divides among blocks and it reads the update's effective rank. T1-T3 test the unrotated number alone. So a pass can come from either of these, and the guard passes separately on a constant offset.

- **Block shares, verified on W1, family C.** Lowercase reads 0.938 and broad cipher 0.816, a gap of 0.122. Their rotated values keep 0.080 of that (66%), and the between-block participation alone differs by 0.071: broad cipher puts 38% of its energy in block 3, lowercase 30%.
- **Rank, verified on W1, family A.** The rotated gap is 0.060, eleven times the neuron-basis gap of 0.0056. The rotated value tells the kinds apart better than the number it is the reference for.

A1 keeps block shares under rotation in the same way, though it was not read. **Fix.** Require each test's gap to exceed its gap on the rotated reference, in the same direction, for W1, W1g and A1 (on the Gaussian for A1 if finding 1's fix is taken). Report the block shares and per-block W1 in `weights.json`.

### 5. T1, T2 and T3_C compare one task with one task, so seed range cannot test kind (medium, a reading, not a bug)
`predictions.json` `tests`, `src/commute/neuron_sparsity.py:147`.

Between seeds a number barely moves, so two different tasks almost always "separate" and only the direction carries information. Verified on W1: seed ranges are 0.003 to 0.005, while the reported-only `e53_cipher` (0.896) sits 0.080 from `broad_cipher` (0.816), both on the new side, against a kind gap of 0.122. The conjunction protects against a false yes. A pure-computation simulation with task differences ten times seed noise and no kind effect gives about 2% for T1, T3_B and T3_C together per number, or about 9% over five numbers, counting T2 as following T1. **Fix.** Keep the rule. Say in the write-up that a pass reads "these tasks differ, in this direction", and put `e53_cipher` into T3_C's new side, or report it beside T3_C on every number.

### 6. The per-window ranking has no GPU control if the 16-neuron update is unreadable (medium-low)
`run.py:315-317`.

C1 ≤ 4 is the only stage-0 check that exercises `window_ranks` and `ranks < k`, and it is skipped when that update is under the floor or its full share is not positive. That is possible here: the update nearly doubles the 16 most active neurons, and the patch keeps down_proj at W0. The E56 checks do not touch the ranks, since k = 12288 is all True. **Fix.** Gate, whether readable or not, on the share at k = 16 equalling the share at k = 12288 within 1e-6. It holds by construction when the ranks are right, and it needs no floor. Divide nothing: compare the raw rises in nats, since the share can be nan.

### 7. W1g on family C may mostly read ordinary text (suspicion)
`run.py:231-245`.

The loss runs over every position of 8 rows of 512 tokens. Only about 9% of broad cipher's training tokens are mapped, and the lowercase loss at step 1 (3.31) is close to plain text's. The gradient may be largely the shared next-token gradient, so T3_C may fail on batch noise between the two batch orders: a false null on one number. Not computed; needs the GPU. **Fix.** Report it first. If W1g's T3_C fails with both kinds close, read the first-batch gradient of the task's text minus that of the cased rows, as an amendment.

### 8. A wrong first batch would only log a line (low)
`run.py:276-277`.

The first-batch loss is logged as SURPRISE, then W1g is scored anyway. The code's batch recipe reads correctly and all 24 filed values match the logs, so this should not fire. The tolerance is 0.1 and no gate reads it. **Fix.** Stop the job on a miss; a wrong batch makes W1g meaningless.

### 9. Small points (low)
- `src/commute/neuron_sparsity.py:10-11`: the docstring says rotation "leaves the model's function unchanged". That is false for c_fc: the GELU sits between c_fc and the neuron, and GELU(Q c_fc x) ≠ Q GELU(c_fc x). The number is unaffected, but the sentence is what makes the guard look stronger than it is (finding 1).
- `run.py:84`: `Out` records `git_head` but not `git_dirty`, which E56's run records. This clone holds other agents' uncommitted edits, among them `src/commute/readout.py`. E63 does not import that module, but in general the record could not show such an edit.

## Verified sound
By reading unless marked as run.
- **Run:** `tests/test_neuron_sparsity.py` passes, 8 of 8. `run.py` imports. All 33 files exist and match their recorded sha256. All 26 states have 24 matrices; c_fc is (3072, 768) and down_proj (768, 3072), matching the docstring. `code_step10` is a weight change (norm 0.42), not whole weights. All windows files hold 403 train and 97 held-out windows of 41 tokens, answer position 20, 32 keys, no `ci_len`. `pile_train` returns 2400 rows of 512 tokens for all three Pile tasks; `scored` works on a bare `Run` and returns 500 windows of 127 tokens.
- **Filed first-step losses:** all 24 match the fine-tunes' logs (E54, E56, E46, E53).
- **Weight readings:** `neuron_energy`'s axes; rotation on rows, columns and shifts; W2's dot product and energy weighting; `sixteen_update` giving exactly 16/12288 and W2 = 1; the Gaussian's 0.9987; the rotated control's 0.222 and its range, which holds down to about 1.6 effective dimensions.
- **Window readings:** `window_ranks` (ties, inverse permutation); `window_edit` (shapes, closures bound at definition, CPU mask moved); W0 and tuned reads on the same batch; odd-index scoring half in both `gains` and `window_curve`; `NextTask`'s target; state-to-task mapping, including `cipher_own` and the `e53_cipher` rename; `half_share_index` on non-monotonic curves; `clean` leaving the in-memory nan for `separation`.
- **Gradient:** the first draw of `Generator(seed)` is the fine-tune's; batch 32 and 8 are right; no `ci_len` is lost; autocast is off inside the net; KL at W0 is zero by construction and the key `kl_other` exists; `decomposed_params` gives the 24 matrices. Family C's text recipes match `stage_finetune` and `finetune_members`.
- **Rules:** `separation` is E51's rule plus the direction; all five numbers read the right reference keys; `answers_yes` needs all four tests and the guard; unreadable never passes.
- **Stage 0's reference to E56:** same path, same card type (L40 node) and same whole-batch W0, so 1e-6 is plausible. `--mem=16G` is what E56's job 1 ran with, and that job also did the split.
- **Memory and time:** capture holds under about 0.5 GB per batch, and 6-10 minutes is plausible inside the 30-minute limit.

## Not run
Any GPU read: A1, C1, W1g and W2 on real states, the real size of the GELU effect (finding 1), the sixteen-neuron update's readability (finding 6), family C's gradient composition (finding 7), `first_step_loss` against the fp32 forward, and stage 0 itself. `make lint` was not run. W2 needs W0 and so was not computed. No model was loaded on any node.
