# Why E44's do-nothing state kept the update: Γ's new values, not a bug

*E44, `experiments/qwen_maps/`, diagnosis written 2026-10-29 from the code and the saved results only; no GPU was used.
Plan: `reports/plans/2026-10-28_plan_qwen-q0q1.md`. Results: `experiments/qwen_maps/results/{C,S,D,K,score,stopped}.json`
and the end of `results/run.log`, from the A100 rerun of commit 81ec101.*

## In plain words

We read the code to find why the state that leaves the fine-tuning update out of its subcomponents still scored as
keeping almost all of it. The importance function reads the fine-tuned model's activations for that state, and the
masks it computes from them bring the masked model almost all the way to the fine-tuned one. This is not a bug but a
wrong expectation in the plan: the pass test cannot fail any state that holds the fine-tuned weights, so what the test
should measure is a question for the user.

## The puzzle

On the 128 maths rows at fine-tuning step 10, three numbers matter. Each is a mean KL in nats per token.

- $K(D) = 1.4156$: KL from the fine-tuned model to the original decomposition with CI masks, where Γ computes the
  masks from the original model's activations. Stage D, `run.py:111-113`.
- $K_0 = 0.0589$: KL from the original model to that same masked decomposition. Stage C, `C.json` `ci.maths`.
- $K = 0.0932$ for do-nothing: KL from the fine-tuned model to the do-nothing state's masked model. Do-nothing keeps
  the original subcomponents and puts all of the update in Δ; its masks are CI masks with Δ off.

The share kept is $r_2 = (K(D) - K)/(K(D) - K_0)$, 0 for a state that drops the update and 1 for one as close to the
fine-tuned model as the original decomposition is to the original model. The arithmetic checks:
$(1.41555 - 0.09325)/(1.41555 - 0.05888) = 1.32231/1.35667 = 0.9747$, as `D.json` reports. At step 1 it is
$(0.28040 - 0.06399)/(0.28040 - 0.05888) = 0.9769$, also as reported. The plan expected do-nothing's ratio to the
original model on maths rows at 8 to 11 and $r_2$ near 0; it read 1.58 and 0.975.

## Diagnosis

**The explanation is (c): the masks Γ computes from the fine-tuned model's activations carry the update.** The code
leaves no other path, as follows.

### What the masked forward uses

`qwen_vpd.masked_logits` (`src/commute/qwen_vpd.py:316-337`) hooks each of the 196 decomposed linear layers and
replaces its output. With Δ off (`deltas is None`) the hook returns `((xV * mask) @ U)` and never reads `out`, the
product with the layer's weight (`qwen_vpd.py:329-330`). So the weights swapped into the model do not enter the
masked logits at any decomposed layer.

The parameters outside those layers are the embedding, the norms and the output head. `set_weights`
(`src/commute/qwen_sft.py:47-60`) swaps only the decomposed layers, and `Job.ft` builds the fine-tuned weights over
`SITES` only (`experiments/qwen_maps/job.py:204-212`). Those other parameters are the original model's in every
forward, masked or not.

The do-nothing state does not move any subcomponent: `site_change` returns `(M, None, None, None)` for it
(`src/commute/qwen_maps.py:218-219`). `applied` then skips the column writes (`qwen_maps.py:250-251`). Its $U$ and
$V$ are the same tensors that the original-model state uses for $K(D)$.

What is left that differs between the two forwards is the mask. `read_rows` computes it once per batch with
`importances(model, tokens, gamma, comps)` while the state's weights are swapped in (`qwen_maps.py:307-309`,
under `applied`'s `set_weights`, `qwen_maps.py:257`). `clean_forward` (`qwen_vpd.py:288-305`) feeds Γ two inputs
from that forward. One is the 29 residual-stream taps of every layer, which a global 32-block transformer reads
together, so a mask at layer 0 can depend on the last layer's residual. The other is each site's readout of
$x_{\text{site}} V$. For do-nothing both come from the fine-tuned model; for $K(D)$ both come from the original one.
The 1.32-nat gap between $K(D)$ and do-nothing's $K$ is therefore made entirely by the difference between those two
sets of masks.

### The other explanations, one by one

- **(a) A layer-local masked forward. Ruled out.** The hooks replace module outputs, so every later layer reads the
  masked activations, not the target's. The only path from the fine-tuned activations into the masked model is
  through Γ's inputs, which is (c).
- **(b) Δ or δW still on. Ruled out.** With `deltas=None` the hook discards `out` (`qwen_vpd.py:329-330`), and no
  non-decomposed parameter carries the update.
- **(d) Different rows, positions, masks or references. Ruled out.** $K(D)$ and the state reads both call
  `Job.read` on `rows("maths")`, the same 128 rows of 512 tokens, averaged over every position
  (`job.py:180-182`, `qwen_maps.py:326`). Both use the same reference: the fine-tuned weights from `Job.ft(t)`, swapped
  in for `"K"` (`qwen_maps.py:320-322`). Both use CI masks with Δ off, the same `comps` object and the same Γ. The one
  intended difference is the activations Γ reads, which the plan's §3 table states in the $r_2$ row. $K_0$ is the
  original model's CI-masked KL on the same rows, as the plan defines it.

### The controls agree with this

The random control's weights are the original ones plus a random change, not the fine-tuned ones. It is not in
`FT_WEIGHTS` (`job.py:41`). Its masks come from activations close to the original model's. Its $K$ is 0.2806 at step
1 and 1.4193 at step 10, within 0.3% of $K(D)$ at both steps, so $r_2$ is about 0. The two halves of the first
control, by contrast, hold the fine-tuned weights. They read the same as do-nothing to four digits at step 1: 0.06399
and 0.06401 against 0.06399. A state's $K$ thus follows whose activations Γ reads, not what its subcomponents hold.

The halves also show how little a change to the subcomponents moves these readings. The lower half moves
subcomponents that Γ rarely switches on in maths text, so CI masks multiply its changes by near 0. It reads 0.093247
at step 10 against do-nothing's 0.093246. The upper half, the most active subcomponents, moves it by 6e-4 nats.

### Why the plan expected otherwise

The plan's §1 item 2 says "CI masks run with Δ off, so the masked model drops δW". The "When r2 adds anything"
paragraph of §3 repeats "do-nothing reads about 0". Both treat the update as living only in the weights. But the
same §3 says "Γ recomputes g on each state's own activations", and for do-nothing those activations are the fine-tuned
model's. The code follows the specification as written. The expected values in the §3 table, do-nothing's maths ratio
8 to 11 and $r_2$ about 0, did not allow for the masks carrying the update.

Whether Γ's new masks describe what the update did, or only let the masked model copy the fine-tuned model's output,
is the question the separate stage 0 table of the other line asks: `reports/analyses/2026-10-29_analysis_e46-regating-table.md`.
On that line's small model, Γ's new values alone raised the masked model's answer a lot for real fine-tunes and little
for random noise. Nothing here settles it for Qwen.

## The other two controls

The diagnosis above is about the third control, do-nothing. The other two failed for their own reasons, so a fix
to the third would not give the run its teeth.

- **First control, the halves.** It was not scored: "confounded by the span". The halves' shares of the update left
  in Δ were 0.71 and 0.52 at step 10, further apart than the filed tolerance. Even unconfounded, both halves hold
  the fine-tuned weights and inherit do-nothing's masks, so both would read about 0.975.
- **Second control, random.** Its largest ratio to the original model was 1.0004 against the 1.5 line. A random
  change to the subcomponents matched in size to the update leaves the masked model as close to its own unmasked
  model as before. Random directions move the output much less than the update's directions do.

## What it means for the decision tree

- This is not a code bug. The code computes what plan §3 specifies; the filed expectations were wrong about it.
  "Control misses → fix; rerun the stage" assumes a fix that keeps the measurement's meaning. There is none here.
- Teeth need the masks of a state whose unmasked model is the fine-tuned one to stop carrying the update. That needs
  one of two changes to what the pass test measures. Either Γ reads the original model's activations for every
  state, as $K(D)$ does, which removes the masks' contribution from the test entirely. Or the test reads something
  other than closeness to the fine-tuned model. Either changes the pass criterion that the user fixed on 2026-10-19.
- So the branch that applies is **"No teeth at step 10 → the line stops; card to the user on the pass criterion"**.
  The card should say plainly that, with Γ reading each state's own activations, any state holding the fine-tuned
  weights keeps about 97% of the update by the masks alone, at steps 1 and 10.
- One reading the score did not use: do-nothing's maths ratio at step 10 is 1.58, over the 1.5 line, so its overall
  pass verdict was "fail" through that ratio. The control's scoring rule reads $r_2$ where it is scored, so that
  failure did not count as teeth.

## A $0 check, if the lead wants direct confirmation

The code settles the explanation by elimination, so no run is needed to choose between (a) to (d). One short cluster
job would confirm it directly and say which of Γ's two inputs carries the update. It is offered, not required.

**Set-up.** One L40, debug QoS, ≤ 30 min, through `/pod-runner`. Use step 10's δW from E40's `step_10.safetensors`,
the first 16 maths rows cut to 256 tokens, batch 1, subcomponents in fp32 and Γ in bf16 as in the run. To fit 46 GB,
keep the fine-tuned weights on the host and copy them in for each swap; memory is then about 38 GB. Per row, run one
clean forward at the fine-tuned weights, keeping the logits, taps and readouts. Run one clean forward at the original
weights, keeping the taps and readouts. Then run the masked forwards below, each with the original subcomponents and Δ
off, scoring KL from the fine-tuned logits. Each forward is one `masked_logits` call on masks from
`lower_importances` on the chosen taps and readouts.

| reading | weights swapped in | Γ's taps from | readouts from | (c) | (a) | (b) | (d) |
|---|---|---|---|---|---|---|---|
| 1 | original | original | original | ≈ $K(D)$, 1.2-1.6 | same | same | far from $K(D)$ |
| 2 | fine-tuned | fine-tuned | fine-tuned | ≈ 0.09 | same | same | — |
| 3 | original | fine-tuned | fine-tuned | = reading 2, to 1e-5 | ≈ reading 1 | ≈ reading 1 | — |
| 4 | fine-tuned | original | original | = reading 1, to 1e-5 | ≈ reading 2 | ≈ reading 2 | — |
| 5 | original | fine-tuned | original | between 1 and 2 | — | — | — |
| 6 | original | original | fine-tuned | between 1 and 2 | — | — | — |

Readings 3 and 4 decide (a) and (b) against (c): under (c) the weights swapped in change nothing, so swapping them
leaves the reading where the masks put it. Reading 1 on these rows checks (d) against stage D. Readings 5 and 6 split
the update's carrier between the global residual taps and the per-site readouts. If the taps carry most of it, the
masks are reading the fine-tuned model's late layers, which points to copying its output rather than describing its
change. Their tolerance between paths is the fp32 noise floor, which readings 3 and 4 measure against 2 and 1.

## Not run

- No GPU job was run or submitted; the check above is a proposal.
- The halves were not rescored with a tighter share tolerance; both would read near do-nothing anyway.
- No version of the pass test with Γ fixed to the original model's activations was run; that is the user's
  decision on the pass criterion.
- The web rows' readings were not examined beyond confirming they sit near the original model's: do-nothing's web
  ratio is 1.04 at step 10.
- Whether Γ's new masks describe the update or copy the fine-tuned output on Qwen was not tested.
