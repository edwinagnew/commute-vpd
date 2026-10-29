# Analysis: do Γ's masks reproduce E40's update where the update is made, or only from the last layers? (E44 Amendments 1 and 1b)

*2026-10-29, analysis agent for the lead. Plan: `reports/plans/2026-10-28_plan_qwen-q0q1.md`, sections "Amendment
record (2026-10-29, Amendment 1: layer-cut check)" and "Amendment record (2026-10-29, Amendment 1b: consistent
inputs)". Predictions `experiments/qwen_maps/results/predictions_amend1.json` and `predictions_amend1b.json`, each
filed before its code and unedited. Code `experiments/qwen_maps/amend1.py`, `amend1b.py`. Results
`experiments/qwen_maps/results/amend1/` and `results/amend1b/`, each with its `run.log`. Figure from
`experiments/qwen_maps/plot_amend1.py`. Notebook `notebooks/E44_qwen_maps.ipynb`, section "Amendment 1". Context: the
diagnosis `reports/analyses/2026-10-29_analysis_e44-no-teeth.md` and Fable's advice
`reports/plans/2026-10-29_advice_e44-pass-test-FABLE.md`. Every number below was read from the results files with `jq`.*

**In plain words.** We fed the importance function either a mix of two models' activations or every activation of
one model whose first sixteen layers alone were fine-tuned, and measured how much of the update its masks reproduce.
The mix recovered none of it, but the consistent model recovered 90%, close to the 93% those sixteen layers make.
For `GOALS.md`, the masks follow whatever activations they are given, so the pass test as built will pass do-nothing
for any update, and these runs cannot say whether that is copying the output or reading the change.

## The answer

Γ's masks follow whatever activations Γ is given. Given one consistent model whose update sits only in the first
sixteen layers, the masks recover 90% of the update's output effect, against the 93% those layers make. Amendment 1's
null, and its filed verdict "copying", came from splicing two models' activations together. So under a pass test that
takes masks from the model being judged, do-nothing will pass for any update. Whether Γ copies the output or reads the
change, these runs cannot tell apart.

On these 16 rows, do-nothing's masks keep 93% of the update, one minus the ratio of the masked KL at the last cut to
that at the first. The 97% quoted for E44 is a different reading, its stage D on 128 rows of 512 tokens with the
original model's noise floor subtracted.

## What we were trying to test

E44 found that the pass test cannot fail do-nothing, the state that leaves the whole fine-tuning update in Δ and
keeps the original subcomponents. The reason is that Γ, the importance function, reads the activations of the model
it is run on. For do-nothing those are the fine-tuned model's activations, and the CI masks Γ computes from them put
the original subcomponents within 0.09 nats of the fine-tuned model. These two amendments ask which of two things that
reproduction is.

- **Γ reads the change.** The update alters the early layers' activations, Γ switches on different subcomponents in
  response, and those subcomponents carry the new behaviour. This is route (b) of `GOALS.md`.
- **Γ copies the output.** Γ reads all 29 residual-stream taps at once, and the last taps nearly fix the next token.
  A mask conditioned on them can pick whatever subcomponents reproduce that token without describing the change.

Γ has two inputs. One is the 29 residual-stream taps: tap $j$ is the input of decoder layer $j$, and tap 28 is the
last layer's output. The other is each decomposed site's readout, its input times the original $V$. In both
amendments the masked model always runs on the original weights, with the original subcomponents and Δ off, and each
reading is its KL from the fine-tuned model, averaged over every position. Only what Γ reads changes.

**Amendment 1 splices Γ's inputs at a layer.** At **cut ℓ**, Γ gets the fine-tuned model's taps 0 to ℓ and its readouts
of layers before ℓ, and the original model's for the rest. The cut ran at ℓ = 0, 8, 16, 24 and 28. Cut 0 is the
original model's masks; cut 28 is do-nothing's.

**Amendment 1b gives Γ one consistent model.** $H(\ell)$ is the unmasked model with fine-tuned weights before layer ℓ
and original weights after. Γ gets every tap and readout of $H(\ell)$'s own forward pass, at ℓ = 16 and 24. Its late taps now agree with its early ones, because they are computed from them. It was run because an
independent check of Amendment 1 pointed out that a Γ reading the change through its effect on the late taps would
fail the splice too.

Four quantities are read, each 0 at the original model and 1 at the fine-tuned one.

- $K(\ell)$ is the masked model's KL from the fine-tuned model with Γ's inputs set at ℓ.
- $m(\ell) = [K(0) - K(\ell)] / [K(0) - K(28)]$ is the share of the masks' reproduction reached under the splice.
- $m_H(\ell)$ is the same share when Γ reads $H(\ell)$, with $K(0)$ and $K(28)$ re-measured in that job.
- $u(\ell) = [\mathrm{KL}(\text{FT} \| W_0) - \mathrm{KL}(\text{FT} \| H(\ell))] / \mathrm{KL}(\text{FT} \| W_0)$ is the
  share of the update's own output effect made by its first ℓ layers. It is the comparator for both: if Γ responds to
  the change, the masks should keep pace with the layers that made it.

Two more readings in Amendment 1 split the carrier at cut 28: the fine-tuned taps with the original readouts, read as
$s_\text{taps}$, and the fine-tuned readouts with the original taps, read as $s_\text{reads}$. Both use the formula of
$m$. Amendment 1b adds one unscored reading: how close the masked model stays to $H(\ell)$ itself when Γ reads
$H(\ell)$, the KL from $H(\ell)$'s own output.

The update is E40's no-warm-up fine-tune of Qwen3-1.7B on maths answers after 10 steps. The text is the first 16 maths
rows, each cut to 256 tokens, at batch 1. Each amendment ran on one cluster L40, $0.

## What we expected, and why

**Amendment 1.** The verdict is read at layer 16 if the first sixteen layers make at least 30% of the update's effect,
else at layer 24. At that layer, $m$ at least half of $u$ reads as Γ reading the change; less reads as copying. The
lean was copying, slightly favoured, because the late taps nearly fix the next token and Γ reads all 29 together.
Copying also predicted that the taps, not the readouts, would carry the reproduction.

**Amendment 1b.** At layer 16, $m_H$ at least half of $u$ reads "reproduces from consistent inputs": the splice made
Amendment 1's null. At most a fifth of $u$ reads "no reproduction from consistent inputs", which would confirm copying
without the splice caveat. Between is "partial". The lean was "reproduces", moderate. The filed reason was that
$H(16)$'s output already carries 93% of the update, so even a Γ that copies the output of whatever model it reads
would reproduce most of it. The filed record says what the run is blind to: with consistent inputs, copying the output
and reading the change through its downstream effect give the same masks.

## The controls

All controls held in both jobs.

| control | rule | read | verdict |
|---|---|---|---|
| A1 floor: the same original-model read twice on row 0 | relative gap ≤ 1e-3 | 0, bitwise equal; tolerance 1e-6 | holds |
| A1 identity: cut 0 and cut 28 against the unchanged code path | relative gap ≤ 1e-6 | 0 and 0 | holds |
| A1 effect: $K(0) - K(28) \ge 0.25$, $K(28)/K(0) \le 0.2$, $\mathrm{KL}(\text{FT} \| W_0) \ge 0.05$ | all three | 1.393, 0.068, 1.817 | holds |
| A1b floor: Γ reading $H(0)$ twice on row 0 | f ≤ 1e-3 | 0 | holds |
| A1b reproduce: $K(0)$, $K(28)$ against Amendment 1's 1.49567488 and 0.10243869 | relative gap ≤ 1e-3 each | 0 and 0, bitwise | holds |

Beside the effect control, against E44 stage D's 128 rows of 512 tokens: $K(0) = 1.4957$ is 1.057 times stage D's
1.4156, and $K(28) = 0.1024$ is 1.099 times its 0.0932. Different rows and lengths, so no limit applies.

## What we observed

### Does Γ reproduce the update from the layers where the update is made?

![m, m_H and u against the cut layer](../../experiments/qwen_maps/figures/amend1_layer_cut.png)

**Reading it.** Under the splice, no; given one consistent model, yes. The blue line is the share of the update's
output effect made by the fine-tuned weights before each cut: 60% by layer 8, 93% by layer 16 and 99.4% by layer 24.
The red line is the masks' share under the splice. It reads 0.0001 at layer 8 and −0.0025 at layer 16, far under the
0.46 that half of the blue line asks for, and reaches 1 only at layer 28. The green line is the masks' share when Γ
reads the partly fine-tuned model itself. It reads 0.90 at layer 16 and 0.996 at layer 24, within three points of the
blue line at both. The green line has no point at layer 8; its dashes there only join 0 to layer 16.

| cut ℓ | $K(\ell)$, splice, nats | $m(\ell)$ | $K$ reading $H(\ell)$, nats | $m_H(\ell)$ | $\mathrm{KL}(\text{FT} \| H(\ell))$, nats | $u(\ell)$ |
|---|---|---|---|---|---|---|
| 0 | 1.4957 | 0 | 1.4957 | 0 | 1.8173 | 0 |
| 8 | 1.4955 | 0.0001 | not run | not run | 0.7342 | 0.596 |
| 16 | 1.4991 | −0.0025 | 0.2414 | 0.900 | 0.1341 | 0.926 |
| 24 | 1.0688 | 0.306 | 0.1078 | 0.996 | 0.0108 | 0.994 |
| 28 | 0.1024 | 1 | 0.1024 | 1 | 0 by definition | 1 |

Layer 16 is the sharpest contrast. Given the fine-tuned activations up to tap 16 and the original model's after, Γ's
masks move the masked model none of the way. Given the same early activations followed by the late ones they actually
produce, the masks move it 90% of the way, to 0.24 nats from the fine-tuned model. So Γ does not ignore the early
change; under the splice, the late taps it reads simply say the original model's output.

### Which of Γ's inputs carries the reproduction?

**Reading it.** The residual taps carry it. The two diamonds at the right of the figure are Amendment 1's split at cut
28. With every tap from the fine-tuned model and every readout from the original, the masked model reaches 0.989 of
the reproduction, at $K = 0.1180$ nats. With the readouts from the fine-tuned model and the taps from the original, it
reaches 0.223, at $K = 1.1845$. The difference is 0.77, past the filed 0.3 line for "taps". The readouts are each
site's own input, so their 0.22 includes the late sites' inputs, which are late activations too.

### How closely do the masked original subcomponents track the model Γ reads?

**Reading it.** About as closely for each partly fine-tuned model as for the original model itself. This is the KL
from the model Γ read to the masked model built from its masks, unscored. It reads 0.066 nats for the original model,
0.079 for $H(16)$, 0.090 for $H(24)$ and 0.102 for the fine-tuned model. The original model's 0.066 is the
decomposition's own faithfulness on these rows, the reference; the rest sit within 0.04 nats of it. So the original
subcomponents, switched on by Γ, describe each hybrid model nearly as well as they describe the model they were
trained on.

| model Γ reads | KL from that model to the masked model, nats |
|---|---|
| original, $H(0)$ | 0.0656 |
| $H(16)$ | 0.0792 |
| $H(24)$ | 0.0902 |
| fine-tuned, $H(28)$ | 0.1024 |

## What we learned

Amendment 1's filed verdict, "copying", stands as filed, but Amendment 1b overturns its reading. The splice showed
only that Γ's masks follow the late taps. When the late taps are the real consequence of the early change, the masks
reproduce 90% of an update made in the first sixteen layers. Γ's masks follow whatever activations they are given.
The original subcomponents they switch on track each such model nearly as well as they track the original.

For the pass test this is decisive in one direction. Under reading A, masks are computed from the model being judged.
Do-nothing's model is the fine-tuned model, so its masks will reproduce the update for any update, however Γ gets
there. These runs cannot say whether to call that copying the output or Γ responding to the change. With consistent
inputs the two are the same function of the activations.

For the user's choice, as an input and not a decision: Fable's option 1 computes every state's masks from the
original model's activations, which is reading B. That removes do-nothing's free pass by construction, because the
masks no longer see the update at all. What it gives up is any credit reading A would give for Γ following the
changed activations. Amendment 1b shows that following is real and large on this update, so option 1 gives up
something. It does not show that the following describes the change rather than copying the output. The plan's branch
for Amendment 1's verdict also asked whether Γ's inputs should be cut to early layers. Amendment 1b answers that a Γ
fed one consistent model already follows early changes, so cutting late taps would not by itself separate the two.

Related evidence from the 67M, E49 in `reports/analyses/2026-10-29_analysis_e49-row-blocks.md`: on new Pile rows, the
adversarial check could not fail plain SGD's do-nothing, while it failed Adam's on every block. There too a test that
passes do-nothing could not credit a split, so its passes carried no weight.

## How compelling is the evidence

The two patterns are large and clean. At layer 16 the spliced masks reach none of their reproduction, and every one of
the 16 rows reads within 0.02 nats of its cut-0 value. The consistent masks reach 90%, and every row's KL at layer 16
lies between 0.13 and 0.36 nats, below its own cut-0 value on all 16 rows. Both floors are exactly 0, and Amendment
1b reproduced Amendment 1's two end readings bitwise, so the two jobs measure the same thing.

It is one update, E40's large format change at step 10, on 16 maths rows of 256 tokens. It has no random-update
comparator on Qwen. The consistent-input run is blind, by design, to copying versus reading the change. On the 67M,
E46 found Γ's new values gave real fine-tunes eight times a random update's gain, which is evidence that Γ there
responds to something specific to the update. That does not separate the two either: a Γ can respond to the update's
output specifically and still copy.

## Surprises

No filed prediction reversed: Amendment 1 leaned copying and read copying; Amendment 1b leaned "reproduces" and read
it. The surprise is in the reading. Amendment 1's result looked like evidence about Γ, and Amendment 1b shows it was
evidence about the splice. Three notes.

- **Amendment 1's reading is overturned.** Its first write-up said do-nothing's pass "is Γ copying the fine-tuned
  output, not reading the change". Amendment 1b shows the masks reproduce an early change from consistent inputs, so
  that sentence no longer holds.
- **The non-monotone flag.** $m(8) = 0.0001$ is above $m(16) = -0.0025$, so Amendment 1's score raised the filed flag
  "non-monotone". The filed rule reports it and does not score it. The step is 0.0035 nats in $K$, a quarter of a
  percent of the 1.39-nat gap, so both values are zero for this purpose.
- **The update's own effect is larger than the reference.** $\mathrm{KL}(\text{FT} \| W_0)$ read 1.82 nats on these
  rows, against E40's 0.52 for the reverse direction on other rows. No reference existed for this direction and these
  rows, and the rule only asked for at least 0.05.

## What next

- **The user's choice of pass test.** These two results go on the card beside Fable's option 1. If the user takes
  option 1, Amendment 2 reruns E44's controls with masks from the original model's activations and a random update
  matched on KL. If the user keeps reading A, every split passes on Qwen and the line moves to describing where δW
  lands.
- **Separating copying from reading the change.** Only a test whose inputs disagree with the output by design could do
  it, and Amendment 1 showed such inputs leave Γ's training distribution. No run is proposed; it matters only if the
  user wants reading A kept in some form.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| A1 control_floor | the same read twice agrees | f ≤ 1e-3 | f = 0 | holds |
| A1 control_identity | the splice at cuts 0 and 28 equals the unchanged path | relative gap ≤ 1e-6 | 0 and 0 | holds |
| A1 control_effect | the reproduction and the update's effect are present | gap ≥ 0.25, ratio ≤ 0.2, KL ≥ 0.05 | 1.393, 0.068, 1.817 | holds |
| A1 verdict | the spliced masks keep pace with the layers that make the change | ℓ* = 16 as u(16) = 0.926 ≥ 0.3; m(16) ≥ 0.5·u(16) = 0.463 → re-gating | m(16) = −0.0025 | **copying**, as filed; flag non-monotone, not scored; reading overturned by A1b |
| A1 carrier | the taps, not the readouts, carry the reproduction | s_taps − s_reads ≥ 0.3 → taps | 0.989 − 0.223 = 0.766 | **taps** |
| A1b control_floor | the same read of H(0) twice agrees | f ≤ 1e-3 | f = 0 | holds |
| A1b control_reproduce | K(0), K(28) equal Amendment 1's | relative gap ≤ 1e-3 each | 0 and 0, bitwise | holds |
| A1b verdict | masks from consistent H(16) reproduce the update | m_H(16) ≥ 0.5·u(16) = 0.463 → reproduces; ≤ 0.185 → none | m_H(16) = 0.900 | **reproduces from consistent inputs**, the filed lean |
| A1b layer 24, unscored | same bins at layer 24 | m_H(24) ≥ 0.497 → reproduces | m_H(24) = 0.996 | reproduces |

## Not run

- Any random-update comparator on Qwen, matched on KL or otherwise.
- Steps 1 and 100, and any update besides E40's.
- The web rows, and maths rows beyond the first 16 or past 256 tokens.
- Stage S, and any split state: only do-nothing's masks and the partly fine-tuned models were read.
- Cuts between 24 and 28 in either amendment, and layer 8 in Amendment 1b.
- Any test that separates copying the output from reading the change on consistent inputs.
- Amendment 2, the pass test with every state's masks from the original model's activations.

## Run and provenance

- **Amendment 1.** Cluster job 44065, one L40, `debug` QoS, $0, with the code of commit 67d9e4d. Stage 1 ended at
  293 s and the job at 345 s, of which 220 s was loading the decomposition. Host memory: the plan asked for
  `--mem=64G`; the lead lowered it to `--mem=48G` after the code review, and the job ran at 48G. The log's host peak,
  the process's maximum resident set, read 52.5 GB, above that limit, and the job was not killed; the resident set
  also counts file-backed pages, which the limit need not charge. GPU peak 44.7e9 bytes, inside the 44 GiB stop rule.
- **Amendment 1b.** Cluster job 44161, one L40, `debug` QoS, $0, `--mem=64G`, with the code of commit dccd40e. The
  controls ended at 78 s and the job at 128 s. GPU peak 45.3 GB, host peak 52.5 GB. Results committed in 6baded8.
- δW for both: `~/commute-data/results-large/qwen_sft/nowarmup_regen/dW/step_10.safetensors`, sha256
  8b8509d2…beb9, matching `experiments/qwen_maps/results/regen_SHA256SUMS`.
- Figure: `experiments/qwen_maps/plot_amend1.py`, reading `amend1_score.json` and `amend1b_score.json` only.
