# Advice: a pass test with teeth for the Qwen splits (E44)

*2026-10-29, Fable, advisory; from the code, `results/*.json`, diagnosis and plans. No run, no edit.*

## In plain words

Every state holding the fine-tuned weights gives Γ the same activations and masks, and the masks alone put the
original subcomponents within 0.09 nats of the fine-tuned model; moving subcomponents under them shifts the reading
by under 0.001 nats. No reading on a state's own activations can separate the splits from
each other or from do-nothing. Teeth need a mask that does not carry the update: reading B, set aside on
2026-10-19.

## 1. What the masks' reproduction means

It is route (b) of `GOALS.md`, "the frozen Γ gating differently on the new activations": 1.32 of 1.42 nats on
unchanged subcomponents. The definition does not say whether Γ *describes* the change or *copies* its
output: Γ reads all 29 residual taps of the unmasked fine-tuned model, and the late taps nearly fix its next token, so a
mask conditioned on them can pick whatever reproduces it. Against copying, E46 (67M) found the gate gain an eighth
as large for a matched random update; E44's random row (r2 ≈ 0) is sized by parameter norm and barely moved the
unmasked model (r1 maths 0.995), so it counts only once matched on KL to start.

## 2. Options

| option | measures | blind to | positive control | cost on saved states |
|---|---|---|---|---|
| **1. Γ on W₀'s activations for every state** (the state's V′ in the readouts): K(D′) under W₀'s masks against K(D) = 1.42 | what the moved subcomponents carry at the old gates: route (a), reading B | route (b): a split onto subcomponents Γ switches on only after the update fails | upper half < lower; KL-matched random ≈ K(D); do-nothing = K(D) exactly | split states not saved: stage S again (~35 min H100, or an L40 on `normal` QoS, $0), then forwards |
| 2. Not closeness: mask cost (mean importance, count on, vs W₀), or VPD's losses with Γ refit per state | sparsity of D′; or reading A as written | frozen Γ: identical across fine-tuned-weight states; refit: ~$10 a state (E15 drifted) | do-nothing's mask cost above W₀'s; KL-matched random | $0, no teeth; refit paid |
| 3. Descriptive placement (Q3): concentration over the random-like null, by layer, think-mode overlap | where δW lands | has no pass line | the null in `placement_*.safetensors` | $0, on disk |

A fixed reference mask (do-nothing's for every state) is answered by `K.json`: 6e-4 nats between the halves. Only
option 1 has a control that must fail. The 67M read d ≤ 0.08 on every ladder task; the same on Qwen is a
finding, not a broken apparatus.

## 3. The six-reading check

Readings 1-4 are settled by the code and by the halves matching (c) to four digits. Readings 5 and 6 are new, but
both inputs come from the fine-tuned model, so they do not decide describing from copying. Reshape it: taps from the fine-tuned model up to layer ℓ and from W₀ after, ℓ in {8, 16, 24, 28}; 16 rows, one
L40, under 30 min, $0, no stage S. Reproduction needing ℓ ≥ 24 means Γ reads the output; ℓ ≤ 16 means it reads the change. One question; run it before the card, not instead.

## 4. Recommendation

Send the card with option 1 as the proposed criterion, phrased as the choice it is: "score splits by what the moved
subcomponents carry at W₀'s gates (reading B), because on Qwen reading A is passed by the masks alone, do-nothing
included". Attach the layer-cut result (Amendment 1); Amendment 2, the last, is option 1's rerun with a KL-matched
random control. E44's predictions stay filed; option 1 files fresh ones: do-nothing 1.42 by definition, random ≈ 1.42, upper
half below lower by a stated margin, none for the splits.

What changes it: the user keeps reading A, so every split passes on Qwen and the line moves to Q3 with no pass
test; or the layer cut shows the late taps carry the update, and the card also asks whether Γ's inputs are cut to
early layers from now on.

Not run: no GPU; the halves' 6e-4 gap is unexplained; whether an L40 holds stage S is unchecked.
