cell: P1.57
rival_a: E_b over K sees a reuse localized at one head whatever the featurization, which predicts the built head-(2,4) V x2 edit (its value rows doubled, so its write is exactly doubled) reads E_b over K at least 0.5 at VPD, near the 0.76 of the built x2 strengthening edit, as at the neuron/head featurization
rival_b: VPD's layer-2 q/k/v subcomponents straddle heads (head-4 weight share median 0.115, max 0.74), so a one-head rescale of v changes the direction of every head-mixed subcomponent's write (kept fraction s_i, redteam-lead 2026-11-23), which predicts V x2 reads E_b over K below 0.3 at VPD while reading at least 0.9 at the neuron/head featurization, where head (2,4) is one variable
rung: premise
smallest_gap: 0.2
spread: 0.04
n: 2
design: paired
unit: featurizations per built edit (VPD and neuron/head), the same built state read at both
spread_source: experiments/rb_states/results/summary.json
spread_of: proxy: E_b over K* at VPD, seed sd of the E220 rl arm (0.16, 0.14, 0.22), the same write share on other states
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23)
stop: If V x2 reads at least 0.5 at VPD, E62's non-separation is not the featurization's spreading and the line on head-localized reuse at VPD closes; if it reads below 0.3 at VPD and at least 0.9 at neurons, the paper states that a change confined to one head reads as mostly new at VPD's head-mixed q/k/v subcomponents, and no further VPD head edits are run
sketch: experiments/head_edit/figures/sketch.png
rehearsal: cluster only: three built states on the MATS cluster debug QoS or one short pod (whip-lead cap $5), stage 0 (identity swap 0 flips, do-nothing E_b over K exactly 1) first, as E277 and E278 rehearsed at full sizes
descriptive: whip-lead 2026-11-23, after the E62 diagnosis: "It decides between a VPD featurization problem and a definition problem"; result to redteam-lead

## What runs (E280, cell P1.57)

Three built edits of M, everything else as M, on head (2,4) (E62's copy head; layer 2, head 4 = dims 512:640):
- V x2: its v_proj rows 512:640 times 2 (the head's output exactly doubled; at VPD the change sits on head-mixed v
  subcomponents, kept fraction s_i each, redteam-lead's derivation);
- O x2: its o_proj input columns 512:640 times 2 (the same function as V x2; at VPD each o subcomponent's change is
  parallel to its own output direction u_i, kept fraction the squared uncentred cosine over positions of (P v_i . x)
  with (v_i . x), redteam-lead 2026-11-23: may read high);
- Q x2: its q_proj rows 512:640 times 2 (attention logits doubled; at VPD the V x2 algebra on the q matrix, each q
  subcomponent keeping s_i, so it should read near V x2 there; at the head featurization the kept fraction is unknown).
Each read by E_b over the needed set K at VPD (E278's apparatus, E268's scan settings) and at the neuron/head
featurization (E277's apparatus), on E62's copy D and read position (the marker). Filed expected values: V x2 and O x2
at neurons at least 0.9 (exact by construction: the positive control); Q x2 at VPD within 0.15 of V x2 at VPD (a check of
the derivation, redteam-lead 2026-11-23); Q x2 at neurons reported; V x2 at VPD the open reading (rivals above); O x2 at VPD
reported. QK refiled before predictions.json (note-2026-11-23-redteam-e280-prerun.md).
