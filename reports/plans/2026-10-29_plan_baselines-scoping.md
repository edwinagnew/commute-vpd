# Plan: which simpler methods must VPD's teach-vs-elicit reading beat? (scoping; B1 = E51)

## The question, in plain words
The project aims to show that a VPD decomposition can tell whether a post-training update teaches the model something
new or brings out (re-weights) something it already had. The obvious objection is that a simpler method, working on
neurons, activations or the weights alone, would have told us the same thing more cheaply, as probes often have in
pragmatic interpretability. This plan lists the baselines a reviewer would demand, says how each gives a verdict on one
update, and proposes the order to run them in. The user picks which to run (task file, 2026-10-29); B1 below is a
$0 small test the task allows now.

The user's request restated in `GOALS.md` → Terms' words: for every update with a known kind, compare VPD's readings of
where the update goes (the route split's re-gating and masked-off shares, behaviour kept under CI masks, the outside
share) against the same verdict from non-VPD methods, scored by the same rule.

## Design check (`/design`)
- **Goal served.** `GOALS.md` → "If no simple Φ works … teaches new capabilities or strengthens existing ones". A
  VPD reading of an update's kind is only worth claiming if it does better than, or differently from, what cheap
  methods give.
- **Nearest tried.** E12 (`reports/analyses/2026-10-20_analysis_new-vs-existing-signatures.md`): only the route split
  and behaviour kept under CI masks separated copy from cipher; the outside share did not. E38: Γ's switched-on set did
  no better than a gradient ranking that uses no Γ. That was the first baseline comparison, and Γ lost it. Nothing
  has compared VPD with neuron, activation or behavioural methods.
- **Patch or step.** A step: it decides whether the VPD reading is a claim at all.
- **Decisions relied on.** Qwen line's E45 anchors and their rule that a row counts only if the gap between anchors
  exceeds the seed spread (`reports/plans/2026-10-28_plan_qwen-line-replan.md`). No departure.
- **What would change what we do.** If a cheap baseline separates every labelled pair that VPD separates and fails
  nowhere VPD succeeds, the teach-vs-elicit claim must move to what VPD names (which subcomponents, which mechanism),
  not to the verdict. If baselines fail somewhere VPD succeeds, that case becomes the project's headline example.
- **Class.** B1 is a small $0 test the task file allows. B2–B5 need the user's choice.

## The central design problem: the labelled set is too easy
On the 67M model the only clean labelled set is E12's: the copy tasks (ident32, ident32b, elicit) against the random
ciphers (rand32, rand32b, teach), two seeds each. These differ at the level of behaviour: the original model gives
the cipher's target about 1e-6, and an induction-style copy is something the model can do somewhere. A behavioural or
steering baseline will very likely separate them, so a win there tells a reviewer nothing about VPD's added value.
Three additions make the comparison able to show a difference:
1. **The null update** (E36's fine-tune that learned nothing, re-gating share 0.82): every method must read it as
   "nothing changed", not as "elicit". VPD's route split already fails this; a baseline that passes it beats VPD here.
2. **A matched random update** of the same size per matrix: the floor for every weight-space and activation number.
3. **A mixed case** where the behaviour is new but built from existing parts (the known limit: VPD reads it as
   "elicited"). No such update exists yet; designing one is a separate `/design` item (not proposed here).
On Qwen, E45's anchors (context distillation = elicit; invented facts = new content) are the second set once trained;
their domain confound (maths against entity recall) affects every method alike.

## The candidates (details and citations: `lit_review/2026-10-29_baselines_notes.md`)

| # | Baseline | Verdict for one update | VPD reading it rivals | 67M cost | Qwen cost |
|---|---|---|---|---|---|
| B1 | **Weight-space statistics** of δW against W₀: share of δW's energy on W₀'s top singular directions, intruder dimensions (Shuttleworth et al. 2024), effective rank, LoRA amplification factor (Hu et al. 2021 §7) | on W₀'s main directions, no intruders = re-weighting; intruders or energy off them = new | outside share | $0, CPU, minutes | $0, cluster CPU |
| B2 | **Neuron gain-patching** (the already-clean objection): rank MLP neurons, both signs, by how much their activation at the trigger shifts under the update; patch the top-k tuned activations into the original model | a few neurons already active at W₀ recover the gain = elicit; it needs neurons idle at W₀, or nothing small recovers it = teach | route split's re-gating share | one debug L40 job, $0 | < 1 GPU-h |
| B3 | **Steering the original model** (Venhoff et al. 2025; Ward et al. 2025; CAA): add the mean tuned-minus-original residual difference at the trigger to the original model, no weight change | share of the behaviour gain recovered | behaviour kept under CI masks | one debug job, $0 | < 1 GPU-h |
| B4 | **Behavioural elicitation**: pass@k and in-context prompting of the original model (Yue et al. 2025; Lin et al. 2023) | the original model reaches the behaviour by sampling or prompting = elicit | the label itself (ground-truth check) | minutes | sampling only |
| B5 | **SAE / activation dictionaries**: a base-model SAE's reconstruction error on tuned activations and the number of latents carrying the gain (Kissane et al. 2024), or a BatchTopK crosscoder with latent scaling (Minder et al. 2025) | gain carried by latents shared with the original model = elicit; tuned-only latents, confirmed by latent scaling and used causally = teach | re-gating vs masked-off | no SAE exists for the 67M: training one ~1 h on RunPod, ~$3 | public Qwen3 SAEs unchecked; a crosscoder on one layer ~5–20 H100-h, $20–80 (gated) |
| B6 | Linear probe for the target on W₀'s activations | decodable before training = existing; necessary, not sufficient | — | minutes | minutes |

**On the already-clean essay** (Cammarata and Tigges, Goodfire, 15 Oct 2026): it argues neurons are mostly
interpretable already (about 90% of neuron signs in Qwen3-32B have one main job in a small audit) and that dictionaries
mostly helped in small models; parameter decompositions are not mentioned. Responses (Nanda, Hilton) are only in
press summaries so far. Its fair demand on us is B2 on Qwen3-1.7B's SwiGLU neurons, both signs. On the 67M the essay
itself concedes that dictionaries help more in small models.

## Recommendation
Run B1 now on E12's eight saved states plus a matched random update ($0, filed predictions, below). Then, if the user
agrees, one 67M debug job for B2 + B3 + B4 on the same eight states plus the null update: all at $0, and they answer the
two strongest objections. B5 follows only once B2–B4 are read: on Qwen if public SAEs exist (a $0 check), otherwise one
67M SAE on RunPod. B6 is cheap, but it can only confirm the label, so I would skip it unless a reviewer asks for it.
Every method is scored by the E45 rule (the gap between kinds exceeds the spread between seeds) and placed in one
table beside E12's VPD rows.

## B1 = E51: weight-space statistics on E12's states (running now)
- **States.** `~/commute-data/results-large/regating_table/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt` (E46's regenerated E12
  fine-tunes; format to be checked by the writer) against W₀ from the 67M target model; a matched-norm Gaussian update per
  matrix as the null.
- **Numbers (per state, aggregated over the 24 matrices weighted by ‖δW‖²):** top-k energy share; intruder count in
  W′'s top singular vectors; effective rank of δW. Writer fixes k and the intruder threshold from Shuttleworth et al.'s
  settings and states them.
- **Control.** A synthetic pure re-weighting δW = ε·(top singular directions of W₀) must read top-k share ≈ 1 and 0
  intruders; the random null must read the share expected for its size (≈ k/d); else the code is wrong.
- **Prediction.** Each statistic separates the kinds (the gap between the copy and cipher means exceeds the larger
  within-kind seed spread) or not. Lean: no separation for top-k share and effective rank, medium, because the outside
  share (a projection onto the decomposition's span) did not separate in E12; intruders unknown, low.
- **What it decides.** If a weight statistic separates as well as the route split, the VPD reading must show what B1
  cannot (behaviour, which parts); if none separates, weight-only methods are answered and B2–B5 remain.

## Not run
B2–B6; the mixed case; Qwen anything (E45's anchors not yet trained).

## B1 result (E51, 2026-10-29, `reports/analyses/2026-10-29_analysis_baseline-weights.md`)
The top-10 energy share and the intruder count do not separate copy from cipher (no state has an intruder at this
update size). Effective rank separates by a hair (gap 24.5 against a seed range of 23.9), against the filed lean, but it
falls with ‖δW‖ across all nine states and every cipher update is larger, so this set cannot tell size from kind. The
learned-nothing update reads off the null on every statistic. Next for B1, if wanted: a size-matched copy/cipher pair.

## B2 + B3 + B4 = E52 (user, 2026-10-29: "go with B2, B3 and B4")
One job on the cluster's `debug` QoS (L40, ≤ 30 min, $0), through a `pod-runner`. States: the eight E12 fine-tunes and
`code_step10` (the fine-tune that learned nothing; its own task's held windows) in
`~/commute-data/results-large/regating_table/`, each applied to W₀ as W₀ + δW; task windows from `windows_<task>.pt`
(each task's held-out windows are read; the steering vector is fit on one half of them and scored on the other).
For each state, the **gain** G = mean over held windows of log P_tuned(target) − log P_W₀(target) at the answer position.
Every recovered share below is (log P_method − log P_W₀) / G. Rule for small gains: if |G| is below the floor (the
larger of 0.05 nats and twice the seed difference of G within its task), the method's verdict is "no change" and its shares are
reported but not scored.

**Stage 0: the controls (first, alone; stop on a miss).**
| check | must read |
|---|---|
| identity patch: patch W₀'s own MLP activations into W₀ | ΔlogP = 0 (≤ 1e-4 nats, fp32) |
| full patch: patch the tuned model's residual stream into W₀ at the last layer's output | share 1 (± 1e-3) |
| G for ident32_s0 and rand32_s0 | > 1 nat (E12's fine-tunes learned both tasks) |

**B2, neuron gain-patching.** Each MLP hidden neuron (all four layers) gets a signed shift: the mean over the fit
half of the windows of its tuned activation minus its W₀ activation, at the answer position. Neurons are ranked by
|shift|. The tuned activations of the top k (k = 1, 4, 16, 64, 256, all) replace W₀'s at every position, and the share
is read on the scoring half. Numbers: k₅₀, the smallest k on the grid with share ≥ 0.5 (∞ if none); the idle share, the
fraction of the top 16 whose W₀ activation at the answer position is under 10% of that neuron's 99th-percentile
|activation| on Pile windows (idle at W₀ = recruited). Reading: elicit = small k₅₀ and low idle share; teach = large or
∞ k₅₀, or a high idle share.
**B3, steering.** v_l = mean over the fit half of (tuned − W₀) residual stream at the answer position, at each layer's
input (l = 0…3); W₀ + v_l at that position, scored on the other half; the layer is chosen on the fit half. Number: the
steering share S. Reading: a single vector can push a mode (copy the key) but cannot encode a 32-word cipher, so elicit =
high S, teach = S near 0.
**B4, prompting the original model.** W₀ with n demonstrations of the same task in the context (marker, key,
target), n = 0, 4, 16, the query's key never among the demonstrations. Number: the prompting share at n = 16. Reading:
the original model performs it in context = elicit. This is the check on the labels themselves.

**Predictions** (each separates when |mean(copy) − mean(cipher)| exceeds the larger within-kind range over its four
states, as in E51):
- P1 (B2): k₅₀ separates, copy lower. Lean: low; neurons are shared across the tasks' answers and the cipher may use
  as few neurons as the copy.
- P2 (B2): the idle share separates, cipher higher. Lean: low.
- P3 (B3): S separates, copy higher. Lean: medium, for the reason above.
- P4 (B4): the prompting share at n = 16 separates, copy higher. Lean: high; the labels were built this way.
- P5 (null): `code_step10`'s |G| falls under the floor, so every method reads it as "no change". Lean: high. VPD's route
  split gave it a re-gating share of 0.82 regardless; a method that reads "no change" beats VPD on it.
Blind spots: B2 patches activations, not weights, so it cannot separate an MLP change from an attention change upstream;
B3 is blind to input-dependent changes by construction; B4 is blind to anything not expressible as in-context examples.

**Decision tree.** P4 fails → the labels are not what we think: stop and tell the user. P1–P3 all fail while P4 passes →
the neuron and activation baselines do not separate the kinds where VPD's route split does: the first point in VPD's
favour, write it up. Any of P1–P3 passes → that baseline separates as well as the route split: write it up as the
objection holding on this set, and propose the size-matched pair and the mixed case (new `/design` items).
