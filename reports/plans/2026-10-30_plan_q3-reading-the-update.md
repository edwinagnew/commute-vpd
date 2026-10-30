# Plan: E57, Q3, reading the update on Qwen3-1.7B: where it lands, and what re-gating means here

*2026-10-30, draft; nothing is filed, coded or run. The user moved the Qwen line to Q3 today after E44 Amendment 2
missed its teeth check (`reports/analyses/2026-10-30_analysis_e44-amend2-reading-b.md`). Parent plan:
`reports/plans/2026-10-28_plan_qwen-line-replan.md` §Q3, §7. Words as in `GOALS.md` → Terms. The smaller choices were
settled by Fable's advice, adopted 2026-10-30, after the user asked that such choices go to Fable (§10).*

**In plain words.** The pass test could not be given teeth on Qwen, so this plan stops trying to score a map and
instead describes what each saved fine-tuning update does to the decomposition. It reads, for each update, how much of
its effect comes from Γ switching existing subcomponents on and off, which subcomponents switch, whether those are the
ones Γ uses when the model thinks, and where the weight change itself sits. Everything runs on free cluster cards from
saved states; the four updates saved today can be read now, and the anchor-1 and own-sample updates wait for the
training under E45 Amendment 2.

## 1. The request, in the Terms' words

For each saved full-weight SFT δW of Qwen3-1.7B (E40's no-warm-up gold update at step 10, E45's warmed-up gold update,
the two seeds of anchor 2, and later anchor 1, the own-sample update and the self-distillation null), and with Γ and
the original subcomponents frozen:

1. **Route split at scale.** Take the least-squares split of δW. Read four CI-masked KLs to the fine-tuned model on
   the update's rows, Δ off: original or split subcomponents, under gates Γ computes from the original model's
   activations or from the fine-tuned model's. Report the re-gating share and the direct share.
2. **Γ switching.** Which alive subcomponents Γ gates on for the fine-tuned model and not for the original one, on the
   same rows.
3. **Think-mode reference.** The set Γ gates on at the unchanged model when thinking is on, read on the answer
   positions. It is compared with a length-matched no-think set and an unrelated-text set. The switched-on set's
   overlap with each is taken against same-size random sets.
4. **Where δW lands.** Norm by layer and matrix kind, effective rank per matrix, and the least-squares split's energy on
   subcomponents Γ gates on at W₀ against the random-update null.

No map is scored for faithfulness. Predictions exist only for the two anchors (§5).

## 2. Design check (`/design`)

- *Goal.* `GOALS.md` "If no simple Φ works", and D3 of the re-plan: does post-training strengthen what exists or build
  something new, read from the decomposition. Rows 1-3 give "re-gating" a concrete meaning at scale. Row 4 is the
  descriptive half of D1.
- *Nearest tried.* E54 (67M): the route split's re-gating share separated an elicited map from a cipher, 0.77 against
  0.24. E44 Amendments 1/1b: Γ's masks follow whatever activations they get, 93% of E40's update kept by masks alone.
  New here: the split's cells at scale, a random-update control for the re-gating share, and the think reference.
- *Patch or step.* A step: the replan's next item, on saved states. Not a fourth E44 amendment.
- *Decisions.* Relies on decisions rows 85-86 (reading B, V′ in readouts) and the E45 rows. The re-gating line's Qwen
  stage 1 stays out (row 94). Departs from none. E45 Job 2's H100 run is dropped (§10.4).
- *What would change next.* The random control's re-gating share decides whether row 1 can carry D3 at all. A
  validated anchor row is the instrument the GSM8K arms and Q5 are read with. With no validated row, D3 is written up
  as "VPD's readings do not separate the anchors at this scale".
- *Control first.* Stage A: the W₀ identity, the reproduction of Amendment 2's numbers and the do-nothing identity,
  alone, with a stop. Then Stage C0, the two random-change controls, alone, before any update's route split (§6).
- *Numbers.* Per state: re-gating share and direct share; switched-on count; overlap excess with the think set; split
  energy on gated-on subcomponents minus the null. Expected values are in §4.
- *Class.* An approved plan's next step (class 3, the replan §Q3) on saved states (class 1). The smaller choices
  (§10) are Fable's advice, adopted 2026-10-30.

## 3. States: what exists now, and what waits on E45 Amendment 2

| state | δW on disk | read in | depends on Amendment 2? |
|---|---|---|---|
| E40 no-warm-up gold, step 10 | `qwen_sft/nowarmup_regen/dW/step_10` | Stage C | no; it carries Amendment 2's reference values |
| E45 gold seed 0, step 100 (maths KL 0.697) | `qwen_updates/dW/gold_s0/step_100` | Stage C | no |
| anchor 2, seeds 0 and 1, step 100 (recall 0.91, 0.905) | `qwen_updates/dW/a2long_s{0,1}/step_100` | Stage C | no. Step 100 is its state under both rules: the filed rule falls back to step 100, unmatched; Amendment 2 matches at step 100 |
| KL-matched Gaussian δW; Amendment 2's in-span random change (a = 17.8) | built in-job | Stage C0 | no |
| anchor 1, seeds 0 and 1; own samples; null | not trained; T own and T null stopped at step 2 | Stage D | **yes**: their training, the format rule and the matched state are Amendment 2's |

E45 Amendment 2 is filed as drafted, with the extra save at the matching point, and runs in parallel. Another agent
writes its code (Fable's advice, adopted 2026-10-30). The think reference (Stage B) needs no update and runs now.
Without anchor 1 no anchor row can be validated, so every prediction in §5 waits on Stage D. Stages A-C are
descriptive.

## 4. The four readings

Common to all: 128 rows of 256 tokens, cut as Amendment 2 did, on E40's shared maths rows (`maths_rows.json`, W₀'s
no-think answers). The route split is also read on each arm's own E45 rows, as filed. KL means mean per-position
KL(fine-tuned ‖ masked), fp32, Δ off. g⁰ is Γ on the original model's forward. g¹ is Γ on the fine-tuned model's
forward. The split subcomponents' readouts use V′ (decisions row 86). "Gated on" means g > 0.01 at ≥ 1% of positions,
E45's filed definition. Code: E44's `Job`, `amend2.Reader`, `qwen_maps.applied`, `qwen_vpd.importances` and
`masked_logits`, and stage S's least-squares split; new code is the cell loop, the g accumulators and Stage B.

**Row 1, route split.** Cells K00 (original subcomponents, g⁰), K01 (original, g¹), K10 (split, g⁰; reading B's K_B)
and K11 (split, g¹). K0 is W₀'s own CI-masked KL on the same rows.
- *Quantity.* Re-gating share (K00 − K01)/(K00 − K11), as filed in E45. Direct share (K00 − K10)/(K00 − K11). K11 − K0
  is reported beside them. Unscored when K11 − K0 > 0.5 (K00 − K0) or K00 − K0 ≤ 0, as filed. That also covers
  K00 − K11 ≤ 0, since then K11 − K0 ≥ K00 − K0.
- *Fixed:* Γ, the original subcomponents, rows, Δ off. *Varied:* subcomponents (original or split), and the
  activations Γ reads.
- *Expected.* W₀: all four cells equal K0 within 1e-6, share undefined. Do-nothing through the state path: K10 = K00
  and K11 = K01 exactly, so the re-gating share is 1 and the direct share 0. E40 step 10 reproduces Amendment 2:
  K00 = 1.3738, K0 = 0.0696, within 1e-3 relative. K01 is expected near 0.09 and K11 near K0, from Amendment 1b,
  so **E40's re-gating share is expected at about 0.9-0.95**.
  **Two random controls** (Stage C0, Fable's advice, adopted 2026-10-30). (i) Off-span: a Gaussian δW, each site's
  Σ-weighted norm as the update's, scaled until its maths KL is within 1.5× E40's; it is also row 4's placement null.
  (ii) In-span: Amendment 2's random change to the alive U and V at a = 17.8 (`qwen_teacher.random_change`, KL ratio
  0.81, no search). A random change uses no existing machinery, so a high share means "Γ follows activations", not
  re-gating. Either at ≥ 0.8 shows no teeth. A low share from (i) alone does not show teeth, since g¹ may simply fail
  to represent an off-span change. **Teeth** only if both sit below E40's share by more than the row bootstrap, that
  is, each gap's 95% interval lies above 0. Otherwise row 1 is **weak**, and the anchors decide.
- *Blind to:* whether g¹ reads the change or copies late-layer output (Amendment 1b could not tell these apart). The
  random controls are the only guard. No constructed positive control exists at scale; on the 67M, E54's elicit map is
  the positive control (0.77).
- *Noise floor:* E45's filed floor, max(0.05 nats, 5 × W₀'s re-masked spread) in the denominator. Also a 2000-sample
  row bootstrap of each share, and the gap between anchor 2's seeds.

**Row 2, Γ switching.** S(state) is the set gated on under g¹ and not under g⁰, on the maths rows. Per-subcomponent
mean g and active share are saved for both, alive only (~2 MB each).
- *Fixed:* rows, Γ, original V. *Varied:* the activations.
- *Expected.* W₀: |S| = 0 exactly. Do-nothing: the state's own S, since S does not depend on the split. Noise floor: the
  "switched-on" set between W₀ on rows 0-63 and on rows 64-127; a state is scored only if
  |S| ≥ max(20, 5 × that count). **Positive control:** W₀ with E45's answer-template system prompt, on the same answer
  tokens. No weight moves, so all change is gating, and |S_prompt| must clear the floor. **S_prompt is anchor 1's
  reference set**: anchor 1 distils that prompt, and its overlap e_P is read in row 3. Random: |S| reported.
- *Blind to:* falls in g that stay above threshold, and changes below 0.01.

**Row 3, think-mode reference.** Three reference sets at W₀, built like S (gated on in the context, not in the plain
no-think rows), on answer positions only:
- **T, think.** The prompt plus W₀'s own think-on response, generated with the model card's thinking sampling
  (T 0.6, top-p 0.95, top-k 20), at most 2048 new tokens; read on the tokens after `</think>`.
- **L, length-matched no-think.** The same no-think answer, placed after an earlier chat turn: another problem and
  W₀'s no-think answer, cut to the think trace's token length. Same answer tokens, same absolute positions, maths text
  before, no think template.
- **C, unrelated text.** Gated on for E44's 128 FineWeb rows and not for the maths rows. These are raw web text, not
  chat (Fable's advice, adopted 2026-10-30; §9).
- **P, the prompt's set.** S_prompt from row 2, anchor 1's own reference.
- *Quantity.* Overlap o_X = |S ∩ X| for X in {T, L, C, P}, as an excess e_X = (o_X − mean)/sd over 1000 random draws of
  |S| alive subcomponents, stratified by site (same count per site as S).
- *Expected.* W₀: S empty, unscored. Positive control: the overlap of anchor 2's seed-0 S with its seed-1 S must sit
  above the 99th percentile of the draws, which shows the statistic can see a shared set. Random δW: e_T reported;
  expected near 0.
- *Blind to:* anything but set membership. T may still track think-style text that L does not match: self-checking
  tokens and the template token. The answer-position restriction and L are the guards.
- *Noise floor:* the draws' distribution; the seed pair's e_T gap; a row bootstrap of e_T.

**Row 4, where δW lands.**
- *Quantities.* (a) Share of ‖δW‖² by layer and by matrix kind. (b) Effective rank per matrix (Roy & Vetterli 2007,
  exp of the entropy of the normalised singular values). (c) E45's filed placement: the share of the least-squares
  split's Σ-weighted piece energy on subcomponents gated on at W₀ for the own rows, minus the same for
  `split.random_like`, the same Gaussian null as C0 (i). The MLP share is reported.
- *Fixed:* the decomposition and Σ. *Varied:* the update.
- *Expected.* W₀: undefined (δW = 0), not read. Random null: kind shares proportional to its Gaussian energy, and
  effective rank near full. **Positive control** for (c): E44's often-on-half state, the update split onto the often-on
  half alone. Its excess is expected near 1 minus the null's share, because the often-on half overlaps the gated-on
  set. Below half of that, (c) is reported as unable to see placement.
- *Blind to:* function. (c) is partly the solver's choice, since the least-norm split spreads any δW. Only the excess
  over the null is read.
- *Noise floor:* the null over 4 seeds; the gap between anchor 2's seeds.

## 5. Predictions: the anchors only (filed in E57's `results/predictions.json` before Stage D)

**Already filed in E45 and scored here, unedited** (`experiments/qwen_updates/results/predictions.json` →
`rules.decomposition_rows`): the re-gating share (decisive: anchor 1 ≥ 0.5, anchor 2 below anchor 1); placement
(decisive: anchor 1 above anchor 2); and behaviour kept (decisive: anchor 1 ≥ 0.8, anchor 2 below). **Behaviour kept is
scored as filed**, under reading A: (K00 − K11)/(K00 − K0), masks from the state's own forward. Reading B's
(K00 − K10)/(K00 − K0) is reported beside it, unfiled and dated; swapping them would be a refile (Fable's advice,
adopted 2026-10-30). E45's validation and t-reading rules apply unchanged.

**New, not filed in E45 ("Q3 material")** (Fable's advice, adopted 2026-10-30).
- *Prompt overlap.* Anchor 1: e_P above the 99th percentile of the site-stratified draws. Anchor 2: e_P at or below
  the 95th. Lean medium. Unscored if |S_prompt| does not clear row 2's split-half floor.
- *Think overlap.* Anchor 2: o_T at or below the draws' 95th percentile, "at chance". Lean medium. Anchor 1's e_T is
  reported, unfiled. Anchor 1 is a format template, so §7's reason to expect it above chance no longer holds.
- *Validated*, for each row, when both anchors are scored on both seeds and the gap of anchor means strictly exceeds
  both the larger within-anchor seed gap and 2 × the gap's row-bootstrap s.e. A validated gap in the wrong direction is
  announced as a reversal.
- Every anchor reading carries the domain confound: "anchor 1 shares its domain, anchor 2 does not".

**Edge cases, run against the rule before filing.**

| case | outcome |
|---|---|
| \|S\| < max(20, 5 × the split-half count) | that state unscored, never "at chance" |
| \|T\|, \|L\| or \|C\| < 20 | the think row unscored for every state; reported |
| \|S_prompt\| below row 2's floor, max(20, 5 × the split-half count) | the prompt row unscored |
| a draw sd of 0 (S fills every alive slot of its sites) | that e_X undefined; unscored |
| e_P between the 95th and 99th percentiles | anchor 1 and anchor 2 both miss |
| o_T above the 95th percentile for anchor 2 | anchor 2 misses |
| an anchor seed unscored | row not validated; single-seed observation |
| a think row whose trace does not close `</think>` in 2048 tokens | dropped from T; > 25% dropped → stop at Stage B |
| NaN or missing | miss (E45 `general`) |

## 6. Control first, and early stops

1. **Stage A, alone.** First, W₀ as the "fine-tuned" model: four cells equal within 1e-6 and |S| = 0, or stop.
   Second, E40 step 10 on the first 16 rows reproduces Amendment 2's K0 (0.0655816) and K(D) (1.4956749), each within
   1e-3, or stop. Third, do-nothing through the state path: re-gating share 1 within 1e-6, or stop. Then GPU peak after
   row 0 must be ≤ 44 GiB, or stop.
2. **Stage C0, alone, before any update's route split.** The two random changes' re-gating shares. A miss does not
   stop the descriptive rows. If either reads ≥ 0.8, row 1 is reported as "no teeth at scale" and a card goes to the
   user before Stage D. The teeth verdict waits for E40's share in Stage C (§4). Filed rows are still scored as filed.
3. **Stage B first minutes.** Trace drop share after the first batch, under 25%. Γ's GPU peak on the longest row, under
   44 GiB. A miss on either stops the job, with the measured length written into this plan.

## 7. Stages, cheapest first (all MATS cluster L40, $0; job IDs recorded)

| stage | what | QoS | ~time | needs |
|---|---|---|---|---|
| A | controls (§6.1); W₀ split-half floor; prompt positive control; row 4 (a)-(b) on the 4 saved δWs and the null | debug | 25 min | nothing |
| B1 | think-on generation, 128 problems, saved as jsonl, resumable | debug | 20 min | A |
| B2 | T, L, C sets; memory and length measured | debug | 25 min | B1 |
| C0 | Gaussian δW: KL search (≤ 6 evals, 16 rows), split, four cells, S; in-span random change at a = 17.8: four cells, S | normal | 50 min | A |
| C | E40 step 10, gold step 100, anchor 2 seeds 0/1 step 100: split (~20 min each, Amendment 2's measured S time), four cells, S, placement | normal, 3 h limit, resumes per state | 2.3 h | C0 read |
| D | anchor 1 seeds 0/1, own samples, null at their matched states; then scoring on CPU | normal | 2.3 h | E45 Amendment 2's Job 1 |

Total about 6.7 h of L40 time, $0. If `normal` is not empty at submission, C and D go to RunPod H100 at ~$4/h,
about $9 each, which is under the gate. Split pieces are not saved (a rerun recomputes them in ~20 min). The g vectors
and every cell's per-row sums are saved (~50 MB in all), to `results/large/` via a project path.

## 8. Gated?

No, and neither is E45 Amendment 2 (Fable's advice, adopted 2026-10-30). A "run" in the gate rule is one Slurm job.
Each job here is $0, at most 3 h, saved per state, and inside the session cap. The H100 fallback for C or D is about
$9 per job, also under the gate. The pre-mortem is filled anyway:

| heading | risk | guard |
|---|---|---|
| recipe, data | none trained; rows from saved files | sha256 of every δW against the saved sums, Stage A |
| think data | traces run past 2048 tokens; Γ's memory on ~2,300-token rows unmeasured | the drop rule and the row-0 peak stop (§6.3) |
| precision | fp32 cells as Amendment 2 | the do-nothing identity at 1e-6 |
| memory | host 74 of 80 GB in Amendment 2 | ask `--mem=96G`; log the host peak after the first state |
| time | split 20 min per state, measured | per-state resume; 2 h 45 min wall exit as `amend2.wall` |
| saves, downloads | none large; cluster NFS | per-state JSON written as it finishes |
| controls | none constructible for row 1 at scale | two random controls first; E54 named as the 67M reference |
| predictions | degenerate overlap rule | §5 edge cases tested on CPU before filing |
| never on a GPU | the cell loop, the g accumulators, Stage B | Stage A runs the cell loop on W₀ and E40 first |

## 9. Departures from the literature

- **Masks from another model's activations (K10, K00 for a changed model).** VPD computes CI from the model being
  decomposed. Here the K10 cell uses the original model's activations with V′ readouts (decisions rows 85-86). K01
  and K11 are VPD's own convention, applied to the fine-tuned model.
- **The route split and its shares are this project's** (`GOALS.md`, "Where an update goes"). No source defines them.
- **"Gated on" adds "≥ 1% of positions"** to VPD's alive threshold, g > 0.01 somewhere. This is E45's filed rule.
- **Think reference.** arXiv 2610.14270 locates a think subspace in activations by swapping it. Here the reference is
  Γ's gated set, and L and C are ours. The model card's thinking sampling is used as written. The 2048-token cap
  departs from its recommended 32,768 output length, and rows that do not finish are dropped.
- **Unrelated chat text becomes FineWeb rows**, already read in E44. C is raw web text, not chat, and the write-up
  says so. This departs from the replan, not a source; it needs no new dataset.
- **KL to start** is per-token KL(W₀ ‖ W₀ + δW) on teacher-forced rows. RL's Razor measures it on the model's own
  samples of the new task.
- **Effective rank** is Roy & Vetterli's, on the raw δW, not input-weighted.

## 10. Choices settled: Fable's advice, adopted 2026-10-30

The user asked that smaller choices go to Fable rather than to them (2026-10-30). Each item below is also a row in
`decisions.md`.

1. **C0 has two random controls**, off-span and in-span. Teeth need both below E40's share beyond the row bootstrap
   (§4, row 1).
2. **Not gated.** A run is one Slurm job; each here is ≤ 3 h, $0 and saved per state. The same holds for E45 Amendment 2.
3. **E45 Amendment 2 is filed as drafted**, with the extra save at the matching point, and runs in parallel.
4. **E45 Job 2 is dropped.** Its placement and behaviour-kept rows come from Q3's C and D cells. Its least-squares
   pass row is "not run: the pass test has no teeth (E44 Amendment 2)".
5. **Behaviour kept is scored as filed**, under reading A, with reading B beside it, unfiled (§5).
6. **Anchor 1's filed overlap is with S_prompt**, not the think set. Anchor 2's think overlap stays "at chance". Anchor
   1's e_T is reported, unfiled (§5).
7. **FineWeb rows are the unrelated set**, named as raw web text (§9).

## 11. Decision tree

- **Stage A fails** → stop and report; nothing else runs.
- **Either C0 random share ≥ 0.8** → row 1 has no teeth at scale. D3 rests on rows 2-4. Card before Stage D.
  **Both below E40's share beyond the row bootstrap** → row 1 has teeth. **Otherwise** → weak, and the anchors decide.
- **T ≈ L** (e_T ≤ e_L on the updates, or |T ∩ L| ≥ 0.8 |T|) → the think reference reads position and maths context,
  not thinking. Row 3 is reported as that and is not used for D3.
- **After D, a validated anchor row** → read gold, own samples, the null and E40 on it with E45's t rule. Then write
  up D3, and Q5's GRPO update goes through the same rows.
- **No validated row** → the write-up says VPD's readings do not separate the anchors at this scale. Card: the
  domain-matched anchor pair (`backlog.md`, 2026-10-28) or close D3.
- **Anchor 1 not learned or not matched under Amendment 2** → no anchor prediction is scored. Q3 ships rows 1-4 on the
  four saved states as description, and a card goes to the user.

## Not in this plan

Any faithfulness score of a map; E45's least-squares pass row (not run: the pass test has no teeth, E44 Amendment 2);
the 20-step adversary; the re-gating line's Qwen
stage 1 (decisions row 94); E40 steps 1 and 100 and gold's earlier states, unless a row needs a dose check; saved
split pieces; a constructed positive control for the route split at scale.
