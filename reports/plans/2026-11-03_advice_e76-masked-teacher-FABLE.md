# Advice: E76, teacher training with masks in its loss (Fable's advice, 2026-11-03, read-only)

*Read: `decisions.md` (last row), `GOALS.md` → Terms, E74's plan, code and u1/u3 JSONs, E73's advice, E72's
`score.json`, `src/commute/qwen_teacher.py`, `experiments/qwen_sft/amend1.py` and its rehearsals, the E40 plan's gates
and rate-scaling advice, the decomposition's `launch_config.yaml`, the RL lead's note. Nothing run; jq on the per-row
files, "per position" = sum over 32 rows × 256 tokens / 8192.*

**In plain words.** The masks at and after a change kill it, so a student trained to match the fine-tuned model under
masks can only do it by routing the behaviour through switched-on subcomponents, mostly in the last blocks: a new
mechanism, not the update's. The run is worth doing as a cheap paired job whose readings tell "re-implemented late"
from "placed"; E74's numbers say the first is the likely outcome. One $0 L40 job, about two hours, control first.

## 1. The loss

VPD's own fit (`artifacts/qwen3-1.7b-p-e9fcffd5/runs/p-e9fcffd5/launch_config.yaml`, `loss_metrics`): faithfulness
(coeff 1000), one reconstruction term (coeff 1; stochastic masks m = g + (1 − g)u, Δ share d ~ U(0, 1), 90% of samples
adversarial by pooled persistent PGD, a subset of sites per sample, hidden-activation auxiliaries at coeff 2),
importance minimality (6.4e-6), nonlinearity locality (3e-6); Γ and components trained together, Muon, batch 2048,
100k steps. The reconstruction target is the clean output of the model being decomposed.

| candidate | reading | verdict |
|---|---|---|
| A. KL(FT clean ‖ student clean), the filed recipe | never run on a real Qwen update at any size (`results/amend1/` holds only stage L); the control fell to 0.72 of its start in 100 steps at 16 columns, 0.79 at 384 | keep as the null arm |
| B. A + w · KL(FT clean ‖ student under stochastic masks from Γ on W0's clean forward with W0's V, Δ share d, Γ frozen) | VPD's reconstruction term with the adversary off and the subcomponent sum frozen at W0's; the student's offsets enter masked on their alive columns and through d · x changeᵀ | **recommended, w = 1** (both terms start near 1.4-1.6 nats on u1, §2) |
| C. B with Γ read on the student's V each step | the consistent VPD choice, but Γ bf16 is 17.1 GB resident (E74's reader: 45.6 GB with fp32 components), so not an L40 job; and it lets the student move g, mixing a re-gating map into the reading | paid card only; read g's move at the end instead |
| D. KL(FT masked ‖ student masked) | under W0's masks FT masked sits 0.01 of its clean effect from W0 masked (τ, §2): the target is W0; under FT's own masks every state reads 0.103-0.105 (E72 `score.json` u1 T1, E44 Amendments 1, 1b): the masks carry it, the weights nothing | out, degenerate both ways |
| E. Γ trained too | a refit: a new decomposition of FT, not a map of this one; `GOALS.md` → Terms fixes teacher training with Γ and Δ frozen | out |

Departures of B from VPD's fit, to state: no adversary (the 90% PGD share), Γ frozen, only the alive cut of U, V moves,
the faithfulness term dropped (with Δ frozen at Δ₀ it reads ‖change − δW‖², the least-squares objective; report it per
block instead), no hidden-activation auxiliaries, Adam at tiny batch, masks from W0's forward not the student's.

## 2. Feasibility against E74's numbers (`experiments/qwen_transmission/results/u1/`)

Per position on u1's maths rows: the whole update moves the clean model by 0.626 nats (`all.json`) and the masked model
by 0.0034 under fixed masks (τ 0.0055) and 0.0080 under stochastic masks (τ 0.0128, `stochastic.json`); the masks' own
damage K0 is 0.067 fixed, 0.044 stochastic. Per block under stochastic masks, τ for δW / for a Gaussian of equal clean
effect: block 0 3.8e-5 / 5.0e-4, block 7 7.6e-6 / 4.8e-4, block 14 4.5e-5 / 1.8e-3, block 21 6.1e-3 / 0.048, block 27
0.65 / 0.64. δW's clean effect sits early (0.025, 0.048, 0.034 nats at blocks 0, 7, 14; 0.004, 0.003 at 21, 27). KL goes
as amplitude squared, so a change the masked forward passes as the clean forward passes δW needs amplitude 1/√τ: 1.25×
in block 27, 4.6× in block 21, 24-45× in blocks 0-14 for a generic direction, more for δW's own. The student will put its
change where it transmits, blocks 21-27, re-expressing an early update as a late one (`direct.json` agrees: δW's clean
effect reaches the logits through later blocks' computation, share 1.4e-6 by the skip alone at block 0). Feasible: a
masked student whose logits match FT. Not feasible on these numbers: one whose change is where δW is.

Three readings make the result no longer a map, each on the fixed rows every 50 steps: (i) clean KL(FT ‖ W0 + change)
above the remainder's 0.53 (E72 `score.json` u1 T2; placements 0.03-0.09, random 1.38); (ii) clean KL(W0 ‖ W0 + change)
above 1.5× δW's 0.626 (size, `GOALS.md`'s ruler); (iii) the share of ‖change‖² in blocks 21-27 more than 3× that share
of ‖δW‖² (arm A below gives the floor). The first-steps check is (iii) at steps 50 and 100 beside the masked term's fall.

## 3. The smallest run

**Card.** A cluster L40, $0, about 200 columns per site: the 480 estimate (`amend1.py` docstring, 34.8 + 0.0218 GB per
column) parked the decomposition on the host, and the masked term needs U, V resident (bf16 9.8 GB) over the 37.0 GB
peak at 384 (`amend1_rehearse_e40a1_r3_c384/B.json`). The columns must be the often-on ones by mean g
(`qwen_maps.halves`'s ordering), not `Job.alive`'s first-by-index cut (`qwen_sft/run.py`, `nonzero()`): the masked term
acts only through what is on (~38 per site per position, E73). Stage 0 reads the share of Σg they hold; no guess. The
paid 141 GB card is the second run: its full-alive-set rate is unmeasured and the filed grid should abort
(`reports/reviews/2026-10-28_review_e40a1-rate-scaling-fable-advice.md`, boundary 2.5e-7 to 7e-7), its memory with the
components resident ~92 GB unmeasured; $10-20 per update at $4.5/h after a $1.5 fetch and a $1.5 rate grid.

**Update.** u1 (E40 step 10): every control number is on its rows. RL's step 100
(`~/commute-data/results-large/qwen_rl/lr1e-6w10/dW/R/step_100.safetensors`, 5.64 GB fp32, KL 0.194 per token, E69
`R.json`) is D1's real update and the scored second job once it has a clean table; u3 (τ 0.0004 / 0.0011) can wait.

**Arms, in order, ≤ 600 steps each at 1e-6 (safe at 96 and 384 columns: falls 0.738, 0.788 at step 100, `_c96/B.json`,
`_c384/B.json`), 2 × 256 tokens per step, masks drawn fresh per step from g cached per row (bf16, 381 MB per row over
745k columns; 64 rows 24 GB host).** Stage 0, fp32 components and Γ as E74's reader: K0 on E74's 32 rows at seed 74
reproduces 0.0440 within 10% (stop); the same with bf16 components on 8 rows gives the dtype floor; cache g; free Γ.
C, control: teacher W0 + `random_change` at 0.01 nats under B's loss; both terms fall, band set from this reading (the
filed 0.5 was never reached, 0.53 at 600 steps, `_c16/rehearsal.json`); the masked term not falling is a gradient bug,
stop. A, null: u1 under the clean term alone. B: u1 under both. Every 50 steps: both KLs on 8 fixed rows, clean KL from
W0, the per-block profile; at the end the student under its own masks (Γ on its forward and V: gain as E68's reader,
and the share of g that moved).

**Time.** Clean steps 0.70-0.82 s at 2 × 128 tokens (`_c96`, `_c384` run.logs); E74's masked forward ~0.17 s per
256-token row (896 in 151 s, `run.log`); so ~2.5 s per step with the backward, ~30 min per arm with readings, stage 0
~10 min, load 6 min: ~1 h 50. Two `debug` jobs (stage 0 + C + A; B from the cached g) or one `normal` job if the queue
is empty. Far under the $60 / 6 h gate.

## 4. How the result is judged

Do-nothing: its placed part is nothing, so the clean table reads W0, 1.635 from FT (u1); the masked term at step 0 is
KL(FT clean ‖ W0 masked), expected near the fixed-mask 1.374 (E73's advice); under its own masks its gain is 0.240 at
K 0.104 (E72 u1 T1), the bar to beat by more than E72's floor w = 0.0075. The positive control is C. "Works" is all of:
B's masked term ends below A's by more than the dtype floor and C's spread; B's clean KL from FT within the placements'
band (≤ 0.09) and from W0 within 1.5 × 0.626; late-block share within 3× δW's; gain under its own masks above 0.240 + w.
"Fails as a map" is (i)-(iii) of §2. "Fails to learn" is B's masked term within the floor of A's at 600 steps: the
finding "the switched-on subcomponents cannot carry the update even when trained to", and the 67M (E34's harness, where
masked tests have teeth) is the cheap place to learn why. Anything between is reported as read; no amendment on it.

## Recommendation, and what would change it

Run B against A with C first, on an L40 with ~200 often-on columns, on u1, $0, ~2 h; file the three "not a map" rules
and set the control's band from stage 0 before the arms. Expected from E74: the masked term falls little, or by a
late-block change that rule (iii) catches. What would change this: a stage-0 coverage under ~50% (the L40 cut cannot
carry the on set; the paid card goes first, with its rate grid); or B's masked term falling at step 50 with a block
profile like A's (then the paid card on RL's step 100 is the next job and the $10-20 is justified).

**Numbers relied on.** `qwen_transmission/results/u1/{all,stochastic,blocks,direct,stream_gap,sharpness}.json`;
`qwen_fixed_masks/results/score.json` (u1 T1, T2; u3 T2); `qwen_sft/results/amend1_rehearse_e40a1_r3_c{16,96,384}/{B,rehearsal}.json`
and run.logs; `qwen_sft/results/predictions_amend1.json` (B_rate, T0); `qwen_rl/results/lr1e-6w10/R.json`; the launch
config above.

**Not read.** The share of Σg the top-200 columns hold; bf16 against fp32 under masks; any masked step time with a
backward; the full alive set's rate; RL's update under any masked read; the 67M with masks in its loss.
