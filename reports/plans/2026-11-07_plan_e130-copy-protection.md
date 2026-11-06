# Plan: is teacher training's protection a constant of the recipe? (E130)

Design, rule and leans are Fable's, adopted unchanged: `~/commute-data/scratch/regating-lead/fable-e128/advice.md` →
"The one run" (written as "E128"; the claimed code is E130). Code `experiments/copy_protection/run.py`, filing
`experiments/copy_protection/results/predictions.json`, job `cluster/copy_protection.sbatch`.

## The question, in plain words
E127 Amendment 2 found that teacher training puts a copied function where E34's adversarial check sees about a third
of its damage: the copy's own weight change, moved into Δ, reads 2.8-4x the excess over W₀ that the copy reads on the
subcomponents (m = 0.36 standard copy, 0.24 rank-8 LoRA copy). That rests on two copies. E122 made two more lowercase
copies by the same recipe (Adam 1e-5, batch 8, 1,000 steps, from W₀'s U, V): cell 1 copies the Adam fine-tune at step
20, cell 2 the rank-8 LoRA at step 2,400. Reading each one's change in Δ says whether the division is the recipe's or
depends on what is copied.

**Claim.** The division is the recipe's: m ≤ 0.5 for every copy trained from zero. **Null.** It depends on the
function: cell 1 m ≥ 0.8, cell 2 recipe-like.

## Design check (`/design`)
Re-gating line's next step inside the user's 2026-11-06 task, gap (a); approved classes 1 and 4 (GPU pass of the
existing harness on saved states). Not gated: debug QoS, ~13 min, $0. Nearest row: E127 Amendment 2 (two copies);
this adds two copies of different functions on the same rows, card and seed.

## States (each read by E34's check as amend2.py: Pile val rows 0-255, two 128-row blocks, seed-0 PGD, 20 steps; ratio over this job's W₀)
| unit | state | file |
|---|---|---|
| W₀ | released model | — |
| c1, c2 free | target W₀ + S_c, alive (U′, V′) | `damage_crossed/results/large/tt_c1.pt`, `tt_c2.pt` (sha256 checked against `damage_crossed.json` `copies.tt_c*.sha256`) |
| c1\|delta, c2\|delta | target W₀ + S_c, subcomponents at U, V | same files, `harness.at(ctx, S_c, None)` |
| cosine (report only) | cos(S_c, own fine-tune), cos(S_c, ft_adam300), input-weighted and Frobenius | `ft_c1.pt`, `ft_c2.pt`, `ft_adam300.pt` |

S_c = U′V′ᵀ − UVᵀ from the bf16 offsets (`saved_states.offsets_state`), the same object for both reads. The only copies
of `tt_c1.pt`/`tt_c2.pt` are in `damage_crossed/results/large/` (none under `ls_start_faith/`; checked by `find`).

## Stage 0: controls (first; any miss stops the job, no rerun)
| check | must read | source |
|---|---|---|
| W₀ mean block KL | 0.69174 ± 1e-3 rel | `ls_start_faith_amend2.json` controls.W0; tolerance amend2.py `repro_tol` |
| c1 free ratio | 2.8418 ± 1e-3 rel | `damage_crossed.json` score.ratios["tt_c1\|A"] |
| c2 free ratio | 1.5341 ± 1e-3 rel | score.ratios["tt_c2\|A"] |
| card, alive U, V, file sha256 | L40; equal to E121's `uv0.pt`; equal to E122's | — |

E127 A1/A2 reproduced every lowercase copy exactly from the same kind of bf16 file on this path.

## Expected values (Fable's leans)
| number | W₀ | do-nothing | control | claim | null / lean |
|---|---|---|---|---|---|
| c1\|delta ratio | 1 | 3.409 | free 2.842 | ≥ 4.68 (m ≤ 0.5) | 2.7-2.9, m 0.95-1.1 |
| c2\|delta ratio | 1 | 3.739 | free 1.534 | ≥ 2.07 | 2.4-3.1, m 0.25-0.4 |
| cosine to own fine-tune | — | 1 | — | no lean | E121's from-zero copies 0.38-0.40 |

## Filed rule
f = 0.20878: the L40 W₀ PGD floor over adversary seeds 0-2, max/min − 1 (`teacher_training_amend8c.json` W0.pgd_floor;
`harness.pgd_floor`), as E127. m_c = (free − 1)/(in Δ − 1), readable only when both excesses ≥ f. Per copy: m ≤ 0.5
"recipe-like", m ≥ 0.8 "no protection", between "partial". Headline: both recipe-like → "recipe constant"; any copy
≥ 0.8 → "function-dependent"; else "partial"; suffix "regularity broken" when a copy's in-Δ ratio exceeds its own
do-nothing. Descriptive: in-Δ excess over the copy's do-nothing excess (known copies 0.67-0.81).
Leans: c1 no protection 0.6 / partial 0.25 / recipe-like 0.15; c2 recipe-like 0.7; headline function-dependent 0.55.

**Edge cases** (run on the CPU worker, `scratch/regating-lead/e130-writer/edge_check.out`): an excess below f, zero or
negative → that copy "unreadable" (writer's completion: with no copy at ≥ 0.8, an unreadable copy makes the headline
"unreadable", not "partial"); ties at 0.5 → recipe-like, at 0.8 → no protection; free = in Δ → m 1; non-finite →
invalid; a control miss → the job stops with `.stopped`, the claim stays at two copies.

## What each number is blind to
One adversary seed and one 20-step local search; m says how much the placement hides, not why. One task, one recipe.
c2's free excess (0.534) is 2.6 f, so its m is coarse (±f per read gives 0.16-0.47 at the lean). The cosine has no rule.

## Stop rules and time
Controls as above; exit 3 if a unit would end past 28 min (`fleet/resubmit --code 3`); a resubmission rereads W₀'s
first block within 1e-6. Five PGD reads at 127-131 s plus the cosine (<1.5 min) and start-up: ~13 min, peak host RSS
~6-7 GB (`--mem 16G`), debug QoS, $0. Every scored number goes to `results/large/run.log` with `STAGE n DONE`.

## Not run
More adversary seeds or steps; held masks; other tasks (emoticon copies have ~0.005 nats free excess); any new copy
or recipe; the CI-weighted placement read (Fable (i), ruled out at $0); a masked term in teacher training (Fable (ii)).
