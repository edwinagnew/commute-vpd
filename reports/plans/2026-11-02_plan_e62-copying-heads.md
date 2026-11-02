# E62, the copying-heads test: plan

*2026-11-02. Approved by the user (`decisions.md`, ask #18). Built from Fable's draft
`reports/plans/2026-11-02_plan_regating-mechanistic-FABLE-DRAFT.md`, option 1. Code `experiments/induction_cue/run.py`,
`src/commute/copying.py`; rules `experiments/induction_cue/results/predictions.json`; job `cluster/induction_cue.sbatch`,
one MATS `debug` job, $0, not gated. Coded, not run.*

**In plain words.** The small model already copies in context: after "A B ... A" it says B. We teach it to copy at a new
marker, where it does not copy yet, and beside that we teach a lookup at the same marker on the same text. We switch off the
heads the model already copies with: the copy map should lose most of what it learned, the lookup little. Then we ask
whether the route split's re-gating share r follows that label. If it does, r reads the mechanism where the model's prior
guess cannot; if both read low, it reads the prior.

**Restated in the Terms' words.** Elicitation here means the fine-tuned behaviour runs through a mechanism W₀ already has
(user, ask #16); carrying to unseen keys is not the label and is not read. The label is causal and uses no decomposition;
r is the candidate instrument scored against it.

## Design check

- **Goal.** `GOALS.md`, "If no simple Φ works": is post-training new capability or strengthened old? First 67M test where "old
  mechanism" and "answer already given" come apart. A step toward the goal, not a patch.
- **Nearest tried.** `backlog.md` 2026-10-22, the user's induction fine-tune, dropped because plain induction is saturated;
  its new-cue variant was never built. E58 A7 built a new rule from an old feature; this redirects an old circuit. E59's
  VPD-free label by linear readout was unreadable; this one is causal and at head level. E55 C3's pruning and reverse
  fine-tuning failed their control and are not used.
- **Decisions.** Row 73 (granularity limit) kept. Row 81: the causal half, with heads as the capability, no linear readout.
  The held-out rule is already demoted (2026-11-02) and not read here. Update size within a recipe is a standing column
  (row 2026-10-31): every state reports ||ΔW||, and both maps of a seed are read at one step, so they share the recipe cell.
- **What changes next.** "r follows mechanism" gives the Qwen KV table its first expectation beyond the prior. "r follows the
  prior" rewrites the 67M claim and leaves E55 C2 as the line's mechanistic test. No label: report why and read C2.
- **Apparatus arm, first and alone.** Stage (a), with a stop. No reference procedure is reproduced except E52's control
  and E54's elicit state in the path control, both checked against their recorded numbers.
- **Class.** Approved by the user after seeing the draft's check.

## Run order and stops

| Stage | What | Stops when |
|---|---|---|
| (a) | W₀ alone. Prefix score of all 24 heads (4 blocks, 6 heads); H = heads over the bar; mean-ablate H, then 5 random sets of \|H\| others, read E52's second-copy top-1 | pattern off `o_proj`'s input by over 1e-4; \|H\| 0, 24 or over 12; base off E52's 0.8247 by over 0.01; top-1 under H not below 0.3; median random draw not over 0.7 |
| (b) | W₀'s prior for B at the marker; zero update; the per-window target path against the lookup-table path | prior over 1e-2; zero update not exactly 0; path differs |
| (c) | Lookup then copy map, three seeds, E58's recipe; pair read at one step; states saved | none: a state under the gate is labelled too small |
| (d) | The label, the floor, the cheap-solution controls | none |
| (e) | Zero update's route gain, then r on all six | zero-update gain over 1e-6 |

`results/run.json` is written after every stage; every scored number is printed. Saved in `results/large/`: `windows.pt`,
`heads.pt`, `{copy,lookup}_s{0,1,2}.pt` at the read step (113 MB each), `run.out`. Estimated 8-15 min on an L40 from E58 A7
(13-17 min for 10 fine-tunes), not measured.

## Open points, settled

**Per-window target.** `route_cells` and `route_split` read `data.target_tokens`, which takes one id, a table indexed by the
token at `half`, or `NEXT`. At the marker the token at `half` is M for every window, so neither map fits (the lookup's key sits
at `half - 1`). `NEXT` plus `ci_len = half + 1` would need no new code, but Γ reads later tokens and would lose the 20 tokens
after the marker that E58's bands were read with. So: a new option `data.LAST` reads the target from an added last column, and
the existing `ci_len = 41` keeps that column from every forward and from Γ. r's meaning is unchanged: same cells, per-window
target, 41-token windows. `ArmAConfig` and `eval_arm_a` gain `ci_len` (default `None`, `toks[:, :None]` is `toks`);
`keydep.route_r` now calls `keydep.route_windows`, same calls. Callers checked by grep, and old argument forms unchanged since
the new branch fires only on -2 and `ci_len` defaults to no cut: `target_tokens` in `local_faith`, `pass_reading`, `ladder`,
`recruitment`, `signatures`, `shift_anatomy`, `regating_table`, `performed_anchor`, `regating_truth_table`; `route_cells` /
`route_split` in `regating`, `shift_map`, `gamma`, `harness`, `pcfg_vpd`, `lora_commute`, `kv_task`, `task_ladder`,
`recruitment_test`, `regating_leak`, `sgd_kv`, `sgd_ladder`, `regating_table`, `recruit_test`, `new_vs_existing`, `open_pair`;
`arm_a_loss` / `eval_arm_a` in `component_training`, `arms`, `adjoint_sgd`, `teacher_training`, `concentration_test`,
`sgd_ladder`, `identity_test`, `lift_test`. `tests/test_kv.py` and `tests/test_decompose.py` pass.

**Control for it.** On E54's saved elicit task (sha256 checked), the lookup table against the last column holding the same
targets: 30-step fine-tune max |ΔW difference| ≤ 1e-6, route cells ≤ 1e-4, and old r within 0.03 of E54's 0.7726. The expected
difference is exactly 0: same tensors, same batches, and E58 A6 reproduced r bitwise. The old path run twice is printed as the
measured floor. A CPU test on the tiny model already gets exact equality.

**The lookup and the pools.** A is 32 tokens (E54's K). The lookup maps each A to its own fixed word by `kv.bijection`. Both maps
train and read on the same A pool and on the same token windows; train and read windows share no row (E12's row split, every 5th
row read). 800 train, 200 read. All pools come from E58's cipher pool (word-like, frequency ranks 1000-2999), kept only if
decode then encode returns the same single id, never the marker, and are cut at random into disjoint A, words, B to train, B to
read. **B cannot leak into the lookup's target**: B and word pools are disjoint, and a window is dropped if A's word occurs in it.
A window is also dropped unless A occurs exactly twice, B once and M once in all 41 tokens. p is uniform on 2..16, so B's
position says nothing and at least one Pile token separates B from the second A. Each A meets many B, and read Bs are never
trained, so B cannot be memorised; this is not a carry test, it closes a shortcut.

**Cheap solutions of the copy map.** Two controls on the read windows, target the original B, gain over G:
- *Pair broken* (first A overwritten by another key, B stays): a true prefix match should fall. Bar ≤ 0.5. A miss sets
  `pair_ignored` beside the verdict: the model copies B out of context without matching A (say, the rare token near the end).
  It may still go through the heads, so the label stands, but the claim weakens to "copies from context".
- *B absent* (B overwritten by another pool token, original B nowhere in the window): a real copy cannot produce it. Bar ≤ 0.2. A
  miss voids the label, `no_label_copy_is_a_lean`: the marker learned a bias toward pool tokens, not copying.

The lookup has the matching check: specific share (own word's gain minus the mean of the other 31 words' over G) ≥ 0.5,
E58's bar, else `no_label_lookup_is_a_lean`.

**How attention is hooked.** `LlamaSimpleMLP` (`param_decomp/pretrain/models/llama_simple_mlp.py`): 4 blocks, 6 heads, head_dim
128, `flash_attention: false`, 6 key-value heads so nothing is repeated, rotary positions. Head outputs are not exposed, but
`o_proj`'s input is them side by side, so a forward pre-hook on `o_proj` ablates a head with **no change to the model code**.
The pattern is recomputed from `q_proj`, `k_proj`, `v_proj` outputs with the module's own `apply_rotary_pos_emb`, and stage
(a) stops if pattern times v differs from `o_proj`'s input by over 1e-4. Ablated reads are target-model only and fp32.

**Mean baseline.** Each head's output averaged over all positions of 512 unedited 41-token Pile spans from the read-side rows,
under W₀: one vector per head, used for every state, so the ablated head carries nothing from the fine-tune. Applied at every
position. Blind spot: the mean is off-distribution on the random control rows, which is what the random-head arm checks.

**The bar.** The paper gives no number ("a clear subset of heads ... with much greater than random chance"). Filed: 5 times the
score of the causal uniform pattern on the same rows, 0.0154 from the row shape, so a bar 0.077; a tie counts. **This multiple is chosen
here, not sourced**, and could not be measured first because the model is not loaded before the job. The top-1 under H control
is what validates it, and the job prints top-1 against the top k heads before any stop. If H only fails on the bar and the curve
shows one k under 0.3, the fix is a dated amendment and a $0 rerun (the 2026-10-29 rule on clear fixes).

**Why the median random draw.** A draw can hold a previous-token head, part of the same circuit (Olsson et al., Argument 5) but
outside H. Requiring every draw over 0.7 would stop the job for that. Previous-token scores are printed so a low draw can be
explained. The same median is the label's floor.

**Under the gate.** The gain gate is E58's 5.148 log-odds, here on the raw gain. A share needs a positive gain: `share_lost` is
nan unless G > 0, and a state under the gate reads `too_small`, is left out, and its r is reported only. A map needs 2 labelled
seeds (E58's `min_entry`), else `no_label_*_did_not_learn`. Read step: first of 30, 100, 300 with both maps over the gate, then
the copy map's, then the lookup's, then 300 (`copying.read_step`). Both maps of a seed are read at one step, for equal update size.

**Rule edge cases.** All in `tests/test_copying.py`: H empty, all, over half (stop); zero or negative gain; a seed under the gate;
one seed left; ties at every bar; a missing state; non-finite r; every verdict branch. A negative share (ablation raises the
gain) reads "new". Verdict order is in `copying.score`; any no-label outcome means r is reported but not scored.

## Numbers and what each should read

| Number | At W₀ | Do-nothing reference | Control | Expected / filed | Blind to |
|---|---|---|---|---|---|
| Prefix score per head | as measured; this is W₀'s number | chance, 0.0154 (uniform pattern) | reconstruction ≤ 1e-4 | 1 to 12 heads over the bar; lean, none in block 0 | heads that copy without attending by prefix; attention is not output |
| Second-copy top-1 | 0.8247 (E52) | random heads: median over 0.7 | H: under 0.3 | as filed; binomial sd 0.011 on 1152 | any copying this easy format does not use |
| Prior: geo-mean P(B) at M | the reading itself | B absent: lower (lean) | stop over 1e-2 | 1e-6 to 1e-4 | is a mean over windows, not the rare windows that already copy |
| Gain G (log-odds) | exactly 0 | zero update exactly 0 | gate 5.148 | E58 fine-tunes read 13-21 at step 30; the copy map may need more steps | which route carries it |
| Share lost under H | undefined by rule, G − G_abl reads exactly 0 | random heads: median under 0.15 | path control; floor | copy ≥ 0.5, lookup ≤ 0.2; row-bootstrap sd printed; three-seed range is its seed noise | a fine-tune keeping H's key and value paths but a new query counts as reuse; subcomponents; other heads the fine-tune recruits |
| Pair-broken ratio | 0 by zero update | n/a | n/a | ≤ 0.5 | matching on tokens other than A |
| B-absent ratio | 0 by zero update | n/a | n/a | ≤ 0.2 | a lean only toward trained Bs (read Bs are untrained) |
| Lookup specific share | undefined | n/a | E58's 0.5 | E58 A7's twin read 0.65-0.72 at step 30 | a lean by word frequency |
| r | undefined, gain exactly 0 (printed with L10 − L00 and L01 − L00) | E58's own reference bands, cipher 0.29-0.31, copy 0.77 | E54's elicit_s0 0.7726 ±0.03 through both paths | see below; noise 0.045 (E58's largest three-seed range) | which subcomponents; r's cut at 0.5 |
| ||ΔW|| | 0 | n/a | n/a | E58 ~5.2 at step 30 | reported beside r, per the 2026-10-31 row |

**Reading r.** Copy map r ≥ 0.5 on every labelled seed, lookup under 0.5 and mean gap ≥ 0.24: "r follows mechanism". Both under
0.5: "r follows the prior". Anything else, both high included: "untested". `near_cut` flags an r within 0.045 of 0.5.

**Prior beside r.** E58's table has r 0.37 at prior 3.7e-5 and 0.77 at 5.9e-5, so the prior alone does not fix r in this range.
A copy-map prior in (1e-3, 1e-2] runs on but is flagged `prior_high`: then the run cannot say mechanism rather than prior. Rare
tokens in context often rise on their own, so this is the likeliest way the design weakens.

## Granularity limit

A new circuit built from old subcomponents, switched on in a new pattern, reads as elicited to VPD. Here the redirected query
is such a case: neither r nor the head ablation can split "same circuit redirected" from "new circuit of old pieces". Every claim
says "elicited at the level of VPD's pieces", and at the level of heads for the label.

## Risks that the run itself tells apart

- The copy map may not reach the gate. Its query must move from M to the token before it, which takes a block for a head to
  read A into the marker's position: the read steps go to 300, and if it still fails the outcome is "did not learn", reported.
- H may not be sufficient alone (shared load, or previous-token heads doing part of it): stage (a) stops, $0.
- bf16 fine-tune against fp32 reads: as E58, gains here are fp32. The Qwen rounding finding (E61) is on another model and at
  its first steps; the first read here is step 30. Not checked on the 67M.
- Not gated, so one fresh review is enough; the draft asks for one on the ablation code before the job.

## Departures from the literature

- **Olsson et al. 2022, prefix-matching score.** As their appendix: 25 random tokens without the most and least common,
  repeated 4 times, a start token, mean of the stripe entries. Departures: 48 sequences, not 10; word-like tokens only; cut-offs
  (ranks under 1000, rarest 10%) fixed here, as they give none; start token `<|endoftext|>`.
- **Their wording and their offset.** Their evaluator sentence says "the tokens that preceded the same token", while their
  definition says the head attends to "the token which induction would suggest comes next". We score the token after the
  earlier copy, the second; the literal reading's score is printed beside.
- **Bar.** They give none; ours is above.
- **Ablation.** They zero a head's contribution, and on small models only. We use the mean, as approved. **The draft's "they
  report zero and mean ablations on larger models" does not match the paper** (full-scale models had no ablations). Zero
  ablation is not run.
- **Copying score.** They require a copying score too for an induction head. We use only prefix matching and let the causal
  control speak.
- **Jain et al. 2024, Def 2** takes a linear readout as the capability; we take a head set and the user's causal requirement.
- **Thresholds.** 0.5 on the share and r is E12's convention; 0.24 is E58's band gap; 0.2 and 0.15 are the draft's. None published.
- **Recipe.** Adam 3e-5, batch 32, KL 1, clip 1 is this project's (E54, E58), from `keydep.finetune`, unchanged except the
  target form, checked by the path control. KL covers the same 40 non-read positions.

## Where each setting comes from

In `predictions.json` under `settings_sources` and each threshold's `_source`. Chosen here, without a source, and said so: seeds,
n_random 5, 512 spans for the mean, bar multiple 5, control bars 0.2 and 0.5, window count 1000 (E54's 500 doubled).

## Not run

Everything. Tests ran on CPU on tiny models, including the hooks and pattern recomputation on a tiny `LlamaSimpleMLP`. The 67M was
never loaded. Not planned: zero ablation, copying score, carry to unseen keys, steering's S, the stand-in ratio.
