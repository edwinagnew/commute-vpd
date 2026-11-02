# E62, the copying-heads test: plan

*2026-11-02. Approved by the user (`decisions.md`, ask #18). Built from Fable's draft
`reports/plans/2026-11-02_plan_regating-mechanistic-FABLE-DRAFT.md`, option 1. Code `experiments/induction_cue/run.py`,
`src/commute/copying.py`; rules `experiments/induction_cue/results/predictions.json`; job `cluster/induction_cue.sbatch`,
one MATS `debug` job, $0, not gated. Coded, not run.*

*Amended 2026-11-02, before the first run, after `reports/reviews/2026-11-02_review_e62-code.md` (F1, F2, F4; F3, F5, F6
below). The lead chose the fixes. Predictions were changed with a dated note in `predictions.json` (`amended_before_first_run`):
the run had not happened, so nothing was read.*

*Amended 2026-11-02 after job 48507 stopped in stage (b): see "Amendment 1" at the end. Where the text below differs
(the 128-token word pool, windows dropped for holding a candidate, what "both high" means), Amendment 1 holds.*

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
| (b) | W₀'s prior for B at the marker; the lookup's words drawn to match it; zero update; the per-window target path against the lookup-table path (old path run twice for the tolerance) | prior over 1e-2; no word set within a factor 2 of B's prior, or a measured ratio over 2; zero update not exactly 0; path differs |
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
targets: 30-step fine-tune max |ΔW difference| ≤ 10 × the same measure between two runs of the old path (review F4: the
tolerance now comes from a measured floor, one extra fine-tune of ~25 s in stage (b)), route cells ≤ 1e-4, and old r within 0.03
of E54's 0.7726. The factor 10 is chosen here, no source: one repeat is one sample of the floor. The expected difference is
exactly 0: same tensors, same batches, and E58 A6 reproduced r bitwise; a repeat difference of exactly 0 therefore demands
exactly 0, and a nonzero one sets the scale. The old path run twice for the route cells is printed as that floor too. A CPU
test on the tiny model already gets exact equality.

**The lookup and the pools.** A is 32 tokens (E54's K). The lookup maps each A to its own fixed word by `kv.bijection`. Both maps
train and read on the same A pool and on the same token windows; train and read windows share no row (E12's row split, every 5th
row read). 800 train, 200 read. All pools come from E58's cipher pool (word-like, frequency ranks 1000-2999), kept only if
decode then encode returns the same single id, never the marker, and are cut at random into disjoint A, B to train, B to
read. (Amendment 1 replaces the word pool of 128 candidates that was cut here; see below.) **B cannot leak into the
lookup's target**: B and the candidates are disjoint. (Amendment 1: no window is dropped for holding a candidate; the
lookup's word for a key is kept out of that key's own windows instead.)

**Matching the prior (review F1).** The test exists to separate "r reads the mechanism" from "r reads W₀'s prior for the
target", and E58 found r tracks the prior closely (rank correlation 0.95 over eight maps). So the lookup's words are not fixed
in advance. Stage (b) measures W₀'s geometric-mean P(B) at the marker on the read windows, then draws each key's word
(`copying.pick_words`, `kv.bijection`'s `ok`, the recipe of E58's prior cipher in `open_pair/amend6.py`) from the word pool
(Amendment 1: from the wider candidates) with that word's mean log P over the key's windows within a factor 2 of P(B). The map's geometric-mean P(word) is then within
a factor 2 of P(B) too. Both priors and their ratio are logged; a key with no word in range, or a measured ratio over 2, stops
the job (exit 1, $0). `prior_match_max` is 2, **with no published source**: chosen so the pair is tighter than E58's 3.7e-5 vs
5.9e-5, which is a factor of 1.6 and read r 0.38 vs 0.77. Yes, 1.6 is inside 2: the bar is looser than that pair, and at
a factor between 1.6 and 2 the prior alone could still move r a little. `score` still refuses a ratio over the bar
(`priors_not_matched`) though the stop makes it unreachable. One lookup arm only.
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

**Under the gate.** The gain gate is E58's 5.148 log-odds (its source was already E58's specific gain). The read step is
chosen on the **specific gain** (review F2): the copy map's gain minus its gain with B absent, the lookup's gain minus the mean
gain of the other words (`heldout_keys.specific`'s definition), because early in a fine-tune the raw gain is partly a general
push toward the pool's tokens (E58: raw 14.0 and specific 1.6 at step 10) and would pick a step where the copy map is still a
lean, then void the label with no later step to look at. Stage (c) computes, for every map at each of steps 30, 100 and 300,
the raw gain, the B-absent gain, the pair-broken gain and the specific gain, prints them and writes them to
`run.json` (`states.*.gains_by_step`; the lookup has no B, so it carries raw, others and specific). A share needs a positive
gain: `share_lost` is nan unless G > 0, and a state whose raw G is under the gate reads `too_small`, is left out, and its r is
reported only (the label's own gate stays on the raw G). A map needs 2 labelled seeds (E58's `min_entry`), else
`no_label_*_did_not_learn`. Read step: first of 30, 100, 300 with both maps' specific gains over the gate, then the copy map's,
then the lookup's, then 300 (`copying.read_step`). Both maps of a seed are read at one step, for equal update size.

**Rule edge cases.** All in `tests/test_copying.py`: H empty, all, over half (stop); zero or negative gain; a seed under the gate;
one seed left; ties at every bar; a missing state; non-finite r; every verdict branch. A negative share (ablation raises the
gain) reads "new". Verdict order is in `copying.score`; any no-label outcome means r is reported but not scored.

## Numbers and what each should read

| Number | At W₀ | Do-nothing reference | Control | Expected / filed | Blind to |
|---|---|---|---|---|---|
| Prefix score per head | as measured; this is W₀'s number | chance, 0.0154 (uniform pattern) | reconstruction ≤ 1e-4 | 1 to 12 heads over the bar; lean, none in block 0 | heads that copy without attending by prefix; attention is not output |
| Second-copy top-1 | 0.8247 (E52) | random heads: median over 0.7 | H: under 0.3 | as filed; binomial sd 0.011 on 1152 | any copying this easy format does not use |
| Prior: geo-mean P(B) at M | the reading itself | B absent: lower (lean) | stop over 1e-2 | 1e-6 to 1e-4; lookup word's within a factor 2 of it (about 5e-7 to 2e-4), a stop otherwise | is a mean over windows, not the rare windows that already copy |
| Gain G (log-odds) | exactly 0 | zero update exactly 0 | gate 5.148 on the specific gain, to pick the read step | E58 fine-tunes read 13-21 at step 30 (specific); the copy map may need more steps | which route carries it |
| Share lost under H | undefined by rule, G − G_abl reads exactly 0 | random heads: median under 0.15 | path control; floor | copy ≥ 0.5, lookup ≤ 0.2; row-bootstrap sd printed; three-seed range is its seed noise | a fine-tune keeping H's key and value paths but a new query counts as reuse; subcomponents; other heads the fine-tune recruits |
| Pair-broken ratio | 0 by zero update | n/a | n/a | ≤ 0.5 | matching on tokens other than A |
| B-absent ratio | 0 by zero update | n/a | n/a | ≤ 0.2 | a lean only toward trained Bs (read Bs are untrained) |
| Lookup specific share | undefined | n/a | E58's 0.5 | E58 A7's twin read 0.65-0.72 at step 30 | a lean by word frequency |
| r | undefined, gain exactly 0 (printed with L10 − L00 and L01 − L00) | E58's own reference bands, cipher 0.29-0.31, copy 0.77 | E54's elicit_s0 0.7726 ±0.03 through both paths | see below; noise 0.045 (E58's largest three-seed range) | which subcomponents; r's cut at 0.5 |
| ||ΔW|| | 0 | n/a | n/a | E58 ~5.2 at step 30 | reported beside r, per the 2026-10-31 row |

**Reading r**, with the priors matched (ratio ≤ 2; otherwise `priors_not_matched`, r reported beside). Copy map r ≥ 0.5 on
every labelled seed, lookup under 0.5 and mean gap ≥ 0.24: "r follows mechanism", r separates the maps where the prior cannot.
Every r of both under 0.5 **and** the mean gap under 0.24: "r follows the prior", meaning that with the priors matched r does
not read this mechanism at this prior (the name is kept from the earlier reading, which could not separate the two).
**Amendment 1:** every r of both at or over 0.5 with the mean gap under 0.24 reads "r follows the prior" too, and copy low with
lookup high at a gap of -0.24 or less reads "r reverses the label" (a surprise, announced). Anything else is "untested": all
low with a gap (copy 0.44 beside lookup 0.10), both high with a gap of 0.24 or more, a missing or non-finite r. `near_cut`
flags an r within 0.045 of 0.5.

**Prior beside r.** E58's table has r 0.37 at prior 3.7e-5 and 0.77 at 5.9e-5, so the prior alone did not fix r at that gap
of 1.6; across all eight of its maps, rank correlation of prior and r is 0.95. Hence the match above: without it "r follows
mechanism" could fire on a prior gap alone. A copy-map prior in (1e-3, 1e-2] runs on but is flagged `prior_high`: then the run
cannot say mechanism rather than prior even with the match (both maps' priors are high). Rare tokens in context often rise on
their own, so this is the likeliest way the design weakens.

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
- **Ablation.** Olsson et al. 2022 zero-ablate a head's contribution, and on small models only; we mean-ablate (each head's
  output replaced by its mean over unedited Pile positions), as approved. **The draft's "they report zero and mean ablations
  on larger models" does not match the paper** (full-scale models had no ablations). Zero ablation is not run.
- **Copying score.** They require a copying score too for an induction head. We use only prefix matching and let the causal
  control speak.
- **Jain et al. 2024, Def 2** takes a linear readout as the capability; we take a head set and the user's causal requirement.
- **Thresholds.** 0.5 on the share and r is E12's convention; 0.24 is E58's band gap; 0.2 and 0.15 are the draft's; the prior
  match of 2 and the path-control factor of 10 are chosen here. None published. The head bar (5 × the uniform pattern) is also
  ours, not the paper's.
- **Gain gate.** E58's 5.148, on its specific gain; here the copy map's specific gain is its gain minus its gain with B absent,
  our analogue (E58 had a value pool to subtract; the copy map has none).
- **Recipe.** Adam 3e-5, batch 32, KL 1, clip 1 is this project's (E54, E58), from `keydep.finetune`, unchanged except the
  target form, checked by the path control. KL covers the same 40 non-read positions.
- **Lookup's candidate words (Amendment 1).** Every intact word-like token of the mined list at any frequency rank, not
  E58's rank-1000-and-up cipher pool: chosen here on Fable's advice, no published source. E58's own prior cipher (band 3e-4
  to 3e-3) hit the same wall, 46 of 128 keys with no admissible value, and was skipped (`experiments/open_pair/results/amend7.json`,
  `tables.ep_skip`).
- **Word kept out of the key's own windows (Amendment 1).** A key's word may not occur in that key's own train or read
  windows, so the copying head cannot produce it; other keys' words may occur. Chosen here on Fable's advice, no source.
- **Per-key factor 2 (Amendment 1).** Kept from the filing (E58 A6's prior cipher recipe, `pick_words`); `prior_match_max` 2
  is still the unsourced bar the plan names above.

## Review items read and skipped

`reports/reviews/2026-11-02_review_e62-code.md`, items F3, F5, F6 were read and not acted on:
- **F3** (verdict names overclaim): read "between" as no clean label. `copy_not_elicitation` with a copy share of 0.62, 0.58 or
  0.47, and `lookup_uses_the_heads` at a lookup share of 0.25, are read that way in every write-up; the names stay.
- **F5** (random draws collapse at large |H|): reported if H comes out that large; the draws are printed.
- **F6** (zero-update stops test little): kept, they cost nothing. L10 − L00 and L01 − L00 at the zero update stay printed.

## Where each setting comes from

In `predictions.json` under `settings_sources` and each threshold's `_source`. Chosen here, without a source, and said so: seeds,
n_random 5, 512 spans for the mean, bar multiple 5, control bars 0.2 and 0.5, window count 1000 (E54's 500 doubled).

## Not run

Everything. Tests ran on CPU on tiny models, including the hooks and pattern recomputation on a tiny `LlamaSimpleMLP`. The 67M was
never loaded. Not planned: zero ablation, copying score, carry to unseen keys, steering's S, the stand-in ratio.

## Amendment 1 (2026-11-02, after job 48507's stop in stage (b))

Advice: `reports/plans/2026-11-02_advice_e62-prior-stop-FABLE.md` (Fable); the lead adopted its items 1-4. Filed text in
`predictions.json` (`amended_after_stop_48507`). Nothing after stage (b) ran, so no fine-tune, label or r was read.

**What the stop read** (job 48507, `results/run.json` as committed in db2a902). Stage (a) passed: one head, (2,4), prefix
score 0.42 against a bar of 0.077; its mean ablation takes the repeated-sequence top-1 from 0.8247 to 0.0095. Stage (b): W0's
geometric-mean prior for B at the marker 4.297e-04 (band `over_band`, which runs on), 6.169e-06 with B absent, 8.468e-05 with
the pair broken. Then the stop: "no admissible value left in the pool for key 7192", because the 128 candidates (ranks
1000-2999, absent from every window) sat near the context-free prior 6e-6, a factor ~70 under B's 4.3e-4. The stop came from
the candidate pool, not from B.

**What was filed before.** `n_word_pool` 128 (pool ranks 1000-2999 cut into A, 128 candidates, B-train, B-read); every window
dropped if it held any candidate; `b_match` "from the 128-token word pool"; `e_r` with "both high" untested and no reversal
cell. All replaced below; no bar moved.

**Changes.**
1. *No bar moves.* `prior_band` [1e-6, 1e-4], `prior_max` 1e-3, `prior_stop` 1e-2, `prior_match_max` 2 and every other
   threshold stay. 4.3e-4 is 0.04% of the mass, far from top-1: "answer already given" does not hold. The 14x lift for B
   anywhere in the window and a further 5x for the pair intact say the copying route already fires weakly at the marker.
2. *The lookup's words.* Candidates are all intact word-like tokens of the mined list at any rank, minus the keys, both B
   pools and the marker (in practice ranks under 1000 and from 3000, as A and B take ranks 1000-2999). Per key, a word within
   a factor 2 of P(B) on that key's read windows, as `pick_words` did. `cue_windows` no longer filters on candidates, so both
   maps keep the same windows; instead `pick_words` rejects a word that occurs in that key's own train or read windows
   (`copying.own_tokens`, `copying.admissible`). The lookup's words are not in its own context, so the copying head cannot
   produce them. Before the draw, stage (b) prints per key the number of admissible candidates and the candidates' log10-prior
   histogram, and writes both to `run.json` `stage_b`. A key with no admissible word still stops the job. Number of
   candidates enough for 32 distinct words in range: not known without the model; that is what the print measures.
3. *Reading r at matched priors near 4e-4.* E58 at step 30: copy map prior 5.9e-5 read r 0.77, elicit 1.3e-3 read 0.73, rank-1
   1.3e-2 read 0.85. So the prior alone predicts copy r near 0.75 and the whole weight of the test is on the lookup, for
   which E58 has no point at this prior.

   | Copy r | Lookup r | Class | Meaning in plain words |
   |---|---|---|---|
   | all >= 0.5 | all < 0.5, mean gap >= 0.24 | `r_follows_mechanism` (filed) | r is high where the old circuit does the work and low where it does not |
   | all >= 0.5 | all >= 0.5, mean gap < 0.24 | `r_follows_prior` (added) | r is high for both: it does not tell the maps apart at this prior, so it follows the prior. Read as "old mechanism, answer weakly present": W0 leans 5x toward B with the pair intact, and r reads high |
   | all < 0.5 | all < 0.5, mean gap < 0.24 | `r_follows_prior` (filed) | r is low for both; against E58's 0.75 it also means that relation fails here |
   | all < 0.5 | all >= 0.5, mean gap <= -0.24 | `r_reverses_label` (added, announce it) | r is high for the map labelled new and low for the one labelled old |
   | anything else | | `untested` | including both high with a gap of 0.24 or more; `near_cut` where within 0.045 |

   The mechanism row is untouched, so no bar moved to rescue the run. The filed rule wrote only both-low because the filed
   band put the prior at 1e-5, where E58 says r is low. One choice not in the advice: both high with a mean gap of -0.24 or
   less (the lookup well above the copy map) also reads `r_follows_prior`, as the filed gap test is signed and no rule
   names that cell; it is a cell nobody expects.
4. *Stage (a) stays.* H is the heads over the bar, i.e. (2,4) alone. Head (1,1) has previous-token score 0.67 (next highest
   0.35, `run.json` `stage_a.previous_token`). Fable calls it a previous-token head and the circuit's other half; **that role
   is a presumption, not measured**: the score is measured, no ablation of (1,1) alone was run, and it is not added to H (at
   the marker both maps need A's identity moved one position forward, so ablating it would leave both maps "old"). One
   random draw of five holding (1,1) is absorbed by the median rule. New, reported only and not a rule: the 24 heads'
   prefix scores under each saved fine-tuned state are printed and stored (`states.*.prefix_scores`), so that if the copy map
   recruits (2,3) or (2,2) (0.05 and 0.04 under the bar), a share under H that falls can be told from a wrong label.

**Rerun behaviour.** `run.py` has no check for existing results: it refuses to start only if `predictions.json` is not
committed unchanged against HEAD (commit this amendment first). The first `dump` of the new run (in stage (a), after the prefix scores) overwrites `results/run.json`; `windows.pt` and `heads.pt` in `results/large/` are overwritten at
once. The run starts a fresh record, so a stale `stopped` key cannot carry over. The earlier stop's numbers stay in git history
(db2a902) and are quoted above: head (2,4), top-1 0.8247 to 0.0095, prior 4.297e-04, B absent 6.169e-06, pair broken 8.468e-05.

**Filed expectations, new.** Every key finds an admissible candidate (else the stop stands); the printed counts measure
whether 32 distinct words fit. The lookup's geometric-mean prior then reads within [P(B)/2, 2 P(B)], about [2.1e-4, 8.6e-4].
The prefix scores under the states carry no expected value: they are reported only.

**Code.** `copying.cue_windows` (no `avoid`), `build_cue` (candidates, no `n_word_pool` setting), `own_tokens`, `admissible`,
`pick_words` (+ `own`), `prior_histogram`, `r_verdict`; `run.py` `stage_b`, `stage_c`, `score_rows`, `prefix_scores_under`.
Tests: `tests/test_copying.py` (`test_pick_words_reject_a_candidate_in_the_keys_own_windows`,
`test_prior_histogram_counts_by_bin`, `test_r_verdict_with_priors_matched`, `test_score_every_outcome`); 28 pass, lint clean.

**Not run.** The model was not loaded and no job submitted. Code reviewed by nobody beyond this amendment's tests; the changed
measurement code has not run on a GPU (the scored run is its own smoke).

## Amendment 2 (2026-11-02)

Spec: Option 1 of `reports/plans/2026-11-02_advice_e62-next-FABLE.md`, adopted by the lead. Rules:
`experiments/induction_cue/results/predictions_amend2.json`. Code: `experiments/induction_cue/amend2.py`, `cluster/induction_cue_amend2.sbatch`.
Coded, not run. **Not gated**: one cluster `debug` job, under 10 min, $0, no fine-tuning.

**In plain words.** E62 taught the 67M to copy at a new marker beside a lookup at the same marker. About half of the copying
map runs through old head (2,4) and none of the lookup does, and r read high on the copying map, unscored. This reads the six
saved states again to ask whether the gain r measures goes through that same head, or whether r is high only because the answer
sits in the window.

**Restated in the Terms' words.** Elicitation: the fine-tuned behaviour runs through a mechanism W₀ already has (user). r is the
route split's re-gating share, (L01 − L00) / (T1 − T0), with L the masked log-odds: subscripts components then gates, 0 W₀, 1 the
state with its least-squares split. The question: with head (2,4) mean-ablated inside the masked forward, original subcomponents and
the same masks, does L01 − L00 fall by the share the target model's gain falls? If yes, r and the label read the same head on this
map; if the masked gain barely falls, the original subcomponents reach B some other way. Beside it: W₀'s own fall under the head
(F3), both circuit heads off together, the pair-broken gain under the head (the check's hypothesis), and whether update size
explains the r gap (F1). The filed verdict `copy_not_elicitation`, r not scored, stands; nothing here rescores it.

**Granularity limit.** A new circuit of old pieces reads as elicited to VPD. Each reading is at the level of heads and VPD's
pieces, and a redirected query still counts as reuse.

### Design check

- **Goal.** `GOALS.md`, new capability or strengthened old: what r can mean when the old mechanism is known by a causal label.
- **Nearest tried.** E62 itself read the label and r separately. Nothing read one inside the other; Fable's option 1 says why other
  separations of "target in context" and "old mechanism" fail on this model.
- **Patch or step.** A step: decides if r is usable beyond "the subcomponents can write this under some mask", not a repair.
- **Decisions.** Row 73 (granularity limit) kept; update size as a standing column (2026-10-31) is row 7. No departure.
- **What changes next.** Head-dependent: report r as reading the label's mechanism on this map, one datum, then the Qwen table.
  Not: report r as weaker than mechanism and stop the 67M copy work (Fable's option 1).
- **Apparatus first.** Row 1 alone, exit 1 on a miss, then row 2. Nothing is reproduced except E62's own reads, each re-read.
- **Class.** 1, analysis of saved states: approved, no sign-off.

### The hook, checked before coding

`o_proj` is itself decomposed (`final_config.yaml`: `h.*.attn.o_proj`, C 1024). It is still the module the forward calls:
`ComponentModel` does not replace modules, it attaches a forward hook (`_attach_forward_hooks`) to each, and that hook returns
`components(x, mask=...)` with `x` the hook's own args (`_components_and_cache_hook`). PyTorch hands a forward hook the args as
left by the pre-hooks, so the edit reaches the components. **`copying.ablated` is unchanged.** `y = att @ v` comes from the q, k and v
components, so the head's value there is replaced in the masked forward too. `ablated`'s callers, E62's `run.py` (lines 140, 152,
157, 216, 305) and `tests/test_copying.py`, are untouched. A test on a real tiny `ComponentModel` shows the components take the mean;
row 1 tests the 67M. One trap: `compute_ci` reads the same module's input, so **Γ must never run inside `ablated`**, or the masks
would change with the head. `amend2.py` reads the gates first and uses them after, so the masks are the same per state.

New code: `copying.AllGatesOn`, which gives `induction` and `ablated` the masked forward; `local_faith.masked_lo_windows` gains
`ci_len`, default none, so its callers keep working (`shift_anatomy`, `regating_leak`, `regating_table`); the rules below.

**Which means.** The masked reads use `heads.pt`, W₀'s target-model means, as every target read does. One replacement vector then
serves every cell, so the head carries nothing of the state or the mask, and the masked and target shares differ only in the forward.
Means re-read under each mask would move with the state, and the masked gain would partly read that move. The cost: the target's mean may sit off
the masked forward's own distribution. L00 and L01 take the same vector, so that offset cancels to first order, and rows 1 and 3's
(3,5) reference show what is left.

### Rows, in run order

| Row | Number | W₀ | Do-nothing or reference | Control | Expected, copy | Expected, lookup |
|---|---|---|---|---|---|---|
| 1 | Top-1 through the masked forward, all gates on | intact about 0.82 | (3,5) off about 0.7, bar 0.66 | (2,4) off about 0.01, bar 0.05; exit 1 | | |
| 2 | Target share lost under (2,4), re-read | undefined | zero update exactly 0 | `run.json` within 1e-3; exit 1 | 0.563, 0.545, 0.471 | 0.000, −0.004, −0.003 |
| 3 | Masked share of L01 − L00 lost under (2,4) | L00 with the head off; lean near 1.57 falls under half | (3,5) under 0.15; zero update's masked gain exactly 0 | cells with the head on equal `run.json` within 1e-3; exit 1 | within 0.092 of the target share (lean) | within 0.092 of about 0 |
| 4 | W₀'s fall under (2,4); state's own fall over its gain | fall in [0, 4.18], lean at least 1 | | | own fall over filed share + 0.092 | small |
| 5 | Target share under (2,4) + (1,1) | top-1 at or under 0.02 | | | 0.7 or more, or within 0.092 of the single share | at most 0.1 |
| 6 | Pair-broken gain lost under (2,4), target only | | | gain re-reads `run.json` within 1e-3 | at most 0.092 (lean), or near the intact share | n/a |
| 7 | r with the update scaled to the other map's size | | | alpha 1 equals `run.json` within 1e-3 | swapped gap at least half of about 0.69 (lean) | |

Outcomes, all per `predictions_amend2.json`. Copy, row 3: `r_reads_the_head`, `head_more_than_target`, `mask_reaches_b_another_way`
(under 0.25), `in_between`, `floor_high`, `too_small`, `unreadable`. Row 5: `circuit`, `no_more`, `between`. Row 6: `outside_head`,
`through_head`, `between`. Row 7: `size_accounts`, `size_does_not`, `unreadable`.

**The tolerance.** Fable's "within 0.1" had no source. Filed 0.092: the larger of three times the sd of a difference of two
independent shares (3 × √2 × 0.0173 = 0.073) and the three-seed range of the share (0.092, the largest difference two states of
one recipe show on the same windows). The range sets it. A masked `share_sd` over 0.0217 weakens that seed's class, as the
tolerance took the target's sd for it; printed, no rule. Row 6's share is on a smaller gain, so its sd is larger and printed too.

**Choices made here, not in the brief.**
- (3,5)'s bar in row 1 is 0.66, not 0.7. Its target reading is 0.7075, under one binomial sd above 0.7. Filed: target minus three sd.
- The intact masked forward must clear that bar too, or the control is empty: a forward that does not copy reads low under any head.
- (2,4)'s bar is 0.05, chosen, no source. A missed hook reads near 0.8.
- The lookup uses the one tolerance, not Fable's 0 ± 0.05.
- `min_gain` is 1.0 (`local_faith.MIN_GAIN`) in all these rows, not E62's 5.148, which would void row 6 on the gains it reads.
- `head_more_than_target` is added: a masked share above the target's by over the tolerance was a case no named outcome covered.
- Row 7 scales each state to the same seed's other map, not to a map mean. Nothing cut: `keydep.route_windows` takes a scaled state as is.
- W₀ lean in the masked forward and in the target are both reported; Fable's 1.57 is the target's.

### Blind to

- **Top-1 control.** Copying on these repeated rows, not at the marker; a hook that works there works anywhere in the forward.
- **Target share.** Heads other than those read; hydra-like compensation after the head goes (McGrath et al. 2023), which makes it low.
- **Masked share.** Γ's reaction to the head: masks stay as read with the head on, by design. Which subcomponents carry the
  gain. A mean off the masked forward's distribution beyond first order.
- **W₀'s fall.** A single mean over windows; the geometric means behind the bracket are another measure.
- **Two heads.** Series against parallel is read only through the sum; heads beyond the two.
- **Pair-broken.** Matching on tokens other than A; only step 30.
- **Scaled r.** A scaled update is not a trained state of that size, so row 7 reads the size of this update's direction, not of a
  shorter fine-tune; a map's ease is not separated from its size.
- **All.** One model, one marker, step 30, three seeds; a fine-tune that keeps the head's key and value paths and changes the query counts as reuse.

### Departures from the literature

- **Olsson et al. 2022** zero-ablate a head's contribution, and on small models only. We mean-ablate, as in E62; zero ablation is not run.
- **Ablation inside a decomposition's masked forward** has no source that we know of; the masks are held fixed.
- **Thresholds.** 0.25, 0.7, 0.1 are Fable's, 0.05, 0.66 and 0.02 chosen or derived here, 0.15 E62's; none published.
- **The half rule of row 7** is the lead's filing, not a published standard.

### Not run

Everything. The rule functions and the hook ran on CPU on a tiny model; the 67M was never loaded. The new masked read has not run
on a GPU; row 1 is its first test. Not planned: later read steps (Fable's option 3), the masked forward under two heads,
attention at the marker (F6), Γ recomputed under ablation.
