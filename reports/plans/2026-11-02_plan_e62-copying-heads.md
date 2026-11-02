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


## Amendment 3 (2026-11-02)

Spec: the lead's choice of the third account in `reports/analyses/2026-11-02_analysis_e62-amend2-masked-head.md`, with
`reports/reviews/2026-11-02_review_e62-amend2-check.md` F1 and F2. Rules: `experiments/induction_cue/results/predictions_amend3.json`.
Code: `experiments/induction_cue/amend3.py`, `cluster/induction_cue_amend3.sbatch`. Coded, not run. **Not gated**: one cluster `debug`
job, under 10 min, $0, no fine-tuning.

**In plain words.** Switching off old head (2,4) costs the target model about half of its copying gain, and costs the masked forward
about a tenth. But that read held Γ's masks as they were written with the head on, and the update reaches the masked forward only
through those masks. This reads Γ again with the head off, to see whether the masked gain falls then.

**Restated in the Terms' words.** r is the route split's re-gating share, (L01 − L00) / (T1 − T0), with L the masked log-odds:
subscripts components then gates, 0 W₀, 1 the state with its least-squares split. The masked gain is L01 − L00. The question: with
head (2,4) mean-ablated both in the forward whose activations Γ reads and in the masked forward, original subcomponents, does the
masked gain fall by the share the target model's gain falls? On the same 200 read windows, six states, with the lookup, head (3,5)
and W₀ read beside. Amendment 2's three accounts: (a) the masks close the head's way, (b) a real other way, (c) the head acts through
the masks. A share near the target's means (c); under 0.25 leaves (a) and (b). Filed verdicts stand; nothing is rescored.

**Granularity limit.** A new circuit of old pieces reads as elicited to VPD. A share near the target's says the masked gain depends
on the head through Γ's inputs. It does not say the same circuit is reused.

### Design check

- **Goal.** `GOALS.md`, new capability or strengthened old: whether r can read a head at all on the one map with a causal label.
- **Nearest tried.** Amendment 2 row 3, the same share with frozen masks; its plan listed this read as not planned. E4's Amendment 1 fixed
  gates to W₀ for another purpose. Nothing has read Γ under ablation.
- **Patch or step.** A step: the report names it as the only read that could reverse the branch.
- **Decisions.** Row 73 (granularity limit) kept. No departure.
- **What changes next.** `head_acts_through_masks`: Amendment 2's stop is withdrawn in favour of "r reads the head, through Γ, on one map",
  and it goes to the user as a reversal. `still_other_way` or `in_between`: the stop stands, with (c) out or partial. `floor_high` or
  `not_specific`: the read cannot separate, the stop stands as "unshown". Report's wording follows the verdict.
- **Apparatus first.** Row 1 alone at W₀, exit 1 on a miss, before any state is read.
- **Class.** 1, analysis of saved states: approved, no sign-off.

### What Γ reads, checked before coding

**`route_split` computes gates from a fresh pass, not a cache.** `route_cells` calls `vpd.compute_ci` per batch, which runs
`model(x, cache_type="input")` then `calc_causal_importances(cache, sampling="continuous")`. Nothing is carried between calls. With no
`mask_infos` the forward is the clean target model at the current weights, components unused, so Γ does not depend on the split.
`continuous` draws no random numbers.

**A pre-hook on `o_proj` reaches Γ at `o_proj` itself and downstream.** `_components_and_cache_hook` is a forward hook that caches `x = args[0]`.
PyTorch hands a forward hook the args as the pre-hooks left them, so `o_proj`'s cached input holds the mean in head 4's slot. Block 2's
`c_fc` and `down_proj` and all of block 3 are downstream (`x = x + attn(rms_1(x))`, then the MLP), so their inputs change too. Blocks 0
and 1 and block 2's q, k, v do not. Row 1 tests all of this on the 67M, `tests/test_copying.py` on a tiny real `ComponentModel`.

**Γ's reaction is wider than the head's causal reach, and that is the main caveat.** Γ is `global_shared_transformer`
(`GlobalSharedTransformerCiFn.forward`): it RMS-normalises each of the 24 inputs, concatenates them, and attends over the row in both
directions. So the head off can move gates at every matrix, including blocks 0 and 1 where no activation changed, and at every position.
Also `o_proj`'s whole input is RMS-normalised, so a replaced slot rescales how the other five read to Γ. Γ never trained on a slot held
constant. A fall can therefore be Γ simply breaking off its distribution, not the head's content reaching the masks (review, F1). The
references below exist to tell those apart.

**Nothing in `copying.ablated` changes.** `amend2.py` is unedited: `amend3.py` imports `Reads`, `NAMES` and `map_of` from it, the pattern of
`experiments/regating_spread/`. `src/commute/copying.py` only gains functions (`gamma_inputs`, `slot_reads`, `slot_class`, `gamma_reading`),
so E62's `run.py`, `plot.py`, `amend2.py` and `plot_amend2.py` are unaffected; `tests/test_copying.py` passes with the new tests.

### Rows, in run order

| Row | Number | W₀ | Do-nothing or reference | Control | Expected, copy | Expected, lookup |
|---|---|---|---|---|---|---|
| 1 | Γ's inputs under the head off; frozen L00 re-read; W₀'s fall with Γ recomputed | fall between 0.039 (frozen) and 1.163 (target), L00 between −10.106 and −11.23 | zero update's recomputed gain exactly 0 | slot holds the mean, upstream exactly unchanged, downstream moved, gates moved; L00 on, (2,4) off, (3,5) off equal `amend2.json` (−10.0672, −10.1059, −10.0637; −8.8342, −8.8370, −8.8329); exit 1 | | fall near the frozen 0.0028 and the target's −0.09 |
| 2 | Share of L01 − L00 lost, head off in the Γ pass and the masked forward | undefined | (3,5) with Γ recomputed for (3,5): under 0.15 in size on all six | L01 and the frozen share equal `amend2.json`; exit 1 | (c): within 0.092 of 0.563, 0.545, 0.471. (a) or (b): under 0.25, frozen read 0.10 to 0.11 | within 0.092 of about 0 on every account |
| 3 | Mean gate at the marker on alive `o_proj` subcomponents of block 2 reading each head's slot | (a): head 4 closed, near 0.03 | the other five heads, same axes | none | reported | reported |

Outcomes, `copying.gamma_reading`: `head_acts_through_masks`, `head_more_than_target`, `still_other_way`, `in_between`, `not_specific`, `floor_high`,
`too_small`, `unreadable`. Lean, weak, filed: not `still_other_way`, since Γ's gates follow activations that the head moves by 7 to 8 log-odds in
the target model. Where it lands is not predicted.

**Telling a fall specific to the copying map from Γ breaking (review, F1).** Ten references go through one bar, `floor_high` at 0.15 in size, which
voids every reading. They are (3,5)'s share on all six states, and W₀'s recomputed fall over the map's mean gain, for (2,4) and (3,5) on both maps.
That quotient is what W₀'s reaction alone moves a net share by. A uniform break of Γ can leave a net share low, so `still_other_way` is guarded too.
Expected: W₀ copy under (2,4) at most 0.084; W₀ lookup under (2,4) near 0.015; W₀ under (3,5) near 0.0003; (3,5) on states near the frozen at most 0.0025.
Separately, the lookup under (2,4) is the same-depth reference, since (3,5) in block 3 changes fewer of Γ's inputs: unless every lookup seed is `within`,
the readings that say the head acts through the masks become `not_specific`. Each state's `gates_moved` prints how far Γ reacted.

**Tolerances, from Amendment 2's measured numbers.**
- Share against target, 0.092: the larger of 3 × √(0.0173² + 0.0063²) = 0.055, the largest row sds of the target and frozen masked shares, and the target's
  three-seed range, 0.5627 − 0.4710 = 0.0917. The range sets it. The recomputed share's sd is unmeasured, so `sd_weak` prints if it exceeds 0.0253, the sd at
  which the first term would reach the tolerance; no rule reads it.
- Repeat, 0 on the L40: Amendment 2 re-read `run.json`'s six shares and every masked cell at a difference of exactly 0 in another job, and `repeat_floor` read 0.
  The frozen path calls the same functions on the same windows. The compute partition holds only L40s. If the job moves to a pod, 1e-3 applies, Amendment 2's
  tolerance for another card, unmeasured and 7e-5 of a share at this gain.
- Slot: 0.5 of the read vector's squared norm and a gate of 0.1 are chosen here, no source. Row 3 stops nothing and no verdict reads it.

**Row 3 is kept.** "Reads head 4's slot" is a threshold on the read vector, so no new definition was needed. If no alive subcomponent passes, it reads `empty`.

### Blind to

- **Share with Γ recomputed.** A redirected query in the state: the masked forward never holds the fine-tuned head, only W₀'s q, k, v under a mask, and the
  update reaches it only through the gates. Reuse that lies in the weight change is not visible here (review, F2), so a low share does not rule out reuse.
  Which of Γ's inputs carries the reaction, as the Γ pass and the masked forward go off together, so the Γ-only corner is not read. A mean that sits off the
  masked forward's distribution, which does not cancel between L00 and L01, as they have different gates (review, C4). Positive control: none reaches a target-sized
  share, as no state is known to depend on a head through Γ; `gates_moved` shows only that Γ reacted.
- **The hook check.** One batch, W₀ only. Tells that the head reaches Γ's inputs, not how Γ uses them.
- **Zero update.** Exactly 0 by determinism whatever the hook does: a reference, not a control (review, Minor).
- **(3,5) as reference.** Block 3, so fewer of Γ's inputs change than under a block 2 head; the lookup covers that depth. The lookup's label is weak on its own
  as its gain is smaller, 5.7 to 6.4.
- **Target against masked.** Target reads are fp32, masked reads bf16 (review, Minor). Target shares come from `amend2.json`, not re-read.
- **Slot gate.** Unweighted, the marker position only, and blind to subcomponents that read several slots, which `slot_reads` leaves out. A closed slot says nothing
  about q, k, v upstream.
- **All.** One model, one marker, step 30, three seeds.

### Departures from the literature

- **Olsson et al. 2022** zero-ablate, on small models. We mean-ablate over ordinary text, as E62.
- **Running a parameter decomposition's importance function on ablated activations** has no source we know of. Γ trained on clean activations; this input is off that distribution by construction.
- **Thresholds.** 0.25 is Fable's, 0.15 E62's, 0.092 derived, 0.5 and 0.1 of the slot chosen here. None published.

### Amendment 2's "Not run" list, quoted

Quoted from the report as it stood when this section was written, while another agent was revising it. The revision names four accounts; the fourth is the review's F2,
the redirected query, which this read cannot see (Blind to).

> Separating the four accounts, each a $0 read on the saved states:
>
> - Γ recomputed with the head off, the one read of the third account, beside the lookup, (3,5) and W₀'s mask recomputed the same way. This is Amendment 3, coded and not run; the plan had listed it as not planned, and `ablated` must wrap Γ deliberately for it.
> - The head off on $L_{11}$ and $U_1$, with its attention, which reads the fourth account: how far the update's weights use the head.
> - An own-mask mean and zero ablation in row 3's cells, which settle whether the fall is removal or injection.
> - The o_proj subcomponents that read head (2,4)'s slot: how much of that slot's weight passes under W₀'s mask and under each state's mask.
> - The masked forward's head output against the mean it is replaced by, under each mask. If it already sits near the mean, the head's way is closed upstream, through q, k or v.
> - Head (2,4)'s attention at the marker in the masked forward, under each mask, beside the target model's. `copying.attention_patterns` exists for the target model.
> - The head off with every mask at 1 on the 200 read windows, $U_0$ and $U_1$. It joins row 1, which ran on repeated sequences, to the windows row 3 read.
> - Each of the 24 heads off in the masked forward at the state's mask: which head, if any, the masked gain does lean on.
> - Both heads off in the masked forward.
>
> Also not run:
>
> - W₀'s level under both heads, computed and discarded, so the state's own fall under both.
> - W₀'s log-odds under the head on repeated sequences, the other half of the 0.5 bar's reference.
> - The target model's attention at the marker, under W₀ and each state; does the fine-tune move the query.
> - Shares at steps 100 and 300, including the hypothesis's remaining prediction.
> - Downstream compensation after ablation.
> - $L_{01}$ for another in-context token; the lookup state's mask read for B; the copying state's mask read for the lookup's word; a map with its target in context and no prefix relation.
> - A head-level label for E58's copy, elicit and rank-1 maps; r for a random update of matched size at this target.
> - Which subcomponents the copying state's mask changes.
> - More seeds, other markers, other key counts; zero ablation.

This amendment runs "Γ recomputed with the head off, the one read of the third account", and a version of the `o_proj` item, the mean gate on those subcomponents, unweighted.

### Not run

Everything: the rule functions and the hook ran on CPU on a tiny model, and `amend3.py` was only imported; the 67M was never loaded. Row 1 is the new read's first test on a GPU. Not planned:
every other item quoted above; Γ recomputed with the head on in the masked forward, the Γ-only corner, which no filed rule reads; Γ recomputed under two heads; recomputed gates at the slot; the review's C1 to C5, which are for the Amendment 2 report.

## Amendment 4 (2026-11-02)

Spec: `reports/plans/2026-11-02_plan_regating-next-FABLE-DRAFT.md`, option 1; the user approved the test ("yes run the free test"). Rules:
`experiments/induction_cue/results/predictions_amend4.json`. Code: `experiments/induction_cue/amend4.py`, `cluster/induction_cue_amend4.sbatch`. Coded, not run. **Not
gated**: one cluster `debug` job, about 6 min, $0, no fine-tuning; one fresh review of the code before the job.

**In plain words.** Switching off old head (2,4) costs the fine-tuned model about half of what it learned, and the decomposition's masks lose most of their gain when
it is off, but only because the mask function reads the fine-tuned model's activations. So that tells us the masks react to the head, not that they found the old
mechanism. This test asks the direct question: does the fine-tune's mask open the same subcomponents the model's own copying opens, and does closing those subcomponents
take the mask's gain away?

**Changed after the code review, before anything ran.** `reports/reviews/2026-11-02_review_e62-amend4-code.md` found no wrong number but three readings that could not stand: nothing tested that `S_old` carries W₀'s copying, `a` mostly
separated a key token from a plain token, and `not_shared`, which stops the 67M copying work, fired on data that refutes the null. Rules, predictions, code and tests were changed together on 2026-11-02; this section states them as they now stand, and "Changed after the code review" below lists each change. The first filing is 56c6981.

**Restated in the Terms' words.** Γ's masks and the least-squares split as in Amendment 3; the masked gain is L01 − L00, original subcomponents, Δ out. The user's request:
on the six saved step-30 states and the 200 read windows, compare each state's mask at the marker with W₀'s own mask at the second A, where W₀ copies on its own, and read
how much of the masked gain goes when the subcomponents W₀'s copying opens are returned to W₀'s gates. No training, no new split, no ablation inside Γ's pass. On the
copying map a mechanism W₀ already has should show; on the lookup, at the same marker and windows, it should not.

**Claim.** On the copying map, the fine-tune's mask at the marker opens the subcomponents W₀'s own mask opens when W₀ copies, and the masked gain runs through them; on the
lookup it does not. **Null.** The copying state's mask opens subcomponents unrelated to W₀'s copying subcomponents, no more aligned than chance and no more than the lookup's;
closing W₀'s copying subcomponents costs the masked gain no more than closing as many random opened subcomponents.

**Granularity limit.** A new circuit built from old subcomponents reads as elicited to VPD. `shared_pieces` says the state's mask opens the subcomponents W₀'s copying opens and
the gain runs through them, at the level of VPD's rank-one subcomponents. It cannot say the same circuit is reused rather than a new circuit of those subcomponents, and the
head label is at the level of heads.

### Design check

- **Goal.** `GOALS.md`, new capability or strengthened old: whether VPD's masks read the mechanism a causal head label names, on the only 67M map that has such a label. A step, not a patch.
- **Nearest tried.** Amendment 3 (TRIED row 136): the masks react to the head through Γ; it could not say which subcomponents. Amendment 2's slot read covered 5% of one head's slot
  (check F2). E55 C2 (row 117) is cancelled, so no other decomposition-level mechanism test exists. Nothing has compared a state's mask with W₀'s own mask at a position where
  W₀ performs the behaviour. E59 and E64 read linear directions, not masks.
- **Decisions.** Row 73 (granularity limit) kept and stated. Row 112 (mechanistic definition; r a candidate instrument to score against a causal check): this is that scoring.
  Row 114 (held-out rule demoted): not read. No departure.
- **What changes next.** `shared_pieces` → the claim and the Qwen expectation. `not_shared` → the stop: r reported as "the original subcomponents can write this target under
  some mask", the 67M copying work on r's strength stops, and the Qwen table carries the split's numbers only. `in_between` → the lead reads which bar missed; the step-300 states
  are the draft's next read only if that matters for Qwen's choice of step. `control_failed_mask` → the masks drop W₀'s copying, to the user, and **the weighted slot read is the next step
  to size it** (then filed, with a reference slot). If the control passes, the weighted re-open is cut. `control_failed_target` → W₀ does not copy through this head here, which says nothing of the masks: to the user; the draft's fallback for it (repeated sequences for `a`) has no code. `old_pieces_not_carrying` → the contrast did not find W₀'s copying subcomponents, so the test says nothing either way, and the numbers are reported. `w0_gate_gain_too_small` → W₀'s gates do not depend on the pair at the second A, so `a` names nothing, and the numbers are reported. `k_too_small` → no k subcomponents could reach the bar; nothing about the mechanism is read.
- **Apparatus first.** W₀'s own mask at the second A, target and masked, intact, head off and pair-broken, runs first and alone with exit 1; then whether `S_old` carries W₀'s copying; the re-read of the masked gain against `amend3.json`
  is the determinism check.
- **Class.** 1, analysis of saved states with a GPU pass of the existing harness: approved, no sign-off. The lead tells the user the Amendment 3 reversal first, as the plan requires.
- **Cut.** The positive control over all 24 heads: Γ's construction predicts its result, and it would move no reading here. The one head worth a number, (2,3), is a reported column, no rule.

### Checked before coding

**Gates are per position and reachable.** `Reads.gates` returns `route_cells(...)["gates"]`, a list per batch of `compute_ci`'s `ci.g[k]`, (B, T, C), on the CPU, T = 41 under `ci_len`. Its `half`
only reads the log-odds, so the gates do not depend on it (`moved` in `amend3.py` indexes `[:, half, alive]`; `tests/test_copying.py` asserts both on a tiny real `ComponentModel`).
`Reads.masked` takes the list, and `masked_lo_windows` reads at any position given. So the read at the second A is the same `Reads` with `half − 1`, and closing is an edit to the list.

**W₀'s gates at half − 1 and half − 2 come from one pass.** `stage1` calls `R.gates(w)` once on the copying map's 200 windows, with no hook, and takes both positions from it; the pair-broken windows have their own pass, also with no hook. The two maps' windows differ
only in the target column, which no forward reads; `load` checks that the first 41 columns are equal.

**The windows.** `cue_windows` sets A at half − 1 and at p, B at p + 1, p ≤ half − 4, and keeps a window only if A occurs exactly twice and B once. So half − 1 is the second A and half − 2 is neither A nor B. `stage1` asserts it on the real
windows (`copying.second_a_holds`) before any forward.

**The head off is in the masked forward only.** `R.masked(w, g0, H)` wraps only `masked_lo_windows` in `ablated`; `g0` was read outside it. The target row, `R.lo(w, H)`, ablates the target model.

**Γ reads the marker when it writes the gates at the second A.** Γ here is bidirectional (`reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`). The draft's line "the model does not know that" holds for the target model and not for Γ: `a` is read on windows
with M at half and the 20 tokens after it in Γ's view. The pair-broken contrast keeps them in both arms, so this largely cancels in `a`.

**The pair-broken windows.** `CP.broken_pair` as `amend2.py`, `control_seed` 62. **Confirmed on the real `windows.pt`** (CPU, no model): on all 200 windows exactly one position differs, the first A; the second A is the same token at the same position; A then occurs once in the window; B, the marker and half − 2 are unchanged. `copying.pair_broken_holds` asserts this at run time. **That W₀ loses the copy there is measured at the marker only**: lean 1.58 (`amend2.json`), prior 3.15e-4 against 6.5e-5 (`run.json`). At the second A it needs a forward, and this agent does not run a GPU, so the confirmation is in the code: the target's lean at the second A under 1.5801 exits 1 as `no_pair_dependence` before `a` is used. A miss goes to the user.

**Two figures in the draft are out of date or too weak.** Its 4.3e-4 for W₀'s P(B) at the marker is job 48507's, on windows Amendment 1 dropped; these 200 windows read 3.154e-4 (`run.json`, `stage_b.prior_copy`). Its P2 implies its P3.

### Definitions

Every gate is Γ's value on the 200 read windows, alive subcomponents only, 9,881 across the 24 matrices (`alive.pt`, counted), `ci_len` 41; every mean is over windows.

- `a` = W₀'s mean gate at the second A (half − 1) on the intact windows minus W₀'s mean gate at the second A on the pair-broken windows: what W₀'s mask adds because a copy is available. Same key, same position, one further Γ pass.
- `a_half2` = W₀'s mean gate at half − 1 minus at half − 2, intact windows, the first filing's `a`. Reported beside, no rule.
- `S_old` = the fewest subcomponents holding half of `a`'s positive mass, largest first, ties in index order, reaching exactly half counts. Its size k is printed.
- `Δ(state)` = the state's mean gate at the marker minus W₀'s mean gate at the marker.
- **Alignment** = cosine(Δ(state), `a`). Chance = the same cosine with `a`'s entries permuted over the alive set, 20 draws, mean and sd.
- **Closing share** = 1 − (masked gain with `S_old`'s gates set to W₀'s values at every position, under the state's mask otherwise) / (masked gain L01 − L00). Reference = the median over 5 draws of k
  subcomponents from those the state opened at the marker (Δ > 0) outside `S_old`; if fewer than k, from all alive outside `S_old`, printed. Closed the same way.

- **W₀'s closing share**, read as stage 4 reads a state, on one set of tokens, the intact ones. Gate gain = masked forward at W₀'s gates minus at the pair-broken gates. Share = 1 − (closed − at the pair-broken gates) / gate gain, with `S_old`'s gates set to their per-window pair-broken values at every position. Reference = 5 draws of k from the entries with `a` > 0 outside `S_old`. Token part = the pair-broken gates on the intact tokens minus on the pair-broken tokens, printed.

### Predictions and verdicts

Bars from E62's filed 0.5 "old" and 0.2 "not", E58's 0.24 band gap; the cosine bar 0.2, the 3 × chance factor and the half are the draft's, no source.

- **Stage 2, `S_old` carries W₀'s copying.** W₀'s closing share ≥ 0.5 and above its random median by ≥ 0.24, P2's bars: a state's closing cannot be read as shared if `S_old` does not carry W₀'s own copying. A gate gain under 1.0 is unscored and `a` then names nothing. Neither is a stop.
- **P1, alignment.** Copying map ≥ 0.2 and ≥ 3 × chance on all three seeds; lookup under half of the copying map's, same seed, on every seed.
- **P2, closing.** Copying map's closing share ≥ 0.5 on every seed and above the random median by ≥ 0.24; lookup's ≤ 0.2.
- **P3, the gap.** Copying minus lookup closing share ≥ 0.24 on every seed. P2 implies it, so it decides no verdict; it is reported, and says whether a P2 miss still shows a gap.

`copying.mask_test`, first that fires: `unreadable` (the control's numbers not finite); `control_failed_target`; `control_failed_mask`; `unreadable` (k = 0, k over 4,940, a seed missing, a gain not finite); `too_small` (any gain under `local_faith.MIN_GAIN`, 1.0); `unreadable` (any other number not finite); `w0_gate_gain_too_small` (stage 2's gate gain under 1.0), `old_pieces_not_carrying` (stage 2 missed), each with P1-P3 still reported; `shared_pieces` (P1-P3 hold on every seed); then the null, copying map alone, either arm on every seed: (a) share − random median under 0.24 and (cosine under 0.2 or under 3 × chance), (b) share at or under the random median. It reads `not_shared` if `largest_k_share` ≥ 0.5 on every seed, else `k_too_small`. `in_between` is anything else. `verdict` is the same dict on a stop and on a finished run.

**Edge cases, run before filing (`tests/test_copying.py`).** `a` with no positive mass: k = 0, `unreadable`. `S_old` disjoint from the opened set: share 0, `not_shared` at a random median of 0 or more; at a negative median, `not_shared` only if the alignment also misses. A
negative closing share: a miss on the copying map, and it holds the bar on the lookup. Ties in `a`'s ordering: stable, index order. Fewer than k opened: the fallback pool, printed. An empty pool: unreachable, as k ≤ half the alive set leaves at least k outside `S_old`; `random_sets`
asserts. A zero or negative gain: `too_small` before any share is read. A zero-norm Δ: cosine nan, `unreadable`. Negative chance: the 0.2 bar decides. All eight combinations of P1-P3 are enumerated: six reach exactly one verdict, the two with P2 and not P3 cannot occur. A 2,520-case grid of cosine, share, median, lookup, `largest_k_share` and carries is checked against the rule written out apart from the code. The review's three failing cases are tests.

**Departures from the draft, each a fix of a rule found degenerate or a stale number, and each changing what the draft wrote.**
1. `not_shared` is in the null's terms, copying map only (review H3). As first written, a copying map at cosine 0.5 and share 0.9 beside a lookup at 0.3 and 0.25 read `not_shared`, and so did cosine 0.19 and share 0.49 against random 0.02; both refute the null, and both now read `in_between`. **This changes the draft's meaning.**
2. "At or under its random median" is read on every seed, as the other arm of that branch is; the draft did not say. One bad seed of three reads `in_between`.
3. The random pool excludes `S_old` throughout. As written, a mask opening exactly `S_old` drew `S_old` as its own reference and read `not_shared`. **This changes the draft's meaning**, only where the opened set is small.
4. P(B)'s bar is 10 × 3.154e-4 = 3.15e-3, not 4.3e-3: the same "ten times the marker's", on the windows actually read.
5. A target fall at the second A under 1.1628 stops the run as `head_not_carrying`. The draft had it as expected only, and with a fall near 0 the quarter bar passes with nothing to test. A false stop costs a resubmission at $0.
6. `largest_k_share`, the closing share of the k largest openings of that state, gates `not_shared`: below 0.5 on any seed the branch reads `k_too_small` (review H3). **This changes the draft's meaning**: the draft's stop no longer fires when no k subcomponents could reach the bar.
7. `a` is the pair-broken contrast, not the draft's half − 2, which is reported beside (review H2). **This changes the draft's meaning.**
8. The new check and verdicts `old_pieces_not_carrying` (review H1) and `w0_gate_gain_too_small` (re-check R1). **This changes the draft's meaning**: either blocks both readings.
9. The draft's within-2 level bar is replaced by the masked lean at least 0.25 of the target's (review M1, re-check R2); `control_failed` is split by cause (review L1).

### Changed after the code review

Why each, from `reports/reviews/2026-11-02_review_e62-amend4-code.md`. Nothing had run, so no filed number was seen.
- **H1.** Stage 2 added, with `copying.old_carries` and `old_pieces_not_carrying`. Without it a closing share of 0.02 could mean "the set never held the copying". The review offered exit 1; it is a verdict, not a stop, so the states are still read. Its optional print of `S_old`'s overlap with (2,4)'s slot is not added.
- **H2.** `a` is the pair-broken contrast; half − 2 is beside it. With half − 2, key-reading subcomponents could fill `S_old`'s half of the mass, leaving copying ones out, so a low share would stop the line wrongly.
- **H3.** `not_shared` rewritten and gated, as above.
- **M1, then R2.** The bar reads the lean, not the level, an offset every later number nets out, already 2.01 at the marker. Levels are still printed. First filed as within 2 of the target's lean; the re-check ran target 1.7 with masked −0.25 and 0.05, which passed with no lean, and target 3.0 with masked 5.2, which stopped. It is now one-sided, at least 0.25 of the target's, as the fall's bar.
- **R1, the re-check.** W₀'s closing share was over the masked lean, which moves tokens and gates, while closing moves gates only. A lean of 3.0 with 2.2 through the tokens capped every set at 0.27, and strong states read `old_pieces_not_carrying` for no reason about `S_old`. It now uses one set of tokens, as stage 4, with the token part printed, and a gate gain under `min_gain` is its own verdict.
- **M2.** `all_alive_share`, W₀ and each state, reported: the read's only end-to-end positive control, and a bound on any `S_old`. 29,031 of 38,912 subcomponents are not alive and cannot be closed.
- **M3.** `largest_outside_share` and `largest_overlap`, reported: "these rather than others".
- **L1.** `control_failed_target` and `control_failed_mask`, with `verdict` one dict shape.
- **L2.** `cos_positive`, reported.

### Reading notes

- **L3.** Alignment reads the marker; closing reads every position. A gain from openings at earlier positions (B, the second A) moves the closing share and not the cosine, so P1 and P2 can disagree for a reason no number names. Read a split between them with that in mind.
- Read `all_alive_share`, `largest_k_share` and `largest_outside_share` beside any P2 miss.

### Stages, in run order

Cheapest first; `results/amend4.json` after each stage and each state; every number printed to the log. A miss in stage 1 or 3 is exit 1; stage 2's is a verdict. Precision on the L40, as Amendment 3: Γ and the masked forward bf16 (autocast), target model fp32, `allow_tf32` off, batch 64.

| Number | W₀ | Do-nothing (zero update) | Control | Expected, copying map | Expected, lookup |
|---|---|---|---|---|---|
| Stage 1. Target model at the second A: P(B), geometric mean; log-odds intact, (2,4) off, pair-broken | P(B) ≥ 10 × the marker's 3.154e-4; fall over 1.1628; lean over 1.5801 | same as W₀ | exit 1 under each: `no_copying`, `head_not_carrying`, `no_pair_dependence`; `control_failed_target` | not read per state | not read |
| Stage 1. Masked at the second A under W₀'s own mask: intact, (2,4) off in the forward, pair-broken under its own mask | lean ≥ 0.25 of the target's; fall ≥ 0.25 of the target's; level printed, no bar | same | exit 1: `mask_no_lean`, `mask_drops_head`; `control_failed_mask` | | |
| Stage 1. k = \|`S_old`\| | printed; tens to a few hundred, `a`'s mass unmeasured | same | exit 1 as `unreadable` if 0 or over 4,940 | | |
| Stage 1. Reported only: `half2_k`, `half2_overlap`, `cos_a_half2` | no expected value | | | | |
| Stage 2. W₀'s gate gain at the second A, one set of tokens | ≥ 1.0 | | under it: `w0_gate_gain_too_small`, no stop | | |
| Stage 2. W₀'s closing share at the second A; its random median | ≥ 0.5, and ≥ 0.24 over the median | | a miss: `old_pieces_not_carrying`, no stop | | |
| Stage 2. Reported only: W₀ with every alive subcomponent closed | about 1, up to non-alive gates | | | | |
| Stage 2. Reported only: the lean's token part | no expected value | | | | |
| Stage 3. L00; re-read L01 − L00 | L00 equals `amend3.json` | gain exactly 0; level under closing exactly L00 | equal `amend3.json`'s `M` within 1e-3 (0 expected on the L40); exit 1 | 13.8 | 5.7-6.4 |
| Stage 4. Alignment cosine | undefined (Δ = 0) | undefined | chance ≈ 0 ± 0.01 (permutation) | ≥ 0.2, lean 0.3-0.5 | under half the copying map's |
| Stage 4. Closing share | undefined (no gain) | 0/0, reported as the level above | random median; 0.05-0.2 | ≥ 0.5 | ≤ 0.2 |
| Stage 4. Closing share of the k largest openings; gates `not_shared` | | | under 0.5: `k_too_small` | at or over the closing share | at or over the closing share |
| Stage 4. Reported only: every alive closed | | | | about 1, up to non-alive gates | about 1 |
| Stage 4. Reported only: k largest outside `S_old`; overlap of the k largest with `S_old` | | | | under the closing share if these rather than others; overlap printed | no expected value |
| Stage 4. Reported only: cosine of the positive parts | undefined | undefined | | near the cosine or under it | no expected value |
| Stage 4. Reported only: half − 2 contrast's cosine and closing share | undefined | | | no expected value | higher than with `a`, as it holds key-reading subcomponents |
| Stage 4. Reported only: (2,3)'s target share | | | | no expected value; W₀'s second prefix head, slot opens as far as (2,4)'s | no expected value |

### Noise floor

**The 1e-3 re-read, from measured evidence; nothing new is guessed.** Amendment 3 re-read `amend2.json`'s L00 and L01 in another job on the L40 at a difference of exactly 0 on all six states and both maps (`amend3.json`, `diff_a2`; both hosts "NVIDIA L40"; the compute partition holds only L40s). Amendment 2 read
`run.json` the same way, and `repeat_floor` read 0. Stage 3 makes the same calls on the same windows with the same batch and flags, so the floor is 0. The tolerance, 1e-3, is the brief's and Amendment 3's other-card value: 7e-5 of a share at a gain of 13.8, so it cannot change a reading.
The run prints `max_diff` and `exact`. A miss is a stop to look at, and a difference that is not 0 on the L40 is worth a look under the stop. The zero update's checks take a tolerance of 0, as Amendment 3 measured.

**The rest.** Row sd by `copying.share_sd`'s bootstrap on the closing share, printed. Three-seed range 0.09 from E62. 20 permutations for the cosine floor. No rule here compares two runs other than stage 3.

### Blind to

- **Alignment and `S_old`.** Subcomponents W₀'s copying uses at the first A or at B rather than at the second A, since `a` is read at one position. Copying that does not change gates between intact and pair-broken windows: subcomponents open in both are not in `a`, however much copying runs through them, and stage 2 cannot find them either. `a` carries whatever else breaking the pair changes, as Γ reads the other key at p. A pair-broken window still holds B, so a lean toward B is not all removed. Γ reads the marker and the 20 tokens after it; both arms hold them. Positive control: stage 2, whether `S_old` carries W₀'s copying; nothing shows a known-shared mask reads high.
- **W₀'s closing.** It acts at every position, so at the first A `S_old` takes gates read on another key.
- **Closing share.** A redirected query that lies in the weight change: the masked forward holds W₀'s q, k, v (account 4, unchanged). Closing returns gates to W₀'s values, so where the state's gate is under W₀'s it raises it; and it acts at every position, so `S_old` disjoint from the opening at the marker is share near 0, not exactly 0. A
  mask whose gain is spread over many subcomponents can miss 0.5 at a small k whatever it shares: `largest_k_share` is the read of that, and it gates `not_shared`. Non-alive subcomponents are never closed: `all_alive_share` bounds that.
- **Random reference.** Δ > 0 has no floor, so most of the pool may be barely open, and the reference is then near 0 and weak; P2's 0.5 does the work, and a near-zero median cannot decide `not_shared` alone, as arm (a) needs the alignment too and arm (b) needs a share at or under it. `n_opened`, `old_opened` and `random_shares` are printed.
- **The 3 × chance bar.** At chance near 0.01 the 0.2 bar decides, so this bar seldom binds.
- **P3.** Decides nothing, as above.
- **Target against masked.** Target reads fp32, masked bf16, so the quarter bars compare differences across precisions. The lean nets the offset (2.01 at the marker) as far as it is the same on both windows; at the marker the two leans are 1.58 and 0.37.
- **All.** One model, one marker, step 30, 32 keys, 200 windows, three seeds, one decomposition seed.

### Departures from the literature

- **Olsson et al. 2022** zero-ablate, on small models. We mean-ablate over ordinary text, as E62.
- **Comparing a parameter decomposition's masks between positions and between a window and its pair-broken twin** to name a mechanism has no source we know of. The VPD paper reads masks per input; `a` as such a difference is ours.
- **Thresholds.** 0.5 and 0.2 E62's, 0.24 E58's, 2 and 0.25 and 0.2 and 3 × and the half the draft's, 3.15e-3 and 1.1628 derived. None published.

### Not run

Everything: the rules ran on CPU, the stages, with every review fix, ran on a stub `Reads` with no model, and the 67M was never loaded. Stage 1 is the first test on a GPU. Not planned: the first-A contrast and `S_old`'s overlap with (2,4)'s slot, the review's two optional prints; the weighted slot read, unless the control fails; the 24-head control, cut; ablation inside Γ's pass; zero ablation; steps 100 and 300; the head off on L11 and U1; a second decomposition seed, the only check that narrows the granularity limit.

## Amendment 5 (2026-11-02)

Spec: `reports/plans/2026-11-02_advice_e62-amend4-control-FABLE.md`, option 1, chosen by the user from a dialog. Rules: `experiments/induction_cue/results/predictions_amend5.json`. Code: `experiments/induction_cue/amend5.py`, `cluster/induction_cue_amend5.sbatch`, `copying.py`'s last section. Coded, not run. **Not gated**: one cluster `debug` job, about 2 min, $0, no fine-tuning; one fresh review of the code before the job. Additions from the Amendment 4 check (`reports/reviews/2026-11-02_review_e62-amend4-check.md`, F1, F2, F4, F5) are below where they fall. Amendment 4's filed verdict, `control_failed_mask`, stands; nothing here rescores it.

**In plain words.** Amendment 4 found that the decomposition's forward, under the mask the original model gets when it copies, still copies but barely notices when the original copying head is switched off. This job first checks that switching a head off works in that forward at all, with every mask at 1. Then it switches off each of the 24 heads in turn, in the real model and in the masked forward, to see whether the masked forward's copy leans on other heads, leans on every head less, or has lost that one head alone.

**Restated in the Terms' words.** At W₀ only, at the second A (half − 1), log-odds of B on E62's 200 read windows. The masked forward holds W₀'s original subcomponents, Δ out, each under Γ's mask at W₀. Read (2,4) mean-ablated in the masked forward with every gate at 1, stopping if its fall is under a quarter of the target model's. Then mean-ablate each of the 24 heads one at a time, in the target model and in the masked forward, always in the forward and never in Γ's pass. Beside it, the two swapped forwards: W₀'s pair-broken gates on the intact tokens, W₀'s intact gates on the pair-broken tokens. The user's words: "whether the mask reroutes copying through other heads or thins the whole pathway".

**Question.** Does W₀'s mask move W₀'s copy at the second A off (2,4) onto other heads (`rerouted`), reduce every head's part alike (`thinned`), or take out (2,4) alone and leave the other heads' parts (`labelled_head_singled_out`)?

**Granularity limit: heads, not circuits.** A verdict names which heads the forward's log-odds of B depends on when each is replaced by its mean. A forward whose copy ran entirely through (2,4) could still run a different circuit of that head's subcomponents, and VPD would read it as the same mechanism; a new circuit built from old subcomponents reads as elicited to VPD. `rerouted` does not say the other head does what (2,4) does in the target model. No verdict here reads a fine-tuned state, new capability against strengthened old, or r. The only check that narrows this limit is a second decomposition seed, not run.

### Design check

- **Goal.** `GOALS.md`, new capability or strengthened old: whether VPD's masks read the mechanism a causal head label names. A cause for Amendment 4's miss at the level the label lives at. A patch on the last experiment, and the sentence the Qwen table's caption needs either way.
- **Nearest tried.** Amendment 4 (TRIED row E62 Amendment 4): stopped at this control, with one head read in this forward at this position. Amendment 3 (row 136) read the head through Γ at the marker. Nothing has switched off any other head in the masked forward at W₀ at the second A.
- **Decisions.** Row 73 (granularity limit) kept and stated. Row 112 (r a candidate instrument scored against a causal check): this asks whether the causal label can be read in this forward at all. No departure.
- **Class.** The user chose it, from the dialog above.
- **Apparatus first.** Head off with every mask at 1, alone, with exit 1; then the re-read of `amend4.json`. Nothing here reproduces a procedure's end state.
- **What changes next.** `rerouted` → the weighted slot read and the re-open at W₀; a state-level test needs its own plan anchored on the masked forward's carriers; the Qwen table carries the split's numbers with the caption that the masked forward at W₀ reroutes the labelled behaviour around its labelled head. `thinned` → the slot read is still the sizing step, a state-level test is not worth building, and `token_reading` says whether account D is live; the caption says "thins". `labelled_head_singled_out` → the mask costs that head and leaves the others: the slot read comes first, as it is that head's slot in question. `mixed` or `no_comparison_head` → the lead reads which heads split, and the caption says the sweep did not separate these. `ablation_does_not_transfer` → an apparatus finding, which touches Amendments 2 and 3's frozen reads: to the user before anything else; next, the hook against the decomposed `o_proj` at the second A with zero ablation beside the mean. `reread_failed` → to the lead; nothing is read.

### What the control adds to Amendment 2's

`amend2.json`'s `row1_control` shows the hook works in the masked forward with every gate at 1: top-1 0.82 to 0.0095, on repeated random tokens. It says nothing at this position, on these windows or in this unit. Stage a reads the same hook at the second A on the cue windows, in log-odds of B, intact and pair broken, against the target's 4.82 and 6.84 from the same windows. It can fail where the top-1 read could not: a top-1 falling from 0.82 to 0.01 needs only that the answer stops being first. If it passes, a low fall under W₀'s mask comes from the mask and not from how the hook acts in a decomposed forward at this position, which removes account B as an apparatus artefact at every gate 1. It does not remove it under W₀'s mask, where the replacement mean still passes masks the all-ones row does not have.

### Definitions

A **fall** is a forward's log-odds of B with no head off minus with the head replaced by its target-model mean (`heads.pt`) at every position, intact windows, mean over windows, gates as read with every head on. T is the target model's, M the masked forward's under W₀'s mask, Mb the masked fall on the pair-broken windows under their own mask, Tb the target's there. **Excess** = M − T, with `excess_sd` its window bootstrap sd. **Fraction** = M / T where T ≥ 0.5. **Pair part** = the fall minus the same head's fall on the pair-broken windows, `pair_part_T` and `pair_part_M`, each with an sd. **Lean** = intact minus pair-broken, each under its own mask. **Token parts**: the masked forward's intact minus pair-broken tokens, gates held at the pair-broken values and at the intact values.

### Verdicts

`copying.sweep_reading`, on the bf16 rows, first that fires; exactly one fires.

1. `unreadable`: a head missing, a number not finite, the masked lean under 1.0, or (2,4)'s T under 0.5.
2. `rerouted`: some head other than (2,4) is a carrier: M ≥ 1.0, M − T ≥ 1.0, M − T ≥ 2 × `excess_sd`, and M − Mb ≥ 1.0. Ties count.
3. `no_comparison_head`: no carrier, and no other head has T ≥ 0.5.
4. `thinned`: every other head with T ≥ 0.5 has a fraction within 0.24 of (2,4)'s. A tie counts.
5. `labelled_head_singled_out`: every such head's fraction is over (2,4)'s by more than 0.24.
6. `mixed`: anything else, including a head whose fraction is under (2,4)'s by more than 0.24.

Before these, in stage a and stage c: `ablation_does_not_transfer` and `reread_failed`, each exit 1. Beside, deciding nothing: `fp32_verdict`, the same rule on the fp32 rows; `precision_moved`; `token_reading`; `ranking_holds`; `falls_over_lean`; `sum_falls_over_lean`.

`copying.token_reading`, account D: `unreadable` (lean under 1.0 or a number not finite); `through_tokens` both token parts over half the lean; `through_gates` both under; `order_dependent` otherwise, ties included.

**Where each bar comes from.** 1.0: the advice's "about a log-odds or more", and `local_faith.MIN_GAIN`; the excess it is set against is 1.40 log-odds, the masked copy outside (2,4) at 3.42 against the target's 2.02. 2 sds: chosen here, no source; the 1.0 does the work at the sds expected. 0.5, the target fall that counts as a denominator: chosen here, half of 1.0, no source, unmeasured per head until this job runs. 0.24: E58's band gap, as E62's `r_gap`; the advice's "similar fraction" has no number. The token bar 0.5: E62's `share_old_min`, here the plain majority. The quarter and 1e-3: Amendment 4's.

**Edge cases, run before filing (`tests/test_copying.py`).** A sweep where no head falls: `no_comparison_head` with no other denominator, `thinned` with one; `unreadable` if the target's labelled head has no fall either. Ties: as above. A zero or negative denominator: a target fall under 0.5 gives no fraction; the labelled head's under it is `unreadable`; a filed target fall at or under 0 is `unreadable` in the control; a masked lean under 1.0 is `unreadable`. A total fall over the masked lean: 24 falls summing over it, or one head's over it, decide nothing and are flagged, since falls need not add up to a lean. A case no branch covers: none, as `mixed` is last; a 1,152-case grid checks the verdict against the rule written out apart from the code. A negative fraction: `mixed`. A general drop (M − Mb under 1.0), an excess under 2 sds, or an M under 1.0: not a carrier.

### Stages, in run order

Cheapest and control first; `results/amend5.json` and `results/large/amend5_windows.pt` (every read, per window) after each stage and before any stop; `results/large/amend5_vectors.pt` (`a`, `a_half2`, `S_old`) straight after Γ's two passes, before anything can stop; every number printed. Same windows, batch 64, `allow_tf32` off, precision as Amendment 4 but for the rows named fp32. The run refuses if `predictions_amend5.json` is not committed unchanged, or if any of its outputs exists.

| Stage and number | W₀ | Do-nothing (no head off, read twice; the zero update reads as W₀ here) | Control | Blind to |
|---|---|---|---|---|
| a. Every mask at 1: fall under (2,4), lean | fall near 4.82; lean near 6.84; **under 1.204 exit 1** | fall 0 | this row is the control, for the sweep | what W₀'s mask does to the replacement mean |
| a. Re-read of `amend4.json`'s stage 1, every number; `exact` | target −4.710, −9.527, −11.545; masked −7.300, −8.216, −11.641; k 172; **over 1e-3 exit 1**; expected 0 | repeat exactly 0, **over 1e-3 exit 1** | | |
| b. Falls and leans with window bootstrap sds | sds unread; `amend4_miss_reported_only` beside, rescoring nothing | | | keys, below |
| b. Masked (2,4) fall: bf16, fp32 under bf16 gates, fp32 under Γ's own fp32 gates | expected within the sd; over it a LOUD line, no stop | | | target model in bf16 |
| b. Three masked levels: mean, sum, distinct values, smallest gap, both precisions | no lattice; the intact level reproduces −7.30000003 through the re-read | | | |
| c. T, Tb, pair_part_T per head | (2,4): T 4.82 within 1e-3, else exit 1; every other T under it (`ranking_holds`); pair_part_T large for (2,4), small for (3,5) | 0 | (2,4)'s pair part is the pair guard's positive control; at the marker 1.073 of 1.163 | |
| c. M, Mb, pair_part_M per head, bf16 | (2,4): M 0.92 within 1e-3; other heads no expected value, which is the reading | 0 | | below |
| c. M32, Mb32 per head | `fp32_verdict` the same as the verdict; a disagreement is a LOUD line, and the reading goes to the user with both | | | |
| d. The four swapped differences | each token part plus its gate part is the lean, by algebra; `token_reading`, no expected value between its branches | | | gates at positions other than the second A move too |

**Why stage b does not stop.** A precision move leaves nothing to save by stopping: every later read costs seconds, and the sweep's fp32 rows are what a move would need next. It is loud, and the reading is the user's. The filed verdict stays bf16's, as Amendment 4 read.

**The checks' additions.** F1: 4.82 mixes what the head does for the pair with what mean ablation at every position does to B's log-odds anyway. Head (2,4) off on the pair-broken windows, target and masked, is `Tb` and `Mb` in the sweep's (2,4) row, and the 23 other heads, (3,5) labelled, are the reference at this position. It decides no verdict; if `pair_part_T` reads small, the fall is not shown to be copying here, and that is reported first. F5: −7.30000003 has no known mechanism, and it enters both masked differences. The three masked levels per window test it: the re-read must reproduce it, and `round_level` gives the distinct values and the gap.

### Noise floor

Window bootstrap, 2000 draws, for every fall, excess and fraction, printed; this gives Amendment 4's miss the sd it lacked. The re-read's floor is 0 on the L40 (`amend3.json`, `diff_a2`), and the do-nothing repeat measures it again in this job. The bf16-against-fp32 difference is measured in stage b, and the rule that uses it is loud only, so it needs no tolerance of its own. The 200 windows share 32 keys, so the bootstrap is a lower bound on the noise over keys; no rule leans on it but the 2 sds. One model, one marker's windows, one decomposition seed.

### Blind to

- **The lean and the falls are not the same kind of number (check F4).** A fall holds the gates fixed and moves one head. The masked lean moves tokens and gates together, each set of tokens under its own mask. So the sweep's falls and the lean are not like for like: a fall of M against a lean of 4.34 is not a share of the copy, "falls over the lean" may happen and decides nothing, and the 1.40 is a difference across two kinds of read. Mb's lower point is under the pair-broken mask too. Stage d splits the lean into tokens and gates, and reads no head.
- **Mean ablation in the masked forward.** The mean is the target model's, passed through masks that were not read for it, so a fall nets a removal and an injection, in the slot of a head whose q, k, v and `o_proj` are all masked. M − Mb guards against an injection that drops B's log-odds whatever the pair; it does not make M a removal alone. Zero ablation and own-mask means are not run.
- **Frozen gates.** Γ reads the head on, so any effect of a head through Γ is out: Amendment 3 read that at the marker, not here.
- **Redundancy.** One head at a time cannot see two heads that back each other up; both read a small fall. `thinned` and `no_comparison_head` cannot tell that from a thin pathway.
- **Spread rerouting.** Twenty heads each carrying 0.07 more than in the target model is no carrier. It reads `thinned`, `mixed` or `labelled_head_singled_out`; the excess column and `sum_falls_over_lean` show it.
- **Fractions.** Heads under T of 0.5 are not compared, however much they carry in the target. A target fall of 0.5 with its sd is a coarse denominator; `fraction_sd` is printed.
- **Downstream heads.** A later head reads what an earlier one wrote, so a fall in block 3 can be the same path as one in block 2. Heads are not independent routes, and `rerouted` names a head the forward depends on, not a separate route.
- **Position and token.** The second A only; other positions and the marker are not read. Word-like windows, B set in by construction, not repeated random tokens.
- **Target's own reading.** T is also an intact-windows number; the pair part sits beside it, and the rules use T for the denominator and the excess, as the advice files them.

### Departures from Fable's advice

1. `labelled_head_singled_out` and `mixed` are added. Amendment 4's numbers fit a third pattern besides rerouting and uniform attenuation: the mask costs (2,4) its fall and leaves the other heads' falls near the target's. Unnamed, it would read as a miss of both.
2. The sweep has pair-broken rows, 96 reads for the advice's 48, as M − Mb guards the carrier and F1 asks for them.
3. The swapped forwards take a reading with a rule where the advice had none, as a number needs an expected value. It decides nothing about the sweep.
4. fp32 rows for all 24 heads, not the reference rows alone, about 15 s, so that a precision move needs no second job. fp32 under Γ's own gates, (2,4) only, so that the check can fail on rounding in the mask as well as in the forward. Unscored.
5. The all-ones row reads through `Reads.masked` under gates of 1, the same forward as `AllGatesOn` (asserted in `tests/test_copying.py`), as `read_lo` needs a net with `at`.

### Departures from the literature

- **Olsson et al. 2022** zero-ablate one head on small models, reading induction on repeated random tokens. We mean-ablate over ordinary text, as E62, on word-like windows at the second A.
- **Switching a head off inside a parameter decomposition's masked forward, gates frozen,** has no source we know of.
- **Thresholds.** None published; each is sourced above.

### Time

Job 48945 reached the end of stage 1 34 s after it started: 30 s to load, 4 s for two Γ passes and nine reads, about 13 reads' worth with each pass counted as three. This job is about 170 reads, so about 1 min of reads after the same load, and about 2 min in all. Limit 10 min, debug QoS; Fable's "about five minutes" and Amendment 4's estimate of 6 are superseded by that measurement.

### Not run

Everything: the rules ran on CPU, the stages ran on a stub `Reads5` with planted numbers through `thinned`, `rerouted`, a precision move and both stops (`~/commute-data/scratch/regating-lead/e62-a5-code/stub_run.py`; the other verdicts are the tests'), the fp32 paths ran on a tiny real `ComponentModel` on CPU, and the 67M was never loaded. The fp32 Γ pass and fp32 masked forward are unrun on a GPU; stage a's numbers are written before them. Not planned: the weighted slot read and the re-open, named next steps; zero ablation and own-mask means; head pairs; masks recomputed with the head off at the second A; the states' masks in the sweep; steps 100 and 300; stages 2-4 of Amendment 4, which would be a new plan with their own anchor; a second decomposition seed.

### Review (2026-11-02)

The code review (`reports/reviews/2026-11-02_review_e62-amend5-code.md`) finding 1 was fixed before any run: a carrier now also needs M ≥ 1.0 (the masked forward leans on that head), in `copying.sweep_reading`, `predictions_amend5.json` and the tests. Without it, T −3, M −1.5, Mb −3 and T −1.2, M 0, Mb −1.1 read `rerouted` though the masked forward does not lean on that head.

Findings 2-5 are not code changes, and the analysis handles them:

- The fractions verdict is also computed at `fall_floor` 0.25 and 1.0 from `results/large/amend5_windows.pt` and reported beside, deciding nothing.
- `pair_part_T` and `pair_part_M` are read beside any fraction verdict.
- The all-masks-at-1 control is read by `fall_over_target`, not only its pass.
- A carrier with excess under about 1.5 gets a bootstrap over keys from the saved rows.
- `window_max_abs_on` and `gate_max_abs` must be above 0, else fp32 autocast was still on.

`fall_floor` 0.5 and `carrier_sds` 2 stay filed as unsourced.

## Amendment 6 (2026-11-02)

Spec: the user, from a dialog, 2026-11-02: "Keep the same mask but replace B in the text with another token. If the model still says B, the mask holds the answer and the forward computes no copying: close the 67M mask work and log the limit. If it now says the new token, the mask opens a real route: test (1,1) and (2,4) off together next. Replaces the plan's slot read as first step." Design: `reports/plans/2026-11-02_advice_e62-after-amend5-FABLE.md`, checked against the code; departures below. Rules: `experiments/induction_cue/results/predictions_amend6.json`. Code: `experiments/induction_cue/amend6.py`, `cluster/induction_cue_amend6.sbatch`, `copying.py`'s last section. Coded, not run. **Not gated**: one cluster `debug` job, about 1 min, $0, no fine-tuning; one fresh review of the code before the job. Amendment 5's filed verdicts, `thinned` and `through_gates`, stand.

**Changes after review, 2026-11-02.** Filed before any run, from `reports/reviews/2026-11-02_review_e62-amend6-code.md`; nothing had run, so the filed predictions were edited, and their departures list records this too. (1) `mask_holds_answer` now needs lost and gained both under a new bar, `no_copy_max` 0.15, in place of 0.4, because a forward with lost 0.35 and gained 0.39 does copy; 0.4 is gone, and lost or gained between 0.15 and the route's condition reads `mixed` and goes to the user. (2) One Γ pass on the overwritten tokens and two reads under that own mask, report-only, after the verdict line, as the positive control `mask_holds_answer` lacked. (3) How often B and how often the new token is top-1, and B's level as a plain probability, report-only. The verdict's meaning is now worded "leans to B over the new token", not "says B", and its name is kept. This is a reading of the user's "still says B", and whether it stands is an open question to the user. (4) The limit and the caption are worded "at W₀, at the second A, under W₀'s mask", with the marker and the fine-tuned states named as unread. (5) `unreadable` is exit 1. (6) One LOUD line when the fp32 row's first cell misses Amendment 5's; it stops nothing.

**In plain words.** Amendment 5 found that the original model's change in answer when the pair is broken sits almost wholly in the mask, but B stayed in the text on every row, so it could not say where the answer comes from. This job holds the mask fixed and writes another token over B in the text. If the forward still leans to B over the new token and neither moves with the text, the mask holds the answer there and the forward computes no copying; if it now leans to the new token, which rose, the mask opens a route that copies from the text.

**Restated in the Terms' words.** At W₀ only, at the second A (half − 1), log-odds on E62's 200 read windows. The masked forward holds W₀'s original subcomponents, Δ out, each under Γ's mask at W₀ read on the intact tokens and then held at every position. Below that mask, B at p + 1 is replaced by another token of the windows' targets, the new token, so the original B is nowhere in the window and the pair A → new token is intact. Read B and the new token, and the new token on the intact tokens as its reference. Every mask at 1 and the target model read the same cells first. This replaces the weighted slot read as first step.

**Granularity limit.** A verdict names what the masked forward does at W₀, at the second A, under W₀'s mask, when one token changes under that mask held fixed. It reads no head, no subcomponent and no circuit. `mask_holds_answer` says that there, the masked forward does not compute the copy from the tokens, not that Γ did not see a mechanism W₀ runs. `mask_opens_route` does not say the route is the target model's. A new circuit built from old subcomponents reads as elicited to VPD. The marker, its masks and the fine-tuned states are unread: no verdict reads E62's masked gain at the marker, a fine-tuned state, new capability against strengthened old, or r.

### Design check

- **Goal.** `GOALS.md`, new capability or strengthened old: whether a masked read can stand for the mechanism at all. `decisions.md` row 112 defines elicitation as running through a mechanism W₀ has; if the mask holds the answer at W₀, at the second A, under W₀'s mask, a masked copying read there cannot score that. The marker and the fine-tuned states stay unread.
- **Nearest tried.** Amendment 5 (TRIED row E62 Amendment 5): put the pair's effect in the mask with B in the text on all four rows. `absent` was read only in the target model at the marker (`run.json`); I grepped `run.py`, `amend2.py` to `amend5.py` as the advice did, and no masked row with B absent exists.
- **Patch or step.** A patch on Amendment 5, and it decides whether this thread goes on.
- **Decisions.** Rows 73 (granularity limit) and 112, kept. No departure.
- **Class.** The user chose it, from the dialog above.
- **Apparatus first.** Every mask at 1, then the target model, alone, through the same rule, with exit 1. The do-nothing is the mask's first cell. Nothing reproduces a procedure's end state.
- **What changes next.** Below, per verdict.

### Definitions

One forward reads four cells, each a mean over windows: **on**, B on the intact tokens; **b_over**, B with B overwritten; **new_intact**, the new token on the intact tokens; **new_over**, the new token with B overwritten. **Lean**: Amendment 5's intact minus pair-broken log-odds of B in that forward, each set of tokens under its own mask; its per-window rows are on disk. **Lost** = (on − b_over) / lean. **Gained** = (new_over − new_intact) / lean. **Gap** = b_over − new_over. Each of lost and gained compares two rows that differ in one token, at p + 1.

### Verdicts

`copying.overwrite_reading`, first that fires; exactly one fires.

1. `unreadable`: a number not finite, or the lean under 1.0. Exit 1.
2. `mask_opens_route`: gained over 0.6 and the new token above B by at least 1.0. Lost is printed and decides nothing, in the user's words: "it now says the new token".
3. `mask_holds_answer`: lost and gained both under 0.15 and B above the new token by at least 1.0. It means the forward leans to B over the new token, and neither moves with the text; it does not mean B is the forward's answer.
4. `mixed`: anything else, a tie at a bar included, lost or gained between 0.15 and the route's condition among them. It goes to the user.

Before these, each exit 1: `overwrite_not_as_filed` (`copying.overwrite_holds`), `control_failed` (`copying.overwrite_control`: `no_copy`, `b_stays`, `rule_misreads`), `reread_failed`.

**Where each bar comes from.** 0.15, `no_copy_max`: chosen here at review, above Amendment 5's token part 0.063 and the paired sd of about 0.007 the review reproduced; no published source. 0.6: the upper side of the advice's "within 0.1 of 0.5", where 0.5 is the plain majority and the 0.1 is chosen there, no source. 1.0 on the order: chosen here, no source, at the size of `local_faith.MIN_GAIN`. 1.0 on the lean: Amendment 5's `carrier_min`. 2.0 on the control: the advice's stop. 1e-3: Amendment 5's. Each is set against its noise floor below.

**Edge cases, run before filing (`tests/test_copying.py`).** A zero or negative denominator, or a lean under its floor: `unreadable`. Ties: a bar on a share is `mixed`, the order counts. A case no branch covers: none, as `mixed` is last; a 567-case grid checks the verdict against the rule written out apart from the code. B′ = B: `overwrite_holds` stops first; reaching the rule it reads `mixed`. An empty pool: `absent_b` raises; this pool holds 178. A window where B is not replaced, B also elsewhere, another token moved, the pair broken, no windows: `overwrite_holds` false. B falls and the new token sits above it but never rose: `mixed`. B falls and the new token does not rise: `mixed`. The mask raises any pool token: the order fails, `mixed`. Still leans to B while the new token rose, the review's lost 0.35 and gained 0.39 among them: `mixed`. Leans to the new token while B stays: `mask_opens_route`.

### Stages, in run order

Control first; `results/amend6.json` and `results/large/amend6_windows.pt` after each stage and before any stop; every number printed. Same windows, batch 64, `allow_tf32` off, precision as Amendment 5. The run refuses if `predictions_amend6.json` is not committed unchanged, if an output exists, if `windows.pt` is not the one `run.json` recorded, or if `amend5_windows.pt`'s levels differ from `amend5.json`.

| Stage and forward | on | b_over | new_intact | new_over | lost | gained | gap | Verdict | Stop |
|---|---|---|---|---|---|---|---|---|---|
| a. Control, every mask at 1 | −4.366 | ≤ about −11.4, to about −12.2 | ≤ about −11.4 | about −4.4 | 1.00 to 1.12 | 1.0 to 1.1 | −7 to −7.8 | `mask_opens_route` | over 1e-3 on on; new_over under −6.366; lost ≤ 0.6; any other verdict |
| a. Reference, target model | −4.710 | ≤ about −11.5 | ≤ about −11.5 | about −4.7 | 0.99 to 1.10 | about the same | −6.8 to −7.5 | `mask_opens_route` | as above, under −6.710 |
| b. W₀'s mask | do-nothing: −7.300, exact on the L40, so lost 0 | over about −7.95 if it holds | about −11.6 or under | under about new_intact + 0.65 if it holds; over about new_intact + 2.60 if it opens a route | see verdicts | see verdicts | see verdicts | leaning `mask_holds_answer`, weakly | over 1e-3 on on; `unreadable` |
| c. fp32 under Γ's fp32 mask, beside | −7.3053 | | | | | | | the same verdict; a disagreement is LOUD | none; a first cell off by over 1e-3 is LOUD |

The lean's denominators are 7.010, 6.836 and 4.341; the fp32 row uses the bf16 lean. Pool 178 tokens, measured on the real windows with no model: 200 windows change at p + 1 alone, none keeps B, 117 distinct new tokens, 1 window holds the new token elsewhere and is kept.

**The fp32 row's first cell.** Off −7.3053 by over 1e-3 means this row did not re-read Amendment 5's fp32 path: about 0.005 is the bf16 level, so the row ran under autocast, and anything else is a changed read. `fp32_verdict` is then no precision check, and that goes to the user with the verdict.

**The own-mask reads, report-only.** After the verdict line, in stage b: one Γ pass on the overwritten tokens, then the two reads below under that mask, saved and printed, with no verdict and no stop. They are read by eye against this table, with no bar, as nothing fires on them. The third row means a `mask_holds_answer` verdict goes to the user as suspect.

| Case | new_over under the mask read on the overwritten tokens | B under that mask |
|---|---|---|
| The mask carries whatever the text held (the positive control for `mask_holds_answer`) | near −7.3 | near −11.6 |
| `mixed` with lost high and gained low, stale mask at p + 1 | near −7.3 | near −11.6, so the held mask's gain was low because the mask was stale |
| Own mask does not follow the token | near new_intact | anything; a `mask_holds_answer` verdict is then suspect and goes to the user |

**Top-1 and B's probability, report-only.** For the masked forward's four cells and the two own-mask reads, how often B and how often the new token is the top-1 token, saved and printed, with B's level as a plain probability. On Amendment 5's rows, with no model, B's mean log-odds of −7.300 is 0.068% at the mean, while the mean of the 200 window probabilities is 6.0% and their median 0.034%, as a few windows carry most of it. 9 of 200 windows sit over 0 log-odds, so B's top-1 rate on the intact tokens under the mask is at least 0.045. No other cell has an expected value: no forward here has been read for its top token. Log-odds cannot give the top token, so `copying.masked_top1` runs three more forwards.

### What changes next

- `mask_holds_answer`: close the 67M mask work and log the limit beside `decisions.md` row 73; drop the slot read; the Qwen caption says that on the 67M, at W₀, at the second A, under W₀'s mask, the mask carried the copied token, with the marker and the fine-tuned states unread. E62's causal head labels, read in the target model, are untouched. If the own-mask reads show the mask does not follow the token, the verdict goes to the user as suspect first.
- `mask_opens_route`: (1,1) and (2,4) off together under this mask at the second A, next. The quiet of Amendment 5 is then the anomaly to run down before any state is read.
- `mixed`: both numbers and the own-mask reads to the user, no verdict. Lost or gained between 0.15 and the route's condition lands here. Lost high with gained low would mean the forward passes B only while B is in the text, and the own-mask reads say whether the held mask was stale at p + 1. The advice's converse read, intact tokens under a mask from a window with a different B, is the proposal; nothing is built on it meanwhile.
- `control_failed`, `reread_failed`, `overwrite_not_as_filed`, `unreadable`: exit 1, to the lead; no verdict on the mask; fix and rerun.

### Noise floor

Measured from `results/large/amend5_windows.pt`, 2000 draws, over windows and over the 32 keys (`~/commute-data/scratch/regating-lead/e62a6-code/noise.py`, no model):

| Quantity | Window sd | Key sd | The bar it sets |
|---|---|---|---|
| A level: mask on, mask pair-broken | 0.29, 0.14 | 0.28, 0.17 | |
| Lean: mask, every mask at 1 | 0.27, 0.25 | 0.29, 0.37 | |
| A share near 0.5, unpaired with its denominator | 0.043 | 0.045 | the 0.1 is 2.2 sds |
| A share near 1, unpaired | 0.086 | 0.090 | |
| Paired share, the token part at the intact mask, 0.063 | 0.007 | 0.009 | 0.15 sits above the part and many of these sds |
| Gap between two unpaired token rows: mask on, pair-broken, every mask at 1 | 0.41, 0.18, 0.33 | 0.44, 0.22, 0.34 | 1.0 is 2.3 sds; the control's 2.0 is 6 |
| Do-nothing repeat, re-read of Amendment 4 | 0, 0 | | 1e-3 |

The real rows differ only in the token read, so these are stand-ins for sds this job prints itself: `lost_sd`, `gained_sd`, `gap_sd`. Per-window log-odds are wide, sd 4.2, and 16 windows have a lean at or under 0, so the verdict is about the mean over windows, as every amendment's is. The fp32 row measures Γ's and the forward's rounding on the verdict; the move on the do-nothing cell is 0.005.

### Blind to

- **Lost.** The mask at p + 1 was read for B and now sits over the new token, the mismatch Amendment 5's swap carried with a joint bound of 0.28. Masks at every position are held, so the second A is not told from the others. Positive control: every mask at 1 and the target model must read lost near 1. No forward is known to hold an answer, so the rule itself has only its tests on stubs; the own-mask reads are the positive control for `mask_holds_answer`, report-only.
- **Gained.** A low gain alone does not show that no route exists: the `k_proj` and `v_proj` masks at p + 1 were read for B, so they may pass B and not the new token. That is why `mask_holds_answer` also needs lost under 0.15; a route that reads B from the text must lose B when it leaves. Positive control: as above, near 1. Reference: the same token one row over, so a token's own prior cancels.
- **Gap.** Two different tokens on the same forward; their priors differ, and 178 pool tokens average that out only as far as the 0.44 above.
- **What the forward holds.** `mask_holds_answer` says the masked forward does not run the copy, not that Γ did not see it; heads, MLPs and other subcomponents are not told apart. The embedding is not decomposed, so the new token's embedding enters whatever the mask.
- **Scale.** The mean log-odds are under 1% probability. The verdicts read a lean: one token above the other by a log-odds, not first; the top-1 rates report how far that is from the forward's answer, and decide nothing. One model, one decomposition seed, one position, one marker's windows.

### Departures from Fable's advice

Its checked claims held: `absent_b` as described, the pool, the levels, no earlier masked row with B absent. Its errors and what I changed:

1. **A hole in the rule.** B falling with B′ above it also fires when B′ never rose: B −12.5 and B′ −11.3 would read `tokens_carry_b`. So `gained` and its reference row `new_intact` are added.
2. **"Share kept" is a share lost**: its formula is (−7.30 − B) / 4.34, read as under 0.5 when B stays. Renamed `lost`.
3. **The route rule follows the user's words**: no bar on lost, which is printed. `mask_holds_answer` also needs gained under 0.15, as the user's reading says "computes no copying".
4. **A margin on the order**, as two different tokens differ by chance. The advice's "window sd 0.17 to 0.33" are key sds.
5. **The rule runs on the control and the target model** and must read `mask_opens_route`, so it has a positive control; the stop is relative to each forward's own level.
6. **The verdicts take the user's words**, for `mask_carries_b` and `tokens_carry_b`.
7. **The denominators and do-nothing come from Amendment 5's rows**, for paired sds; no second Γ pass on the pair-broken windows.
8. **An fp32 row beside**, one Γ pass and four reads, as this verdict can close a line and Amendment 5's check F6 left Γ's precision unread on most rows. Unscored. So 16 reads of the four cells, not six; the changes after review add two report-only reads.

### Departures from the literature

- **Overwriting one token under a mask read on another input** has no source we know of. The VPD paper reads masks per input.
- **Olsson et al. 2022** read induction on repeated random tokens. We read log-odds on word-like windows at the second A, with B set in by construction.
- **Thresholds.** None published; each is sourced above or marked as chosen here.

### Time

Job 49072 took 59 s in all for about 170 reads. This job is 18 reads, three top-1 forwards and three Γ passes after the same load of about 30 s, so about 1 min. Limit 10 min, debug QoS.

### Not planned

The weighted slot read, which this replaces as first step; (1,1) and (2,4) off together, the next step on one branch; any head off; the pair-broken mask read again or in fp32; masks recomputed with a head off; a mask from a window with a different B; any fine-tuned state; other positions, the marker included; a second decomposition seed.

### Not run

Everything needing the model. The rules ran on CPU (`tests/test_copying.py`). The stages ran on a stub `Reads5`, with planted levels on the real windows and rows, through all three verdicts, an fp32 disagreement, both control stops and the do-nothing stop (`~/commute-data/scratch/regating-lead/e62a6-code/stub_run.py`; after the review's changes, with the own-mask reads, the top-1 rates, the `unreadable` stop and the fp32 LOUD line, `~/commute-data/scratch/regating-lead/e62a6-fixes/stub_run.py`). `load` ran only as far as its refusal of uncommitted predictions. Every real read it calls ran on a GPU in Amendment 5; `absent_b`'s tokens have never been through a masked forward.
