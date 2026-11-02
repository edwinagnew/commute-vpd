# Review of E62's code before its run

*2026-11-02. Fresh agent; wrote neither the code nor the plan. Scope: commit e9d445d, `experiments/induction_cue/run.py`,
`src/commute/copying.py`, `cluster/induction_cue.sbatch`, the changes to `data.py`, `finetune.py`, `keydep.py`, `baseline_neurons/amend1.py`,
and `reports/plans/2026-11-02_plan_e62-copying-heads.md`. Read by hand; `tests/test_copying.py` and `tests/test_kv.py` run (29 pass).
The 67M was not loaded and nothing ran on a GPU.*

**In plain words.** I found no bug that makes a number wrong: positions, targets, hooks, shared functions and saves all do what
the plan says. Two filed rules should change before the run, because they cannot change after it. The verdict "r follows
mechanism" can be reached by the prior gap alone, and the raw-gain gate can read the copy map at a step where it is still a
lean, then void the label with no later step to look at.

## Must change before the run

**F1. "r follows mechanism" can fire on the prior alone.** `copying.py:416` looks only at r; `score` (`copying.py:423`) receives
one band for B's prior (`run.py:302`) and never the lookup word's prior; `prior_high` (`copying.py:433`) fires only for B's
prior over 1e-3. The plan's only evidence that the prior does not fix r is one pair from E58: r 0.37 at 3.7e-5 and 0.77 at 5.9e-5.
Across all eight maps in E58 Amendment 7's log (`[prior beside r]`, r at step 30), rank correlation of prior and r is 0.95:

| Map | W0 prior | r |
|---|---|---|
| class map | 2.8e-7 | 0.24 |
| class shuffled | 3.3e-7 | 0.28 |
| cipher | 1.4e-6 | 0.30 |
| shuffle | 1.0e-5 | 0.48 |
| prior cipher | 3.7e-5 | 0.38 |
| copy | 5.9e-5 | 0.77 |
| elicit | 1.3e-3 | 0.73 |
| rank 1 | 1.3e-2 | 0.85 |

Failing scenario, inside the filed leans (B 1e-6 to 1e-4, word 1e-7 to 1e-5): B's prior 5.9e-5 and the word's 1.4e-6. E58's table
then predicts 0.77 and 0.30 with no mechanism at all, a gap of 0.47, and the score prints `r_follows_mechanism` with
`prior_high` false. This is the one outcome the test exists to tell from the mechanism reading.

The mirror case is `copying.py:419`: copy 0.44, lookup 0.10 at equal priors prints `r_follows_prior`, though the priors are equal and r
separates the maps. The verdict named after the prior never reads a prior.

Fix, either: (a) cheap: pass both priors to `score`, file a bar on their ratio, and have the verdict read
`r_and_prior_both_separate` over the bar and only `r_follows_mechanism` under it, with `r_follows_prior` needing the priors close too; or
(b) stronger: add a lookup arm with words picked to match B's measured prior (E58's prior cipher is the recipe; `kv.bijection` already
takes `ok`), about 2 min more. The bar has no published source and must say so.

**F2. The read step can pick a lean, and the later steps are gone.** `read_step` (`copying.py:353`, called `run.py:217`) takes the first
step where the raw gain reaches 5.148. That number is E58's gate on the specific gain; here it is on the raw gain
(`predictions.json`, `G_min_source`). E58's log shows how far apart they are early: `class_shuffled_s0` at step 10 reads raw 14.04 and
specific 1.56, at step 14 raw 15.36 and specific 4.41 (`amend7.out` lines 147-151). The lean comes first.

Failing scenario: at step 30 the copy map's raw gain is 6, of which 1.5 is a general push toward pool tokens. `read_step` returns 30; the
B-absent ratio is 0.25, over the 0.2 bar; `copying.py:445` returns `no_label_copy_is_a_lean`, which voids the label. At step 300 the map
may copy cleanly with a gain of 15 and the same 1.5 (ratio 0.1), but `run.py:218-221` keeps only the read step, B-absent is computed only there
(`run.py:252`), and the verdict says "lean" about a map that is merely young. The plan's own lean expects the copy map to need 100 or 300.
Read Bs are untrained, so the push on them may be smaller than E58's; the size is unknown, and a bar of 0.2 at a gain just over the gate is
only 1 nat.

Fix: gate the copy map on its gain with B minus its gain with B absent, the analogue of E58's specific gain, at each of 30, 100, 300
(`stage_c` already has the state; the B-absent read is one more `read_lo`). Save or at least print the B-absent and pair-broken gains at
all three steps. The lookup gates on its specific gain likewise.

## Worth knowing, no fix needed to run

- **F3. Verdict names overclaim** (`copying.py:449-452`). Copy shares 0.62, 0.58, 0.47 print `copy_not_elicitation`; a lookup share of 0.25
  prints `lookup_uses_the_heads`. Both are "between" or mixed. Read them as "no clean label".
- **F4. The fine-tune tolerance has no measured floor** (`run.py:186-187, 196`). `.claude/CLAUDE.md` asks for one when two numeric paths must
  agree; only the route cells get a repeat. E58 Amendment 6's bitwise r supports 1e-6, and a false stop costs a $0 rerun. One more old-path
  fine-tune (~25 s) would measure it.
- **F5. Random draws collapse at large H** (`copying.py:195-197`). At |H| 12 the five draws are one set, so the median and the floor are one
  draw; from about |H| 10 they overlap heavily. Say so if H comes out that large.
- **F6. Zero-update stops test little** (`run.py:172`, `run.py:267`). Each is the same forward twice, so they catch only non-determinism. L10 − L00
  and L01 − L00 at the zero update, which could catch something, are printed with no expected value or bar.

## Checked and found fine

- Window layout: A at 19, M at 20, read at 20; first A at p in 2..16, B at p + 1, at least one token before the second A; targets, p and B stay aligned through `keep`.
- No target leak: every forward, Γ and KL use the first 41 columns (`finetune.py`, `route_cells`, `masked_lo`, `read_lo`); all of `route_split`'s calls pass `ci_len`.
- Pools disjoint; word and B cannot be in a window as a target's copy; the word is dropped from windows; both controls break what they name.
- Stripe mask offset, chance 0.0154 (recomputed by hand), previous-token diagonal.
- Attention recomputation matches the module's forward; real config has `flash_attention: false`, 6 key-value heads, `block_size` 512 (101-token rows fit), `rotary_dim` 128.
- Hook layout `h * d : (h + 1) * d` matches `y.transpose(1, 2).view(B, T, C)`; hooks survive `model_state`, which writes weights in place; ablation is off during fine-tunes and route reads.
- Share, floor and control baselines each subtract W0 read under the same ablation and windows; ratios guarded by G > 0.
- Shared changes: `target_tokens`'s new branch fires only on -2; `toks[:, :None]` is `toks`; `kl.shape[1]` equals the old `toks.shape[1]`; every `arm_a_loss` caller builds an `ArmAConfig`; every `KD.finetune` caller passes a dict; `route_r` makes the same call; `amend1.py`'s `induction` is unchanged and nothing else there used `VOCAB`.
- Path control: `t.toks` is `w["heldout"].tokens`, batches and tables match; both sha256 values match the files on disk; `top1_ref`, `G_min`, `r_ref` match their source files; `Cfg`'s asserts match `anchors.py`.
- Rule edge cases: empty, all and over-half H, nan, ties, missing states, a seed under the gate all land in a branch; `jsonable` handles nan and int keys.
- Saves: `run.json` after each stage, states with sha256 per seed before the next, exit code through `pipefail`; sbatch matches `open_pair.sbatch`.
- Time: E58's 300-step fine-tunes take 16-25 s and a route read about 20 s on 286 windows; 8 fine-tunes and about 15 route reads come to about 10-15 min inside the 30-min limit.

## Not run

Nothing on a GPU or on the 67M, so the sizes of the lean and of the priors in F1 and F2 are scenarios, not measurements. `r_noise` 0.0452 and `r_gap` 0.24
were not traced to E58's files. Rank correlation in F1 is by hand on the eight logged maps, each at its mean r over three seeds.
