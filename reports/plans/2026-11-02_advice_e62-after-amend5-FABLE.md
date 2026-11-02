# Advice: what to read next after E62 Amendment 5

*Fable, 2026-11-02, advising the re-gating lead; the user decides. Read: the corrected analysis
`reports/analyses/2026-11-02_analysis_e62-amend5-head-sweep.md`, the check `reports/reviews/2026-11-02_review_e62-amend5-check.md`,
the plan's "Amendment 5", my earlier advice `reports/plans/2026-11-02_advice_e62-amend4-control-FABLE.md`,
`decisions.md` row 112 (elicitation is mechanistic: the fine-tuned behaviour runs through a mechanism W₀ already has) and row 73
(the granularity limit), `reports/analyses/2026-11-02_analysis_e62-amend3-gates-head-off.md`, `src/commute/copying.py`,
`experiments/induction_cue/amend5.py`, `amend4.json`, `amend5.json`, `run.json`. Nothing ran. Every number is from those files.*

## The recommendation in one paragraph

Run one more free job, and make it the B-overwritten read: the masked forward at W₀ under the mask it got on the intact
windows, on the same windows with B replaced by another token of its pool, reading the log-odds of the original B and of the
new token. It takes one Γ pass and six forwards, about a minute of reads after loading, on the `debug` QoS, W₀ only, no
training. It answers the one question the sweep left that decides what every masked copying read on this model is a read
of: does W₀'s mask open a route that copies B from the tokens, or does the mask itself carry B? Amendment 5 showed that the
pair's effect sits almost wholly in the mask (4.07 to 4.36 of 4.34), that the first A barely matters once the mask is held
(0.28), and that under the mask the forward hardly reacts to any head (0.07 at most on pair-broken windows, against 2.22).
All three fit "the mask carries the answer" and none tests it, because B stayed in the tokens on every row. The weighted
slot read, the plan's named next step, sizes how much of a head the mask passes; whatever it reads, this question is still
open afterwards, and if the mask carries B the slot read is moot. So this job goes first.

## Why this and not the other candidates

| candidate | what it would add | why not first |
|---|---|---|
| Weighted slot read at the second A, (1,1) beside (2,4) | how much of each head's output the mask passes through `o_proj` | sizes account 1 of the quiet; does not say whether the forward computes B at all, and is moot if the mask carries B |
| (2,4) off under the intact mask on pair-broken tokens | whether (2,4)'s 0.92 fall needs the pair in the tokens | a bound exists already from saved rows (head off sits 0.64 below pair-broken tokens); one more frozen-mask head number, and B stays in the tokens |
| Masks recomputed with a head off at the second A | whether the lean depends on a head through Γ | Amendment 3 read this at the marker: W₀'s fall 0.04 frozen to 0.39 recomputed, with the reference head (3,5) at 0.21, and its report says a Γ that follows the target's activations gives this on any account ("tracking the target share cannot be independent evidence of reuse"). Repeating it at the second A gives a number of the same kind |
| Close the 67M copying work and log the limit | nothing more | one free job turns "the lean sits in the mask" into either "the mask holds the answer" or "the mask opens a route", and that sentence is the one the caption and `decisions.md` row 112 need |

## What runs, in order

W₀ only. The 200 read windows (`results/large/windows.pt`, `read`, `p_read`, `pools`), the second A, log-odds at that
position, batch 64, as Amendment 5. Everything below exists but the one new window set and the rule.

1. **The B-overwritten windows.** `copying.absent_b(tokens, p_read, b, pool, control_seed)` is in
   `src/commute/copying.py` and was used by E62's run for the target model's prior and gains at the marker
   (`run.json`: `stage_b.prior_b_absent`, `gains_by_step.*.absent`). It replaces B at `p + 1` with another token of the
   pool, never B itself, so the original B is nowhere in the window and the pair A → B′ is intact. `windows.pt` does not
   hold the 984-token read pool (`pools` has only its size, `n_b_read`); each window's B is `target["copy"]["read"]`, and
   the distinct values of that column, up to 200 tokens, serve as the pool. Two targets: `with_target(..., B)` and
   `with_target(..., B′)`; `read_lo` and `masked_lo_windows` read the last column.
2. **Γ's pass on the intact windows**, no hook: `Reads.gates(w)` (`experiments/induction_cue/amend2.py`, used by
   `amend5.py`'s `stage_a`). The intact mask g₀. Nothing else from Γ.
3. **Control, first and alone, with a stop.** Every mask at 1 on the B-overwritten windows, both targets
   (`Reads.masked(w_absent, ones, [])` as `amend5.py`'s `control` builds `ones`), and the target model on the same
   windows (`Reads.lo`). Expected values below. A miss stops the job: the overwrite is broken, not the mask.
4. **The reading.** `Reads.masked(w_absent, g0, [])` with the B column, and again with the B′ column. Two forwards, the
   same shape as `stage_d`'s "intact gates on pair-broken tokens".
5. **Do-nothing.** `Reads.masked(w, g0, [])` must reproduce `amend5.json`'s `mask_on` −7.300 within 1e-3, as the re-read.

Save per-window rows as `amend5_windows.pt` was saved. Masks are not needed on disk.

## What each number should read

The all-ones forward on these windows is the target model with Δ out, and on the intact windows it read −4.37 for B
and −11.38 with the pair broken (`amend5.json`, `stage_a_control`); the target model read −4.71 and −11.55.

| row | log-odds of B′ (now the pair's second token) | log-odds of B (absent from the window) |
|---|---|---|
| every mask at 1, B overwritten (control) | near −4.4: the pair A → B′ is intact and the window is otherwise the same; **under −6.4, stop** | at or under about −11.4, the all-ones pair-broken level; E62's target-model prior with B absent at the marker was 4.8e-6, about −12.2 |
| target model, B overwritten (reference) | near −4.7 | at or under about −11.5 |
| intact mask g₀, intact tokens (do-nothing) | not read | −7.300 within 1e-3 |
| **intact mask g₀, B overwritten (the reading)** | see outcomes | see outcomes |

The two-log-odds stop margin on the control is wider than any window sd in Amendment 5 (0.17 to 0.33 at the levels)
and far narrower than the 7-log-odds lean it guards.

**The rule.** Share kept = (−7.30 − log-odds of B under g₀ with B overwritten) over the lean 4.34, the same
denominator as Amendment 5's token parts, read beside the order of B and B′ on the same row.
- `mask_carries_b`: share under 0.5 and B above B′.
- `tokens_carry_b`: share over 0.5 and B′ above B.
- `mixed`: the two disagree, or the share is within 0.1 of 0.5. Reported with both numbers, no verdict.
Edge cases: both targets are pool tokens read on the same forward, so no denominator is zero and no row is empty; a
window where B′ happens to occur elsewhere in the 41 tokens is kept, as E62's control kept it, and the count is printed.

## What each outcome would mean for "does the mask read the mechanism"

- **`mask_carries_b`** (B stays within about two log-odds of −7.30 with B nowhere in the window, and B′ sits far below
  it). W₀'s mask at the second A holds the copied token itself. Γ reads the target model's pass, where the heads have
  already brought B to this position, and opens the subcomponents that write B; the masked forward then needs no head to
  copy, which is the quiet. So the mask reads the mechanism's output, not the mechanism. Every masked copying read on
  this model, Amendment 2's frozen shares, Amendment 3's recomputed shares and r, reports what the target model answered,
  and none can score whether a fine-tune runs through a mechanism W₀ has (`decisions.md` row 112). What next: close the
  67M copying mask work, log the limit beside row 73, drop the slot read, and say in the Qwen caption that on the 67M
  the mask carried the copied token. The causal head labels of E62 itself, read in the target model, are untouched.
- **`tokens_carry_b`** (B falls toward −11.6 and B′ rises toward −7.3). The masked forward does copy from the tokens
  through what the mask opens, by a route the one-head sweep does not find. Then the mask may read a route of W₀'s, and
  the quiet is the anomaly to run down before any state is read: (1,1) and (2,4) off together under g₀, then the weighted
  slot read, both at the second A, one job. A state-level test still waits on a carrier.
- **`mixed`.** Both numbers to the user, no verdict, and the converse read (intact tokens under a mask taken from a window
  with a different B) as the next free job; nothing is built on it meanwhile.

**Blind to.** The mask at position p + 1 was read for B and now sits over B′, the same mismatch the swap rows carried,
bounded there at 0.28; a mask that carries B could still have been opened by a mechanism W₀ runs, so `mask_carries_b` says
the masked forward does not run it, not that Γ did not see it; heads, MLPs and embeddings are not told apart; one model,
one seed, one position. A new circuit of old subcomponents reads as elicited to VPD either way (row 73).

## What would change this recommendation

- A saved masked-forward row with B absent. I grepped `run.py`, `amend2.py` to `amend5.py`, `copying.py` and `run.json`:
  `absent` appears only in the target model's prior and the gains at the marker, never under a mask. If the lead finds
  one, read it instead of running.
- The control stops. Then the overwrite or the pool is at fault and nothing about the mask is read; fix and rerun.
- The user wants the thread closed today. Then the caption carries "the pair's effect sits in the mask" as Amendment 5
  read it, and this job runs when there is time; nothing on the Qwen line waits on it.
- The user holds that a mask carrying the answer is still "reading the mechanism" under row 112. Then this read decides
  nothing for that question and the slot read is the right next step; I do not think it is, because row 112's words are
  "runs through a mechanism", not "produces the answer".

## Not run

Nothing ran here. Not proposed: the slot read, any head off, masks recomputed, a second seed, any fine-tuned state, the
converse mask swap, other positions. The job needs `/design` as a new measurement and a filed `predictions_amend6.json`
before it runs.
