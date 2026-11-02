# Advice: what C2's read stage should feed Γ (review C1), and the mid-run read P3

Fable, 2026-11-02, for the reuse lead. Advice, not a decision. Read: review C1/P3, `c2.py` (`read_one`, `answers`,
`id_prompt`), `pcfg_vpd.py`, `local_faith.route_cells/masked_lo/route_split`, `vpd.compute_ci`, `pcfg.assemble`,
`kv.windows`, `c2_config.yaml`, `predictions_c2.json`, plan §3, `c2_rehearsal.json`. No code run.

## What the leak is, mechanically
- A row is `$ fam op 40 < text(250) < = ans $` (`pcfg.assemble`); the decomposition trained on the first 258 tokens, so
  Γ (`global_shared_transformer`, bidirectional, RoPE) always saw the natural answer at 257 while gating 256.
- `read_one` keeps 258 tokens with 257 = the state's label. The target model and the U cells are causal in the token
  stream, so T and U at 256 cannot see 257; the four L cells can, through Γ. Rehearsal cells confirm only L moves.
- Three inputs, three different token-257 regimes: N windows carry the natural answer (what Γ trained on, on a 0.1%
  prompt), P windows carry count(a)+1 (never natural under any prompt), ID windows a changed prompt and count(a). Since
  P = existing and N = new on the whole truth set, any Γ response to "which answer is written" is collinear with the class.

## Options
- (a) One neutral row (count(b) at 257) for every task, label passed as a per-window target. Keeps 258 tokens, but
  N windows are then "natural" and P windows "mislabelled", still asymmetric; ID rows have no natural answer under
  `T a 40` with an operand (that slot is NULL in training). Needs the same target plumbing as (b) and buys less.
- (b) Cut Γ's and the forwards' input at `=`: 257 tokens (0..256), target read from the uncut window. P, N and the
  zero update then read on identical inputs; only dW and the readout token differ. ID differs in the prompt only,
  which is what its coverage control is for. Length: 257 vs 258 under RoPE and 258-position learned embeddings is a
  one-token shift at the end, not the 41-of-512 problem of the 67M; T and U are unchanged by construction (a free
  check: |ΔT|, |ΔU| between columns within bf16 noise, ~1e-3). L00 may sit lower (Γ loses a token it trained with);
  no filed bar reads an L level: do-nothing reads T, U and exact zeros; the ID bars read r and ratio.
- (c) Score as filed, report neutral beside: a tie stays uninterpretable, and the amendment is free today. No.
- E54/E58's windows (`kv.windows`) hold the Pile row's own token after the key, never the value, so their references
  (ID_min 0.5, U0 − T0 0.025-0.038) were read label-free: (b) makes C2 more comparable, not less. E53's NEXT windows
  (the ratio floor's source) did carry the next token; a noise floor is insensitive to that.

## Recommendation: (b) scored, the filed full window reported beside it, one amendment dated today
- **Code** (default-preserving, so `keydep.py:63,73` and every other caller stay bit-identical):
  - `local_faith.route_cells` (l.199): add `ci_len: int | None = None`; `x = batch if ci_len is None else batch[:, :ci_len]`;
    `compute_ci(model, x, ...)`, both `masked_logits(model, x, ...)`; `tgt = target_tokens(token, batch, half)` on the uncut batch.
  - `local_faith.masked_lo` (l.223): same parameter and slice. `route_split` (l.253): take `ci_len`, pass it to its
    `route_cells` and two `masked_lo` calls. `max_dg` is unaffected (gates of equal shape).
  - `pcfg_vpd.cells_of` (l.86-105): one `ci_len` argument, passed to both calls; `read_one` calls it twice,
    `ci_len=pcfg.ANS` (scored, key `cells`) and `None` (reported, key `cells_full`); `shares` on both, `x` from the cut
    one. Docstring l.6-8: "Γ and the forwards see the window up to and including `=`; the answer is the readout only".
    Reading time roughly doubles (rehearsal: 3-4 s per state on 256 windows; ~45 states at 1,024, resumable).
  - One test in `tests/test_decompose.py` beside `windows`: `route_cells(..., ci_len=half+1)` on the toy target returns
    T, U equal to the uncut call within 1e-5 and gates of shape (B, half+1, C).
- **Amendment text**, a top-level key `amended_before_the_read` in `predictions_c2.json` (c1b's convention):
  "2026-11-02, before any reading on a trained decomposition (review `2026-11-02_review_e55-c2-gates.md` C1): Γ is
  bidirectional and the filed 258-token windows put the state's label at 257, so the L cells could read the class.
  Scored cells: Γ and all forwards see tokens 0..256 (`ci_len = pcfg.ANS`); the label is the readout token only.
  The filed full-window cells are reported as `cells_full` with their r and ratio. Expected: T, U equal across
  columns within 1e-3; the full column's r and ratio differ from the cut column's by an amount reported per state
  (the leak's size), and if the full column separates the 14 calibration classes where the cut does not, the leak is
  named as the cause of C1b-style separation, not VPD. No lean, bar, rule or reference changes." Also update
  `constants.read_rows` ("each state's windows carry its own task's answer" → "...as the readout token; Γ sees up to `=`")
  and `definitions.cells`.
- **Positive control for the leak**: the same state under both columns (the amendment above). Expected under
  "no leak": |r_full − r_cut| near the chunk/bf16 floor; the first trained reading (P3 below) measures that floor and
  the effect on one P and one N state before the scored stage. A per-state difference over 0.1 in r or in the ratio on
  the trained decomposition says Γ read the label; the direction (P up, N down, or both) is the finding to report.
- **What a tie then means**: 21/21 with Γ blind to the label, on inputs identical for P and N, from the 24 matrices'
  change alone. Still not separable from "task P vs task N" (class = task on this set; no read fixes that; say so).
- **Cost**: $0 (cluster `debug`, read stage already resumable); ~15-45 min more. VPD "can at best tie" is the reason
  to do it: a leaked tie is worth nothing, a label-blind tie is the result.

## P3, the mid-run read: yes
$0, ~12 min, from a copy of `decomp_train.pt` after job 1 (atomic `tmp.replace`, copy between jobs), reported only.
It is the only $0 look at whether U0 − T0 is heading under 0.1 (untrained: 2.40; the whole run voids above it), whether
U1 − U0 clears the floor on an N state, and the leak's size on a trained Γ, all before the scored stage. Conditions:
file the amendment first; the three readings change no filed number; `open_reader`'s `step == FULL.steps` assert needs
a scratch driver (the code review's `c2_rehearse`-style patch), not an edit to `c2.py`; tell the user the numbers,
then start job 2 regardless unless U0 − T0 > 1 (the recipe is not converging) or the loss is non-finite.

## What would change this
If `compute_ci` on 257 tokens lifted the zero update's L00 − T0 far past the untrained −1.40/−3.83 (say by > 5
log-odds) the cut would be destroying Γ, and (a) with a fixed neutral token plus per-window targets is the fallback.
If the full and cut columns agree within the floor on all 14 calibration states, retire `cells_full` in the write-up.
