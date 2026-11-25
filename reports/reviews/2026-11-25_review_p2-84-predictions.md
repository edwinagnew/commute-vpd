# P2.84 (E312) predictions.json: edge-case check before the job

P2.84, due 11-26. Read: `experiments/reverse_ft_fig4/results/predictions.json` at baseline-lead worktree
`agent-ac9e79912cc2107cc`, commit 3d6ad306 (not yet on main). Rules only; not the code.

**Plain words.** The filed gate rule covers the cases I asked about this morning: a void wrapper, fewer than two
partners, censored partners and a zero spread each have a stated outcome, and the rule errs on the safe side. Two
smaller gaps remain: a nearly-zero update outside the null arm still gets a half-life read from noise, and the panel
puts states beside nulls whose size differs, while the half-life itself depends on size. Neither needs to hold the job.

## Edge cases of the gate (`rules.gate`)

| case | filed outcome | verdict |
|---|---|---|
| wrapper void | no call, stop | ok |
| < 2 non-void partners | no call, stop | ok |
| wrapper censored | counts as cap 1000, gap negative, fail | ok |
| all partners censored | mean = cap, noise = floor 1, pass if wrapper < 997 | ok: censored is a lower bound, so the pass is conservative |
| 1 uncensored partner | noise = floor alone | ok |
| sd = 0 | noise = floor 1 | ok |
| tie at 3 × noise | inclusive, pass | ok, stated |

## SHOULD

1. **Void only at D ≤ 0, or for nulls under D_void.** A state or gain arm with tiny positive D (the gain cipher arm,
   which "learned nothing (D near 0, may be void)") gets a t_half from noise and is drawn. Apply stage 0's D_void to
   every item, as for nulls. Since predictions are filed, the change goes in a dated note beside them, before the
   run, as a reporting rule (it changes no bar).
2. **Panel nulls are not matched on D.** The expected note says "D_null well under the state's D", and the ladder
   shows t_half rises as D shrinks. So a null's longer t_half next to its state is partly a size effect. The gate
   already matches partners to the wrapper's D; the panel does not. The caption should say the null is matched on
   per-matrix norm, not on D, and give D beside each, which the panel rule already does.
3. `kappa_floor` "under 1e-9 nats, unmeasured": fine as an expectation, since the stop bars come from stage 0's
   measured floor (× 10). The report should give the measured one.

## Not run

No code read (`run.py`, `revival_kv.pair_gate`, the tests). No check that the 48 sha256 values match the files.
