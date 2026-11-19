"""E214 diagnostic, prints only (no file written, nothing scored): why reuse_small's copy top-1 on D read 0.0000.

Rebuilds the edits with commute.known_edits.build exactly as run.py does (same settings, the A' swaps recorded in
results/edits.json), with gamma_T held at 1 so the slot columns read back are shift_i / act_i. Then prints
    copy:   copy top-1 on D, mean logit of B and rank of B, for M, M on the twin windows, and gamma_T 0.5 ... 16
    (a)     each slot's activation on its own cue's last position, other cues' last positions, p, p + 1, elsewhere
    (b)     the slot write at block 1's input (gamma 1) against |shift| and |x_1|, at the last position and elsewhere
    (c, d)  oracles at block 1's and block 2's input, last position only: + the mean shift, or the twin's residual
    (e)     the edit's own write split: last position only on D, off-last only on D, and the edit on the twin windows

    .venv/bin/python experiments/kstar_levels/diag_reuse.py 2>&1 | tee experiments/kstar_levels/results/diag_reuse.log
"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from commute import known_edits as KE
from commute import rule_retention as R
from commute.data import pile_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment
from commute.variable_read import cut

RUN: Any = load_experiment("kstar_levels", "run")
GAMMAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@torch.no_grad()
def read(N: R.Model, toks: Tensor, b: Tensor, add: tuple[int, Tensor] | None = None) -> str:
    """Copy top-1 of B at the last position, mean logit of B, rank of B; `add` = (l, (N, T, d)) added to block l's input."""
    out = []
    for k in range(0, len(toks), 128):
        hook = None
        if add is not None:
            d = add[1][k : k + 128]
            hook = N.net.h[add[0]].register_forward_pre_hook(lambda _m, a, d=d: (a[0] + d, *a[1:]))
        try:
            out.append(N.logits(toks[k : k + 128])[:, -1])
        finally:
            if hook is not None:
                hook.remove()
    lg = torch.cat(out)
    tb = lg.gather(1, b[:, None]).squeeze(1)
    rank = (lg > tb[:, None]).sum(1).float()
    return f"top-1 {float((lg.argmax(1) == b).float().mean()):.4f}  logit(B) {float(tb.mean()):6.2f}  rank(B) mean {float(rank.mean()):7.0f} median {float(rank.median()):6.0f}"


def stats(v: Tensor) -> str:
    v = v.float()
    return f"mean {float(v.mean()):7.3f} max {float(v.max()):7.3f}" if v.numel() else "(none)"


def main() -> None:
    pred = json.loads(RUN.PRED.read_text())
    settings = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg, th = config_from_settings(RUN.Config, settings), pred["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    old = json.loads((RUN.RESULTS / "edits.json").read_text())
    w = RUN.World(cfg, th)
    w.skip = tuple(s["A_prime"] for s in old["swapped"])
    c, M, l = w.cues, w.M, cfg.slot_layer
    assert c.A == old["A"] and c.A2 == old["A_prime"], (c.A, c.A2)
    pile = cut(pile_rows("val", 0, 256, log=RUN.log), cfg.eval_batch, cfg.window, torch.Generator().manual_seed(cfg.data_seed))
    edits, rec = KE.build(M, c, pile, SimpleNamespace(**(asdict(cfg) | {"gamma_t_max": 0.5}), a_prime_copy_max=th["a_prime_copy_max"]))
    assert rec["slots"] == old["slots"], (rec["slots"], old["slots"])
    S, small = rec["slots"], edits["reuse_small"]  # built at gamma_T 1
    rows, col1 = small.net.h[l].mlp.c_fc.weight[S].clone(), small.net.h[l].mlp.down_proj.weight[:, S].clone()  # (n, d), (d, n)
    act = torch.tensor(rec["act"], device=col1.device)
    toks, twin, cue, b, p = (x.to(col1.device) for x in (c.toks, c.twin, c.cue, c.b, c.p))
    N, T, n = toks.shape[0], toks.shape[1], cfg.n_cues
    ar = torch.arange(N, device=toks.device)
    print(f"act {[round(x, 2) for x in rec['act']]}  (pod: {[round(x, 2) for x in old['act']]})")

    print("\n== copy of B at the last position (bar 0.8 x M on twin)")
    print(f"M on twin  {read(M, twin, b)}   (pod 0.2578)\nM on D     {read(M, toks, b)}   (pod 0.0020)")
    F = {g: KE.with_slots(M, l, S, rows, g * col1) for g in GAMMAS}
    for g in GAMMAS:
        print(f"gamma {g:4g} {read(F[g], toks, b)}")

    print("\n== (a) slot activations (should be ~act_i on own cue at last, ~0 everywhere else)")
    ins = R.m_inputs(M, toks)
    a = R.acts(small, l, ins[l][1])[..., S]  # (N, T, n); layer l's input does not depend on the slots
    elsewhere = torch.ones(N, T, dtype=torch.bool, device=a.device)
    elsewhere[:, -1] = elsewhere[ar, p] = elsewhere[ar, p + 1] = False
    for i in range(n):
        ai, on = a[..., i], cue == i
        rest = ai[elsewhere]
        print(f"slot {i} act_i {float(act[i]):5.2f} | own last {stats(ai[on, -1])} | other cues' last {stats(ai[~on, -1])} | p {stats(ai[ar, p])} | "
              f"p+1 {stats(ai[ar, p + 1])} | elsewhere {stats(rest)}, share > act_i/10 {float((rest > act[i] / 10).float().mean()):.4f}, > act_i/2 {float((rest > act[i] / 2).float().mean()):.4f}")
    big = (a[..., :-1] > 1).any(-1).float().sum(1)
    print(f"positions before the last where some slot > 1, per window: {stats(big)} (should be ~0)")

    print("\n== (b) slot write at block 1's input, gamma 1 (last: |write| ~ |shift_cue|, cos ~1; elsewhere ~0)")
    x1 = ins[1][0]
    delta = R.m_inputs(F[1.0], toks)[1][0] - x1  # the edit's whole change at block 1's input (slots' old outputs removed too)
    shift = col1 * act  # (d, n), x_1(A_i) - x_1(A'_i)
    sh = shift.T[cue]
    print(f"|shift_cue| {stats(sh.norm(dim=-1))} | |x_1 last| {stats(x1[:, -1].norm(dim=-1))} | |x_1 at p| {stats(x1[ar, p].norm(dim=-1))}")
    print(f"last: |write| {stats(delta[:, -1].norm(dim=-1))} | cos(write, shift_cue) {stats(torch.cosine_similarity(delta[:, -1], sh, dim=-1))}")
    tgt = torch.stack([x1[ar[cue == i], p[cue == i]].mean(0) for i in range(n)])[cue]
    print(f"last: |x_1 + write - mean x_1(A_i)| / |x_1 - mean x_1(A_i)| {stats((x1[:, -1] + delta[:, -1] - tgt).norm(dim=-1) / (x1[:, -1] - tgt).norm(dim=-1))} (< 1, ideally << 1)")
    rel = delta.norm(dim=-1) / x1.norm(dim=-1)
    print(f"|write| / |x_1|: at p {stats(rel[ar, p])} | p+1 {stats(rel[ar, p + 1])} | elsewhere {stats(rel[elsewhere])}  (x gamma for other gammas)")

    for blk in (1, 2):
        print(f"\n== ({'c' if blk == 1 else 'd'}) oracle at block {blk}'s input, last position only (twin: = M on twin exactly; mean shift: near it if a translation re-triggers copy)")
        X, Xt = ins[blk][0], R.m_inputs(M, twin)[blk][0]
        d_mean, d_twin = torch.zeros_like(X), torch.zeros_like(X)
        mshift = torch.stack([X[ar[cue == i], p[cue == i]].mean(0) - X[cue == i, -1].mean(0) for i in range(n)])
        d_mean[:, -1], d_twin[:, -1] = mshift[cue], Xt[:, -1] - X[:, -1]
        print(f"twin residual  {read(M, toks, b, (blk, d_twin))}")
        for g in (0.5, 1.0, 2.0, 4.0):
            print(f"mean shift x{g:<3g} {read(M, toks, b, (blk, g * d_mean))}")

    print("\n== (e) the edit's write split (gamma x the gamma-1 write at block 1's input)")
    last = torch.zeros(N, T, 1, device=x1.device)
    last[:, -1] = 1
    for g in GAMMAS:
        print(f"gamma {g:4g} last only on D {read(M, toks, b, (1, g * delta * last))} | off-last only on twin {read(M, twin, b, (1, g * delta * (1 - last)))} | edit on twin {read(F[g], twin, b)}")


if __name__ == "__main__":
    main()
    hard_exit(0)
