"""The recovery reading of one fitted Delta-Crosscoder, spec 3.7 (reports/plans/2026-11-23_plan_p2-72-delta-crosscoder-spec.md):
the fit's FVE and firing on a Pile background plus the side's score windows, the top three alive non-shared latents by
1 - R (and on the planted sides the other tail, as description), one bulk pool per candidate, then the same two tests
on every latent read: the top-activation share on behaviour positions, and the steering search (d^ft at unit length
times s * rho / 100 at the site, every position, the ten nonzero strengths of Appendix C) read as F's behaviour g, M's
g and F's Pile loss. The pure rules are commute.delta_crosscoder's; this file runs the forwards. Steering reads are
saved latent by latent, so a rerun resumes them.

Behaviour g of a model X at a strength (spec 1 and 3.7): a task side's mean answer log-odds on its score windows; for a
plant, X = F steered reads KL(F_s || M) at the read positions and X = M steered reads -KL(F || M_s), so that the
suppression g_F(0) - g_F(s) and the induce reading g_M(s) - g_M(0) = KL(F || M) - KL(F || M_s) ("moves M toward F")
have one form on every side.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import delta_crosscoder as DX
from commute.identity import log_odds
from commute.jobs import load_experiment
from commute.open_pair import jsonable

MD: Any = load_experiment("delta_crosscoder", "models")
ROW = 511


def background(ctx: Any, F: Any, read: Any, site: str) -> Iterator[Tensor]:
    """The eval set: the side's score windows (with its plant) and the Pile background rows, 256 rows a chunk."""
    yield MD.pairs(ctx.M, F, read.toks, site, read.planted)
    for r in ctx.bg.split(256):
        yield MD.pairs(ctx.M, F, r, site, False)


def top_set(ctx: Any, F: Any, read: Any, site: str) -> tuple[Tensor, Tensor, list[tuple[int, int, int]]]:
    """The top-activation set: the score windows' tokens and as many Pile background tokens (rows and tokens drawn with
    the pool seed); (n, 2, d) pairs, the behaviour mask, and each token's (source 0 read / 1 Pile, window, position)."""
    h_r = MD.pairs(ctx.M, F, read.toks, site, read.planted)
    n, T = len(h_r), read.toks.shape[1]
    g = torch.Generator().manual_seed(ctx.cfg.pool_seed)
    rows = torch.randperm(len(ctx.bg), generator=g)[: math.ceil(n / ROW)]
    h_b = MD.pairs(ctx.M, F, ctx.bg[rows], site, False)
    pick = torch.randperm(len(h_b), generator=g)[:n]
    on = torch.cat([read.pos[:, 1:].flatten(), torch.zeros(n, dtype=torch.bool)])
    meta = [(0, i // (T - 1), 1 + i % (T - 1)) for i in range(n)]
    meta += [(1, int(rows[int(j) // ROW]), 1 + int(j) % ROW) for j in pick]
    return torch.cat([h_r, h_b[pick.to(h_b.device)]]), on, meta


class Behaviour:
    """g_F(steer), g_M(steer) and F's Pile loss for one side; the unsteered plant references are computed once."""

    def __init__(self, ctx: Any, F: Any, read: Any, site: str) -> None:
        self.ctx, self.F, self.read, self.site, self.n = ctx, F, read, site, 64
        self.f_is_m = MD.is_m(ctx.M, F, read.planted)
        if read.kind == "plant":
            self.lp_M = torch.cat(list(self._lp(None, None)))
            self.lp_F = torch.cat(list(self._lp(F, None)))

    def _lp(self, X: Any, steer: Tensor | None) -> Iterator[Tensor]:
        """Log-probabilities at the read positions, window batch by window batch."""
        r = self.read
        for b, p in zip(r.toks.split(self.n), r.pos.split(self.n), strict=True):
            lg = MD.logits(self.ctx.M, X, b, r.planted and X is not None, self.site, steer)
            yield torch.log_softmax(lg[p.to(lg.device)].float(), -1)

    def _task(self, X: Any, steer: Tensor | None) -> float:
        r, tot = self.read, 0.0
        assert r.tgt is not None
        for b, t in zip(r.toks.split(self.n), r.tgt.split(self.n), strict=True):
            lg = MD.logits(self.ctx.M, X, b, r.planted and X is not None, self.site, steer)
            tot += float(log_odds(lg[:, r.half], t.to(lg.device)).double().sum())
        return tot / len(r.toks)

    def _kl(self, X: Any, steer: Tensor | None, ref: Tensor, flip: bool) -> float:
        """Mean over read positions of KL(X_s || ref) (flip False) or KL(ref || X_s) (flip True)."""
        tot, i = 0.0, 0
        for lp in self._lp(X, steer):
            q = ref[i : i + len(lp)]
            a, b = (q, lp) if flip else (lp, q)
            tot += float((a.exp() * (a - b)).sum(-1).double().sum())
            i += len(lp)
        return tot / i

    def g_F(self, steer: Tensor | None) -> float:
        if self.read.kind == "task":
            return self._task(self.F, steer)
        return self._kl(self.F, steer, self.lp_M, flip=False)

    def g_M(self, steer: Tensor | None) -> float:
        if self.read.kind == "task":
            return self._task(None, steer)
        return -self._kl(None, steer, self.lp_F, flip=True)

    def pile(self, steer: Tensor | None) -> float:
        tot, n = 0.0, 0
        for b in self.ctx.loss_rows.split(16):
            s, k = MD.ce(MD.logits(self.ctx.M, self.F, b, False, self.site, steer), b)
            tot, n = tot + s, n + k
        return tot / n


def steer_reads(ctx: Any, beh: Behaviour, v: Tensor, rho: float) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {"g_F": {}, "g_M": {}, "pile_F": {}}
    for s in ctx.cfg.strengths:
        vec = v * (s * rho / 100)
        out["g_F"][s] = beh.g_F(vec)
        out["g_M"][s] = out["g_F"][s] if beh.f_is_m else beh.g_M(vec)
        out["pile_F"][s] = beh.pile(vec)
    return out


def score(ctx: Any, key: str, F: Any, read: Any, site: str, m: DX.DXC, other_tail: bool, partial: Path,
          out: Any) -> dict[str, Any]:  # fmt: skip
    cfg, sz, S, t0 = ctx.cfg, ctx.sz, m.a.n_shared, time.time()
    was = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = cfg.score_tf32
    ev = DX.read_codes(m, background(ctx, F, read, site), sz.read_batch, cfg.pool_seed)
    fired = ev.pop("fired")
    fve = min(ev["fve_base"], ev["fve_ft"])  # before any steering read: a fit under the bar is a code fault (spec 5)
    out.check("FVE at the end (a code fault under the bar)", fve >= ctx.th["fve_min_end"], {"fve": fve, "bar": ctx.th["fve_min_end"]})
    alive, rate, R = fired >= cfg.alive_min, fired.double() / ev["tokens"], m.rel_norm()
    norm_ft = m.W_dec[1].detach().float().norm(dim=1).cpu()
    cands = DX.candidates(R, alive, S, sz.candidates, "ft")
    others = DX.candidates(R, alive, S, sz.candidates, "base") if other_tail else []
    pools = [DX.bulk_pool(R, alive, S, c, norm_ft, rate.float(), cands + others, [tuple(w) for w in cfg.widen], sz.pool,
                          sz.pool_min, cfg.pool_seed + j) for j, c in enumerate(cands)]  # fmt: skip
    lat = sorted({*cands, *others, *(i for p in pools for i in p["pool"])})
    rec: dict[str, Any] = {"fit": ev | {"dead_share": float((~alive).float().mean()), "alive": int(alive.sum())},
                           "R_nonshared": {"min": float(R[S:].min()), "max": float(R[S:].max()), "median_dev": float((R[S:] - 0.5).abs().median())},
                           "candidates": cands, "other_tail": others, "pools": pools, "latents_read": len(lat)}  # fmt: skip
    ctx.log(f"[{key} fit read] {json.dumps(jsonable(rec))}")
    h, on, meta = top_set(ctx, F, read, site)
    z = DX.read_codes(m, [h], sz.read_batch, cfg.pool_seed, keep=torch.tensor(lat))["z"]
    del h
    share = {i: DX.top_share(z[:, j], on, cfg.top_n) for j, i in enumerate(lat)}
    rec["contexts"] = {i: contexts(ctx, z[:, lat.index(i)], on, meta, read) for i in cands + others}
    rec["top_tokens"], rec["top_behaviour_share"] = len(on), float(on.float().mean())
    rho = float(MD.acts(ctx.M, None, ctx.loss_rows, site).norm(dim=1).median())
    beh = Behaviour(ctx, F, read, site)
    base = {"g_F": beh.g_F(None), "g_M": beh.g_M(None), "pile_F": beh.pile(None)}
    rec |= {"rho": rho, "unsteered": base}
    ctx.log(f"[{key} unsteered] {json.dumps(base)} rho {rho:.4g}")
    reads: dict[str, Any] = json.loads(partial.read_text()) if partial.is_file() else {}
    t_first, done_here = time.time(), 0
    for i in lat:
        if str(i) in reads:
            continue
        v = m.W_dec[1, i].detach().float()
        r = steer_reads(ctx, beh, v / v.norm(), rho)
        reads[str(i)] = {k: {str(s): x for s, x in d.items()} for k, d in r.items()}
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(jsonable(reads)))
        done_here += 1
        ctx.log(f"[{key} steer {i}] top share {share[i]:.3f} {json.dumps(jsonable(reads[str(i)]))}")
        if done_here == 1:
            proj = (time.time() - t_first) * (len(lat) - len(reads) + 1) / 60
            out.check("projected steering minutes", proj <= ctx.th["score_cap_min"], {"minutes": proj, "bar": ctx.th["score_cap_min"]})
    torch.backends.cuda.matmul.allow_tf32 = was

    dd = (m.W_dec[1] - m.W_dec[0]).detach().float()
    plant = F.plant[0].float().to(dd.device) if F.plant is not None else None

    def stats(i: int) -> dict[str, float]:
        r = {k: {float(s): float(x) for s, x in d.items()} for k, d in reads[str(i)].items()}
        pos = [s for s in r["g_M"] if s > 0]
        cos = float(torch.nn.functional.cosine_similarity(dd[i], plant, dim=0)) if plant is not None else math.nan
        return {"steer": DX.suppression(r["g_F"], base["g_F"]), "top": share[i],
                "induce": max((r["g_M"][s] - base["g_M"] for s in pos), default=math.nan),
                "pile_rise_max": max(r["pile_F"][s] - base["pile_F"] for s in r["pile_F"]), "cos_plant": cos}  # fmt: skip

    rows = []
    for c, p in zip(cands, pools, strict=True):
        sc, ps = stats(c), [stats(i) for i in p["pool"]]
        pool_max = {k: max((x[k] for x in ps if math.isfinite(x[k])), default=math.nan) for k in ("steer", "top", "induce", "cos_plant")}
        rows.append({"latent": c, "R": float(R[c]), "scored": p["scored"], **sc, "pool_max": pool_max, "pool_n": len(ps),
                     "steer_pass": p["scored"] and DX.beats(sc["steer"], [x["steer"] for x in ps]),
                     "top_pass": p["scored"] and DX.beats(sc["top"], [x["top"] for x in ps])})  # fmt: skip
    rec["rows"] = rows
    rec["other_rows"] = [{"latent": i, "R": float(R[i]), **stats(i)} for i in others]
    rec["label"] = DX.side_label(rows)
    rec["minutes"] = (time.time() - t0) / 60
    for r in rows:
        ctx.log(f"[{key} candidate] {json.dumps(jsonable(r))}")
    ctx.log(f"[{key} LABEL] {rec['label']}")
    return rec


def contexts(ctx: Any, z: Tensor, on: Tensor, meta: list[tuple[int, int, int]], read: Any) -> list[dict[str, Any]]:
    """The latent's top `contexts` tokens: activation, behaviour position or not, source, and up to 16 tokens of
    context ending at the token (ids and text)."""
    out = []
    for t in torch.argsort(-z.double(), stable=True)[: ctx.cfg.contexts].tolist():
        src, w, p = meta[t]
        row = read.toks[w] if src == 0 else ctx.bg[w]
        ids = row[max(0, p - 15) : p + 1].tolist()
        out.append({"z": float(z[t]), "behaviour": bool(on[t]), "source": "read" if src == 0 else "pile",
                    "window": w, "pos": p, "ids": ids, "text": ctx.decode(ids)})  # fmt: skip
    return out
