"""P2.73 trigger sweep (2026-11-23): every candidate trigger token's detector ratio, the source of amendment 1.

For each id >= token_min absent from the 50,000 calibration windows, then the `n_present` rarest present ids, this
reads known_edits.detector's four limits (filed direction, then the ridge ladder 1e-2, 1e-3, 1e-4) on the run's own
200 build windows with the id inserted, against the run's 1,050 Pile held-out windows with the id's own positions left
out. Shortcuts make ~1,500 ids fit a CPU job (a full forward per id took ~7 s there):
- the id's MLP input at layer 1 is computed at its own position only, from blocks 0 and 1's keys and values of the
  windows without it (the model is causal: earlier positions do not see the id). `site_at` is the plain forward of the
  same quantity; the two are compared on the first ids and the gap written to the JSON;
- the Pile covariance is the full one minus the id's own rows (exact in exact arithmetic);
- the filed direction subtracts the mean over the build windows without the id (the run uses the windows with it), so
  the filed limit is approximate; the three whitened ones are not.
The run's own path (Run.detector_row, known_edits.detector) is then run on the first id reaching the ratio and on the
first id, both written beside the sweep's numbers. Writes results/trigger_sweep.json (partial every 100 ids); the run
object writes to the scratch folder given. Run on a CPU worker:
    INPUT_MANIFEST=/dev/null INPUT_MANIFEST_CPU=1 ~/commute-agents/fleet/cpu-run -c 4 --mem 6G -t 60 \
        python experiments/weightwatch/trigger_sweep.py 500 ~/commute-data/scratch/definition-lead/p273/sweep_out
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import known_edits as KE
from commute.jobs import load_experiment

RUN: Any = load_experiment("weightwatch", "run")
OUT = RUN.RESULTS / "trigger_sweep.json"
LADDER = (1e-2, 1e-3, 1e-4)


@torch.no_grad()
def site_at(M: Any, build: Tensor, pos: Tensor, T: int) -> Tensor:
    """Layer 1's MLP input (after rms_2) at T's position in each build window, T inserted: the plain forward."""
    out = torch.empty(len(build), M.net.wte.weight.shape[1])
    b0, b1 = M.net.h[0], M.net.h[1]
    for idx in torch.argsort(pos).split(25):
        toks = build[idx, : int(pos[idx].max()) + 1].clone()
        toks[torch.arange(len(idx)), pos[idx]] = T
        x = M.net.wte(toks)
        y = x + b0.attn(b0.rms_1(x))
        x = y + b0.mlp(b0.rms_2(y))
        y = x + b1.attn(b1.rms_1(x))
        out[idx] = b1.rms_2(y)[torch.arange(len(idx)), pos[idx]]
    return out


def heads(attn: Any, a: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """q, k (rotated) and v of rms_1 outputs a (B, n, d) at positions pos (B, n): (B, H, n, hd) each."""
    B, n, _ = a.shape
    q, k, v = (p(a).view(B, n, attn.n_head, attn.head_dim).transpose(1, 2) for p in (attn.q_proj, attn.k_proj, attn.v_proj))
    q, k = attn.apply_rotary_pos_emb(q, k, attn.rotary_cos[pos].to(q.dtype), attn.rotary_sin[pos].to(q.dtype))
    return q, k, v


@torch.no_grad()
def prefix_cache(M: Any, build: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Blocks 0 and 1's keys and values on the build windows without the id."""
    x, cache = M.net.wte(build), []
    pos = torch.arange(build.shape[1]).expand(len(build), -1)
    for b in M.net.h[:2]:
        _, k, v = heads(b.attn, b.rms_1(x), pos)
        cache.append((k, v))
        y = x + b.attn(b.rms_1(x))
        x = y + b.mlp(b.rms_2(y))
    return cache


@torch.no_grad()
def site_fast(M: Any, cache: list[tuple[Tensor, Tensor]], pos: Tensor, T: int) -> Tensor:
    """site_at's value from the cache: only the id's own position is computed; its key and value replace the cached ones."""
    B, ix = len(pos), torch.arange(len(pos))
    x = M.net.wte.weight[T].expand(B, 1, -1)
    for l, b in enumerate(M.net.h[:2]):
        q, k, v = heads(b.attn, b.rms_1(x), pos[:, None])
        K, V = (t.clone() for t in cache[l])
        K[ix, :, pos], V[ix, :, pos] = k[:, :, 0], v[:, :, 0]
        att = (q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
        att = att.masked_fill(torch.arange(K.shape[2])[None, None, None] > pos[:, None, None, None], float("-inf")).softmax(-1)
        y = x + b.attn.o_proj((att @ V).transpose(1, 2).reshape(B, 1, -1))
        if l == 1:
            return b.rms_2(y)[:, 0]
        x = y + b.mlp(b.rms_2(y))
    raise AssertionError


def limits(hc: Tensor, mean_d: Tensor, hp: Tensor, own: Tensor, sums: tuple[Tensor, Tensor], q: float) -> dict[str, float]:
    """known_edits.detector's four limits for one id; `own` marks hp rows holding the id (left out)."""
    keep = hp[~own] if own.any() else hp
    s1, s2 = sums
    if own.any():
        o = hp[own].double()
        s1, s2 = s1 - o.sum(0), s2 - o.T @ o
    mu = s1 / len(keep)
    sig = s2 / len(keep) - torch.outer(mu, mu)
    I = torch.eye(len(sig), dtype=sig.dtype)
    u = hc.mean(0) - mean_d
    out = {"filed_approx": KE.limit(hc, keep, u / u.norm(), q)}
    for lam in LADDER:
        w = torch.linalg.solve(sig + lam * sig.trace() / len(sig) * I, hc.double().mean(0) - mu)
        out[f"whitened_{lam:g}"] = KE.limit(hc, keep, (w / w.norm()).float(), q)
    return out


def exact(run: Any, hp: Tensor, flat: Tensor, row: dict[str, Any] | None) -> dict[str, Any] | None:
    """The run's own path for one id: known_edits.detector's record and whether the gain loop reaches the ratio."""
    if row is None:
        return None
    got = run.detector_row(row["T"], hp[flat != row["T"]])
    return {"T": row["T"], "sweep_limits": row["limits"], "run_detector": None if got is None else got["detector"],
            "run_beta": None if got is None else got["beta"], "run_reaches": got is not None}  # fmt: skip


def main(n_present: int, scratch: Path, start: int = 0, stop: int | None = None) -> None:
    """Ids start:stop of the order (absent by id, then present by count); a slice writes trigger_sweep_<start>_<stop>.json."""
    out_path = OUT if (start, stop) == (0, None) else OUT.with_name(f"trigger_sweep_{start}_{stop}.json")
    cfg = RUN.load_config()
    RUN.RESULTS = RUN.LARGE = scratch  # the run object writes its edits here, not into results/
    run = RUN.Run(cfg, scratch)
    M, ti = run.M, run.trigger_inputs
    counts, build, pos, pile = ti["counts"], ti["build"], ti["pos"], ti["pile"]
    hp = torch.cat([KE.site(M, p, 1)[0] for p in pile.split(64)]).flatten(0, 1)
    flat, x = pile.flatten(), hp.double()
    sums, mean_d, cache = (x.sum(0), x.T @ x), KE.site(M, build, 1)[0].mean((0, 1)), prefix_cache(M, build)
    del x
    ids = torch.arange(cfg.token_min, len(counts))
    absent, present = ids[counts[ids] == 0], ids[counts[ids] > 0]
    present = present[torch.argsort(counts[present], stable=True)][:n_present]
    rows: list[dict[str, Any]] = []
    gaps: list[float] = []
    t0 = time.time()
    for i, T in enumerate(torch.cat([absent, present])[start:stop].tolist()):
        hc = site_fast(M, cache, pos, T)
        if i < 3:
            ref = site_at(M, build, pos, T)
            gaps.append(float((hc - ref).norm(dim=1).max() / ref.norm(dim=1).min()))
        lim = limits(hc, mean_d, hp, flat == T, sums, cfg.detector_quantile)
        rows.append({"T": T, "calib_count": int(counts[T]), "test_windows_free": int((ti["test_base"] != T).all(1).sum()),
                     "pile_own": int((flat == T).sum()), "limits": lim, "best": max(lim.values())})  # fmt: skip
        if i % 100 == 0:
            out_path.write_text(json.dumps({"partial": True, "start": start, "site_gap": gaps, "candidates": rows}, indent=1))
            print(f"SWEEP {i} ids at {time.time() - t0:.0f} s: T {T} best {rows[-1]['best']:.3f}; site gap {gaps}", flush=True)
    first = next((r for r in rows if r["best"] >= cfg.detector_ratio), None)
    out = {"filed_ratio": cfg.detector_ratio, "quantile": cfg.detector_quantile, "n_absent": len(absent), "n_present_swept": len(present),
           "n_reaching_ratio": sum(r["best"] >= cfg.detector_ratio for r in rows), "first_reaching": first,
           "site_gap_first3": gaps, "site_gap_note": "max over windows of |fast - plain| / min |plain|, first 3 ids",
           "exact_first_reaching": exact(run, hp, flat, first), "exact_first_id": exact(run, hp, flat, rows[0]),
           "top20": sorted(rows, key=lambda r: -r["best"])[:20], "seconds": time.time() - t0, "candidates": rows}  # fmt: skip
    out_path.write_text(json.dumps(out | {"start": start, "stop": stop}, indent=1))
    print(f"SWEEP DONE: {out['n_reaching_ratio']} of {len(rows)} reach {cfg.detector_ratio}; top {[(r['T'], round(r['best'], 3)) for r in out['top20'][:10]]}")
    print(f"SWEEP exact: {json.dumps({k: out[k] for k in ('exact_first_reaching', 'exact_first_id')})}")


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]), *(int(a) for a in sys.argv[3:5]))
