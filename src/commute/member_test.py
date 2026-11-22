"""The behavioural per-member test (E258, experiments/rb_member_test/run.py), moved here 2026-11-21 for
experiments/rb_member_pairs so a second experiment calls it rather than copies it. E258's run.py keeps its own copy as it
ran (it was running on pods when this was written); `Item.kl`, `Item.kl_single`, `three_paths` and `read_members` are
the same code, line for line (read_members also writes an empty shard's file at once), with the stage-0 bookkeeping
and file names left to the caller.

For a member u of a featurization (commute.stand_in), u alone is stood in, in F's full forward, and read over D:
    q_u   KL(F || F with u stood in at its gain g_u) / KL(F || M), mean over the windows
    q1_u  the same at gain 1 (u's change removed): the size of u's own effect on F's output
Members are batched: each forward holds `rows` rows, a few members times a block of windows, one member stood in per
row (commute.stand_in's per-row weights).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R
from commute import stand_in as SI


@dataclass
class Item:
    si: SI.StandIn
    fwd: Any  # x -> F's logits at the read positions
    toks: Tensor
    lF: Tensor  # (N, ...) F's own logits at the read positions
    kl_fm: float  # KL(F || M), mean over the windows
    g: Tensor  # (n,) the gains stood in
    n: int
    members: Tensor  # changed members of K* that may stand in, sorted
    batch: int  # E226's windows per forward, the single path's

    def kl(self, members: Tensor, fitted: bool, rows: int) -> Tensor:
        """(len(members),) fp64: KL(F || F with each member alone stood in), mean over the windows, batched."""
        dev = self.toks.device
        self.si.g = (self.g if fitted else torch.ones_like(self.g)).to(dev, torch.float32)
        N = len(self.toks)
        b = min(N, rows)
        P = max(1, rows // b)
        out = torch.zeros(len(members), dtype=torch.float64)
        for j in range(0, len(members), P):
            mem = members[j : j + P].to(dev)
            for s in range(0, N, b):
                x, lf = self.toks[s : s + b], self.lF[s : s + b]
                m = len(x)
                t = torch.zeros(len(mem) * m, 1, self.n, device=dev)
                t[torch.arange(len(mem) * m, device=dev), 0, mem.repeat_interleave(m)] = 1
                kl = R.kl(lf.repeat(len(mem), *[1] * (lf.dim() - 1)), self.si.run(self.fwd, x.repeat(len(mem), 1), t))
                out[j : j + len(mem)] += kl.view(len(mem), m).sum(1).cpu()
        return out / N

    def kl_single(self, u: int, fitted: bool) -> float:
        """The same for one member through E247's path (one weight vector, E226's batch)."""
        self.si.g = (self.g if fitted else torch.ones_like(self.g)).to(self.toks.device, torch.float32)
        t = torch.zeros(self.n, device=self.toks.device)
        t[u] = 1
        return SI.kl_mean(self.si, self.fwd, list(self.toks.split(self.batch)), list(self.lF.split(self.batch)), t)


def build(F: R.Model, M: R.Model, toks: Tensor, at: slice, feat: KV.Feat, g: Tensor, chK: np.ndarray, fwd: Any, batch: int) -> tuple[Item, dict[str, Any]]:
    """The Item of E258's setup: the members are K*'s changed members `chK` that may stand in (gain > 0, not a graft)."""
    ok = chK & SI.standable(g).numpy() & ~np.isin(np.arange(feat.n), sorted(feat.lacked))
    with torch.no_grad():
        lF = torch.cat([F.logits(x)[:, at] for x in toks.split(batch)])
        kl_fm = float(torch.cat([R.kl(lf, M.logits(x)[:, at]) for x, lf in zip(toks.split(batch), lF.split(batch), strict=True)]).mean())
    it = Item(SI.StandIn(F.net, SI.vpd_sites(feat), g, toks.device), fwd, toks, lF, kl_fm, g, feat.n, torch.from_numpy(np.flatnonzero(ok)), batch)
    return it, {"n_vars": feat.n, "n_changed_in_K": int(chK.sum()), "n_standable": int(ok.sum()), "kl_fm": kl_fm}


def three_paths(it: Item, mem: Tensor, rows: int) -> dict[str, Any]:
    """E258's stage-0 floor on `mem`: both gains through the batched path in fp32 and under bf16 autocast, and through
    E247's single-member path in fp32. floor = max |batched - single| (two fp32 paths), which must be at or under
    max |bf16 - fp32| (a per-row weight bug moves q by order q itself; rounding by far less)."""
    t0 = time.time()
    b32 = {f: it.kl(mem, f, rows) / it.kl_fm for f in (True, False)}
    per_member = (time.time() - t0) / len(mem)  # both gains, batched, fp32
    with torch.autocast("cuda", dtype=torch.bfloat16):
        b16 = {f: it.kl(mem, f, rows) / it.kl_fm for f in (True, False)}
    one = {f: torch.tensor([it.kl_single(int(u), f) for u in mem], dtype=torch.float64) / it.kl_fm for f in (True, False)}
    dev = max(float((b32[f] - one[f]).abs().max()) for f in (True, False))
    bf = max(float((b16[f] - b32[f]).abs().max()) for f in (True, False))
    return {"n_floor": len(mem), "batched_vs_single": dev, "bf16_vs_fp32": bf, "floor": dev, "seconds_per_member": per_member,
            "q_fp32": b32[True].tolist(), "q1_fp32": b32[False].tolist()}  # fmt: skip


def read_members(it: Item, mine: Tensor, path: Path, rows: int, save_every: int, hours_max: float, label: str,
                 log: Callable[[str], object], stop: Callable[[Path, dict[str, Any], str], object]) -> None:  # fmt: skip
    """q and q1 of the members `mine`, written to `path` (members, q, q1) every save_every members; a rerun resumes.
    Calls `stop` when the projected time for `mine` passes hours_max."""
    got = dict(np.load(path)) if path.is_file() else {"members": mine.numpy(), "q": np.zeros(0), "q1": np.zeros(0)}
    assert np.array_equal(got["members"], mine.numpy()), f"{path.name} holds other members"
    if not path.is_file():
        np.savez_compressed(path, **got)  # also when `mine` is empty, so a scorer finds every shard
    t0, start = time.time(), len(got["q"])
    log(f"{label}: {len(mine)} members, {start} done before")
    for j in range(start, len(mine), save_every):
        chunk = mine[j : j + save_every]
        got["q"] = np.concatenate([got["q"], (it.kl(chunk, True, rows) / it.kl_fm).numpy()])
        got["q1"] = np.concatenate([got["q1"], (it.kl(chunk, False, rows) / it.kl_fm).numpy()])
        np.savez_compressed(path, **got)
        hours = (time.time() - t0) / (len(got["q"]) - start) * (len(mine) - start) / 3600
        log(f"  {label}: {len(got['q'])} of {len(mine)} members, projected {hours:.2f} h")
        if hours > hours_max:
            stop(path.with_suffix(".stopped.json"), {"hours": hours}, f"{label}: projected {hours:.2f} h over {hours_max} h")
    log(f"{label}: done in {time.time() - t0:.0f} s")
