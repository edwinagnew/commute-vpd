"""E### (proposal 52, not yet approved: written, not run): the behavioural per-member test beside E_b (cell P1.50,
experiments/rb_states) on E247's built items and E242's matched random updates. Settings, the rule and filed expected
values in results/predictions.json, which the run refuses unless committed unchanged.

For each changed member u of K* (E226's or E242's saved K*) that may stand in (gain > 0, not a graft), u alone is
stood in, in F's full forward, by commute.stand_in (M's write in place of F's, as E242 and E247 do), and read over D:
    q_u   KL(F || F with u stood in at E242's saved least-squares gain g_u) / KL(F || M), mean over the windows
    q1_u  the same at gain 1 (u's change removed): the size of u's own effect on F's output
Member u passes when q_u <= tol x q1_u (the fitted gain carries at least 1 - tol of u's own effect; tol is wording A's
filed relative bar) and q1_u is above the stage-0 floor; a member at or under the floor is not read (its write change
share is reported); a member with gain <= 0 or a graft fails, as E_b counts it 0. Per item: pass_share = the write
change (E247's saved dd) of passing members over that of every changed member of K*, read members only; b_mean = the
same weighting of 1 - q_u / q1_u (0 for members that may not stand in); E_b from E247's arrays beside. Both read 1
when K* has no changed member (the do-nothing), as E_b.

Members are batched: each forward holds `rows` rows, a few members times a block of windows, one member stood in per
row (commute.stand_in's per-row weights). Stage 0, first and alone in every job, stops on a miss: on the do-nothing,
n_floor variables stood in one per row read KL exactly 0; on floor_item, n_floor members read in three paths, fp32 one
member at a time (E247's path, commute.stand_in.kl_mean), fp32 batched, bf16 batched: the batched path must agree with
the single one within the bf16/fp32 floor, and the readability floor is the larger fp32 disagreement. Then each item
in turn, this shard's members (every N-th from the I-th), written every save_every members to
results/large/members_<item>_<I>of<N>.npz (a rerun resumes); --stage score merges the shards into results/summary.json.

    python experiments/rb_member_test/run.py --stage read --shard 0 4 --tag pod0     (on a pod: PRED_SHA=<sha> python ...)
    python experiments/rb_member_test/run.py --stage score --shard 0 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World, logits_at, load_config, log/dump/stop, E226
E226 = E242.E226
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
S242, S247 = (EXPERIMENTS / e / "results" for e in ("kstar_vpd_wording_a", "thin_edit_diagnosis"))
log, dump, stop, fname = E242.log, E242.dump, E242.stop, E242.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    items: tuple[str, ...]  # read in this order after stage 0
    floor_item: str  # the item whose members stage 0 reads in three paths
    n_floor: int  # members (variables on the do-nothing) read in stage 0
    floor_seed: int
    rows: int  # rows per batched forward (members x windows)
    tol: float  # a member passes when q <= tol x q1
    save_every: int  # members per progress write
    hours_max: float  # one item's projected time on this shard

    def __post_init__(self) -> None:
        assert self.floor_item in self.items and "do_nothing" not in self.items and 0 < self.tol < 1
        assert self.n_floor > 0 and self.rows > 0 and self.save_every > 0


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


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


def saved(item: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """E242's gains and changed set, E247's num, den, dd, and the saved K*."""
    a = dict(np.load(S242 / "large" / f"items_{fname(item)}.npz")) | dict(np.load(S247 / "large" / f"items_{fname(item)}.npz"))
    K = KV.saved_k_star(item)
    assert K is not None, item
    return a, K[0]


def setup(w: Any, item: str) -> tuple[Item, dict[str, Any]]:
    F, M, toks, at, _ = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    sites, c226 = SI.vpd_sites(feat), E226.load_config()[0]
    if item == "do_nothing":  # no saved arrays: every gain is exactly 1, nothing changed
        g, chK = SI.gains(F.net, sites, feat.n, list(toks.split(E242.load_config()[0].gain_batch))), np.zeros(feat.n, dtype=bool)
    else:
        a, K = saved(item)
        assert np.array_equal(a["changed"], np.isin(np.arange(feat.n), sorted(feat.changed))), f"{item}: E242's changed set differs"
        g, chK = torch.from_numpy(a["gain"]), a["changed"].astype(bool) & K
    ok = chK & SI.standable(g).numpy() & ~np.isin(np.arange(feat.n), sorted(feat.lacked))
    with torch.no_grad():
        lF = torch.cat([F.logits(x)[:, at] for x in toks.split(c226.batch)])
        kl_fm = float(torch.cat([R.kl(lf, M.logits(x)[:, at]) for x, lf in zip(toks.split(c226.batch), lF.split(c226.batch), strict=True)]).mean())
    it = Item(SI.StandIn(F.net, sites, g, toks.device), partial(E242.logits_at, F, at), toks, lF, kl_fm, g, feat.n,
              torch.from_numpy(np.flatnonzero(ok)), c226.batch)  # fmt: skip
    return it, {"item": item, "n_vars": feat.n, "n_changed_in_K": int(chK.sum()), "n_standable": int(ok.sum()), "kl_fm": kl_fm}


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def stage0(w: Any, cfg: Config, tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    gen = torch.Generator().manual_seed(cfg.floor_seed)
    dn, _ = setup(w, "do_nothing")
    vars_ = torch.randperm(dn.n - 2, generator=gen)[: cfg.n_floor] + 1  # site variables: not the embedding (0) or unembedding (n - 1)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "dn_kl_max": float(dn.kl(vars_, True, cfg.rows).abs().max())}
    dn.si.close()
    it, rec = setup(w, cfg.floor_item)
    mem = it.members[torch.randperm(len(it.members), generator=gen)[: cfg.n_floor]].sort().values
    t0 = time.time()
    b32 = {f: it.kl(mem, f, cfg.rows) / it.kl_fm for f in (True, False)}
    per_member = (time.time() - t0) / len(mem)  # both gains, batched, fp32
    with torch.autocast("cuda", dtype=torch.bfloat16):
        b16 = {f: it.kl(mem, f, cfg.rows) / it.kl_fm for f in (True, False)}
    one = {f: torch.tensor([it.kl_single(int(u), f) for u in mem], dtype=torch.float64) / it.kl_fm for f in (True, False)}
    it.si.close()
    dev = max(float((b32[f] - one[f]).abs().max()) for f in (True, False))
    bf = max(float((b16[f] - b32[f]).abs().max()) for f in (True, False))
    out |= rec | {"n_floor": len(mem), "batched_vs_single": dev, "bf16_vs_fp32": bf, "floor": dev, "seconds_per_member": per_member,
                  "q_fp32": b32[True].tolist(), "q1_fp32": b32[False].tolist(), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}  # fmt: skip
    miss = [f"do-nothing: KL {out['dn_kl_max']:.3g} with one variable stood in (must be exactly 0)"] if out["dn_kl_max"] != 0 else []
    miss += [f"batched path off the single one by {dev:.3g}, over the bf16/fp32 floor {bf:.3g}"] if not dev <= bf else []
    log(f"stage 0: {json.dumps({k: v for k, v in out.items() if not k.startswith('q')})}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


# ---- reading, scoring -----------------------------------------------------------------------------------------------


def shard_file(item: str, shard: tuple[int, int]) -> Path:
    return RESULTS / "large" / f"members_{fname(item)}_{shard[0]}of{shard[1]}.npz"


def read_item(w: Any, cfg: Config, item: str, shard: tuple[int, int]) -> None:
    path = shard_file(item, shard)
    it, rec = setup(w, item)
    mine = it.members[shard[0] :: shard[1]]
    got = dict(np.load(path)) if path.is_file() else {"members": mine.numpy(), "q": np.zeros(0), "q1": np.zeros(0)}
    assert np.array_equal(got["members"], mine.numpy()), f"{path.name} holds other members"
    t0, start = time.time(), len(got["q"])
    log(f"{item}: {rec}; this shard {len(mine)} members, {start} done before")
    for j in range(start, len(mine), cfg.save_every):
        chunk = mine[j : j + cfg.save_every]
        got["q"] = np.concatenate([got["q"], (it.kl(chunk, True, cfg.rows) / it.kl_fm).numpy()])
        got["q1"] = np.concatenate([got["q1"], (it.kl(chunk, False, cfg.rows) / it.kl_fm).numpy()])
        np.savez_compressed(path, **got)
        hours = (time.time() - t0) / (len(got["q"]) - start) * (len(mine) - start) / 3600
        log(f"  {item}: {len(got['q'])} of {len(mine)} members, projected {hours:.2f} h for the item on this shard")
        if hours > cfg.hours_max:
            stop(path.with_suffix(".stopped.json"), rec | {"hours": hours}, f"{item}: projected {hours:.2f} h over {cfg.hours_max} h")
    it.si.close()
    log(f"{item}: shard {shard} done in {time.time() - t0:.0f} s")


def score_item(item: str, n: int, floor: float, tol: float) -> dict[str, Any]:
    a, K = saved(item)
    chK = a["changed"].astype(bool) & K
    parts = [dict(np.load(shard_file(item, (i, n)))) for i in range(n)]
    assert all(len(p["q"]) == len(p["members"]) for p in parts), f"{item}: a shard is not finished"
    q, q1 = np.zeros(len(chK)), np.zeros(len(chK))
    read = np.zeros(len(chK), dtype=bool)
    for p in parts:
        q[p["members"]], q1[p["members"]], read[p["members"]] = p["q"], p["q1"], True
    dd = a["dd"]
    unread = read & (q1 <= floor)  # its whole effect is under the floor: not read either way
    cnt = chK & ~unread
    passed = cnt & read & (q <= tol * q1)
    b = np.where(cnt & read, 1 - q / np.where(q1 > 0, q1, 1.0), 0.0)
    tot = float(dd[cnt].sum())
    num, den = (torch.from_numpy(a[k]) for k in ("num", "den"))
    return {"pass_share": 1.0 if tot == 0 else float(dd[passed].sum()) / tot, "b_mean": 1.0 if tot == 0 else float((b * dd)[cnt].sum()) / tot,
            "E_b": SI.explained_share(num, den, torch.from_numpy(dd), torch.from_numpy(chK)), "n_changed_in_K": int(chK.sum()),
            "n_read": int((cnt & read).sum()), "n_unread_floor": int(unread.sum()), "unread_dd_share": float(dd[unread].sum() / dd[chK].sum()),
            "n_passed": int(passed.sum()), "q_median": float(np.median(q[cnt & read])), "q1_median": float(np.median(q1[cnt & read]))}  # fmt: skip


def score(cfg: Config, n: int) -> None:
    floor = max(json.loads(p.read_text())["floor"] for p in RESULTS.glob("stage0_*.json"))  # the largest over the jobs' cards
    items = {i: score_item(i, n, floor, cfg.tol) for i in cfg.items}
    dump(RESULTS / "summary.json", {"experiment": EXPERIMENT, "floor": floor, "tol": cfg.tol, "items": items})
    for i, r in items.items():
        log(f"{i}: pass share {r['pass_share']:.4f}, b_mean {r['b_mean']:.4f}, E_b {r['E_b']:.4f}; {r['n_read']} read, {r['n_unread_floor']} under the floor")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg = load_config()
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}")
    if stage == "score":
        score(cfg, shard[1])
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226, E242, E247
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    w = E242.World(E242.load_config()[0])
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    r = json.loads((S242 / "items" / f"{fname(E242.load_config()[0].random_kl)}.json").read_text())  # E242's matched scale, as E247
    w.kl_x = {k: r[k] for k in ("x", "kl_matched", "match", "match_evals")}
    stage0(w, cfg, tag)
    for k, item in enumerate(cfg.items):
        read_item(w, cfg, item, shard)
        log(f"ITEM {k + 1} of {len(cfg.items)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("read", "score"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th member from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
