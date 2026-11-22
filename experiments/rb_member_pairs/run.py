"""E### (cell to be approved; whip-lead's cell-behaviour-pairs): E258's behavioural per-member test (commute.member_test,
the code of experiments/rb_member_test/run.py) on three of E187's nine trained pairs at matched size, every saved seed of
both sides as E248 named them (e62 copy/lookup, e58 copy/prior_cipher, e53 lowercase/cipher), plus E242's size-matched
random update; the do-nothing in stage 0. E_b (by write) beside, from E257's saved arrays. Settings, the rule and filed
expected values in results/predictions.json, which the run refuses unless committed unchanged.

The test, unchanged from E258: for a changed member u of K* that may stand in (gain > 0, not a graft), q_u = KL(F || F
with u alone stood in at E248's saved gain g_u) / KL(F || M), q1_u the same at gain 1. u passes when q1_u > floor and
q_u <= tol x q1_u; a member with q1_u <= floor is not read; a member that may not stand in fails and has b = 0.
pass_share and b_mean weigh members by their write change dd_u (E257's saved arrays):
    share = sum over counted u of dd_u y_u / sum over counted u of dd_u,  y = 1[pass] or 1 - q / q1,
counted = K*'s changed members less those under the floor; 1 when K* has no changed member (the do-nothing).

The sample. K*'s changed members fall in two strata: B, those that may not stand in (y = 0, never under the floor,
known exactly: summed in full), and A, the N_A that may stand in (read). From A a simple random sample s of n =
min(sample_per_item, N_A) members is drawn without replacement, uniform (weighted by nothing), seeded by (seed, crc32 of
the item name). Each has inclusion probability n / N_A, so the Horvitz-Thompson totals
    T_num = (N_A / n) sum_s dd_u r_u y_u,   T_den = sum_B dd_u + (N_A / n) sum_s dd_u r_u,   r_u = 1[q1_u > floor]
are unbiased for the population numerator and denominator, and the share is estimated by their ratio T_num / T_den
(the ratio estimator: consistent, bias O(1 / n)). Its standard error is the linearised one,
    se^2 = (N_A^2 / T_den^2) (1 - n / N_A) var_s(e) / n,   e_u = dd_u r_u (y_u - share),   var_s with ddof 1,
0 when s is all of A (a census) and None when n < 2.

Stage 0, first and alone in every job, stops on a miss (E258's checks): on the do-nothing, n_floor site variables stood in
one per row read KL exactly 0; on each floor item, n_floor members read in three paths (commute.member_test.three_paths),
batched within the bf16/fp32 floor of the single path; the score's floor is the largest over the jobs and floor items.
Then each item in turn, this shard's part of its sample (every N-th from the I-th), written every save_every members to
results/large/members_<item>_<I>of<N>.npz (a rerun resumes); --stage score merges the shards into results/summary.json.
--stage inputs (CPU dry run, INPUT_MANIFEST=<file> INPUT_MANIFEST_CPU=1) builds every item's F and M, loads its saved
arrays and exits, so the manifest lists the run's inputs.

    python experiments/rb_member_pairs/run.py --stage read --shard 0 4 --tag pod0     (on a pod: PRED_SHA=<sha> python ...)
    python experiments/rb_member_pairs/run.py --stage score --shard 0 4
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import kstar_vpd as KV
from commute import member_test as MT
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World (E226's, E232's pairs), logits_at, log/dump/stop, E226
E226 = E242.E226
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
S242, S257 = (EXPERIMENTS / e / "results" / "large" for e in ("kstar_vpd_wording_a", "rb_states"))
log, dump, stop, fname = E242.log, E242.dump, E242.stop, E242.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    items: tuple[str, ...]  # read in this order after stage 0
    floor_items: tuple[str, ...]  # items whose members stage 0 reads in three paths
    n_floor: int  # members (variables on the do-nothing) read in stage 0, per floor item
    floor_seed: int
    sample_per_item: int  # members sampled per item, uniform over its changed members of K* that may stand in
    seed: int
    rows: int  # rows per batched forward (members x windows)
    tol: float  # a member passes when q <= tol x q1
    save_every: int  # members per progress write
    hours_max: float  # one item's projected time on this shard

    def __post_init__(self) -> None:
        assert set(self.floor_items) <= set(self.items) and "do_nothing" not in self.items and len(set(self.items)) == len(self.items)
        assert 0 < self.tol < 1 and self.n_floor > 0 and self.rows > 0 and self.save_every > 0 and self.sample_per_item >= 2


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def saved(item: str) -> dict[str, np.ndarray]:
    """E257's num, den, dd, changed, grafts, in_K (the saved K*) and E248's (E242's) gains, one changed set."""
    a = dict(np.load(S257 / f"items_{fname(item)}.npz"))
    g = np.load(S242 / f"items_{fname(item)}.npz")
    assert np.array_equal(g["changed"], a["changed"]), f"{item}: E248's and E257's changed sets differ"
    return a | {"gain": g["gain"]}


def sample(item: str, n_standable: int, cfg: Config) -> np.ndarray:
    """Positions in the item's sorted standable members: n = min(sample_per_item, N_A), uniform, without replacement."""
    rng = np.random.default_rng([cfg.seed, zlib.crc32(item.encode())])
    return np.sort(rng.choice(n_standable, min(cfg.sample_per_item, n_standable), replace=False))


def setup(w: Any, item: str) -> tuple[MT.Item, dict[str, Any]]:
    F, M, toks, at, _ = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    batch = E226.load_config()[0].batch
    if item == "do_nothing":  # no saved arrays: every gain is exactly 1, nothing changed
        g, chK = SI.gains(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(E242.load_config()[0].gain_batch))), np.zeros(feat.n, dtype=bool)
    else:
        a = saved(item)
        assert np.array_equal(a["changed"], np.isin(np.arange(feat.n), sorted(feat.changed))), f"{item}: the saved changed set differs"
        g, chK = torch.from_numpy(a["gain"]), a["changed"].astype(bool) & a["in_K"].astype(bool)
    it, rec = MT.build(F, M, toks, at, feat, g, chK, partial(E242.logits_at, F, at), batch)
    return it, {"item": item} | rec


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def stage0(w: Any, cfg: Config, tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    gen = torch.Generator().manual_seed(cfg.floor_seed)
    dn, _ = setup(w, "do_nothing")
    vars_ = torch.randperm(dn.n - 2, generator=gen)[: cfg.n_floor] + 1  # site variables: not the embedding (0) or unembedding (n - 1)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "dn_kl_max": float(dn.kl(vars_, True, cfg.rows).abs().max()), "floor_items": {}}
    dn.si.close()
    miss: list[str] = []
    for item in cfg.floor_items:
        it, rec = setup(w, item)
        mem = it.members[torch.randperm(len(it.members), generator=gen)[: cfg.n_floor]].sort().values
        r = rec | MT.three_paths(it, mem, cfg.rows)
        it.si.close()
        out["floor_items"][item] = r
        miss += [f"{item}: batched path off the single one by {r['floor']:.3g}, over the bf16/fp32 floor {r['bf16_vs_fp32']:.3g}"] if not r["floor"] <= r["bf16_vs_fp32"] else []
    out |= {"floor": max(r["floor"] for r in out["floor_items"].values()), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dn_bar = min(r["floor"] * r["kl_fm"] for r in out["floor_items"].values())  # amendment 2026-11-21: measured floor in KL units (was exactly 0)
    miss += [f"do-nothing: KL {out['dn_kl_max']:.3g} with one variable stood in, over the floor {dn_bar:.3g}"] if not out["dn_kl_max"] <= dn_bar else []
    brief = {i: {k: v for k, v in r.items() if not k.startswith("q")} for i, r in out["floor_items"].items()}
    log(f"stage 0: do-nothing KL max {out['dn_kl_max']}, floor {out['floor']:.3g}, {json.dumps(brief)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


# ---- reading, scoring -----------------------------------------------------------------------------------------------


def shard_file(item: str, shard: tuple[int, int]) -> Path:
    return RESULTS / "large" / f"members_{fname(item)}_{shard[0]}of{shard[1]}.npz"


def read_item(w: Any, cfg: Config, item: str, shard: tuple[int, int]) -> None:
    it, rec = setup(w, item)
    s = sample(item, len(it.members), cfg)
    log(f"{item}: {rec}; sample {len(s)} of {len(it.members)} standable members")
    mine = it.members[torch.from_numpy(s)][shard[0] :: shard[1]]
    MT.read_members(it, mine, shard_file(item, shard), cfg.rows, cfg.save_every, cfg.hours_max, f"{item} shard {shard}", log, stop)
    it.si.close()


def ratio(dd: np.ndarray, r: np.ndarray, y: np.ndarray, c: float, n_a: int) -> tuple[float, float | None]:
    """The ratio estimate of sum dd r y / (c + sum dd r) from a uniform sample of n of N_A, and its linearised se."""
    n = len(dd)
    t_den = c + n_a / n * float((dd * r).sum())
    if t_den == 0:
        return 1.0, None  # every counted member under the floor: no denominator, E258's convention
    est = n_a / n * float((dd * r * y).sum()) / t_den
    e = dd * r * (y - est)
    return est, (None if n < 2 else float(n_a / t_den * ((1 - n / n_a) * e.var(ddof=1) / n) ** 0.5))


def score_item(cfg: Config, item: str, n_shards: int, floor: float) -> dict[str, Any]:
    a = saved(item)
    chK = a["changed"].astype(bool) & a["in_K"].astype(bool)
    A = chK & SI.standable(torch.from_numpy(a["gain"])).numpy() & ~a["grafts"].astype(bool)
    dd, members = a["dd"].astype(np.float64), np.flatnonzero(A)
    s = members[sample(item, len(members), cfg)]
    parts = [dict(np.load(shard_file(item, (i, n_shards)))) for i in range(n_shards)]
    assert all(len(p["q"]) == len(p["members"]) for p in parts), f"{item}: a shard is not finished"
    q, q1, m = (np.concatenate([p[k] for p in parts]) for k in ("q", "q1", "members"))
    order = np.argsort(m)
    q, q1, m = q[order], q1[order], m[order]
    assert np.array_equal(m, s), f"{item}: the shards hold other members than the sample"
    num, den, ddt = (torch.from_numpy(a[k]) for k in ("num", "den", "dd"))
    out: dict[str, Any] = {"E_b": SI.explained_share(num, den, ddt, torch.from_numpy(chK)),
                           "n_changed_in_K": int(chK.sum()), "n_standable": len(members), "n_sampled": len(s)}  # fmt: skip
    if not chK.any():
        return out | {"pass_share": 1.0, "pass_share_se": 0.0, "b_mean": 1.0, "b_mean_se": 0.0}  # the do-nothing convention
    c, r = float(dd[chK & ~A].sum()), q1 > floor
    b = np.where(r, 1 - q / np.where(r, q1, 1.0), 0.0)
    for k, y in (("pass_share", (q <= cfg.tol * q1) & r), ("b_mean", b)):
        out[k], out[f"{k}_se"] = ratio(dd[s], r, y.astype(np.float64), c, len(members)) if len(s) else (0.0 if c > 0 else 1.0, 0.0)
    return out | {"unstandable_dd_share": c / float(dd[chK].sum()), "unread_floor_dd_share_est": len(members) / max(len(s), 1) * float(dd[s][~r].sum()) / float(dd[chK].sum()),
                  "n_sampled_under_floor": int((~r).sum()), "n_sampled_passed": int(((q <= cfg.tol * q1) & r).sum()),
                  "q_median": float(np.median(q[r])) if r.any() else None, "q1_median": float(np.median(q1[r])) if r.any() else None}  # fmt: skip


def score(cfg: Config, n: int) -> None:
    floor = max(json.loads(p.read_text())["floor"] for p in RESULTS.glob("stage0_*.json"))  # the largest over the jobs' cards
    items = {i: score_item(cfg, i, n, floor) for i in cfg.items}
    dump(RESULTS / "summary.json", {"experiment": EXPERIMENT, "floor": floor, "tol": cfg.tol, "items": items})
    for i, r in items.items():
        log(f"{i}: b_mean {r['b_mean']:.4f} (se {r['b_mean_se']}), pass share {r['pass_share']:.4f} (se {r['pass_share_se']}), E_b {r['E_b']:.4f}; "
            f"{r['n_sampled']} of {r['n_standable']} sampled")  # fmt: skip


def inputs(w: Any, cfg: Config) -> None:
    """The CPU dry run's path: every item's F, M and saved arrays opened, nothing read."""
    for item in ("do_nothing", *cfg.items):
        w.make(item)
        if item != "do_nothing":
            saved(item)
        log(f"inputs: {item} opened")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg = load_config()
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}")
    if stage == "score":
        score(cfg, shard[1])
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226, E242, E247, E258
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    w = E242.World(E242.load_config()[0])
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    if stage == "inputs":
        inputs(w, cfg)
        return 0
    stage0(w, cfg, tag)
    for k, item in enumerate(cfg.items):
        read_item(w, cfg, item, shard)
        log(f"ITEM {k + 1} of {len(cfg.items)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("read", "score", "inputs"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th sampled member from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
