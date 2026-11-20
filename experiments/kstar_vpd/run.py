"""P1.37: R, f, the site-remainder share and |K*| at the VPD featurization of the 67M (subcomponents plus one remainder
per decomposed matrix), on the known-answer items first and then E187's nine pairs. Plan
reports/plans/2026-11-20_plan_p1-37-vpd.md and its appendix; filed settings, thresholds and rules in
results/predictions.json, which the run refuses unless committed unchanged. The measurement is commute.kstar_vpd.
Nothing is trained.

Items, each on its own windows D as its source experiment read it:
    stage 0  do_nothing                                   E214's cue windows, last position (F = M)
    stage 1  new_few, new_thin                            E214's built new edits, its cue windows, last position
             x2, x10, moved, rebuilt                      E195's reuse items as E214 read them, copy24 held-out, every position
    stage 2  every saved state of E187's nine pairs      as E221 read them: 512 read windows, last position
Stage 1 items get three searches (all of D, and its two halves for the band); every other item one. Per item,
results/items/<item>.json is written when it ends (with the split's norm shares per matrix and in total), and
results/large/<out>_<item>.npz holds the per-variable arrays (user amendment 2026-11-20): s (2 x n, fp32; row 0 every
position, row 1 the read positions), changed (bool), and per search <search>_in_K_polished, <search>_in_K_prefix (bool)
and <search>_attribution (rankings x n, fp32), with site_names and site_off for the variable layout (embedding 0, each
site's subcomponents from site_off then its remainder, the grafts, the unembedding last). A rerun skips an item whose
JSON exists. Stage 0 runs first and alone on every shard (results/stage0_<tag>.json, skipped once done;
exit 1 on a miss). Every guard of predictions.json R0 stops the job on a miss.

    python experiments/kstar_vpd/run.py --tag pod0 --shard 0 4 2>&1 | tee -a experiments/kstar_vpd/results/run_pod0.log
    python experiments/kstar_vpd/run.py --tag rehearse --out rehearse --items new_thin e58/cipher_s0   (the rehearsal)
    (on a pod: PRED_SHA=<sha> python ...)
    python experiments/kstar_vpd/score.py     (no GPU: band, R1, R1b, P-moved and T4 from the item files present)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

KL: Any = load_experiment("kstar_levels", "run")  # E214: World (do_nothing, the edits, x2, x10, moved, rebuilt), Config
KP: Any = load_experiment("kstar_pairs", "run")  # E221: World (E187's states), load_config, pairs, load_checks, fname
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop = KL.log, KL.dump, KL.stop
E214_ITEMS = ("do_nothing", "new_few", "new_thin", "x2", "x10", "moved", "rebuilt")  # built by E214's World
KP_TH = KP.load_config()[1]  # E221's load-check thresholds (kl_repro_rel, size_rel)
CHECKS = ("split_fp64", "F_from_pieces", "M_from_pieces")  # per-site guards, each against th["<name>_tol"]


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    alpha: float
    split_tol: float  # singular values below split_tol x max count as zero in the split's bases (split.bases)
    batch: int  # windows per F_K forward
    grade_batch: int  # windows per grade batch
    reranks: int
    eval_cap: int  # closure evaluations per search
    chunk_div: int  # the polish's first chunk is |prefix| // chunk_div
    split_seed: int  # the two halves of D
    reuse: tuple[str, ...]
    new: tuple[str, ...]

    def __post_init__(self) -> None:
        assert 0 < self.alpha < 1 and self.reranks >= 0 and self.eval_cap > 0 and self.chunk_div >= 1
        assert set(self.reuse) == {"x2", "x10", "moved", "rebuilt"} and set(self.new) == {"new_few", "new_thin"}

    @property
    def known(self) -> tuple[str, ...]:
        return (*self.new, *self.reuse)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def e187_states() -> list[str]:
    cfg, _ = KP.load_config()
    return list(dict.fromkeys(i for p, (r, n) in KP.pairs(cfg).items() if p.startswith("e187_") for i in (*r, *n)))


class World:
    """E214's and E221's item builders, the decomposition, and the split's bases; each loaded when first needed."""

    def __init__(self) -> None:
        from commute import vpd

        pred = json.loads(KL.PRED.read_text())
        s = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
        self.kl = KL.World(config_from_settings(KL.Config, s), pred["thresholds"])
        self.kp: Any = None
        model = self.kl.net.model
        self.uv, self.delta = vpd.component_weights(model), vpd.weight_deltas(model)
        self.keys = vpd.module_keys(model)
        w0 = vpd.checkpoint_weights()[0]
        self.w0_check = [k for k in self.keys if not torch.equal(self.kl.M.net.get_submodule(k).weight.cpu(), w0[k])]
        self.bases: dict[str, Any] = {}

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        """F, M, D, the read positions and E221's load record (E187 states only)."""
        if item in E214_ITEMS:
            F, toks, at, piece, _ = self.kl.make(item)
            assert piece is None
            return F, self.kl.M, toks, at, {}
        if self.kp is None:
            self.kp = KP.World(KP.load_config()[0])
            assert all(torch.equal(self.kp.M.net.get_submodule(k).weight, self.kl.M.net.get_submodule(k).weight) for k in self.keys)
        F, toks, at, rec = self.kp.make(item)
        return F, self.kp.M, toks, at, rec


# ---- one item -------------------------------------------------------------------------------------------------------


def guards(item: str, feat: KV.Feat, diffs: list[str], gates: dict[str, float], th: dict[str, Any]) -> list[str]:
    miss = [f"{k} {k2} {v:.3g}" for k, c in feat.checks.items() for k2, v in c.items()
            if k2 in CHECKS and not v <= th[f"{k2}_tol"]]  # fmt: skip
    miss += [f"parameters outside the decomposition differ: {diffs[:4]}"] if diffs else []
    miss += [f"gate {k} {v:.3g}" for k, v in gates.items() if not v <= th["gate_tol"]]
    if item == "do_nothing" and (feat.changed or any(s.changed for s in feat.sites)):
        miss.append(f"do-nothing has {len(feat.changed)} changed variables")
    return miss


def membership(K: list[int], n: int) -> np.ndarray:
    keep = np.zeros(n, dtype=bool)
    keep[K] = True
    return keep


def one_search(ctx: KV.Ctx, s: Tensor, feat: KV.Feat, cfg: Config, label: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The search's record (for the item JSON) and its per-variable arrays: K* membership (polished and prefix sets, bool)
    and the attribution scores of every ranking (fp32, rankings x variables)."""
    t0, bar = time.time(), (1 - cfg.alpha) * float(ctx.kl_none.mean())
    f = KV.search(lambda keep: KV.kl_mean(ctx, keep), lambda c: KV.attribution(ctx, c), feat.n, bar, cfg.reranks, cfg.eval_cap, cfg.chunk_div, log)
    rec: dict[str, Any] = KV.read(f, s, feat) | {"bar": bar, "seconds": time.time() - t0}
    arrays = {"in_K_polished": membership(f.polished, feat.n), "in_K_prefix": membership(f.prefix, feat.n),
              "attribution": torch.stack(f.attributions).float().numpy()}  # fmt: skip
    p, q = rec["prefix"], rec["polished"]
    log(f"  {label}: |K*| {q['size']} (prefix {p['size']}), R read {q['R_read']:.6f} (prefix {p['R_read']:.6f}), R all {q['R_all']:.6f}, "
        f"f {q['f']:.4f}, site-remainder share {q['site_remainder_share']:.4f}, graft share {q['graft_share']:.4f}; "
        f"{rec['evals']} evaluations{' (cap hit, chunk ' + str(rec['chunk_min']) + ')' if rec['capped'] else ''}, {rec['seconds']:.0f} s")  # fmt: skip
    return rec, arrays


def read_item(w: World, item: str, cfg: Config, th: dict[str, Any], out: Path) -> dict[str, Any]:
    path = out / f"{KP.fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    F, M, toks, at, rec = w.make(item)
    if not w.bases:
        w.bases = KV.bases(w.uv, cfg.split_tol)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    ctx = KV.prepare(F, M, feat, toks, at, cfg.batch)
    base = {"experiment": EXPERIMENT, "item": item, "n_windows": len(toks), "read_positions": "last" if at == KP.LAST else "all", "n_vars": feat.n,
            "n_changed": len(feat.changed), "n_grafts": len(feat.grafts), "kl_none": float(ctx.kl_none.mean()), "kl_fm": float(ctx.kl_fm.mean()),
            "checks": feat.checks, "split_shares": KV.norm_shares(feat), "gates": KV.gates(ctx, feat.n)} | rec  # fmt: skip
    log(f"{item}: {len(toks)} windows, {feat.n} variables, {len(feat.changed)} changed, {len(feat.grafts)} grafts; KL(F || F_empty) "
        f"{base['kl_none']:.6g}, KL(F || M) {base['kl_fm']:.9g}; gates {base['gates']}; worst split check "
        f"{max(c[k] for c in feat.checks.values() for k in CHECKS):.3g}; split shares over all 24 {base['split_shares']['total']}")  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    miss = guards(item, feat, KV.undecomposed_differences(M, F, w.keys), base["gates"], th) + KP.load_checks(item, base, KP_TH)
    if miss:
        stop(out / f"{KP.fname(item)}.stopped.json", base, f"{item}: " + "; ".join(miss))
    s = KV.grades(M, F, feat, toks, at, cfg.grade_batch)
    base["max_dev_s_unchanged"] = float((1 - s[:, [j for j in range(feat.n) if j not in feat.changed]]).abs().max())
    if not base["kl_none"] > th["kl_floor"]:
        dump(path, base | {"reading": "no behaviour"})
        return base | {"reading": "no behaviour"}
    variants = {"main": torch.arange(len(toks))}
    if item in cfg.known:
        perm = torch.randperm(len(toks), generator=torch.Generator().manual_seed(cfg.split_seed))
        variants |= {"half1": perm[: len(toks) // 2], "half2": perm[len(toks) // 2 :]}
    done = {k: one_search(ctx if k == "main" else ctx.subset(idx, cfg.batch), s, feat, cfg, f"{item} {k}") for k, idx in variants.items()}
    searches = {k: r for k, (r, _) in done.items()}
    arrays: dict[str, Any] = {f"{k}_{name}": v for k, (_, a) in done.items() for name, v in a.items()}
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / f"{out.name}_{KP.fname(item)}.npz", s=s.float().numpy(), changed=membership(sorted(feat.changed), feat.n),
                        site_names=np.array([x.name for x in feat.sites]), site_off=np.array([x.off for x in feat.sites]), **arrays)  # fmt: skip
    res = base | {"reading": "read", "searches": searches, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, res)
    log(f"{item}: done in {res['seconds']:.0f} s; peak GPU memory {res['peak_gpu_gb']:.2f} GB")
    ctx.hy.close()
    del ctx, F
    torch.cuda.empty_cache()
    return res


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def stage0(w: World, cfg: Config, th: dict[str, Any], tag: str, n_items: int) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    it = read_item(w, "do_nothing", cfg, th, RESULTS / f"stage0_{tag}")
    m = it["searches"]["main"]
    sec = m["seconds"] / max(m["evals"], 1)
    out = {"experiment": EXPERIMENT, "w0_differs": w.w0_check, "max_dev_s": it["max_dev_s_unchanged"], "seconds_per_eval": sec,
           "projected_hours": sec * cfg.eval_cap * n_items / 3600} | {f"{v}_{k}": m[v][k] for v in ("prefix", "polished") for k in ("f", "R_read", "R_all")}  # fmt: skip
    miss = [f"VPD target weights differ from M at {w.w0_check}"] if w.w0_check else []
    miss += [f"do-nothing: {out}"] if not (out["max_dev_s"] <= th["s_tol"] and all(out[f"{v}_f"] == 0 and abs(out[f"{v}_{g}"] - 1) <= th["s_tol"]
                                                                                    for v in ("prefix", "polished") for g in ("R_read", "R_all"))) else []  # fmt: skip
    miss += [f"projected {out['projected_hours']:.1f} h for {n_items} searches over {th['shard_hours_max']} h"] if out["projected_hours"] > th["shard_hours_max"] else []
    log(f"stage 0: {sec:.3f} s per evaluation, {out['projected_hours']:.2f} h projected for this shard; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def main(chosen: list[str] | None, shard: tuple[int, int], tag: str, out: str) -> int:
    cfg, th = load_config()
    every = [*cfg.known, *e187_states()]
    todo = chosen or every[shard[0] :: shard[1]]
    assert set(todo) <= set(every), f"unknown items {sorted(set(todo) - set(every))}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E214
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} of {len(every)} items {todo}; settings {asdict(cfg)}; thresholds {th}")
    w = World()
    stage0(w, cfg, th, tag, sum(3 if i in cfg.known else 1 for i in todo))
    for k, item in enumerate(todo):
        read_item(w, item, cfg, th, RESULTS / out)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=None, help="items by name (default: this shard of every item, known items first)")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this pod's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--out", default="items", help="results subdirectory for the item files (the rehearsal writes elsewhere)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.items, (args.shard[0], args.shard[1]), args.tag, args.out)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
