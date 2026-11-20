"""Side check of E226 (cell P1.37): one K* search per item run to convergence, without the filed evaluation cap. No
filed reading reads it, and it has no predictions of its own. Why it exists: the filed eval_cap of 2000 stops every
search early (reports/analyses/2026-11-20_analysis_e226-search-cap.md), and a definition search is never capped for
cost (user, 2026-11-16). It measures whether R moves from the capped set to the converged one: a move under the band
0.014 means the capped readings stand.

Each item is built exactly as run.py builds it (its World, KV.featurize, KV.prepare, KV.grades, its guards), and one
search runs on all of D with the filed settings except the cap, set to 10**9. search() needs no change for this: its
bisection and polish both end on their own (the polish when every member of the prefix has been dropped in a passing
chunk or tried alone at chunk 1), so `capped` reads False when it converged. Writes results/converge/<item>.json, with
a partial file (kept-set size, evaluations) every ~30 min while the search runs.

    PRED_SHA is not needed (require_filed is not called; the filed settings are only read):
    python experiments/kstar_vpd/converge.py --items do_nothing 2>&1 | tee -a experiments/kstar_vpd/results/converge_do_nothing.log
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute.jobs import hard_exit, load_experiment

RUN: Any = load_experiment("kstar_vpd", "run")
OUT = RUN.RESULTS / "converge"
CAP = 10**9
LOG_EVERY, PARTIAL_SECONDS = 500, 1800
log, dump = RUN.log, RUN.dump


def filed_reading(item: str) -> dict[str, Any] | None:
    """The scored run's main search for the item (capped), if its file is here."""
    paths = [RUN.RESULTS / "items" / f"{RUN.KP.fname(item)}.json"]
    paths += sorted(RUN.RESULTS.glob("stage0_*/do_nothing.json")) if item == "do_nothing" else []
    for p in paths:
        if p.is_file() and "searches" in (d := json.loads(p.read_text())):
            return {"file": str(p.relative_to(RUN.HERE)), "main": d["searches"]["main"]}
    return None


def watched(ev: Any, bar: float, item: str) -> Any:
    """ev with a progress line every LOG_EVERY evaluations and a partial JSON every PARTIAL_SECONDS. The kept size is
    the last passing set's: the search's current set is always the last set that passed."""
    st = {"evals": 0, "kept": None, "t0": time.time(), "last_dump": time.time()}

    def f(keep: Tensor) -> float:
        v = ev(keep)
        st["evals"] += 1
        st["kept"] = int(keep.sum()) if v <= bar else st["kept"]
        el = time.time() - st["t0"]
        if st["evals"] % LOG_EVERY == 0:
            log(f"  {item}: {st['evals']} evaluations, kept {st['kept']}, {el / 60:.1f} min")
        if time.time() - st["last_dump"] > PARTIAL_SECONDS:
            dump(OUT / f"{RUN.KP.fname(item)}.partial.json", {"item": item, "evals": st["evals"], "kept_size": st["kept"], "seconds": el})
            st["last_dump"] = time.time()
        return v

    return f


def converge(w: Any, item: str, cfg: Any, th: dict[str, Any]) -> None:
    path = OUT / f"{RUN.KP.fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return
    F, M, toks, at, rec = w.make(item)
    w.bases = w.bases or KV.bases(w.uv, cfg.split_tol)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    ctx = KV.prepare(F, M, feat, toks, at, cfg.batch)
    base = {"kl_none": float(ctx.kl_none.mean()), "kl_fm": float(ctx.kl_fm.mean()), "gates": KV.gates(ctx, feat.n)} | rec
    miss = RUN.guards(item, feat, KV.undecomposed_differences(M, F, w.keys), base["gates"], th) + RUN.KP.load_checks(item, base, RUN.KP_TH)
    assert not miss and base["kl_none"] > th["kl_floor"], f"{item}: {miss or 'no behaviour'}"
    s = KV.grades(M, F, feat, toks, at, cfg.grade_batch)
    t0, bar = time.time(), (1 - cfg.alpha) * base["kl_none"]
    ev = watched(lambda keep: KV.kl_mean(ctx, keep), bar, item)
    found = KV.search(ev, lambda c: KV.attribution(ctx, c), feat.n, bar, cfg.reranks, CAP, cfg.chunk_div, log)
    r = KV.read(found, s, feat)
    out = {"experiment": RUN.EXPERIMENT, "item": item, "side_check": "converged search, cap 10**9; no filed reading reads it",
           "evals": found.evals, "seconds": time.time() - t0, "bar": bar, "n_vars": feat.n, "prefix_size": r["prefix"]["size"],
           "polished_size": r["polished"]["size"], "read": r, "filed_capped": filed_reading(item)}  # fmt: skip
    dump(path, out)
    q = r["polished"]
    log(f"{item}: converged |K*| {q['size']} (prefix {r['prefix']['size']}), R read {q['R_read']:.6f}, R all {q['R_all']:.6f}, "
        f"f {q['f']:.4f}; {found.evals} evaluations, capped {found.capped}, {out['seconds']:.0f} s")  # fmt: skip
    ctx.hy.close()
    torch.cuda.empty_cache()


def main(items: list[str]) -> int:
    cfg, th = RUN.load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as run.py
    OUT.mkdir(parents=True, exist_ok=True)
    w = RUN.World()
    for item in items:
        converge(w, item, cfg, th)
        log(f"CONVERGE {item} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=["do_nothing", "new_thin", "x10"])
    code = 1
    try:
        code = main(ap.parse_args().items)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
