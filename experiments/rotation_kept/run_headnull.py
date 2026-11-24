"""P1.60 run 2 (E286's head (2, 4) block with a null; whip-lead 2026-11-23, relaying the user): copy head (2, 4) read
alone at the head featurization on E62's six fine-tunes and on six random updates, each matched in size to one of them
(`needed_neurons.matched_random`: a Gaussian per parameter tensor F changed, at the Frobenius norm of F - M on that
tensor). Settings and filed values in predictions.json's "headnull" block (added before this ran); run.py stays as it ran.

Per item, two readings of the one variable L2.h4:
    whole_d   E277's: `needed_neurons.write_pass` and its scan on all 512 windows of D at the last position; the kept
              share explained / dd (the paper's 0.095-0.141 copy, 0.015-0.020 lookup) and the head's flips
    copy      E286's head24, on D's 206 copy windows (head_edit.targets) at every position: strict, own-space and
              needed-space shares (the reference set R from this item's own scan of the copy windows, E268's settings),
              the rotation share and its dimension-count null, and the head's flips
Stage 0 runs first in every job (skipped once results/headnull/stage0_<tag>.json says done); any miss stops the job:
the identity interchange (x' = x) on the copy windows changes no argmax; the do-nothing reads dd exactly 0 and every
share 1; E280's built v2 (head 4's value rows doubled) reads strict and kept 1, own equal to the share its own space holds.
The six fine-tunes must repeat E286's saved strict and own and E277's saved kept share within the transcription tolerance.
results/headnull/<item>.json; a rerun skips written items.

    python experiments/rotation_kept/run_headnull.py --tag h0 --shard 0 2
    python experiments/rotation_kept/run_headnull.py --summary          (CPU)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import known_edits as KE
from commute import needed as IX
from commute import needed_neurons as NN
from commute import rotation_kept as RK
from commute import stand_in as SI
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

RUN: Any = load_experiment("rotation_kept", "run")  # E286: CFG, Nn (E277's e62 world and featurization), log/dump/stop
NNR, KN, HE = RUN.NNR, RUN.KN, RUN.HE  # E277 (write_pass, matched_random, HEAD24), E268 (scan settings), E280 (edited, targets)
OUT = RUN.RESULTS / "headnull"
BLOCK = json.loads(RUN.PRED.read_text())["headnull"]
CFG, NC, KC, HC = RUN.CFG, NNR.load_config(), KN.load_config(), HE.load_config()
U = NNR.HEAD24
log, dump, stop = RUN.log, RUN.dump, RUN.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "headnull" "settings"
    bases: tuple[str, ...]
    control: str
    random_seed: int
    control_tol: float
    saved_rel_tol: float  # H1 only: amended 2026-11-23 from the measured cross-card floor

    def __post_init__(self) -> None:
        assert self.bases == CFG.head_states and self.control in HE.EDITS and 0 < self.control_tol < CFG.transcription_rel_tol <= self.saved_rel_tol


H = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in BLOCK["settings"].items()})


def state(nn: Any, item: str) -> tuple[Any, Any, Tensor, slice, dict[str, Any]]:
    """F, M, D and the read position: a built edit of M on E280's D, a fine-tune, or (`<base>_rand`) M plus a Gaussian
    matched per tensor to that fine-tune's F - M (seed random_seed + the base's index)."""
    if item in ("do_nothing", H.control):
        toks, at = HE.data(nn.world.w, HC)
        return HE.edited(nn.world.w, HC, item), nn.world.w.kl.M, toks, at, {}
    base = item.removesuffix("_rand")
    F, M, toks, at, rec = nn.world.make(base)
    if item == base:
        return F, M, toks, at, rec
    dw, rec = NNR.matched_random(F.net, M.net, H.random_seed + H.bases.index(base))
    del F
    Fr = KE.clone(M)
    p = dict(Fr.net.named_parameters())
    with torch.no_grad():
        for k, d in dw.items():
            p[k].add_(d.to(p[k].device, p[k].dtype))
    return Fr, M, toks, at, rec | {"matched_to": base}


def scan_at(F: Any, feat: Any, toks: Tensor, at: slice, c: Any) -> tuple[IX.Scan, int]:
    run = NN.cache(F.net, toks, at, c.cache_batch)
    ex, pt = IX.derangements(len(toks), c.derangements, min(c.examples, len(toks)), c.seed)
    sc = NN.scan(F.net, feat, run, at, ex, pt, IX.rows_per_forward(c.rows, c.logit_rows, run.top.shape[1]), log)
    del run
    return sc, len(ex)


def whole_d(nn: Any, F: Any, M: Any, toks: Tensor, at: slice) -> dict[str, Any]:
    """E277's reading (read_item and head24_columns) of head (2, 4) on all of D."""
    num, den, dd, _, _ = NNR.write_pass(F, M, toks, nn.feat, NC.cache_batch)
    sc, n = scan_at(F, nn.feat, toks, at, NC)
    d = float(dd[U])
    return {"n_windows": len(toks), "dd": d, "kept": float(SI.explained(num, den)[U]) / d if d > 0 else 1.0, "flips": int(sc.flips[U]), "n_pairs": n}


def copy_windows(nn: Any, F: Any, M: Any, toks: Tensor, at: slice) -> dict[str, Any]:
    """E286's head24 reading on the copy windows, with the needed space from this item's own scan of them."""
    copy, _ = HE.targets(toks)
    tc = toks[copy.to(toks.device)]
    sc, n = scan_at(F, nn.feat, tc, at, KC)
    sites, ch = nn.pair(F, M, tc)
    st = RK.sets(sites, sc.flips, sc.first_pair >= 0, ch, CFG.top_frac, CFG.seed, N=[U])
    V = RK.gather(sites, RK.accumulate(F.net, sites, RK.wanted(sites, st), list(tc.split(CFG.cache_batch))))
    rows = RK.read(sites, st, V, CFG.own_energy, CFG.rank_rtol)
    r, p = rows[0], RK.pooled(rows, False)
    return {"n_windows": len(tc), "changed": bool(ch[U]), "dd": r["dd"], "strict": p["strict"], "own": p["own"], "needed": p["needed"],
            "rotation": p["rotation"], "rotation_nu_null": p["rotation_nu_null"], "own_null": r["dim_O"] / r["d_out"],
            "needed_null": r["dim_S"] / r["d_out"], "dim_O": r["dim_O"], "dim_S": r["dim_S"], "n_ref": r["n_ref"], "own_held": r["own_held"],
            "flips": int(sc.flips[U]), "n_pairs": n, "size_K": int((sc.first_pair >= 0).sum()), "order_misses": RK.ordering_misses(rows, CFG.order_tol)}  # fmt: skip


def saved(item: str) -> dict[str, Any] | None:
    """A fine-tune's saved values: E286's head24 item (copy windows) and E277's arrays (all of D)."""
    p = RUN.RESULTS / "items" / f"{RUN.fname('head24', item)}.json"
    if not p.is_file():
        return None
    h, z = json.loads(p.read_text()), np.load(NNR.RESULTS / "large" / f"items_{NNR.fname(item)}.npz")
    kept = float(SI.explained(torch.from_numpy(z["num"]), torch.from_numpy(z["den"]))[U]) / float(z["dd"][U])
    return {"copy": {k: h[k] for k in ("strict", "own", "flips")}, "whole_d": {"kept": kept, "flips": int(z["flips"][U])}}


def read_item(nn: Any, item: str) -> dict[str, Any]:
    path = OUT / f"{item.replace('/', '__')}.json"
    if path.is_file() and "stopped" not in (done := json.loads(path.read_text())):
        log(f"{item}: done before, skipped")
        return done
    t0 = time.time()
    F, M, toks, at, rec = state(nn, item)
    out = {"experiment": BLOCK["experiment"], "item": item, "head": "L2.h4"} | rec
    out |= {"whole_d": whole_d(nn, F, M, toks, at), "copy": copy_windows(nn, F, M, toks, at)}
    del F
    torch.cuda.empty_cache()
    miss = [f"ordering {m}" for m in out["copy"]["order_misses"]]
    if (sv := saved(item)) is not None:
        out["saved"] = sv
        tr = {f"{w}.{k}": RUN.rel(out[w][k], sv[w][k]) for w, k in (("copy", "strict"), ("copy", "own"), ("whole_d", "kept"))}
        out["transcription"] = tr
        miss += [f"{k} {v!r} off the saved value" for k, v in tr.items() if v > H.saved_rel_tol]
    out["seconds"] = time.time() - t0
    if miss:
        stop(path, out, f"{item}: " + "; ".join(miss))
    dump(path, out)
    w, c = out["whole_d"], out["copy"]
    log(f"{item}: all of D kept {w['kept']:.6g}, flips {w['flips']} of {w['n_pairs']}; copy windows strict {c['strict']:.6g}, own {c['own']:.6g} "
        f"(null {c['own_null']:.4f}), needed {c['needed']:.6g} (null {c['needed_null']:.4f}, {c['n_ref']} ref), rotation {c['rotation']}, "
        f"flips {c['flips']} of {c['n_pairs']}; saved {out.get('saved')}; {out['seconds']:.0f} s")  # fmt: skip
    return out


def stage0(nn: Any, tag: str) -> None:
    path = OUT / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    t0, out = time.time(), {"experiment": BLOCK["experiment"], "card": torch.cuda.get_device_name()}
    F, _, toks, at, _ = state(nn, H.control)
    tc = toks[HE.targets(toks)[0].to(toks.device)]
    run = NN.cache(F.net, tc, at, KC.cache_batch)
    ex = torch.arange(min(KC.examples, len(tc)))
    sc = NN.scan(F.net, nn.feat, run, at, ex, ex, IX.rows_per_forward(KC.rows, KC.logit_rows, run.top.shape[1]), log)
    out["identity"] = {"flips": int(sc.flips.sum()), "max_dev": sc.max_dev, "cases_per_s": sc.rows / sc.seconds}
    del F, run
    miss = [f"identity: {out['identity']['flips']} argmax changes"] if out["identity"]["flips"] else []
    dn, ct, tol = read_item(nn, "do_nothing"), read_item(nn, H.control), H.control_tol
    miss += [f"do_nothing: {w} dd {dn[w]['dd']!r}" for w in ("whole_d", "copy") if dn[w]["dd"] != 0]
    miss += [f"do_nothing: {k} {dn['copy'][k]!r}" for k in ("strict", "own", "needed") if dn["copy"][k] != 1.0]
    c = ct["copy"]
    miss += [f"{H.control}: kept {ct['whole_d']['kept']!r}"] if ct["whole_d"]["kept"] < 1 - tol else []
    miss += [f"{H.control}: strict {c['strict']!r}"] if c["strict"] < 1 - tol else []
    miss += [f"{H.control}: own {c['own']!r} off its own space's held share {c['own_held']!r}"] if abs(c["own"] - c["own_held"]) > tol else []
    miss += [f"{H.control}: needed {c['needed']!r} below own"] if c["needed"] < c["own"] - tol else []
    out |= {"do_nothing": dn, H.control: ct, "seconds": time.time() - t0}
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log(f"STAGE 0 DONE in {out['seconds']:.0f} s")


def summary() -> None:
    every = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json")) if not p.stem.startswith(("stage0_", "summary"))}
    rows = []
    for b in H.bases:
        for item in (b, f"{b}_rand"):
            r = every.get(item.replace("/", "__"))
            if r is not None:
                w, c = r["whole_d"], r["copy"]
                rows.append({"item": item, "arm": b.split("/")[1].split("_")[0], "random": item != b, "kept_all_D": w["kept"], "flips_all_D": w["flips"],
                             "strict_copy": c["strict"], "own_copy": c["own"], "own_null": c["own_null"], "needed_copy": c["needed"],
                             "needed_null": c["needed_null"], "rotation": c["rotation"], "rotation_nu_null": c["rotation_nu_null"],
                             "flips_copy": c["flips"], "n_pairs": c["n_pairs"]})  # fmt: skip
    dump(OUT / "summary.json", {"experiment": BLOCK["experiment"], "rows": rows})
    log(f"headnull summary: {len(rows)} of {2 * len(H.bases)} items")


def main(shard: tuple[int, int], tag: str) -> int:
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E277 and E286
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"{BLOCK['experiment']} start, shard {shard}, tag {tag}; settings {asdict(H)}")
    nn = RUN.Nn("head24")
    stage0(nn, tag)
    todo = [x for b in H.bases[shard[0] :: shard[1]] for x in (b, f"{b}_rand")]
    for n, item in enumerate(todo):
        read_item(nn, item)
        log(f"ITEM {n + 1} of {len(todo)} DONE ({item})")
    log(f"{BLOCK['experiment']} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th base (and its random) from the I-th")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/headnull/stage0_<tag>.json")
    ap.add_argument("--summary", action="store_true", help="CPU: the item files into results/headnull/summary.json")
    args = ap.parse_args()
    require_filed(RUN.PRED)
    code = 1
    try:
        if args.summary:
            summary()
            code = 0
        else:
            code = main((args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
