"""prior_vs_share (E133, cell P2.3): over every saved map on the 67M, does the original model's prior for the trained
answer order the maps the way the re-gating share does? Plan: reports/plans/2026-11-09_plan_e133-prior-vs-share.md.
Rules and thresholds: results/predictions.json; the maps and every key read: results/inventory.json (both filed and
committed before the run; the run refuses otherwise, never edited after).

The prior of a map is the original model's geometric-mean P(trained answer) at the trigger on the windows the share was
read on (`keydep.geo_rate` of `steering.load_task`'s log P, the path of `open_pair/amend3.py`; `copying.read_lo` for the
label pairs). No share is read here: every share and gain comes from a saved JSON. No fine-tuned state is loaded.

  stage 0  the control, alone: the forward reproduces two saved priors (copy, 96 keys; E54's elicit); a miss exits 1 ->
           results/stage0.json
  stage 1  the prior of every map: saved where a file holds it, computed by the forward where none does ->
           results/stage1.json
  stage 2  the saved shares, one point per map (the mean over seeds, the seed range kept); Spearman of share against
           log prior, its bootstrap interval over maps, the permutation null, the maps off the line, the reading ->
           results/stage2.json
Each stage prints `STAGE n DONE` after its file is written, and every scored number goes to the log. No verdict of
reused or new: the built labels are quoted, and the rule reads orderings only.
"""

from __future__ import annotations

import json
import math
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute import label_reads as LR
from commute import open_pair as OP
from commute import steering as ST
from commute.jobs import hard_exit, require_filed
from commute.local_faith import MIN_GAIN

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, INV = RESULTS / "predictions.json", RESULTS / "inventory.json"
HALF = 20  # the label pairs' trigger position: both were built with 20 tokens each side (predictions.json, their experiments')
LEN = 2 * HALF + 1
log = KD.log
_cache: dict[str, Any] = {}


def load(rel: str) -> Any:
    if rel not in _cache:
        _cache[rel] = json.loads((REPO / rel).read_text())
    return _cache[rel]


def lookup(rel: str, keys: list[Any]) -> Any:
    x = load(rel)
    for k in keys:
        x = x[k]
    return x


def dump(name: str, out: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- the prior


def label_windows(w: dict[str, Any]) -> Tensor:
    """The label pair's read windows with each one's label token as the last column (`label_reads.labelled`, as the
    pair's route read made them). Case label: first n_read of E112's fit windows, label from the word; repeat label: the
    first n_route fit windows, label per window."""
    t, n, a = torch.load(REPO / w["path"], weights_only=False), w["n_read"], w["arm"]
    if w["task"] == "case_label":
        toks = t["fit"][:n]
        y = torch.tensor([t["y"][a][int(k)] for k in toks[:, HALF - 1]])
    else:
        toks, y = t["toks"]["fit"][:n], t["y"][a]["fit"][:n]
    return LR.labelled(toks, y, t["ab"], HALF).tokens


def forward_prior(net: Any, row: dict[str, Any]) -> float:
    w = row["windows"]
    if w["format"] == "kv":
        return KD.geo_rate(ST.load_task(net, REPO / w["path"], row["name"], row["built"]).lp0)
    assert w["format"] == "label_task", row["name"]
    return KD.geo_rate(CP.read_lo(net, label_windows(w), HALF, LEN, readout=ST.logp_at))


def batched_lp(net: Any, t: ST.Task, batch: int = 64) -> Tensor:
    """W0's log P(target) in batches of 64: the one rounding reference the job has (reported only)."""
    return torch.cat([ST.logp_at(net.run(t.toks[s : s + batch])[0], t.half, t.tgt(t.toks[s : s + batch])) for s in range(0, len(t.toks), batch)])


# ----------------------------------------------------------------------------- stages


def stage0(net: Any, P: dict[str, Any]) -> dict[str, Any]:
    """The control, alone, on the path every computed prior takes."""
    S, tol = P["stage0"], P["stage0"]["tol_rel"]
    rec: dict[str, Any] = {"rows": {}, "tol_rel": tol}
    for c in S["controls"]:
        t = ST.load_task(net, REPO / c["windows"], c["name"], "control")
        p, ref = KD.geo_rate(t.lp0), lookup(c["file"], c["key"])
        rec["rows"][c["name"]] = {"prior": p, "saved": ref, "rel": p / ref - 1, "n_windows": len(t.toks), "ok": abs(p / ref - 1) <= tol}
        rec["rows"][c["name"]]["batched_vs_full_lp"] = float((batched_lp(net, t) - t.lp0).mean().abs())
        log(f"[stage0 {c['name']}] {json.dumps(rec['rows'][c['name']])}")
    rec["pass"] = all(r["ok"] for r in rec["rows"].values())
    dump("stage0.json", rec)
    return rec


def stage1(net: Any, maps: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for row in maps:
        src = row["prior"]
        p = lookup(src["file"], src["key"]) if src["source"] == "saved" else forward_prior(net, row)
        rec[row["name"]] = {"prior": p, "source": src["source"]}
        log(f"[stage1 {row['name']}] prior {p:.4e} ({src['source']})")
        dump("stage1.json", rec)
    return rec


def point(row: dict[str, Any], prior: float) -> dict[str, Any]:
    """One point per map: the mean share over the seeds at or above the gain floor, the range kept."""
    rs = [float(lookup(*s["r"])) for s in row["seeds"]]
    gs = [float(lookup(*s["gain"])) for s in row["seeds"]]
    kept = [r for r, g in zip(rs, gs, strict=True) if g >= MIN_GAIN and OP.finite(r)]
    p = {k: row[k] for k in ("name", "built", "family", "step")} | {"prior": prior, "seeds_r": rs, "seeds_gain": gs, "n_seeds": len(kept)}
    if not kept:
        return p | {"excluded": "no seed at or above the gain floor"}
    p |= {"share": sum(kept) / len(kept), "lo": min(kept), "hi": max(kept)}
    if not (OP.finite(prior) and prior > 0):
        return p | {"excluded": "prior not finite and positive"}
    return p | {"log10_prior": math.log10(prior)}


# ----------------------------------------------------------------------------- the score


def bootstrap(y: list[float], x: list[float], n_boot: int, seed: int) -> tuple[float, float, int]:
    """Percentile 2.5 and 97.5 of Spearman over resampled maps (seed fixed); a resample with a constant side is dropped.
    nan when under 90% of resamples are valid."""
    gen, n, out = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        i = gen.integers(0, n, n)
        rho = OP.spearman([y[k] for k in i], [x[k] for k in i])
        if not math.isnan(rho):
            out.append(rho)
    if len(out) < 0.9 * n_boot:
        return math.nan, math.nan, len(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


def permutation_null(y: list[float], x: list[float], n_perm: int, seed: int, rho: float) -> tuple[float, float]:
    """The 95th percentile of Spearman with the shares shuffled over the maps, and the one-sided p of the observed rho."""
    gen, out = np.random.default_rng(seed), []
    for _ in range(n_perm):
        p = gen.permutation(len(y))
        out.append(OP.spearman([y[k] for k in p], x))
    arr = np.array([v for v in out if not math.isnan(v)])
    if len(arr) == 0 or math.isnan(rho):
        return math.nan, math.nan
    return float(np.percentile(arr, 95)), float((1 + (arr >= rho).sum()) / (1 + len(arr)))


def off_line(x: list[float], y: list[float], lo: list[float], hi: list[float], k: float) -> tuple[list[float], float, list[int]]:
    """Leave-one-out residuals of share on log10 prior (a line fitted on the others, read at the map); sd = 1.4826 x MAD
    of the residuals. A map is off the line when |residual| > k sd and > its own seed range."""
    X, Y = np.array(x), np.array(y)
    n = len(X)
    if n < 4 or np.ptp(X) == 0 or np.ptp(Y) == 0:
        return [math.nan] * n, math.nan, []
    e = np.array([Y[i] - np.polyval(np.polyfit(np.delete(X, i), np.delete(Y, i), 1), X[i]) for i in range(n)])
    sd = float(1.4826 * np.median(np.abs(e - np.median(e))))
    idx = [i for i in range(n) if abs(e[i]) > k * sd and abs(e[i]) > hi[i] - lo[i]] if sd > 1e-9 else []
    return [float(v) for v in e], sd, idx


def reading(rho: float, lo: float, hi: float, null95: float, n: int, T: dict[str, Any]) -> str:
    """The filed rule: orderings only (see predictions.json rules, edge_cases)."""
    if n < T["min_maps"] or math.isnan(rho) or math.isnan(lo) or math.isnan(hi) or math.isnan(null95):
        return "no_call"
    if rho >= T["agree"] and lo >= T["weaker"] and rho > null95:
        return "agree"
    if rho <= T["weaker"] and hi < T["agree"]:
        return "weaker"
    return "no_call"


def score(pts: list[dict[str, Any]], T: dict[str, Any]) -> dict[str, Any]:
    ok = [p for p in pts if "excluded" not in p]
    y, x = [p["share"] for p in ok], [p["log10_prior"] for p in ok]
    n, rho = len(ok), OP.spearman(y, x) if len(ok) >= 3 else math.nan
    ci_lo, ci_hi, n_valid = bootstrap(y, x, T["n_boot"], T["seed_boot"]) if n >= 3 else (math.nan, math.nan, 0)
    null95, p_perm = permutation_null(y, x, T["n_perm"], T["seed_perm"], rho) if n >= 3 else (math.nan, math.nan)
    res, sd, idx = off_line(x, y, [p["lo"] for p in ok], [p["hi"] for p in ok], T["off_k"])
    for p, e in zip(ok, res, strict=True):
        p["residual"] = e
    off = [{"name": ok[i]["name"], "built": ok[i]["built"], "residual": res[i], "seed_range": ok[i]["hi"] - ok[i]["lo"]} for i in idx]
    return {"n_maps": n, "spearman": rho, "ci95": [ci_lo, ci_hi], "n_boot_valid": n_valid, "null95": null95, "p_perm": p_perm, "residual_sd": sd,
            "off_line": off, "reading": reading(rho, ci_lo, ci_hi, null95, n, T), "excluded": [{"name": p["name"], "why": p["excluded"]} for p in pts if "excluded" in p],
            "points": pts}  # fmt: skip


def stage2(priors: dict[str, Any], maps: list[dict[str, Any]], T: dict[str, Any]) -> dict[str, Any]:
    rec = score([point(row, priors[row["name"]]["prior"]) for row in maps], T)
    dump("stage2.json", rec)
    for p in rec["points"]:
        log(f"[score {p['name']}] {json.dumps(OP.jsonable(p))}")
    log(f"[score] {json.dumps(OP.jsonable({k: v for k, v in rec.items() if k != 'points'}))}")
    return rec


def main() -> None:
    require_filed(PRED)
    require_filed(INV)
    P, maps = json.loads(PRED.read_text()), json.loads(INV.read_text())["maps"]
    missing = [row["windows"]["path"] for row in maps if not (REPO / row["windows"]["path"]).is_file()]
    if missing:
        sys.exit(f"windows files missing: {missing}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # the earlier runs' setting
    net = ST.Net("cuda")
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()} maps {len(maps)}")
    s0 = stage0(net, P)
    if not s0["pass"]:
        log("FAIL: the control missed; stopping")
        hard_exit(1)
    print("STAGE 0 DONE", flush=True)
    s1 = stage1(net, maps)
    print("STAGE 1 DONE", flush=True)
    stage2(s1, maps, P["rules"])
    print("STAGE 2 DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
