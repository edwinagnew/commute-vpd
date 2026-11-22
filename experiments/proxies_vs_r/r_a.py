"""Cell P2.51: R under wording A -> analyse.py's `--r` JSON, then the proxy scoring, in one command.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/r_a.py --source vpd   # E242, VPD featurization
    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/r_a.py --source sae   # E245, SAE featurization

Writes results/r_a_<source>.json (built edits, nine pairs as reuse side minus new side, five key tasks as the mean over
seeds 0-2) and runs analyse.main on it; --no-analyse stops after the JSON. A missing file or section is left out with a
printed line; an item not read stays null. Sources: E242's results/summary.json (R_A; D1_pairs.gaps; D3_keys), E245's
summary_built.json (R_A_tol.t1, the filed tolerance 0.1), summary_keys.json (keys.<task>_s<seed>.R_A.t1) and, when it
exists, summary_pairs.json read as {"gaps": {"e187_<pair>": gap}} (E245 had not filed that format when this was written).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import analyse as A

X = A.X
HERE = A.HERE
TOL = "t1"  # E245's filed tolerance 0.1 (predictions.json settings: side_tols 0.05 and 0.2 are the sensitivity pair)
SEEDS = (0, 1, 2)


def num(v: Any) -> float | None:
    """A reading from a number, None, or an E242/E245 record ({"R_A": ...} or {"t1": ...})."""
    if isinstance(v, dict):
        return num(v["R_A"]) if "R_A" in v else num(v.get(TOL))
    return v


def task_mean(xs: list[Any]) -> float | None:
    vals = [num(x) for x in xs]
    return fmean(vals) if len(vals) == len(SEEDS) and all(v is not None for v in vals) else None  # type: ignore[arg-type]


def vpd() -> dict[str, Any]:
    s = A.load("kstar_vpd_wording_a/results/summary.json")
    if s is None:
        return {}
    out: dict[str, Any] = {"built": {e: num(s["built"][e]) for e in A.BUILT if e in s["built"]}}
    if "D1_pairs" in s:
        out["pairs"] = {p: num(g) for p, g in s["D1_pairs"]["gaps"].items()}
    if "D3_keys" in s:
        d3 = s["D3_keys"]  # one item per seed, "<task>_s<seed>": {"R_A": [reading or null]}
        out["key"] = {t: task_mean([d3[f"{t}_s{k}"]["R_A"][0] for k in SEEDS if f"{t}_s{k}" in d3]) for t in A.KEY}
    return out


def sae(pairs_path: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    built, keys = (A.load(f"sae_r_a/results/summary_{g}.json") for g in ("built", "keys"))
    pairs = json.loads(Path(pairs_path).read_text()) if pairs_path else A.load("sae_r_a/results/summary_pairs.json")
    if built is not None:
        out["built"] = {e: num(built["items"][e]["R_A_tol"]) for e in A.BUILT if e in built["items"]}
    if keys is not None:
        k = keys["keys"]
        out["key"] = {t: task_mean([k[f"{t}_s{s}"]["R_A"] for s in SEEDS if f"{t}_s{s}" in k]) for t in A.KEY}
    if pairs is not None:
        out["pairs"] = {p: num(g) for p, g in pairs["gaps"].items()}
    return out


def random_note(source: str) -> str | None:
    """The matched random-change band for R_A, from E251: at SAE its 15 random updates at the key tasks' states against
    the 15 key states; at VPD its two E241 random updates (built size, KL 2f) against the key states' and built edits'
    range (E242). The VPD pair is not matched to the key tasks' size."""
    e251 = A.load("sae_r_a_random/results/summary.json")
    if e251 is None:
        return None
    if source == "sae":
        ft = [v["key_R_A"] for v in e251["per_state"].values()]
        rnd = [v["random_R_A"] for v in e251["per_state"].values()]
        where = "15 random updates at the key states' sizes (E251)"
    else:
        s = A.load("kstar_vpd_wording_a/results/summary.json")
        ft = [num(v["R_A"][0]) for v in s["D3_keys"].values()] + [v for e in A.BUILT if (v := num(s["built"].get(e))) is not None]
        rnd = [v["vpd"] for v in e251["D1"].values()]
        where = "E241's 2 random updates, built size and KL 2f, read by E251 (not matched to the key tasks)"
    return f"{A.inside(rnd, min(ft), max(ft))} inside; random {min(rnd):.4g} to {max(rnd):.4g}, fine-tunes {min(ft):.4g} to {max(ft):.4g}; {where}"  # type: ignore[type-var,arg-type]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("vpd", "sae"), required=True)
    ap.add_argument("--no-analyse", action="store_true")
    ap.add_argument("--sae-pairs", help="E245 pairs summary, if not yet in sae_r_a/results/")
    args = ap.parse_args()
    sections = vpd() if args.source == "vpd" else sae(args.sae_pairs)
    for sec in A.SETS:
        got = sections.get(sec)
        if got is None:
            print(f"{args.source}: section {sec} omitted (no summary yet)")
        else:
            read = sum(v is not None for v in got.values())
            print(f"{args.source}: {sec} read {read} of {len(got)}")
    path = HERE / "results" / f"r_a_{args.source}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": f"r_a_{args.source}", "featurization": args.source, "random": random_note(args.source), **sections}, indent=1))
    print(path)
    return 0 if args.no_analyse else A.main(["--r", str(path)])


if __name__ == "__main__":
    raise SystemExit(main())
