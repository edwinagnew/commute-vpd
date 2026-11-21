"""E242's scorecard from the item files only (no GPU): the band, P1 (every reuse edit above both new edits on R_A), P2
(the size-matched random below the lowest reuse edit), P2k (the KL-matched random, the same), and the descriptive readings D1 (the nine pairs at matched size),
D2 (E62's copy/lookup pair), D3 (E237's key tasks), each beside the old squared-cosine R.

    ~/commute-agents/fleet/cpu-run python3 experiments/kstar_vpd_wording_a/score.py      (writes results/summary.json)
"""

from __future__ import annotations

import json
from statistics import fmean
from typing import Any

from commute.jobs import load_experiment

RUN: Any = load_experiment("kstar_vpd_wording_a", "run")


def verdict(gap: float, band: float) -> str:
    return "holds" if gap > band else "fails" if gap <= 0 else "no call"


def main() -> None:
    cfg, th = RUN.load_config()
    its = {p.stem: json.loads(p.read_text()) for p in (RUN.RESULTS / "items").glob("*.json") if not p.stem.endswith(".stopped")}
    read = {k: v for k, v in its.items() if v.get("reading") == "read"}
    has = lambda i: RUN.fname(i) in read

    def ra(i: str, k: str = "main") -> float:
        return read[RUN.fname(i)]["searches"][k]["R_A"]

    def old(i: str) -> float:
        return read[RUN.fname(i)]["R_old"]

    controls = (cfg.random, cfg.random_kl)
    out: dict[str, Any] = {"experiment": RUN.EXPERIMENT, "built": {i: {"R_A": ra(i), "R_old": old(i)} for i in (*cfg.built, *controls) if has(i)}}
    if all(has(i) for i in cfg.built):
        d = [abs(ra(i, "half1") - ra(i, "half2")) for i in cfg.built]
        band = max(d) if max(d) > 0 else th["band_min"]
        low = min(ra(i) for i in cfg.reuse)
        g1 = low - max(ra(i) for i in cfg.new)
        out |= {"band": band, "band_degenerate": max(d) == 0, "P1": {"gap": g1, "verdict": verdict(g1, band)}}
        for rule, i in (("P2", cfg.random), ("P2k", cfg.random_kl)):  # a control not read: its rule is not read
            out[rule] = {"gap": low - ra(i), "verdict": verdict(low - ra(i), band)} if has(i) else {"gap": None, "verdict": "not read"}

    def gap(r: list[str], n: list[str]) -> dict[str, float] | None:
        if not all(has(i) for i in (*r, *n)):
            return None
        return {"R_A": fmean(map(ra, r)) - fmean(map(ra, n)), "R_old": fmean(map(old, r)) - fmean(map(old, n))}

    pairs = {p: gap(r, n) for p, (r, n) in RUN.pair_sides().items()}
    done = {p: v for p, v in pairs.items() if v is not None}
    out["D1_pairs"] = {"gaps": pairs, "n_read": len(done), "reuse_higher_R_A": sum(v["R_A"] > 0 for v in done.values()),
                       "reuse_higher_R_old": sum(v["R_old"] > 0 for v in done.values())}  # fmt: skip
    out["D2_e62"] = gap([i for i in cfg.e62 if "/copy" in i], [i for i in cfg.e62 if "/lookup" in i])
    keys = RUN.key_items()
    tasks = sorted({i.split("/")[1].rsplit("_s", 1)[0] for i in keys})
    out["D3_keys"] = {t: {"R_A": [ra(i) if has(i) else None for i in keys if f"/{t}_s" in i], "R_old": [old(i) if has(i) else None for i in keys if f"/{t}_s" in i]}
                      for t in tasks}  # fmt: skip
    (RUN.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
