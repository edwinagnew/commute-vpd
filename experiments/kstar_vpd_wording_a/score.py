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

    def ra(i: str, k: str = "main") -> float:  # R_A; None when the item's (1 - alpha) KL(F || M) is under the floor
        return read[RUN.fname(i)]["searches"][k]["R_A"]

    def old(i: str) -> float:
        return read[RUN.fname(i)]["R_old"]

    controls = (cfg.random, cfg.random_kl)
    out: dict[str, Any] = {"experiment": RUN.EXPERIMENT, "built": {i: {"R_A": ra(i), "R_A_abs": read[RUN.fname(i)]["searches"]["main"].get("R_A_abs"), "R_old": old(i)}
                                                                   for i in (*cfg.built, *controls) if has(i)}}  # fmt: skip
    if all(has(i) for i in cfg.built):
        ok = [i for i in cfg.built if ra(i) is not None]  # a built item under the floor is left out of the rules, and listed
        d = [abs(a - b) for i in ok if (a := ra(i, "half1")) is not None and (b := ra(i, "half2")) is not None]
        band = max(d) if d and max(d) > 0 else th["band_min"]
        reuse, new = [ra(i) for i in cfg.reuse if i in ok], [ra(i) for i in cfg.new if i in ok]
        out |= {"band": band, "band_degenerate": not d or max(d) == 0, "left_out_under_floor": [i for i in cfg.built if i not in ok]}
        low = min(reuse) if reuse else None
        g1 = low - max(new) if low is not None and new else None
        out["P1"] = {"gap": g1, "verdict": verdict(g1, band) if g1 is not None else "not read"}
        for rule, i in (("P2", cfg.random), ("P2k", cfg.random_kl)):  # a control not read: its rule is not read
            r = ra(i) if has(i) else None
            out[rule] = {"gap": low - r, "verdict": verdict(low - r, band)} if low is not None and r is not None else {"gap": None, "verdict": "not read"}

    def gap(r: list[str], n: list[str]) -> dict[str, float | None] | None:
        if not all(has(i) for i in (*r, *n)):
            return None
        a, b = [x for i in r if (x := ra(i)) is not None], [x for i in n if (x := ra(i)) is not None]  # states under the floor left out
        return {"R_A": fmean(a) - fmean(b) if a and b else None, "R_old": fmean(map(old, r)) - fmean(map(old, n))}

    pairs = {p: gap(r, n) for p, (r, n) in RUN.pair_sides().items()}
    done = {p: v for p, v in pairs.items() if v is not None}
    pos = lambda k: sum((x := v[k]) is not None and x > 0 for v in done.values())
    out["D1_pairs"] = {"gaps": pairs, "n_read": len(done), "reuse_higher_R_A": pos("R_A"), "reuse_higher_R_old": pos("R_old")}
    out["D2_e62"] = gap([i for i in cfg.e62 if "/copy" in i], [i for i in cfg.e62 if "/lookup" in i])
    keys = RUN.key_items()
    tasks = sorted({i.split("/")[1].rsplit("_s", 1)[0] for i in keys})
    out["D3_keys"] = {t: {"R_A": [ra(i) if has(i) else None for i in keys if f"/{t}_s" in i], "R_old": [old(i) if has(i) else None for i in keys if f"/{t}_s" in i]}
                      for t in tasks}  # fmt: skip
    (RUN.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
