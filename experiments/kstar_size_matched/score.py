"""P1.38's per-pair table, the filed rule and the rank correlations, from saved files only (no GPU):

    python experiments/kstar_size_matched/score.py      (writes results/summary.json; then plot.py)

A pair's R gap is E221's: the mean over the reuse side's items minus the mean over the new side's, each item's R the
mean over its main search's orders (E221 score.py `gap`, `item_value`). At saved size every item is E221's file; at
matched size the larger side's items are this run's scaled files and the smaller side's stay E221's. Update size enters
the rank correlations as minus its gap (a smaller reuse-side update positive), as in E221.
"""

from __future__ import annotations

import json
from typing import Any

from commute.jobs import load_experiment

S: Any = load_experiment("kstar_pairs", "score")  # gap, se, item_value, spearman, sign, old_readings, load_items, LEVELS
RUN: Any = load_experiment("kstar_size_matched", "run")  # load_config, RESULTS, PRED, EXPERIMENT, E205
GRADES = ("read", "all")
Pairs = dict[str, tuple[list[str], list[str]]]


def load_mine() -> dict[tuple[str, str], dict[str, Any]]:
    its = (json.loads(p.read_text()) for p in (RUN.RESULTS / "items").glob("*.json") if not p.stem.endswith(".stopped"))
    return {(it["item"], it["level"]): it for it in its if it.get("reading") == "read"}


def matched(unscaled: Pairs) -> Pairs:
    """E187's pairs with the larger side's states replaced by their scaled items (E205's names)."""
    pr = RUN.E205.load_pairs()
    out = {}
    for p, (r, n) in unscaled.items():
        pid = p.removeprefix("e187_")
        big = pr.m67[pid]["larger"]
        out[p] = ([pr.item(s, pid).name for s in r] if big == "reuse" else r, [pr.item(s, pid).name for s in n] if big == "new" else n)
    return out


def partial(x: list[float], y: list[float], z: list[float], n_min: int) -> float | None:
    """Spearman of x and y with z held fixed: the partial correlation of the three rank vectors; None as `spearman` is,
    or when z orders x or y perfectly (nothing left to correlate)."""
    rxy, rxz, ryz = S.spearman(x, y, n_min), S.spearman(x, z, n_min), S.spearman(y, z, n_min)
    if rxy is None or rxz is None or ryz is None or max(abs(rxz), abs(ryz)) >= 1:
        return None
    return (rxy - rxz * ryz) / ((1 - rxz**2) * (1 - ryz**2)) ** 0.5


def both(a: dict[str, float | None], b: dict[str, float | None]) -> list[str]:
    return [p for p in a if a[p] is not None and b.get(p) is not None]


def rule(gun: dict[str, float | None], gm: dict[str, float | None], th: dict[str, Any]) -> dict[str, Any]:
    """The filed rule: a pair keeps its sign iff its saved-size and matched gaps have the same non-zero sign; holds iff
    at least kept_min of the nine keep it; 'not read' unless all nine pairs have both gaps."""
    P = both(gun, gm)
    kept = [p for p in P if S.sign(gun[p]) * S.sign(gm[p]) > 0]
    out = {"n": len(P), "kept": len(kept), "flipped_or_zero": [p for p in P if p not in kept], "reuse_higher_matched": sum(gm[p] > 0 for p in P)}  # type: ignore[operator]
    return out | {"holds": "not read" if len(P) < 9 else len(kept) >= th["kept_min"]}


def level(d: dict[tuple[str, str], Any], L: str, un: Pairs, mt: Pairs, size: dict[str, dict[str, float | None]], th: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in GRADES:
        get = S.at_level(d, L, f"R_{g}")
        gun, gm = S.gaps(get, un), S.gaps(get, mt)
        out[g] = {"gap_unscaled": gun, "gap_matched": gm, "se_unscaled": {p: S.se(get, *v) for p, v in un.items()},
                  "se_matched": {p: S.se(get, *v) for p, v in mt.items()}, "signs": rule(gun, gm, th)}  # fmt: skip
    gun, gm = out["read"]["gap_unscaled"], out["read"]["gap_matched"]
    scaled, get = RUN.scaled_items(), S.at_level(d, L, "R_read")

    def c2(i: str) -> float | None:  # rival b: a scaled state's 1 - R is c^2 times its saved-size 1 - R
        if i not in scaled:
            return get(i)
        r = get(scaled[i].state)
        return None if r is None else 1 - scaled[i].c ** 2 * (1 - r)

    out["read"]["gap_c2_law"] = S.gaps(c2, mt)
    P = both(gun, gm)
    dR = [gm[p] - gun[p] for p in P]  # type: ignore[operator]
    dsize = [-(size["matched"][p] - size["unscaled"][p]) for p in P]  # type: ignore[operator]
    out["within_pair_rho"] = {"n": len(P), "rho": S.spearman(dR, dsize, th["n_min"]), "of": "matched minus saved R gap against minus the matched minus saved size gap"}
    return out


def size_gaps(d: dict[tuple[str, str], Any], pairs: Pairs) -> dict[str, float | None]:
    sizes = {i: rec["size"]["whole"] for (i, _), rec in d.items()}
    return S.gaps(lambda i: sizes.get(i), pairs)


def correlations(d: dict[tuple[str, str], Any], L: str, cfg: Any, th: dict[str, Any], every: Pairs, mt: Pairs, size_m: dict[str, float | None]) -> dict[str, Any]:
    """(i) over E221's 14 saved-size pairs: Spearman of the R gap with each earlier reading's gap, plain and with minus
    the size gap held fixed; (ii) over the nine matched pairs, Spearman with E205's matched held-model S gaps (size held
    by construction); (iii) pooled over the 14 saved-size and 9 matched readings: Spearman of the R gap with minus the size gap."""
    get = S.at_level(d, L, "R_read")
    gR, small = S.gaps(get, every), {p: -v if v is not None else None for p, v in size_gaps(d, every).items()}
    out: dict[str, Any] = {"partial_given_size": {}}
    for name, o in S.old_readings(cfg, every).items():
        if o.get("sign_only"):
            continue
        P = [p for p in both(gR, o["gaps"]) if small[p] is not None]
        x, y, z = [gR[p] for p in P], [o["gaps"][p] for p in P], [small[p] for p in P]
        out["partial_given_size"][name] = {"n": len(P), "rho": S.spearman(x, y, th["n_min"]), "partial": partial(x, y, z, th["n_min"])}  # type: ignore[arg-type]
    e205 = json.loads((RUN.E205.RESULTS / "summary.json").read_text())["pairs"]
    gm = S.gaps(get, mt)
    for r in ("3", "1"):
        P = [p for p in gm if gm[p] is not None]
        out[f"matched_vs_S{r}"] = {"n": len(P), "rho": S.spearman([gm[p] for p in P], [e205[p[5:]][f"reading {r}"]["gap_matched"] for p in P], th["n_min"])}
    x = [v for v in gR.values() if v is not None] + [v for v in gm.values() if v is not None]
    z = [small[p] for p in gR if gR[p] is not None] + [-size_m[p] for p in gm if gm[p] is not None]  # type: ignore[operator]
    out["pooled_vs_smaller_update"] = {"n": len(x), "rho": S.spearman(x, z, th["n_min"]) if None not in z else None}
    return out


def main() -> int:
    cfg, th = RUN.load_config()
    d = S.load_items() | load_mine()
    every = S.RUN.pairs(cfg)
    un = {p: v for p, v in every.items() if p.startswith("e187_")}
    mt = matched(un)
    size = {"unscaled": size_gaps(d, un), "matched": size_gaps(d, mt)}
    m67 = RUN.E205.load_pairs().m67
    small_side = {p: S.mean(m67[p[5:]]["sizes"]["new" if m67[p[5:]]["larger"] == "reuse" else "reuse"]) for p in un}
    rel = {p: abs(v) / small_side[p] if v is not None else None for p, v in size["matched"].items()}
    out: dict[str, Any] = {"experiment": RUN.EXPERIMENT, "n_items_read": len({i for i, _ in load_mine()}), "size_gap": size,
                           "control_size_matched": {"rel_gap": rel, "holds": "not read" if None in rel.values() else max(rel.values()) <= th["size_match_rel"]}}  # type: ignore[type-var]  # fmt: skip
    out["levels"] = {L: level(d, L, un, mt, size, th) | {"correlations": correlations(d, L, cfg, th, every, mt, size["matched"])} for L in S.LEVELS}
    out["P_signs_block"] = out["levels"]["block"]["read"]["signs"]
    out["plot"] = {"n": "2 to 8 seeds per side, 9 pairs; read grade", "panels": {L: [
        {"pair": p[5:], "unscaled": v["gap_unscaled"][p], "se": v["se_unscaled"][p] or 0.0, "matched": v["gap_matched"][p]} for p in un]
        for L, v in ((L, out["levels"][L]["read"]) for L in S.LEVELS)}}  # fmt: skip
    (RUN.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    for L in S.LEVELS:
        print(L, out["levels"][L]["read"]["signs"], "within-pair rho", out["levels"][L]["within_pair_rho"]["rho"])
    print("P (block, read):", out["P_signs_block"]["holds"], "| size-matched control:", out["control_size_matched"]["holds"], "| items read", out["n_items_read"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
