"""P1.36's per-pair table and filed rules (results/predictions.json "rules") from the item files present, under both
grades: "read" (the last position, main, predictions.json "main_grade") and "all" (every position, side). No GPU:

    python experiments/kstar_pairs/score.py            (writes results/summary.json)

A pair's gap is the mean over its reuse items minus the mean over its new items; an item's R, f, remainder share and |K*|
are the means over its main search's orders. An old reading's gap is oriented so that positive means the reuse side
reads more like reuse, by its own report's rule; update size enters as minus its gap (a smaller reuse-side update
positive). Every old reading is read from its saved results; `old_readings` names each file.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from commute.jobs import EXPERIMENTS, load_experiment

RUN: Any = load_experiment("kstar_pairs", "run")  # load_config, pairs, items, fname, RESULTS, E205
RESULTS = RUN.RESULTS
LEVELS = ("head_mlp", "mlp_attn", "block")
GRADES = ("read", "all")
X = EXPERIMENTS
Get = Callable[[str], float | None]  # an item (E196 name or E187 state) -> its value, None if not saved
E136_NAMES = {"copy24": "c3/copy", "cipher24": "c3/cipher", "ident32": "e46/ident32", "rand32": "e46/rand32", "case_real": "case_label/real",
              "case_twin": "case_label/twin", "repeat_real": "fresh/real", "repeat_twin": "fresh/twin"}  # fmt: skip


def jload(p: Path) -> Any:
    return json.loads(p.read_text()) if p.is_file() else None


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


# ---- this run's items -------------------------------------------------------------------------------------------------


def load_items(out: str = "items") -> dict[tuple[str, str], dict[str, Any]]:
    its = (json.loads(p.read_text()) for p in (RESULTS / out).glob("*.json") if not p.stem.endswith(".stopped"))
    return {(it["item"], it["level"]): it for it in its if it.get("reading") == "read"}


def item_value(rec: dict[str, Any] | None, key: str) -> float | None:
    """The mean over the main search's orders of `key` (R_read, R_all, f, remainder_share, size); None if not read."""
    return None if rec is None else mean([r[key] for r in rec["searches"]["main"]])


def gap(get: Get, reuse: list[str], new: list[str]) -> float | None:
    """mean over reuse - mean over new; None if any item's value is missing."""
    a, b = [get(i) for i in reuse], [get(i) for i in new]
    if any(v is None for v in a + b):
        return None
    return mean([v for v in a if v is not None]) - mean([v for v in b if v is not None])


def se(get: Get, reuse: list[str], new: list[str]) -> float | None:
    a, b = [get(i) for i in reuse], [get(i) for i in new]
    if any(v is None for v in a + b) or min(len(a), len(b)) < 2:
        return None
    return (statistics.variance([float(v) for v in a if v is not None]) / len(a) + statistics.variance([float(v) for v in b if v is not None]) / len(b)) ** 0.5


# ---- the old readings -------------------------------------------------------------------------------------------------


def e196_name(item: str, alias: dict[str, str]) -> str:
    """The E196 name of an item (E196 members read under their E187 state name map back)."""
    stem, s = item.rsplit("_s", 1)
    back = {v: k for k, v in alias.items()}
    return f"{back.get(stem, stem)}_s{s}"


def scorecard_gap(method: str) -> Callable[[str], float | None]:
    """The 67M scorecard's own oriented gap for an E187 pair (experiments/method_table/results/final.json)."""
    cells = {(c["pair"], c["method"]): c for c in jload(X / "method_table" / "results" / "final.json")["cells"]}
    return lambda pair: (c := cells.get((pair, method))) and c["gap"]


def crosscoder_gap(pairs: dict[str, tuple[list[str], list[str]]]) -> Callable[[str], float | None]:
    """E203's as-built crosscoder gap (seed 0 only) for the E187 pair whose two sides hold the cell's two states."""
    cells = [c for c in jload(X / "crosscoder" / "results" / "run2" / "summary.json")["cells"] if c["column"] == "crosscoder"]

    def get(pair: str) -> float | None:
        r, n = pairs[f"e187_{pair}"]
        hit = [c["gap"] for c in cells if c["sides"]["reuse"].replace("__", "/") in r and c["sides"]["new"].replace("__", "/") in n]
        return hit[0] if len(hit) == 1 else None

    return get


def rung_key(r: dict[str, Any]) -> float:
    """E184's ranking (rung, then n; lower is more reuse-like; rung 4 has no n), as one number, minus: positive is reuse-like."""
    return -(r["rung"] + (r["n"] or 0) / 100)


def old_readings(cfg: Any, pairs: dict[str, tuple[list[str], list[str]]]) -> dict[str, dict[str, Any]]:
    """Per reading: its per-pair oriented gaps (None where not saved), whether it is read by sign only, and its source."""
    alias = dict(cfg.e196_as_e187)
    E205_ITEMS = RUN.E205.ITEMS

    def per_item(e196: Callable[[str], float | None], e187: Callable[[str], float | None] | None) -> dict[str, float | None]:
        out = {}
        for p, (r, n) in pairs.items():
            if p.startswith("e196_"):
                out[p] = gap(lambda i: e196(e196_name(i, alias)), r, n)
            else:
                out[p] = gap(e187, r, n) if e187 else None
        return out

    def field(path: Callable[[str], Path], f: Callable[[Any], float | None]) -> Callable[[str], float | None]:
        def get(i: str) -> float | None:
            d = jload(path(i))
            return None if d is None else f(d)

        return get

    rho = field(lambda i: X / "rho_saved_items/results/items" / f"{i}.json", lambda d: mean(d["replace"]["rho_main"]) if d["replace"]["reading"] == "read" else None)
    ret = field(lambda i: X / "rule_retention_pairs/results/items" / f"{i}.json", lambda d: mean(d["A"]["contrast"]["retention"]) if d["A"]["contrast"]["reading"] == "read" else None)
    rung = field(lambda i: X / "simplest_intervention/results" / f"{i}.json", lambda d: rung_key(d["readings"]["all"]["0.9"]))
    labels = {it["item"]: it.get("G") for it in jload(X / "old_task_labels/results/labels.json")["items"]}
    share136 = jload(X / "share_untrained/results/stage1.json")["states"]

    def share196(i: str) -> float | None:
        stem, s = i.rsplit("_s", 1)
        return share136.get(f"{E136_NAMES[stem]}_s{s}", {}).get("trained") if stem in E136_NAMES else None

    def S(r: str) -> tuple[Get, Get]:
        return (field(lambda i: X / "held_model_retention/results/items" / f"{i}.json", lambda d: d["readings"][r]["S"]),
                field(lambda i: E205_ITEMS / f"{RUN.fname(i)}.json", lambda d: d["readings"][r]["S"]))  # fmt: skip

    def pair_level(f: Callable[[str], float | None]) -> dict[str, float | None]:
        return {p: f(p.removeprefix("e187_")) if p.startswith("e187_") else None for p in pairs}

    out: dict[str, dict[str, Any]] = {
        "rho_E208": {"gaps": per_item(rho, None), "source": "experiments/rho_saved_items/results/items/<item>.json .replace.rho_main (mean over orders)"},
        "retention_A_E196": {"gaps": per_item(ret, None), "source": "experiments/rule_retention_pairs/results/items/<item>.json .A.contrast.retention (mean over orders)"},
        "S3_held_model": {"gaps": per_item(*S("3")), "source": "E197 experiments/held_model_retention/results/items/<item>.json and E205 experiments/held_model_size_matched/results/items/<state>.json .readings['3'].S"},
        "S1_held_model": {"gaps": per_item(*S("1")), "source": "as S3_held_model, .readings['1'].S"},
        "share": {"gaps": {**per_item(share196, None), **{k: v for k, v in pair_level(scorecard_gap("share")).items() if k.startswith("e187_")}},
                  "source": "E196 pairs: experiments/share_untrained/results/stage1.json .states[<E136 name>].trained; E187 pairs: experiments/method_table/results/final.json share .gap"},
        "steering": {"gaps": pair_level(scorecard_gap("steering")), "source": "experiments/method_table/results/final.json steering .gap"},
        "neuron_top": {"gaps": pair_level(scorecard_gap("neuron_top")), "source": "experiments/method_table/results/final.json neuron_top .gap"},
        "k_half": {"gaps": pair_level(scorecard_gap("k_half")), "source": "experiments/method_table/results/final.json k_half .gap (log10 new / reuse)"},
        "crosscoder_E203": {"gaps": pair_level(crosscoder_gap(pairs)), "source": "experiments/crosscoder/results/run2/summary.json column crosscoder .gap (seed 0)"},
        "rung_E184": {"gaps": per_item(rung, None), "sign_only": True, "source": "experiments/simplest_intervention/results/<item>.json .readings.all['0.9'] (rung, n), minus"},
        "G_E142": {"gaps": per_item(lambda i: labels.get(i), None), "sign_only": True, "source": "experiments/old_task_labels/results/labels.json .items[].G"},
    }  # fmt: skip
    return out


# ---- the rules --------------------------------------------------------------------------------------------------------


def ranks(v: list[float]) -> list[float]:
    """Ranks from 1, ties given their average rank."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: list[float], y: list[float], n_min: int) -> float | None:
    """Pearson correlation of the ranks; None when n < n_min or either side is constant (no order to compare)."""
    if len(x) < n_min:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def tally(gR: dict[str, float | None], gX: dict[str, float | None], g_small: dict[str, float | None], th: dict[str, Any], sign_only: bool = False) -> dict[str, Any]:
    """The T4 rule on the pairs where both gaps exist: signs agree (both non-zero, same sign) on every pair AND
    Spearman >= rho_bar -> 'tracks'; else 'does not track'; 'not read' under n_min pairs or with a constant side;
    a sign-only reading is never called. Beside it, R's and X's Spearman with the smaller update on the same pairs."""
    P = [p for p in gR if gR[p] is not None and gX.get(p) is not None]
    x, y = [float(gR[p]) for p in P], [float(gX[p]) for p in P]  # type: ignore[arg-type]
    agree = sum(sign(a) * sign(b) > 0 for a, b in zip(x, y, strict=True))
    out: dict[str, Any] = {"n": len(P), "pairs": P, "sign_agree": agree, "rho": None if sign_only else spearman(x, y, th["n_min"])}
    Q = [p for p in P if g_small.get(p) is not None]
    out |= {"rho_R_smaller_update": spearman([float(gR[p]) for p in Q], [float(g_small[p]) for p in Q], th["n_min"]),  # type: ignore[arg-type]
            "rho_X_smaller_update": spearman([float(gX[p]) for p in Q], [float(g_small[p]) for p in Q], th["n_min"])}  # type: ignore[arg-type]  # fmt: skip
    if sign_only:
        return out | {"call": f"sign only: {agree} of {len(P)}"}
    if out["rho"] is None:
        return out | {"call": "not read"}
    return out | {"call": "tracks" if agree == len(P) and out["rho"] >= th["rho_bar"] else "does not track"}


Items = dict[tuple[str, str], dict[str, Any]]
Pairs = dict[str, tuple[list[str], list[str]]]


def at_level(d: Items, L: str, key: str) -> Get:
    return lambda i: item_value(d.get((i, L)), key)


def gaps(get: Get, pairs: Pairs) -> dict[str, float | None]:
    return {p: gap(get, r, n) for p, (r, n) in pairs.items()}


def level(d: Items, L: str, th: dict[str, Any], pairs: Pairs, old: dict[str, dict[str, Any]], size: dict[str, dict[str, float | None]], items: list[str]) -> dict[str, Any]:
    """One level: per grade the R gaps, their seed se and the T4 tally against every reading; then f, the remainder share and the f check."""
    small, small_mlp = ({p: -v if v is not None else None for p, v in size[k].items()} for k in ("whole", "mlp"))
    lv: dict[str, Any] = {}
    for g in GRADES:
        get = at_level(d, L, f"R_{g}")
        gR = gaps(get, pairs)
        per = {"gap_R": gR, "se_R": {p: se(get, r, n) for p, (r, n) in pairs.items()}}
        per |= {"smaller_update": tally(gR, small, small, th), "smaller_mlp_update": tally(gR, small_mlp, small, th),
                "K_size": tally(gR, gaps(at_level(d, L, "size"), pairs), small, th)}  # fmt: skip
        lv[g] = per | {name: tally(gR, o["gaps"], small, th, o.get("sign_only", False)) for name, o in old.items()}
    lv["read_vs_all"] = tally(lv["read"]["gap_R"], lv["all"]["gap_R"], small, th)
    lv["f"] = gaps(at_level(d, L, "f"), pairs)
    lv["remainder_share_max"] = max((v for i in items if (v := at_level(d, L, "remainder_share")(i)) is not None), default=None)
    lv["f_check_misses"] = [i for i in items if (rec := d.get((i, L))) and any(
        not u["changed"] for r in rec["searches"]["main"] for u in r["K"] if u["name"] not in ("embed", "unembed"))]  # fmt: skip
    return lv


def score(d: Items, cfg: Any, th: dict[str, Any], pairs: Pairs, old: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sizes = {i: rec["size"] for (i, _), rec in d.items()}  # an item's update size, the same at every level
    size = {k: gaps(lambda i, k=k: sizes[i][k] if i in sizes else None, pairs) for k in ("whole", "mlp")}
    out: dict[str, Any] = {"main_grade": json.loads(RUN.PRED.read_text())["main_grade"], "n_items_read": len(sizes)}
    out["levels"] = {L: level(d, L, th, pairs, old, size, RUN.items(cfg)) for L in LEVELS}
    main = json.loads(RUN.PRED.read_text())["main_grade"]
    complete = all(v is not None for L in LEVELS for v in out["levels"][L][main]["gap_R"].values())
    rho_small = [out["levels"][L][main]["smaller_update"]["rho"] for L in LEVELS]
    rho_S3 = [out["levels"][L][main]["S3_held_model"]["rho"] for L in LEVELS]
    out["P1"] = "not read" if not complete or None in rho_small else all(r >= th["rho_bar"] for r in rho_small if r is not None)
    out["P2"] = "not read" if not complete or None in rho_S3 else all(r < th["rho_bar"] for r in rho_S3 if r is not None)
    out["size"] = size
    return out


def control(pairs: dict[str, tuple[list[str], list[str]]], size_whole: dict[str, float | None], th: dict[str, Any]) -> dict[str, Any]:
    """Positive control of the gap and rank code: this run's size gaps against E187's saved sizes on its nine pairs."""
    m67 = RUN.E205.load_pairs().m67
    saved = {p: mean(m67[p[5:]]["sizes"]["reuse"]) - mean(m67[p[5:]]["sizes"]["new"]) if p.startswith("e187_") else None for p in pairs}
    t = tally(size_whole, saved, {}, th)
    return t | {"holds": "not read" if t["n"] < 9 else t["sign_agree"] == t["n"] and t["rho"] is not None and t["rho"] >= th["control_rho"]}


def main() -> int:
    cfg, th = RUN.load_config()
    pairs = RUN.pairs(cfg)
    d = load_items()
    out = score(d, cfg, th, pairs, old_readings(cfg, pairs))
    out["control"] = control(pairs, out["size"]["whole"], th)
    out["experiment"] = RUN.EXPERIMENT
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    m = out["main_grade"]
    for L in LEVELS:
        row = out["levels"][L][m]
        print(L, {k: (v["call"], v["n"], v["sign_agree"], v["rho"]) for k, v in row.items() if isinstance(v, dict) and "call" in v})
    print("P1", out["P1"], "P2", out["P2"], "control", out["control"]["holds"], "items read", out["n_items_read"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
