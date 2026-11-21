"""Cell P2.51 ($0, no run): rank agreement of cheap proxies with the reuse score R, from saved readings only.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/analyse.py                # size-removed R
    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/analyse.py --r normalised  # P1.43's N
    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/analyse.py --r PATH.json   # e.g. R under A

R sources: "size_removed" (P1.42: minus the residual of log(1 - R) on log update size; the five key tasks get the same
fitted law applied to E237's R at VPD), "normalised" (P1.43's N; no key tasks), or a JSON file
{"name": str, "built": {edit: R}, "pairs": {"e187_<pair>": R reuse side minus new side}, "key": {task: R}}, any
section may be missing. When definition-lead's P1.46 lands, write its readings in that form and pass the path.

Item sets: the six built edits (values), the nine built pairs (reuse side minus new side), the five 96-key tasks of
E237 at one common update size (task means over seeds 0-2). Every proxy is oriented so that higher reads as reuse by
its own definition. A missing reading is None ("not read"), never imputed. Ranks tie values equal to 6 significant
figures (x2 and moved share one size; x2 and x10 one effective rank, by construction). Writes results/table_<name>.json
and figures/proxies_vs_r_<name>.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy.stats import rankdata

from commute import figures as F

ROOT = Path(__file__).resolve().parents[2]
X = ROOT / "experiments"
HERE = Path(__file__).resolve().parent
BUILT = ("x2", "x10", "moved", "rebuilt", "new_few", "new_thin")
PAIRS = ("e62_copy_lookup", "e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3",
         "open_pair_c3_copy", "e58_rank1_96", "e58_copy_prior_cipher")  # fmt: skip
KEY = ("copy", "rank1", "elicit", "cipher", "prior_cipher")
SETS = ("built", "pairs", "key")
PROXIES = {  # name: (label, built from the VPD decomposition, as R is)
    "r": ("re-gating share, log-odds form", True),
    "prior": ("original model's prior on the answer", False),
    "S": ("P1.45's S, re-gated VPD pieces", True),
    "size": ("update size, smaller is reuse", False),
    "erank": ("update's effective rank", False),
    "steering": ("steering share", False),
    "crosscoder": ("crosscoder, Minder et al.", False),
}
Series = dict[str, float | None]


def load(rel: str) -> Any:
    p = X / rel
    return json.loads(p.read_text()) if p.exists() else None


def cells(key: str, value: str, rows: list[dict[str, Any]], **match: str) -> dict[str, float]:
    return {r[key]: r[value] for r in rows if all(r.get(k) == v for k, v in match.items())}


# ---- R --------------------------------------------------------------------------------------------------------------


def r_size_removed() -> dict[str, Series]:
    p142 = load("r_reanalysis/results/summary.json")
    a, b = p142["fits"]["R"]["a_b"]
    tab = load("key_grid/results/summary.json")["table"]
    key: Series = {t: fmean(-(math.log(1 - r) - (a + b * math.log(s))) for r, s in zip(tab["R_vpd"][t]["seeds"], tab["size"][t]["seeds"], strict=True))
           for t in KEY}  # fmt: skip
    return {"built": p142["known"]["R_size_removed"]["values"], "pairs": p142["pairs"]["R_size_removed"]["gaps"], "key": key}


def r_normalised() -> dict[str, Series]:
    p143 = load("kstar_vpd_norm/results/summary.json")
    return {"built": p143["known"]["N"], "pairs": p143["pairs_N"]["gaps"], "key": {}}


def r_source(arg: str) -> tuple[str, dict[str, Series]]:
    if arg == "size_removed":
        return arg, r_size_removed()
    if arg == "normalised":
        return arg, r_normalised()
    d = json.loads(Path(arg).read_text())
    return d["name"], {s: d.get(s, {}) for s in SETS}


# ---- proxies, each oriented higher = reuse ---------------------------------------------------------------------------


def proxies() -> dict[str, dict[str, Series]]:
    p142 = load("r_reanalysis/results/summary.json")
    pairs_json = load("kstar_pairs/results/pairs.json")["pairs"]
    final = load("method_table/results/final.json")["cells"]
    e187 = [c for c in load("size_matched/results/summary.json")["cells"] if c["model"] == "67M"]
    kg = load("key_grid/results/summary.json")
    cc = load("crosscoder/results/run2/summary.json")["cells"]
    names = {c["pair_name"]: c["pair"] for c in final if c["method"] == "share"}
    gating = load("gating_vpd/results/summary.json")  # P1.45; None until it lands

    def per_pair(d: dict[str, float]) -> Series:
        return {f"e187_{p}": d.get(p) for p in PAIRS}

    def key_mean(reading: str) -> Series:
        return {t: kg["table"][reading][t]["mean"] for t in KEY}

    out: dict[str, dict[str, Series]] = {
        "r": {"built": load("random_kl_share/results/known_answer/score.json")["rules"]["r"]["values"],
              "pairs": per_pair(cells("pair", "matched_gap", e187, reading="share")), "key": key_mean("r")},
        "prior": {"built": {}, "pairs": per_pair(cells("pair", "gap", final, method="prior")), "key": kg["log10_prior"]},
        "S": {s: (gating or {}).get("proxy_view", {}).get(s, {}) for s in SETS},
        "size": {"built": {i: -v for i, v in p142["sizes_known"].items()},
                 "pairs": {p: pairs_json[p]["smaller_update"] for p in per_pair({})}, "key": {}},
        "erank": {"built": load("known_answer_baselines/results/score.json")["methods"]["erank"]["P1b"]["values"],
                  "pairs": per_pair(cells("pair", "gap", final, method="erank")), "key": key_mean("erank")},
        "steering": {"built": {}, "pairs": per_pair(cells("pair", "matched_gap", e187, reading="steering")), "key": {}},
        "crosscoder": {"built": {}, "key": {},
                       "pairs": per_pair({names[c["row"]]: c["gap"] for c in cc if c["column"] == "crosscoder_matched"})},
    }  # fmt: skip
    return out


# ---- agreement -------------------------------------------------------------------------------------------------------


def common(a: Series, b: Series) -> list[str]:
    return [i for i in a if i in b and a[i] is not None and b[i] is not None]


def spearman(a: Series, b: Series) -> dict[str, Any]:
    ids = common(a, b)
    if len(ids) < 3:
        return {"rho": None, "n": len(ids), "items": ids}
    ra, rb = (rankdata([float(f"{d[i]:.6g}") for i in ids]) for d in (a, b))
    rho = None if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])
    return {"rho": rho, "n": len(ids), "items": ids}


def inside(values: list[float], lo: float, hi: float) -> str:
    return f"{sum(lo <= v <= hi for v in values)} of {len(values)}"


def random_column() -> dict[str, str]:
    """Matched random change against the fine-tunes' range: E237's 15 random changes at the 15 scaled states' sizes;
    the crosscoder from E235's Gaussian at site B on its one pair; r has no random reading (E228)."""
    tab = load("key_grid/results/summary.json")["table"]

    def kg(reading: str) -> str:
        ft = [v for t in KEY for v in tab[reading][t]["seeds"]]
        rnd = [v for t in KEY for v in tab[reading][t]["random"]]
        return inside(rnd, min(ft), max(ft)) + f" inside; random {min(rnd):.4g} to {max(rnd):.4g}, fine-tunes {min(ft):.4g} to {max(ft):.4g}"

    b = load("crosscoder_proper/results/run1/stage1.json")["sites"]["B"]
    ft = [*b["pair"]["reuse"], *b["pair"]["new"]]
    return {"r": "not defined: no random change gains 1 log-odds on the answer (E228)", "prior": "does not apply: a property of the task",
            "S": "pending", "size": "matched by construction", "erank": kg("erank"), "steering": "not read",
            "crosscoder": inside(b["gauss_specific"], min(ft), max(ft)) + " inside, one pair, last block (E235)",
            "R": kg("R_vpd") + " (key tasks: R at VPD as filed, equal sizes, so size removal keeps the order)"}  # fmt: skip


def table(r: dict[str, Series], px: dict[str, dict[str, Series]]) -> dict[str, Any]:
    size, prior = px["size"], px["prior"]
    rnd = random_column()
    rows: dict[str, Any] = {}
    for name, sets in [*px.items(), ("R", r)]:
        row: dict[str, Any] = {"label": PROXIES.get(name, ("the reuse score R", True))[0],
                               "circular": PROXIES.get(name, ("", True))[1], "random_inside_finetune_range": rnd[name]}  # fmt: skip
        if name != "R":
            row["vs_R"] = {s: spearman(sets.get(s, {}), r.get(s, {})) for s in SETS}
        row["vs_size"] = {s: spearman(sets.get(s, {}), size[s]) for s in ("built", "pairs")} if name != "size" else None
        row["vs_prior"] = {s: spearman(sets.get(s, {}), prior[s]) for s in ("pairs", "key")} if name != "prior" else None
        row["values"] = sets
        rows[name] = row
    return rows


# ---- figure ------------------------------------------------------------------------------------------------------


def scaled_ranks(d: Series, ids: list[str]) -> list[float]:
    rk = rankdata([float(f"{d[i]:.6g}") for i in ids])
    return [float((x - 1) / (len(ids) - 1)) for x in rk]


def plot(name: str, rows: dict[str, Any], r: dict[str, Series]) -> Path:
    fig, axes = F.figure(F.FULL_WIDTH, ncols=4, nrows=2)
    look = {"built": ("six built edits", "#0072B2", "o"), "pairs": ("nine built pairs, gap", "#D55E00", "s"),
            "key": ("five key tasks, equal size", "#009E73", "^")}  # fmt: skip
    data: list[dict[str, Any]] = []
    for ax, p in zip(axes.flat, [*PROXIES, None], strict=True):
        if p is None:
            ax.axis("off")
            continue
        drawn = False
        for s, (lab, col, mk) in look.items():
            ids = common(rows[p]["values"].get(s, {}), r.get(s, {}))
            if len(ids) < 3:
                continue
            xs, ys = scaled_ranks(r[s], ids), scaled_ranks(rows[p]["values"][s], ids)
            ax.scatter(xs, ys, color=col, marker=mk, s=22, label=lab, alpha=0.85)
            data += [{"proxy": p, "set": s, "item": i, "R": r[s][i], "proxy_value": rows[p]["values"][s][i], "R_rank": x, "proxy_rank": y}
                     for i, x, y in zip(ids, xs, ys, strict=True)]  # fmt: skip
            drawn = True
        rhos = [f"{s} {rows[p]['vs_R'][s]['rho']:+.2f}" for s in SETS if rows[p]["vs_R"][s]["rho"] is not None]
        ax.set_title(PROXIES[p][0] + ("\n(built from VPD, circular)" if PROXIES[p][1] else "") + ("\n" + ", ".join(rhos) if rhos else ""), fontsize=7)
        if not drawn:
            ax.text(0.5, 0.5, "pending: P1.45 not yet read" if p == "S" else "not read on 3 or more items", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.8, label="ranks agree")
        ax.set_xlabel("R rank in its item set", fontsize=7)
        ax.set_ylabel("proxy rank in its item set", fontsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=7, frameon=False)
    words = {"size_removed": "R with update size removed", "normalised": "size-normalised R"}.get(name, name)
    fig.suptitle(f"Which cheap proxies rank items the way {words} does? Ranks within each item set, 0 lowest", fontsize=9)
    return F.save(fig, HERE / "figures" / f"proxies_vs_r_{name}.png", data=data,
                  n="six built edits (one fine-tune each), nine built pairs (seed means per side), five key tasks (seeds 0-2); "
                    "titles give the Spearman rank correlation per item set, a description at these n")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", default="size_removed", help="size_removed, normalised, or a JSON file (see the docstring)")
    name, r = r_source(ap.parse_args().r)
    rows = table(r, proxies())
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / f"table_{name}.json").write_text(json.dumps({"cell": "P2.51", "R": name, "rows": rows}, indent=1))
    for p, row in rows.items():
        vs_r = {s: (v["rho"] if v["rho"] is None else round(v["rho"], 2), v["n"]) for s, v in row.get("vs_R", {}).items()}
        print(p, "vs R", vs_r, "| size", row["vs_size"] and {s: v["rho"] and round(v["rho"], 2) for s, v in row["vs_size"].items()},
              "| prior", row["vs_prior"] and {s: v["rho"] and round(v["rho"], 2) for s, v in row["vs_prior"].items()}, "|", row["random_inside_finetune_range"])  # fmt: skip
    print(plot(name, rows, r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
