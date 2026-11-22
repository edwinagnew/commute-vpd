"""E248 (cell P1.48) figure and side reading, from the saved item JSONs only (no model, no GPU): per item, the share of
its circuit K* the update left untouched (1 - changed members of K* / |K*|, changed = standable + grafts + gain <= 0)
and R under wording A among K*'s changed members only, 1 - (kept F's + cannot stand in) / changed. Untouched members
stand in exactly, so R_A = untouched share + (1 - untouched share) * the changed-only reading. Writes
results/changed_only.json and figures/changed_only_pairs.png (E187's nine pairs, with E242's thin new and rebuilt reuse
edits as the case where the two readings differ).

    ~/commute-agents/fleet/cpu-run python3 experiments/kstar_vpd_wording_a/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

RUN: Any = load_experiment("kstar_vpd_wording_a", "run")
HERE = Path(__file__).resolve().parent
COLORS = {"reuse": "#0072B2", "new": "#D55E00"}
PAIR_NAMES = {"e187_e62_copy_lookup": "copy / lookup", "e187_e46_copy_random": "identity / random map",
              "e187_e53_lowercase_cipher": "lowercase / cipher", "e187_e54_matched": "elicit / cipher",
              "e187_e54_own": "elicit / own cipher", "e187_open_pair_c3": "rank-1 / cipher (c3)",
              "e187_open_pair_c3_copy": "copy / cipher (c3)", "e187_e58_rank1_96": "rank-1 / cipher (96 keys)",
              "e187_e58_copy_prior_cipher": "copy / prior cipher"}  # fmt: skip


def item(name: str) -> dict[str, Any]:
    d = json.loads((HERE / "results" / "items" / f"{RUN.fname(name)}.json").read_text())
    s = d["searches"]["main"]
    changed = s["standable_changed_in_K"] + s["grafts_in_K"] + s["nonpositive_gain_in_K"]
    not_m = s["kept_F"] + s["grafts_in_K"] + s["nonpositive_gain_in_K"]  # members that keep F's mechanism
    return {"item": name, "size_K": s["size_K"], "changed_in_K": changed, "untouched": 1 - changed / s["size_K"],
            "R_A": s["R_A"], "R_A_changed": 1 - not_m / changed, "capped": s["capped"]}  # fmt: skip


def side(names: list[str]) -> dict[str, Any]:
    rows = [item(n) for n in names]
    return {"R_A": fmean(r["R_A"] for r in rows), "R_A_changed": fmean(r["R_A_changed"] for r in rows),
            "untouched": fmean(r["untouched"] for r in rows), "untouched_max": max(r["untouched"] for r in rows),
            "capped": sum(r["capped"] for r in rows), "n": len(rows), "items": rows}  # fmt: skip


def readings() -> dict[str, Any]:
    cfg, _ = RUN.load_config()
    pairs: dict[str, dict[str, Any]] = {p: {"reuse": side(r), "new": side(n)} for p, (r, n) in RUN.pair_sides().items()}
    for v in pairs.values():
        v["gap_R_A"] = v["reuse"]["R_A"] - v["new"]["R_A"]
        v["gap_R_A_changed"] = v["reuse"]["R_A_changed"] - v["new"]["R_A_changed"]
    keys = RUN.key_items()
    tasks = sorted({i.split("/")[1].removesuffix("_scaled").rsplit("_s", 1)[0] for i in keys})  # score.py's split kept the seed
    every = [item(i) for p in RUN.pair_sides().values() for s in p for i in s] + [item(i) for i in (*keys, *cfg.e62)]
    return {"experiment": "E248", "pairs": pairs,
            "pairs_reuse_higher_R_A": sum(v["gap_R_A"] > 0 for v in pairs.values()),
            "pairs_reuse_higher_R_A_changed": sum(v["gap_R_A_changed"] > 0 for v in pairs.values()),
            "e62_own_size": {"reuse": side([i for i in cfg.e62 if "/copy" in i]), "new": side([i for i in cfg.e62 if "/lookup" in i])},
            "keys": {t: side([i for i in keys if f"/{t}_s" in i]) for t in tasks},
            "trained_untouched_max": max(r["untouched"] for r in every), "trained_items": len({r["item"] for r in every}),
            "built": {b: item(b) for b in ("new_thin", "rebuilt")}}  # fmt: skip


def figure(out: dict[str, Any]) -> None:
    fig, axs = FG.figure(FG.FULL_WIDTH, ncols=2)
    cols = [(PAIR_NAMES[p], v["reuse"], v["new"]) for p, v in out["pairs"].items()]
    b = out["built"]
    data = []
    for ax, key, title in zip(axs, ("R_A", "R_A_changed"), ("all of K*", "K*'s changed variables only"), strict=True):
        for x, (name, r, n) in enumerate(cols):
            for s, kind in ((r, "reuse"), (n, "new")):
                ax.scatter([x], [s[key]], color=COLORS[kind], s=24, label=f"{kind} side, mean over seeds" if x == 0 else None)
                data.append({"panel": key, "pair": name, "side": kind, "value": s[key], "untouched": s["untouched"]})
        x = len(cols) + 0.5
        for nm, kind, mk in (("rebuilt", "reuse", "D"), ("new_thin", "new", "D")):
            ax.scatter([x], [b[nm][key]], color=COLORS[kind], marker=mk, s=28, label="built rebuilt reuse edit (E242)" if nm == "rebuilt" else "built thin new edit (E242)")
            data.append({"panel": key, "pair": "built edits", "side": nm, "value": b[nm][key], "untouched": b[nm]["untouched"]})
        ax.axhline(0.29, color="#555555", linewidth=0.8, linestyle=":", label="random update, matched size (E242)")
        ax.set_xticks([*range(len(cols)), len(cols) + 0.5], [c[0] for c in cols] + ["built edits"], rotation=60, ha="right", fontsize=6)
        ax.set_ylim(0.2, 1.02)
        ax.set_ylabel("R under wording A")
        ax.set_title(title, fontsize=8)
    axs[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Reuse sides read higher in all nine pairs, with or without the untouched variables", fontsize=9)
    n = (f"9 pairs, 2 to 8 seeds per side, larger side scaled to the smaller's size; every trained K* has at most "
         f"{out['trained_untouched_max']:.4f} of its variables untouched, so the panels agree except on the built edits")
    FG.save(fig, HERE / "figures" / "changed_only_pairs.png", n=n, data=data)


def main() -> None:
    out = readings()
    (HERE / "results" / "changed_only.json").write_text(json.dumps(out, indent=1))
    figure(out)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"} | {"gaps": {p: (v["gap_R_A"], v["gap_R_A_changed"]) for p, v in out["pairs"].items()}}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
