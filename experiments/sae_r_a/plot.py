"""E245 figure and key summary, from the saved item JSONs only (no model, no GPU).

Writes results/summary_keys.json (per key item: R_A at the three tolerances, the remainder share, |K*|, whether each
search hit its cap, how many variables keep F's mechanism and how many of those are site remainders; per task: mean and
range) with, under "built_descriptive", the share of K*'s changed members that can stand in, at SAE latents and at
E242's VPD reading of the same built items. Figure: figures/r_a_sae_vs_vpd.png.

    ~/commute-agents/fleet/cpu-run python3 experiments/sae_r_a/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ITEMS = HERE / "results" / "items"
LARGE = HERE / "results" / "large"  # the keep masks (gitignored); counts are left out if it is absent
VPD = HERE.parent / "kstar_vpd_wording_a" / "results" / "items"
WIDTH = 16_385  # variables per SAE site: 16,384 latents and the remainder (commute.sae_feat)
REUSE, NEW = ("x2", "x10", "rebuilt", "moved"), ("new_thin", "new_few")
TASKS = {"copy": "copy", "rank1": "top", "elicit": "mid", "prior_cipher": "prior-matched cipher", "cipher": "cipher"}
TOLS = {"t05": "tol0.05", "t1": "tol0.1", "t2": "tol0.2"}
SAE_C, VPD_C, RAND_C = "#0072B2", "#D55E00", "#555555"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def changed_only(s: dict[str, Any]) -> float | None:
    """1 - (members that keep F's mechanism + grafts + gains <= 0) / K*'s changed members: R_A without K*'s unchanged-site
    members, which stand in exactly. The decision card reports the same share at VPD."""
    if s.get("R_A") is None:
        return None
    bad = s["kept_F"] + s["grafts_in_K"] + s["nonpositive_gain_in_K"]
    return 1 - bad / (s["standable_changed_in_K"] + s["grafts_in_K"] + s["nonpositive_gain_in_K"])


def kept(name: str) -> dict[str, int] | None:
    npz = LARGE / f"items_{name}.npz"
    z = np.load(npz) if npz.exists() else None
    if z is None or "main_keeps_F_tol0.1" not in z.files:  # absent, or the item was not read
        return None
    idx = np.flatnonzero(z["main_keeps_F_tol0.1"])
    return {"kept_F_tol0.1": len(idx), "kept_remainders_tol0.1": int(np.sum(idx % WIDTH == WIDTH - 1))}


def key_record(name: str) -> dict[str, Any]:
    d = load(ITEMS / f"{name}.json")
    m = d["searches"]["main"]
    return {
        "R_A": {k: m[t]["R_A"] for k, t in TOLS.items()},
        "remainder_share": d["remainder_share"],
        "kl_fm": d["kl_fm"],
        "size_K": m["size_K"],
        "K_search_capped": m["capped_K"],
        "stand_in_capped": {k: m[t]["capped"] for k, t in TOLS.items()},
        **(kept(name) or {}),
    }


def span(xs: list[float]) -> dict[str, float]:
    return {"mean": fmean(xs), "min": min(xs), "max": max(xs)}


def built_record(name: str) -> dict[str, Any]:
    sae, vpd = load(ITEMS / f"{name}.json"), load(VPD / f"{name}.json")
    s, v = sae["searches"]["main"]["tol0.1"], vpd["searches"]["main"]  # a tolerance not read holds only R_A None
    return {
        "sae": {"R_A": s["R_A"], "floor_1_minus_f": s.get("floor_1_minus_f"), "R_A_changed_only": changed_only(s),
                "size_K": sae["searches"]["main"]["size_K"], "stand_in_capped": s.get("capped"), **(kept(name) or {})},
        "vpd": {"R_A": v.get("R_A"), "floor_1_minus_f": v.get("floor_1_minus_f"), "R_A_changed_only": changed_only(v)},
    }  # fmt: skip


def summary() -> dict[str, Any]:
    keys = {f"{t}_s{s}": key_record(f"e58-{t}_s{s}_scaled") for t in TASKS for s in range(3)}
    tasks = {t: {"R_A_t1": span([keys[f"{t}_s{s}"]["R_A"]["t1"] for s in range(3)]),
                 "remainder_share": span([keys[f"{t}_s{s}"]["remainder_share"] for s in range(3)])} for t in TASKS}  # fmt: skip
    built = {n: built_record(n) for n in (*REUSE, *NEW)}
    random = {n: load(VPD / f"{n}.json")["searches"]["main"]["R_A"] for n in ("random_size_241001", "random_kl_241002")}
    every = [k["R_A"]["t1"] for k in keys.values()]
    return {
        "experiment": "E245",
        "note": "written by experiments/sae_r_a/plot.py from results/items/*.json, the keep masks in results/large, and "
        "E242's items (experiments/kstar_vpd_wording_a/results/items). Key items: every K* search hit its 2000-evaluation "
        "cap; no stand-in search did. R_A_changed_only is descriptive, not a filed reading.",
        "keys": keys,
        "tasks": tasks,
        "keys_range": {"R_A_t1": span(every), "remainder_share": span([k["remainder_share"] for k in keys.values()])},
        "built_descriptive": built,
        "vpd_random": random,
    }


def plot(s: dict[str, Any]) -> None:
    """One panel, one axis: built items at SAE and VPD with each one's lowest possible value, VPD's random updates,
    then the key tasks at SAE (three seeds each)."""
    fig, ax = FG.figure((7.0, 6.2))
    rows = [*REUSE, *NEW]
    data: list[dict[str, Any]] = []
    for i, n in enumerate(rows):
        b = s["built_descriptive"][n]
        for side, c, mk, dy in (("sae", SAE_C, "o", -0.15), ("vpd", VPD_C, "s", 0.15)):
            r, fl = b[side]["R_A"], b[side]["floor_1_minus_f"]
            data.append({"group": "built", "item": n, "featurization": side, "R_A_tol0.1": r, "floor_1_minus_f": fl})
            if r is None:
                continue
            first = i == 0
            ax.scatter([r], [i + dy], color=c, marker=mk, s=30, zorder=3,
                       label=("SAE latents" if side == "sae" else "VPD subcomponents") if first else None)  # fmt: skip
            ax.scatter([fl], [i + dy], facecolors="none", edgecolors=c, marker=mk, s=30,
                       label=f"lowest possible at {side.upper()}: share of K* at untouched sites" if first else None)  # fmt: skip
        if b["sae"]["R_A"] is None:
            ax.annotate("not read: output change under the noise floor", (0.22, i), fontsize=7, va="center", color="#555555")
    at = len(rows)
    for j, r in enumerate(s["vpd_random"].values()):
        ax.scatter([r], [at + 0.15 * (2 * j - 1)], color=RAND_C, marker="X", s=30, label=None if j else "random update, VPD")
        data.append({"group": "built", "item": "random", "featurization": "vpd", "R_A_tol0.1": r})
    keys = {TASKS[t]: [s["keys"][f"{t}_s{k}"]["R_A"]["t1"] for k in range(3)] for t in TASKS}
    for i, (label, vals) in enumerate(keys.items(), start=at + 1):
        ax.scatter(vals, [i] * 3, color=SAE_C, marker="o", s=22, alpha=0.6, linewidths=0)
        ax.annotate(f"3 seeds, {min(vals):.4f} to {max(vals):.4f}", (0.22, i), fontsize=7, va="center", color="#555555")
        data += [{"group": "keys", "item": f"{label} s{k}", "featurization": "sae", "R_A_tol0.1": v} for k, v in enumerate(vals)]
    names = [f"{n}, built {'reuse' if n in REUSE else 'new'}" for n in rows] + ["random update"]
    names += [f"key task: {k}" for k in keys]
    for y in (len(REUSE) - 0.5, at - 0.5, at + 0.5):
        ax.axhline(y, color="#bbbbbb", linewidth=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("R under wording A, bar a tenth of KL(F || M)")
    ax.set_title("At SAE latents R puts the built reuse edits above the new ones;\n"
                 "at VPD it does not. Every key task reads near 1 at SAE latents", fontsize=9)  # fmt: skip
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.35, -0.1), ncol=2, frameon=False)
    FG.save(fig, HERE / "figures" / "r_a_sae_vs_vpd.png", data=data,
            n="Built: one fine-tune per item, 75 copy windows or 512 cue windows; keys: 3 seeds per task, 512 windows. "
            "Every K* search hit its cap; R is a lower bound where the stand-in search did too (x10, rebuilt, both new).")  # fmt: skip


def main() -> None:
    s = summary()
    (HERE / "results" / "summary_keys.json").write_text(json.dumps(s, indent=1) + "\n")
    plot(s)


if __name__ == "__main__":
    main()
