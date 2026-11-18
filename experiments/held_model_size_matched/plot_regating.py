"""P2.35, zero-run: how far compare-lead's re-gating share agrees with the held-model reading S on E187's nine pairs.

Inputs, read only:
    results/summary.json (this folder, E205)    S per state, reading 3 (per-site map) and reading 1 (one gain per unit),
                                                 saved and matched sizes; gap = mean S of the reuse side minus the new side
    ~/commute-agents/compare-lead/experiments/size_matched/results/summary.json (E187)
                                                 the share's saved gap and its per-state matched values
Outputs:
    figures/regating_vs_S.png   two panels with shared y: x = S gap per pair, y = share gap per pair; open = saved sizes,
                                filled = larger side shrunk to match. 0 lines on both axes = no difference between sides.
    results/regating_vs_S.json  per pair the four gaps, sign agreement counts and Spearman rank correlations

    ~/commute-agents/fleet/cpu-run python experiments/held_model_size_matched/plot_regating.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from commute.figures import FULL_WIDTH, METHODS, figure, save

HERE = Path(__file__).resolve().parent
SHARE = Path.home() / "commute-agents/compare-lead/experiments/size_matched/results/summary.json"
READINGS = {"3": "per-site linear map (main)", "1": "one gain per unit (side)"}
SIZES = {"saved": "saved sizes", "matched": "larger side shrunk to match"}
NAMES = {"e62_copy_lookup": "copy at a new marker vs lookup", "e58_copy_prior_cipher": "copy vs prior-matched cipher"}
N_TEXT = "9 pairs (E187's built pairs), 2 to 8 seeds per side; S on 512 read windows per state, the share on E187's rows"


def share_gaps() -> dict[str, dict[str, float]]:
    cells = json.loads(SHARE.read_text())["cells"]
    out = {}
    for c in cells:
        if c["model"] == "67M" and c["reading"] == "share":
            m = c["matched_values"]
            matched = mean(m["reuse"]) - mean(m["new"])
            assert abs(matched - c["matched_gap"]) < 1e-12, c["pair"]
            out[c["pair"]] = {"saved": c["gap"], "matched": c["matched_gap"]}
    return out


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(order):
        r[i] = float(k)
    assert len(set(xs)) == len(xs), "ties: use average ranks"
    return r


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def table() -> dict[str, Any]:
    pairs = json.loads((HERE / "results/summary.json").read_text())["pairs"]
    share = share_gaps()
    assert set(pairs) == set(share), (sorted(pairs), sorted(share))
    rows = {}
    for pair, p in pairs.items():
        rows[pair] = {"c": p["c"], "larger": p["larger"], "share": share[pair]}
        for r in READINGS:
            g = p[f"reading {r}"]
            rows[pair][f"S{r}"] = {"saved": g["gap_original"], "matched": g["gap_matched"]}
    agree: dict[str, Any] = {}
    for r in READINGS:
        for size in SIZES:
            xs = [v[f"S{r}"][size] for v in rows.values()]
            ys = [v["share"][size] for v in rows.values()]
            agree[f"S{r}_{size}"] = {
                "sign_agree": sum((x > 0) == (y > 0) for x, y in zip(xs, ys, strict=True)),
                "spearman": spearman(xs, ys),
                "S_gap_min": min(xs), "S_gap_max": max(xs), "share_gap_min": min(ys), "share_gap_max": max(ys),
            }  # fmt: skip
    return {"cell": "P2.35", "n_pairs": len(rows), "pairs": rows, "agreement": agree}


def draw(t: dict[str, Any]) -> None:
    fig, axs = figure(FULL_WIDTH, ncols=2)
    color = METHODS["regating"].color
    data = []
    for k, (ax, (r, title)) in enumerate(zip(axs, READINGS.items(), strict=True)):
        for size, name in SIZES.items():
            xs = [v[f"S{r}"][size] for v in t["pairs"].values()]
            ys = [v["share"][size] for v in t["pairs"].values()]
            face = "white" if size == "saved" else color
            ax.scatter(xs, ys, s=26, marker="o", facecolors=face, edgecolors=color, linewidths=1.0, label=name)
            data += [{"reading": r, "size": size, "pair": p, "S_gap": x, "share_gap": y}
                     for p, x, y in zip(t["pairs"], xs, ys, strict=True)]  # fmt: skip
        for pair, v in t["pairs"].items():  # each pair's saved and matched points joined, so size's move is visible
            ax.plot([v[f"S{r}"]["saved"], v[f"S{r}"]["matched"]], [v["share"]["saved"], v["share"]["matched"]],
                    color=color, linewidth=0.5, alpha=0.5)  # fmt: skip
            if v[f"S{r}"]["saved"] < 0:  # the pair whose S gap disagrees in sign with its share gap
                ax.annotate(NAMES[pair], (v[f"S{r}"]["saved"], v["share"]["saved"]), xytext=(4, -12),
                            textcoords="offset points", fontsize=6, color="#333333")  # fmt: skip
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_ylim(-0.05, 0.8)
        a = t["agreement"][f"S{r}_saved"]
        ax.set_title(f"{title}: sign agrees on {a['sign_agree']} of 9 at saved sizes", fontsize=8)
        ax.set_xlabel("gap in held-model reading S, reuse side minus new side", fontsize=7)
        if k == 0:
            ax.set_ylabel("gap in re-gating share, reuse minus new", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Re-gating share gaps sit near 0.4 to 0.7 on every pair; the held-model gaps are all under 0.08", fontsize=9)
    save(fig, HERE / "figures/regating_vs_S.png", n=N_TEXT, data=data)


def main() -> None:
    t = table()
    (HERE / "results/regating_vs_S.json").write_text(json.dumps(t, indent=1) + "\n")
    draw(t)
    print(json.dumps(t["agreement"], indent=1))


if __name__ == "__main__":
    main()
