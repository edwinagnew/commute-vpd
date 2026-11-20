"""E221's per-pair table and figure: each pair's R gap (reuse minus new, main grade) against its smaller-update gap, one
panel per level, with the do-nothing reference at (0, 0). Writes results/pairs.json (every per-pair and per-item number
the report cites, from the item files and score.py's old readings) and figures/r_gap_vs_size.png. No GPU:

    python experiments/kstar_pairs/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as fg
from commute.jobs import load_experiment

S: Any = load_experiment("kstar_pairs", "score")
HERE = Path(__file__).resolve().parent
LEVELS = {"head_mlp": "heads and MLP layers", "mlp_attn": "MLP and attention layers", "block": "blocks"}


def item_rows(d: Any, items: list[str]) -> dict[str, dict[str, Any]]:
    """Per item and level: R under both grades, f, |K*| (means over orders), the first order's K*, and the update size."""
    out: dict[str, dict[str, Any]] = {}
    for i in items:
        row: dict[str, Any] = {"size": d[(i, "block")]["size"]["whole"]}
        for L in LEVELS:
            rec = d[(i, L)]
            row[L] = {k: S.item_value(rec, k) for k in ("R_read", "R_all", "f", "size", "remainder_share")}
            row[L]["K_order0"] = [u["name"] for u in rec["searches"]["main"][0]["K"]]
            row[L]["polish_capped"] = sum(bool(r.get("polish_capped")) for r in rec["searches"]["main"])
        out[i] = row
    return out


def pair_rows(summary: dict[str, Any], old: dict[str, dict[str, Any]], pairs: dict[str, Any], d: Any) -> dict[str, dict[str, Any]]:
    """Per pair: its items, the smaller-update gap, and per level each side's mean R, R's gaps, se, f gap and every old
    reading's gap."""
    out: dict[str, dict[str, Any]] = {}
    for p, (r, n) in pairs.items():
        size = summary["size"]["whole"][p]
        row: dict[str, Any] = {"reuse": r, "new": n, "smaller_update": -size, "old": {k: o["gaps"][p] for k, o in old.items()}}
        for L in LEVELS:
            lv, get = summary["levels"][L], S.at_level(d, L, "R_read")
            row[L] = {"R_reuse_read": S.mean([get(i) for i in r]), "R_new_read": S.mean([get(i) for i in n]), "gap_R_read": lv["read"]["gap_R"][p],
                      "gap_R_all": lv["all"]["gap_R"][p], "se_R_read": lv["read"]["se_R"][p], "gap_f": lv["f"][p]}  # fmt: skip
        out[p] = row
    return out


def variants(d: Any, pairs: dict[str, Any], rows: dict[str, dict[str, Any]], th: dict[str, Any]) -> dict[str, Any]:
    """At the two exhaustive levels, each saved variant's R gaps (read grade): how many pairs keep the main gap's sign,
    and the Spearman of the variant's gaps with the smaller update."""
    out: dict[str, Any] = {}
    for L in ("mlp_attn", "block"):
        for v in ("alpha0.8", "alpha0.95", "restricted", "half1", "half2"):
            get = lambda i, L=L, v=v: d[(i, L)]["searches"][v][0]["R_read"]
            g = {p: S.gap(get, r, n) for p, (r, n) in pairs.items()}
            same = sum(S.sign(g[p]) == S.sign(rows[p][L]["gap_R_read"]) for p in pairs)
            rho = S.spearman([g[p] for p in pairs], [rows[p]["smaller_update"] for p in pairs], th["n_min"])
            out[f"{L}_{v}"] = {"gaps": g, "sign_same_as_main": same, "rho_smaller_update": rho}
    return out


def figure(rows: dict[str, dict[str, Any]], summary: dict[str, Any]) -> None:
    fig, axes = fg.figure(size=(7.5, 3.4), ncols=3)
    data = []
    for ax, (L, title) in zip(axes, LEVELS.items(), strict=True):
        for src, colour, label in (("e196", "#0072B2", "five task and twin pairs, 2 seeds a side"), ("e187", "#D55E00", "nine scorecard pairs, 2-8 seeds a side")):
            P = [p for p in rows if p.startswith(src)]
            x, y = [rows[p]["smaller_update"] * 1e3 for p in P], [rows[p][L]["gap_R_read"] * 1e3 for p in P]
            ax.scatter(x, y, color=colour, s=20, alpha=0.85, linewidths=0, label=label)
            data += [{"level": L, "pair": p, "smaller_update_x1e3": a, "gap_R_read_x1e3": b} for p, a, b in zip(P, x, y, strict=True)]
        ax.scatter([0], [0], color="#000000", marker="X", s=40, label="do-nothing, R = 1 on both sides")
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.axvline(0, color="#999999", linewidth=0.6)
        rho = summary["levels"][L]["read"]["smaller_update"]["rho"]
        ax.set_title(f"{title}\nrank correlation {rho:.2f}", fontsize=8)
        ax.set_xlabel("smaller update on the reuse side,\nsize gap x 1000", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("R gap, reuse minus new, x 1000", fontsize=7)
    fig.suptitle("A smaller update on the reuse side goes with a higher R gap at every level, but the ranks agree below 0.7", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fg.save(fig, HERE / "figures" / "r_gap_vs_size.png", data=data,
            n="one point per pair, 14 pairs; a gap is the mean over a side's seeds; R graded at the last position of 512 windows per fine-tune")  # fmt: skip


def main() -> None:
    cfg, th = S.RUN.load_config()
    pairs = S.RUN.pairs(cfg)
    summary = json.loads((S.RESULTS / "summary.json").read_text())
    d = S.load_items()
    rows = pair_rows(summary, S.old_readings(cfg, pairs), pairs, d)
    out = {"pairs": rows, "variants": variants(d, pairs, rows, th), "items": item_rows(d, S.RUN.items(cfg))}
    (S.RESULTS / "pairs.json").write_text(json.dumps(out, indent=1))
    figure(rows, summary)


if __name__ == "__main__":
    main()
