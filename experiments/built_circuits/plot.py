"""E301's figure, its rows for figures/data/definition.csv and the filed description P1 (no GPU).

    ~/commute-agents/fleet/cpu-run python experiments/built_circuits/plot.py

Reads results/items/<feat>/<circuit>__<state>.json (run.py read). Writes figures/built_circuits.png (per circuit and
featurization, each edit's E_b over K beside its own matched random), results/definition_rows.csv (feat, block, item,
side, value, field, source; fields E_b_K, E_b_K_no_leftover, leftover_share) and results/summary.json (delta_f, P1 per
circuit and featurization, P1 on E298's Fig 2 rows where they exist, P2).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E298 = HERE.parent / "fig2_standard" / "results" / "items"
FEATS = {"vpd": "VPD", "neurons": "top", "sae": "SAE"}  # definition.csv's feat names
CIRCUITS = ("induction", "bigram")
REUSE, NEW = ("x2", "x10", "rebuilt"), ("new_few", "new_thin")
EDITS = (*REUSE, *NEW)
FIELDS = ("E_b_K", "E_b_K_no_leftover", "leftover_share")


def item(root: Path, feat: str, name: str) -> dict[str, Any] | None:
    p = root / feat / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def side(state: str) -> str:
    return "random" if state.startswith("rand_") else "reuse" if state in REUSE else "new" if state in NEW else "control"


def p1(vals: dict[str, dict[str, Any] | None], delta: float | None, rand: str) -> str:
    """Filed P1: min E_b_K over the reuse edits above max over the new edits and randoms by more than delta_f; a gap at
    most delta_f is 'does not hold'; any of the ten states unread or with K and changed empty is 'no call'."""
    names = [*EDITS, *(rand.format(e) for e in EDITS)]
    got = [vals.get(n) for n in names]
    if delta is None or any(g is None or g["n_changed_K"] == 0 for g in got):
        return "no call"
    eb = {n: g["E_b_K"] for n, g in zip(names, got, strict=True) if g is not None}
    gap = min(eb[e] for e in REUSE) - max(v for n, v in eb.items() if n not in REUSE)
    return "holds" if gap > delta else "does not hold"


def floor(feat: str) -> float | None:
    a, b = (item(RESULTS / "items", feat, f"induction__{s}") for s in ("x2", "x2_s269"))
    return abs(a["E_b_K"] - b["E_b_K"]) if a and b else None


def collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = [], {"delta_f": {f: floor(f) for f in FEATS}, "P1": {}, "P1_fig2": {}, "P2": {}}
    for f, name in FEATS.items():
        d = summary["delta_f"][f]
        for c in CIRCUITS:
            vals = {s: item(RESULTS / "items", f, f"{c}__{s}") for s in ("do_nothing", *EDITS, *(f"rand_{e}" for e in EDITS))}
            rows += [{"feat": name, "block": f"built_{c}", "item": s, "side": side(s), "value": v[k], "field": k, "source": "E301"}
                     for s, v in vals.items() if v is not None for k in FIELDS]  # fmt: skip
            summary["P1"][f"{c}/{f}"] = p1(vals, d, "rand_{}")
            summary["P2"][f"{c}/{f}"] = {e: vals[e]["E_b_K"] - vals[f"rand_{e}"]["E_b_K"] for e in EDITS if vals[e] and vals[f"rand_{e}"]}  # type: ignore[index]
        fig2 = {s: item(E298, f, f"built__{s}") for s in (*EDITS, *(f"rand_{e}" for e in EDITS))}
        summary["P1_fig2"][f] = p1(fig2, d, "rand_{}")
    return rows, summary


def draw(rows: list[dict[str, Any]]) -> None:
    v = {(r["feat"], r["block"], r["item"]): r["value"] for r in rows if r["field"] == "E_b_K"}
    fig, axes = FG.figure((10.0, 6.0), ncols=3, nrows=2)
    for i, c in enumerate(CIRCUITS):
        for j, name in enumerate(FEATS.values()):
            ax = axes[i][j]
            for state, method, off in (("{}", "eb", -0.12), ("rand_{}", "random", 0.12)):
                pts = [(k + off, v[(name, f"built_{c}", state.format(e))]) for k, e in enumerate(EDITS) if (name, f"built_{c}", state.format(e)) in v]
                st = FG.METHODS[method]
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=st.color, marker=st.marker, s=22, label=st.label)
            ax.set_xticks(range(len(EDITS)), list(EDITS), fontsize=7)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("write share over the needed set")
            ax.set_title(f"{c}, {name}", fontsize=8)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Built edits on two more circuits: each edit's write share beside its matched random", fontsize=9)
    FG.save(fig, HERE / "figures" / "built_circuits.png", n="one state per point; 256 pairs per needed set; one random per edit",
            data=[r for r in rows if r["field"] == "E_b_K"])  # fmt: skip


def main() -> None:
    rows, summary = collect()
    with (RESULTS / "definition_rows.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, ["feat", "block", "item", "side", "value", "field", "source"])
        wr.writeheader()
        wr.writerows(rows)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if rows:
        draw(rows)


if __name__ == "__main__":
    main()
