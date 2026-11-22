"""P2.59 (E265): the ADL column beside the other baselines on the nine built pairs, and against definition (b)'s E_b.

Reads results/amend1/column.json (ADL), method_table/results/final.json (share, neuron, steering), crosscoder/results/run2
summary.json (crosscoder, as the paper scorecard reads it) and rb_states/results/summary_items.json (E_b, E257). Writes
results/amend1/compare.json and figures/adl_vs_eb.png.

    ~/commute-agents/fleet/cpu-run python3 experiments/adl/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as F
from commute.jobs import EXPERIMENTS, load_experiment

HERE = Path(__file__).resolve().parent
AN = load_experiment("proxies_vs_r", "analyse")  # P2.51's spearman
FINAL = EXPERIMENTS / "method_table" / "results" / "final.json"
XC = EXPERIMENTS / "crosscoder" / "results" / "run2" / "summary.json"
E257 = EXPERIMENTS / "rb_states" / "results" / "summary_items.json"


def read(p: Path) -> Any:
    return json.loads(p.read_text())


def main() -> None:
    adl = {c["pair"]: c for c in read(HERE / "results" / "amend1" / "column.json")["cells"] if c["group"] == "built"}
    final = [c for c in read(FINAL)["cells"] if c["group"] == "built"]
    names = {c["pair"]: c["pair_name"] for c in final}
    xc = {c["row"]: c["gap"] for c in read(XC)["cells"] if c["column"] == "crosscoder"}
    eb = {k.removeprefix("e187_"): v["gap_E_b"] for k, v in read(E257)["pairs"].items()}
    rows: dict[str, dict[str, Any]] = {}
    for p, a in adl.items():
        r = {m: c["gap"] for c in final if c["pair"] == p for m in [c["method"]] if m in ("share", "neuron", "steering")}
        rows[p] = r | {"crosscoder": xc[names[p]], "adl": a["gap"], "adl_call": a["call"], "E_b": eb[p], "name": names[p]}
    sign = lambda m: sum((rows[p][m] > 0) == (rows[p]["E_b"] > 0) for p in rows)  # noqa: E731
    out = {"pairs": rows, "n": len(rows),
           "same_sign_as_E_b": {m: sign(m) for m in ("share", "neuron", "steering", "crosscoder", "adl")},
           "adl_new_higher": sum(r["adl"] < 0 for r in rows.values()),
           "spearman_gap_vs_E_b": {m: AN.spearman({p: rows[p][m] for p in rows}, {p: rows[p]["E_b"] for p in rows})["rho"]
                                   for m in ("share", "neuron", "steering", "crosscoder", "adl")}}  # fmt: skip
    (HERE / "results" / "amend1" / "compare.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    fig, ax = F.figure((4.6, 3.6))
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    for p, r in rows.items():
        ax.scatter(r["adl"], r["E_b"], color="#0072B2", s=26)
        ax.annotate(p, (r["adl"], r["E_b"]), xytext=(3, 2), textcoords="offset points", fontsize=6, color="#333333")
    ax.set_xlabel("activation difference lens gap, reuse minus new")
    ax.set_ylabel("definition (b) write share gap, reuse minus new")
    ax.set_title("Does the lens's pair gap follow the definition's?", fontsize=9)
    F.save(fig, HERE / "figures" / "adl_vs_eb.png", n="9 built pairs, 2-8 seeds per side",
           data=[{"pair": p, "adl_gap": r["adl"], "E_b_gap": r["E_b"]} for p, r in rows.items()])


if __name__ == "__main__":
    main()
