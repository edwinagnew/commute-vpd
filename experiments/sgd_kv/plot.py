"""Merge sgd_kv's shard files and draw the one figure Q1 reads: per task, each state's adversarial Pile ratio against
the pass line and that shard's inconclusive band, with E10's recorded Adam 1e-5 ratio beside it (left), and its held-out
KL elsewhere against the rate rule's limit (right). Prints Q1, P2 and P3. Writes results/sgd_kv.json (merged) and
figures/kv.png.

Usage (repo root): uv run python experiments/sgd_kv/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LINE = 1.5
ADAM_REF = "adam_1e-05"
COLOURS = {ADAM_REF: "C1", "adam_3.33e-06": "C3", "adam_1e-06": "C4"}  # SGD: C0


def merge() -> dict[str, Any]:
    shards = sorted((HERE / "results").glob("sgd_kv_shard*.json"))
    tasks: dict[str, Any] = {}
    floors: dict[str, float] = {}
    for p in shards:
        s = json.loads(p.read_text())
        for t, rec in s["tasks"].items():
            tasks[t], floors[t] = rec, s["W0"]["pgd_floor"]
    sc = {t: r["score"] for t, r in tasks.items() if "score" in r}
    q1 = sc.get("rand32")
    decision = ("not scored" if q1 is None else "yes: E10's rand32 failure was an Adam/rate artefact" if q1["any_new_arm_passes"]
                else "rerun the inconclusive states first" if q1["any_new_arm_inconclusive"]
                else "no: the failure holds across optimisers and rates")
    return {"shards": [p.name for p in shards], "tasks": tasks, "pgd_floor": floors,
            "Q1": {"rand32": q1 and q1["verdicts"], "decision": decision},
            "P2": {t: s["P2_sgd_learned"] for t, s in sc.items()}, "P3": {t: s["P3_sgd_less_damage"] for t, s in sc.items()}}


def main() -> None:
    m = merge()
    (HERE / "results" / "sgd_kv.json").write_text(json.dumps(m, indent=1, sort_keys=True, default=float))
    print(json.dumps({k: m[k] for k in ("Q1", "P2", "P3")}, indent=1))
    names = list(m["tasks"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4))
    for i, t in enumerate(names):
        rec, f = m["tasks"][t], m["pgd_floor"][t]
        ax.fill_between([i - 0.4, i + 0.4], LINE / (1 + f), LINE * (1 + f), color="0.9", lw=0)
        ax.scatter(i - 0.35, rec["e10"][ADAM_REF]["pile"]["pgd_kl"], facecolors="none", edgecolors="C1",
                   label="Adam 1e-5, E10's reading" if i == 0 else None, zorder=3)
        arms = rec["arms"]
        for j, (name, a) in enumerate(arms.items()):
            x = i - 0.2 + 0.5 * j / max(len(arms) - 1, 1)
            c = COLOURS.get(name, "C0")
            ax.scatter(x, a["pile"]["pgd_kl"], color=c, label=name if i == 0 else None, zorder=3)
            bx.scatter(x, a["heldout"]["kl_other"], color=c, zorder=3)
        bx.hlines(rec["limit"], i - 0.4, i + 0.4, color="k", lw=0.8)
    ax.axhline(LINE, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("adversarial Pile KL / W0's (grey: inconclusive band)")
    bx.set_ylabel("held-out KL elsewhere, nats (line: rate rule's limit)")
    for a in (ax, bx):
        a.set_xticks(range(len(names)), names)
    ax.legend(fontsize=7)
    fig.suptitle("Key-to-value tasks: SGD (selected rate) and Adam at three rates, each at its first learned checkpoint")
    fig.tight_layout()
    out = HERE / "figures" / "kv.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
