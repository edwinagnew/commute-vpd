"""P2.87: per reuse task, the fine-tunes' log-odds shares beside the random-subspace updates' at the first state that gains
1 log-odds ("at 1") and at the recipe's last step ("end"); and the filed reading (`rules`) -> results/summary.json. $0,
CPU, saved files only:

    ~/commute-agents/fleet/cpu-run python3 experiments/subspace_share/plot.py      # figures/shares.png, results/summary.json

Rows are Fig `known`'s 8 reuse tasks in known3.json's order. A seed that never reached gain 1 has no at-1 share; a task
none of whose seeds did says so with its largest gain. Values only: no verdict and no counts of comparisons.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import rules

from commute.figures import FULL_WIDTH, METHODS, dots, figure, save

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
KNOWN3 = HERE.parents[0] / "known_3seeds" / "results" / "known3.json"
KINDS = {"fine-tune": ("built_reuse", -0.25), "at1": ("random", 0.0), "end": ("size", 0.25)}
LABELS = {"fine-tune": "fine-tune (task built to reuse)", "at1": "random-subspace update, first state at gain 1",
          "end": "random-subspace update, step 300"}  # fmt: skip


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan  # nan is written as the string "nan"


def reuse_states() -> dict[str, list[str]]:
    """task -> its three reuse states, known3.json's order (e54/elicit is the reuse side of two pairs: once)."""
    out: dict[str, list[str]] = {}
    for c in json.loads(KNOWN3.read_text())["cells"]:
        if c["method"] == "share":
            out.setdefault(c["states"]["reuse"][0].rsplit("_s", 1)[0], c["states"]["reuse"])
    return out


def records(tasks: dict[str, list[str]]) -> list[dict[str, Any]]:
    out = []
    for task, states in tasks.items():
        for s in states:
            path = RESULTS / "updates" / f"{s.replace('/', '_')}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text())
            out.append({"task": task, "state": s, "kind": "fine-tune", "share": num(rec["fine_tune"]["r_route_split"]),
                        "gain": num(rec["fine_tune"]["gain"]), "step": "", "norm_rel": 1.0})  # fmt: skip
            for kind in ("at1", "end"):
                r = rec[kind]
                out.append({"task": task, "state": s, "kind": kind, "share": num(r["r"]), "gain": num(r.get("gain", r.get("max_gain"))),
                            "step": r["step"], "norm_rel": num(r.get("norm_rel"))})  # fmt: skip
    return out


def summary(tasks: dict[str, list[str]], data: list[dict[str, Any]]) -> dict[str, Any]:
    P = json.loads((RESULTS / "predictions.json").read_text())["rules"]
    out: dict[str, Any] = {}
    for kind in ("at1", "end"):
        means = {t: rules.task_mean([p["share"] for p in data if p["task"] == t and p["kind"] == kind]) for t in tasks}
        out[kind] = rules.reading(means, P["a_min"], P["b_max"], P["min_tasks"])
    return out


def main() -> None:
    tasks = reuse_states()
    data = records(tasks)
    fig, ax = figure(size=(FULL_WIDTH[0] * 0.8, 4.2))
    for kind, (method, off) in KINDS.items():
        groups = {t: [p["share"] for p in data if p["task"] == t and p["kind"] == kind and math.isfinite(p["share"])] for t in tasks}
        dots(ax, groups, method=method, xlabel="re-gating share, log-odds form", offset=off)
        ax.scatter([], [], s=22, color=METHODS[method].color, marker=METHODS[method].marker, label=LABELS[kind])
    for i, t in enumerate(tasks):  # a task none of whose seeds reached gain 1
        at1 = [p for p in data if p["task"] == t and p["kind"] == "at1"]
        if at1 and not any(math.isfinite(p["share"]) for p in at1):
            ax.annotate(f"no seed reached gain 1; largest {max(p['gain'] for p in at1):.2f}", (0.05, i), fontsize=6,
                        color=METHODS["random"].color, va="center", xycoords=("axes fraction", "data"))  # fmt: skip
    ax.set_title("Fine-tunes' share beside updates trained inside a random subspace", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, HERE / "figures/shares.png", data=data,
         n="per task: 3 fine-tune seeds; 3 random-subspace updates (each its own subspace and seed, rank-8-LoRA trainable "
           "count, the task's recipe, one lr multiplier), read at the first step with held-out gain >= 1 and at step 300; 67M model")  # fmt: skip
    s = summary(tasks, data)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
