"""Figure and scorecard for ladder_second_seed (E26). JSON only.

figures/seeds.png: per task (columns) and arm (rows), the adversarial (PGD) ratio at steps 100 and 1000: E21's seed 0
(L40), this job's seed 0 at 1000 (the gate) and seed 1, with this job's pass and fail lines. Hollow markers: held-out
P(target) below 0.9. results/scorecard.json: P1, P2, P3 read by the filed rules.

Usage (repo root): uv run python experiments/ladder_second_seed/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
E21 = HERE.parent / "sgd_ladder_long" / "results"
TASKS = ("r0_t352_v310", "r1_t534_v369", "r2_t275_v2667")
Q_TASKS = TASKS[:2]
STEPS = ("100", "1000")
YTICKS = (0.9, 1, 1.25, 1.5, 2, 3, 5, 10, 20, 40)


def load() -> dict[str, dict[str, Any]]:
    return {t: json.loads((RESULTS / f"ladder_second_seed_{t}.json").read_text()) for t in TASKS
            if (RESULTS / f"ladder_second_seed_{t}.json").is_file()}


def verdicts(d: dict[str, Any], arm: str, steps: tuple[str, ...]) -> list[str]:
    st = d["states"][f"seed{d['seed']}"].get(arm, {})
    return [st[s]["score"]["verdict"] if s in st else "missing" for s in steps]


def outcome_p1(v: list[str]) -> str:
    if "missing" in v:
        return "not run"
    if any(x in ("fail", "below P 0.9") for x in v):
        return "one-seed pass"
    return "replicates" if all(x == "pass" for x in v) else "unresolved"


def outcome_p2(v: list[str]) -> str:
    if "missing" in v:
        return "not run"
    if "pass" in v:
        return "seed noise"
    return "replicates" if all(x == "fail" for x in v) else "unresolved"


def scorecard(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The filed rules; a task whose G1 failed is listed as unscored and left out of P1/P2."""
    scored = [t for t in Q_TASKS if t in runs and runs[t].get("score", {}).get("scored")]
    v1 = [x for t in scored for x in verdicts(runs[t], "adam_1e-05", STEPS)]
    v2 = [x for t in scored for x in verdicts(runs[t], "sgd_0.03", ("1000",))]
    out: dict[str, Any] = {"scored_tasks": scored, "unscored_tasks": [t for t in Q_TASKS if t not in scored],
                           "P1": {"verdicts": v1, "outcome": outcome_p1(v1) if len(scored) == 2 else "unscored"},
                           "P2": {"verdicts": v2, "outcome": outcome_p2(v2) if len(scored) == 2 else "unscored"}}
    r2 = runs.get(TASKS[2])
    if r2 is not None and "score" in r2:
        v3 = verdicts(r2, "sgd_0.1", ("1000",)) + verdicts(r2, "adam_1e-05", ("1000",))
        out["P3"] = {"verdicts": v3, "outcome": "yes" if "pass" in v3 else "no (expected)"}
    return out


def points(states: dict[str, Any], arm: str) -> list[tuple[int, float, bool]]:
    """(step, PGD ratio, P >= 0.9) for each scored step of `arm` in a states record."""
    return [(int(s), r["pile"]["pgd_kl"], r["heldout"]["p_o"] >= 0.9) for s, r in states.get(arm, {}).items()
            if s in STEPS]


def draw(ax: Any, pts: list[tuple[int, float, bool]], colour: str, marker: str, label: str, line: bool,
         shift: float) -> None:
    """The points at step x shift (side by side on the log axis), filled when P(target) >= 0.9."""
    if not pts:
        return
    xy = sorted((x * shift, y, learned) for x, y, learned in pts)
    if line:
        ax.plot([p[0] for p in xy], [p[1] for p in xy], color=colour, lw=1, alpha=0.6)
    for x, y, learned in xy:
        ax.scatter(x, y, s=50, marker=marker, color=colour if learned else "none", edgecolors=colour, zorder=3,
                   label=label)
        label = "_"


def figure(runs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for j, task in enumerate(TASKS):
        d, e21 = runs.get(task), json.loads((E21 / f"sgd_ladder_long_{task}.json").read_text())
        sgd = f"sgd_{e21['ref_3a']['sgd_lr']:g}"
        for i, (arm, name) in enumerate(((sgd, f"SGD {sgd[4:]}"), ("adam_1e-05", "Adam 1e-5"))):
            ax = axes[i, j]
            draw(ax, points(e21["states"], arm), "0.55", "s", "seed 0, E21 (L40)", True, 0.9)
            if d is not None:
                draw(ax, points(d["states"].get("seed0", {}), arm), "C2", "D", "seed 0, this job", False, 1.0)
                draw(ax, points(d["states"].get(f"seed{d['seed']}", {}), arm), "C0", "o", f"seed {d['seed']}", True, 1.1)
                f = d["W0"]["pgd_floor"]
                ax.axhline(LINE / (1 + f), color="g", ls="--", lw=0.8, label=f"pass line (f {f:.3f})")
                ax.axhline(LINE * (1 + f), color="r", ls="--", lw=0.8, label="fail line")
            ax.set_xscale("log")
            ax.set_xticks([100, 1000], ["100", "1000"])
            if i == 0:  # SGD spans 0.9 to 40: log axis with plain labels
                ax.set_yscale("log")
                lo, hi = ax.get_ylim()
                ax.set_yticks([t for t in YTICKS if lo <= t <= hi], [f"{t:g}" for t in YTICKS if lo <= t <= hi])
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{task}: {name}", fontsize=10)
            if j == 0:
                ax.set_ylabel("adversarial ratio to W0")
            if i == 1:
                ax.set_xlabel("component-training step")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Adversarial ratio by batch-order seed (hollow: held-out P(target) below 0.9)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "seeds.png", dpi=150)


def main() -> None:
    runs = load()
    sc = scorecard(runs)
    (RESULTS / "scorecard.json").write_text(json.dumps(sc, indent=1))
    print(json.dumps(sc, indent=1))
    figure(runs)


if __name__ == "__main__":
    main()
