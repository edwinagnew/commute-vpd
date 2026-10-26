"""Figure and scorecard for teacher_training Amendment 2, dropping the rows where the emoticon trigger is partly on
(plan: reports/2026-10-23_plan_teacher-training.md -> "Amendment 2"), read from the JSONs only.

  amend2.png             top: held-out P(target) of the Pile-text copy per emoticon teacher in its four versions, in
                         order of how much trigger text is removed (B, B-masked, B-dropped at 0.5, B-dropped at 0.1;
                         the first three from teacher_training_amend1.json), filled at step 1000 and hollow at step 100,
                         with the teacher's P and the 0.9 line; bottom: the same four versions' cosine with delta-W
                         beside arm A's, the adjoint split's and a random direction's
  scorecard_amend2.json  C1, G1 and P1-P2 by the filed rules (amend2.py's `score`, loaded as tests/test_teacher_amend2.py
                         does)

Usage (repo root): uv run python experiments/teacher_training/plot_amend2.py [--results DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
TEACHERS = ("sgd", "adam")
TEACHER_NAME = {"sgd": "SGD 0.01 teacher", "adam": "Adam 3e-5 teacher"}
VERSIONS = ("B", "B-masked", "B-dropped", "B-dropped-0.1")  # by how much trigger text is removed
VERSION_NAME = {"B": "all rows", "B-masked": "loss off at\nthe trigger", "B-dropped": "g > 0.5 rows\nreplaced",
                "B-dropped-0.1": "g > 0.1 rows\nreplaced"}
REFS = ("A", "adjoint", "null")
REF_NAME = {"A": "copy on\ntask text", "adjoint": "adjoint\nsplit", "null": "random\ndirection"}
SHORT_NAME = {**{v: n.replace(" rows\n", "\n").replace("off at", "off").replace("the trigger", "at trigger")
                 for v, n in VERSION_NAME.items()}, **REF_NAME}  # the cosine panel's narrower columns
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COLOR = {"B": "#eb6834", "A": "#2a78d6", "adjoint": MUTED, "null": INK}  # plot.py's colours for the same arms
BOX = {"facecolor": "white", "edgecolor": "none", "pad": 0.5}
W0_P_EMO = 0.004  # the unchanged model's held-out P(target) on emoticon (plan's measurement table; not in the JSONs)
Rec = dict[str, Any]


def load_module(name: str) -> Any:
    """amend2.py and its helper module run.py, loaded by path as tests/test_teacher_amend2.py does (no sys.path edit)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def versions(am1: Rec, am2: Rec, opt: str) -> dict[str, Rec]:
    """The four Pile-text copies of one teacher at step 1000: Amendment 1's three, then this amendment's."""
    return {**{v: am1["teachers"][opt][v] for v in VERSIONS[:3]}, VERSIONS[3]: am2["teachers"][opt][VERSIONS[3]]}


def rows_removed(am1: Rec, am2: Rec) -> dict[str, int]:
    return {"B": 0, "B-masked": 0, "B-dropped": am1["trigger"]["rows"], "B-dropped-0.1": am2["trigger"]["above_0.1"]["rows"]}


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_p(ax: Axes, am1: Rec, am2: Rec, p_min: float) -> None:
    removed, n = rows_removed(am1, am2), len(VERSIONS)
    for g, opt in enumerate(TEACHERS):
        x0, vs = (n + 1) * g, versions(am1, am2, opt)
        p_t = am2["teachers"][opt]["repro"]["p_target"]
        ax.plot([x0 - 0.4, x0 + n - 0.6], [p_t] * 2, color=INK, lw=1.4)
        ax.text(x0 - 0.4, p_t + 0.01, f"teacher {p_t:.3f}", fontsize=7, va="bottom")
        for k, v in enumerate(VERSIONS):
            r = vs[v]
            ax.plot([x0 + k] * 2, [r["p_target_100"], r["p_target"]], color=COLOR["B"], lw=1, alpha=0.6)
            ax.scatter(x0 + k, r["p_target_100"], s=45, marker="s", color="none", edgecolors=COLOR["B"], zorder=3)
            ax.scatter(x0 + k, r["p_target"], s=45, marker="s", color=COLOR["B"], zorder=3)
            ax.text(x0 + k + 0.12, r["p_target"] - 0.02, f"{r['p_target']:.3f}", fontsize=7, va="top", bbox=BOX)
        ax.text(x0 + (n - 1) / 2, -0.36, TEACHER_NAME[opt], ha="center", fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.axhline(p_min, color=INK, ls="--", lw=0.8)
    ax.axhline(W0_P_EMO, color=MUTED, ls=":", lw=1)
    ax.text(-0.45, p_min - 0.01, f"{p_min:g}", fontsize=7, va="top")
    ax.text(-0.45, W0_P_EMO + 0.01, f"unchanged model {W0_P_EMO:g}", fontsize=7, color=MUTED, va="bottom")
    ax.set_xticks([(n + 1) * g + k for g in range(len(TEACHERS)) for k in range(n)],
                  [f"{VERSION_NAME[v]}\n{removed[v]} removed" for _ in TEACHERS for v in VERSIONS], fontsize=6.5)
    ax.set_xlim(-0.6, 2 * n + 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out P(target)")
    ax.set_title("Does the Pile-text copy still learn as more of the trigger is removed?\n"
                 "filled: step 1000, hollow: step 100; x: of the 8,192 Pile rows, how many were replaced by clean spare rows",
                 fontsize=9, loc="left")
    style(ax)


def panel_cos(ax: Axes, am1: Rec, am2: Rec) -> None:
    cols = (*VERSIONS, *REFS)
    n = len(cols)
    for g, opt in enumerate(TEACHERS):
        x0, vs, ref = (n + 1) * g, versions(am1, am2, opt), am1["teachers"][opt]["cos_ref"]
        for k, c in enumerate(cols):
            v = (vs[c]["cos"] if c in VERSIONS else ref[c])["all"]
            ax.bar(x0 + k, v, width=0.7, color=COLOR.get(c, COLOR["B"]))
            ax.text(x0 + k, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5)
        ax.text(x0 + (n - 1) / 2, -0.3, TEACHER_NAME[opt], ha="center", fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_xticks([(n + 1) * g + k for g in range(len(TEACHERS)) for k in range(n)],
                  [SHORT_NAME[c] for _ in TEACHERS for c in cols], fontsize=6.5)
    ax.set_xlim(-0.6, 2 * n + 0.4)
    ax.set_ylabel("cosine of the weight change with δW")
    ax.set_title("Does removing the trigger change where the copy heads? cosine with the fine-tune's δW\n"
                 "orange: the four Pile-text copies; blue: the task-text copy; grey: the adjoint split", fontsize=9,
                 loc="left")
    style(ax)
    ax.grid(False, axis="x")


def fig_amend2(am1: Rec, am2: Rec, p_min: float, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9.2))
    panel_p(ax1, am1, am2, p_min)
    panel_cos(ax2, am1, am2)
    fig.suptitle("Emoticon, copies trained on Pile text: the trigger rows removed at two thresholds", fontsize=10,
                 x=0.01, ha="left")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=3)
    fig.savefig(out / "amend2.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'amend2.png'}")


def scorecard(am2: Rec, pred: Rec) -> dict[str, Rec]:
    load_module("run")
    sc = load_module("amend2").score(am2)
    rules = {p["id"]: p["rule"] for p in pred["predictions"]}
    out: dict[str, Rec] = {
        "C1": {"claim": "the unchanged model reads its known Pile values", "rule": pred["controls"]["C1"],
               "read": {k: am2["W0"][k] for k in ("pgd_floor", "stochastic_floor", "pgd_rel", "ci_rel")},
               "verdict": "pass" if am2["W0"]["pass"] else "fail"}}
    for opt in TEACHERS:
        g1 = am2["G1"][opt]
        out[f"G1_{opt}"] = {"claim": f"the regenerated {opt} teacher reads Amendment 1's reproduction",
                            "rule": pred["controls"]["G1"], "read": {"p_target": g1["p_target"], "size": g1["size"]},
                            "verdict": "pass" if g1["pass"] else "fail"}
    for opt in TEACHERS:
        out[f"P1_{opt}"] = {"claim": "the partly-on rows carried the Pile-only result", "rule": rules["P1_loophole"],
                            "read": {"p_target": sc[opt]["p_target"], "verdict": sc[opt]["verdict"]}, "verdict": sc[opt]["P1"]}
    for opt in TEACHERS:
        out[f"P2_{opt}"] = {"claim": "the copy's cosine with delta-W is unchanged", "rule": rules["P2_cosine_unchanged"],
                            "read": {"cos": sc[opt]["cos"]}, "verdict": "pass" if sc[opt]["P2"] else "fail"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=None, help="figure and scorecard here (default figures/, results/)")
    args = ap.parse_args()
    am1 = json.loads((args.results / "teacher_training_amend1.json").read_text())
    am2 = json.loads((args.results / "teacher_training_amend2.json").read_text())
    pred = json.loads((args.results / "predictions_amend2.json").read_text())
    sc = scorecard(am2, pred)
    for k, r in sc.items():
        print(f"{k:8} {r['verdict']:<16} {r['claim']}\n     read: {json.dumps(r['read'], default=float)}")
    (args.out or args.results).mkdir(parents=True, exist_ok=True)
    (args.out or args.results).joinpath("scorecard_amend2.json").write_text(json.dumps(sc, indent=1, default=float))
    fig_amend2(am1, am2, pred["config"]["p_min"], args.out or HERE / "figures")


if __name__ == "__main__":
    main()
