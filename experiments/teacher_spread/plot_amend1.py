"""Figures for teacher_spread Amendment 1 (E37 A1), read from results/amend1.json, with E37's own states from
results/teacher_spread.json as references. Plan: reports/plans/2026-10-26_plan_teacher-spread.md -> "Amendment 1";
analysis: reports/analyses/2026-10-26_analysis_teacher-spread.md -> "Amendment 1".

  amend1_steps.png  top: the check ratio (harness PGD KL over W0's) against teacher-training steps for arm 1 (from W0's
                    subcomponents) and arm 3 (from the scaled adjoint split), with W0's 1 and seed spread, the pass and
                    fail lines, E37's second batch order and do-nothing, and M0/M0b at their step 1000; middle: the
                    held-out gap to the fine-tuned model, KL(fine-tuned || state), for the same reads; bottom: the
                    training KL, single logged steps from results/large/amend1_run.log (arm 1's run first, then
                    arm 3's) and the 50-step means before each read from the JSON
  amend1_m0.png     check ratio against damage elsewhere, KL(W0 || state) on plain Pile: M0 and M0b against M0's filed
                    region around W0, beside every teacher-trained read and do-nothing

Usage (repo root): ~/commute-agents/fleet/cpu-run uv run python experiments/teacher_spread/plot_amend1.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INK, MUTED, GRID, BAND, INCONCL = "#0b0b0b", "#8a8984", "#e6e5e0", "#f0efec", "#fbeee0"
PASS_C, FAIL_C = "#2e8b57", "#c0392b"
ARM = {  # arm 1 keeps E37's colour for the same recipe (plot.py, lowercase copy)
    "arm1": {"c": "#2a78d6", "m": "o", "ls": "-", "name": "arm 1: teacher training from the released subcomponents"},
    "arm3": {"c": "#6b4fbb", "m": "s", "ls": "--", "name": "arm 3: teacher training from the adjoint split"},
}
M0_C = "#b5651d"
M0_NAME = {"M0": "masked recipe, fine-tuned model = W₀, released coefficients",
           "M0b": "the same at coefficients 0.05"}


def style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def series(a: dict[str, Any], arm: str, key: str) -> tuple[list[int], list[float]]:
    """(steps, values) of one arm, step 0 included: arm 1 starts at W0's subcomponents, arm 3 at its read arm3_0."""
    st, d = a["states"], a["score"]["derived"]
    reads = sorted(int(n.split("_")[1]) for n in st if n.startswith(arm + "_"))
    start = "W0" if arm == "arm1" else "arm3_0"
    val = {"check": lambda n: 1.0 if n == "W0" else d[n]["check"], "gap": lambda n: st[n]["gap"]}[key]
    return [0, *[r for r in reads if r]], [val(start), *[val(f"{arm}_{r}") for r in reads if r]]


def lines(a: dict[str, Any]) -> tuple[float, float, float]:
    """Pass and fail lines on this job's W0 floor (faith_score.verdict), and W0's spread over adversary seeds."""
    f = a["C1"]["pgd_floor"]
    spread = max(p["pgd_kl"] for p in a["W0_pile"].values()) / a["W0_pile"]["0"]["pgd_kl"]
    return LINE / (1 + f), LINE * (1 + f), spread


def panel_check(ax: Axes, a: dict[str, Any], e37: dict[str, Any]) -> None:
    pass_l, fail_l, spread = lines(a)
    d = a["score"]["derived"]
    ax.axhspan(1, spread, color=BAND, lw=0)
    ax.axhspan(pass_l, fail_l, color=INCONCL, lw=0)
    ax.axhline(pass_l, color=PASS_C, ls="--", lw=0.9)
    ax.axhline(fail_l, color=FAIL_C, ls="--", lw=0.9)
    dn = e37["states"]["do-nothing"]["pile"]["pgd_kl"]
    ax.axhline(dn, color=INK, ls=":", lw=0.8)
    for arm, s in ARM.items():
        x, y = series(a, arm, "check")
        ax.plot(x, y, color=s["c"], ls=s["ls"], lw=1.5, marker=s["m"], ms=7, label=s["name"], zorder=4)
        for xi, yi in zip(x, y, strict=True):
            if xi:
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(6, 5 if arm == "arm1" else -11), textcoords="offset points",
                            fontsize=7, color=INK)
    ax.annotate("1.05: the adjoint split itself", (0, d["arm3_0"]["check"]), xytext=(12, -16), textcoords="offset points",
                fontsize=7, color=INK)
    o1 = e37["states"]["lower_s1"]["pile"]["pgd_kl"]
    ax.scatter(1000, o1, s=50, marker="o", facecolors="white", edgecolors=ARM["arm1"]["c"], lw=1.5, zorder=5,
               label="E37: the same recipe at a second batch order")
    ax.annotate(f"{o1:.2f}", (1000, o1), xytext=(-8, 3), textcoords="offset points", fontsize=7, ha="right")
    for name, mk in (("M0", "D"), ("M0b", "d")):
        r = d[f"{name}_1000"]
        ax.scatter(1000, r["check"], s=60, marker=mk, color=M0_C, zorder=5, label=M0_NAME[name])
        ax.annotate(f"{r['check']:.2f}", (1000, r["check"]), xytext=(7, -3), textcoords="offset points", fontsize=7)
    x_txt = 4150
    ax.text(x_txt, spread * 1.0, f"unchanged model across 3 adversary seeds: 1 to {spread:.2f}", fontsize=7,
            color=MUTED, ha="right", va="bottom")
    ax.text(x_txt, pass_l * 0.99, f"pass line {pass_l:.2f}", fontsize=7, color=PASS_C, ha="right", va="top")
    ax.text(x_txt, fail_l * 1.01, f"fail line {fail_l:.2f}; between the lines: inconclusive", fontsize=7, color=FAIL_C,
            ha="right", va="bottom")
    ax.text(x_txt, dn * 0.985, f"E37: the fine-tuned model left in Δ (do-nothing), {dn:.2f}", fontsize=7, color=INK,
            ha="right", va="top")
    ax.set_ylim(0.8, 5.15)
    ax.set_ylabel("check ratio: adversary's KL over\nthe unchanged decomposition's")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)
    ax.set_title("Does training longer, or starting from the adjoint split, bring the check under the pass line?\n"
                 "No: both arms end between 1.8 and 2.3 times the unchanged decomposition; the masked recipe reads "
                 "higher at W₀", fontsize=9, loc="left")
    style(ax)


def panel_gap(ax: Axes, a: dict[str, Any]) -> None:
    w0_gap = a["states"]["W0"]["gap"]
    ax.axhline(w0_gap, color=MUTED, ls=":", lw=0.9)
    ax.text(4150, w0_gap * 1.08, f"unchanged model {w0_gap:.3f}; the fine-tuned model reads 0", fontsize=7,
            color=MUTED, ha="right", va="bottom")
    for arm, s in ARM.items():
        x, y = series(a, arm, "gap")
        ax.plot(x, y, color=s["c"], ls=s["ls"], lw=1.5, marker=s["m"], ms=7, zorder=4)
        for xi, yi in zip(x, y, strict=True):
            if xi:
                ax.annotate(f"{yi:.4f}", (xi, yi), xytext=(6, 5 if arm == "arm1" else -11), textcoords="offset points",
                            fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 0.5)
    ax.set_xlim(-150, 4250)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlabel("teacher-training steps (Adam 1e-5, batches of 8 lowercase rows)")
    ax.set_ylabel("gap to the fine-tuned model:\nKL(fine-tuned ‖ state), nats per token")
    style(ax)


def logged_steps(log: Path) -> dict[str, tuple[list[int], list[float]]]:
    """Single-step training KL from the run log's "[component training ...] step N: loss X" lines: arm 1's run, then
    arm 3's (the step count restarts between them)."""
    runs: list[tuple[list[int], list[float]]] = []
    for m in re.finditer(r"\[component training lr [^\]]+\] step (\d+): loss ([\d.]+)", log.read_text()):
        n, v = int(m.group(1)), float(m.group(2))
        if not runs or n < runs[-1][0][-1]:
            runs.append(([], []))
        runs[-1][0].append(n)
        runs[-1][1].append(v)
    assert len(runs) == 2, f"expected arm 1's and arm 3's runs in {log}, found {len(runs)}"
    return dict(zip(("arm1", "arm3"), runs, strict=True))


def panel_train(ax: Axes, a: dict[str, Any], log: Path) -> None:
    st = a["states"]
    for arm, (x, y) in logged_steps(log).items():
        s = ARM[arm]
        ax.plot(x, y, color=s["c"], ls=s["ls"], lw=0.8, alpha=0.5, zorder=3)
        ax.scatter(x, y, s=14, marker=s["m"], facecolors="white", edgecolors=s["c"], lw=0.8, zorder=4)
        reads = sorted(int(n.split("_")[1]) for n in st if n.startswith(arm + "_") and "train_kl" in st[n])
        means = [st[f"{arm}_{r}"]["train_kl"] for r in reads]
        ax.scatter(reads, means, s=50, marker=s["m"], color=s["c"], zorder=5)
        for r, v in zip(reads, means, strict=True):
            ax.annotate(f"{v:.4f}", (r, v), xytext=(6, 6 if arm == "arm1" else -12), textcoords="offset points",
                        fontsize=7)
    k2, k4 = st["arm1_2000"]["train_kl"], st["arm1_4000"]["train_kl"]
    ax.text(3000, 0.12, f"arm 1's 50-step mean fell {1 - k4 / k2:.0%} from step 2,000 to 4,000;\n"
            "the plan calls a run flat below a 10% fall per doubling", fontsize=7, ha="center", color=INK)
    ax.scatter([], [], s=14, marker="o", facecolors="white", edgecolors=INK, label="single logged steps")
    ax.scatter([], [], s=50, marker="o", color=INK, label="mean of the 50 steps before each read")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.72), frameon=False)
    ax.set_yscale("log")
    ax.set_ylim(0.002, 0.5)
    ax.set_xlabel("teacher-training steps (Adam 1e-5, batches of 8 lowercase rows)")
    ax.set_ylabel("training KL:\nKL(fine-tuned ‖ student) on the batch")
    ax.set_title("And the training KL is still falling at step 4,000", fontsize=9, loc="left")
    style(ax)


def fig_steps(a: dict[str, Any], e37: dict[str, Any], log: Path) -> None:
    fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(8.6, 10.2), sharex=True,
                                        gridspec_kw={"height_ratios": (1.7, 1, 1)})
    panel_check(top, a, e37)
    panel_gap(mid, a)
    mid.set_xlabel("")
    mid.set_title("Meanwhile the gap to the fine-tuned model on held-out lowercase text keeps falling", fontsize=9,
                  loc="left")
    panel_train(bot, a, log)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_steps.png", dpi=150)
    plt.close(fig)


def fig_m0(a: dict[str, Any], e37: dict[str, Any]) -> None:
    pass_l, fail_l, _ = lines(a)
    d = a["score"]["derived"]
    dmax, cmax = a["amend1"]["m0_damage_max"], a["amend1"]["m0_check_max"]  # the filed M0 limits
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.add_patch(plt.Rectangle((-0.01, 0.8), dmax + 0.01, cmax - 0.8, color="#e3f1e8", lw=0, zorder=0))
    ax.text(dmax + 0.004, 1.13, f"← M0 had to stay in the green box:\ndamage ≤ {dmax}, check ≤ {cmax}", fontsize=7, color=PASS_C,
            va="bottom")
    ax.axhline(pass_l, color=PASS_C, ls="--", lw=0.9)
    ax.axhline(fail_l, color=FAIL_C, ls="--", lw=0.9)
    ax.text(0.14, pass_l * 0.99, f"pass line {pass_l:.2f}", fontsize=7, color=PASS_C, ha="right", va="top")
    ax.text(0.14, fail_l * 1.01, f"fail line {fail_l:.2f}", fontsize=7, color=FAIL_C, ha="right", va="bottom")
    ax.scatter(0, 1, s=60, marker="o", facecolors="white", edgecolors=INK, lw=1.5, zorder=5, label="unchanged model, W₀")
    for arm, s in ARM.items():
        names = [n for n in d if n.startswith(arm + "_") and n != "arm3_0"]
        ax.scatter([d[n]["damage"] for n in names], [d[n]["check"] for n in names], s=45, marker=s["m"], color=s["c"],
                   zorder=4, label=s["name"].split(":")[0] + ", every read")
    ax.scatter(d["arm3_0"]["damage"], d["arm3_0"]["check"], s=45, marker="s", facecolors="white",
               edgecolors=ARM["arm3"]["c"], lw=1.5, zorder=4, label="the adjoint split, arm 3's start")
    dn = e37["states"]["do-nothing"]
    ax.scatter(dn["damage"], dn["pile"]["pgd_kl"], s=55, marker="o", color=INK, zorder=4,
               label="the fine-tuned model left in Δ (do-nothing, E37)")
    for name, mk in (("M0", "D"), ("M0b", "d")):
        r = d[f"{name}_1000"]
        ax.scatter(r["damage"], r["check"], s=70, marker=mk, color=M0_C, zorder=6, label=M0_NAME[name] + ", step 1000")
        ax.annotate(f"check {r['check']:.2f}, damage {r['damage']:.3f}", (r["damage"], r["check"]), xytext=(9, -3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlim(-0.01, 0.265)
    ax.set_ylim(0.8, 5.15)
    ax.set_xlabel("damage elsewhere: KL(W₀ ‖ state) on held-out plain Pile rows, nats per token")
    ax.set_ylabel("check ratio: adversary's KL over\nthe unchanged decomposition's")
    ax.set_title("With nothing to copy, does the masked recipe leave the decomposition where it was?\n"
                 "No: it barely changes the model's predictions, yet the check reads higher than after copying the "
                 "fine-tune",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, columnspacing=1.0)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_m0.png", dpi=150)
    plt.close(fig)


def main() -> None:
    a = json.loads((HERE / "results" / "amend1.json").read_text())
    e37 = json.loads((HERE / "results" / "teacher_spread.json").read_text())
    assert a["status"] == "done", a["status"]
    a["amend1"] = json.loads((HERE / "results" / "amend1_predictions.json").read_text())["amend1"]
    FIG.mkdir(exist_ok=True)
    fig_steps(a, e37, HERE / "results" / "large" / "amend1_run.log")
    fig_m0(a, e37)
    print(f"wrote {FIG / 'amend1_steps.png'} and {FIG / 'amend1_m0.png'}")


if __name__ == "__main__":
    main()
