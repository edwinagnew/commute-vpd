"""E109 (larger_size) figures from results/*.json, the stored sets here and E102/E107's stored results; no model. Colours as
E107's plots: the true-answer (known) fine-tunes purple, the wrong-answer arm teal, the fictional-author arm orange,
references grey.

  figures/seeds.png    seed overlap against cross-arm overlap, at the matched size and at the five-epoch end, on the
                       TOFU rows and on the answer tokens: each arm's two seeds, wrong in known (four pairs), known in
                       fictional (the filed x), a Gaussian change's set, and the filed power line (1 + x) / 2
  figures/learned.png  the share of 217 rows whose wrong sentence the model prefers, unchanged model, true-answer and
                       wrong-answer fine-tunes at both sizes, with the filed learned bar 0.75

Matched TOFU numbers are E107's (job 53575, label_vs_style/results/score.json); the rest are this run's (jobs 54053,
54062). Overlaps not in score.json (fictional seeds, Gaussian floor, the dry run's stand-in) are computed here
from the stored sets, as qwen_neurons.overlap: the share of the first set inside the second.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/larger_size/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG, SETS = HERE / "results", HERE / "figures", HERE / "results" / "sets"
E107 = HERE.parent / "label_vs_style" / "results"
E102 = HERE.parent / "qwen_carrying" / "results" / "states"
KAP = HERE.parent / "known_answer_pair" / "results"
PURPLE, TEAL, ORANGE, GREY, INK, GRID = "#8a4fbf", "#00a088", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"


def js(path: Path) -> Any:
    return json.loads(path.read_text())


def share(a: Path, b: Path, key: str = "carry") -> float:
    sa, sb = set(js(a)[key]), set(js(b)[key])
    return len(sa & sb) / len(sa)


def panels() -> dict[str, dict[str, dict[str, Any]]]:
    """Per probe and size: the scored numbers (from score files) and the descriptive ones (from sets)."""
    sc, e107 = js(RES / "score.json"), js(E107 / "score.json")["tofu"]["wrong"]
    s = lambda name, probe: SETS / f"{name}__{probe}.json"
    tofu_m = {"w_known": e107["w_known"], "w_wrong": e107["w_wrong"], "cross": e107["cross"], "x": e107["x"],
              "fict": share(E102 / "a2_s142_matched.json", E102 / "a2_s42_matched.json"),
              "gauss": share(E102 / "gauss_tofu.json", KAP / "neurons_amend1" / "a1_matched__tofu.json")}
    tofu_e = {k: sc["tofu"][k] for k in ("w_known", "w_wrong", "cross", "x")} | {
        "fict": share(s("FICT_learned142", "tofu"), s("FICT_learned42", "tofu")),
        "gauss": share(E102 / "gauss_tofu.json", s("KNOWN_end42", "tofu")),
        "standin": share(E102 / "a2_s142_full.json", E102 / "a2_s42_full.json")}
    t_m = {k: sc["T_matched"][k] for k in ("w_known", "w_wrong", "cross", "x")} | {
        "fict": share(s("FICT_matched142", "T"), s("FICT_matched42", "T"), "carry_abs"),
        "gauss": share(s("GAUSS", "T"), s("KNOWN_matched42", "T"), "carry_abs")}
    t_e = {k: sc["T"][k] for k in ("w_known", "w_wrong", "cross", "x")} | {
        "fict": share(s("FICT_learned142", "T"), s("FICT_learned42", "T"), "carry_abs"),
        "gauss": sc["reported"]["gauss_in_known_42"]}
    assert abs(t_e["gauss"] - share(s("GAUSS", "T"), s("KNOWN_end42", "T"), "carry_abs")) < 1e-12
    return {"tofu": {"matched": tofu_m, "end": tofu_e}, "T": {"matched": t_m, "end": t_e}}


def plot_seeds(p: dict[str, dict[str, dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    titles = {"tofu": "TOFU rows, whole answer (scored)", "T": "true answer tokens only (scored)"}
    for ax, probe in zip(axes, ("tofu", "T"), strict=True):
        for i, size in enumerate(("matched", "end")):
            d = p[probe][size]
            line = (1 + d["x"]) / 2
            ax.hlines(line, i - 0.38, i + 0.38, color=INK, ls="--", lw=1.4, label="power line (1 + x) / 2" if i == 0 else None)
            ax.scatter(i - 0.27, d["w_known"], s=90, color=PURPLE, zorder=3, label="true-answer arm, its two seeds" if i == 0 else None)
            ax.scatter(i - 0.15, d["w_wrong"], s=90, color=TEAL, zorder=3, label="wrong-answer arm, its two seeds" if i == 0 else None)
            ax.scatter([i - 0.03] * 4, d["cross"], s=40, marker="x", color=TEAL, zorder=3,
                       label="wrong in true-answer, four seed pairs" if i == 0 else None)
            ax.scatter(i + 0.09, d["x"], s=80, marker="s", color=ORANGE, zorder=3,
                       label="x: true-answer in fictional, highest pair" if i == 0 else None)
            ax.scatter(i + 0.21, d["fict"], s=80, marker="o", facecolor="white", edgecolor=ORANGE, lw=2, zorder=3,
                       label="fictional arm, its two seeds" if i == 0 else None)
            ax.scatter(i + 0.32, d["gauss"], s=70, marker="v", color=GREY, zorder=3,
                       label="random Gaussian change in true-answer" if i == 0 else None)
            if "standin" in d:
                ax.scatter(i + 0.21, d["standin"], s=80, marker="D", facecolor="white", edgecolor=GREY, lw=1.5, zorder=3,
                           label="dry run's stand-in seed level")
            word = "power" if d["w_known"] > line else "no power"
            if probe == "tofu" and size == "matched":  # E107 filed no power gate; the line is drawn for comparison
                word = "above the line\n(E107's job, no gate filed)"
            ax.text(i, 0.08, word, ha="center", color=INK, fontsize=10, weight="bold")
        ax.set_xticks([0, 1], ["matched size\n|δW| about 0.49", "five-epoch end\n|δW| about 8.3"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(titles[probe], fontsize=11)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("share of one 1,024-neuron set inside another")
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    for h, lab in zip(h2, l2, strict=True):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Is the carrying set reproducible across data-order seeds at the five-epoch end?", fontsize=12)
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(FIG / "seeds.png", dpi=130)
    plt.close(fig)


def plot_learned() -> None:
    ctrl = js(RES / "control.json")
    w107 = [js(E107 / f"arm_WRONG{s}.json")["prefers_wrong"]["matched"] for s in (42, 142)]
    wend = [js(RES / f"arm_WRONG_end{s}.json")["sentences"]["prefers_wrong"] for s in (42, 142)]
    bars = [("unchanged\nmodel", [ctrl["W0"]["prefers_wrong"]], GREY),
            ("true-answer\nmatched", [ctrl[f"KNOWN_matched{s}"]["prefers_wrong"] for s in (42, 142)], PURPLE),
            ("true-answer\nend", [ctrl[f"KNOWN_end{s}"]["prefers_wrong"] for s in (42, 142)], PURPLE),
            ("wrong-answer\nmatched", w107, TEAL), ("wrong-answer\nend", wend, TEAL)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (_, vals, c) in enumerate(bars):
        xs = [i] if len(vals) == 1 else [i - 0.18, i + 0.18]
        ax.bar(xs, vals, width=0.32, color=c, alpha=0.4 if "matched" in bars[i][0] else 1.0)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.axhline(0.75, color=INK, ls="--", lw=1.2)
    ax.text(1.5, 0.77, "filed learned bar 0.75", fontsize=9, color=INK, ha="center")
    ax.axhline(ctrl["W0"]["prefers_wrong"], color=GREY, ls=":", lw=1.2)
    ax.text(2.0, ctrl["W0"]["prefers_wrong"] + 0.02, "unchanged model's level", fontsize=9, color=GREY, ha="center")
    ax.set_xticks(range(len(bars)), [b[0] for b in bars])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of 217 rows preferring the wrong sentence")
    ax.set_title("Has each fine-tune learned the answers it was trained on? (bars: seeds 42, 142)", fontsize=11)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "learned.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    p = panels()
    for probe, sizes in p.items():
        for size, d in sizes.items():
            print(probe, size, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(c, 4) for c in v])
                                           for k, v in d.items()}))
    plot_seeds(p)
    plot_learned()


if __name__ == "__main__":
    main()
