"""E103 Amendment 2's figures from results/amend2_*.json, amend1_train.json, neurons_amend1.json and neurons_amend2.json;
no model. Colours as plot.py: the known-facts arm (Amendment 1's sentence run) purple, the fictional arm orange, the
Gaussian change grey. Seed 42 is a filled marker, seed 142 a hollow one.

  figures/amend2_gain.png        content gain on the sentence reader against |dW| through warm-up, both known-facts seeds,
                                 with C1's bar, the fictional reference and the template's weight change
  figures/amend2_overlap.png     carrying-set overlaps on each row set: each arm's two seeds, the four cross-arm pairs and
                                 the Gaussian change, with the filed rule's two lines
  figures/amend2_pair_rules.png  the four matched pair rules in units of their bands, seed 42 and seed 142
  figures/amend2_options.png     at Amendment 1's end state, the change from W0 in the right and wrong options' log P,
                                 answer words and end token apart
  figures/amend2_recall.png      share of the 217 real facts under each reader, W0 and Amendment 1's end state

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/known_answer_pair/plot_amend2.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
PURPLE, ORANGE, BLUE, GREY, INK, GRID = "#8a4fbf", "#eb6834", "#2a78d6", "#7a7a75", "#52514e", "#e4e3df"
TEMPLATE_NORM = 0.4865  # E91's reference, run.py REF_NORM
RULES = [("real", "engaged"), ("tofu", "engaged"), ("real", "prompt_overlap"), ("tofu", "prompt_overlap")]
WORDS = {"engaged": "engaged share", "prompt_overlap": "prompt overlap", "real": "real rows", "tofu": "TOFU rows"}


def js(name: str) -> Any:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def gain(a1: dict[str, Any], a2: dict[str, Any], ref: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for tr, mfc, lab in ((a1, PURPLE, "seed 42"), (a2, "none", "seed 142")):
        norm = {r["step"]: r["dw_norm"] for r in tr["steps"]}
        steps = sorted(int(t) for t in tr["gains"])
        ax.plot([norm[t] for t in steps], [tr["gains"][str(t)]["content"] for t in steps], "-o", color=PURPLE, mfc=mfc,
                ms=4, lw=1, label=f"known-facts fine-tune, {lab}, steps 2 to {steps[-1]}")
        m = tr["matched"]
        ax.plot([m["norm"]], [m["gain"]["content"]], "*", color=PURPLE, mfc=mfc, ms=15)
    ax.plot([], [], "*", color=PURPLE, ms=12, label="matched step, each seed")
    ax.axhline(ref["content_gain"], color=ORANGE, ls="--", lw=1.2,
               label=f"fictional matched state on its own answers, {ref['content_gain']:.2f}")
    ax.axhline(ref["bar"], color=INK, ls=":", lw=1.2, label=f"control's bar, half of that, {ref['bar']:.2f}")
    ax.axhline(0, color=GREY, lw=1, label="unchanged model and do-nothing, 0")
    ax.axvline(TEMPLATE_NORM, color=BLUE, lw=0.8, alpha=0.6, label="template's weight change, 0.4865")
    ax.set_xlabel("size of the weight change |dW|")
    ax.set_ylabel("answer-word gain on the trained sentences,\nnats a token over the unchanged model")
    ax.set_title("A second seed gains on the answer words as the first did, and clears the control's bar", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    save(fig, "amend2_gain.png")


def carry(path: Path) -> set[int]:
    return set(json.loads(path.read_text())["carry"])


def overlap(ident: dict[str, Any]) -> None:
    e102 = HERE.parent / "qwen_carrying" / "results" / "states"
    gauss = {"real": RES / "neurons" / "gauss__real.json", "tofu": e102 / "gauss_tofu.json"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    labels = ["known 142\nin known 42", "fictional 142\nin fictional 42", "known 42\nin fict. 42", "known 42\nin fict. 142",
              "known 142\nin fict. 42", "known 142\nin fict. 142", "known 142 in\nGaussian"]
    for ax, rows in zip(axes, ("real", "tofu"), strict=True):
        r = ident[rows]
        g = len(carry(RES / "neurons_amend2" / f"known_s142_matched__{rows}.json") & carry(gauss[rows])) / 1024
        vals = [r["known"], r["fict"], *r["cross"], g]
        cols = [PURPLE, ORANGE, *[INK] * 4, GREY]
        for i, (v, c) in enumerate(zip(vals, cols, strict=True)):
            ax.bar(i, v, color=c, width=0.6, hatch="//" if 2 <= i <= 5 else None, fill=not 2 <= i <= 5, edgecolor=c)
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.axhline(r["cross_max"] + r["band"], color=INK, ls="--", lw=1.1,
                   label=f"highest cross-arm share plus the seed band, {r['cross_max'] + r['band']:.2f}")
        ax.axhline(r["fict"] - r["band"], color=ORANGE, ls=":", lw=1.3,
                   label=f"fictional seeds' share minus the band, {r['fict'] - r['band']:.2f}")
        ax.axhline(0.006, color=GREY, ls=":", lw=1, label="two random sets, about 0.006")
        ax.set_xticks(range(len(vals)), labels, fontsize=6.6)
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}: {r['verdict']}", fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
        style(ax)
    axes[0].set_ylabel("share of the first set's 1,024 neurons\ninside the second set")
    fig.suptitle("Each arm's two seeds share far more of their carrying neurons than any known-facts and fictional pair",
                 fontsize=10)
    save(fig, "amend2_overlap.png")


def pair_rules(neu1: dict[str, Any], neu2: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhspan(-1, 1, color=GREY, alpha=0.15, lw=0, label="inside the rule's band: level")
    for i, (rows, k) in enumerate(RULES):
        name = f"matched_{rows}_{k}"
        for dx, r, mfc, lab in ((-0.12, neu1["rules"][name], PURPLE, "known-facts seed 42"),
                                (0.12, neu2["pair_rules_s142"][name], "none", "known-facts seed 142")):
            ax.plot(i + dx, r["gap"] / r["band"], "o", color=PURPLE, mfc=mfc, mew=1.8, ms=9, label=lab if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(4), [f"{WORDS[k]}\n{WORDS[rows]}" for rows, k in RULES], fontsize=8)
    ax.set_ylim(-2, 2.2)
    ax.set_ylabel("known-facts state minus the fictional seeds' mean,\nin units of the rule's band")
    ax.set_title("The second known-facts seed also sits level on all four neuron readings", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=3)
    save(fig, "amend2_pair_rules.png")


def option_change(opt: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Mean over the 217 rows of the end state minus W0, per option kind: answer-word mean and end token."""
    out: dict[str, dict[str, list[float]]] = {"right": {"content": [], "end": []}, "wrong": {"content": [], "end": []}}
    for w, e in zip(opt["states"]["W0"]["items"], opt["states"]["a1_end"]["items"], strict=True):
        for k, (tw, te) in enumerate(zip(w["tokens"], e["tokens"], strict=True)):
            kind = out["right" if k == 0 else "wrong"]
            kind["content"].append(statistics.mean(te[:-1]) - statistics.mean(tw[:-1]))
            kind["end"].append(te[-1] - tw[-1])
    return {k: {p: statistics.mean(v) for p, v in d.items()} for k, d in out.items()}


def options(opt: dict[str, Any]) -> None:
    ch = option_change(opt)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, (kind, c) in enumerate((("right", PURPLE), ("wrong", GREY))):
        vals = [ch[kind]["content"], ch[kind]["end"]]
        xs = [i + (j - 0.5) * 0.36 for i in range(2)]
        ax.bar(xs, vals, width=0.34, color=c, label=f"{kind} option" + ("s, mean of three" if kind == "wrong" else ""))
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v - 1.4, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=1, label="no change from the unchanged model")
    ax.set_xticks(range(2), ["answer words, mean a token", "end-of-turn token after the option"], fontsize=8.5)
    ax.set_ylim(-30, 3)
    ax.set_ylabel("Amendment 1's end state minus the unchanged model,\nlog P in nats")
    ax.set_title("After training on sentences, ending the turn after a bare option costs about 25 nats", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    save(fig, "amend2_options.png")


def swapped_share(opt: dict[str, Any]) -> float:
    """The end state's share with each option's end token replaced by W0's, mean log P otherwise as the share reads it."""
    hits = []
    for w, e in zip(opt["states"]["W0"]["items"], opt["states"]["a1_end"]["items"], strict=True):
        means = [(sum(te[:-1]) + tw[-1]) / len(te) for tw, te in zip(w["tokens"], e["tokens"], strict=True)]
        hits.append(all(means[0] > m for m in means[1:]))
    return statistics.mean(hits)


def recall(opt: dict[str, Any]) -> None:
    w0, end = opt["states"]["W0"], opt["states"]["a1_end"]
    groups = [("multiple choice,\nend token included\n(the filed share)", w0["summary"]["share"], end["summary"]["share"]),
              ("multiple choice,\nanswer words only", w0["summary"]["content_share"], end["summary"]["content_share"]),
              ("multiple choice,\nunchanged model's\nend token", w0["summary"]["share"], swapped_share(opt)),
              ("greedy answer,\n'a few words'", w0["greedy"]["few_words"]["recall"], end["greedy"]["few_words"]["recall"]),
              ("greedy answer,\n'one sentence'", w0["greedy"]["sentence"]["recall"], end["greedy"]["sentence"]["recall"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for j, (lab, c) in enumerate((("unchanged model", GREY), ("Amendment 1's end state", PURPLE))):
        xs = [i + (j - 0.5) * 0.36 for i in range(len(groups))]
        vals = [g[1 + j] for g in groups]
        ax.bar(xs, vals, width=0.34, color=c, label=lab)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.25, color=GREY, ls="--", lw=1, label="multiple-choice chance, one of four")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of the 217 real facts: picked right,\nor reference words in the greedy answer")
    ax.set_title("The end state's drop is the end token: without it, or asked to answer, it gets the facts", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=3)
    save(fig, "amend2_recall.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    neu2, opt = js("neurons_amend2.json"), js("amend2_options.json")
    gain(js("amend1_train.json"), js("amend2_train.json"), js("amend2_ref.json")["reference"])
    overlap(neu2["identity"])
    pair_rules(js("neurons_amend1.json"), neu2)
    options(opt)
    recall(opt)
    print(f"wrote {sorted(p.name for p in FIG.glob('amend2_*.png'))}")


if __name__ == "__main__":
    main()
