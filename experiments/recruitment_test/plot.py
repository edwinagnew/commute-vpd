"""Figures for recruitment_test (E38) from the results JSONs only (no model).

  recruitment.png  stage 1: removed share of the gain when the recruited set is taken out of the fine-tuned weights,
                   per task-seed, against 20 random same-size sets, the ' http' fine-tune's recruited set, and the
                   filed bars (0.5 pass, 0.2 fail); the busy set's share, off the scale, as labelled arrows.
  w0.png           stage 1, P3: the same removal from the original weights W0 against the filed band (|share| <= 0.1),
                   beside the removal from the fine-tuned weights; the difference is exploratory.
  offtarget.png    stage 1: off-target KL at the 40 non-trigger positions for each removal, log scale, against the
                   fine-tune's own off-target KL and the filed P2 bar (3 x the fine-tune's own).
  topk.png         stage 1: removed share and off-target KL when the k pieces with the largest rise are removed.
  routing.png      stage 2: cosine of the fine-tune's mean input change at the trigger with the target's natural
                   direction, per distinct matrix input (q, k, v share one input), seeds 0 and 1 and the ' http' null.
  census.png       stage 3: how often each recruited piece is on in Pile text, against the five random sets.

Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/recruitment_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, VIOLET, AQUA = (
    "#2a78d6",
    "#eb6834",
    "#6250d6",
    "#1baf7a",
)  # S, null set, busy set, W0 removal
TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
NAME = {
    "emoticon": "emoticon",
    "r1_t534_v369": "' which' → ' was'",
    "r2_t359_v816": "' we' → ' just'",
}
SEEDS = ("0", "1")


def load(k: int) -> dict[str, Any]:
    return json.loads((RES / f"stage{k}.json").read_text())


def style(ax: Any, zero: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color=MUTED, lw=0.8)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def keys() -> list[str]:
    return [f"{t}/s{s}" for t in TASKS for s in SEEDS]


def xlabels() -> list[str]:
    return [f"{NAME[k.split('/')[0]]}\nseed {k[-1]}" for k in keys()]


def fig_recruitment(st: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    top = 0.72
    for i, k in enumerate(ks):
        r = st[k]
        rr = np.array(r["random"]["rho"])
        ax.scatter(
            i - 0.18 + np.linspace(-0.08, 0.08, len(rr)),
            rr,
            s=9,
            color=MUTED,
            zorder=3,
            label="20 random sets of the same size" if i == 0 else None,
        )
        ax.plot(
            [i - 0.3, i - 0.06],
            [rr.mean()] * 2,
            color=INK,
            lw=1.4,
            label="random sets' mean" if i == 0 else None,
        )
        ax.bar(
            i + 0.05,
            r["rho_S"],
            0.22,
            color=BLUE,
            label="recruited set" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.27],
            [r["rho_null"]],
            marker="D",
            s=30,
            color=ORANGE,
            zorder=4,
            label="the ' http' fine-tune's recruited set" if i == 0 else None,
        )
        ax.annotate(
            f"busy set\n{r['rho']['busy']:.1f} ↑",
            (i + 0.05, top - 0.01),
            ha="center",
            va="top",
            fontsize=7,
            color=VIOLET,
        )
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(
        len(ks) - 0.5, 0.5, "filed pass bar 0.5", ha="right", va="bottom", fontsize=7
    )
    ax.axhline(0.2, color=INK, ls=":", lw=1)
    ax.text(
        len(ks) - 0.5, 0.2, "filed fail bar 0.2", ha="right", va="bottom", fontsize=7
    )
    ax.set_xticks(range(len(ks)), xlabels(), fontsize=8)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylim(-0.3, top)
    ax.set_ylabel(
        "share of the fine-tune's gain removed\n(0 = no effect, 1 = fine-tune undone)"
    )
    ax.set_title(
        "Does removing the recruited subcomponents from the fine-tuned weights undo the fine-tune?",
        fontsize=10,
    )
    style(ax)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
    )
    save(fig, "recruitment.png")


def fig_w0(st: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axhspan(
        -0.1,
        0.1,
        color=GRID,
        alpha=0.7,
        lw=0,
        label="filed expectation at W0: within ±0.1",
    )
    for i, k in enumerate(ks):
        r = st[k]
        ax.plot(
            [i - 0.12, i + 0.12],
            [r["rho_S_W0"], r["rho_S"]],
            color=MUTED,
            lw=1,
            zorder=2,
        )
        ax.scatter(
            [i - 0.12],
            [r["rho_S_W0"]],
            s=45,
            facecolor="white",
            edgecolor=AQUA,
            lw=2,
            zorder=3,
            label="removed from the original weights W0" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.12],
            [r["rho_S"]],
            s=45,
            color=BLUE,
            zorder=3,
            label="removed from the fine-tuned weights" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.12],
            [r["random_mean"]],
            s=25,
            marker="_",
            color=INK,
            lw=1.5,
            zorder=3,
            label="random sets' mean, fine-tuned weights" if i == 0 else None,
        )
        d = r["rho_S"] - r["rho_S_W0"]
        ax.text(i, 0.63, f"difference\n{d:+.2f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(ks)), xlabels(), fontsize=8)
    ax.set_ylim(-0.15, 0.72)
    ax.set_ylabel(
        "fall in the target's log-odds at the trigger\nas a share of the fine-tune's gain"
    )
    ax.set_title(
        "Is the recruited set inert in the original model? Removing it there already lowers the target",
        fontsize=10,
    )
    style(ax)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
    )
    save(fig, "w0.png")


def fig_offtarget(st: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, k in enumerate(ks):
        r = st[k]
        kr = np.array(r["random"]["kl_other"])
        ax.plot(
            [i - 0.38, i + 0.38],
            [r["kl_ft"]] * 2,
            color=INK,
            lw=1.4,
            label="the fine-tune's own off-target KL to W0" if i == 0 else None,
        )
        ax.plot(
            [i - 0.38, i + 0.38],
            [3 * r["kl_ft"]] * 2,
            color=INK,
            lw=1,
            ls="--",
            label="filed bar for the recruited set: 3 × the fine-tune's own"
            if i == 0
            else None,
        )
        ax.scatter(
            i - 0.24 + np.linspace(-0.06, 0.06, len(kr)),
            kr,
            s=8,
            color=MUTED,
            zorder=3,
            label="20 random sets of the same size" if i == 0 else None,
        )
        ax.scatter(
            [i - 0.06],
            [r["kl_other"]["S"]],
            s=45,
            color=BLUE,
            zorder=4,
            label="recruited set, from the fine-tuned weights" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.06],
            [r["kl_other_S_W0"]],
            s=45,
            facecolor="white",
            edgecolor=AQUA,
            lw=2,
            zorder=4,
            label="recruited set, from W0" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.18],
            [r["kl_other"]["null"]],
            s=30,
            marker="D",
            color=ORANGE,
            zorder=4,
            label="the ' http' fine-tune's recruited set" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.3],
            [r["kl_other"]["busy"]],
            s=40,
            marker="s",
            color=VIOLET,
            zorder=4,
            label="busy set" if i == 0 else None,
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)), xlabels(), fontsize=8)
    ax.set_ylabel(
        "KL at the 40 non-trigger positions, nats\n(to the model before the removal)"
    )
    ax.set_title(
        "Is removing the recruited set confined to the trigger? Off-target damage of each removal",
        fontsize=10,
    )
    style(ax, zero=False)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.17),
    )
    save(fig, "offtarget.png")


def fig_topk(st: dict[str, Any]) -> None:
    kk = [25, 50, 100, 200, 400]
    colors = dict(zip(TASKS, (BLUE, ORANGE, AQUA), strict=True))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for t in TASKS:
        for s, ls in zip(SEEDS, ("-", "--"), strict=True):
            r = st[f"{t}/s{s}"]
            lab = f"{NAME[t]}, seed {s}"
            a1.plot(
                kk,
                [r["rho"][f"top{k}"] for k in kk],
                ls=ls,
                marker="o",
                ms=4,
                color=colors[t],
                lw=1.6,
                label=lab,
            )
            a2.plot(
                kk,
                [r["kl_other"][f"top{k}"] for k in kk],
                ls=ls,
                marker="o",
                ms=4,
                color=colors[t],
                lw=1.6,
                label=lab,
            )
            a1.scatter(
                [r["n"]["S"]],
                [r["rho_S"]],
                marker="*",
                s=90,
                color=colors[t],
                edgecolor=INK,
                lw=0.5,
                zorder=4,
            )
        a2.axhline(
            np.mean([st[f"{t}/s{s}"]["kl_ft"] for s in SEEDS]),
            color=colors[t],
            lw=0.8,
            ls=":",
        )
    a1.axhline(1, color=INK, lw=0.8, ls="--")
    a1.text(26, 1.02, "fine-tune undone", fontsize=7)
    a1.text(26, 1.75, "★ = the recruited set at its own size", fontsize=7)
    a2.text(26, 0.12, "dotted: each task's own fine-tune, KL to W0", fontsize=7)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks(kk, [str(k) for k in kk])
        a.set_xlabel("k, pieces removed (largest rise in trigger importance first)")
        style(a, zero=a is a1)
    a2.set_yscale("log")
    a1.set_ylabel("share of the fine-tune's gain removed")
    a2.set_ylabel("off-target KL, nats")
    a1.set_title("Target removed", fontsize=9)
    a2.set_title("Off-target damage", fontsize=9)
    a2.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "Does removing more of the most-risen pieces undo more of the fine-tune, and at what off-target cost?",
        fontsize=10,
    )
    save(fig, "topk.png")


def inputs() -> list[tuple[str, str]]:
    """Distinct matrix inputs in forward order: q, k and v share one; layer 0's attention input is the embedding."""
    out = []
    for layer in range(4):
        if layer:
            out.append((f"h.{layer}.attn.q_proj", f"L{layer} attn in"))
        out += [
            (f"h.{layer}.attn.o_proj", f"L{layer} attn out"),
            (f"h.{layer}.mlp.c_fc", f"L{layer} MLP in"),
            (f"h.{layer}.mlp.down_proj", f"L{layer} MLP hidden"),
        ]
    return out


def fig_routing(s2: dict[str, Any]) -> None:
    mats = inputs()
    x = np.arange(len(mats))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    for ax, t in zip(axes, TASKS, strict=True):
        rec = s2["tasks"][t]
        ax.axhline(0.3, color=INK, ls="--", lw=1)
        rc = [rec["seeds"]["0"]["trace"][m]["random_cos"] for m, _ in mats]
        ax.fill_between(
            x,
            -np.array(rc),
            rc,
            color=GRID,
            step="mid",
            label="random direction, ±1/√d",
        )
        ax.axvspan(len(mats) - 4.5, len(mats) - 0.5, color=BLUE, alpha=0.06, lw=0)
        for s, mk in zip(SEEDS, ("o", "s"), strict=True):
            ax.scatter(
                x,
                [rec["seeds"][s]["trace"][m]["cos"] for m, _ in mats],
                marker=mk,
                s=28,
                color=BLUE,
                facecolor=BLUE if s == "0" else "white",
                zorder=3,
                label=f"fine-tune to the target, seed {s}",
            )
        ax.scatter(
            x,
            [rec["null"][m]["cos"] for m, _ in mats],
            marker="D",
            s=24,
            color=ORANGE,
            zorder=3,
            label="' http' fine-tune (null)",
        )
        verdict = {
            True: "filed rule met",
            False: "filed rule missed",
            "unscored": "not scored",
        }[rec["P4_pass"]]
        ax.set_title(
            f"{NAME[t]}: {rec['n_positions']} Pile positions before the target; {verdict}",
            fontsize=9,
        )
        style(ax)
    axes[0].text(
        len(mats) - 2.5,
        0.62,
        "layer 3: most of the\nrecruited set",
        ha="center",
        fontsize=7,
        color=BLUE,
    )
    axes[0].text(-0.4, 0.32, "filed bar 0.3", fontsize=7)
    axes[1].set_ylabel(
        "cosine of the mean input change at the trigger\nwith the target's natural direction"
    )
    axes[2].set_xticks(x, [lab for _, lab in mats], rotation=45, ha="right", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=4)
    axes[0].set_ylim(-0.65, 0.75)
    fig.suptitle(
        "Does the fine-tune move each matrix's input at the trigger toward where the original model predicts "
        "the target?",
        fontsize=10,
    )
    save(fig, "routing.png")


def fig_census(s3: dict[str, Any], big: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rng = np.random.default_rng(0)
    for i, t in enumerate(TASKS):
        s = np.array([p["on_rate"] for p in s3["tasks"][t]["pieces_S"]])
        r = np.array([p["on_rate"] for sel in big[t] for p in sel])
        r = r[r > 0]
        ax.scatter(
            np.log10(r),
            i + 0.18 + rng.uniform(-0.1, 0.1, len(r)),
            s=5,
            color=MUTED,
            alpha=0.5,
            label="five random sets of the same size" if i == 0 else None,
        )
        ax.scatter(
            np.log10(s),
            i - 0.18 + rng.uniform(-0.1, 0.1, len(s)),
            s=9,
            color=BLUE,
            label="recruited set, seed 0" if i == 0 else None,
        )
        for y, v, c in ((i - 0.18, np.median(s), BLUE), (i + 0.18, np.median(r), INK)):
            ax.plot([np.log10(v)] * 2, [y - 0.15, y + 0.15], color=c, lw=2)
            ax.text(
                np.log10(v),
                y + 0.17,
                f"median {100 * v:.1f}%",
                fontsize=7,
                ha="center",
                color=c,
            )
    ax.set_yticks(
        range(len(TASKS)),
        [f"{NAME[t]}\n{len(s3['tasks'][t]['pieces_S'])} pieces" for t in TASKS],
        fontsize=8,
    )
    tick = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    ax.set_xticks(np.log10(tick), [f"{100 * v:g}%" for v in tick])
    ax.set_xlabel(
        "share of Pile positions where the piece's causal importance exceeds 0.5 (original model)"
    )
    ax.set_title(
        "Are the recruited pieces rare specialists? How often each is on in ordinary text",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, "census.png")


def main() -> None:
    s1, s2, s3 = load(1), load(2), load(3)
    big = json.loads((RES / "large" / "stage3_pieces.json").read_text())
    fig_recruitment(s1["states"])
    fig_w0(s1["states"])
    fig_offtarget(s1["states"])
    fig_topk(s1["states"])
    fig_routing(s2)
    fig_census(s3, big)
    print(f"wrote {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
