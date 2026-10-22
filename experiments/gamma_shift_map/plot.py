"""Figures for gamma_shift_map (E30) from results/gamma_shift_map.json.

  pile.png    per update, each Pile reading of (a) s = 0, (b) s, (c) the null's s, (d) shuffled s and do-nothing as a
              ratio to W0, with W0 at 1 and the adversarial pass / fail lines 1.5 / (1 + f) and 1.5 (1 + f)
  topk.png    share of the whole s's trigger recovery kept by the top-k entries of |s|, per update, the 0.5 line (P5)
  split.png   s by layer and by sign: share of sum |s| beside the share of trigger recovery
  trigger.png trigger recovery with the change at every position, per arm, against the trigger-only figure
  adversarial.png  adversarial ratio (log) and sparsity ratio to W0 per arm and update, with the fail and cheating lines
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "gamma_shift_map.json"
FIG = HERE / "figures"
ARMS = (("a_zero", "s = 0", "#9e9e9e"), ("b_shift", "shifted (s)", "#1f77b4"), ("c_null", "' http' s", "#ff7f0e"),
        ("d_shuffled", "s shuffled", "#8c564b"), ("nothing", "do-nothing", "#2ca02c"))
METRICS = (("ci_masked_kl", "CI-masked"), ("stochastic_kl", "stochastic"), ("pgd_kl", "adversarial"),
           ("unmasked_kl", "unmasked"), ("sparsity", "sparsity"))


def short(label: str) -> str:
    return label.replace("emoticon/", "emoticon ").replace("r1_t534_v369/", "r1_t534 ").replace("_", " ")


def pile(d: dict, f: float) -> None:
    rat = d["score"]["ratios"]
    fig, axes = plt.subplots(len(rat), 1, figsize=(9, 3.2 * len(rat)), squeeze=False)
    for ax, (label, r) in zip(axes[:, 0], rat.items(), strict=True):
        w = 0.16
        for i, (arm, name, col) in enumerate(ARMS):
            if arm in r:
                ax.bar([j + (i - 2) * w for j in range(len(METRICS))], [r[arm][m] for m, _ in METRICS], w, color=col,
                       label=name)
        ax.axhline(1.0, color="k", lw=0.8)
        for y in (1.5 / (1 + f), 1.5 * (1 + f)):
            ax.axhline(y, color="r", ls=":", lw=0.8)
        ax.set_xticks(range(len(METRICS)), [n for _, n in METRICS])
        ax.set_ylabel("ratio to W0")
        ax.set_title(short(label), fontsize=10)
    axes[0, 0].legend(fontsize=7, ncol=5, loc="upper left")
    fig.suptitle("D' on the Pile against the fine-tuned model (red: adversarial pass / fail lines)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pile.png", dpi=130)


def topk(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, u in d["updates"].items():
        if "readout" not in u:
            continue
        sh = u["readout"]["shift"]["topk_share_of_whole"]
        ks = sorted(sh, key=int)
        ax.plot([int(k) for k in ks], [sh[k] for k in ks], "o-", label=short(label))
    ax.axhline(0.5, color="r", ls=":", lw=0.8)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("k (largest entries of |s| kept, rest zero)")
    ax.set_ylabel("share of the whole s's trigger recovery")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "topk.png", dpi=130)


def split(d: dict) -> None:
    ups = {k: u["readout"]["shift"] for k, u in d["updates"].items() if "readout" in u}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    w = 0.8 / max(len(ups), 1)
    for i, (label, ro) in enumerate(ups.items()):
        whole = ro["R_whole"]
        lay = [str(x) for x in range(4)]
        axes[0].bar([j + i * w for j in range(4)], [ro["mass"]["by_layer"].get(x, 0.0) for x in lay], w * 0.45,
                    color=f"C{i}", label=f"{short(label)}: sum |s|")
        axes[0].bar([j + i * w + w * 0.45 for j in range(4)], [ro["R_by_layer"][x] / whole for x in lay], w * 0.45,
                    color=f"C{i}", alpha=0.45, hatch="//", label=f"{short(label)}: recovery")
        halves = (ro["mass"]["by_sign"]["up"], ro["R_halves"]["up"] / whole, ro["mass"]["by_sign"]["down"],
                  ro["R_halves"]["down"] / whole)
        axes[1].bar([j + i * w for j in range(4)], halves, w * 0.9, color=f"C{i}", label=short(label))
    axes[0].set_xticks(range(4), [f"layer {x}" for x in range(4)])
    axes[0].set_ylabel("share of the whole s")
    axes[0].legend(fontsize=6)
    axes[1].set_xticks(range(4), ["up: sum |s|", "up: recovery", "down: sum |s|", "down: recovery"], fontsize=8)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "split.png", dpi=130)


def trigger(d: dict) -> None:
    labels = list(d["updates"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.2
    for i, (arm, name, col) in enumerate(ARMS[:4]):
        ax.bar([j + (i - 1.5) * w for j in range(len(labels))], [d["updates"][k]["R_everywhere"][arm] for k in labels],
               w, color=col, label=name)
    for j, k in enumerate(labels):
        ref = d["updates"][k]["G1"]["ref"]["R_shared"]
        ax.plot([j - 2 * w, j + 2 * w], [ref, ref], "k--", lw=1, label="s at the trigger only (earlier run)" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(labels)), [short(k) for k in labels])
    ax.set_ylabel("trigger recovery, share of own values")
    ax.set_title("Does the fixed change, applied everywhere, still reproduce the trigger behaviour?", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "trigger.png", dpi=130)


def adversarial(d: dict, f: float) -> None:
    rat = d["score"]["ratios"]
    labels = list(rat)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.16
    for metric, ax in zip(("pgd_kl", "sparsity"), axes, strict=True):
        for i, (arm, name, col) in enumerate(ARMS):
            ax.bar([j + (i - 2) * w for j in range(len(labels))], [rat[k][arm][metric] for k in labels], w, color=col,
                   label=name)
        ax.axhline(1, color="k", lw=0.8, label="W0 (unchanged model)")
        ax.set_xticks(range(len(labels)), [short(k) for k in labels], fontsize=8)
    axes[0].axhline(1.5 * (1 + f), color="r", ls=":", lw=1, label="fail line")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("adversarial KL on the Pile, ratio to W0")
    axes[1].axhline(1.05, color="r", ls=":", lw=1, label="cheating line 1.05")
    axes[1].set_ylabel("sparsity loss, ratio to W0")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("Is the shifted decomposition still faithful on the Pile, and what does it cost in sparsity?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "adversarial.png", dpi=130)


def main() -> None:
    d = json.loads(OUT.read_text())
    FIG.mkdir(exist_ok=True)
    topk(d)
    split(d)
    trigger(d)
    if "score" in d:
        f = d["score"]["floors"]["pgd_f"]
        pile(d, f)
        adversarial(d, f)
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
