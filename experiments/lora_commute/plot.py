"""Figures for lora_commute from results/lora_commute.json (or the smoke file with --smoke).

  route_split.png   one question: which route does each arm's trigger gain take? Stacked d / r / i / e_gate / e_delta
                    per arm: LoRA_r do-nothing and lifted (scored step), ft do-nothing and lifted (step 10), pb_r,
                    pb_full, pb_all (step 30).
  vs_rank.png       does commuting get harder with rank? Left: direct share d of lift_r and pb_r against rank, with
                    lift_ft, pb_full and pb_all as horizontal references. Right: Pile PGD ratio of the same arms, the
                    1.5 line and the W0 floor.

Usage (repo root):  uv run python experiments/lora_commute/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SHARES = ("d", "r", "i", "e_gate", "e_delta")
COLORS = {"d": "#1f77b4", "r": "#ff7f0e", "i": "#bcbd22", "e_gate": "#7f7f7f", "e_delta": "#d62728"}


def load(smoke: bool) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"lora_commute{'_smoke' if smoke else ''}.json").read_text())


def scored_step(arm: dict[str, Any], sizes: dict[str, Any]) -> str:
    return str(arm["scored_step"] or sizes["lora_ckpts"][-1])


def arms(out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """name -> measurement record (the one each prediction reads)."""
    S = out["sizes"]
    rows: dict[str, dict[str, Any]] = {}
    ft = out["ft"]["steps"][str(S["ft_ckpts"][0])]
    rows["ft nothing"], rows["ft lift"] = ft["nothing"], ft["lift"]
    for r in S["ranks"]:
        a = out["lora"][str(r)]
        st = a["steps"][scored_step(a, S)]
        rows[f"LoRA_{r} nothing"], rows[f"lift_{r}"] = st["nothing"], st["lift"]
    last = str(S["pb_steps"])
    for name, a in out["pb"].items():
        rows[name] = a["steps"][last]
    return rows


def route_split(out: dict[str, Any], path: Path) -> None:
    rows = arms(out)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows)), 4.5))
    x = range(len(rows))
    pos = [0.0] * len(rows)
    neg = [0.0] * len(rows)
    for s in SHARES:
        vals = [rows[n]["routes"]["shares"][s] for n in rows]
        bottoms = [pos[i] if v >= 0 else neg[i] for i, v in enumerate(vals)]
        ax.bar(x, vals, bottom=bottoms, color=COLORS[s], label=s, width=0.7)
        for i, v in enumerate(vals):
            if v >= 0:
                pos[i] += v
            else:
                neg[i] += v
    ax.axhline(1, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(rows), rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("share of the trigger log-odds gain")
    ax.set_title("Which route does each arm's gain take at the trigger? (held-out emoticon windows)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def vs_rank(out: dict[str, Any], path: Path) -> None:
    S = out["sizes"]
    rows = arms(out)
    ranks = S["ranks"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(ranks, [rows[f"lift_{r}"]["routes"]["shares"]["d"] for r in ranks], "o-", label="lift_r (LoRA lifted)")
    ax.plot(ranks, [rows[f"pb_r{r}"]["routes"]["shares"]["d"] for r in ranks], "s-", label="pb_r (rank-r pull-back)")
    for name, ls in (("ft lift", "--"), ("pb_full", ":"), ("pb_all", "-.")):
        ax.axhline(rows[name]["routes"]["shares"]["d"], ls=ls, color="k", lw=1, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank r")
    ax.set_ylabel("direct share d")
    ax.set_title("Does the direct share fall with rank?")
    ax.legend(fontsize=8)
    ax = axes[1]
    have = [r for r in ranks if "pile" in rows[f"lift_{r}"]]
    if have:
        ax.plot(have, [rows[f"lift_{r}"]["pile"]["pgd_kl"] for r in have], "o-", label="lift_r")
    have_pb = [r for r in ranks if "pile" in rows[f"pb_r{r}"]]
    if have_pb:
        ax.plot(have_pb, [rows[f"pb_r{r}"]["pile"]["pgd_kl"] for r in have_pb], "s-", label="pb_r")
    for name, ls in (("ft lift", "--"), ("ft nothing", (0, (1, 1))), ("pb_full", ":"), ("pb_all", "-.")):
        if "pile" in rows[name]:
            ax.axhline(rows[name]["pile"]["pgd_kl"], ls=ls, color="k", lw=1, label=name)
    ax.axhline(1.5, color="r", lw=0.8, label="1.5 (fails)")
    if out["W0"]["pgd_floor"] is not None:
        ax.axhline(1 + 3 * out["W0"]["pgd_floor"], color="g", lw=0.8, label="W0 + 3 floors")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank r")
    ax.set_ylabel("Pile PGD KL ratio to W0")
    ax.set_title("Is the lifted decomposition adversarially fragile?")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = load(args.smoke)
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    route_split(out, figs / f"route_split{suffix}.png")
    vs_rank(out, figs / f"vs_rank{suffix}.png")
    print(f"wrote {figs}")


if __name__ == "__main__":
    main()
