"""E258 and E259 from their summary.json files and E258's saved members, $0, no GPU:
  figures/behaviour_vs_write.png  the behavioural change-weighted mean b_mean beside the write share E_b, per item;
  results/member_check.json       where E258's b_mean comes from: per built item, the write-change share of members
                                  whose stand-in at the fitted gain reads worse than removing their change (q > q1), the
                                  five largest members (write-change share, gain, q, q1, 1 - q/q1, own write share
                                  explained) and the member that moves b_mean most.

    ~/commute-agents/fleet/cpu-run --mem 6G python3 experiments/rb_member_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from commute import figures as FG
from commute.jobs import EXPERIMENTS, load_experiment

HERE = Path(__file__).resolve().parent
BUILT = {"x2": "gain x2", "x10": "gain x10", "rebuilt": "rebuilt", "new_few": "new,\nfew parts", "new_thin": "new,\nthin",
         "random_size_241001": "random\n(size)", "random_kl_241002": "random\n(KL)"}  # fmt: skip
SIDES = {"e62/copy": "e62\ncopy", "e62-lookup": "e62\nlookup", "e58/copy": "e58\ncopy", "e58-prior": "e58 prior\ncipher",
         "e53/lower": "e53\nlowercase", "e53-": "e53\nciphers", "random_size": "random\n(size)"}  # fmt: skip
BEHAVIOUR, WRITE = "steering", "regating"  # marker styles only: behaviour orange diamonds, write blue circles


def _row(cols: dict[str, np.ndarray], u: int) -> dict[str, float]:
    return {"member": int(u)} | {k: float(v[u]) for k, v in cols.items()}


def member_check() -> dict[str, Any]:
    run = load_experiment("rb_member_test", "run")
    floor = json.loads((run.RESULTS / "summary.json").read_text())["floor"]
    out: dict[str, Any] = {}
    for item in BUILT:
        a, K = run.saved(item)
        chK = a["changed"].astype(bool) & K
        q, q1, read = np.zeros(len(chK)), np.zeros(len(chK)), np.zeros(len(chK), dtype=bool)
        for i in range(4):
            p = dict(np.load(run.shard_file(item, (i, 4))))
            q[p["members"]], q1[p["members"]], read[p["members"]] = p["q"], p["q1"], True
        cnt = chK & ~(read & (q1 <= floor))  # run.score_item's counted members
        dd, n, m = a["dd"], a["num"], a["den"]
        tot = float(dd[cnt].sum())
        b = np.where(cnt & read, 1 - q / np.where(q1 > 0, q1, 1.0), 0.0)
        own = np.where((m > 0) & (n > 0), (n - m) ** 2 / np.where(m > 0, m, 1.0) / np.where(dd > 0, dd, 1.0), 0.0)
        contrib = np.where(cnt, b * dd / tot, 0.0)

        cols = {"dd_share": dd / tot, "gain": a["gain"], "q": q, "q1": q1, "b": b, "own_write_share": own, "b_mean_part": contrib}
        top = np.argsort(-np.where(cnt, dd, -1.0))[:5]
        out[item] = {"b_mean": float(contrib.sum()), "dd_share_q_above_q1": float(dd[cnt & read & (q > q1)].sum() / tot),
                     "b_mean_top5": float(contrib[top].sum()), "median_abs_gain_minus_1": float(np.median(np.abs(a["gain"][cnt] - 1))),
                     "median_rel_q_q1": float(np.median(np.abs(q - q1)[cnt & read] / q1[cnt & read])),
                     "top5": [_row(cols, u) for u in top], "most_negative": _row(cols, int(np.argmin(contrib)))}  # fmt: skip
    (HERE / "results" / "member_check.json").write_text(json.dumps(out, indent=1))
    return out


def figure() -> None:
    built = json.loads((HERE / "results" / "summary.json").read_text())["items"]
    pairs = json.loads((EXPERIMENTS / "rb_member_pairs" / "results" / "summary.json").read_text())["items"]
    rows = [{"panel": "built", "item": i, "side": BUILT[i], "b_mean": built[i]["b_mean"], "b_mean_se": 0.0, "E_b": built[i]["E_b"]} for i in BUILT]
    rows += [{"panel": "pairs", "item": i, "side": next(v for k, v in SIDES.items() if i.startswith(k)), "b_mean": r["b_mean"],
              "b_mean_se": r["b_mean_se"], "E_b": r["E_b"]} for i, r in pairs.items()]  # fmt: skip
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    for ax, panel, names in ((axes[0], "built", list(BUILT.values())), (axes[1], "pairs", list(SIDES.values()))):
        mine = [r for r in rows if r["panel"] == panel]
        for key, method, dx in (("E_b", WRITE, -0.15), ("b_mean", BEHAVIOUR, 0.15)):
            st = FG.METHODS[method]
            xs = [names.index(r["side"]) + dx for r in mine]
            label = "write share $E_b$" if key == "E_b" else "behaviour, change-weighted $b_{mean}$"
            if key == "b_mean" and panel == "pairs":
                ax.errorbar(xs, [r[key] for r in mine], yerr=[2 * r["b_mean_se"] for r in mine], fmt="none", ecolor=st.color, elinewidth=0.6, alpha=0.6)
            ax.scatter(xs, [r[key] for r in mine], color=st.color, marker=st.marker, s=20, alpha=0.85, linewidths=0, label=label)
        ax.axhline(built["random_size_241001"]["E_b"], color="#555555", linestyle=":", linewidth=0.8, label="random update, write share")
        ax.set_xticks(range(len(names)), names, fontsize=6)
        ax.set_xlabel("built item" if panel == "built" else "task side of a trained pair", fontsize=7)
        ax.set_ylim(-0.6, 0.85)
        ax.set_title("built edits, every member read" if panel == "built" else "trained pairs, 2,000 members sampled per seed", fontsize=8)
    axes[0].set_ylabel("share of the change explained by a gain")
    axes[1].sharey(axes[0])
    axes[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Read one part at a time, behaviour misses the gain edits that the write share finds", fontsize=9)
    FG.save(fig, HERE / "figures" / "behaviour_vs_write.png",
            n="one reading per item over all of D, VPD featurization, 67M; pairs: one point per seed, bars 2 se of the sample", data=rows)  # fmt: skip


if __name__ == "__main__":
    member_check()
    figure()
