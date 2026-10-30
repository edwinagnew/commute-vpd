"""Figures and report tables for E55 Stage C1 and Amendment 5b, read from results/c1.json, amend5b.json and amend5.json
only; no model. Analysis: reports/analyses/2026-10-30_analysis_e55-c1.md.

  c1_scores.png    agreement with the truth per instrument and per nuisance labeller, on every state and on one vote per
                   training run, with what each instrument needed (the binomial bar and the better nuisance labeller +
                   margin); steering without the two unreadable R votes (G < 0.01) as a hollow marker, post hoc
  c1_prune.png     per state, the largest count-a gain from pruning the top-K units over the random draws, by truth class,
                   with the 3 sd the filed rule needed; the LOW control's accuracy loss (importance pruning) beside it
  c1_reverse.png   per state, reverse fine-tuning steps to 95% count a, one panel per reverse lr, with scratch's steps and
                   the filed half-of-scratch line
  c1_steer.png     per state, the steering gain share S against fine-tuning steps, by truth class, the filed 0.5 line;
                   R's states (G < 0.01) marked unreadable
  amend5b_saves.png  Amendment 5b: the two count-a checks by save for the fast natural-row seeds 1-3 beside seed 0
                   (Amendment 5), LOW first, Stage A's bars

Also prints the report's per-state table and the post-hoc steering score without the unreadable votes.

Usage (repo root): PYTHONPATH=experiments/pcfg_anchors ~/commute-agents/fleet/cpu-run .venv/bin/python \
    experiments/pcfg_anchors/plot_c1.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from c1 import load_state, nuisance_labels, per_run
from plot import FIG, INK, MUTED, finish, load, plt

from commute import revival

EX_C, NEW_C, B_C, CTL_C = "#2e9e5b", "#2a78d6", "#8a8984", "#eb6834"  # truth: existing, new, class B; LOW control
TRUTH_C = {"existing": EX_C, "new": NEW_C, "B": B_C}
TRUTH_W = {"existing": "truth: reuses the old count", "new": "truth: new count", "B": "class B (unscored)"}
G_MIN = 0.01  # lead_decisions (c): a steering vote with G below this is unreadable (rule unchanged)


def prune_gain(p: dict[str, Any]) -> tuple[float, float, int]:
    """(gain, 3 sd, K) at the K with the larger gain over sd: top-K accuracy minus the random mean (revival_z)."""
    zs = {k: revival.revival_z(p[str(k)]["top"], p[str(k)]["random"], p["n_eval"]) for k in revival.PRUNE_K}
    k = max(zs, key=lambda q: zs[q][0] / zs[q][1])
    return zs[k][0], revival.SD_MULT * zs[k][1], k


def truth_of(c: dict[str, Any], sid: str) -> str:
    t = c["truth"]
    return t["scored"].get(sid) or t["steering_only"].get(sid) or "B"


def jitter(n: int) -> np.ndarray:
    return np.linspace(-0.25, 0.25, n) if n > 1 else np.zeros(1)


def fig_scores(c: dict[str, Any], post: dict[str, dict[str, Any]], out: str = "c1_scores.png") -> None:
    rows = [("pruning revival", "prune"), ("reverse fine-tuning", "reverse"), ("steering", "steer"),
            ("nuisance: fine-tuned <= 1K steps", "nuisance_short"), ("nuisance: fine-tune lr <= 1e-5", "nuisance_slow")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, which, words in zip(axes, ("all", "per_run"), ("every truth state (margin 2)", "one vote per training run (margin 1)"), strict=True):
        for i, (name, key) in enumerate(rows):
            s = c["scores"][key][which]
            inst = not key.startswith("nuisance")
            ax.barh(i, s["agree"] / s["n"], color=CTL_C if inst else MUTED, alpha=0.85 if inst else 0.5, height=0.6)
            ax.text(s["agree"] / s["n"] + 0.01, i, f"{s['agree']}/{s['n']}", va="center", fontsize=8)
            if inst:
                need = max(s["bar"], max(s["nuisance_agree"].values()) + s["margin"])
                ax.plot([s["bar"] / s["n"]] * 2, [i - 0.38, i + 0.38], color=INK, lw=1.6,
                        label="binomial bar" if i == 0 and which == "all" else None)  # fmt: skip
                ax.plot([need / s["n"]] * 2, [i - 0.38, i + 0.38], color="#c0392b", lw=2.2,
                        label="needed: better nuisance labeller + margin" if i == 0 and which == "all" else None)  # fmt: skip
        p = post[which]
        p_need = max(p["bar"], max(p["nuisance_agree"].values()) + p["margin"])
        ax.scatter(p["agree"] / p["n"], 2.45, s=60, facecolor="white", edgecolor=INK, zorder=5, clip_on=False,
                   label="steering without R's unreadable votes (post hoc, not a score)" if which == "all" else None)  # fmt: skip
        ax.text(p["agree"] / p["n"] + 0.015, 2.45, f"{p['agree']}/{p['n']} without R, needs {p_need}", va="center", fontsize=7)
        ax.axvline(0.5, color=MUTED, ls=":", lw=0.9)
        ax.text(0.5, 4.45, "chance", fontsize=7.5, color=MUTED, ha="center")
        ax.set_xlim(0.4, 1.0)
        ax.set_title(words, fontsize=9.5, loc="left")
        ax.set_xlabel("share of states whose label agrees with the truth")
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=3)
    finish(fig, out, "On every state, no instrument beats the better nuisance labeller by the filed margin;\n"
           "on one vote per run, pruning and steering do", bottom=0.2)  # fmt: skip


def fig_prune(c: dict[str, Any], out: str = "c1_prune.png") -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    groups = {"existing": [], "new": [], "B": []}
    for sid, r in c["states"].items():
        if "prune" in r:
            groups[truth_of(c, sid)].append(prune_gain(r["prune"]))
    for x, (t, vals) in enumerate(groups.items()):
        vals.sort()
        xs = x + jitter(len(vals))
        ax.scatter(xs, [v[0] for v in vals], color=TRUTH_C[t], s=34, zorder=3, label=TRUTH_W[t])
        ax.scatter(xs, [v[1] for v in vals], marker="_", color=INK, s=60, zorder=4,
                   label="3 sd the filed rule needed (gain above it -> 'existing')" if x == 0 else None)  # fmt: skip
    ctl = c["controls"]["prune_LOW_importance"]
    loss = [float(np.mean(ctl[str(k)]["random"]) - ctl[str(k)]["top"]) for k in revival.PRUNE_K]
    ax.scatter([3 - 0.1, 3 + 0.1], loss, marker="D", color=CTL_C, s=40, zorder=3,
               label="LOW control: count-a accuracy lost by importance pruning, K = 5, 15")  # fmt: skip
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.text(3.45, 0.01, "random draws", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(range(4), ["reuses the old count\n(15 states)", "new count\n(11 states)", "class B\n(10 states)", "LOW control\n(pass needs 3 sd)"])  # fmt: skip
    ax.set_ylabel("count-a accuracy after pruning top-K units\nminus mean over 20 random K-unit draws")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    finish(fig, out, "Pruning revives count a by 0.35 or more in every 'reuses' state and by at most 0.08 in 'new' ones;\n"
           "the filed 3 sd rule calls six of the eleven 'new' states 'existing'")  # fmt: skip


def fig_reverse(c: dict[str, Any], out: str = "c1_reverse.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, lr, words in zip(axes, ("1e-06", "1e-05"), ("slow fine-tunes (lr 1e-5), reversed at 1e-6", "fast fine-tunes (lr 1e-4), reversed at 1e-5"), strict=True):
        sc = revival.scratch_value(c["controls"]["scratch"][lr]["steps"])
        pts = [(truth_of(c, s), revival.scratch_value(r["reverse"]["steps"])) for s, r in c["states"].items()
               if "reverse" in r and f"{r['reverse']['lr']:.0e}" == lr]  # fmt: skip
        for x, t in enumerate(("existing", "new", "B")):
            ys = sorted(v for tt, v in pts if tt == t)
            ax.scatter(x + jitter(len(ys)), ys, color=TRUTH_C[t], s=34, zorder=3, label=TRUTH_W[t] if lr == "1e-06" else None)
        ax.axhline(sc, color=INK, lw=1.2)
        ax.text(2.4, sc, f"scratch (NEVER): {sc:,} steps" + (" (cap)" if sc == revival.CAP else ""), fontsize=7.5, ha="right", va="bottom")
        ax.axhline(sc * revival.EXISTING_FRAC, color="#c0392b", ls="--", lw=1.2)
        ax.text(2.4, sc * revival.EXISTING_FRAC, "half of scratch: below -> 'existing'", fontsize=7.5, ha="right", va="bottom", color="#c0392b")
        ax.set_xticks(range(3), ["reuses", "new", "class B"])
        ax.set_title(words, fontsize=9.5, loc="left")
        ax.set_yscale("log")
    axes[0].set_ylabel("reverse fine-tuning steps to 95% count a")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    finish(fig, out, "Reverse fine-tuning reaches 95% count a well inside half of scratch's steps in almost every state;\n"
           "within each rate, 'reuses' states are faster than 'new' ones")  # fmt: skip


def fig_steer(c: dict[str, Any], out: str = "c1_steer.png") -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    pos = {1000: 0, 2000: 1, 10000: 2}
    seen: set[str] = set()
    for i, (sid, r) in enumerate(c["states"].items()):
        t, step, st = truth_of(c, sid), int(sid.split("/")[-1]), r["steer"]
        x = pos[step] + ((i % 9) - 4) * 0.05
        if st["G_score"] < G_MIN:
            ax.scatter(x, np.nan_to_num(st["S"]), marker="x", color=INK, s=50, zorder=4,
                       label="R, HIGH base (gain G < 0.01: unreadable)" if "R" not in seen else None)  # fmt: skip
            seen.add("R")
            continue
        ax.scatter(x, st["S"], color=TRUTH_C[t], s=34, zorder=3, label=TRUTH_W[t] if t not in seen else None)
        seen.add(t)
    ax.axhline(revival.STEER_MIN, color="#c0392b", ls="--", lw=1.2)
    ax.text(2.45, revival.STEER_MIN + 0.01, "filed line: above -> 'existing'", fontsize=7.5, ha="right", color="#c0392b")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.text(2.45, 0.01, "do nothing (no shift)", fontsize=7.5, ha="right", color=MUTED)
    ax.set_xticks(range(3), ["1K steps", "2K steps", "10K steps"])
    ax.set_xlabel("fine-tuning steps of the state")
    ax.set_ylabel("steering gain share S: share of the fine-tune's gain\nin log P(its answer) recovered by one mean shift")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    finish(fig, out, "Every 1K state clears the filed steering line, 'new' ones included;\n"
           "'reuses' states sit at 0.84 or above, 'new' ones at 0.67 or below")  # fmt: skip


def fig_amend5b(out: str = "amend5b_saves.png") -> None:
    r5, r5b, bars = load("amend5.json"), load("amend5b.json"), load("predictions.json")["a2_bars"]
    low = r5b["W0"]["reference"]["check_count_a"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, t, bar, words in ((axes[0], "B", bars["B_a"], "behaviour check"), (axes[1], "P", bars["P_a"], "patch check")):
        ax.axhspan(bar["hi"], 1.05, color="#e8f2e0", zorder=0)
        for y, w in ((bar["hi"], "upper bar"), (bar["lo"], "lower bar")):
            ax.axhline(y, color=INK, ls=":", lw=0.9)
            ax.text(3.3, y + 0.01, w, ha="right", va="bottom", fontsize=7.5)
        for src, arm, col, w in ((r5, "WNf_s0", MUTED, "seed 0 (Amendment 5, reference)"), (r5b, "WNf_s1", CTL_C, "seeds 1-3 (Amendment 5b)"),
                                 (r5b, "WNf_s2", CTL_C, None), (r5b, "WNf_s3", CTL_C, None)):  # fmt: skip
            ys = [low[t]] + [src["arms"][arm]["states"][s]["check_count_a"][t] for s in ("500", "1000", "2000")]
            ax.plot(range(4), ys, color=col, marker="o", ms=4, lw=1.3, label=w if t == "B" else None)
        ax.set_xticks(range(4), ["LOW\n(no fine-tune)", "500", "1K", "2K"])
        ax.set_xlabel("fine-tuning steps")
        ax.set_title(words, fontsize=9.5, loc="left")
    axes[0].set_ylabel("share answering through count a")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower center")
    finish(fig, out, "Fast natural-row fine-tunes, seeds 1-3: through count a at 500 and 1K, neither by 2K, as seed 0")


def post_hoc_steer(c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Steering scored as filed but without the votes whose gain G is below G_MIN (labelled post hoc, not a re-scoring)."""
    states = {s: load_state(s) for s in c["states"]}
    truth = c["truth"]["scored"] | c["truth"]["steering_only"]
    keep = {s: v for s, v in truth.items() if c["states"][s]["steer"]["G_score"] >= G_MIN}
    lab = {s: r["labels"]["steer"] for s, r in c["states"].items()}
    nuis = nuisance_labels(states)
    return {"all": revival.score(lab, keep, nuis, revival.MARGIN_ALL),
            "per_run": revival.score(lab, per_run(keep, states), nuis, revival.MARGIN_RUN)}  # fmt: skip


def print_table(c: dict[str, Any]) -> None:
    sc = {k: revival.scratch_value(v["steps"]) for k, v in c["controls"]["scratch"].items()}
    print("| state | truth | count a before | pruning gain (K) | 3 sd | prune | reverse steps / scratch | reverse | G | S (block) | steer |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for sid, r in c["states"].items():
        lab, st = r["labels"], r["steer"]
        if "prune" in r:
            g, sd3, k = prune_gain(r["prune"])
            rv = r["reverse"]
            pr = f"{r['prune']['acc0']:.3f} | {g:+.3f} ({k}) | {sd3:.3f} | {lab['prune']}"
            lr = f"{rv['lr']:.0e}"
            rev = f"{revival.scratch_value(rv['steps']):,} / {sc[lr]:,} | {lab['reverse']}"
        else:
            pr, rev = "- | - | - | -", "- | -"
        flag = " unreadable" if st["G_score"] < G_MIN else ""
        print(f"| {sid} | {truth_of(c, sid)} | {pr} | {rev} | {st['G_score']:.4f} | {np.nan_to_num(st['S']):.3f} ({st['block']}) | {lab['steer']}{flag} |")
    ctl = c["controls"]["prune_LOW_importance"]
    for k in revival.PRUNE_K:
        d, sd = revival.revival_z(ctl[str(k)]["top"], ctl[str(k)]["random"], ctl["n_eval"])
        print(f"LOW importance K={k}: top {ctl[str(k)]['top']:.4f} random {np.mean(ctl[str(k)]['random']):.5f} drop {-d:.4f} 3sd {3 * sd:.5f} rows {-d * 4000:.0f}")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    c = load("c1.json")
    post = post_hoc_steer(c)
    print("post-hoc steering without G < 0.01 votes:", {w: {q: post[w][q] for q in ("n", "agree", "bar", "nuisance_agree", "passes")} for w in post})
    print_table(c)
    fig_scores(c, post)
    fig_prune(c)
    fig_reverse(c)
    fig_steer(c)
    fig_amend5b()


if __name__ == "__main__":
    main()
