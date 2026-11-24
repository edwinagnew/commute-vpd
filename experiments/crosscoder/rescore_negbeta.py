"""Cell P2.70: every saved crosscoder of E203, E212, E235, E243 and E244 rescored with Minder et al. App. E.2's exclusion
(a latent with a negative base beta, error or reconstruction, is not fine-tune-specific), now inside
`commute.crosscoder.specific`. CPU only, from the saved `results/large` records (relative norm, alive mask, per-latent
carried part s, the latent-scaled indices and their betas); no model, no crosscoder forward.

Old = the groups rebuilt with the thresholds alone (the rule the runs used), checked against the saved fine-tune-specific
mask and the reported parts; new = the same groups with `specific`'s filter. Writes results/negbeta_rescore.json and
figures/negbeta_ladder.png.

    ~/commute-agents/fleet/cpu-run --mem 8G -t 60 python3 experiments/crosscoder/rescore_negbeta.py
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute.figures import figure, save
from commute.jobs import config_from_settings
from commute.open_pair import jsonable

EXP = Path(__file__).resolve().parents[1]
LARGE = Path.home() / "commute-data" / "results-large"
OUT = Path(__file__).resolve().parent / "results" / "negbeta_rescore.json"
Reported = Callable[[str, str], dict[str, Any] | None]  # (file stem, read task) -> the run's own record


def cfg_of(pred: Path, *path: str) -> CC.Cfg:
    S = json.loads(pred.read_text())["settings"]
    for k in path:
        S = S[k]
    return config_from_settings(CC.Cfg, S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})


def read(p: Path, *path: str) -> dict[str, Any] | None:
    if not p.exists():
        return None
    o = json.loads(p.read_text())
    for k in path:
        o = o[k]
    return o


def e203(stem: str, _task: str) -> dict[str, Any] | None:
    R = EXP / "crosscoder/results/run2"
    special = {"self": ("stage0a.json", "self"), "planted": ("stage0b.json", "planted"),
               "e54__elicit_s0__seed1": ("stage1.json", "reuse_seed1"), "random_like__e54__elicit_s0": ("stage1.json", "random")}  # fmt: skip
    return read(R / special[stem][0], special[stem][1]) if stem in special else read(R / "sides" / f"{stem}.json")


def e244(stem: str, _task: str) -> dict[str, Any] | None:
    R = EXP / "key_grid_baselines/results/run/crosscoder"
    return read(R / "stage0a.json", "self") if stem == "self" else read(R / "sides" / f"{stem}.json")


def items(run: str) -> Reported:
    return lambda stem, _task: read(EXP / f"crosscoder_proper/results/{run}/items/{stem}.json", "read")


def e212(stem: str, task: str) -> dict[str, Any] | None:
    job = "gauss" if stem.startswith("gauss") else stem
    return read(EXP / f"realistic_tasks/results/run3_{job}.json", "side", "reads", task)


PRED_CC, PRED_P = EXP / "crosscoder/results/run2/predictions.json", EXP / "crosscoder_proper/results/predictions.json"
RUNS: dict[str, tuple[str, CC.Cfg, Reported]] = {  # E-code: (results-large folder, the run's Cfg, its reported records)
    "E203": ("crosscoder/run2", cfg_of(PRED_CC, "full"), e203),
    "E212": ("realistic_tasks/run3", cfg_of(EXP / "realistic_tasks/results/predictions_run3.json", "full"), e212),
    "E235": ("crosscoder_proper/run1", cfg_of(PRED_P, "cc", "full"), items("run1")),
    "E243": ("crosscoder_proper/run2", cfg_of(PRED_P, "cc", "full"), items("run2")),
    "E244": ("key_grid_baselines/run/crosscoder", cfg_of(PRED_CC, "full"), e244),
}


def old_rule(cfg: CC.Cfg, b: dict[str, Tensor]) -> Tensor:
    """The thresholds alone, as `specific` read before cell P2.70."""
    return (b["nu_r"] < cfg.nu_r) & (b["nu_eps"] < cfg.nu_eps)


def rebuilt(cfg: CC.Cfg, r: dict[str, Any], spec_J: Tensor) -> dict[str, dict[str, Tensor]]:
    """`CC.groups` at N and at the wide N under one rule. The saved relative norm is float16: its ties move the top-N
    boundary by up to 55 latents and its rounding moves latents across the shared band's edges, so the candidates at N
    are taken from the saved groups (the specific and artefact latents are exactly the candidates outside the shared
    band; one inside it is shared under either rule). The wide N has no saved mask: rebuilt from the float16 norm."""
    rn, saved = r["relative_norm"].float(), r["groups"]
    spec = torch.zeros(len(rn), dtype=torch.bool)
    spec[r["scaled"]] = spec_J
    cands = {"": saved["ft_specific"] | saved["ft_artefact"], "_wide": CC.top(rn, cfg.top_n_wide)}
    return {tag: CC.groups(cfg, rn, r["alive"], c, spec) for tag, c in cands.items()}


def reading(g: dict[str, dict[str, Tensor]], r: dict[str, Any]) -> dict[str, Any]:
    """The three parts the filter can move, the counts, both forms of d and the lead latent's flag, at N and wide N."""
    s, alive, out = r["s"].double(), r["alive"], {}
    for tag, m in g.items():
        p = {k: float(s[m[k]].sum()) for k in ("ft_specific", "shared", "ft_artefact")}
        both = p["shared"] + p["ft_specific"]
        out |= {f"parts{tag}": p, f"counts{tag}": {k: int(v.sum()) for k, v in m.items()},
                f"shared_frac{tag}": p["shared"] / both if both else math.nan, **{f"{k}{tag}": v for k, v in CC.d_forms(p).items()}}  # fmt: skip
    if not bool(torch.isnan(s).all()):
        lead = int(torch.where(alive, s.nan_to_num(-math.inf), -math.inf).argmax())
        out["lead"] = {"latent": lead, "s": float(s[lead]), "specific": bool(g[""]["ft_specific"][lead])}
    return out


def off(mine: dict[str, Any], theirs: dict[str, Any] | None) -> dict[str, float] | None:
    """Largest absolute gap between the rebuilt old reading and the run's reported parts, and counts (None: no record)."""
    if theirs is None:
        return None
    gap = {}
    for k in ("parts", "counts", "parts_wide", "counts_wide"):
        g = [abs(v - theirs[k][q]) for q, v in mine[k].items() if k in theirs
             and isinstance(theirs[k].get(q), int | float) and math.isfinite(theirs[k][q])]  # fmt: skip
        gap[k] = max(g) if g else math.nan
    return gap


def one(cfg: CC.Cfg, r: dict[str, Any], theirs: dict[str, Any] | None) -> dict[str, Any]:
    """Old = the saved groups at N (the run's own), new = the saved groups with the latents the filter drops moved as
    `CC.groups` moves them; at the wide N both are rebuilt."""
    b, saved = r["betas"], r["groups"]
    g_old, g_new = rebuilt(cfg, r, old_rule(cfg, b)), rebuilt(cfg, r, CC.specific(cfg, b))
    same = all(bool((g_old[""][k] == saved[k]).all()) for k in ("ft_specific", "ft_artefact"))
    g_new[""] = {k: saved[k] ^ g_old[""][k] ^ g_new[""][k] for k in saved}
    g_old[""] = saved
    dropped = saved["ft_specific"] & ~g_new[""]["ft_specific"]
    rn, s = r["relative_norm"].float(), r["s"].double()
    edge = dropped & (((rn - cfg.shared[0]).abs() < 1e-3) | ((rn - cfg.shared[1]).abs() < 1e-3))
    near = torch.zeros(len(rn), dtype=torch.bool)  # descriptive only: both ratios above -0.02, a base beta near zero
    near[r["scaled"]] = torch.minimum(b["nu_r"], b["nu_eps"]) > -0.02
    old = reading(g_old, r)
    return {"old": old, "new": reading(g_new, r), "specific_and_artefact_match_saved": same,
            "max_gap_vs_reported": off(old, theirs), "dropped": int(dropped.sum()),
            "dropped_part": float(s[dropped].sum()), "dropped_part_near_zero": float(s[dropped & near].sum()),
            "dropped_to_shared": int((dropped & g_new[""]["shared"]).sum()), "dropped_at_band_edge": int(edge.sum()),
            "neg_eps_base": int((b["eps_base"] < 0).sum()), "neg_r_base": int((b["r_base"] < 0).sum()), "scaled": len(r["scaled"])}  # fmt: skip


def e203_tally(rows: dict[str, Any]) -> dict[str, Any]:
    """E203's four tallies (results/run2/summary.json: pairs whose side built to reuse reads higher), old and new."""
    form = {"crosscoder": "d_authors", "crosscoder_matched": "d_authors", "crosscoder_alt": "d_alt", "crosscoder_alt_matched": "d_alt"}
    cells = json.loads((EXP / "crosscoder/results/run2/summary.json").read_text())["cells"]
    return {c: {w: sum(rows[x["sides"]["reuse"]][w][form[c]] > rows[x["sides"]["new"]][w][form[c]] for x in cells if x["column"] == c)
                for w in ("old", "new")} for c in form}  # fmt: skip


def e244_tasks(rows: dict[str, Any]) -> dict[str, Any]:
    """E244 per task: the three fine-tunes' d_authors, their spread, and how many read below the task's three random
    updates (all three lower), old and new; the random updates have no specific latent under either rule."""
    out = {}
    for t in ("copy", "rank1", "elicit", "prior_cipher", "cipher"):
        rnd = [rows[f"random_e58-{t}_s{i}"]["new"]["d_authors"] for i in range(3)]
        out[t] = {"random": rnd} | {w: {"d": (d := [rows[f"e58-{t}_s{i}_scaled"][w]["d_authors"] for i in range(3)]),
                                       "spread": max(d) - min(d), "below_random": sum(v < min(rnd) for v in d)}
                                    for w in ("old", "new")}  # fmt: skip
    return out


def planted(res: dict[str, Any]) -> dict[str, Any]:
    """Each planted control's specific part, old and new, with the bar 0.5 (E212 also needs its lead latent specific)."""
    R = res["runs"]
    keys = {"E235 block 2 head": ("E235", [f"head.A.c{c}" for c in range(3)]), "E235 block 3 head": ("E235", [f"head.B.c{c}" for c in range(3)]),
            **{f"E243 rung {k}": ("E243", [f"head-r{k}.A.c{c}" for c in range(3) if f"head-r{k}.A.c{c}" in R["E243"]]) for k in (1, 2, 4, 5)},
            "E212 real-text plant": ("E212", ["planted.L1"]), "E203 planted direction": ("E203", ["planted"])}  # fmt: skip
    return {name: {w: {"specific": [R[e][k][w]["parts"]["ft_specific"] for k in ks],
                       "lead_specific": [R[e][k][w].get("lead", {}).get("specific") for k in ks]} for w in ("old", "new")}
            for name, (e, ks) in keys.items()}  # fmt: skip


def figure_ladder(res: dict[str, Any]) -> None:
    """The planted heads' specific part against squared activation change per token, old and new, bar 0.5."""
    fig, ax = figure((6.0, 3.6))
    data = []
    for e, keys in (("E235", [f"head.{s}.c{c}" for s in "AB" for c in range(3)]), ("E243", list(res["runs"]["E243"]))):
        for k in (k for k in keys if k.startswith("head")):
            run = "run1" if e == "E235" else "run2"
            x = float(json.loads((EXP / f"crosscoder_proper/results/{run}/items/{k}.json").read_text())["energy"])
            site = "block 3" if ".B." in k else "block 2"
            for w in ("old", "new"):
                v = max(res["runs"][e][k][w]["parts"]["ft_specific"], 1e-7)
                data.append({"item": f"{e} {k}", "site": site, "rule": w, "energy": x, "specific": v})
    for (site, w), style in {("block 2", "old"): ("#009E73", "o", "none"), ("block 2", "new"): ("#009E73", "o", "#009E73"),
                             ("block 3", "old"): ("#D55E00", "s", "none"), ("block 3", "new"): ("#D55E00", "s", "#D55E00")}.items():  # fmt: skip
        pts = [d for d in data if d["site"] == site and d["rule"] == w]
        label = f"{site}, {'thresholds only (as run)' if w == 'old' else 'with the negative-beta filter'}"
        ax.scatter([d["energy"] for d in pts], [d["specific"] for d in pts], edgecolors=style[0], marker=style[1], facecolors=style[2], s=30, label=label)
    ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--", label="planted bar, 0.5")
    ax.set(xscale="log", yscale="log", ylim=(5e-8, 1.5), xlabel="planted head's squared activation change per token",
           ylabel="share of the change carried\nby fine-tune-specific latents")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title("With the filter no planted head clears the bar, and the ladder's top falls to near zero", fontsize=9)
    save(fig, OUT.parent.parent / "figures/negbeta_ladder.png",
         n="One point per crosscoder: block 2 heads, 1 to 3 crosscoder seeds per size (E235 and E243); block 3 head, "
           "3 crosscoder seeds (E235). Values under 1e-7 drawn at 1e-7.", data=data)  # fmt: skip


def main() -> None:
    res: dict[str, Any] = {"rule": "fine-tune-specific = top N by relative norm, eps_base >= 0, r_base >= 0, nu_r < 0.5, nu_eps < 0.2 "
                                   "(Minder et al. 2025, arXiv 2504.02922 v4, App. E.2 and glossary)", "runs": {}}  # fmt: skip
    for code, (folder, cfg, reported) in RUNS.items():
        rows = res["runs"][code] = {}
        for f in sorted((LARGE / folder).glob("*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=True)
            reads = d.get("reads", {"": d})  # E212 keeps one record per read task; texts.pt holds tokens only
            for task, r in ((t, r) for t, r in reads.items() if isinstance(r, dict) and "betas" in r):
                key = f"{f.stem}.{task}" if task else f.stem
                rows[key] = one(cfg, r, reported(f.stem, task))
                x = rows[key]
                o, n = x["old"], x["new"]
                print(f"{code} {key:48s} specific {o['parts']['ft_specific']:.4g} -> {n['parts']['ft_specific']:.4g}  "
                      f"d_authors {o['d_authors']:.4g} -> {n['d_authors']:.4g}  count {o['counts']['ft_specific']} -> "
                      f"{n['counts']['ft_specific']}  match {x['specific_and_artefact_match_saved']} "
                      f"edge {x['dropped_at_band_edge']} gap {x['max_gap_vs_reported']}", flush=True)  # fmt: skip
            del d
    res["paper"] = {"planted": planted(res), "e203_tally": e203_tally(res["runs"]["E203"]), "e244_tasks": e244_tasks(res["runs"]["E244"])}
    OUT.write_text(json.dumps(jsonable(res), indent=1) + "\n")
    print("wrote", OUT)
    figure_ladder(res)


if __name__ == "__main__":
    main()
