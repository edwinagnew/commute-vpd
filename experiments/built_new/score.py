"""P1.63's scoring ($0, CPU, saved arrays only): both runs of the cell, E301 (experiments/built_circuits, the bigram and
induction circuits) and E304 (this folder, new edits on three more cue draws), with E298's rule unchanged: every value
is experiments/fig2_standard/eb.py `from_npz` on the read's saved per-variable sums, the one expression behind every
Fig 2 number (rule fixed at 937ba9a7), checked against the value the read job wrote to items/<feat>/<state>.json.

Reads from --root (the clone the Slurm jobs and pods write to: results/large/<feat>/, items/, stage0/), so a later read
(E301's induction items) drops in with one rerun; a state not yet home is listed under "missing". Writes, in this tree:
    experiments/built_circuits/results/definition_rows_p163.csv   E301's rows (block built_<circuit>)
    experiments/built_new/results/definition_rows_p163.csv        E304's rows (block built_new_<draw>)
    experiments/built_new/results/summary_p163.json               controls, builds, rows, delta_f, E301's P1 and P2,
        E304's P, and a diagnostic per state: the write share over every changed variable and K's share of the write change
    experiments/built_new/figures/p163_built.png (.pdf, .csv)     the new dots beside Fig 2's built rows

    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/built_new/score.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import figures as FG
from commute.jobs import EXPERIMENTS, load_experiment

EB: Any = load_experiment("fig2_standard", "eb")
S298: Any = load_experiment("fig2_standard", "score")  # Fig 2's own arrays (`reused`) and its side names
HERE = Path(__file__).resolve().parent
MAIN = Path("/mnt/nw/home/e.agnew/commute-agents/regating-lead/experiments")
FEATS = {"vpd": "VPD", "neurons": "top", "sae": "SAE"}  # definition.csv's feat names
REUSE, NEW = ("x2", "x10", "rebuilt"), ("new_few", "new_thin")
COLS = ["feat", "block", "item", "side", "value", "field", "source", "value_no_leftover", "leftover_share"]


def e301_states(root: Path) -> list[tuple[str, str]]:
    """(circuit, state) in E301's filed order: Amendment 2 drops the new edits and their randoms on `no_new` circuits."""
    s = json.loads((root / "built_circuits" / "results" / "predictions.json").read_text())["settings"]
    out = []
    for c in s["circuits"]:
        edits = REUSE if c in s["no_new"] else (*REUSE, *NEW)
        out += [(c, "do_nothing"), *((c, e) for e in edits), *((c, f"rand_{e}") for e in edits)]
        out += [(c, "x2_s269")] if c == s["repeat_circuit"] else []
    return out


def e304_states(root: Path) -> list[tuple[str, str]]:
    draws = [d["draw"] for d in json.loads((root / "built_new" / "results" / "build.json").read_text())["draws"]]
    return [(d, s) for d in draws for s in (*NEW, *(f"rand_{e}" for e in NEW))]


def side(state: str) -> tuple[str, str]:
    """(item, side) as Fig 2's rows name them: a matched random is (<its edit>, random)."""
    if state.startswith("rand_"):
        return state.removeprefix("rand_"), "random"
    return state, ({"do_nothing": "control", "x2_s269": "floor_repeat"} | S298.BUILT_SIDE)[state]


def diagnostic(z: dict[str, np.ndarray], feat: str) -> dict[str, float]:
    """Beside E_b over K: the same expression over every changed variable (K dropped), and K's share of the write change."""
    every = EB.readings(*(torch.from_numpy(np.asarray(z[k])) for k in ("num", "den", "dd")), torch.from_numpy(z["changed"].astype(bool)),
                        torch.from_numpy(EB.leftover_mask(feat, len(z["changed"]))), torch.ones(len(z["changed"]), dtype=torch.bool))  # fmt: skip
    dd, ch, K = np.asarray(z["dd"]), z["changed"].astype(bool), z["in_K"].astype(bool)
    return {"E_b_changed": every["E_b_K"], "dd_share_in_K": float(dd[ch & K].sum() / dd[ch].sum()) if dd[ch].sum() > 0 else 0.0}


def score(root: Path, exp: str, folder: str, states: list[tuple[str, str]], block: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows, missing, differ = [], [], []
    res = root / folder / "results"
    for feat, lab in FEATS.items():
        for site, state in states:
            name = EB.fname(f"{site}/{state}")
            p = res / "large" / feat / f"{name}.npz"
            if not p.is_file():
                missing.append(f"{exp} {feat} {site}/{state}")
                continue
            z = dict(np.load(p))
            r = EB.from_npz(z, feat) | diagnostic(z, feat)
            if (j := res / "items" / feat / f"{name}.json").is_file() and json.loads(j.read_text())["E_b_K"] != r["E_b_K"]:
                differ.append(f"{exp} {feat} {site}/{state}")
            item, sd = side(state)
            rows.append({"feat": lab, "block": block.format(site), "item": item, "side": sd, "value": r["E_b_K"], "field": "E_b_K", "source": exp,
                         "value_no_leftover": r["E_b_K_no_leftover"], "site": site, "state": state} | r)  # fmt: skip
    return rows, missing, differ


def fig2(root: Path) -> list[dict[str, Any]]:
    """Fig 2's built edits and their own randoms (E298's rows), with the same diagnostic from the arrays E298 scored."""
    S298.LARGE = {k: root / p.relative_to(EXPERIMENTS) for k, p in S298.LARGE.items()}
    with (EXPERIMENTS / "fig2_standard" / "results" / "definition_rows.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["block"] == "built" and r["item"] in (*REUSE, *NEW)]
    inv = {v: k for k, v in FEATS.items()}
    for r in rows:
        state = f"built/{'rand_' if r['side'] == 'random' else ''}{r['item']}"
        p = root / "fig2_standard" / "results" / "large" / inv[r["feat"]] / f"{EB.fname(state)}.npz"
        z = dict(np.load(p)) if p.is_file() else S298.reused(inv[r["feat"]], state)[0]
        assert EB.from_npz(z, inv[r["feat"]])["E_b_K"] == float(r["value"]), f"Fig 2 {r['feat']} {state}: arrays do not give the row"
        r |= {"value": float(r["value"]), "site": "fig2", "state": state.split("/")[1]} | diagnostic(z, inv[r["feat"]])
    return rows


def controls(root: Path) -> dict[str, Any]:
    """Every stage-0 control file: do-nothing (0 changed, dd max 0, E_b 1, all exact) and identity (0 flips; max_dev)."""
    out: dict[str, Any] = {}
    for folder in ("built_circuits", "built_new", "built_new/results/l40check"):
        d = root / folder / ("stage0" if folder.endswith("l40check") else "results/stage0")
        for kind in ("do_nothing", "identity"):
            for p in sorted(d.glob(f"*_{kind}_*.json")):
                j = json.loads(p.read_text())
                held = j["n_changed"] == 0 and j["dd_max"] == 0 and j["E_b"] == 1 if kind == "do_nothing" else j["flips"] == 0
                out[f"{folder}/{p.stem}"] = {"held": held} | {k: j.get(k) for k in ("n_changed", "dd_max", "E_b", "flips", "max_dev", "min_margin", "card")
                                                                if k in j}  # fmt: skip
    return out


def get(rows: list[dict[str, Any]], feat: str, site: str, state: str) -> float | None:
    return next((r["value"] for r in rows if r["feat"] == feat and r["site"] == site and r["state"] == state), None)


def p1(rows: list[dict[str, Any]], states: list[tuple[str, str]], delta: dict[str, float | None]) -> dict[str, Any]:
    """E301's P1 per circuit and featurization: min E_b_K over the reuse edits minus max over the circuit's other filed
    states (its new edits and every random); 'holds' when the gap exceeds delta_f, 'no call' until delta_f and every state."""
    out = {}
    for c in dict.fromkeys(s for s, _ in states):
        filed = [st for s, st in states if s == c and st not in ("do_nothing", "x2_s269")]
        for f in FEATS.values():
            v = {st: get(rows, f, c, st) for st in filed}
            if any(x is None for x in v.values()):
                out[f"{c}/{f}"] = {"verdict": "no call", "why": "states not home"}
                continue
            gap = min(v[e] for e in REUSE) - max(x for st, x in v.items() if st not in REUSE)  # type: ignore[type-var]
            d = delta[f]
            out[f"{c}/{f}"] = {"gap": gap, "delta_f": d, "verdict": "no call" if d is None else "holds" if gap > d else "does not hold"}
    return out


def p304(rows: list[dict[str, Any]], f2: list[dict[str, Any]], delta: dict[str, float | None]) -> dict[str, Any]:
    """E304's P: per new edit and featurization, its distance above the highest of Fig 2's two new edits and outside
    their range. Rival A: every edit within delta_f of that range at every featurization; rival B: some edit above the
    highest by more than delta_f at two or more featurizations; else neither. 'no call' until delta_f exists."""
    per: dict[str, dict[str, float]] = {}
    for r in (r for r in rows if r["side"] == "new"):
        new2 = [x["value"] for x in f2 if x["feat"] == r["feat"] and x["side"] == "new"]
        per.setdefault(f"{r['site']}/{r['state']}", {})[r["feat"]] = r["value"] - max(new2) if r["value"] > max(new2) else min(0.0, r["value"] - min(new2))
    if any(d is None for d in delta.values()):
        return {"verdict": "no call", "why": "delta_f not measured yet (E301 induction x2_s269)", "outside_fig2_range": per}
    above = {k: sum(v[f] > delta[f] for f in v) for k, v in per.items()}  # type: ignore[operator]
    a = all(abs(v[f]) <= delta[f] for v in per.values() for f in v)  # type: ignore[operator]
    return {"verdict": "rival A" if a else "rival B" if max(above.values()) >= 2 else "neither", "outside_fig2_range": per, "delta_f": delta}


def draw(rows: list[dict[str, Any]], f2: list[dict[str, Any]], out: Path) -> None:
    groups = [("Fig 2 site, reuse", "fig2", "reuse"), ("Fig 2 site, new", "fig2", "new"), ("bigram, reuse", "bigram", "reuse"),
              ("bigram, new", "bigram", "new"), ("induction, reuse", "induction", "reuse"), ("3 new cue draws, new", "draws", "new")]  # fmt: skip
    every = f2 + rows
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=3)
    data = []
    for ax, f in zip(axes, FEATS.values(), strict=True):
        for method, off, rnd in (("eb", -0.15, False), ("random", 0.15, True)):
            g: dict[str, list[float]] = {}
            for label, site, sd in groups:
                pick = [r for r in every if r["feat"] == f and (r["site"] == site or (site == "draws" and r["site"].startswith("s186")))
                        and r["side"] == ("random" if rnd else "reuse" if sd == "reuse" else "new") and (not rnd or (r["item"] in REUSE) == (sd == "reuse"))]  # fmt: skip
                g[label] = [max(r["value"], 1e-4) for r in pick]
                data += [{"feat": f, "group": label, "random": rnd, "item": r["item"], "site": r["site"], "value": r["value"]} for r in pick]
            FG.dots(ax, g, method=method, xlabel="write share over the needed set", offset=off)
        ax.set_xscale("log")
        ax.set_xlim(5e-5, 2)
        ax.set_xlabel("write share over the needed set", fontsize=7)
        ax.axhline(1.5, color="#bbbbbb", linewidth=0.6)
        ax.set_title(f"{FEATS_LONG[f]}", fontsize=9)
        if f != "VPD":
            ax.set_yticklabels([])
    for m in ("eb", "random"):
        axes[0].scatter([], [], color=FG.METHODS[m].color, marker=FG.METHODS[m].marker, s=22, label="built edit" if m == "eb" else FG.METHODS[m].label)
    fig.legend(fontsize=7, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.95), frameon=False)
    fig.suptitle("Do built edits on other circuits and cue draws read like Fig 2's (top two rows)?", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    FG.save(fig, out, data=data, n="one point per state; each edit's own size-matched random is drawn just below its row; 256 pairs per needed set; "
            "values under 1e-4 drawn at 1e-4")  # fmt: skip


FEATS_LONG = {"VPD": "VPD subcomponents", "top": "top neurons", "SAE": "SAE latents"}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(root: Path) -> None:
    s301, s304 = e301_states(root), e304_states(root)
    r301, m301, d301 = score(root, "E301", "built_circuits", s301, "built_{}")
    r304, m304, d304 = score(root, "E304", "built_new", s304, "built_new_{}")
    f2 = fig2(root)
    delta = {f: abs(a - b) if (a := get(r301, f, "induction", "x2")) is not None and (b := get(r301, f, "induction", "x2_s269")) is not None else None
             for f in FEATS.values()}  # fmt: skip
    builds = {c: json.loads((root / "built_circuits" / "results" / f"build_{c}.json").read_text()) for c, _ in dict.fromkeys(s301)}
    norms = {f"E301 {c} {k}": abs(v["size2_rand"] / v["size2_F"] - 1) for c, b in builds.items() for k, v in b.items() if k.startswith("rand_")}
    b304 = json.loads((root / "built_new" / "results" / "build.json").read_text())
    norms |= {f"E304 {d['draw']} {k}": abs(d[k]["size2_rand"] / d[k]["size2_F"] - 1) for d in b304["draws"] for k in d if k.startswith("rand_")}
    gam = {d["draw"]: {k: d[k] for k in ("gamma_new_few", "gamma_new_thin", "top1_new_few", "top1_new_thin")} for d in b304["draws"]}
    gam["bigram (E301)"] = {k: builds["bigram"]["new"][k] for k in ("gamma_new_few", "gamma_new_thin", "top1_new_few", "top1_new_thin")} if "bigram" in builds else {}
    summary = {"cell": "P1.63", "root": str(root), "controls": controls(root), "norm_rel_dev": norms, "gammas": gam, "missing": m301 + m304,
               "json_differs_from_npz": d301 + d304, "delta_f": delta, "P1_E301": p1(r301, s301, delta), "P_E304": p304(r304, f2, delta),
               "rows": r301 + r304, "fig2_built_rows": f2}  # fmt: skip
    write_rows(EXPERIMENTS / "built_circuits" / "results" / "definition_rows_p163.csv", r301)
    write_rows(HERE / "results" / "definition_rows_p163.csv", r304)
    (HERE / "results" / "summary_p163.json").write_text(json.dumps(summary, indent=1))
    draw(r301 + r304, f2, HERE / "figures" / "p163_built.png")
    held = {k: v["held"] for k, v in summary["controls"].items()}
    print(f"controls held: {sum(held.values())} files of {len(held)}; failing: {[k for k, v in held.items() if not v]}")
    print(f"rows: E301 {len(r301)}, E304 {len(r304)}; missing {len(summary['missing'])}: {summary['missing']}; json differs: {summary['json_differs_from_npz']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MAIN, help="the experiments/ folder the read jobs write to (read only)")
    main(ap.parse_args().root)
