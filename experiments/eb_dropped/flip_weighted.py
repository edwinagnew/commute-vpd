"""P1.57 (descriptive, $0, CPU): the flip-weighted kept share E_flip beside E_b, from saved per-variable arrays, as
filed in experiments/eb_dropped/results/flip_weighted_filing.md (with its same-day amendment).

E_flip = sum_{u in K & changed} flips_u * kept_u / sum flips_u, kept_u = explained_u / dd_u (`stand_in.explained`), over
the variables with dd_u > 0, without a floor and with dd_u >= f * sum(dd over K & changed), f in FLOORS. E_b =
`stand_in.explained_share(num, den, dd, K & changed)`, recomputed and checked against each item's saved E_b.

Sources: neurons/heads E277 (experiments/needed_neurons, every items_*.npz present: e62, built, trained, panel) and
E280 (experiments/head_edit); VPD E62 = E275's K, flips (claim2_needed) with E257's num, den, dd, changed (rb_states);
VPD built = E268's K, flips (kstar_needed) with E247's num, den, dd (thin_edit_diagnosis) and E242's changed
(kstar_vpd_wording_a), as kstar_needed/run.py `saved(item, "built")` pairs them. Rerun when more npz land:
    fleet/cpu-run python3 experiments/eb_dropped/flip_weighted.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import stand_in as SI
from commute.rule_retention import D_MLP, N_HEAD

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "eb_dropped" / "results"
FLOORS = (1e-8, 1e-6, 1e-4)
EB_TOL = 1e-12  # recomputed E_b against the saved one (relative); a miss means the wrong arrays: the script stops
BUILT = ("x2", "x10", "rebuilt", "new_few", "new_thin", "random_size_241001", "random_kl_241002")
E62 = [f"e62-{t}_s{s}" for t in ("copy", "lookup") for s in range(3)] + [f"e62-lookup_s{s}_scaled_e62_copy_lookup" for s in range(3)]
PAIRS = [("copy24", "cipher24"), ("ident32", "rand32"), ("case_real", "case_twin"), ("repeat_real", "repeat_twin"), ("twin", "fact")]  # E229: (reuse, new)
PER_LAYER = N_HEAD + D_MLP + 1  # E277's order per layer: heads, neurons, the remainder (needed_neurons.variables)


def neuron_name(i: int) -> str:
    l, j = divmod(i, PER_LAYER)
    return f"L{l}.h{j}" if j < N_HEAD else f"L{l}.n{j - N_HEAD}" if j < N_HEAD + D_MLP else f"L{l}.rem"


def load(p: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    z = np.load(p)
    return {k: z[k] for k in keys}


def e_flip(flips: np.ndarray, kept: np.ndarray, sel: np.ndarray) -> float | None:
    w = flips[sel].astype(np.float64)
    return None if w.sum() == 0 else float((w * kept[sel]).sum() / w.sum())


def read(a: dict[str, np.ndarray], eb_saved: float, named: bool) -> dict[str, Any]:
    num, den, dd = (torch.from_numpy(a[k].astype(np.float64)) for k in ("num", "den", "dd"))
    sel_t = torch.from_numpy(a["in_K"].astype(bool) & a["changed"].astype(bool))
    eb = SI.explained_share(num, den, dd, sel_t)
    if abs(eb - eb_saved) > EB_TOL * max(abs(eb_saved), 1e-300):
        raise SystemExit(f"E_b recomputed {eb!r} != saved {eb_saved!r}: wrong arrays")
    ddn, ex = dd.numpy(), SI.explained(num, den).numpy()
    sel = sel_t.numpy() & (ddn > 0)
    kept = np.where(ddn > 0, ex / np.where(ddn > 0, ddn, 1.0), np.nan)
    flips, tot = a["flips"].astype(np.int64), float(ddn[sel].sum())
    idx = np.flatnonzero(sel)
    top = idx[np.argsort(-flips[idx], kind="stable")[:5]]
    return {"E_b": eb, "E_b_saved": eb_saved, "E_flip": e_flip(flips, kept, sel),
            "E_flip_floor": {f"{f:g}": e_flip(flips, kept, sel & (ddn >= f * tot)) for f in FLOORS},
            "n_floor": {f"{f:g}": int((sel & (ddn >= f * tot)).sum()) for f in FLOORS},
            "n": int(sel.sum()), "n_K_changed_dd0": int(sel_t.numpy().sum() - sel.sum()), "flips_sum": int(flips[sel].sum()),
            "top5": [{"var": neuron_name(int(u)) if named else int(u), "flips": int(flips[u]), "kept": float(kept[u]),
                      "dd_share": float(ddn[u] / tot)} for u in top]}  # fmt: skip


def item_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def neurons() -> dict[str, dict[str, Any]]:
    """Every E277 item with an npz (E_b saved in its item JSON), then E280's at both featurizations."""
    out: dict[str, dict[str, Any]] = {}
    nn = EXP / "needed_neurons" / "results"
    for p in sorted((nn / "large").glob("items_*.npz")):
        f = p.stem.removeprefix("items_")
        a = load(p, ("in_K", "flips", "num", "den", "dd", "changed"))
        out[f"{f}|neurons"] = {"item": f, "featurization": "neurons", "source": "E277"} | read(a, item_json(nn / "items" / f"{f}.json")["E_b_K"], True)
    he = EXP / "head_edit" / "results"
    for p in sorted((he / "large").glob("items_*.npz")):
        name, feat = p.stem.removeprefix("items_").rsplit("_", 1)
        a = load(p, ("in_K", "flips", "num", "den", "dd", "changed"))
        out[f"e280_{name}|{feat}"] = {"item": f"e280_{name}", "featurization": feat, "source": "E280"} | read(
            a, item_json(he / "items" / f"{name}_{feat}.json")["E_b_K"], feat == "neurons")
    return out


def vpd() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    c2, rb = EXP / "claim2_needed" / "results", EXP / "rb_states" / "results" / "large"
    for f in E62:
        a = load(c2 / "large" / f"items_{f}.npz", ("in_K", "flips")) | load(rb / f"items_{f}.npz", ("num", "den", "dd", "changed"))
        out[f"{f}|vpd"] = {"item": f, "featurization": "vpd", "source": "E275+E257"} | read(a, item_json(c2 / "items" / f"{f}.json")["E_b_K"], False)
    kn = EXP / "kstar_needed" / "results"
    for f in BUILT:
        a = (load(kn / "large" / f"items_{f}.npz", ("in_K", "flips")) | load(EXP / "thin_edit_diagnosis" / "results" / "large" / f"items_{f}.npz", ("num", "den", "dd"))
             | load(EXP / "kstar_vpd_wording_a" / "results" / "large" / f"items_{f}.npz", ("changed",)))  # fmt: skip
        out[f"{f}|vpd"] = {"item": f, "featurization": "vpd", "source": "E268+E247+E242"} | read(a, item_json(kn / "items" / f"{f}.json")["E_b_K"], False)
    return out


def expected(item: str, feat: str) -> tuple[str, tuple[float, float] | None]:
    """The amendment's expected values (not bars); the range is this script's reading of "about": a factor 3 around
    'about 0.01' (0.01-0.02 for new_*), at least 0.5 (VPD) or 0.9 ('near 1', neurons) for x2/x10, 1 to 1e-9 for E280 V, O."""
    if item.startswith("random_") or item.endswith("_rand") or "__random_" in item:
        return "about 0.01", (0.01 / 3, 0.03)
    if item in ("new_few", "new_thin"):
        return "about 0.01-0.02", (0.01 / 3, 0.06)
    if item in ("x2", "x10"):
        return (">= 0.5", (0.5, 1.0)) if feat == "vpd" else ("near 1", (0.9, 1.0))
    if item in ("e280_v2", "e280_o2") and feat == "neurons":
        return "exactly 1", (1 - 1e-9, 1 + 1e-9)
    return "", None


def kind(item: str) -> str:
    if item.startswith("e62"):
        return "E62 (restatement)"
    if item.startswith("panel__"):
        return "panel (held out)"
    if item in BUILT or item.startswith("e280_"):
        return "control"
    return "trained (held out)"


def label(r: dict[str, Any]) -> list[str]:
    """'not as expected' outside the expected range; 'rests on near-rounding variables' when E_flip moves by more than
    10% of its f = 1e-8 value between f = 1e-8 and 1e-6 (the reading rule sent by the lead, 2026-11-23)."""
    out, v = [], r["E_flip"]
    lo_hi = r["expected_range"]
    if lo_hi is not None and (v is None or not lo_hi[0] <= v <= lo_hi[1]):
        out.append("not as expected")
    a, b = r["E_flip_floor"]["1e-08"], r["E_flip_floor"]["1e-06"]
    if a is not None and (b is None or abs(a - b) > 0.1 * abs(a)):
        out.append("rests on near-rounding variables")
    return out


def pair_diffs(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per E229 pair and seed with both items read: reuse minus new under E_b and E_flip (and each one's random)."""
    out = []
    for reuse, new in PAIRS:
        for s in (0, 1):
            g = {k: rows.get(f"{k}|neurons") for k in (f"{reuse}_s{s}", f"{new}_s{s}", f"{reuse}_s{s}_rand", f"{new}_s{s}_rand")}
            ru, nw = g[f"{reuse}_s{s}"], g[f"{new}_s{s}"]
            if ru is None or nw is None:
                continue
            d = {k: None if ru[k] is None or nw[k] is None else ru[k] - nw[k] for k in ("E_b", "E_flip")}
            out.append({"pair": f"{reuse}/{new}", "seed": s, "E_b_reuse_minus_new": d["E_b"], "E_flip_reuse_minus_new": d["E_flip"],
                        "rand": {k: (v["E_b"], v["E_flip"]) if v else None for k, v in g.items() if k.endswith("_rand")}})  # fmt: skip
    return out


def summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Three-seed means of copy and lookup (and the scaled lookups) at each featurization, and copy / lookup."""
    out: dict[str, Any] = {}
    for feat, pre in (("neurons", "e62__"), ("vpd", "e62-")):
        m: dict[str, Any] = {}
        for key, names in (("copy", [f"{pre}copy_s{s}" for s in range(3)]), ("lookup", [f"{pre}lookup_s{s}" for s in range(3)]),
                           ("lookup_scaled", [f"e62-lookup_s{s}_scaled_e62_copy_lookup" for s in range(3)])):  # fmt: skip
            got = [rows[f"{n}|{feat}"] for n in names if f"{n}|{feat}" in rows]
            m[key] = {q: float(np.mean([g[q] for g in got])) if len(got) == 3 else None for q in ("E_b", "E_flip")}
        for q in ("E_b", "E_flip"):
            c, lk = m["copy"][q], m["lookup"][q]
            m[f"copy_over_lookup_{q}"] = c / lk if c is not None and lk else None
        out[feat] = m
    return out


def fmt(v: float | None) -> str:
    return "undefined" if v is None else f"{v:.4g}"


def table(rows: dict[str, dict[str, Any]]) -> str:
    order = {"trained (held out)": 0, "panel (held out)": 1, "control": 2, "E62 (restatement)": 3}
    lines = ["# Flip-weighted kept share E_flip beside E_b (P1.57, descriptive)", "",
             "Filing: `flip_weighted_filing.md` (with its amendment). Written by `experiments/eb_dropped/flip_weighted.py`;",
             "per-item details (n, flips, top-5 variables) in `flip_weighted.json`. Held-out items and controls first; E62",
             "copy against lookup is a restatement (the weighting was chosen after seeing it). 'not as expected': outside the",
             "amendment's expected value as `expected()` reads it; 'rests on near-rounding variables': E_flip moves by more",
             "than 10% between dd floors f = 1e-8 and 1e-6.", "",
             "| group | item | featurization | E_b | E_flip | f=1e-8 | f=1e-6 | f=1e-4 | n | expected | label |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in sorted(rows.values(), key=lambda r: (order[r["group"]], r["item"], r["featurization"])):
        fl = r["E_flip_floor"]
        lines.append(f"| {r['group']} | {r['item']} | {r['featurization']} | {fmt(r['E_b'])} | {fmt(r['E_flip'])} | {fmt(fl['1e-08'])} | "
                     f"{fmt(fl['1e-06'])} | {fmt(fl['0.0001'])} | {r['n']} | {r['expected']} | {', '.join(r['labels'])} |")  # fmt: skip
    pairs = pair_diffs(rows)
    lines += ["", "Trained pairs at neurons, reuse minus new: " + ("none read yet." if not pairs else ""), ""]
    if pairs:
        lines += ["| pair | seed | E_b diff | E_flip diff |", "|---|---|---|---|"]
        lines += [f"| {p['pair']} | {p['seed']} | {fmt(p['E_b_reuse_minus_new'])} | {fmt(p['E_flip_reuse_minus_new'])} |" for p in pairs]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = neurons() | vpd()
    for r in rows.values():
        r["group"] = kind(r["item"])
        r["expected"], r["expected_range"] = expected(r["item"], r["featurization"])
        r["labels"] = label(r)
    nn_present = {r["item"] for r in rows.values() if r["source"] == "E277"}
    want = [f"{m}_s{s}{x}" for a, b in PAIRS for m in (a, b) for s in (0, 1) for x in ("", "_rand")] + list(BUILT)
    res = {"cell": "P1.57", "filing": "experiments/eb_dropped/results/flip_weighted_filing.md", "floors": list(FLOORS),
           "e277_missing": [w for w in want if w not in nn_present], "e277_panel_present": sorted(i for i in nn_present if i.startswith("panel__")),
           "summary_e62": summary(rows), "trained_pairs": pair_diffs(rows), "items": rows}  # fmt: skip
    (OUT / "flip_weighted.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "flip_weighted.md").write_text(table(rows))
    print(table(rows))
    print(json.dumps({k: res[k] for k in ("e277_missing", "e277_panel_present", "summary_e62", "trained_pairs")}, indent=1))


if __name__ == "__main__":
    main()
