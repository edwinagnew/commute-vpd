"""P1.49 ($0, saved readings only; no model, no GPU): R_b, the user's option (b) of 2026-11-21 (answer to ask 58), on
every saved item that has the per-member write changes it needs, beside R under wording A and K*'s untouched share.

R_b is E247's reading (experiments/thin_edit_diagnosis/run.py `shares`): over K*'s changed members, the share of their
write change dd = |write_F - write_M|^2 (summed over every position of D in F's own run) carried by the members that
stand in, R_b = 1 - c_kept. Which members stand in is E242's saved search result (experiments/kstar_vpd_wording_a,
results/large/items_<item>.npz `<search>_keeps_F`), unchanged. dd, num and den are E247's (results/large/items_<item>.npz),
computed with commute.stand_in.write_change; only E247's seven items have them.

Descriptive, not a definition (labelled so in the summary): two order-free variants that use no search, each member
read on its own. M's mechanism at the member's own least-squares gain g = num / den explains (num - den)^2 / den of its
write change (zero for g <= 0, which may not stand in, and where M writes zero):
    E_b   explained write change over all of K*'s changed members / their write change
    T_b   write change of the members whose own explained share is at least tau / their write change (tau 0.9, the
          bar's 1 - 0.1, with 0.8 and 0.95 beside it)

Edge cases: an item with no changed K* member, or with zero write change over them (the do-nothing), has no
denominator; R_b, E_b and T_b are then set to 1, as R_A reads 1 on the do-nothing (a convention, stated in the
summary). Ties do not enter R_b (a sum over a saved set). Tiny changes: R_b is recomputed with members under
eps x the largest member's dd dropped from both sums, eps in EPS.

    ~/commute-agents/fleet/cpu-run python3 experiments/option_b_rescore/score.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
E242, E247, E226 = (EXPERIMENTS / e / "results" for e in ("kstar_vpd_wording_a", "thin_edit_diagnosis", "kstar_vpd"))
BUILT = ("x2", "x10", "rebuilt", "new_few", "new_thin", "random_size_241001", "random_kl_241002")  # E247's items
SEARCHES = {"bar 0.1 (filed)": "main", "bar 0.05": "main_tol0.05", "bar 0.2": "main_tol0.2", "absolute bar": "main_abs"}
TAUS = (0.8, 0.9, 0.95)
EPS = (1e-9, 1e-6, 1e-3)
REPRO_REL = 1e-9  # the same sums over the same saved arrays: E247's c_kept must come back to rounding


def ratio(top: float, bottom: float) -> float:
    return 1.0 if bottom == 0 else top / bottom  # the stated convention for an empty or unchanged K*


def k_star(item: str) -> np.ndarray:
    p = E226 / "large" / f"items_{item}.npz"
    k = np.load(p if p.is_file() else E242 / "large" / f"kstar_{item}.npz")
    return k["main_in_K_polished"].astype(bool)


def explained_share(num: np.ndarray, den: np.ndarray, dd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per member: the write change M's mechanism at its own positive gain explains, and that as a share of dd."""
    g = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    expl = np.where((den > 0) & (g > 0), (num - den) ** 2 / np.where(den > 0, den, 1.0), 0.0)
    return expl, np.where(dd > 0, expl / np.where(dd > 0, dd, 1.0), 0.0)


def diagnosis(d: np.ndarray, own: np.ndarray, kept: np.ndarray, stood: np.ndarray) -> dict[str, Any]:
    """Step 1: are the kept members the largest ones (a count-minimal search), or the ones M's mechanism cannot write?"""
    ch = kept | stood
    n_kept = int(kept.sum())
    order = np.argsort(-d[ch], kind="stable")
    top = np.zeros(int(ch.sum()), dtype=bool)
    top[order[:n_kept]] = True
    kept_c = kept[ch]
    return {"n_changed_in_K": int(ch.sum()), "n_kept": n_kept, "kept_count_share": ratio(n_kept, int(ch.sum())),
            "kept_dd_share": ratio(float(d[kept].sum()), float(d[ch].sum())),
            "top_n_kept_dd_share": ratio(float(d[ch][top].sum()), float(d[ch].sum())),
            "kept_among_top_n_kept": ratio(int((kept_c & top).sum()), n_kept),
            "kept_median_dd_rank_pct": float(np.median(np.argsort(order)[kept_c]) / max(1, int(ch.sum()) - 1)) if n_kept else None,
            "own_explained_kept_dd_weighted": ratio(float((own * d)[kept].sum()), float(d[kept].sum())),
            "own_explained_stood_dd_weighted": ratio(float((own * d)[stood].sum()), float(d[stood].sum())),
            "own_explained_kept_median": float(np.median(own[kept])) if n_kept else None,
            "own_explained_stood_median": float(np.median(own[stood])) if stood.any() else None}  # fmt: skip


def read_item(item: str) -> dict[str, Any]:
    a, w = np.load(E242 / "large" / f"items_{item}.npz"), np.load(E247 / "large" / f"items_{item}.npz")
    r242 = json.loads((E242 / "items" / f"{item}.json").read_text())
    K, ch = k_star(item), a["changed"].astype(bool)
    num, den, d = w["num"], w["den"], w["dd"]
    expl, own = explained_share(num, den, d)
    chK = ch & K
    out: dict[str, Any] = {"item": item, "R_A": r242["searches"]["main"]["R_A"], "size_K": int(K.sum()),
                           "untouched_share": ratio(int((K & ~ch).sum()), int(K.sum())), "R_b": {}}  # fmt: skip
    for label, s in SEARCHES.items():
        if f"{s}_keeps_F" not in a.files:
            continue
        kept = chK & a[f"{s}_keeps_F"].astype(bool)
        out["R_b"][label] = 1 - ratio(float(d[kept].sum()), float(d[chK].sum()))
        if s == "main":
            out["R_A_changed_only"] = 1 - ratio(int(kept.sum()), int(chK.sum()))
            out["diagnosis"] = diagnosis(d, own, kept, chK & ~kept)
            big = {e: d >= e * d[chK].max() for e in EPS}
            out["R_b_drop_tiny"] = {f"{e:g}": 1 - ratio(float(d[kept & b].sum()), float(d[chK & b].sum())) for e, b in big.items()}
    out["E_b"] = ratio(float(expl[chK].sum()), float(d[chK].sum()))
    out["T_b"] = {f"{t:g}": ratio(float(d[chK & (own >= t)].sum()), float(d[chK].sum())) for t in TAUS}
    c247 = json.loads((E247 / "summary.json").read_text())["items"][item]["c_kept"]
    assert abs((1 - out["R_b"]["bar 0.1 (filed)"]) / c247 - 1) <= REPRO_REL, f"{item}: c_kept does not repeat E247's {c247}"
    return out


def do_nothing() -> dict[str, Any]:
    """E247's stage 0 (no changed variable, every write change exactly 0) and E242's do-nothing (R_A 1 by definition)."""
    s0 = json.loads((E247 / "stage0.json").read_text())
    assert s0["n_changed"] == 0 and s0["dd_max"] == 0, "E247's do-nothing has a change"
    return {"item": "do_nothing", "R_A": 1.0, "untouched_share": 1.0, "R_b": {"bar 0.1 (filed)": 1.0}, "E_b": 1.0,
            "T_b": {f"{t:g}": 1.0 for t in TAUS}, "note": "no changed member: R_b, E_b, T_b set to 1 by the stated convention"}  # fmt: skip


NOT_COMPUTABLE = {
    "E248 nine pairs (76 fine-tunes), E62 copy/lookup (6), E237 keys (15), VPD": "E242's run.py saves gains, changed and keeps_F "
    "(kstar_vpd_wording_a/results/large/items_*.npz) but no per-member write change; E226 saves cos^2 grades and attributions only",
    "E245 built, pairs, keys and E251 randoms, SAE": "sae_r_a(_random)/results/large/items_*.npz save gain, changed, K* and keeps_F "
    "per tolerance; no per-latent write change",
    "moved": "not read under wording A (KL(F || M) under the floor), not in E247",
}


def main() -> None:
    items = [do_nothing(), *(read_item(i) for i in BUILT)]
    out = {"cell": "P1.49", "reading": "R_b = 1 - c_kept as E247 defined it; E_b and T_b are descriptive order-free variants, not definitions",
           "convention": "no changed K* member or zero write change over them: R_b = E_b = T_b = 1",
           "items": {r["item"]: r for r in items}, "not_computable_at_0": NOT_COMPUTABLE}  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in items:
        print(json.dumps({k: r.get(k) for k in ("item", "R_A", "untouched_share", "R_b", "E_b", "T_b")}))


if __name__ == "__main__":
    main()
