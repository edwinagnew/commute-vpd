"""E274 (cell P1.55): the squared-cosine share made admissible and read over the needed set K, from saved files ($0).

P1.53 (reports/analyses/2026-11-22_analysis_reading-admissibility.md) found E226's squared-cosine share
(`commute.kstar_vpd.grades`) inadmissible under the frozen definition (theory/2026-11-22_definition_frozen.md): a
cosine per setting, so the gain may vary with the setting; the sign ignored; settings from M's run as well as F's.
Here each variable u gets one gain across all settings, positive only, on the settings of F's own run on D:

    cos2_u = 1 - min_{c > 0} sum |write_F - c write_M|^2 / sum |write_F|^2 = num^2 / (den |F|^2)   (num > 0)

from the per-variable sums `commute.stand_in.write_change` saved for every item E268 (P1.54) read: num = sum
<write_F, write_M>, den = sum |write_M|^2, dd = sum |write_F - write_M|^2 over every position of D in F's own run
(each mechanism evaluated on the one matrix input y of F's run), so |F|^2 = dd + 2 num - den. Read over E268's K
(`in_K`) and the changed variables, beside E_b over the same set (recomputed: the transcription check), and against
the reference verdict R = K from the same arrays at E268's kept floor. Settings and filed expected values in
results/predictions.json; the run refuses unless it is committed unchanged.

Stage 0 (edge cases, synthetic kept variables, the do-nothing on x2's saved path) runs first; any miss stops the run.
Then every item E268 read, into results/summary.json. CPU, a few seconds.

    ~/commute-agents/fleet/cpu-run --mem 4G python3 experiments/cos_admissible/run.py [OUT_DIR]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
from torch import Tensor

from commute import stand_in as SI
from commute.jobs import EXPERIMENTS, config_from_settings, require_filed

HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E268 = EXPERIMENTS / "kstar_needed" / "results"
E247 = EXPERIMENTS / "thin_edit_diagnosis" / "results" / "large"  # built edits' num, den, dd
E242 = EXPERIMENTS / "kstar_vpd_wording_a" / "results" / "large"  # built edits' changed sets


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    built: tuple[str, ...]
    gain_edits: tuple[str, ...]
    random_edits: tuple[str, ...]
    control: str
    top_tol: float
    transcription_rel_tol: float
    kept_floor: float
    synthetic_gains: tuple[float, ...]

    def __post_init__(self) -> None:
        assert set(self.gain_edits) | set(self.random_edits) | {self.control} <= set(self.built), self
        assert 0 < self.top_tol < 1e-6 and 0 < self.transcription_rel_tol < 1e-6 and 0 < self.kept_floor < 1, self
        assert self.synthetic_gains and all(c > 0 for c in self.synthetic_gains), self


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def stop(msg: str) -> NoReturn:
    print(f"STOP: {msg}", flush=True)
    sys.exit(1)


# ---- the reading ----------------------------------------------------------------------------------------------------


def cos2(num: Tensor, den: Tensor, dd: Tensor) -> tuple[Tensor, float]:
    """(n,) fp64 cos2_u (predictions.json readings) and its largest unclipped excess over 1."""
    nF = (dd + 2 * num - den).clamp(min=0)
    pos = (num > 0) & (den > 0) & (nF > 0)
    raw = torch.where(pos, num**2 / torch.where(pos, den * nF, 1.0), 0.0)
    out = torch.where((den == 0) & (nF == 0), 1.0, raw.clamp(max=1.0))
    return out, float((raw - 1).max()) if len(raw) else 0.0


def mean_over(c: Tensor, sel: Tensor) -> float:
    """The mean of c over sel; 1 when sel is empty (no changed needed variable)."""
    return float(c[sel].mean()) if bool(sel.any()) else 1.0


def kept_by_share(num: Tensor, den: Tensor, dd: Tensor, floor: float, tol: float) -> Tensor:
    """(n,) bool: the reference kept test, explained / dd >= 1 - floor (E268's), or a write change within fp64 rounding
    of M's write, dd <= tol den. Amended 2026-11-23 after the filed run (which had dd = 0 exactly): on 60 changed
    needed variables dd / den was 4e-28 to 3e-17, the change being rounding alone, where the explained part
    (num - den)^2 / den is a difference of two equal sums and the share is noise; the next smallest dd / den on any item
    is 7.4e-7, ten orders above."""
    share = SI.explained(num, den) / torch.where(dd > 0, dd, 1.0)
    return (dd <= tol * den) | (share >= 1 - floor)


# ---- saved arrays ---------------------------------------------------------------------------------------------------


def fname(item: str) -> str:
    return item.replace("/", "-")


def arrays(rec: dict[str, Any], cfg: Config) -> dict[str, Tensor]:
    """num, den, dd (fp64), changed, K (bool) for one E268 item record, from the files E268 read."""
    f = fname(rec["item"])
    if rec["item"] in cfg.built:
        e = np.load(E247 / f"items_{f}.npz")
        a = {k: e[k] for k in ("num", "den", "dd")} | {"changed": np.load(E242 / f"items_{f}.npz")["changed"]}
    else:
        z = np.load(EXPERIMENTS.parent / rec["arrays"])
        a = {k: z[k] for k in ("num", "den", "dd", "changed")}
    a["K"] = np.load(E268 / "large" / f"items_{f}.npz")["in_K"]
    out = {k: torch.from_numpy(np.asarray(v)) for k, v in a.items()}
    out["changed"], out["K"] = out["changed"].bool(), out["K"].bool()
    if len({len(v) for v in out.values()}) != 1:
        stop(f"{rec['item']}: array lengths differ {[(k, len(v)) for k, v in out.items()]}")
    return out


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def edge_cases() -> list[str]:
    def t(*v: float) -> Tensor:
        return torch.tensor(v, dtype=torch.float64)

    cases = {  # name: (num, den, dd, expected cos2_u)
        "both write zero": (t(0.0), t(0.0), t(0.0), 1.0),
        "M zero, F not": (t(0.0), t(0.0), t(2.0), 0.0),
        "F zero, M not": (t(0.0), t(3.0), t(3.0), 0.0),
        "negative gain": (t(-3.0), t(3.0), t(12.0), 0.0),
        "zero gain": (t(0.0), t(3.0), t(5.0), 0.0),
    }
    miss = [f"{k}: {float(cos2(n, d, x)[0][0])} != {want}" for k, (n, d, x, want) in cases.items() if float(cos2(n, d, x)[0][0]) != want]
    empty = mean_over(t(0.5, 0.2), torch.zeros(2, dtype=torch.bool))
    return miss + ([f"empty K: {empty} != 1.0"] if empty != 1.0 else [])


def stage0(cfg: Config, rec: dict[str, Any]) -> dict[str, Any]:
    a = arrays(rec, cfg)
    den = a["den"]
    on = den > 0
    miss = edge_cases()
    short = {}
    for c in cfg.synthetic_gains:
        v, _ = cos2(c * den, den, (c - 1) ** 2 * den)
        short[str(c)] = float((1 - v[on]).max())
    neg, _ = cos2(-den, den, 4 * den)
    miss += [f"synthetic kept c={c}: shortfall {s!r} > top_tol" for c, s in short.items() if s > cfg.top_tol]
    miss += [f"synthetic c=-1: max {float(neg[on].max())} != 0"] if bool((neg[on] != 0).any()) else []
    un = ~a["changed"]
    cu, _ = cos2(a["num"], den, a["dd"])
    dn = mean_over(cu, torch.zeros_like(un))
    miss += [f"{cfg.control} unchanged: max dd {float(a['dd'][un].abs().max())}"] if bool((a["dd"][un] != 0).any()) else []
    miss += [f"{cfg.control} unchanged: min cos2 {float(cu[un].min())}"] if bool((cu[un] != 1.0).any()) else []
    miss += [f"do-nothing cos2_K {dn} != 1.0"] if dn != 1.0 else []
    out = {"synthetic_shortfall": short, "floor": max(short.values()), "n_unchanged": int(un.sum()), "do_nothing_cos2_K": dn, "misses": miss}
    print(f"stage 0: {json.dumps(out)}", flush=True)
    if miss:
        stop("stage 0: " + "; ".join(miss))
    return out


# ---- items ----------------------------------------------------------------------------------------------------------


def read(rec: dict[str, Any], cfg: Config) -> dict[str, Any]:
    a = arrays(rec, cfg)
    num, den, dd = a["num"], a["den"], a["dd"]
    sel = a["K"] & a["changed"]
    c, over = cos2(num, den, dd)
    eb = SI.explained_share(num, den, dd, sel)
    rel = abs(eb - rec["E_b_K"]) / abs(rec["E_b_K"]) if rec["E_b_K"] else abs(eb)
    at_top, kept = c >= 1 - cfg.top_tol, kept_by_share(num, den, dd, cfg.kept_floor, cfg.top_tol)
    nF = (dd + 2 * num - den).clamp(min=0)
    rc = (dd / torch.where(nF > 0, nF, 1.0))[sel & (nF > 0)]
    share = SI.explained(num, den) / torch.where(dd > 0, dd, 1.0)
    on = sel & (num > 0) & (den > 0) & (nF > 0) & (dd > cfg.top_tol * den)  # the identity's terms all defined
    want = ((1 - share) * dd / torch.where(nF > 0, nF, 1.0))[on]  # 1 - cos2_u = (1 - share_u) dd / |F|^2
    out = {"item": rec["item"], "stage": rec["stage"], "size_K": int(a["K"].sum()), "n_changed_K": int(sel.sum()),
           "cos2_K": mean_over(c, sel), "E_b_K": eb, "E_b_K_e268": rec["E_b_K"], "E_b_K_rel_dev": rel,
           "top": bool(at_top[sel].all()), "R_eq_K": bool(kept[sel].all()), "n_top": int((at_top & sel).sum()),
           "n_kept": int((kept & sel).sum()), "n_disagree": int(((at_top ^ kept) & sel).sum()),
           "rel_change_median": float(rc.median()) if len(rc) else None, "overshoot": over,
           "identity_max_rel": float(((1 - c[on] - want).abs() / want).max()) if len(want) else None}  # fmt: skip
    if rel > cfg.transcription_rel_tol:
        stop(f"{rec['item']}: E_b over K {eb!r} off E268's {rec['E_b_K']!r} (relative {rel:.3g})")
    if over > cfg.top_tol:
        stop(f"{rec['item']}: cos2 exceeds 1 by {over:.3g}: num, den, dd do not fit one pair of writes")
    print(f"{rec['item']}: cos2 over K {out['cos2_K']:.6f}, E_b over K {eb:.6f}; {out['n_changed_K']} changed in K, top {out['n_top']}, "
          f"kept {out['n_kept']}, disagree {out['n_disagree']}; dd/|F|^2 median {out['rel_change_median']}", flush=True)  # fmt: skip
    return out


def ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        r[v == x] = r[v == x].mean()
    return r


def agreement(rows: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    c, e = np.array([r["cos2_K"] for r in rows]), np.array([r["E_b_K"] for r in rows])
    by = {r["item"]: r["cos2_K"] for r in rows}
    gain = min(by[i] for i in cfg.gain_edits)
    rest = [by[i] for i in cfg.built if i not in cfg.gain_edits and i != "rebuilt"]
    ch = [r for r in rows if r["n_changed_K"] > 0]
    short, size = np.array([1 - r["cos2_K"] for r in ch]), np.array([r["rel_change_median"] for r in ch])
    return {"spearman": float(np.corrcoef(ranks(c), ranks(e))[0, 1]),
            "spearman_shortfall_vs_change": float(np.corrcoef(ranks(short), ranks(size))[0, 1]),
            "identity_max_rel": max(r["identity_max_rel"] for r in ch if r["identity_max_rel"] is not None),
            "median_abs_gap": float(np.median(np.abs(c - e))),
            "built_order_holds": bool(gain > by["rebuilt"] > max(rest)),
            "gain_mean_minus_random_mean": float(np.mean([by[i] for i in cfg.gain_edits]) - np.mean([by[i] for i in cfg.random_edits])),
            "admissible": all(r["top"] == r["R_eq_K"] for r in rows), "n_disagree_total": sum(r["n_disagree"] for r in rows)}  # fmt: skip


def main(out_dir: Path) -> None:
    cfg = load_config()
    print(f"{EXPERIMENT} start; settings {asdict(cfg)}", flush=True)
    recs = {k: r for k, r in json.loads((E268 / "summary.json").read_text())["items"].items() if r.get("reading") == "read"}
    s0 = stage0(cfg, recs[cfg.control])
    rows = [read(recs[k], cfg) for k in sorted(recs)]
    summ = {"experiment": EXPERIMENT, "settings": asdict(cfg), "stage0": s0, "agreement": agreement(rows, cfg), "items": {r["item"]: r for r in rows}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(f"agreement: {json.dumps(summ['agreement'])}", flush=True)
    print(f"{EXPERIMENT} DONE: {len(rows)} items into {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    require_filed(PRED)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results")
