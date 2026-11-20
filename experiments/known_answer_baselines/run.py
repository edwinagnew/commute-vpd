"""E227 (cell P2.43): the re-gating share, effective rank and neurons for half the gain (k_half) read on E214's
known-answer items, scored by R's rule (every reuse item above every new item, by more than a split-half band). Plan
reports/plans/2026-11-20_plan_p2-43-known-answer-baselines.md; settings, expected values and rules in
results/predictions.json (the run refuses unless it is committed unchanged). The readings are commute.realistic_reads
(share_of, k_half_of; E211's readers) and commute.update_spectrum (erank); the items are E214's builders
(experiments/kstar_levels/run.py, World.make), read on the windows E214 read them on. Nothing is trained.

One job, stages in order, each writing its JSON as it ends and printing every number:
  0  controls, alone, exit 1 on a miss: E53's lowercase_s0 read by these readers reproduces its saved share (E136),
     k_half and r_half numerator (E98) and erank (E141); then E214's edits are built (so every cue item shares one set
     of windows) and the do-nothing item (zero change, E214's cue windows) reads every numerator and gain exactly 0
  1  new_few, new_thin, x2, x10, moved, rebuilt: per item its change of M's 24 decomposed matrices (or "does not apply"
     when F has neurons or parameters M lacks), a gate (M plus that change gives F's logits, exit 1 on a miss), then the
     share and k_half on all rows and on two halves, and erank -> results/items/<item>.json (a rerun skips it)
  2  score: each method's band and rules P1 and P1b -> results/score.json (also alone, no GPU: `run.py score`)

    sbatch --comment=E227 cluster/known_answer_baselines.sbatch          (2>&1 into cluster/logs; ~15 min on an L40)
    python experiments/known_answer_baselines/run.py score
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from commute import kstar_vpd as KV
from commute import realistic_reads as RR
from commute import update_spectrum as US
from commute.data import target_tokens
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

KL: Any = load_experiment("kstar_levels", "run")  # E214: World (the items), Config, COPY24 through KL.P
WIN: Any = load_experiment("share_untrained", "windows")  # E136's read windows
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS = RESULTS / "predictions.json", RESULTS / "items"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
CUE_ITEMS = ("do_nothing", "new_few", "new_thin")  # E214's cue windows, last position; the others copy24's, at half
METHODS = ("share", "k_half", "erank")
E136, E98 = EXPERIMENTS / "share_untrained" / "results", EXPERIMENTS / "neuron_r" / "results"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_stage(path, jsonable(obj))


def stop(path: Path, obj: dict[str, Any], why: str) -> None:
    dump(path, obj | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    reuse: tuple[str, ...]
    new: tuple[str, ...]
    split_seed: int
    control_state: str  # E136 inventory name of the positive control

    def __post_init__(self) -> None:
        assert set(self.reuse) == {"x2", "x10", "moved", "rebuilt"} and set(self.new) == {"new_few", "new_thin"}


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def e214_world() -> Any:
    """E214's item builders as E226 builds them; E214's build record goes to our results, not E214's folder."""
    KL.RESULTS = RESULTS / "e214_build"  # World.edits writes RESULTS / "edits.json"
    KL.RESULTS.mkdir(parents=True, exist_ok=True)
    pred = json.loads(KL.PRED.read_text())
    s = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return KL.World(config_from_settings(KL.Config, s), pred["thresholds"])


# ---- one item -------------------------------------------------------------------------------------------------------


def item_read(w: Any, item: str) -> tuple[Any, RR.Read]:
    """F and the item's rows in the LM layout: cue items' target y_i at the last position (E214's lookup answer), copy24
    items' target the task table's answer at half (the answer E195's circuit C was found on)."""
    F, toks, _, piece, _ = w.make(item)
    assert piece is None
    if item in CUE_ITEMS:
        c = w.cues
        return F, RR.next_layout(item, toks.cpu(), torch.tensor(c.y)[c.cue], toks.shape[1] - 1)
    d = torch.load(KL.P.COPY24, weights_only=False)
    h = d["heldout"]
    assert torch.equal(h.tokens, toks.cpu())
    return F, RR.next_layout(item, h.tokens, target_tokens(d["table"], h.tokens, h.half), h.half)


def weight_change(net: Any, M: Any, F: Any) -> tuple[dict[str, torch.Tensor] | None, str]:
    """F - M on the 24 decomposed matrices, or None with the reason the item is not such a change."""
    if F.extra:
        return None, f"F appends neurons at layers {sorted(F.extra)}: not a change of M's 24 matrices"
    if diffs := KV.undecomposed_differences(M, F, list(net.W0)):
        return None, f"parameters outside the 24 matrices differ: {diffs[:4]}"
    return {k: (F.net.get_submodule(k).weight.detach().float() - w).cpu() for k, w in net.W0.items()}, ""


@torch.no_grad()
def gate(net: Any, F: Any, r: RR.Read, dW: dict[str, torch.Tensor]) -> float:
    """Max |F's logits - (M + dW)'s logits| over max |F's| at the read position, first 64 rows."""
    t = r.toks[:64].to(net.device)
    a = F.logits(t)[:, r.pos]
    with net.at(dW):
        b = net.run(t)[0][:, r.pos]
    return float((a - b).abs().max() / a.abs().max())


def readings(net: Any, r: RR.Read, dW: dict[str, torch.Tensor], S: dict[str, Any]) -> dict[str, Any]:
    sh, kh = RR.share_of(net, r, dW), RR.k_half_of(net, r, dW, S)["lo"]
    return {"n_rows": len(r.toks), "share": sh["share"]["lo"], "share_num": sh["numerator"]["lo"], "gain": sh["gain"]["lo"],
            "k_half": kh["k_half"], "r_half": kh["r_half"], "r_half_num": kh["r_half_num"], "G": kh["G"],
            "k_grid_numerators": kh["numerators"]}  # fmt: skip


def read_item(w: Any, item: str, cfg: Config, th: dict[str, Any], S: dict[str, Any]) -> dict[str, Any]:
    path = ITEMS / f"{item}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    F, r = item_read(w, item)
    dW, why = weight_change(w.net, w.M, F)
    rec: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "n_rows": len(r.toks), "pos": r.pos}
    if dW is None:
        rec |= {"reading": "does not apply", "why": why}
        log(f"{item}: does not apply ({why})")
        dump(path, rec)
        return rec
    rec["gate"] = gate(w.net, F, r, dW)
    rec["erank"], rec["norm2"] = (sp := US.state(dW))["erank"], sp["norm2"]
    log(f"{item}: {len(r.toks)} rows, pos {r.pos}; gate {rec['gate']:.3g} (tol {th['gate_tol']}); erank {rec['erank']}, |dW|^2 {rec['norm2']:.6g}")
    if not rec["gate"] <= th["gate_tol"]:
        stop(path, rec, f"{item}: M plus its weight change does not give F's logits ({rec['gate']:.3g})")
    perm = torch.randperm(len(r.toks), generator=torch.Generator().manual_seed(cfg.split_seed))
    sets = {"all": r, "half1": RR.subset(r, perm[: len(perm) // 2]), "half2": RR.subset(r, perm[len(perm) // 2 :])}
    rec["rows"] = {}
    for name, rr in sets.items():
        rec["rows"][name] = readings(w.net, rr, dW, S)
        log(f"{item} {name}: " + json.dumps(jsonable({k: v for k, v in rec["rows"][name].items() if k != "k_grid_numerators"})))
    rec |= {"reading": "read", "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, rec)
    log(f"{item}: done in {rec['seconds']:.0f} s, peak GPU {rec['peak_gpu_gb']:.2f} GB")
    del F
    torch.cuda.empty_cache()
    return rec


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def stage0(w: Any, cfg: Config, th: dict[str, Any], S: dict[str, Any]) -> None:
    path = RESULTS / "stage0.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log("stage 0: done before")
        return
    inv = json.loads((E136 / "inventory.json").read_text())
    name, tol = cfg.control_state, th["reproduce_tol"]
    task = inv["states"][name]["task"]
    wr = WIN.load_read(task, inv["tasks"][task])
    r = RR.next_layout(task, wr.toks, target_tokens(wr.token, wr.toks, wr.half), wr.half)
    dW = {k: v.float() for k, v in torch.load(EXPERIMENTS.parent / inv["states"][name]["file"], weights_only=False).items()}
    got = readings(w.net, r, dW, S) | {"erank": US.state(dW)["erank"]}
    e98 = json.loads((E98 / "carrying.json").read_text())["rows"][name.split("/")[1]]
    svd = json.loads((EXPERIMENTS / "update_svd/results/results.json").read_text())["pairs"]["e53_lowercase_cipher"]["reuse"][name.split("/")[1]]
    saved = {"share": json.loads((E136 / "stage1.json").read_text())["states"][name]["trained"], "k_half": e98["k_half"],
             "r_half_num": e98["r_half_num"], "erank": svd["erank"]}  # fmt: skip
    out: dict[str, Any] = {"experiment": EXPERIMENT, "control": {k: {"got": got[k], "saved": saved[k], "tol": tol[k]} for k in saved}}
    log(f"stage 0 control {name}: {json.dumps(jsonable(out['control']))}")
    miss = [k for k, v in out["control"].items() if not (v["got"] is not None and abs(v["got"] - v["saved"]) <= v["tol"])]
    _ = w.edits  # build E214's edits before the do-nothing read: a detector swap rebuilds the cue windows (E214's edits.json)
    F, rd = item_read(w, "do_nothing")
    zero, _ = weight_change(w.net, w.M, F)
    assert zero is not None
    nothing = readings(w.net, rd, zero, S) | {"erank": US.state(zero)["erank"], "gate": gate(w.net, F, rd, zero)}
    out["do_nothing"] = nothing
    log(f"stage 0 do-nothing: {json.dumps(jsonable({k: v for k, v in nothing.items() if k != 'k_grid_numerators'}))}")
    miss += [f"do-nothing {k} {nothing[k]}" for k in ("share_num", "gain", "r_half_num", "G") if nothing[k] != 0]
    miss += [f"do-nothing k_half {nothing['k_half']}, erank {nothing['erank']}"] if not (nothing["k_half"] is None and nothing["erank"] is None) else []
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


# ---- score (no GPU) -------------------------------------------------------------------------------------------------


def value(rec: dict[str, Any], method: str, rows: str = "all") -> float:
    """The item's reading oriented so that higher = more reuse: share as read, -log2 k_half (fewer neurons = reuse),
    erank as read (E141 P3: the reuse side higher); nan when unreadable or the item does not apply."""
    if rec["reading"] != "read":
        return math.nan
    if method == "erank":
        return math.nan if rec["erank"] is None else float(rec["erank"])
    x = rec["rows"][rows][method]
    if method == "k_half":
        return -math.log2(x) if x else math.nan
    return float(x) if isinstance(x, int | float) else math.nan  # the share is "nan" (a string) once written


def band(recs: dict[str, dict[str, Any]], method: str, band_min: dict[str, float]) -> dict[str, Any]:
    if method == "erank":
        return {"band": 0.0, "source": "exact: erank is a function of the weight change alone, no rows"}
    d = [abs(value(r, method, "half1") - value(r, method, "half2")) for r in recs.values()]
    d = [x for x in d if math.isfinite(x)]
    if not d or max(d) == 0:
        return {"band": band_min[method], "source": "band_min (no finite non-zero split-half gap)", "gaps": d}
    return {"band": max(d), "source": "largest split-half gap over the items", "gaps": d}


def rule(recs: dict[str, dict[str, Any]], method: str, reuse: tuple[str, ...], new: tuple[str, ...], b: float) -> dict[str, Any]:
    v = {i: value(recs[i], method) for i in (*reuse, *new)}
    bad = [i for i, x in v.items() if not math.isfinite(x)]
    if bad:
        return {"verdict": "not read", "unreadable": bad, "values": v}
    gap = min(v[i] for i in reuse) - max(v[i] for i in new)
    flipped = min(v[i] for i in new) - max(v[i] for i in reuse)  # reported only: the order the other way round
    return {"verdict": "pass" if gap > b else "fail", "gap": gap, "band": b, "reversed_gap": flipped, "values": v}


def score(cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    recs = {i: json.loads((ITEMS / f"{i}.json").read_text()) for i in (*cfg.reuse, *cfg.new)}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "methods": {}}
    for m in METHODS:
        b = band(recs, m, th["band_min"])
        out["methods"][m] = {"band": b, "P1": rule(recs, m, cfg.reuse, cfg.new, b["band"]),
                             "P1b": rule(recs, m, tuple(i for i in cfg.reuse if i != "moved"), cfg.new, b["band"])}  # fmt: skip
        log(f"score {m}: band {b['band']:.4g}; P1 {json.dumps(jsonable(out['methods'][m]['P1']))}; P1b {json.dumps(jsonable(out['methods'][m]['P1b']))}")
    out["passing"] = {k: [m for m in METHODS if out["methods"][m][k]["verdict"] == "pass"] for k in ("P1", "P1b")}
    dump(RESULTS / "score.json", out)
    log(f"methods passing: {out['passing']}")
    return out


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, th = load_config()
    if argv == ["score"]:
        score(cfg, th)
        return 0
    assert not argv, f"usage: run.py [score], got {argv}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E211 and E214
    S = json.loads((E98 / "predictions.json").read_text())["settings"]
    log(f"{EXPERIMENT} start; settings {asdict(cfg)}; thresholds {th}")
    w = e214_world()
    stage0(w, cfg, th, S)
    for k, item in enumerate((*cfg.new, *cfg.reuse)):
        read_item(w, item, cfg, th, S)
        log(f"ITEM {k + 1} of {len(cfg.new) + len(cfg.reuse)} DONE ({item})")
    score(cfg, th)
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
