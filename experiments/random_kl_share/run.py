"""E228 (cell P2.44, widened by the user 2026-11-20 12:24): the re-gating share in Qwen's KL form (primary) and in the
67M's log-odds form r (beside it) on every saved fine-tune of the nine built pairs and of the scorecard's four
shuffled-label twins, on random weight changes matched to each fine-tune's output change (KL to W0 on its read rows),
and on one structured control of matched KL that reads 0 by construction. Plan
reports/plans/2026-11-20_plan_p2-44-random-kl-matched.md; settings, expected values and rules in results/predictions.json
(the run refuses unless it is committed unchanged). Nothing is trained.

Readings of a change, from one pass of forwards (`masked_reads.route_cells_by`) on the fine-tune's E136/E147 read rows,
fed whole as their source fed them (`realistic_reads.target_read`; Gamma reads the whole row):
  kl_share  E57's re-gating share (K00 - K01) / (K00 - K11), each K a row's KL summed over every position up to the read
            position; K10, K11 on the components carrying the change's least-squares split onto round 1's alive
            subcomponents; nan when K00 - K11 <= 0; `qwen_reading.shares`' status reported
  r         GOALS.md's (L01 - L00) / (T1 - T0) in log-odds of the trained answer at the read position, nan under a gain of 1
Changes per fine-tune: the fine-tune itself (all rows and two halves); three Gaussian directions at its per-matrix norms
(`finetune.random_direction`) scaled by `perturb.match` to its KL; the structured control: a Gaussian change of the last
decomposed matrix (h.3.mlp.down_proj) projected inside its alive span, at the fine-tune's total norm, scaled the same
way. Gamma reads no input that matrix moves, so the gates do not change (K01 = K00 exactly, a stop if not) while the
split expresses it: kl_share reads 0 by construction.

Jobs (one per pod, any order; each first runs the zero-change check on its first state's rows, exit 1 on a miss):
  states I N   every N-th state from the I-th: the fine-tune (r must reproduce its saved share, exit 1 on a miss), its
               draws and the structured control -> results/states/<state>.json as each state ends; a rerun skips them
  score        no GPU: bands, per-state verdicts, rules K1 and P1, the pair rule, the control check -> score.json
  manifest N   no GPU: the inputs outside git per shard (for sync) -> results/manifest_<I>_of_<N>.txt

    PRED_SHA=<sha> python experiments/random_kl_share/run.py states 0 3 2>&1 | tee -a experiments/random_kl_share/results/pod0.log
    python experiments/random_kl_share/run.py score
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from commute import component_training as CT
from commute import finetune as FT
from commute import masked_reads as MR
from commute import perturb as PT
from commute import qwen_reading as QR
from commute import realistic_reads as RR
from commute import split
from commute import steering as ST
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

WIN: Any = load_experiment("share_untrained", "windows")  # E136's read windows
SM: Any = load_experiment("size_matched", "run")  # E187: the nine built pairs' state names and their inventory
R96: Any = load_experiment("refit_96key", "run")  # E147: the 96-key states, their inventory and saved reads
HERE = Path(__file__).resolve().parent
RESULTS, STATES = HERE / "results", HERE / "results" / "states"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E136 = EXPERIMENTS / "share_untrained" / "results"
E147 = EXPERIMENTS / "refit_96key" / "results"
ALIVE = EXPERIMENTS / "transport_test" / "results" / "large" / "alive.pt"  # round 1's alive masks (the paper's rule)
LAST = "h.3.mlp.down_proj"  # the last decomposed matrix: no decomposed matrix reads its output
TWINS = {"e123_case_label": "case_label", "e125_repeat_label": "fresh_reuse_pair"}  # scorecard twin -> E136 pair id
E58_TWINS = ("e58_elicit_shuffle", "e58_class_twin")
KEYS = ("kl_share", "r")


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
    seeds: tuple[int, ...]
    control_seed: int
    scale_cap: float
    kl_rel_tol: float
    split_seed: int
    batch: int  # rows per KL forward
    split_tol: float

    def __post_init__(self) -> None:
        assert len(set(self.seeds)) == len(self.seeds) >= 2 and self.control_seed not in self.seeds


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def fname(state: str) -> str:
    return state.replace("/", "_")


# ---- the catalogue: pairs, states, rows, saved shares ---------------------------------------------------------------


def catalogue() -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, Any], dict[str, Any]]:
    """The thirteen pairs (E187's nine built pairs, then the scorecard's four twins), each (reuse side, new side) in
    E136/E147 state names; one inventory holding every task and state; E147's inventory (its saved reads)."""
    names, _, _ = SM.pairs()
    e147 = json.loads((E147 / "inventory.json").read_text())
    _, full = R96.inventories(e147)
    e136 = {p["id"]: (p["reuse"], p["new"]) for p in full["pairs"]}
    pairs = {p: names[p] for p in SM.BUILT}
    pairs |= {t: e136[p] for t, p in TWINS.items()}
    for p in e147["pairs"]:
        if p["id"] in E58_TWINS:
            pairs[p["id"]] = ([R96.state_name(p["reuse"], s) for s in e147["seeds"]], [R96.state_name(p["new"], s) for s in e147["seeds"]])
    return pairs, full, e147


def every_state(pairs: dict[str, tuple[list[str], list[str]]]) -> list[str]:
    return list(dict.fromkeys(s for r, n in pairs.values() for s in (*r, *n)))


def saved_r(state: str, e147: dict[str, Any], stage1: dict[str, Any]) -> float:
    if state.startswith("e58/"):
        m, s = state[4:].rsplit("_s", 1)
        return float(R96.saved_read(e147, m, int(s))["r"])
    return float(stage1[state]["trained"])


def state_read(state: str, inv: dict[str, Any]) -> tuple[RR.Read, dict[str, Tensor]]:
    """A fine-tune's read rows, fed as its source fed them, its target apart; and its saved weight change, fp32 CPU."""
    task = inv["states"][state]["task"]
    w = WIN.load_read(task, inv["tasks"][task])
    d = torch.load(EXPERIMENTS.parent / inv["states"][state]["file"], weights_only=False)
    return RR.target_read(task, w.toks, w.token, w.half, w.ci_len), {k: v.float() for k, v in d.items()}


# ---- KL, the split, the readings ------------------------------------------------------------------------------------


@torch.no_grad()
def kl_of(net: ST.Net, r: RR.Read, dW: dict[str, Tensor], batch: int) -> float:
    """Mean KL(W0 + dW || W0) per row and position over every position up to the read position (the plain causal
    model, no masks): the change's output size on the read rows."""
    tot, n = 0.0, 0
    for s in range(0, len(r.toks), batch):
        t = r.toks[s : s + batch, : r.pos + 1]
        p0 = net.run(t)[0].log_softmax(-1)
        with net.at(dW):
            p1 = net.run(t)[0].log_softmax(-1)
        k = (p1.exp() * (p1 - p0)).sum(-1)
        tot, n = tot + float(k.double().sum()), n + k.numel()
    return tot / n


class Splitter:
    """Round 1's alive masks and the least-squares split's bases at W0 (once), and the components carrying a change's split."""

    def __init__(self, net: ST.Net, tol: float) -> None:
        self.model = net.model
        self.alive = {k: v.cpu().bool() for k, v in torch.load(ALIVE, weights_only=False)["alive"].items()}
        self.uv = {k: (u.cpu(), v.cpu()) for k, (u, v) in CT.alive_uv(self.model, self.alive).items()}
        t0 = time.time()
        self.bases = split.bases(self.uv, tol)
        log(f"split bases: {len(self.bases)} matrices, {sum(int(a.sum()) for a in self.alive.values())} alive, {time.time() - t0:.0f} s")

    def comps(self, dW: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
        st = split.split_state(dW, self.uv, self.bases)
        return CT.components_with(self.model, self.alive, {k: (v[1], v[2]) for k, v in st.items()})


def summary(cells: MR.Cells, K: MR.KLCells, idx: Tensor, n_pos: int) -> dict[str, Any]:
    """Both readings on the rows idx: r from the log-odds cells, kl_share from the KL sums (n_pos positions a row)."""
    m = lambda c: float(cells[c]["lo"][idx].mean())
    num, gain = m("L01") - m("L00"), m("T1") - m("T0")
    sums = {k: float(v[idx].sum()) for k, v in K.items()}
    s = QR.shares(sums, len(idx) * n_pos)
    return {"kl_share": s["regating"] if sums["K00"] - sums["K11"] > 0 else math.nan, "kl_status": s["status"],
            "K": {k: v / (len(idx) * n_pos) for k, v in sums.items()}, "r": RR.ratio(num, gain), "numerator": num, "gain": gain}  # fmt: skip


def readings(net: ST.Net, sp: Splitter, r: RR.Read, dW: dict[str, Tensor], halves: Tensor | None = None) -> dict[str, Any]:
    t0 = time.time()
    cells, K = MR.route_cells_by(net.model, net.W0, dW, r.toks, r.pos, r.units_of(r.tgt), r.pos, sp.comps(dW))
    out = summary(cells, K, torch.arange(len(r.toks)), r.pos + 1)
    if halves is not None:
        h = len(halves) // 2
        out["halves"] = [summary(cells, K, halves[:h], r.pos + 1), summary(cells, K, halves[h:], r.pos + 1)]
    return out | {"read_seconds": time.time() - t0}


def structured(dW: dict[str, Tensor], b: split.Basis, seed: int) -> dict[str, Tensor]:
    """The structured control's direction: a Gaussian on LAST with its part outside the alive span removed (the split
    expresses the rest), at the fine-tune's total Frobenius norm; every other matrix zero."""
    D = torch.randn(dW[LAST].shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    D = D - split.obstruction(D, b)
    total = math.sqrt(sum(float(v.double().pow(2).sum()) for v in dW.values()))
    return {k: (D * (total / float(D.norm()))).float() if k == LAST else torch.zeros_like(v) for k, v in dW.items()}


def matched(net: ST.Net, sp: Splitter, r: RR.Read, d: dict[str, Tensor], target: float, cfg: Config) -> dict[str, Any]:
    """Direction d scaled so its KL matches target, and both readings of the scaled change."""
    k1 = kl_of(net, r, d, cfg.batch)
    if not (math.isfinite(k1) and k1 > 0):
        return {"kl_at_1": k1, "match": "unscaled KL not positive"}
    m = PT.match(lambda x: kl_of(net, r, PT.scaled(d, x), cfg.batch), target, cfg.scale_cap, cfg.kl_rel_tol)
    return {"kl_at_1": k1, "c": m["x"], "kl": m["f"], "match": m["status"], "evals": len(m["grid"]) + len(m["trace"])} | readings(net, sp, r, PT.scaled(d, m["x"]))


# ---- jobs -----------------------------------------------------------------------------------------------------------


def zero_check(net: ST.Net, sp: Splitter, state: str, inv: dict[str, Any], cfg: Config, tag: str) -> None:
    """The do-nothing: a zero change reads r's numerator and gain, the KL, and K00 - K0, K01 - K0, K10 - K00, K11 - K01
    exactly 0 (E57's identity check: the unchanged model as the changed one, a zero split equal to the release)."""
    path = RESULTS / f"zero_{tag}.json"
    r, dW = state_read(state, inv)
    zero = {k: torch.zeros_like(v) for k, v in dW.items()}
    z = readings(net, sp, r, zero) | {"kl": kl_of(net, r, zero, cfg.batch), "state": state}
    K = z["K"]
    z["exact"] = {"numerator": z["numerator"], "gain": z["gain"], "kl": z["kl"], "K00-K0": K["K00"] - K["K0"], "K01-K0": K["K01"] - K["K0"],
                  "K10-K00": K["K10"] - K["K00"], "K11-K01": K["K11"] - K["K01"]}  # fmt: skip
    log(f"[zero {state}] {json.dumps(jsonable(z))}")
    if bad := {k: v for k, v in z["exact"].items() if v != 0}:
        stop(path, z, f"a zero change does not read exactly 0: {bad}")
    dump(path, z | {"done": True})
    log("STAGE zero DONE")


def one_state(net: ST.Net, sp: Splitter, state: str, inv: dict[str, Any], saved: float, cfg: Config, th: dict[str, Any]) -> None:
    path = STATES / f"{fname(state)}.json"
    rec: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    if rec.get("done"):
        log(f"{state}: done before, skipped")
        return
    t0 = time.time()
    r, dW = state_read(state, inv)
    if "own" not in rec:
        perm = torch.randperm(len(r.toks), generator=torch.Generator().manual_seed(cfg.split_seed))
        rec = {"experiment": EXPERIMENT, "state": state, "n_rows": len(r.toks), "pos": r.pos, "saved_r": saved, "draws": {},
               "own": readings(net, sp, r, dW, perm), "kl_target": kl_of(net, r, dW, cfg.batch)}  # fmt: skip
        log(f"[{state} own] kl {rec['kl_target']:.6g}; {json.dumps(jsonable(rec['own']))}")
        if not abs(rec["own"]["r"] - saved) <= th["reproduce_tol"]:
            stop(path, rec, f"{state}: r {rec['own']['r']:.4f} does not reproduce its saved {saved:.4f}")
        dump(path, rec)
    for seed in cfg.seeds:
        if str(seed) not in rec["draws"]:
            rec["draws"][str(seed)] = matched(net, sp, r, FT.random_direction(dW, seed), rec["kl_target"], cfg)
            dump(path, rec)
            log(f"[{state} random {seed}] {json.dumps(jsonable(rec['draws'][str(seed)]))}")
    if "control" not in rec:
        c = matched(net, sp, r, structured(dW, sp.bases[LAST], cfg.control_seed), rec["kl_target"], cfg)
        rec["control"] = c
        log(f"[{state} control] {json.dumps(jsonable(c))}")
        if "K" in c and not abs(c["K"]["K01"] - c["K"]["K00"]) <= th["control_gate_tol"] * abs(c["K"]["K00"]):  # Gamma reads no input LAST moves
            stop(path, rec, f"{state}: the structured control moved the gates (K01 {c['K']['K01']} != K00 {c['K']['K00']})")
    rec |= {"done": True, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, rec)
    log(f"STATE {state} DONE in {rec['seconds']:.0f} s, peak GPU {rec['peak_gpu_gb']:.2f} GB")


def states_job(cfg: Config, th: dict[str, Any], i: int, n: int) -> None:
    pairs, inv, e147 = catalogue()
    todo = every_state(pairs)[i::n]
    stage1 = json.loads((E136 / "stage1.json").read_text())["states"]
    net = ST.Net("cuda")
    assert LAST == f"h.{len(net.net.h) - 1}.mlp.down_proj" and LAST in net.W0
    sp = Splitter(net, cfg.split_tol)
    zero_check(net, sp, todo[0], inv, cfg, f"{i}_of_{n}")
    for k, state in enumerate(todo):
        one_state(net, sp, state, inv, saved_r(state, e147, stage1), cfg, th)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({state})")
    log(f"{EXPERIMENT} states {i} {n} DONE")


# ---- score (no GPU) -------------------------------------------------------------------------------------------------


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan  # nan is written as the string "nan"


def band(recs: dict[str, dict[str, Any]], key: str, band_min: float) -> dict[str, Any]:
    """The pooled split-half spread: the root mean square over the fine-tunes of |half 1 - half 2| (an estimate of
    twice the sd of a whole-row reading); band_min when no gap is finite or all are 0."""
    g = [abs(num(x["own"]["halves"][0][key]) - num(x["own"]["halves"][1][key])) for x in recs.values()]
    g = [v for v in g if math.isfinite(v)]
    rms = math.sqrt(fmean(v * v for v in g)) if g else 0.0
    return {"band": rms if rms > 0 else band_min, "n": len(g), "source": "rms split-half gap" if rms > 0 else "band_min"}


def verdict(rec: dict[str, Any], key: str, b: float, min_scored: int) -> dict[str, Any]:
    own = num(rec["own"][key])
    scored = [num(d[key]) for d in rec["draws"].values() if d.get("match") == "ok" and math.isfinite(num(d.get(key)))]
    out = {"own": own, "scored": scored}
    if not math.isfinite(own) or len(scored) < min_scored:
        return out | {"verdict": "refused"}
    return out | {"verdict": "low" if max(scored) < own - b else "reaches"}


def overall(vs: list[str]) -> str:
    read = [v for v in vs if v != "refused"]
    if not read:
        return "silent: refused"
    n = sum(v == "reaches" for v in read)
    return "rival a: low" if n == 0 else "rival b: reaches" if 2 * n > len(read) else f"mixed: {n} of {len(read)} reach"


def pair_rule(recs: dict[str, dict[str, Any]], reuse: list[str], new: list[str], key: str) -> dict[str, Any]:
    """E51's pair rule on the fine-tunes' own readings: separates iff |mean reuse - mean new| > the larger side range."""
    a, b = [num(recs[s]["own"][key]) for s in reuse], [num(recs[s]["own"][key]) for s in new]
    if not all(math.isfinite(v) for v in a + b):
        return {"verdict": "not read", "reuse": a, "new": b}
    gap, rng = fmean(a) - fmean(b), max(max(a) - min(a), max(b) - min(b))
    return {"verdict": "separates" if abs(gap) > rng else "no", "gap": gap, "range": rng, "reuse": a, "new": b}


def score(th: dict[str, Any]) -> dict[str, Any]:
    pairs, _, _ = catalogue()
    recs = {s: json.loads((STATES / f"{fname(s)}.json").read_text()) for s in every_state(pairs) if (STATES / f"{fname(s)}.json").is_file()}
    recs = {s: v for s, v in recs.items() if v.get("done")}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "n_states": len(recs), "n_planned": len(every_state(pairs)), "bands": {}, "rules": {}}
    for rule, key in (("K1", "kl_share"), ("P1", "r")):
        b = band(recs, key, th["band_min"])
        per = {s: verdict(v, key, b["band"], th["min_scored"]) for s, v in recs.items()}
        out["bands"][key] = b
        out["rules"][rule] = {"reading": key, "overall": overall([v["verdict"] for v in per.values()]), "states": per,
                              "counts": {x: sum(v["verdict"] == x for v in per.values()) for x in ("low", "reaches", "refused")}}  # fmt: skip
        log(f"score {rule} ({key}): band {b['band']:.4g}; {out['rules'][rule]['overall']}; {out['rules'][rule]['counts']}")
    ctl = {s: num(v["control"].get("kl_share")) for s, v in recs.items() if v["control"].get("match") == "ok"}
    low = [s for s, x in ctl.items() if x < num(recs[s]["own"]["kl_share"]) - out["bands"]["kl_share"]["band"]]
    out["control"] = {"kl_share": ctl, "low_on": len(low), "matched_on": len(ctl), "check": "pass" if ctl and len(low) == len(ctl) else "miss"}
    log(f"structured control: low on {len(low)} of {len(ctl)} matched states -> {out['control']['check']}")
    out["pairs"] = {p: {k: pair_rule(recs, r, n, k) for k in KEYS} for p, (r, n) in pairs.items() if all(s in recs for s in (*r, *n))}
    dump(RESULTS / "score.json", out)
    return out


def manifest(n: int) -> None:
    """The inputs outside git per shard: the saved updates and read windows its states use, and the shared ones."""
    pairs, inv, _ = catalogue()
    shared = ["artifacts/vpd/", "artifacts/sets/train_0_4096.npy", "artifacts/sets/val_0_256.npy", str(ALIVE.relative_to(EXPERIMENTS.parent))]  # E53 and label rows
    for i in range(n):
        todo = every_state(pairs)[i::n]
        files = sorted({inv["states"][s]["file"] for s in todo} | {inv["tasks"][inv["states"][s]["task"]].get("path", "") for s in todo} - {""})
        p = RESULTS / f"manifest_{i}_of_{n}.txt"
        p.write_text("\n".join([f"# E228 shard {i} of {n}: {len(todo)} states", *shared, *files]) + "\n")
        log(f"{p.name}: {len(todo)} states, {len(files)} files")


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, th = load_config()
    if argv == ["score"]:
        score(th)
    elif len(argv) == 2 and argv[0] == "manifest":
        manifest(int(argv[1]))
    else:
        assert len(argv) == 3 and argv[0] == "states", f"usage: run.py states I N | score | manifest N, got {argv}"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E211
        log(f"{EXPERIMENT} {' '.join(argv)} start; settings {asdict(cfg)}; thresholds {th}")
        states_job(cfg, th, int(argv[1]), int(argv[2]))
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
