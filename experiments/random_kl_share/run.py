"""E228 (cell P2.44): the re-gating share on random weight changes of the 67M, each scaled so its output change (mean
KL to W0 on a fine-tune's read rows) matches that fine-tune's, the control the share failed on Qwen (P3.8). Plan
reports/plans/2026-11-20_plan_p2-44-random-kl-matched.md; settings, expected values and rules in results/predictions.json
(the run refuses unless it is committed unchanged). The share is commute.realistic_reads.share_of (E211's reader, GOALS.md
Terms r); the random change is E46's (commute.finetune.random_direction, the fine-tune's per-matrix norms, times c from
commute.perturb.match). Nothing is trained.

Jobs, each writing its JSON as it goes and printing every number:
  ctl          alone, exit 1 on a miss: a zero change on the first state's rows reads share numerator, gain and both KLs
               exactly 0; each fine-tune's share on its E136 read rows reproduces E136's saved share; its KL at both
               sites (every position up to the read position, and the read position) and its split-half shares -> ctl.json
  draws STATE  after ctl: per site and seed, the Gaussian direction scaled to the fine-tune's KL at that site, and the
               share read on it -> draws_<state>.json after every draw
  score        no GPU: the band and rules P1 (site all) and P2 (site read) -> score.json

    sbatch --comment=E228 cluster/random_kl_share.sbatch ctl
    sbatch --comment=E228 --dependency=afterok:<ctl> cluster/random_kl_share.sbatch draws c3/copy_s0   (and each state)
    python experiments/random_kl_share/run.py score
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

from commute import finetune as FT
from commute import perturb as PT
from commute import realistic_reads as RR
from commute import steering as ST
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

WIN: Any = load_experiment("share_untrained", "windows")  # E136's read windows
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, CTL = RESULTS / "predictions.json", RESULTS / "ctl.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E136 = EXPERIMENTS / "share_untrained" / "results"
SITES = ("all", "read")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: dict[str, Any]) -> None:
    write_stage(path, jsonable(obj))


def stop(path: Path, obj: dict[str, Any], why: str) -> None:
    dump(path, obj | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    states: tuple[str, ...]
    seeds: tuple[int, ...]
    scale_cap: float
    kl_rel_tol: float
    split_seed: int
    batch: int  # rows per KL forward

    def __post_init__(self) -> None:
        assert len(set(self.states)) == len(self.states) >= 1 and len(set(self.seeds)) == len(self.seeds) >= 3


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def fname(state: str) -> str:
    return state.replace("/", "_")


# ---- the rows, the update, the KL -----------------------------------------------------------------------------------


def state_read(state: str) -> tuple[RR.Read, dict[str, torch.Tensor]]:
    """A fine-tune's E136 read rows (its task's held-out windows, its target at the answer position) in the LM layout,
    and its saved weight change, fp32 CPU."""
    inv = json.loads((E136 / "inventory.json").read_text())
    task = inv["states"][state]["task"]
    w = WIN.load_read(task, inv["tasks"][task])
    r = RR.next_layout(task, w.toks, target_tokens(w.token, w.toks, w.half), w.half)
    d = torch.load(EXPERIMENTS.parent / inv["states"][state]["file"], weights_only=False)
    return r, {k: v.float() for k, v in d.items()}


@torch.no_grad()
def kl_of(net: ST.Net, r: RR.Read, dW: dict[str, torch.Tensor], site: str, batch: int) -> float:
    """Mean KL(W0 + dW || W0) per row and position: every position up to the read position ("all"), or the read
    position alone ("read")."""
    tot, n = 0.0, 0
    for s in range(0, len(r.toks), batch):
        t = r.toks[s : s + batch, : r.pos + 1]
        p0 = net.run(t)[0].log_softmax(-1)
        with net.at(dW):
            p1 = net.run(t)[0].log_softmax(-1)
        k = (p1.exp() * (p1 - p0)).sum(-1)
        k = k if site == "all" else k[:, -1:]
        tot, n = tot + float(k.double().sum()), n + k.numel()
    return tot / n


def share(net: ST.Net, r: RR.Read, dW: dict[str, torch.Tensor]) -> dict[str, float]:
    s = RR.share_of(net, r, dW)
    return {"share": s["share"]["lo"], "numerator": s["numerator"]["lo"], "gain": s["gain"]["lo"]}


# ---- jobs -----------------------------------------------------------------------------------------------------------


def ctl(net: ST.Net, cfg: Config, th: dict[str, Any]) -> None:
    out: dict[str, Any] = {"experiment": EXPERIMENT, "card": torch.cuda.get_device_name(), "states": {}}
    saved = json.loads((E136 / "stage1.json").read_text())["states"]
    r, dW = state_read(cfg.states[0])
    zero = {k: torch.zeros_like(v) for k, v in dW.items()}
    out["zero"] = share(net, r, zero) | {f"kl_{s}": kl_of(net, r, zero, s, cfg.batch) for s in SITES}
    log(f"[zero] {json.dumps(jsonable(out['zero']))}")
    if bad := {k: v for k, v in out["zero"].items() if k != "share" and v != 0}:
        stop(CTL, out, f"a zero change does not read exactly 0: {bad}")
    for state in cfg.states:
        t0 = time.time()
        r, dW = state_read(state)
        perm = torch.randperm(len(r.toks), generator=torch.Generator().manual_seed(cfg.split_seed))
        halves = [share(net, RR.subset(r, perm[: len(perm) // 2]), dW), share(net, RR.subset(r, perm[len(perm) // 2 :]), dW)]
        rec = share(net, r, dW) | {"saved": saved[state]["trained"], "halves": halves, "n_rows": len(r.toks), "pos": r.pos}
        rec |= {f"kl_{s}": kl_of(net, r, dW, s, cfg.batch) for s in SITES} | {"seconds": time.time() - t0}
        out["states"][state] = rec
        dump(CTL, out)
        log(f"[ctl {state}] {json.dumps(jsonable(rec))}")
        if not abs(rec["share"] - rec["saved"]) <= th["reproduce_tol"]:
            stop(CTL, out, f"{state}'s share {rec['share']:.4f} does not reproduce E136's {rec['saved']:.4f}")
    dump(CTL, out | {"done": True})
    log("STAGE ctl DONE")


def draws(net: ST.Net, cfg: Config, th: dict[str, Any], state: str) -> None:
    c = json.loads(CTL.read_text())
    assert c.get("done") and state in c["states"], "run ctl first"
    path = RESULTS / f"draws_{fname(state)}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"experiment": EXPERIMENT, "state": state, "draws": {}}
    r, dW = state_read(state)
    for site in SITES:
        target = c["states"][state][f"kl_{site}"]
        for seed in cfg.seeds:
            key = f"{site}_s{seed}"
            if key in out["draws"]:
                continue
            t0, d = time.time(), FT.random_direction(dW, seed)
            k1 = kl_of(net, r, d, site, cfg.batch)  # first-minute check: the unscaled direction's KL is finite and positive
            if not (math.isfinite(k1) and k1 > 0):
                stop(path, out, f"{key}: the unscaled random direction reads KL {k1}")
            m = PT.match(lambda x, d=d, site=site: kl_of(net, r, PT.scaled(d, x), site, cfg.batch), target, cfg.scale_cap, cfg.kl_rel_tol)
            rec = {"site": site, "seed": seed, "kl_target": target, "kl_at_1": k1, "c": m["x"], "kl": m["f"], "match": m["status"],
                   "evals": len(m["grid"]) + len(m["trace"])} | share(net, r, PT.scaled(d, m["x"]))  # fmt: skip
            rec["kl_other_site"] = kl_of(net, r, PT.scaled(d, m["x"]), "read" if site == "all" else "all", cfg.batch)
            rec["seconds"] = time.time() - t0
            out["draws"][key] = rec
            dump(path, out)
            log(f"[{state} {key}] {json.dumps(jsonable(rec))}")
    dump(path, out | {"done": True})
    log(f"STAGE draws {state} DONE")


# ---- score (no GPU) -------------------------------------------------------------------------------------------------


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan  # nan is written as the string "nan"


def verdict(own: float, b: float, rs: list[dict[str, Any]], min_scored: int) -> dict[str, Any]:
    """One fine-tune at one site: 'refused' if fewer than min_scored draws are matched with a scored share; else 'low'
    if every scored draw's share is below own - b (strictly), else 'reaches'."""
    scored = [num(d["share"]) for d in rs if d["match"] == "ok" and math.isfinite(num(d["share"]))]
    sd = float(torch.tensor(scored).std()) if len(scored) > 1 else math.nan
    rec = {"own": own, "band": b, "scored": scored, "n_draws": len(rs), "sd": sd, "gains": [num(d["gain"]) for d in rs]}
    if len(scored) < min_scored:
        return rec | {"verdict": "refused"}
    return rec | {"verdict": "low" if max(scored) < own - b else "reaches"}


def score(cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    c = json.loads(CTL.read_text())["states"]
    gaps = [abs(num(s["halves"][0]["share"]) - num(s["halves"][1]["share"])) for s in c.values()]
    gaps = [g for g in gaps if math.isfinite(g)]
    b = max(gaps) if gaps and max(gaps) > 0 else th["band_min"]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "band": b, "band_gaps": gaps, "rules": {}}
    for rule, site in (("P1", "all"), ("P2", "read")):
        per = {}
        for state in cfg.states:
            p = RESULTS / f"draws_{fname(state)}.json"
            rs = [d for d in json.loads(p.read_text())["draws"].values() if d["site"] == site] if p.is_file() else []
            per[state] = verdict(num(c[state]["share"]), b, rs, th["min_scored"]) if rs else {"verdict": "not run"}
        vs = [v["verdict"] for v in per.values()]
        overall = ("not run" if "not run" in vs else "rival b: reaches" if "reaches" in vs else "rival a: low" if "low" in vs else "silent: refused")
        out["rules"][rule] = {"site": site, "states": per, "overall": overall}
        log(f"score {rule} (site {site}): {overall}; " + "; ".join(f"{s} {v['verdict']}" for s, v in per.items()))
    dump(RESULTS / "score.json", out)
    return out


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, th = load_config()
    if argv == ["score"]:
        score(cfg, th)
        return 0
    assert argv[:1] == ["ctl"] and len(argv) == 1 or argv[:1] == ["draws"] and len(argv) == 2 and argv[1] in cfg.states, f"usage: run.py ctl | draws STATE | score, got {argv}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E211
    log(f"{EXPERIMENT} {' '.join(argv)} start; settings {asdict(cfg)}; thresholds {th}")
    net = ST.Net("cuda")
    ctl(net, cfg, th) if argv[0] == "ctl" else draws(net, cfg, th, argv[1])
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
