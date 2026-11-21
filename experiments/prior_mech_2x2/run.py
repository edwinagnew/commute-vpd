"""E_TBD (cell P2.54) run 1: which saved fine-tunes run through the copy head? Head (2,4), the one head over E62's
prefix-matching bar, is switched off (mean-ablated with E62's saved base-weight means, `copying.ablated`) in M and in
each fine-tuned state F, and the target's log-odds is read on the state's own read rows. Per state: the share of F's
gain over M that is lost, net of M's own fall (E62's `copying.share_lost`, 1 - G_off / G); the same with each of
E62's five random single heads off (the null); M's own fall per task. Nothing is trained. Plan
reports/plans/2026-11-21_plan_prior-mech-2x2.md; rules and expected values in results/predictions.json (the run
refuses unless it is committed unchanged). Stage 2 (run 2, the low-prior copy fine-tune) is `train.py`.

States: E237's 15 key states (copy, rank1, elicit, cipher, prior_cipher x seeds 0-2) at E237's common update size
(each saved update times E237's c for its task, experiments/key_grid/results/stage0.json), and E62's six states (copy
behind a new marker and its lookup twin, seeds 0-2) at their saved size. Run order, cheapest and control first:
  control  M on E52's repeated random tokens: second-copy top-1 with (2,4) off, and with each random head off
           (E62's `control_a`, exit 1 unless "pass"); the zero update reads G = G_off = 0 exactly (exit 1 otherwise)
  repeat   E62's six states: G and G_off against E62's saved reading (exit 1 beyond G_repro_rel)
  key      the 15 key states, one JSON each                                   -> results/stage1/<state>.json
  summary  per task means, M's fall, the filed rules                          -> results/stage1/summary.json
Every scored number is also printed. A rerun skips states already written.

    sbatch --comment=E_TBD cluster/prior_mech_2x2.sbatch
"""

from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP
from commute import steering as ST
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)
from commute.open_pair import jsonable

E228: Any = load_experiment("random_kl_share", "run")  # state_read: a state's read rows and saved update
R96: Any = load_experiment("refit_96key", "run")  # inventories: E147's key states beside E136's (E62's among them)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "stage1"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E147_INV = EXPERIMENTS / "refit_96key" / "results" / "inventory.json"
E237_C = EXPERIMENTS / "key_grid" / "results" / "stage0.json"
E62 = EXPERIMENTS / "induction_cue" / "results"
log = E228.log
Rows = dict[str, Tensor]  # "on", "off", "rand<i>": per-row log-odds of the target


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    key_tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    e62_tasks: tuple[str, ...]
    head: tuple[int, int]
    n_random: int  # E62's random single-head draws (copying.random_others)
    random_seed: int
    induction: dict[str, int | float]  # E52's repeated-random-token control, E62's settings
    batch: int
    n_boot: int
    boot_seed: int

    def __post_init__(self) -> None:
        assert len(set(self.key_tasks)) == len(self.key_tasks) == 5 and self.seeds == (0, 1, 2)
        assert self.head == (2, 4) and self.e62_tasks == ("copy", "lookup") and self.n_random >= 3

    def e62_states(self) -> list[str]:
        return [f"e62/{t}_s{s}" for t in self.e62_tasks for s in self.seeds]

    def key_states(self) -> list[str]:
        return [R96.state_name(t, s) for t in self.key_tasks for s in self.seeds]

    def draws(self) -> list[list[tuple[int, int]]]:
        return CP.random_others([self.head], 4, 6, self.n_random, self.random_seed)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def fname(state: str) -> str:
    return state.replace("/", "-")


class Reader:
    """M, E62's head means, every state's read rows; per-row target log-odds with a set of heads off."""

    def __init__(self, cfg: Config) -> None:
        self.cfg, self.net = cfg, ST.Net("cuda")
        heads = torch.load(E62 / "large" / "heads.pt", weights_only=False)
        assert [tuple(h) for h in heads["H"]] == [cfg.head], heads["H"]
        self.means = heads["means"].to(self.net.device)
        _, self.inv = R96.inventories(json.loads(E147_INV.read_text()))
        self.c = json.loads(E237_C.read_text())["c"]
        self.base: dict[str, Rows] = {}

    def task(self, state: str) -> str:
        return self.inv["states"][state]["task"]

    def scale(self, state: str) -> float:
        """E237's c for a key state's task; 1 for E62's states (read at their saved size)."""
        return self.c[self.task(state).removeprefix("e58/")] if state.startswith("e58/") else 1.0

    @torch.no_grad()
    def lo(self, r: Any, dW: dict[str, Tensor] | None, heads: list[tuple[int, int]]) -> Tensor:
        off = CP.ablated(self.net, heads, self.means) if heads else nullcontext()
        b = self.cfg.batch
        with self.net.at(dW), off:
            return torch.cat([ST.logodds_at(self.net.run(r.toks[s : s + b])[0], r.pos, r.tgt[s : s + b]) for s in range(0, len(r.toks), b)])

    def rows(self, r: Any, dW: dict[str, Tensor] | None) -> Rows:
        out = {"on": self.lo(r, dW, []), "off": self.lo(r, dW, [self.cfg.head])}
        return out | {f"rand{i}": self.lo(r, dW, d) for i, d in enumerate(self.cfg.draws())}

    def of_M(self, state: str, r: Any) -> Rows:
        if (t := self.task(state)) not in self.base:
            self.base[t] = self.rows(r, None)
        return self.base[t]


# ---- the control, alone, first -------------------------------------------------------------------------------------


def control(rd: Reader, th: dict[str, Any]) -> None:
    """M's copying falls with (2,4) off and holds with random heads off; the zero update reads 0 exactly."""
    path, net, C = OUT / "control.json", rd.net, rd.cfg.induction
    if path.is_file() and json.loads(path.read_text()).get("status") == "pass":
        log("control passed before")
        return
    base = CP.induction(net, C)["top1_second"]
    with CP.ablated(net, [rd.cfg.head], rd.means):
        off = CP.induction(net, C)["top1_second"]
    rand = []
    for d in rd.cfg.draws():
        with CP.ablated(net, d, rd.means):
            rand.append(CP.induction(net, C)["top1_second"])
    out: dict[str, Any] = {"top1_base": base, "top1_off": off, "top1_random": rand, "draws": rd.cfg.draws()}
    out["status"] = CP.control_a(base, off, rand, th)
    log(f"[control] second-copy top-1: M {base:.4f}, (2,4) off {off:.4f} (bar < {th['top1_H_max']}), random heads off {rand} (median bar > {th['top1_random_min']}) -> {out['status']}")
    state = rd.cfg.key_states()[0]
    r, dW = E228.state_read(state, rd.inv)
    zero = rd.rows(r, {k: torch.zeros_like(v) for k, v in dW.items()})
    m = rd.of_M(state, r)
    out["zero_max_abs"] = {k: float((zero[k] - m[k]).abs().max()) for k in m}
    log(f"[control] zero update on {state}'s rows, max |F - M| per head set {out['zero_max_abs']} (must be 0)")
    if out["status"] != "pass" or any(v != 0 for v in out["zero_max_abs"].values()):
        stop(path, out, f"control: {out['status']}, zero update {out['zero_max_abs']}", log)
    write_stage(path, out)


# ---- one state -------------------------------------------------------------------------------------------------------


def share_of(f: Rows, m: Rows, key: str) -> tuple[float, float, float]:
    """(G, G_off, share lost) with the heads `key` off: ratio of row means, E62's form."""
    G, G_off = float((f["on"] - m["on"]).mean()), float((f[key] - m[key]).mean())
    return G, G_off, CP.share_lost(G, G_off)


def read_state(rd: Reader, state: str, th: dict[str, Any]) -> dict[str, Any]:
    path = OUT / f"{fname(state)}.json"
    if path.is_file():
        return json.loads(path.read_text())
    t0, (r, dW), c = time.time(), E228.state_read(state, rd.inv), rd.scale(state)
    m, f = rd.of_M(state, r), rd.rows(r, {k: v * c for k, v in dW.items()})
    G, G_off, share = share_of(f, m, "off")
    rand = [share_of(f, m, f"rand{i}")[2] for i in range(rd.cfg.n_random)]
    sd = CP.share_sd(f["on"] - m["on"], f["off"] - m["off"], rd.cfg.n_boot, rd.cfg.boot_seed)
    rec = {"experiment": EXPERIMENT, "state": state, "task": rd.task(state), "c": c, "n_rows": len(r.toks), "G": G, "G_off": G_off,
           "share_lost": share, "share_sd": sd, "random_shares": rand, "random_median": statistics.median(rand),
           "label": CP.share_label(G, G_off, th), "M_fall": float((m["on"] - m["off"]).mean()), "seconds": time.time() - t0}  # fmt: skip
    write_stage(path, jsonable(rec))
    log(f"[{state}] c {c:.4f}; G {G:.4f}, G_off {G_off:.4f}, share lost {share:.4f} (sd {sd:.4f}); random-head shares {[round(x, 4) for x in rand]}; "
        f"label {rec['label']}; M's fall {rec['M_fall']:.4f}; {rec['seconds']:.0f} s")  # fmt: skip
    torch.cuda.empty_cache()
    return rec


def repeat(rd: Reader, th: dict[str, Any]) -> None:
    """E62's six states read again: G and G_off against E62's run.json, relative tolerance G_repro_rel."""
    saved, miss = json.loads((E62 / "run.json").read_text())["states"], []
    for state in rd.cfg.e62_states():
        rec, ref = read_state(rd, state, th), saved[state.removeprefix("e62/")]
        dev = {k: abs(rec[k] / ref[s] - 1) for k, s in (("G", "G"), ("G_off", "G_abl"))}
        log(f"[repeat {state}] G {rec['G']:.4f} vs E62 {ref['G']:.4f}; G_off {rec['G_off']:.4f} vs E62 {ref['G_abl']:.4f}; rel dev {dev}")
        miss += [f"{state} {k} rel dev {v:.2e}" for k, v in dev.items() if not v <= th["G_repro_rel"]]
    if miss:
        stop(OUT / "repeat.json", {"miss": miss}, "E62 repeat: " + "; ".join(miss), log)


# ---- summary and the filed rules -------------------------------------------------------------------------------------


READABLE = ("old", "new", "between")  # copying.share_label's labels with a finite share over the gain gate


def task_mean(recs: list[dict[str, Any]]) -> float:
    """Mean share over seeds; nan unless every seed is readable (finite, gain at or over G_min)."""
    return fmean(r["share_lost"] for r in recs) if all(r["label"] in READABLE for r in recs) else math.nan


def verdict(holds: bool, *means: float) -> str:
    return "not_read" if not all(math.isfinite(m) for m in means) else "holds" if holds else "misses"


def rules(by: dict[str, list[dict[str, Any]]], th: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on task means. nu, the null's floor: the largest |median random-head share| over readable states."""
    m = {t: task_mean(v) for t, v in by.items()}
    nu = max(abs(r["random_median"]) for v in by.values() for r in v if r["label"] in READABLE)
    new, old = th["share_new_max"], th["share_old_min"]
    copy, pc, ci = m["e58/copy"], m["e58/prior_cipher"], m["e58/cipher"]
    over_null = all(r["share_lost"] > r["random_median"] for r in by["e58/copy"] if r["label"] in READABLE)
    P = {"P1_copy_uses_head": verdict(copy >= new and over_null, copy),
         "P2_ciphers_do_not": verdict(ci <= new and pc <= new, ci, pc),
         "P3_matched_prior_pair_split": verdict(copy - pc > 2 * nu, copy, pc),
         "P4_e62_copy_over_lookup": verdict(m["e62/copy"] - m["e62/lookup"] > 2 * nu, m["e62/copy"], m["e62/lookup"]),
         "P5_rank1_elicit_do_not": verdict(m["e58/rank1"] <= new and m["e58/elicit"] <= new, m["e58/rank1"], m["e58/elicit"])}  # fmt: skip
    kind = {t: "not_read" if not math.isfinite(v) else "uses_head" if v >= old else "not_via_head" if v <= new else "between" for t, v in m.items()}
    return {"task_mean_share": m, "task_kind": kind, "null_floor": nu, "rules": P}


def summary(cfg: Config, th: dict[str, Any], recs: list[dict[str, Any]]) -> None:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        by.setdefault(r["task"], []).append(r)
    out = {"experiment": EXPERIMENT, "M_fall": {t: v[0]["M_fall"] for t, v in by.items()}} | rules(by, th)
    write_stage(OUT / "summary.json", jsonable(out))
    for t, v in by.items():
        log(f"[summary {t}] mean share lost {out['task_mean_share'][t]:.4f} ({out['task_kind'][t]}); seeds {[r['share_lost'] for r in v]}; M's fall {out['M_fall'][t]:.4f}")
    log(f"[summary] null floor (largest |median random-head share|) {out['null_floor']:.4f}; rules {out['rules']}")


def main() -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as E62 and E237 read
    log(f"{EXPERIMENT} run 1 start; settings {asdict(cfg)}; thresholds {th}")
    OUT.mkdir(parents=True, exist_ok=True)
    rd = Reader(cfg)
    control(rd, th)
    repeat(rd, th)
    recs = [read_state(rd, s, th) for s in cfg.e62_states() + cfg.key_states()]
    summary(cfg, th, recs)
    log(f"{EXPERIMENT} run 1 DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
