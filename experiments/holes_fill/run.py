"""E143 (holes_fill), cell P2.10: steering in the fair form and the neuron version on every scorecard pair with saved updates
where either is missing, beside the share on the same rows. Plan reports/plans/2026-11-10_plan_E143-holes.md; card
2026-11-10_plan_p2-10-card.md; filed rule results/predictions.json (the run refuses unless it is committed unchanged), applied
by rules.py. Sides are named as each pair was built; no verdict on reuse.

Pairs, states, read rows: E136's `experiments/share_untrained/results/inventory.json` and its `windows.load_read`, the rows
E136 read the share on (its stage1.json `trained`, the share column here). Every reading is the log-odds of each row's target
at the read position over the whole vocabulary, as a share of the fine-tune's own gain on those rows (fp32 target model):
  neuron version  W0 with each block's MLP firing kept on its TopK neurons (k 64; k matched to the decomposition's down_proj
                  on-count, E136's stage0 k_m), the sets from the state's own firing (N01) or W0's (N00): mean(N01 - N00) / gain;
  steering        `fair_steering`: pooled and per-key vectors (key = the token at the read position) from every training
                  window, (layer, strength) chosen on up to n_select of them, read on the rows; random, cross-applied, cosines.

  stage 0  controls, alone, exit 1 on a miss: zero update reads exactly 0 (neurons both k, steering every setting), strength 0
           reads 0 on a real update; the share of two saved states reproduces E136's; E140's neuron reading of case real s0
           reproduces through this path.
  stage 1  every state (written as it lands), then cross-application and cosines per pair, then the score.

    sbatch --comment=E143 cluster/holes_fill.sbatch
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rules
import torch
from torch import Tensor

from commute import fair_steering as FS
from commute import masked_reads as MR
from commute import performed, vpd
from commute import steering as ST
from commute.data import pile_rows, target_tokens
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN, route_cells, route_split
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E136, E140 = REPO / "experiments" / "share_untrained" / "results", REPO / "experiments" / "real_vs_twin" / "results"
WIN = load_experiment("share_untrained", "windows")
PAIRS = ("e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3", "open_pair_c3_copy", "e62_copy_lookup")
NOT_STEERED = ("e62_copy_lookup",)  # read in the fair form by E135
KINDS, KS = ("pooled", "perkey"), ("k64", "kmatched")
DOWN = tuple(f"h.{l}.mlp.down_proj" for l in FS.LAYERS)


def log(msg: str) -> None:
    print(msg, flush=True)


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


class Stopped(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


@dataclass
class Task:
    """A task's read rows (E136's `Read`), the windows its fine-tunes trained on (steering's fitting rows; None where steering
    is not read) and the per-key pool (the task's keys; None where per-key vectors are not read)."""

    read: Any
    train: Tensor | None
    pool: Tensor | None
    lo0: Tensor = field(default_factory=lambda: torch.empty(0))  # W0's log-odds on the read rows

    def x(self, b: Tensor) -> Tensor:
        return b[:, : self.read.ci_len] if self.read.ci_len else b

    def lo(self, lg: Tensor, b: Tensor) -> Tensor:
        """Log-odds of each row's target, from the logits at the read position."""
        return log_odds(lg.float(), target_tokens(self.read.token, b.to(lg.device), self.read.half)).double().cpu()

    def idx(self, toks: Tensor) -> Tensor:
        return torch.zeros(len(toks), dtype=torch.long) if self.pool is None else FS.key_index(toks[:, self.read.half], self.pool)


def e53_train(name: str) -> Tensor:
    """E53's hand-write windows (`performed_anchor.Run.task`): the task's own text of Pile train rows 0-255, window draw as E53."""
    pa = load_experiment("regating_truth_table", "run").performed_anchor()
    pi = {"lowercase": {}, "cipher": pa.e53_pi(), "broad_cipher": pa.a1_maps()["broad_cipher"]}[name]
    text, at = pa.task_text(name, pile_rows("train", 0, 4096)[:256], pi)
    return performed.windows_at(text, at, pa.CFG.half, pa.CFG.n_write_windows, pa.CFG.window_seed)[0]


def load_task(name: str, spec: dict[str, Any]) -> Task:
    r = WIN.load_read(name, spec)
    if spec["format"] == "kv":  # the answer is table[key]: a per-key vector is handed it (stated in the plan)
        w = torch.load(REPO / spec["path"], weights_only=False)
        assert w["train"].half == r.half
        return Task(r, w["train"].tokens, torch.tensor(w["task"].keys))
    if spec["format"] == "e53":  # no key in the window fixes the answer: pooled only
        return Task(r, e53_train(spec["task"]), None)
    return Task(r, None, None)


def batches(t: Task, toks: Tensor) -> list[Tensor]:
    return [toks[s : s + FS.BATCH] for s in range(0, len(toks), FS.BATCH)]


def lo_at(net: ST.Net, t: Task, toks: Tensor, dW: ST.DW | None = None) -> Tensor:
    with net.at(dW):
        return torch.cat([t.lo(net.run(t.x(b))[0][:, t.read.half], b) for b in batches(t, toks)])


def lo_steered(net: ST.Net, t: Task, toks: Tensor, v: Tensor, layer: int, strength: float) -> Tensor:
    return t.lo(FS.logits_steered(net, t.x(toks), t.idx(toks), t.read.half, v, layer, strength), toks)


def share(num: float, gain: float, floor: float) -> float:
    return num / gain if gain >= floor else math.nan


@dataclass
class Job:
    net: ST.Net
    inv: dict[str, Any]
    c: dict[str, Any]
    th: dict[str, float]
    tasks: dict[str, Task] = field(default_factory=dict)
    ks: dict[str, dict[int, int]] = field(default_factory=dict)

    def task(self, name: str) -> Task:
        if name not in self.tasks:
            t = load_task(name, self.inv["tasks"][name])
            t.lo0 = lo_at(self.net, t, t.read.toks)
            self.tasks[name] = t
        return self.tasks[name]

    def dW(self, state: str) -> ST.DW:
        d = torch.load(REPO / self.inv["states"][state]["file"], weights_only=False)
        return {k: v.float() for k, v in d.items()}

    def neuron(self, t: Task, dW: ST.DW, k: dict[int, int], gain: float) -> dict[str, Any]:
        cells: dict[str, list[Tensor]] = {c: [] for c in ("T0", "T1", "N00", "N01")}
        for b in batches(t, t.read.toks):
            for c, lg in MR.topk_logits(self.net, dW, t.x(b), t.read.half, k).items():
                cells[c].append(t.lo(lg, b))
        m = {c: float(torch.cat(v).mean()) for c, v in cells.items()}
        return {"share": share(m["N01"] - m["N00"], gain, self.th["gain_floor"]), "numerator": m["N01"] - m["N00"],
                "gain_cells": m["T1"] - m["T0"], "N00_minus_T0": m["N00"] - m["T0"]}  # fmt: skip

    def kind(self, t: Task, v: dict[int, Tensor], gain: float, gen: torch.Generator) -> dict[str, Any]:
        """The filed selection (largest mean log-odds rise on the selection rows), then the reads at that setting."""
        assert t.train is not None
        sel = t.train[: self.c["n_select"]]
        sel0 = lo_at(self.net, t, sel)
        grid = {(l, s): float((lo_steered(self.net, t, sel, v[l], l, s) - sel0).mean()) for l in FS.LAYERS for s in self.c["strengths"]}
        setting = FS.pick(grid)
        rec: dict[str, Any] = {"grid": {f"{l},{s}": x for (l, s), x in grid.items()}, "setting": setting}
        if setting is None:
            return {**rec, "share": math.nan}
        l, s = setting
        toks = t.read.toks
        rec["rise"] = float((lo_steered(self.net, t, toks, v[l], l, s) - t.lo0).mean())
        rec["share"] = share(rec["rise"], gain, self.th["gain_floor"])
        rec["off_kl"] = FS.off_kl(self.net, t.x(toks), t.read.half, t.idx(toks), (v[l], l, s))
        rnd = [float((lo_steered(self.net, t, toks, FS.random_like(v[l], gen), l, s) - t.lo0).mean()) for _ in range(self.c["n_random"])]
        rec["random"] = [share(x, gain, self.th["gain_floor"]) for x in rnd]
        rec["random_mean"] = statistics.fmean(rec["random"])
        return rec

    def state(self, name: str, i: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """One state's record, and its vectors at every layer (kept for cross-application and cosines)."""
        t, dW = self.task(self.inv["states"][name]["task"]), self.dW(name)
        gain = float((lo_at(self.net, t, t.read.toks, dW) - t.lo0).mean())
        rec: dict[str, Any] = {"gain": gain, "neuron": {kn: self.neuron(t, dW, k, gain) for kn, k in self.ks.items()}}
        vs: dict[str, Any] = {}
        if t.train is not None:
            gen = torch.Generator().manual_seed(self.c["random_seed"] + i)
            n_keys = 1 if t.pool is None else len(t.pool)
            vs = FS.vectors(self.net, t.x(t.train), t.idx(t.train), n_keys, dW, t.read.half)
            rec["off_kl_state"] = FS.off_kl(self.net, t.x(t.read.toks), t.read.half, dW=dW)
            rec["steering"] = {kd: self.kind(t, vs[kd], gain, gen) for kd in KINDS if kd == "pooled" or t.pool is not None}
        return rec, vs

    def cross(self, a: str, b: str, st: dict[str, Any], vs: dict[str, Any]) -> dict[str, float]:
        """a's vector at a's setting, read on b's rows and targets over b's gain (each kind whose keys cover b's rows)."""
        tb, pa = self.task(self.inv["states"][b]["task"]), self.task(self.inv["states"][a]["task"]).pool
        out = {}
        for kd, rec in st[a].get("steering", {}).items():
            if rec["setting"] is None or (kd == "perkey" and not (pa is not None and torch.isin(tb.read.toks[:, tb.read.half], pa).all())):
                continue
            l, s = rec["setting"]
            t = Task(tb.read, tb.train, pa if kd == "perkey" else None, tb.lo0)
            out[kd] = share(float((lo_steered(self.net, t, t.read.toks, vs[a][kd][l], l, s) - t.lo0).mean()), st[b]["gain"], self.th["gain_floor"])
        return out


# ----------------------------------------------------------------------------- stage 0


def stage0(job: Job) -> dict[str, Any]:
    th, net, out = job.th, job.net, {}
    k_m = json.loads((E136 / "stage0.json").read_text())["k_m"]
    job.ks = {"k64": {l: int(job.c["k_fixed"]) for l in FS.LAYERS},"kmatched": {l: int(k_m[key]) for l, key in enumerate(DOWN)}}
    say("0 k per block", job.ks)
    first = "e46/ident32_s0"
    t, zero = job.task(job.inv["states"][first]["task"]), {k: torch.zeros_like(w) for k, w in net.W0.items()}
    for kn, k in job.ks.items():
        cells = [MR.topk_logits(net, zero, t.x(b), t.read.half, k) for b in batches(t, t.read.toks)]
        out[f"zero_neuron_{kn}"] = max(float((c[a] - c[b]).abs().max()) for c in cells for a, b in (("N01", "N00"), ("T1", "T0")))
    assert t.train is not None and t.pool is not None
    sel = t.train[: job.c["n_select"]]
    lg0 = torch.cat([net.run(b)[0][:, t.read.half] for b in batches(t, sel)])
    for name, dW, strengths in (("zero_steering", zero, job.c["strengths"]), ("strength0_real", job.dW(first), (0.0,))):
        vs = FS.vectors(net, t.train, t.idx(t.train), len(t.pool), dW, t.read.half)  # every training window, as stage 1
        out[name] = max(float((FS.logits_steered(net, sel, t.idx(sel), t.read.half, vs[kd][l], l, s) - lg0).abs().max())
                        for kd in KINDS for l in FS.LAYERS for s in strengths)  # fmt: skip
    say("0 zero update and strength 0, largest |change|", out)
    if bad := [k for k, v in out.items() if not v <= th["zero_tol"]]:
        raise halt(0, out, f"not exactly 0: {bad}")
    out["k"], out["reproduce"] = job.ks, {}
    saved = json.loads((E136 / "stage1.json").read_text())["states"]
    comps0 = vpd.component_weights(net.model)
    for state in ("e46/ident32_s0", "e62/copy_s0"):
        r, dW = job.task(job.inv["states"][state]["task"]).read, job.dW(state)
        base = route_cells(net.model, r.toks, r.half, r.token, r.batch, r.ci_len)
        with model_state(net.model, target={k: w + dW[k].to(w.device) for k, w in net.W0.items()}):
            got = route_split(net.model, r.toks, r.half, r.token, base, comps0, r.batch, r.ci_len)["shares"]["r"]
        out["reproduce"][state] = {"r": got, "saved": saved[state]["trained"], "tol": th["r_tol"]}
    lab, dW = job.task("case_label/real"), job.dW("case_label/real_s0")
    rec = job.neuron(lab, dW, job.ks["k64"], float((lo_at(net, lab, lab.read.toks, dW) - lab.lo0).mean()))
    e140 = json.loads((E140 / "stage3.json").read_text())["states"]["case_real_s0"]["neuron"]["k64"]["share"]["lo"]
    out["reproduce"]["case_label/real_s0 neuron k64"] = {"r": rec["share"], "saved": e140, "tol": th["neuron_tol"]}
    say("0 reproductions", out["reproduce"])
    if bad := [k for k, v in out["reproduce"].items() if not abs(v["r"] - v["saved"]) <= v["tol"]]:
        raise halt(0, out, f"does not reproduce: {bad}")
    return out


def halt(n: int, out: dict[str, Any], why: str) -> Stopped:
    out["stopped"] = why
    write(n, out)
    return Stopped(why)


# ----------------------------------------------------------------------------- stage 1


def stage1(job: Job, t_job: float) -> dict[str, Any]:
    pairs = {p["id"]: p for p in job.inv["pairs"] if p["id"] in PAIRS}
    order = list(dict.fromkeys(s for p in PAIRS for s in pairs[p]["reuse"] + pairs[p]["new"]))
    out: dict[str, Any] = {"states": {}, "cross": {}, "cosines": {}}
    vs: dict[str, Any] = {}
    t0 = time.time()
    for i, name in enumerate(order):
        out["states"][name], vs[name] = job.state(name, i)
        st = out["states"][name]
        say(f"1 {name}", {"gain": st["gain"], **{f"neuron_{k}": st["neuron"][k]["share"] for k in KS},
                          **{f"steer_{k}": {x: r.get(x) for x in ("setting", "share", "random_mean", "off_kl")} for k, r in st.get("steering", {}).items()}})  # fmt: skip
        write(1, out)
        minutes = ((time.time() - t0) / (i + 1) * len(order) + (t0 - t_job)) / 60  # the steered states come first
        out["projected_minutes"] = minutes
        log(f"[1 time] {i + 1} of {len(order)} states; projected {minutes:.1f} min for the job")
        if i >= 3 and minutes > job.th["projected_limit_min"]:
            raise halt(1, out, f"projected {minutes:.1f} min over {job.th['projected_limit_min']}")
    for p in PAIRS:
        if p in NOT_STEERED:
            continue
        reuse, new = pairs[p]["reuse"], pairs[p]["new"]
        out["cross"][p] = {f"{a} on {b}": job.cross(a, b, out["states"], vs) for x, y in zip(reuse, new) for a, b in ((x, y), (y, x))}
        names = reuse + new
        out["cosines"][p] = {"names": names, **{str(l): FS.cosine_matrix([vs[n]["pooled"][l] for n in names]).tolist() for l in FS.LAYERS}}
        say(f"1 cross {p}", out["cross"][p])
        write(1, out)
    out["score"] = score(job.th, pairs, out["states"])
    say("1 score", out["score"])
    write(1, out)
    return out


def score(th: dict[str, float], pairs: dict[str, Any], st: dict[str, Any]) -> dict[str, Any]:
    """The filed rule (rules.py) per reading and pair; each family against the share on the same pairs."""
    e136 = json.loads((E136 / "stage1.json").read_text())["states"]
    get = {"share": lambda n: e136[n]["trained"], **{f"neuron_{k}": (lambda n, k=k: st[n]["neuron"][k]["share"]) for k in KS},
           **{f"steer_{k}": (lambda n, k=k: st[n]["steering"][k]["share"] if k in st[n].get("steering", {}) else None) for k in KINDS}}  # fmt: skip
    calls: dict[str, dict[str, Any]] = {m: {} for m in get}
    for p in PAIRS:
        for m, f in get.items():
            reuse, new = [f(n) for n in pairs[p]["reuse"]], [f(n) for n in pairs[p]["new"]]
            if None not in reuse + new:
                calls[m][p] = rules.call(reuse, new, th)
    fam = {"steering": {p: rules.wider(calls["steer_pooled"][p], calls["steer_perkey"].get(p)) for p in calls["steer_pooled"]},
           "neuron": {p: rules.wider(calls["neuron_k64"][p], calls["neuron_kmatched"][p]) for p in PAIRS}}  # fmt: skip
    comp = {f: rules.compare(c, calls["share"]) for f, c in fam.items()}
    each = {m: rules.compare(calls[m], calls["share"]) for m in get if m != "share"}
    return {"calls": calls, "families": fam, "compare": comp, "each_reading": each, "outcome": rules.outcome(comp)}


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    c, th = P["settings"], P["thresholds"]
    assert P["experiment"].startswith("E143") and th["gain_floor"] == MIN_GAIN and c["strengths"][0] == 0
    inv = json.loads((E136 / "inventory.json").read_text())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = Job(ST.Net("cuda"), inv, c, th)
    try:
        write(0, stage0(job))
        print("STAGE 0 DONE", flush=True)
        stage1(job, t_job)
        print("STAGE 1 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
