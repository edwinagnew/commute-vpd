"""E190 (neuron_jobs_k), cell P2.30: E189 (experiments/neuron_jobs, stopped in stage 0 on its top-K check) rerun with a K
grid. Plan reports/plans/2026-11-16_plan_p2-30-neuron-jobs-k.md; card 2026-11-16_plan_p2-30-card.md; filed rules
results/predictions.json (the run refuses unless it is committed unchanged). Descriptive: no verdict on reuse.

Everything is E189's (`neuron_jobs/run.py`, imported, frozen as it ran) except, per the plan:
  1. gain kept, its random-K floor and the pair call at each k in settings.k (240, 768, 1536 of 3072); retention is read
     once per state (it does not depend on k);
  2. C0_topk (the often/rare down_proj update) recorded at each k, not a stop; C0_zero and C0_retention stop as before;
  3. W0 under the mask, per k, no update: each task's target log-odds masked minus unmasked (and over the unmasked mean),
     and the Pile next-token loss the mask adds (nats, val rows 4048-4304);
  4. projected-time stop at thresholds.projected_limit_min.

    sbatch --comment=E190 cluster/neuron_jobs_k.sbatch
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import masked_reads as MR
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
NR = load_experiment("neuron_jobs", "run")  # E189 as it ran: Job, pile_top, own_topk, random_masks, rows_edit, down_on, order, firsts
JR = load_experiment("neuron_jobs", "rules")  # retention, kept, tally
E155 = load_experiment("neuron_overlap", "rules")  # call, floor
N, W, LOSS_BATCH = ST.N_LAYERS, ST.D_MLP, 16
say, log = NR.say, NR.log


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


class Stopped(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


def halt(n: int, out: dict[str, Any], why: str) -> Stopped:
    out["stopped"] = why
    write(n, out)
    return Stopped(why)


@torch.no_grad()
def cells(net: ST.Net, t: Any, ks: list[int], seed: int, counts: dict[int, dict[int, Tensor]] | None = None) -> dict[str, Tensor]:
    """E189's `cells` at every k: firing at the read position f, unmasked log-odds T, and per k M<k> (own top-k, sequential)
    and R<k> (random-k; one generator per k seeded `seed` per pass, so W0 and the state get the same masks)."""
    gens, half = {k: torch.Generator(net.device).manual_seed(seed) for k in ks}, t.read.half
    out: dict[str, list[Tensor]] = {c: [] for c in ("f", "T", *(f"{c}{k}" for k in ks for c in "MR"))}
    for s in range(0, len(t.read.toks), MR.BATCH):
        b = t.read.toks[s : s + MR.BATCH]
        x = t.x(b)
        lg, cap = net.run(x, capture=True)
        out["f"].append(torch.stack([cap[f"mlp{l}"][:, half] for l in range(N)], dim=1).cpu())
        out["T"].append(t.lo(lg[:, half], b))
        for k in ks:
            out[f"M{k}"].append(t.lo(net.run(x, NR.own_topk(k, None if counts is None else counts[k]))[0][:, half], b))
            out[f"R{k}"].append(t.lo(net.run(x, MR.mask_edit(NR.random_masks(x.shape, k, gens[k], net.device)))[0][:, half], b))
    return {c: torch.cat(v) for c, v in out.items()}


class Job(NR.Job):
    """E189's Job (Pile rows, W0 Pile table, pairs, dW, state cache) with the task cells and the reads at every k."""

    def task(self, state: str) -> tuple[Any, dict[str, Tensor], dict[int, dict[int, Tensor]]]:
        name = self.inv["states"][state]["task"]
        if name not in self.tasks:
            t = NR.HF.Task(NR.WIN.load_read(name, self.inv["tasks"][name]), None, None)
            counts = {k: {l: torch.zeros(W, dtype=torch.long) for l in range(N)} for k in self.c["k"]}
            self.tasks[name] = (t, cells(self.net, t, self.c["k"], self.c["random_seed"], counts), counts)
        return self.tasks[name]

    def kepts(self, state: str, dW: ST.DW) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """The state's cells and gain kept per k, without the Pile pass."""
        t, c0, _ = self.task(state)
        with self.net.at(dW):
            c1 = cells(self.net, t, self.c["k"], self.c["random_seed"])
        d = {c: float((c1[c] - c0[c]).mean()) for c in c1 if c != "f"}
        per_k = {f"k{k}": {"kept": JR.kept(d[f"M{k}"], d["T"]), "kept_random": JR.kept(d[f"R{k}"], d["T"]),
                           "num_M": d[f"M{k}"], "num_R": d[f"R{k}"]} for k in self.c["k"]}  # fmt: skip
        return c1, {"gain": d["T"], **per_k}

    def read(self, state: str, dW: ST.DW) -> dict[str, Any]:
        c1, rec = self.kepts(state, dW)
        with self.net.at(dW):
            top1 = NR.pile_top(self.net, self.rows, self.c["n_top_windows"])
        ids = (c1["f"] - self.task(state)[1]["f"]).abs().mean(dim=0).topk(self.c["n_changed"], dim=-1).indices
        per = JR.retention(self.top0, top1, (ids + torch.arange(N)[:, None] * W).flatten())
        return {"retention": float(per.mean()), **rec, "top_changed": ids.tolist(), "retention_per_neuron": per.tolist()}


def brief(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in ("top_changed", "retention_per_neuron", "cols")}


@torch.no_grad()
def pile_loss(net: ST.Net, rows: Tensor, edit: ST.Edit | None) -> float:
    """Mean next-token cross-entropy (nats) over every position of `rows`."""
    tot, n = 0.0, 0
    for b in rows.split(LOSS_BATCH):
        lg = net.run(b, edit)[0][:, :-1]
        tot += float(torch.nn.functional.cross_entropy(lg.flatten(0, 1), b[:, 1:].flatten().to(lg.device), reduction="sum"))
        n += lg.shape[0] * lg.shape[1]
    return tot / n


def w0_masked(job: Job, states: list[str]) -> dict[str, Any]:
    """Per k, no update: each task's mean masked minus unmasked target log-odds (and over the unmasked mean), the Pile loss added."""
    split, a, b = job.c["pile_loss_rows"]
    assert split == "val" and 4048 <= a < b <= 8192
    rows = pile_rows("val", 0, 8192)[a:b]
    base = pile_loss(job.net, rows, None)
    out: dict[str, Any] = {"pile_loss_unmasked": base}
    for k in job.c["k"]:
        tasks = {}
        for s in states:
            t, c0, _ = job.task(s)
            T, M = float(c0["T"].mean()), float(c0[f"M{k}"].mean())
            tasks[t.read.name] = {"T": T, "M": M, "diff": M - T, "diff_over_T": (M - T) / T if T != 0 else float("nan")}
        out[f"k{k}"] = {"tasks": tasks, "pile_loss_added": pile_loss(job.net, rows, NR.own_topk(k)) - base}
        say(f"0 W0 masked k{k}", {"pile_loss_added": out[f"k{k}"]["pile_loss_added"], "diff_over_T": {n: v["diff_over_T"] for n, v in tasks.items()}})
    return out


def controls(job: Job, out: dict[str, Any]) -> None:
    """C0_zero and C0_retention, as E189's stage 0 (stop on a miss)."""
    ref, W0 = job.c["reference_state"], job.net.W0
    out["zero"] = job.read(ref, {k: torch.zeros_like(w).cpu() for k, w in W0.items()})
    z = out["zero"]
    say("0 zero", brief(z))
    if not (z["retention"] == 1.0 and z["gain"] == 0.0 and all(z[f"k{k}"]["num_M"] == 0.0 for k in job.c["k"])):
        raise halt(0, out, "C0_zero: zero update does not read retention exactly 1.0 with M and T numerators exactly 0")
    out["reference"] = job.state(ref)
    say("0 reference", brief(out["reference"]))
    for how in ("harder", "swapped"):
        out[how] = job.read(ref, NR.rows_edit(W0, out["reference"]["top_changed"], how, job.c["harder_factor"]))
        say(f"0 {how}", brief(out[how]))
    h, s = out["harder"]["retention"], out["swapped"]["retention"]
    out["C0_retention"] = {"harder": h, "swapped": s, "pass": h > s + job.th["retention_control_margin"]}
    say("0 C0_retention", out["C0_retention"])
    if not out["C0_retention"]["pass"]:
        raise halt(0, out, "C0_retention: harder not above swapped by the margin")


def topk_check(job: Job) -> dict[str, Any]:
    """C0_topk at each k, recorded: the reference's down_proj update on the k most (often) / least (rare) kept neurons."""
    ref = job.c["reference_state"]
    counts, real, out = job.task(ref)[2], job.dW(ref), {}
    for k in job.c["k"]:
        rec: dict[str, Any] = {}
        for which, sign in (("often", 1), ("rare", -1)):
            cols = {l: (sign * counts[k][l]).topk(k).indices for l in range(N)}
            r = job.kepts(ref, NR.down_on(real, cols))[1]
            rec[f"{which}_own"], rec[f"{which}_random"], rec[f"{which}_gain"] = r[f"k{k}"]["kept"], r[f"k{k}"]["kept_random"], r["gain"]
        rec["gap_over_0.4"] = rec["often_own"] - rec["often_random"] > 0.4  # E189's filed gap; nan reads False
        out[f"k{k}"] = rec
        say(f"0 C0_topk k{k}", rec)
    return out


def stage0(job: Job, pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    controls(job, out)
    out["C0_topk"] = topk_check(job)
    out["W0_masked"] = w0_masked(job, NR.order(job, pairs))
    return out


def stage1(job: Job, pairs: list[str], t_job: float) -> dict[str, Any]:
    names, t0, n_random = NR.order(job, pairs), time.time(), len(NR.firsts(job, pairs))
    out: dict[str, Any] = {"states": {}}
    for i, name in enumerate(names):
        out["states"][name] = job.state(name)
        say(f"1 {name}", brief(out["states"][name]))
        write(1, out)
        minutes = ((time.time() - t0) / (i + 1) * (len(names) + n_random) + (t0 - t_job)) / 60  # stage 2 reads n_random more
        out["projected_minutes"] = minutes
        log(f"[1 time] {i + 1} of {len(names)} states; projected {minutes:.1f} min for the job")
        if i >= 3 and minutes > job.th["projected_limit_min"]:
            raise halt(1, out, f"projected {minutes:.1f} min over {job.th['projected_limit_min']}")
    return out


def score(job: Job, pairs: list[str], states: dict[str, Any], rnd: dict[str, Any]) -> dict[str, Any]:
    """E155's call per pair on retention and on gain kept at each k; floors against the side's random update."""
    reads = {"retention": lambda r: r["retention"], **{f"kept_k{k}": (lambda r, k=k: r[f"k{k}"]["kept"]) for k in job.c["k"]}}
    out: dict[str, Any] = {}
    for p in pairs:
        sd = dict(zip(JR.SIDES, job.names[p]))
        rec: dict[str, Any] = {}
        for name, get in reads.items():
            side = {s: [get(states[n]) for n in sd[s]] for s in JR.SIDES}
            rec[name] = {**E155.call(side["reuse"], side["new"]), "values": side,
                         "floor": {s: E155.floor(v, get(rnd[sd[s][0]])) for s, v in side.items()}}  # fmt: skip
        for k in job.c["k"]:
            rec[f"kept_random_k{k}"] = {"values": {s: [states[n][f"k{k}"]["kept_random"] for n in sd[s]] for s in JR.SIDES},
                                        "random_update": {s: rnd[sd[s][0]][f"k{k}"]["kept_random"] for s in JR.SIDES}}  # fmt: skip
        out[p] = rec
    return out


def tally(sc: dict[str, Any], topk: dict[str, Any]) -> dict[str, Any]:
    """P1-P4 through E189's tally on the filed k (P2 at 1536, P3 and the rest at 240); P5 from stage 0's C0_topk."""
    def view(k: int) -> dict[str, Any]:
        return {p: {"retention": s["retention"], "kept": s[f"kept_k{k}"], "kept_random": s[f"kept_random_k{k}"]} for p, s in sc.items()}

    at240, at1536 = JR.tally(view(240)), JR.tally(view(1536))
    p5 = bool(topk["k1536"]["gap_over_0.4"] and not topk["k240"]["gap_over_0.4"])
    return {"P1": at240["P1"], "P2": at1536["P2"], "P3": at240["P3"], "P4": at240["P4"], "P5": {"holds": p5}}


def stage2(job: Job, pairs: list[str], states: dict[str, Any], topk: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"random": {}}
    for i, s in enumerate(NR.firsts(job, pairs)):
        out["random"][s] = job.read(s, NR.NO.random_like(job.dW(s), job.c["random_seed"] + 1 + i))
        say(f"2 random~{s}", brief(out["random"][s]))
        write(2, out)
    out["score"] = score(job, pairs, states, out["random"])
    out["predictions"] = tally(out["score"], topk)
    say("2 score", {p: {r: {k: v[r][k] for k in ("call", "gap", "band")} for r in v if not r.startswith("kept_random")} for p, v in out["score"].items()})
    say("2 predictions", out["predictions"])
    write(2, out)
    return out


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E190") and P["settings"]["k"] == [240, 768, 1536]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = Job(ST.Net("cuda"), P)  # E189's W0 Pile table is recomputed here (the same rows; E189's file is left as it is)
    pairs = list(P["pairs"])
    assert set(pairs) == set(job.names), (pairs, sorted(job.names))
    try:
        s0 = stage0(job, pairs)
        write(0, s0)
        print("STAGE 0 DONE", flush=True)
        st = stage1(job, pairs, t_job)
        print("STAGE 1 DONE", flush=True)
        stage2(job, pairs, st["states"], s0["C0_topk"])
        print("STAGE 2 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
