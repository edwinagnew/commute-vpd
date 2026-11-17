"""E155 (neuron_overlap), cell P2.18: do the same top-firing neurons stay on after a fine-tune? Plan
reports/plans/2026-11-11_plan_p2-18-neuron-overlap.md; card 2026-11-11_plan_p2-18-card.md; filed rule results/predictions.json
(the run refuses unless it is committed unchanged), applied by rules.py. Descriptive: no verdict on reuse; sides named as built.

Pairs, states, read rows: E143's (E136's `inventory.json` and `windows.load_read`, the rows the share was read on). Per state,
on every read row at the read position, each block's MLP firing (down_proj's input, fp32 target model) at W0 and at W0 + the
saved update, and per block the top-k sets by |firing| (`masked_reads.top_overlap`, the sets of E143's `top_masks`):
  overlap   Jaccard of the two top-240 sets (7.8% of 3072, Goodfire's share), mean over rows, then over the 4 blocks;
  on_both   the squared firing change summed over neurons in both top-240 sets, over that summed over every neuron, pooled
            over rows and blocks (nan when the update changes no firing).
k 64 is recorded as a second form, not called.

  stage 0  alone, first: a zero update reads overlap exactly 1.0 and on_both nan at both k (exit 1 otherwise); then the
           positive control, a hand-built 'same neurons, harder' update (each block's c_fc scaled by 1 + d, d matched to
           e46/ident32_s0's c_fc size) against a random update of the same per-matrix size: recorded, not a stop.
  stage 1  every saved state, written as each lands; the seed spread per side (the call's band) comes from these.
  stage 2  one random update per side (matched Frobenius norm per matrix to the side's first seed, fixed seed), the calls.

    sbatch --comment=E155 cluster/neuron_overlap.sbatch
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import rules
import torch

from commute import masked_reads as MR
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
INV = REPO / "experiments" / "share_untrained" / "results" / "inventory.json"
WIN = load_experiment("share_untrained", "windows")
PAIRS = ("e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3", "open_pair_c3_copy", "e62_copy_lookup")
READINGS = ("overlap", "on_both")


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


def halt(n: int, out: dict[str, Any], why: str) -> Stopped:
    out["stopped"] = why
    write(n, out)
    return Stopped(why)


@torch.no_grad()
def measure(net: ST.Net, read: Any, dW: ST.DW, ks: dict[str, int]) -> dict[str, Any]:
    """Per k: overlap, on_both and their per-block parts, W0 against W0 + dW on the read rows at the read position."""
    acc = {kn: {l: {"jaccard": [], "d2_both": 0.0, "d2_all": 0.0} for l in range(ST.N_LAYERS)} for kn in ks}
    for s in range(0, len(read.toks), MR.BATCH):
        b = read.toks[s : s + MR.BATCH]
        x = b[:, : read.ci_len] if read.ci_len else b
        f0 = net.run(x, capture=True)[1]
        with net.at(dW):
            f1 = net.run(x, capture=True)[1]
        for kn, k in ks.items():
            for l, r in MR.top_overlap(f0, f1, dict.fromkeys(range(ST.N_LAYERS), k), read.half).items():
                a = acc[kn][l]
                a["jaccard"].append(r["jaccard"])
                a["d2_both"] += float(r["d2_both"].sum())
                a["d2_all"] += float(r["d2_all"].sum())
    return {kn: summarise(blocks) for kn, blocks in acc.items()}


def summarise(blocks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    jac = {l: torch.cat(a["jaccard"]) for l, a in blocks.items()}
    both, every = sum(a["d2_both"] for a in blocks.values()), sum(a["d2_all"] for a in blocks.values())
    return {"overlap": statistics.fmean(float(j.mean()) for j in jac.values()), "on_both": both / every if every > 0 else math.nan,
            "per_block": {l: {"overlap": float(jac[l].mean()), "overlap_min": float(jac[l].min()), "d2_both": a["d2_both"], "d2_all": a["d2_all"]}
                          for l, a in blocks.items()}}  # fmt: skip


def random_like(dW: ST.DW, seed: int) -> ST.DW:
    """Gaussian per matrix, scaled to that matrix's Frobenius norm in dW (zero where dW is zero); one generator, sorted keys."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for key in sorted(dW):
        r = torch.randn(dW[key].shape, generator=gen)
        out[key] = r * (float(dW[key].norm()) / float(r.norm()))
    return out


class Job:
    def __init__(self, net: ST.Net, inv: dict[str, Any], P: dict[str, Any]) -> None:
        self.net, self.inv, self.c, self.th = net, inv, P["settings"], P["thresholds"]
        self.ks = {f"k{k}": int(k) for k in self.c["k"]}
        self.reads: dict[str, Any] = {}

    def read(self, state: str) -> Any:
        task = self.inv["states"][state]["task"]
        if task not in self.reads:
            self.reads[task] = WIN.load_read(task, self.inv["tasks"][task])
        return self.reads[task]

    def dW(self, state: str) -> ST.DW:
        d = torch.load(REPO / self.inv["states"][state]["file"], weights_only=False)
        return {k: v.float() for k, v in d.items()}

    def at(self, state: str, dW: ST.DW) -> dict[str, Any]:
        return measure(self.net, self.read(state), dW, self.ks)


def stage0(job: Job) -> dict[str, Any]:
    ref, W0 = job.c["reference_state"], job.net.W0
    zero = {k: torch.zeros_like(w).cpu() for k, w in W0.items()}
    out: dict[str, Any] = {"zero": job.at(ref, zero)}
    say("0 zero update", {kn: {r: v[r] for r in READINGS} for kn, v in out["zero"].items()})
    bad = [kn for kn, v in out["zero"].items() if not (v["overlap"] == 1.0 and math.isnan(v["on_both"]) and all(b["overlap_min"] == 1.0 for b in v["per_block"].values()))]
    if bad:
        raise halt(0, out, f"zero update does not read overlap exactly 1.0 with on_both nan at {bad}")
    real = job.dW(ref)
    scaled = {k: (w.cpu() * (float(real[k].norm()) / float(w.norm())) if k.endswith("mlp.c_fc") else torch.zeros_like(w).cpu()) for k, w in W0.items()}
    out["scale"] = {k: float(v.norm()) / float(W0[k].norm()) for k, v in scaled.items() if k.endswith("mlp.c_fc")}
    out["scaled"], out["scaled_random"] = job.at(ref, scaled), job.at(ref, random_like(scaled, job.c["random_seed"]))
    k = f"k{job.c['k'][0]}"
    out["positive_control"] = {r: {"scaled": out["scaled"][k][r], "random": out["scaled_random"][k][r],
                                   "pass": rules.positive_control(out["scaled"][k][r], out["scaled_random"][k][r])} for r in READINGS}  # fmt: skip
    say("0 positive control", out["positive_control"])
    return out


def pairs_of(job: Job) -> dict[str, Any]:
    return {p["id"]: p for p in job.inv["pairs"] if p["id"] in PAIRS}


def order(job: Job) -> list[str]:
    """Every saved state of the pairs, each once (a side shared by two pairs is read once)."""
    pairs = pairs_of(job)
    return list(dict.fromkeys(s for p in PAIRS for s in pairs[p]["reuse"] + pairs[p]["new"]))


def firsts(job: Job) -> list[str]:
    """Each side's first seed, each once: the random update of a side is matched to it."""
    pairs = pairs_of(job)
    return list(dict.fromkeys(pairs[p][side][0] for p in PAIRS for side in ("reuse", "new")))


def stage1(job: Job, t_job: float) -> dict[str, Any]:
    names, t0, n_random = order(job), time.time(), len(firsts(job))
    out: dict[str, Any] = {"states": {}}
    for i, name in enumerate(names):
        out["states"][name] = job.at(name, job.dW(name))
        say(f"1 {name}", {kn: {r: v[r] for r in READINGS} for kn, v in out["states"][name].items()})
        write(1, out)
        minutes = ((time.time() - t0) / (i + 1) * (len(names) + n_random) + (t0 - t_job)) / 60  # stage 2 reads n_random more
        out["projected_minutes"] = minutes
        log(f"[1 time] {i + 1} of {len(names)} states; projected {minutes:.1f} min for the job")
        if i >= 3 and minutes > job.th["projected_limit_min"]:
            raise halt(1, out, f"projected {minutes:.1f} min over {job.th['projected_limit_min']}")
    return out


def stage2(job: Job, states: dict[str, Any]) -> dict[str, Any]:
    pairs = pairs_of(job)
    out: dict[str, Any] = {"random": {}}
    for i, s in enumerate(firsts(job)):
        out["random"][s] = job.at(s, random_like(job.dW(s), job.c["random_seed"] + 1 + i))
        say(f"2 random~{s}", {kn: {r: v[r] for r in READINGS} for kn, v in out["random"][s].items()})
        write(2, out)
    out["score"] = score(job, pairs, states, out["random"])
    say("2 score", out["score"])
    write(2, out)
    return out


def score(job: Job, pairs: dict[str, Any], states: dict[str, Any], rnd: dict[str, Any]) -> dict[str, Any]:
    """The filed rule (rules.py) per pair and reading, at the first k; each side against its random update."""
    k = f"k{job.c['k'][0]}"
    out: dict[str, Any] = {}
    for p in PAIRS:
        rec: dict[str, Any] = {}
        for r in READINGS:
            side = {sd: [states[n][k][r] for n in pairs[p][sd]] for sd in ("reuse", "new")}
            rec[r] = {**rules.call(side["reuse"], side["new"]), "values": side,
                      "floor": {sd: rules.floor(v, rnd[pairs[p][sd][0]][k][r]) for sd, v in side.items()}}  # fmt: skip
        out[p] = rec
    return out


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E155") and P["settings"]["k"][0] == 240
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = Job(ST.Net("cuda"), json.loads(INV.read_text()), P)
    try:
        write(0, stage0(job))
        print("STAGE 0 DONE", flush=True)
        st = stage1(job, t_job)
        print("STAGE 1 DONE", flush=True)
        stage2(job, st["states"])
        print("STAGE 2 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
