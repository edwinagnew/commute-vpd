"""E189 (neuron_jobs), cell P2.29: Goodfire-style neuron baselines on the nine built 67M pairs. Plan
reports/plans/2026-11-16_plan_p2-29-neuron-jobs.md; card 2026-11-16_plan_p2-29-card.md; filed rules results/predictions.json
(the run refuses unless it is committed unchanged). Descriptive: no verdict on reuse; sides named as built.

Pairs, states, read rows: E187's `size_matched.pairs()` (E155's seven pairs and E147's two 96-key pairs; E136's
`windows.load_read` rows). Firing = each block's MLP neurons (down_proj's input, `steering.Net` site mlp<l>), fp32. Per state:
  retention  the 32 neurons per block of largest mean |firing change| on the read rows at the read position; per neuron its
             top 20 of 2,000 Pile windows (val rows 2048-4048) by peak firing over positions, at W0 and at the state;
             shared / 20, mean over the 128 neurons (rules.retention);
  kept       mean(M1 - M0) / mean(T1 - T0) in log-odds of the target at the read position (E143's units, holes_fill Task.lo);
             T unmasked, M each block keeping its 240 neurons of largest |firing| per window and position, taken from the
             firing inside the masked forward (rules.keep_top); kept_random the same with a random 240 per block, window and
             position (one generator seeded random_seed per pass, so W0 and state get the same masks).

  stage 0  alone, first, exit 1 on a miss: (a) zero update: retention exactly 1.0, mean(M1 - M0) and mean(T1 - T0) exactly 0;
           (b) on e46/ident32_s0's top-changed neurons, c_fc rows x 1.5 ("harder") against cyclically shifted ("swapped"):
           harder > swapped + 0.5; (c) e46/ident32_s0's down_proj update on the 240 neurons per block most often in W0's
           own top-240 masks on its read rows: kept > kept_random + 0.4 (the 240 least often recorded only).
  stage 1  every state, written as it lands; projected-time stop after four states.
  stage 2  one random update per side (E155's random_like), the calls (E155's rules.call / floor), P1-P4 tallies.

    sbatch --comment=E189 cluster/neuron_jobs.sbatch
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
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
SM = load_experiment("size_matched", "run")  # pairs(): names, inventory
HF = load_experiment("holes_fill", "run")  # Task: x (ci_len cut), lo (log-odds of the target)
WIN = load_experiment("share_untrained", "windows")  # load_read
NO = load_experiment("neuron_overlap", "run")  # random_like
E155 = load_experiment("neuron_overlap", "rules")  # call, floor
JR = load_experiment("neuron_jobs", "rules")  # keep_top, retention, kept, tally
N, W, PILE_BATCH = ST.N_LAYERS, ST.D_MLP, 32
READINGS = ("retention", "kept")


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


class _Done(Exception):
    """Raised at the last block's firing: the Pile pass needs no logits."""


@torch.no_grad()
def pile_top(net: ST.Net, rows: Tensor, n_top: int) -> Tensor:
    """(n_top, N * W) window ids: per neuron (flat l * W + i) the rows of largest peak firing (max over positions, signed)."""
    peaks: list[Tensor] = []
    seen: dict[int, Tensor] = {}

    def keep(x: Tensor, l: int) -> Tensor:
        seen[l] = x.amax(dim=1)
        if l == N - 1:
            raise _Done
        return x

    edit = {f"mlp{l}": (lambda x, l=l: keep(x, l)) for l in range(N)}
    for b in rows.split(PILE_BATCH):
        try:
            net.run(b, edit)
        except _Done:
            pass
        peaks.append(torch.cat([seen[l] for l in range(N)], dim=-1))
    return torch.cat(peaks).topk(n_top, dim=0).indices.cpu()


def own_topk(k: int, counts: dict[int, Tensor] | None = None) -> ST.Edit:
    """Each block keeps its k largest |firing| per window and position, from the firing in this forward (sequential)."""

    def f(x: Tensor, l: int) -> Tensor:
        y = JR.keep_top(x, k)
        if counts is not None:
            counts[l] += (y != 0).sum(dim=(0, 1)).cpu()
        return y

    return {f"mlp{l}": (lambda x, l=l: f(x, l)) for l in range(N)}


def random_masks(shape: torch.Size, k: int, gen: torch.Generator, device: str) -> dict[int, Tensor]:
    """Per block, (B, L, W) bool: k uniformly random neurons per window and position."""
    out = {}
    for l in range(N):
        idx = torch.rand((*shape, W), generator=gen, device=device).topk(k, dim=-1).indices
        out[l] = torch.zeros((*shape, W), dtype=torch.bool, device=device).scatter(-1, idx, True)
    return out


@torch.no_grad()
def cells(net: ST.Net, t: Any, k: int, seed: int, counts: dict[int, Tensor] | None = None) -> dict[str, Tensor]:
    """On the task's read rows under the weights in place: firing at the read position f (R, N, W); log-odds of the target
    T (unmasked), M (own top-k, sequential), R (random-k)."""
    gen, half = torch.Generator(net.device).manual_seed(seed), t.read.half
    out: dict[str, list[Tensor]] = {c: [] for c in ("f", "T", "M", "R")}
    for s in range(0, len(t.read.toks), MR.BATCH):
        b = t.read.toks[s : s + MR.BATCH]
        x = t.x(b)
        lg, cap = net.run(x, capture=True)
        out["f"].append(torch.stack([cap[f"mlp{l}"][:, half] for l in range(N)], dim=1).cpu())
        out["T"].append(t.lo(lg[:, half], b))
        out["M"].append(t.lo(net.run(x, own_topk(k, counts))[0][:, half], b))
        out["R"].append(t.lo(net.run(x, MR.mask_edit(random_masks(x.shape, k, gen, net.device)))[0][:, half], b))
    return {c: torch.cat(v) for c, v in out.items()}


class Job:
    def __init__(self, net: ST.Net, P: dict[str, Any]) -> None:
        self.net, self.c, self.th = net, P["settings"], P["thresholds"]
        self.names, _, self.inv = SM.pairs()
        split, a, b = self.c["pile_rows"]
        assert (split, b - a) == ("val", 2000)
        self.rows = pile_rows("val", 0, 8192)[a:b]  # E101's cached rows (val 0-8192), sliced
        self.top0 = pile_top(net, self.rows, self.c["n_top_windows"])
        self.tasks: dict[str, tuple[Any, dict[str, Tensor], dict[int, Tensor]]] = {}
        self.done: dict[str, dict[str, Any]] = {}  # a state shared by two pairs is read once

    def task(self, state: str) -> tuple[Any, dict[str, Tensor], dict[int, Tensor]]:
        """The state's task, its W0 cells and how often each neuron was in W0's own top-k masks (every row and position)."""
        name = self.inv["states"][state]["task"]
        if name not in self.tasks:
            t = HF.Task(WIN.load_read(name, self.inv["tasks"][name]), None, None)
            counts = {l: torch.zeros(W, dtype=torch.long) for l in range(N)}
            self.tasks[name] = (t, cells(self.net, t, self.c["k"], self.c["random_seed"], counts), counts)
        return self.tasks[name]

    def dW(self, state: str) -> ST.DW:
        d = torch.load(REPO / self.inv["states"][state]["file"], weights_only=False)
        return {k: v.float() for k, v in d.items()}

    def read(self, state: str, dW: ST.DW) -> dict[str, Any]:
        """Both readings of W0 + dW on the state's task (see the module docstring)."""
        t, c0, _ = self.task(state)
        with self.net.at(dW):
            c1 = cells(self.net, t, self.c["k"], self.c["random_seed"])
            top1 = pile_top(self.net, self.rows, self.c["n_top_windows"])
        ids = (c1["f"] - c0["f"]).abs().mean(dim=0).topk(self.c["n_changed"], dim=-1).indices  # (N, n_changed)
        per = JR.retention(self.top0, top1, (ids + torch.arange(N)[:, None] * W).flatten())
        gain, num, num_r = (float((c1[c] - c0[c]).mean()) for c in ("T", "M", "R"))
        return {"retention": float(per.mean()), "kept": JR.kept(num, gain), "kept_random": JR.kept(num_r, gain),
                "gain": gain, "num_M": num, "num_R": num_r, "top_changed": ids.tolist(), "retention_per_neuron": per.tolist()}  # fmt: skip

    def state(self, name: str) -> dict[str, Any]:
        if name not in self.done:
            self.done[name] = self.read(name, self.dW(name))
        return self.done[name]


def brief(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ("retention", "kept", "kept_random", "gain", "num_M", "num_R")}


def rows_edit(W0: ST.DW, ids: list[list[int]], how: str, factor: float) -> ST.DW:
    """c_fc rows of `ids` per block times `factor` ("harder") or cyclically shifted by one ("swapped"); all else W0."""
    out = {k: torch.zeros_like(w).cpu() for k, w in W0.items()}
    for l, rows in enumerate(ids):
        key, i = f"h.{l}.mlp.c_fc", torch.tensor(rows)
        w = W0[key].cpu()
        out[key][i] = w[i] * (factor - 1) if how == "harder" else w[i.roll(-1)] - w[i]
    return out


def down_on(dW: ST.DW, cols: dict[int, Tensor]) -> ST.DW:
    """dW's down_proj update kept on each block's columns `cols[l]`; every other matrix and column zero."""
    out = {k: torch.zeros_like(v) for k, v in dW.items()}
    for l, c in cols.items():
        key = f"h.{l}.mlp.down_proj"
        out[key][:, c] = dW[key][:, c]
    return out


def stage0(job: Job) -> dict[str, Any]:
    ref, W0, c, th = job.c["reference_state"], job.net.W0, job.c, job.th
    assert all(job.net.net.h[l].mlp.c_fc.bias is None for l in range(N))  # no MLP biases: "and biases" changes nothing
    torch.save({"top": job.top0.to(torch.int16), "pile_rows": c["pile_rows"], "flat": "l * 3072 + i"}, LARGE / "pile_top20_w0.pt")
    out: dict[str, Any] = {"zero": job.read(ref, {k: torch.zeros_like(w).cpu() for k, w in W0.items()})}
    z = out["zero"]
    say("0 zero", brief(z))
    if not (z["retention"] == 1.0 and z["num_M"] == 0.0 and z["gain"] == 0.0):
        raise halt(0, out, "C0_zero: zero update does not read retention exactly 1.0 with M and T numerators exactly 0")
    out["reference"] = job.state(ref)
    say("0 reference", brief(out["reference"]))
    for how in ("harder", "swapped"):
        out[how] = job.read(ref, rows_edit(W0, out["reference"]["top_changed"], how, c["harder_factor"]))
        say(f"0 {how}", brief(out[how]))
    out["C0_retention"] = {"harder": out["harder"]["retention"], "swapped": out["swapped"]["retention"],
                           "pass": out["harder"]["retention"] > out["swapped"]["retention"] + th["retention_control_margin"]}  # fmt: skip
    say("0 C0_retention", out["C0_retention"])
    if not out["C0_retention"]["pass"]:
        raise halt(0, out, "C0_retention: harder not above swapped by the margin")
    counts, real = job.task(ref)[2], job.dW(ref)
    for which, sign in (("often", 1), ("rare", -1)):
        cols = {l: (sign * counts[l]).topk(c["k"]).indices for l in range(N)}
        out[which] = {**job.read(ref, down_on(real, cols)), "cols": {l: v.tolist() for l, v in cols.items()}}
        say(f"0 {which}", brief(out[which]))
    o = out["often"]
    out["C0_topk"] = {"often_own": o["kept"], "often_random": o["kept_random"], "rare_own": out["rare"]["kept"],
                      "rare_random": out["rare"]["kept_random"], "pass": o["kept"] > o["kept_random"] + th["topk_control_margin"]}  # fmt: skip
    say("0 C0_topk", out["C0_topk"])
    if not out["C0_topk"]["pass"]:  # nan on either side fails
        raise halt(0, out, "C0_topk: own top-K not above random-K by the margin on the often-kept update")
    return out


def order(job: Job, pairs: list[str]) -> list[str]:
    return list(dict.fromkeys(s for p in pairs for side in job.names[p] for s in side))


def firsts(job: Job, pairs: list[str]) -> list[str]:
    """Each side's first seed, each once: the side's random update is matched to it."""
    return list(dict.fromkeys(job.names[p][i][0] for p in pairs for i in (0, 1)))


def stage1(job: Job, pairs: list[str], t_job: float) -> dict[str, Any]:
    names, t0, n_random = order(job, pairs), time.time(), len(firsts(job, pairs))
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


def stage2(job: Job, pairs: list[str], states: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"random": {}}
    for i, s in enumerate(firsts(job, pairs)):
        out["random"][s] = job.read(s, NO.random_like(job.dW(s), job.c["random_seed"] + 1 + i))
        say(f"2 random~{s}", brief(out["random"][s]))
        write(2, out)
    out["score"] = score(job, pairs, states, out["random"])
    out["predictions"] = JR.tally(out["score"])
    say("2 score", {p: {r: {k: v[r][k] for k in ("call", "gap", "band")} for r in READINGS} for p, v in out["score"].items()})
    say("2 predictions", out["predictions"])
    write(2, out)
    return out


def score(job: Job, pairs: list[str], states: dict[str, Any], rnd: dict[str, Any]) -> dict[str, Any]:
    """E155's call per pair and reading; each side's floor against its random update; the random-K values per side."""
    out: dict[str, Any] = {}
    for p in pairs:
        sd = dict(zip(JR.SIDES, job.names[p]))
        rec: dict[str, Any] = {}
        for r in READINGS:
            side = {s: [states[n][r] for n in sd[s]] for s in JR.SIDES}
            rec[r] = {**E155.call(side["reuse"], side["new"]), "values": side,
                      "floor": {s: E155.floor(v, rnd[sd[s][0]][r]) for s, v in side.items()}}  # fmt: skip
        rec["kept_random"] = {"values": {s: [states[n]["kept_random"] for n in sd[s]] for s in JR.SIDES},
                              "random_update": {s: rnd[sd[s][0]]["kept_random"] for s in JR.SIDES}}  # fmt: skip
        out[p] = rec
    return out


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E189") and LARGE.is_dir(), "run fleet/link-data first"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = Job(ST.Net("cuda"), P)
    pairs = list(P["pairs"])
    assert set(pairs) == set(job.names), (pairs, sorted(job.names))
    try:
        write(0, stage0(job))
        print("STAGE 0 DONE", flush=True)
        st = stage1(job, pairs, t_job)
        print("STAGE 1 DONE", flush=True)
        stage2(job, pairs, st["states"])
        print("STAGE 2 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
