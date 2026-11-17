"""E198 (cell P2.31): a model-diffing baseline for the part 2 scorecard. A BatchTopK crosscoder with latent scaling
(Minder et al. 2025; method from Lindsey et al. 2024) between the base 67M and each saved fine-tune of the nine built
pairs, as built and size-matched (E187's one scalar c on the larger side), at the middle residual (block 2's input).
Plan reports/plans/2026-11-17_plan_p2-31-crosscoder.md; settings, stop bars and expected values results/predictions.json
(the run refuses unless it is committed unchanged). Descriptive: sides named as built, no reuse/new verdict.
The crosscoder, latent scaling and the reading are `commute.crosscoder`; the figure is plot.py (CPU, after the run).

Per side: train on Pile train rows 0-32767 plus the side's task fit-half windows (task_frac of tokens), one pass, every
position but 0; score on held-out data: latent groups and latent scaling on Pile val rows 0-255 plus the task's score
half; the change reading on the task's score half. Same crosscoder seed and data order for every side.

  stage 0, alone, a miss writes, logs STOP and exits 1 (results/stage0.json):
    (a) self: the base against itself through the fine-tune path (a zero update): activations equal, no relative norm
        off 0.5, no fine-tune-only latent, FVE and dead share within the filed bars;
    (b) planted: the fine-tune side is the base plus one random direction at each task window's answer position; at
        least one fine-tune-specific latent, and fine-tune-specific latents carry at least the filed part of the change.
  stage 1, one pair (e54_matched, seed 0 of each side), a random update of the reuse state's per-matrix norms (the
    do-nothing reference), and the reuse state again with crosscoder seed 1 (results/stage1.json): FVE within the
    filed drop of (a)'s, the two crosscoder seeds' d within the filed bar, some of the change reconstructed.
  stage 2, every other state, then the size-matched states, one file each in results/sides/ as it lands; the projected
    time checked after the first. `--shard i/n` splits stage 2 over jobs.
Prints every scored number and "STAGE n DONE".

    sbatch --comment=E198 cluster/crosscoder.sbatch            # stages 0 and 1
    sbatch --comment=E198 cluster/crosscoder.sbatch 2 0/4      # stage 2, shard 0 of 4 (after stage 1 passed)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, SIDES, LARGE = RESULTS / "predictions.json", RESULTS / "sides", RESULTS / "large"
SM = load_experiment("size_matched", "run")  # pairs(): the share's states per pair (E136, E147 names) and their inventory
W = load_experiment("share_untrained", "windows")  # load_read: each task's read windows, as the share read them
M67 = REPO / "experiments/size_matched/results/m67.json"  # E187's c and larger side per pair


def log(msg: str) -> None:
    print(msg, flush=True)


class Stop(Exception):
    """A filed stop: the stage file records why; the job exits 1."""


def check(out: dict[str, Any], path: Path, name: str, ok: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"pass": ok, **detail}
    log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
    if not ok:
        out["stopped"] = name
        write(path, out)
        raise Stop(name)


def write(path: Path, out: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(path)


@dataclass
class Side:
    key: str  # file name stem
    task: Any  # share_untrained.windows.Read
    dW: ST.DW  # the update (zeros for the self control)
    plant: Tensor | None = None  # (d,) added to the fine-tune side at each task window's answer position
    seed: int = 0  # crosscoder seed


class Grab(Exception):
    """Raised by the site hook once the activation is taken, so the rest of the forward is skipped."""


class Job:
    def __init__(self, P: dict[str, Any]) -> None:
        self.P, S = P, P["settings"]
        self.cfg = CC.Cfg(**(S["crosscoder"] | {"shared": tuple(S["crosscoder"]["shared"])}))
        self.layer, self.chunk_rows, self.task_frac = S["layer"], S["chunk_rows"], S["task_frac"]
        self.net = ST.Net("cuda")
        self.names, _, self.inv = SM.pairs()
        self.m67 = json.loads(M67.read_text())["pairs"]
        self.train_rows = pile_rows("train", *S["pile_train"])
        self.val = pile_rows("val", *S["pile_val"])
        self.reads = {t: W.load_read(t, spec) for t, spec in self.inv["tasks"].items()}
        self.zero = {k: torch.zeros_like(w) for k, w in self.net.W0.items()}
        self.epochs, self.scale = S["epochs"], 1.0
        h = self.acts(self.train_rows[:64], None)  # one scalar for both models: base mean norm to sqrt(d) (Anthropic's SAE practice)
        self.scale = math.sqrt(self.cfg.d) / float(h.norm(dim=1).mean())
        log(f"[job] activation scale {self.scale:.5g} (base mean norm {float(h.norm(dim=1).mean()):.4g} at block {self.layer}'s input)")

    def windows(self, task: Any, half: str) -> Tensor:
        toks = task.toks[: , : task.ci_len] if task.ci_len else task.toks
        return toks[0::2] if half == "fit" else toks[1::2]

    @torch.no_grad()
    def acts(self, toks: Tensor, dW: ST.DW | None, plant: Tensor | None = None, pos: int = -1, rows: int = 64) -> Tensor:
        """(n, d): block `layer`'s input at every position but 0, under W0 (None: the plain path) or W0 + dW, in fp32
        with TF32 off (the crosscoder's own matmuls use TF32); `plant` added at position `pos`; times the job's scale."""
        out: list[Tensor] = []

        def grab(x: Tensor) -> Tensor:
            out.append(x[:, 1:].clone() if plant is None else _plant(x, plant, pos)[:, 1:])
            raise Grab

        torch.backends.cuda.matmul.allow_tf32 = False
        with self.net.at(dW):
            for b in toks.split(rows):
                try:
                    self.net.run(b, edit={f"in{self.layer}": grab})
                except Grab:
                    pass
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.cat(out).reshape(-1, out[0].shape[-1]) * self.scale

    def pair_acts(self, toks: Tensor, side: Side, task_rows: bool) -> Tensor:
        plant = side.plant if task_rows else None
        return torch.stack([self.acts(toks, None), self.acts(toks, side.dW, plant, side.task.half)])

    def mix(self, side: Side) -> tuple[Tensor, int, int]:
        """The task fit windows, task windows per Pile chunk (task_frac of tokens) and the run's steps over all epochs."""
        fit = self.windows(side.task, "fit")
        per_chunk = self.chunk_rows * (self.train_rows.shape[1] - 1)
        n_win = math.ceil(self.task_frac / (1 - self.task_frac) * per_chunk / (fit.shape[1] - 1))
        n_chunks = math.ceil(len(self.train_rows) / self.chunk_rows)
        return fit, n_win, self.epochs * n_chunks * (per_chunk + n_win * (fit.shape[1] - 1)) // self.cfg.batch

    def batches(self, side: Side):
        """(2, batch, d) batches: Pile chunks of `chunk_rows` rows mixed with task fit windows drawn with replacement,
        shuffled within the chunk; `epochs` passes, the same order for every side."""
        gen = torch.Generator().manual_seed(self.P["settings"]["data_seed"])
        (fit, n_win, n_steps), B, buf, done = self.mix(side), self.cfg.batch, None, 0
        for _ in range(self.epochs):
            for rows in self.train_rows.split(self.chunk_rows):
                pick = fit[torch.randint(len(fit), (n_win,), generator=gen)]
                h = torch.cat([self.pair_acts(rows, side, False), self.pair_acts(pick, side, True)], 1)
                h = h[:, torch.randperm(h.shape[1], generator=gen).to(h.device)]
                buf = h if buf is None else torch.cat([buf, h], 1)
                while buf.shape[1] >= B and done < n_steps:
                    yield buf[:, :B]
                    buf, done = buf[:, B:], done + 1

    def run_side(self, side: Side, save_dec: bool = False) -> dict[str, Any]:
        """Train one crosscoder, score it, save its per-latent vectors; the summary record."""
        t0 = time.time()
        cc, n = CC.Crosscoder(self.cfg, side.seed, "cuda"), self.mix(side)[2]
        log(f"[side {side.key}] training {n} steps")
        tr = CC.train(cc, self.batches(side), n, log)
        score = self.windows(side.task, "score")
        h_task = self.pair_acts(score, side, True)
        h_val = torch.cat([self.pair_acts(self.val, side, False), h_task], 1)
        g, info = CC.groups(cc, h_val)
        ch = CC.change_parts(cc, h_task, g)
        rn = info.pop("relative_norm")
        alive = ~g["dead"]
        large = {"relative_norm": rn.half(), "alive": alive.cpu(), "s": ch.pop("s").float(), "scaled": info.pop("scaled").cpu(),
                 "scaling": {k: v.float() for k, v in info.pop("scaling").items()}, "groups": {k: v.cpu() for k, v in g.items()}}  # fmt: skip
        if save_dec:
            large["W_dec"] = cc.W_dec.detach().half().cpu()
        LARGE.mkdir(parents=True, exist_ok=True)
        torch.save(large, LARGE / f"{side.key}.pt")
        rec = {"key": side.key, "seed": side.seed, "scale": self.scale, **tr, **info, **ch, "minutes": (time.time() - t0) / 60,
               "max_rn_dev_alive": float((rn[alive.cpu()].float() - 0.5).abs().max()) if bool(alive.any()) else math.nan}  # fmt: skip
        log(f"[side {side.key}] {json.dumps(jsonable({k: rec[k] for k in ('d', 'parts', 'counts', 'fve_base', 'fve_ft', 'l0', 'change_energy', 'theta', 'minutes')}))}")
        return rec

    # ----------------------------------------------------------------------------- the sides

    def state(self, name: str, c: float = 1.0, seed: int = 0, key: str | None = None) -> Side:
        st = self.inv["states"][name]
        d = torch.load(REPO / st["file"], weights_only=False)
        dW = {k: v.float().to(self.net.W0[k].device) * c for k, v in d.items()}
        return Side(key or name.replace("/", "__"), self.reads[st["task"]], dW, seed=seed)

    def random_like(self, name: str) -> Side:
        """A Gaussian update at `name`'s per-matrix norms, the do-nothing reference."""
        s, gen = self.state(name), torch.Generator().manual_seed(self.P["settings"]["random_seed"])
        R = {}
        for k, v in s.dW.items():
            r = torch.randn(v.shape, generator=gen).to(v.device)
            R[k] = r * (v.norm() / r.norm())
        return Side(f"random_like__{s.key}", s.task, R)

    def planted(self, task: str) -> Side:
        """The base plus one random unit direction times the median base norm at the answer position, task windows only."""
        t, gen = self.reads[task], torch.Generator().manual_seed(self.P["settings"]["plant_seed"])
        h = self.acts(self.windows(t, "fit")[:, : t.half + 1], None).reshape(-1, t.half, self.cfg.d)[:, -1]
        v = torch.randn(self.cfg.d, generator=gen)
        size = h.norm(dim=1).median().cpu() / self.scale  # the plant is added before the scale
        return Side("planted", t, self.zero, plant=(v / v.norm() * size).cuda())

    def stage2_sides(self) -> list[tuple[str, float, str]]:
        """(state, c, key): every state of the nine pairs once, then each pair's larger side times its c."""
        out = {s: (s, 1.0, s.replace("/", "__")) for r, n in self.names.values() for s in r + n}
        for pair, (r, n) in self.names.items():
            p = self.m67[pair]
            for s in r if p["larger"] == "reuse" else n:
                key = f"{s.replace('/', '__')}__matched_{pair}"
                out[key] = (s, float(p["c"]), key)
        return list(out.values())


def _plant(x: Tensor, v: Tensor, pos: int) -> Tensor:
    x = x.clone()
    x[:, pos] += v.to(x.dtype)
    return x


# ----------------------------------------------------------------------------- stages


def stage0(job: Job, T: dict[str, Any]) -> dict[str, Any]:
    path, out = RESULTS / "stage0.json", {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name()}
    task = job.reads[job.P["settings"]["control_task"]]
    toks = job.windows(task, "fit")[:8]
    diff = float((job.acts(toks, None) - job.acts(toks, job.zero)).abs().max())  # the plain path against the update path
    check(out, path, "a self path: equal activations", diff <= T["self_act_diff"], {"max_abs": diff, "bar": T["self_act_diff"]})
    rec = out["self"] = job.run_side(Side("self", task, job.zero), save_dec=True)
    write(path, out)
    c = rec["counts"]
    check(out, path, "a self: relative norms at 0.5", rec["max_rn_dev_alive"] <= T["self_rn_dev"], {"max_dev": rec["max_rn_dev_alive"], "bar": T["self_rn_dev"]})
    check(out, path, "a self: no fine-tune-only latent", c["ft_specific"] + c["ft_artefact"] == 0, {"counts": c})
    check(out, path, "a self: FVE", min(rec["fve_base"], rec["fve_ft"]) >= T["fve_min"], {"fve": [rec["fve_base"], rec["fve_ft"]], "bar": T["fve_min"]})
    dead = c["dead"] / job.cfg.latents
    check(out, path, "a self: dead share", dead <= T["dead_max"], {"dead": dead, "bar": T["dead_max"]})
    rec = out["planted"] = job.run_side(job.planted(job.P["settings"]["control_task"]), save_dec=True)
    write(path, out)
    ok = rec["counts"]["ft_specific"] >= 1 and rec["parts"]["ft_specific"] >= T["planted_part"]
    check(out, path, "b planted: found and carried", ok, {"ft_specific": rec["counts"]["ft_specific"], "part": rec["parts"]["ft_specific"], "bar": T["planted_part"]})
    out["pass"] = True
    write(path, out)
    return out


def stage1(job: Job, T: dict[str, Any], fve0: float) -> None:
    path, out = RESULTS / "stage1.json", {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name()}
    reuse, new = job.P["settings"]["first_pair_states"]
    for key, side in (("reuse", job.state(reuse)), ("new", job.state(new)), ("random", job.random_like(reuse)),
                      ("reuse_seed1", job.state(reuse, seed=1, key=f"{reuse.replace('/', '__')}__seed1"))):  # fmt: skip
        rec = out[key] = job.run_side(side, save_dec=True)
        if key in ("reuse", "new"):
            write(SIDES / f"{side.key}.json", rec | {"state": side.key.replace("__", "/"), "c": 1.0})
        write(path, out)
        f = min(rec["fve_base"], rec["fve_ft"])
        check(out, path, f"1 FVE {key}", f >= fve0 - T["fve_drop"], {"fve": f, "self": fve0, "drop": T["fve_drop"]})
    dd = abs(out["reuse"]["d"] - out["reuse_seed1"]["d"])
    check(out, path, "1 crosscoder reseed", dd <= T["reseed_d"], {"d0": out["reuse"]["d"], "d1": out["reuse_seed1"]["d"], "bar": T["reseed_d"]})
    expl = {k: 1 - out[k]["parts"]["error"] for k in ("reuse", "new")}
    check(out, path, "1 change reconstructed", max(expl.values()) >= T["explained_min"], {"explained": expl, "bar": T["explained_min"]})
    out["pass"] = True
    write(path, out)


def stage2(job: Job, T: dict[str, Any], shard: tuple[int, int], t_job: float) -> None:
    todo = [s for i, s in enumerate(job.stage2_sides()) if i % shard[1] == shard[0] and not (SIDES / f"{s[2]}.json").is_file()]
    log(f"[stage 2] shard {shard[0]}/{shard[1]}: {len(todo)} sides to train")
    for i, (state, c, key) in enumerate(todo):
        rec = job.run_side(job.state(state, c, key=key))
        write(SIDES / f"{key}.json", rec | {"state": state, "c": c})
        minutes = (time.time() - t_job) / 60 / (i + 1) * len(todo)
        log(f"[stage 2] {i + 1}/{len(todo)} done; projected shard minutes {minutes:.0f}")
        if minutes > T["projected_limit_min"]:
            log(f"STOP: projected {minutes:.0f} min over {T['projected_limit_min']}: tell the lead")
            raise Stop("projected time")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="?", default="01", choices=["01", "2"])
    ap.add_argument("shard", nargs="?", default="0/1")
    a = ap.parse_args()
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if P["experiment"] == "E???":
        raise SystemExit("refusing to start: predictions.json has no experiment number")
    torch.backends.cuda.matmul.allow_tf32 = True  # the crosscoder's matmuls; `Job.acts` turns it off for the model
    T = P["thresholds"]
    job = Job(P)
    try:
        if a.stages == "01":
            s0 = stage0(job, T)
            print("STAGE 0 DONE", flush=True)
            stage1(job, T, min(s0["self"]["fve_base"], s0["self"]["fve_ft"]))
            print("STAGE 1 DONE", flush=True)
        else:
            s1 = RESULTS / "stage1.json"
            if not (s1.is_file() and json.loads(s1.read_text()).get("pass")):
                raise SystemExit("refusing stage 2: results/stage1.json has not passed")
            i, n = (int(x) for x in a.shard.split("/"))
            stage2(job, T, (i, n), t_job)
            print("STAGE 2 DONE", flush=True)
    except Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
