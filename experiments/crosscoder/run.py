"""E203 (cell P2.31, run 2 of 2): the crosscoder baseline as its authors run it. A BatchTopK crosscoder trained with
Minder et al.'s pinned code and settings, between the base 67M and each saved fine-tune of the nine built pairs, as built
and size-matched (E187's c on the larger side), at block 2's input. Fine-tune-specific latents: the top N by relative
norm that pass latent scaling. The reading d = shared part - fine-tune-specific part of the change on the task's
score-half windows. Plan reports/plans/2026-11-17_plan_p2-31-run2.md (+ _appendix.md); settings, bars and expected
values results/run2/predictions.json (refuses unless committed unchanged). Run 1 (E198) is this file's previous version
in git; its record (results/predictions.json, stage0.json, stage01_pod.log) is untouched. Glue: `commute.crosscoder`.

Data per side, as the source's two halves (FineWeb + chat there): Pile train rows and the side's task fit-half windows,
equal tokens, drawn with replacement (the task text is too small to fill its half without repeats), shuffled within
chunks of `chunk_rows` Pile rows (their 1M-token local shuffling), every position but 0, `epochs` passes, the same order
for every side. The normaliser and the latent-scaling tokens come from the first chunk.

  prep            (CPU) assemble the cached Pile train rows the full run reads (artifacts/sets/train_0_<n>.npy)
  inputs i/n      (CPU) print the files a stage-2 shard reads, for the pod's upload
  0a, 0b, 0       stage 0 alone: (a) the base against itself, (b) a planted direction; scored at the snapshots
  t               step time at the full size, fp32 then TF32 (300 steps each)
  1               e54_matched seed 0 both sides, the random update, a crosscoder reseed; scored at the snapshots
  2 i/n           every other state, then the size-matched states, shard i of n; one file each in results/run2/sides/
  --rehearse      the rehearsal sizes (settings.rehearsal), outputs under results/run2_rehearsal/
  --stop-after S  save a checkpoint at step S and exit 3 (the rehearsal's resume test)
Prints every scored number and "STAGE x DONE".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import steering as ST
from commute.data import SETS_DIR
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRED = HERE / "results/run2/predictions.json"
SM = load_experiment(
    "size_matched", "run"
)  # pairs(): the share's states per pair and their inventory
W = load_experiment(
    "share_untrained", "windows"
)  # load_read: each task's read windows, as the share read them
M67 = (
    REPO / "experiments/size_matched/results/m67.json"
)  # E187's c and larger side per pair
ROW = 511  # tokens a 512-token Pile row gives with position 0 dropped
OUT: dict[str, Path] = {}  # results, sides, large: set in main by --rehearse


def log(msg: str) -> None:
    print(msg, flush=True)


class Stop(Exception):
    """A filed stop: the stage file records why; the job exits 1."""


def write(path: Path, out: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(path)


def check(
    out: dict[str, Any], path: Path, name: str, ok: bool, detail: dict[str, Any]
) -> None:
    out.setdefault("checks", {})[name] = {"pass": ok, **detail}
    log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
    if not ok:
        out["stopped"] = name
        write(path, out)
        raise Stop(name)


def rows(split: str, n: int) -> Tensor:
    """Pile rows 0..n of a split from the smallest cached file starting at 0 that holds them (`prep` makes the train one)."""
    have = sorted(
        (int(p.stem.split("_")[2]), p) for p in SETS_DIR.glob(f"{split}_0_*.npy")
    )
    path = next((p for m, p in have if m >= n), None)
    if path is None:
        raise SystemExit(
            f"no cached {split} file with {n} rows from 0 under {SETS_DIR}: run `run.py prep` first"
        )
    return torch.from_numpy(np.load(path, mmap_mode="r")[:n].copy()).long()


def train_rows(cfg: CC.Cfg) -> int:
    """Pile rows for half of one epoch's tokens, rounded up to whole chunks."""
    per_epoch = cfg.tokens / cfg.epochs / 2 / ROW
    return math.ceil(per_epoch / cfg.chunk_rows) * cfg.chunk_rows


def prep(cfg: CC.Cfg) -> None:
    """Rows 0..n of Pile train from the cached files 0_32768, 32768_32768 and 65536_524288, as one file."""
    n, parts, start = train_rows(cfg), [], 0
    for s, m in ((0, 32768), (32768, 32768), (65536, 524288)):
        assert s == start, (s, start)
        take = min(m, n - start)
        parts.append(np.load(SETS_DIR / f"train_{s}_{m}.npy", mmap_mode="r")[:take])
        start += take
        if start == n:
            break
    assert start == n, (start, n)
    out = SETS_DIR / f"train_0_{n}.npy"
    np.save(out, np.concatenate(parts))
    log(f"[prep] {out} ({n} rows, {out.stat().st_size / 1e6:.0f} MB)")


@dataclass
class Side:
    key: str  # file name stem
    task: Any  # share_untrained.windows.Read
    dW: ST.DW  # the update (zeros for the self control)
    plant: Tensor | None = (
        None  # (d,) added to the fine-tune side at each task window's answer position
    )
    seed: int = 0  # crosscoder seed


class Grab(Exception):
    """Raised by the site hook once the activation is taken, so the rest of the forward is skipped."""


def stage2_sides(
    names: dict[str, Any], m67: dict[str, Any]
) -> list[tuple[str, float, str]]:
    """(state, c, key): every state of the nine pairs once, then each pair's larger side times its c."""
    out = {s: (s, 1.0, s.replace("/", "__")) for r, n in names.values() for s in r + n}
    for pair, (r, n) in names.items():
        p = m67[pair]
        for s in r if p["larger"] == "reuse" else n:
            key = f"{s.replace('/', '__')}__matched_{pair}"
            out[key] = (s, float(p["c"]), key)
    return list(out.values())


class Job:
    def __init__(self, P: dict[str, Any], cfg: CC.Cfg) -> None:
        self.P, self.cfg, S = P, cfg, P["settings"]
        self.layer = S["layer"]
        self.net = ST.Net("cuda")
        self.names, _, self.inv = SM.pairs()
        self.m67 = json.loads(M67.read_text())["pairs"]
        self.train = rows("train", train_rows(cfg))
        self.val = rows("val", cfg.val_rows)
        self.reads = {t: W.load_read(t, spec) for t, spec in self.inv["tasks"].items()}
        self.zero = {k: torch.zeros_like(w) for k, w in self.net.W0.items()}
        log(f"[job] {cfg}\n[job] {len(self.train)} Pile train rows, {cfg.steps} steps")

    # ------------------------------------------------------------------------- activations

    @torch.no_grad()
    def acts(
        self,
        toks: Tensor,
        dW: ST.DW | None,
        plant: Tensor | None = None,
        pos: int = -1,
        n: int = 64,
    ) -> Tensor:
        """(m, d) raw block-`layer` inputs at every position but 0, under W0 (None) or W0 + dW, fp32 (TF32 is off for the
        whole job); `plant` added at position `pos`."""
        out: list[Tensor] = []

        def grab(x: Tensor) -> Tensor:
            x = x.clone()
            if plant is not None:
                x[:, pos] += plant.to(x.dtype)
            out.append(x[:, 1:])
            raise Grab

        with self.net.at(dW):
            for b in toks.split(n):
                try:
                    self.net.run(b, edit={f"in{self.layer}": grab})
                except Grab:
                    pass
        return torch.cat(out).reshape(-1, out[0].shape[-1])

    def pair(self, toks: Tensor, side: Side, task: bool) -> Tensor:
        """(m, 2, d): base then fine-tune; the plant only on task windows."""
        return torch.stack(
            [
                self.acts(toks, None),
                self.acts(toks, side.dW, side.plant if task else None, side.task.half),
            ],
            1,
        )

    def windows(self, task: Any, half: str) -> Tensor:
        toks = task.toks[:, : task.ci_len] if task.ci_len else task.toks
        return toks[0::2] if half == "fit" else toks[1::2]

    # ------------------------------------------------------------------------- the training stream

    def plan(self, side: Side) -> list[tuple[int, int, int, int]]:
        """(epoch, chunk, Pile rows, task windows) in stream order; chunk order permuted per epoch."""
        C, T = self.cfg.chunk_rows, self.windows(side.task, "fit").shape[1] - 1
        n = math.ceil(len(self.train) / C)
        out = []
        for e in range(self.cfg.epochs):
            order = torch.randperm(
                n,
                generator=torch.Generator().manual_seed(
                    self.P["settings"]["data_seed"] + e
                ),
            )
            for c in order.tolist():
                r = min(C, len(self.train) - c * C)
                out.append((e, c, r, math.ceil(r * ROW / T)))
        return out

    def chunk(self, side: Side, e: int, c: int, n_win: int) -> Tensor:
        """One chunk's (m, 2, d) raw pairs: its Pile rows and n_win task fit windows, shuffled together."""
        gen = torch.Generator().manual_seed(
            self.P["settings"]["data_seed"] * 100_003 + e * 1_009 + c
        )
        fit = self.windows(side.task, "fit")
        pick = fit[torch.randint(len(fit), (n_win,), generator=gen)]
        h = torch.cat(
            [
                self.pair(
                    self.train[c * self.cfg.chunk_rows :][: self.cfg.chunk_rows],
                    side,
                    False,
                ),
                self.pair(pick, side, True),
            ]
        )
        return h[torch.randperm(len(h), generator=gen).to(h.device)]

    def stream(self, side: Side, start: int) -> Iterator[Tensor]:
        """(B, 2, d) batches from step `start` on: whole chunks before it are skipped without computing them."""
        B, T = self.cfg.batch, self.windows(side.task, "fit").shape[1] - 1
        skip, buf = start * B, None
        for e, c, r, n_win in self.plan(side):
            size = r * ROW + n_win * T
            if skip >= size:
                skip -= size
                continue
            h = self.chunk(side, e, c, n_win)[skip:]
            skip, buf = 0, h if buf is None else torch.cat([buf, h])
            while len(buf) >= B:
                yield buf[:B]
                buf = buf[B:]

    # ------------------------------------------------------------------------- one crosscoder

    def run_side(self, side: Side, snap: bool, save_dec: bool, early: Callable[[dict[str, Any]], None] | None = None,
                 stop_after: int | None = None) -> dict[str, Any]:  # fmt: skip
        t0, cfg = time.time(), self.cfg
        e, c, _, n_win = self.plan(side)[0]
        h0 = self.chunk(side, e, c, n_win)
        mean, std = CC.normaliser(h0)
        h_ls, h_task = (
            h0[: cfg.ls_tokens].clone(),
            self.pair(self.windows(side.task, "score"), side, True),
        )
        del h0
        tr = CC.trainer(cfg, mean, std, side.seed)
        ck, start, snaps, curve = OUT["large"] / "ckpt" / f"{side.key}.pt", 0, [], []
        if ck.is_file():
            s = torch.load(ck, weights_only=False)
            start, snaps, curve = CC.load_state(tr, s), s["snaps"], s["curve"]
            log(f"[side {side.key}] resumed at step {start}")
        marks = {int(f * cfg.steps) for f in cfg.snapshots} if snap else set()
        log(f"[side {side.key}] training steps {start}..{cfg.steps}")

        def hook(step: int) -> None:
            if step in marks:
                snaps.append(
                    self.score(
                        tr.ae, side, h_ls, h_task, step, snaps[-1] if snaps else None
                    )[0]
                )
                self.show(side, snaps[-1])
                if early is not None:
                    early(snaps[-1])
            if step % cfg.ckpt_every == 0 or step in marks or step == stop_after:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(CC.state(tr, step) | {"snaps": snaps, "curve": curve}, ck)
            if step == stop_after:
                log(f"[side {side.key}] stopped after step {step} (--stop-after)")
                hard_exit(3)

        curve += CC.train(tr, self.stream(side, start), start, cfg.steps, log, hook)
        rec, large = self.score(
            tr.ae, side, h_ls, h_task, cfg.steps, snaps[-1] if snaps else None
        )
        if save_dec:
            large["W_dec"] = tr.ae.decoder.weight.detach().half().cpu()
        OUT["large"].mkdir(parents=True, exist_ok=True)
        torch.save(large, OUT["large"] / f"{side.key}.pt")
        rec |= {
            "key": side.key,
            "seed": side.seed,
            "snapshots": snaps,
            "curve": curve,
            "minutes": (time.time() - t0) / 60,
        }
        self.show(side, rec)
        ck.unlink(missing_ok=True)
        return rec

    def val_chunks(self, side: Side, h_task: Tensor) -> Iterator[Tensor]:
        for r in self.val.split(self.cfg.chunk_rows):
            yield self.pair(r, side, False)
        yield h_task

    def score(
        self,
        ae: Any,
        side: Side,
        h_ls: Tensor,
        h_task: Tensor,
        step: int,
        prev: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Every number of one crosscoder at `step`: FVE, the groups at N and at the wide N, the parts, d, the lead latent
        (the one carrying most of the change), the overlap of the specific set with the previous score, the planted cosine."""
        cfg, rn = self.cfg, CC.relative_norm(ae)
        ev = CC.fve(ae, self.val_chunks(side, h_task))
        alive = ev.pop("fired") >= cfg.alive_min
        ch = CC.carried(ae, h_task)
        s = ch.pop("s")
        lead = (
            int(torch.where(alive, s.nan_to_num(-math.inf), -math.inf).argmax())
            if ch["energy"] > 0
            else -1
        )
        J = torch.nonzero(
            CC.top(rn, cfg.top_n_wide) | (torch.arange(len(rn)) == lead)
        ).squeeze(1)
        b = CC.betas(ae, h_ls, J)
        spec = torch.zeros(len(rn), dtype=torch.bool)
        spec[J] = CC.specific(cfg, b)
        rec: dict[str, Any] = {"step": step, "tokens": step * cfg.batch, **ev, **ch, "theta": float(ae.threshold),
                               "dead": float((~alive).float().mean()),
                               "max_rn_dev_alive": float((rn[alive] - 0.5).abs().max()) if bool(alive.any()) else math.nan}  # fmt: skip
        g = {}
        for tag, n in (("", cfg.top_n), ("_wide", cfg.top_n_wide)):
            g[tag] = CC.groups(cfg, rn, alive, CC.top(rn, n), spec)
            p = CC.parts(s, ch["bias"], g[tag])
            rec |= {
                f"parts{tag}": p,
                f"counts{tag}": {k: int(m.sum()) for k, m in g[tag].items()},
                f"d{tag}": p["shared"] - p["ft_specific"],
            }
        rec["spec"] = torch.nonzero(g[""]["ft_specific"]).squeeze(1).tolist()
        if prev is not None:
            a, bb = set(prev["spec"]), set(rec["spec"])
            rec["jaccard_prev"] = len(a & bb) / len(a | bb) if a | bb else math.nan
        if lead >= 0:
            i = int((J == lead).nonzero())
            rec["lead"] = {"latent": lead, "s": float(s[lead]), "rn": float(rn[lead]), "rank": 1 + int((rn > rn[lead]).sum()),
                           "nu_r": float(b["nu_r"][i]), "nu_eps": float(b["nu_eps"][i]), "specific": bool(g[""]["ft_specific"][lead])}  # fmt: skip
        if side.plant is not None:
            Wr, _ = CC.raw(ae)
            dd = (Wr[1] - Wr[0])[g[""]["ft_specific"].to(Wr.device)]
            cos = torch.nn.functional.cosine_similarity(
                dd, side.plant[None].to(dd.device), dim=1
            )
            rec["cos_planted_best"] = float(cos.max()) if len(cos) else math.nan
        large = {"relative_norm": rn.half(), "alive": alive, "s": s.float(), "scaled": J, "groups": g[""],
                 "betas": {k: v.float() for k, v in b.items()}}  # fmt: skip
        return rec, large

    def show(self, side: Side, r: dict[str, Any]) -> None:
        keys = (
            "tokens",
            "fve_base",
            "fve_ft",
            "fve_change",
            "dead",
            "d",
            "d_wide",
            "parts",
            "counts",
            "lead",
            "jaccard_prev",
            "cos_planted_best",
            "minutes",
        )
        log(
            f"[side {side.key}] {json.dumps(jsonable({k: r[k] for k in keys if k in r}))}"
        )

    # ------------------------------------------------------------------------- the sides

    def state(
        self, name: str, c: float = 1.0, seed: int = 0, key: str | None = None
    ) -> Side:
        st = self.inv["states"][name]
        d = torch.load(REPO / st["file"], weights_only=False)
        dW = {k: v.float().to(self.net.W0[k].device) * c for k, v in d.items()}
        return Side(
            key or name.replace("/", "__"), self.reads[st["task"]], dW, seed=seed
        )

    def random_like(self, name: str) -> Side:
        """A Gaussian update at `name`'s per-matrix norms, the do-nothing reference."""
        s, gen = (
            self.state(name),
            torch.Generator().manual_seed(self.P["settings"]["random_seed"]),
        )
        R = {}
        for k, v in s.dW.items():
            r = torch.randn(v.shape, generator=gen).to(v.device)
            R[k] = r * (v.norm() / r.norm())
        return Side(f"random_like__{s.key}", s.task, R)

    def planted(self, task: str) -> Side:
        """The base plus one random unit direction times the median base norm at the answer position (run 1's seed)."""
        t, gen = (
            self.reads[task],
            torch.Generator().manual_seed(self.P["settings"]["plant_seed"]),
        )
        h = self.acts(self.windows(t, "fit")[:, : t.half + 1], None).reshape(
            -1, t.half, self.cfg.d
        )[:, -1]
        v = torch.randn(self.cfg.d, generator=gen)
        return Side(
            "planted",
            t,
            self.zero,
            plant=(v / v.norm() * h.norm(dim=1).median().cpu()).cuda(),
        )


# ----------------------------------------------------------------------------- stages


def stage0(job: Job, T: dict[str, Any], which: str, stop_after: int | None) -> None:
    path = OUT["results"] / "stage0.json"
    out = json.loads(path.read_text()) if path.is_file() else {}
    out |= {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name()}
    task = job.reads[job.P["settings"]["control_task"]]
    if "a" in which:
        toks = job.windows(task, "fit")[:8]
        diff = float((job.acts(toks, None) - job.acts(toks, job.zero)).abs().max())
        check(
            out,
            path,
            "a self path: equal activations",
            diff <= T["self_act_diff"],
            {"max_abs": diff, "bar": T["self_act_diff"]},
        )

        def early(r: dict[str, Any]) -> None:
            if r["step"] == min(int(f * job.cfg.steps) for f in job.cfg.snapshots):
                check(
                    out,
                    path,
                    "a self: FVE at the first snapshot",
                    min(r["fve_base"], r["fve_ft"]) >= T["fve_early"],
                    {"fve": [r["fve_base"], r["fve_ft"]], "bar": T["fve_early"]},
                )

        rec = out["self"] = job.run_side(
            Side("self", task, job.zero), True, True, early, stop_after
        )
        write(path, out)
        check(
            out,
            path,
            "a self: relative norms at 0.5",
            rec["max_rn_dev_alive"] <= T["self_rn_dev"],
            {"max_dev": rec["max_rn_dev_alive"], "bar": T["self_rn_dev"]},
        )
        check(
            out,
            path,
            "a self: no fine-tune-specific latent",
            rec["counts"]["ft_specific"] == 0,
            {"counts": rec["counts"]},
        )
        check(
            out,
            path,
            "a self: FVE",
            min(rec["fve_base"], rec["fve_ft"]) >= T["fve_min"],
            {"fve": [rec["fve_base"], rec["fve_ft"]], "bar": T["fve_min"]},
        )
        check(
            out,
            path,
            "a self: dead share",
            rec["dead"] <= T["dead_max"],
            {"dead": rec["dead"], "bar": T["dead_max"]},
        )
        out["pass_a"] = True
        write(path, out)
    if "b" in which:
        rec = out["planted"] = job.run_side(
            job.planted(job.P["settings"]["control_task"]), True, True
        )
        write(path, out)
        lead = rec.get("lead", {})
        ok = (bool(lead.get("specific")) or not T["planted_lead_specific"]) and rec[
            "parts"
        ]["ft_specific"] >= T["planted_part"]
        check(
            out,
            path,
            "b planted: found, specific and carried",
            ok,
            {
                "lead": lead,
                "part": rec["parts"]["ft_specific"],
                "bar": T["planted_part"],
            },
        )
        out["pass_b"] = True
        write(path, out)


def timing(job: Job, cfg: CC.Cfg) -> None:
    """ms per step at the run's size, fp32 (the source) and TF32, 300 steps each after 50 warm ones; peak memory."""
    side = Side("self", job.reads[job.P["settings"]["control_task"]], job.zero)
    out: dict[str, Any] = {"card": torch.cuda.get_device_name()}
    h = next(job.stream(side, 0))
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        tr = CC.trainer(cfg, *CC.normaliser(h), 0)
        for i in range(
            50
        ):  # step numbers past the k annealing and the threshold start, as most of a run is
            tr.update(5000 + i, h.clone())
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(50, 350):
            tr.update(5000 + i, h.clone())
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 300 * 1000
        out["tf32" if tf32 else "fp32"] = {
            "ms_per_step": ms,
            "train_minutes": ms * cfg.steps / 60_000,
        }
        log(
            f"[timing] tf32={tf32}: {ms:.2f} ms/step, {ms * cfg.steps / 60_000:.1f} min of steps per crosscoder"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    write(OUT["results"] / "timing.json", out)


def stage1(job: Job, T: dict[str, Any]) -> None:
    path, out = (
        OUT["results"] / "stage1.json",
        {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name()},
    )
    s0 = json.loads((OUT["results"] / "stage0.json").read_text())
    if not (s0.get("pass_a") and s0.get("pass_b")):
        raise SystemExit("refusing stage 1: stage0.json has not passed both controls")
    fve0 = min(s0["self"]["fve_base"], s0["self"]["fve_ft"])
    reuse, new = job.P["settings"]["first_pair_states"]
    for key, side in (("reuse", job.state(reuse)), ("new", job.state(new)), ("random", job.random_like(reuse)),
                      ("reuse_seed1", job.state(reuse, seed=1, key=f"{reuse.replace('/', '__')}__seed1"))):  # fmt: skip
        rec = out[key] = job.run_side(side, True, True)
        if key in ("reuse", "new"):
            write(
                OUT["sides"] / f"{side.key}.json",
                rec | {"state": side.key.replace("__", "/"), "c": 1.0},
            )
        write(path, out)
        f = min(rec["fve_base"], rec["fve_ft"])
        check(
            out,
            path,
            f"1 FVE {key}",
            f >= fve0 - T["fve_drop"],
            {"fve": f, "self": fve0, "drop": T["fve_drop"]},
        )
    dd = abs(out["reuse"]["d"] - out["reuse_seed1"]["d"])
    check(
        out,
        path,
        "1 crosscoder reseed",
        dd <= T["reseed_d"],
        {"d0": out["reuse"]["d"], "d1": out["reuse_seed1"]["d"], "bar": T["reseed_d"]},
    )
    expl = {k: 1 - out[k]["parts"]["error"] for k in ("reuse", "new")}
    check(
        out,
        path,
        "1 change reconstructed",
        max(expl.values()) >= T["explained_min"],
        {"explained": expl, "bar": T["explained_min"]},
    )
    out["pass"] = True
    write(path, out)


def stage2(job: Job, T: dict[str, Any], shard: tuple[int, int], t_job: float) -> None:
    s1 = OUT["results"] / "stage1.json"
    if not (s1.is_file() and json.loads(s1.read_text()).get("pass")):
        raise SystemExit("refusing stage 2: stage1.json has not passed")
    done = {
        s.replace("/", "__") for s in job.P["settings"]["first_pair_states"]
    }  # read in stage 1
    todo = [s for i, s in enumerate(stage2_sides(job.names, job.m67))
            if i % shard[1] == shard[0] and s[2] not in done and not (OUT["sides"] / f"{s[2]}.json").is_file()]  # fmt: skip
    log(f"[stage 2] shard {shard[0]}/{shard[1]}: {len(todo)} sides to train")
    for i, (state, c, key) in enumerate(todo):
        rec = job.run_side(job.state(state, c, key=key), False, False)
        write(OUT["sides"] / f"{key}.json", rec | {"state": state, "c": c})
        minutes = (time.time() - t_job) / 60 / (i + 1) * len(todo)
        log(
            f"[stage 2] {i + 1}/{len(todo)} done ({key}); projected shard minutes {minutes:.0f}"
        )
        if minutes > T["projected_limit_min"]:
            log(
                f"STOP: projected {minutes:.0f} min over {T['projected_limit_min']}: tell the lead"
            )
            raise Stop("projected time")


def inputs(shard: tuple[int, int]) -> None:
    """The files a stage-2 shard reads beyond the common ones (artifacts/vpd, the Pile sets, the task windows)."""
    names, _, inv = SM.pairs()
    m67 = json.loads(M67.read_text())["pairs"]
    sides = [
        s for i, s in enumerate(stage2_sides(names, m67)) if i % shard[1] == shard[0]
    ]
    files = sorted({inv["states"][s]["file"] for s, _, _ in sides})
    tasks = sorted(
        {
            inv["tasks"][inv["states"][s]["task"]].get(
                "path", "(built from the Pile val set)"
            )
            for s, _, _ in sides
        }
    )
    size = sum((REPO / f).stat().st_size for f in files)
    print(json.dumps({"shard": f"{shard[0]}/{shard[1]}", "sides": [k for _, _, k in sides], "states": files, "tasks": tasks,
                      "state_mb": round(size / 1e6)}, indent=1))  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "inputs", "0", "0a", "0b", "t", "1", "2"])
    ap.add_argument("shard", nargs="?", default="0/1")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    t_job, P = time.time(), json.loads(PRED.read_text())
    tag = "rehearsal" if a.rehearse else "full"
    S = P["settings"][tag]
    cfg = CC.Cfg(
        **(S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})
    )
    sub = "run2_rehearsal" if a.rehearse else "run2"
    OUT.update(
        results=HERE / "results" / sub,
        sides=HERE / "results" / sub / "sides",
        large=HERE / "results/large" / sub,
    )
    shard = tuple(int(x) for x in a.shard.split("/"))
    if a.stage == "prep":
        prep(cfg)
        return 0
    if a.stage == "inputs":
        inputs((shard[0], shard[1]))
        return 0
    require_filed(PRED)
    if P["experiment"] != "E203":
        raise SystemExit(
            "refusing to start: results/run2/predictions.json is not E203's"
        )
    torch.backends.cuda.matmul.allow_tf32 = (
        False  # fp32 throughout, as the source's code runs
    )
    T = P["thresholds"][tag]
    job = Job(P, cfg)
    try:
        if a.stage in ("0", "0a", "0b"):
            stage0(job, T, "ab" if a.stage == "0" else a.stage[1], a.stop_after)
            print(f"STAGE {a.stage} DONE", flush=True)
        elif a.stage == "t":
            timing(job, cfg)
            print("STAGE t DONE", flush=True)
        elif a.stage == "1":
            stage1(job, T)
            print("STAGE 1 DONE", flush=True)
        else:
            stage2(job, T, (shard[0], shard[1]), t_job)
            print("STAGE 2 DONE", flush=True)
    except Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
