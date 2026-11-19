"""E212 (realistic_tasks run 3), cell P2.38 run 3 of 3: the crosscoder of P2.31/E203 (Minder et al. 2025's BatchTopK
crosscoder with latent scaling, their pinned trainer, E203's settings) between the original 67M and each of the six
real-text fine-tunes run 2 (E211) saved, seed 0 at each pair's larger matched size, with a Gaussian update and E203's
planted direction as controls. Plan reports/plans/2026-11-18_plan_p2-38-run3.md, card ..._p2-38-run3-card.md; settings,
bars and predictions results/predictions_run3.json (refuses unless committed unchanged). Shared steps: commute.crosscoder
(the trainer glue and E203's reading, d in both forms); pair gaps and the card call: score3.py on the CPU.

One stream for every job, so jobs differ only in the update: E203's Pile half (chunks of `chunk_rows` Pile train rows
from `pile_start`) and, in each chunk, as many tokens of the six run-2 tasks' training texts, drawn with a generator
seeded by the chunk alone: the label tasks' 512 pool windows (L1 and L3 share them) at most `repeat_cap` draws a window
an epoch (E203's cap), the four LM tasks' 2,400 training rows filling the rest in equal parts; every position but 0;
shuffled within the chunk; `epochs` passes, the chunk order permuted per epoch. Normaliser and latent-scaling tokens
from the first chunk (E203).

  prep     (GPU, cluster) each task's training units and run-2 read rows from run 2's own loaders (run2.task_data) into
           results/large/run3/texts.pt; every other job reads only it, the cached Pile rows and its update
  planted  E203's planted control on this stream: one random direction (E203's plant seed) at the median base norm of the
           label read position, added on the fine-tune side at every `every`-th position of each task unit (`every` =
           the label window's length, so a label window gets it at its read position); read on the label read rows.
           A miss is recorded and reported beside, not a stop
  gauss    a Gaussian update at T2 seed 0's per-matrix norms (run 2's commute.finetune.random_direction, draw 0), read on
           every side's read rows; d_authors on T2's rows at or over the filed bar exits 1 and the sides are not run
  L1 L3 T1 T2 T3 T4  one side each, read on its task's run-2 read rows; refuses until run3_gauss.json has passed
  --rehearsal   the rehearsal sizes; outputs under results/run3_rehearsal/ and results/large/run3_rehearsal/
  --stop-after S   checkpoint after step S and exit 3 (the rehearsal's resume test); the same line again resumes
Each job writes results/run3_<job>.json at its end or at a filed stop, prints every scored number and "STAGE <job> DONE".
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import finetune as FT
from commute import steering as ST
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable
from commute.realistic import update_size

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, TEXTS = RESULTS / "predictions_run3.json", RESULTS / "large/run3/texts.pt"
SIDES = ("L1", "L3", "T1", "T2", "T3", "T4")
LABEL = ("L1", "L3")
ROW = 511  # tokens a 512-token row gives with position 0 dropped


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stop(Exception):
    """A filed stop: the job's file records why; the job exits 1."""


class Out:
    def __init__(self, path: Path, job: str, experiment: str) -> None:
        self.path, self.d = path, {"job": job, "experiment": experiment, "card": torch.cuda.get_device_name(), "checks": {}}

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jsonable(self.d), indent=1))
        tmp.replace(self.path)

    def check(self, name: str, ok: bool, detail: dict[str, Any], stop: bool = True) -> None:
        self.d["checks"][name] = {"pass": ok, **detail}
        log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
        if not ok and stop:
            self.d["stopped"] = name
            self.write()
            raise Stop(name)


@dataclass
class Side:
    key: str
    dW: ST.DW
    reads: tuple[str, ...]  # the read rows it is scored on; the first is its own (snapshots, stop rule)
    plant: Tensor | None = None


class Job:
    def __init__(self, P: dict[str, Any], cfg: CC.Cfg, th: dict[str, Any], out: Out, large: Path, stop_after: int | None) -> None:
        self.S, self.cfg, self.th, self.out, self.large, self.stop_after = P["settings"], cfg, th, out, large, stop_after
        self.layer, self.t0 = self.S["layer"], time.time()
        self.net = ST.Net("cuda")
        texts = torch.load(TEXTS, weights_only=False)
        self.units = {k: v.long() for k, v in texts["units"].items()}
        self.reads = {k: v.long() for k, v in texts["reads"].items()}
        self.every = self.units["L1"].shape[1]
        n_pile = math.ceil(cfg.tokens / cfg.epochs / 2 / ROW / cfg.chunk_rows) * cfg.chunk_rows  # E203's Pile half
        self.pile, self.val = cached_rows("train", cfg.pile_start, n_pile), cached_rows("val", 0, cfg.val_rows)
        self.n_chunks = n_pile // cfg.chunk_rows
        self.per_chunk = self.counts()
        size = cfg.chunk_rows * ROW + sum(n * (self.units[t].shape[1] - 1) for t, n in self.per_chunk.items())
        self.order = [(e, c) for e in range(cfg.epochs)
                      for c in torch.randperm(self.n_chunks, generator=torch.Generator().manual_seed(self.S["data_seed"] + e)).tolist()]  # fmt: skip
        self.sizes = [size] * len(self.order)
        task = sum(n * (self.units[t].shape[1] - 1) for t, n in self.per_chunk.items())
        out.d["stream"] = {"per_chunk": self.per_chunk, "chunk_tokens": size, "task_share": task / size, "chunks": len(self.order),
                           "tokens": sum(self.sizes), "plant_every": self.every}  # fmt: skip
        log(f"[job] {cfg}\n[job] stream {json.dumps(out.d['stream'])}")
        assert sum(self.sizes) // cfg.batch >= cfg.steps, f"the stream gives {sum(self.sizes)} tokens, {cfg.steps} steps need more"
        self.zero = {k: torch.zeros_like(w) for k, w in self.net.W0.items()}
        u = self.units["L1"][:8]
        diff = float((CC.acts(self.net, u, None, self.layer) - CC.acts(self.net, u, self.zero, self.layer)).abs().max())
        out.check("self path: a zero update gives W0's activations", diff <= th["self_act_diff"], {"max_abs": diff, "bar": th["self_act_diff"]})

    def counts(self) -> dict[str, int]:
        """Task units per chunk: an equal sixth of the Pile half's tokens each, the label tasks capped at repeat_cap draws
        of a window an epoch (E203's cap), the LM tasks sharing what the label tasks leave."""
        half, L = self.cfg.chunk_rows * ROW, {}
        for t in LABEL:
            n, T = self.units[t].shape
            L[t] = min(math.ceil(half / len(SIDES) / (T - 1)), max(1, self.cfg.repeat_cap * n // self.n_chunks))
        rest = half - sum(n * (self.units[t].shape[1] - 1) for t, n in L.items())
        lm = [t for t in SIDES if t not in LABEL]
        return L | {t: math.ceil(rest / len(lm) / (self.units[t].shape[1] - 1)) for t in lm}

    def chunk(self, i: int, side: Side) -> Tensor:
        e, c = self.order[i]
        gen = torch.Generator().manual_seed(self.S["data_seed"] * 100_003 + e * 1_009 + c)
        C = self.cfg.chunk_rows
        h = [CC.pairs(self.net, self.pile[c * C : (c + 1) * C], side.dW, self.layer)]
        for t, n in self.per_chunk.items():
            u = self.units[t]
            h.append(CC.pairs(self.net, u[torch.randint(len(u), (n,), generator=gen)], side.dW, self.layer, side.plant, self.every))
        x = torch.cat(h)
        return x[torch.randperm(len(x), generator=gen).to(x.device)]

    def h_read(self, side: Side, name: str) -> Tensor:
        return CC.pairs(self.net, self.reads[name], side.dW, self.layer, side.plant, self.every)

    def ev(self, ae: Any, side: Side) -> dict[str, Any]:
        """FVE, firing counts and L0 on the validation mix, the same for every job: Pile val rows and every read set."""
        pile = (CC.pairs(self.net, r, side.dW, self.layer) for r in self.val.split(self.cfg.chunk_rows))
        return CC.fve(ae, itertools.chain(pile, (self.h_read(side, t) for t in SIDES if t != "L3")))  # one chunk at a time

    def fit(self, side: Side) -> dict[str, Any]:
        cfg, th, out, stop_after = self.cfg, self.th, self.out, self.stop_after
        h0 = self.chunk(0, side)
        tr, h_ls = CC.trainer(cfg, *CC.normaliser(h0), 0), h0[: cfg.ls_tokens].clone()
        del h0
        h_own = self.h_read(side, side.reads[0])
        ck, start, snaps, curve = self.large / "ckpt" / f"{side.key}.pt", 0, [], []
        if ck.is_file():
            s = torch.load(ck, weights_only=False)
            start, snaps, curve = CC.load_state(tr, s), s["snaps"], s["curve"]
            log(f"[{side.key}] resumed at step {start}")
        marks, t_train = sorted({int(f * cfg.steps) for f in cfg.snapshots}), time.time()

        def hook(step: int) -> None:
            if step == th["curve_step"] + 1:
                f = next(r["fve_batch"] for r in curve if r["step"] == th["curve_step"])
                out.check(f"batch FVE at step {th['curve_step']}", f >= th["fve_curve_min"], {"fve_batch": f, "bar": th["fve_curve_min"]})
            if step in marks:
                r = CC.score(cfg, tr.ae, self.ev(tr.ae, side), h_ls, h_own, side.plant)[0]
                snaps.append({"step": step, "tokens": step * cfg.batch} | {k: r[k] for k in ("fve_base", "fve_ft", "dead", "d_authors", "d_alt", "specific", "spec")})
                log(f"[{side.key} snapshot] {json.dumps(jsonable({k: v for k, v in snaps[-1].items() if k != 'spec'}))}")
                if step == marks[0]:
                    f, proj = min(r["fve_base"], r["fve_ft"]), (time.time() - self.t0 + (time.time() - t_train) / max(1, step - start) * (cfg.steps - step)) / 60
                    out.check("FVE at the first snapshot", f >= th["fve_early"], {"fve": f, "bar": th["fve_early"]})
                    out.check("dead share at the first snapshot", r["dead"] <= th["dead_max"], {"dead": r["dead"], "bar": th["dead_max"]})
                    out.check("projected job minutes", proj <= th["projected_limit_min"], {"minutes": proj, "bar": th["projected_limit_min"]})
            if step % cfg.ckpt_every == 0 or step in marks or step == stop_after:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(CC.state(tr, step) | {"snaps": snaps, "curve": curve}, ck)
            if step == stop_after:
                log(f"[{side.key}] stopped after step {step} (--stop-after)")
                hard_exit(3)

        CC.train(tr, CC.batches(self.sizes, lambda i: self.chunk(i, side), cfg.batch, start), start, cfg.steps, cfg.log_every, curve, log, hook)
        ev, reads, large = self.ev(tr.ae, side), {}, {}
        for t in side.reads:
            reads[t], large[t] = CC.score(cfg, tr.ae, ev, h_ls, h_own if t == side.reads[0] else self.h_read(side, t), side.plant)
            log(f"[{side.key} read on {t}] {json.dumps(jsonable({k: v for k, v in reads[t].items() if k != 'spec'}))}")
        self.large.mkdir(parents=True, exist_ok=True)
        ae = {k: v.bfloat16() if v.numel() > 1_000_000 else v for k, v in tr.ae.state_dict().items()}
        torch.save({"ae": ae, "reads": large}, self.large / f"{side.key}.pt")
        ck.unlink(missing_ok=True)
        own = reads[side.reads[0]]
        rec = out.d["side"] = {"key": side.key, "reads": reads, "d_authors": own["d_authors"], "d_alt": own["d_alt"], "snapshots": snaps,
                               "curve": curve, "resumed_at": start, "minutes": (time.time() - self.t0) / 60,
                               "crosscoder": str((self.large / f"{side.key}.pt").relative_to(REPO))}  # fmt: skip
        out.check("FVE at the end", min(own["fve_base"], own["fve_ft"]) >= th["fve_min"], {"fve": [own["fve_base"], own["fve_ft"]], "bar": th["fve_min"]})
        out.check("dead share at the end", own["dead"] <= th["dead_max"], {"dead": own["dead"], "bar": th["dead_max"]})
        return rec

    def update(self, name: str) -> ST.DW:
        """A run-2 update (bf16 on disk) in fp32 on the GPU, its size checked against its target."""
        dW = {k: v.float().cuda() for k, v in torch.load(REPO / self.S["states"][name], weights_only=False).items()}
        size, tgt = update_size(dW, self.net.W0), self.S["targets"][name]
        lo, hi = self.th["size_band"]
        self.out.check(f"{name} update size", lo * tgt <= size <= hi * tgt, {"size": size, "target": tgt, "band": [lo, hi]})
        return dW


# ----------------------------------------------------------------------------- jobs


def prep(out: Out) -> None:
    """Each task's training units and run-2 read rows, by run 2's own loaders, into TEXTS (int32)."""
    R2 = load_experiment("realistic_tasks", "run2")
    C2, net = config_from_settings(R2.Config, json.loads(R2.PRED.read_text())["settings"]), ST.Net("cuda")
    units, reads = {}, {}
    for t in SIDES:
        r, data, _ = R2.task_data(net, C2, t)
        units[t], reads[t] = (data.parts["pool"][0] if t in LABEL else data[1]).int(), r.toks.int()
        log(f"[prep {t}] units {tuple(units[t].shape)}, read rows {tuple(reads[t].shape)}")
    assert torch.equal(units["L1"], units["L3"]) and torch.equal(reads["L1"], reads["L3"]), "L1 and L3 must share texts"
    TEXTS.parent.resolve().mkdir(parents=True, exist_ok=True)
    torch.save({"units": units, "reads": reads}, TEXTS)
    out.d |= {"units": {t: list(v.shape) for t, v in units.items()}, "reads": {t: list(v.shape) for t, v in reads.items()},
              "texts": str(TEXTS.relative_to(REPO)), "sha256": hashlib.sha256(TEXTS.read_bytes()).hexdigest()}  # fmt: skip


def planted(job: Job) -> dict[str, Any]:
    """E203's planted direction: a random unit vector (plant seed) times the median base norm at the label read position."""
    v = torch.randn(job.cfg.d, generator=torch.Generator().manual_seed(job.S["plant_seed"]))
    L = job.reads["L1"]
    h = CC.acts(job.net, L, None, job.layer).reshape(len(L), L.shape[1] - 1, -1)[:, -1]
    side = Side("planted", job.zero, ("L1",), (v / v.norm() * h.norm(dim=1).median().cpu()).cuda())
    rec = job.fit(side)
    own, th = rec["reads"]["L1"], job.th
    lead = own.get("lead", {})
    ok = (bool(lead.get("specific")) or not th["planted_lead_specific"]) and own["specific"] >= th["planted_part"]
    job.out.check("planted: found, specific and carried", ok, {"lead": lead, "part": own["specific"], "bar": th["planted_part"]}, stop=False)
    return rec | {"planted_pass": ok}


def gauss(job: Job) -> dict[str, Any]:
    src = job.update(job.S["gauss_of"])
    R = FT.random_direction({k: v.cpu() for k, v in src.items()}, job.S["gauss_draw"])
    # random_direction's fp32 norms leave the total size off by ~7e-5 (rehearsal 61482); rescale each matrix in float64
    R = {k: (r.double() * (src[k].double().norm().cpu() / r.double().norm())).float().cuda() for k, r in R.items()}
    rel = abs(update_size(R, job.net.W0) / update_size(src, job.net.W0) - 1)
    job.out.check("Gaussian size equals its source's", rel <= 1e-5, {"rel": rel})
    own = job.S["gauss_of"]
    rec = job.fit(Side(f"gauss__{own}", R, (own, *(t for t in SIDES if t != own))))
    bar = job.th["gauss_stop"]
    job.out.check(f"stop rule: Gaussian d_authors on {own}'s rows under {bar}", rec["d_authors"] < bar, {"d_authors": rec["d_authors"], "bar": bar})
    return rec


def side(job: Job, name: str, results: Path) -> dict[str, Any]:
    g = results / "run3_gauss.json"
    if not (g.is_file() and json.loads(g.read_text()).get("done")):
        raise SystemExit(f"refusing {name}: {g.name} has not passed (the Gaussian's stop rule reads first)")
    return job.fit(Side(name, job.update(name), (name,)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("prep", "planted", "gauss", *SIDES))
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E212"), "predictions_run3.json is not E212's"
    tag = "rehearsal" if a.rehearsal else "full"
    S = P["settings"][tag]
    cfg = config_from_settings(CC.Cfg, S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})
    results = RESULTS / "run3_rehearsal" if a.rehearsal else RESULTS
    out = Out(results / f"run3_{a.job}.json", a.job, P["experiment"])
    out.d |= {"tag": tag, "stop_after": a.stop_after}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as the source's code runs
    try:
        if a.job == "prep":
            prep(out)
        else:
            job = Job(P, cfg, P["thresholds"][tag], out, RESULTS / "large" / ("run3_rehearsal" if a.rehearsal else "run3"), a.stop_after)
            out.d["side"] = planted(job) if a.job == "planted" else gauss(job) if a.job == "gauss" else side(job, a.job, results)
    except Stop as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    out.d["done"] = True
    out.write()
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; wrote {out.path.relative_to(REPO)}")
    print(f"STAGE {a.job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
