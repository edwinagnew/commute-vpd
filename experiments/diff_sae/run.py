"""Cell P2.75, run 1 (E-number in results/predictions.json): the diff-SAE with latent scaling (Dumas, Minder and Nanda
2025; diffing-toolkit@617d926 configs/diffing/method/sae_difference.yaml) on the 67M, on run 1's controls. Spec
reports/plans/2026-11-23_plan_p2-75-diff-sae-spec.md (appendix ..._appendix.md); settings, bars and the gate's order
results/predictions.json (refuses unless committed unchanged). Sides and streams are E235's and E243's (`Job` below is
E243's, which is E235's), called, not copied; the SAE, its trainer and loop are dictionary_learning's, called as they
run; latent scaling is the toolkit's `closed_form_scalars` from its pinned file (commute.diff_sae).

Item keys (E235's what.site.cSEED[.kK]):
  plant.A.c0                 E212's planted direction on E212's L1 rows (stage 0a, first and alone)
  head-r1.A.c{0,1}           E243's grafted head scaled to rung 1 (the gate)
  random_d{0,1}-r1.A.c0      E243's random draws matched to rung 1 on rung 1's stream and read rows (the gate's null)
  head-r5.A.c0, head.B.c0, gauss.B.c0   descriptive controls (stage 0c); head-r1.A.c0.k33 the conditional k-33 fit
Each item runs its stages cheapest first and writes results/<tag>/items/<key>.json as each ends (normaliser, training,
latent scaling and the reading, the KL patches); the SAE and per-latent arrays go to results/large/<tag>/<key>.pt.

    PRED_SHA=<sha> python experiments/diff_sae/run.py item plant.A.c0 2>&1 | tee -a experiments/diff_sae/results/run1/pod0.log
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")  # trainSAE's and the scaler's progress bars, 100k lines a fit

import torch
from dictionary_learning import BatchTopKSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
from dictionary_learning.training import trainSAE
from torch import Tensor

from commute import crosscoder as CC
from commute import diff_sae as DS
from commute import graft as G
from commute.data import cached_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    manifest_only,
    note_inputs,
    require_filed,
)
from commute.open_pair import jsonable

RUN: Any = load_experiment("crosscoder_proper", "run")  # E235: Item, Side, Out, Stop, exact, its Job
RUN2: Any = load_experiment("crosscoder_proper", "run2")  # E243: Scaled, its Job (the rungs and random draws)
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
TEXTS = EXPERIMENTS / "realistic_tasks" / "results" / "large" / "run3" / "texts.pt"  # E212's units and read rows
OUT: dict[str, Path] = {}
log = RUN.log


@dataclass(frozen=True)
class Config:  # settings.run
    plant_seed: int  # E212's
    plant_task: str  # E212's L1
    toolkit_root: str  # the pinned latent_scaling/ folder
    toolkit_sha256: dict[str, str]
    rung1: tuple[str, ...]
    gauss: tuple[str, ...]


class ValSet:
    """trainSAE's validation data: fixed (val_batch, d) batches of D, re-iterable, with a length (dl training.py 141)."""

    def __init__(self, batches: list[Tensor]) -> None:
        self.b = batches

    def __len__(self) -> int:
        return len(self.b)

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.b)


class Job(RUN2.Job):
    def __init__(self, P1: dict[str, Any], P2: dict[str, Any], P: dict[str, Any], size: str) -> None:
        S, th = P["settings"], P["thresholds"][size]
        self.sae, self.c = config_from_settings(DS.Cfg, S["sae"][size]), settings(Config, S["run"])
        e = P1["settings"]["cc"][size]
        cc = config_from_settings(CC.Cfg, e | {"shared": tuple(e["shared"]), "snapshots": tuple(e["snapshots"])})
        sz = self.sae
        cc = dataclasses.replace(cc, tokens=sz.stream_tokens, epochs=sz.epochs, chunk_rows=sz.chunk_rows, pile_start=sz.pile_start,
                                 pool_start=sz.pool_start, pool_rows=sz.pool_rows, repeat_cap=sz.repeat_cap)  # fmt: skip
        full = P1["settings"]["cc"]["full"]
        assert all(getattr(cc, k) == full[k] for k in ("tokens", "epochs", "chunk_rows", "pile_start", "pool_start", "pool_rows", "repeat_cap")), "not E235's stream"
        super().__init__(settings(RUN.Config, P1["settings"]["run"]), cc, th, RUN2.settings(P2))
        self.th, self.capture = th, None
        self.P = DS.pinned(Path(self.c.toolkit_root).expanduser(), self.c.toolkit_sha256)
        note_inputs(*(Path(self.c.toolkit_root).expanduser() / f for f in DS.PINNED))
        self.val_rows = cached_rows("train", sz.val_start, sz.val_rows)
        self.kl_rows = cached_rows("val", 0, sz.kl_rows)

    # ---- sides ------------------------------------------------------------------------------------------------------

    def side(self, it: Any) -> Any:
        if it.what == "plant":
            return self.plant_side(it)
        if it.what == "head":  # E235's head at site B at scale 1, wrapped so that a zero update can be applied
            s0 = RUN.Job.side(self, it)
            wo = G.out_param(s0.ft.net.h[self.cfg.graft_layers[self.cfg.heads[it.site]]].attn.graft)
            return RUN2.Scaled(**vars(s0), rung=-1, wo=(wo, wo.detach().clone()))
        if "-r" not in it.what:  # E235's Gaussian at site B
            return RUN.Job.side(self, it)
        s = super().side(it)
        if it.what.startswith("random_d"):  # the null trains, scales and reads on its rung's stream (spec 5, stage 0b)
            h = super().side(RUN.Item(f"head{it.what[it.what.index('-r'):]}", it.site, 0))
            s.source, s.read = h.source, h.read
            assert s.source[0] == "graft" and torch.equal(s.source[1], h.source[1]) and torch.equal(s.read, h.read)
        return s

    def plant_side(self, it: Any) -> Any:
        """E212's plant: a random unit vector (its seed) times the median base norm at the L1 read position, added at
        that position of each L1 unit (E212 `planted`, realistic_tasks/run3.py 230-235)."""
        t = torch.load(TEXTS, weights_only=False)
        units, reads = t["units"][self.c.plant_task].long(), t["reads"][self.c.plant_task].long()
        site = self.cfg.sites[it.site]
        with RUN.exact():
            h = CC.site_acts(self.net, reads, None, site).reshape(len(reads), reads.shape[1] - 1, -1)[:, -1]
        v = torch.randn(self.sae.d, generator=torch.Generator().manual_seed(self.c.plant_seed))
        vec = (v / v.norm() * h.norm(dim=1).median().cpu()).cuda()
        return RUN.Side(it, site, self.net, self.zero, ("fit", units), reads, (vec, units.shape[1] - 1))

    def parents(self, s: Any) -> list[tuple[int, int]]:
        """Train-split parent rows of the side's stream, as intervals: the Pile half, the task-window source, fill."""
        tx, n_pile = self.text(s), self.n_chunks * self.cc.chunk_rows
        kind = s.source[0]
        src = {"kv": (self.cc.pool_start, self.cc.pool_start + self.cc.pool_rows), "graft": (0, self.cfg.graft_rows),
               "fit": (0, RUN.TASK_ROWS) if s.item.what != "plant" else None}[kind]  # fmt: skip
        out = [(self.cc.pile_start, self.cc.pile_start + n_pile)] + ([src] if src else [])
        return out + ([(self.cfg.fill_start, self.cfg.fill_start + self.n_chunks * tx["fill_rows"])] if tx["fill_rows"] else [])

    def zero_change(self, s: Any) -> Any:
        """The side with its change set to zero: no plant, a zero dW, or a scaled side at 0 (restored by `before`)."""
        if s.plant is not None:
            return dataclasses.replace(s, plant=None)
        if isinstance(s, RUN2.Scaled):
            s.apply(0.0)
            return s
        return dataclasses.replace(s, dW=self.zero)

    def before(self, s: Any, out: Any) -> None:
        """Before step 0: the validation rows outside the stream's, D exactly 0 under a zero update, then E243's or
        E235's own checks (the rung match, the Gaussian match, a graft's write)."""
        lo, hi = self.sae.val_start, self.sae.val_start + self.sae.val_rows
        par = self.parents(s)
        out.check("validation rows share no parent row with the stream", all(b <= lo or a >= hi for a, b in par), {"val": [lo, hi], "stream": par})
        z = self.pair(s.read[:8], self.zero_change(s), True)
        diff = float((z[:, 1] - z[:, 0]).abs().max())
        out.check("D exactly 0 under a zero update", diff == 0.0, {"max_abs": diff})
        if isinstance(s, RUN2.Scaled) and s.rung >= 0:
            super().before(s, out)  # E243's: the zero path, the match to the rung, the write
        elif s.item.what != "plant":
            if isinstance(s, RUN2.Scaled):
                s.apply(1.0)  # E235's head at its own scale
            RUN.Job.before(self, s, out)
        out.d["energy"] = self.energy(s, s.dW)

    def pair(self, toks: Tensor, s: Any, task: bool) -> Tensor:
        x = super().pair(toks, s, task)
        if self.capture is not None:  # the dataset a piece belongs to: the Pile half, or the task half (windows, fill)
            pile = toks.untyped_storage().data_ptr() == self.pile.untyped_storage().data_ptr()
            self.capture.append(("pile" if pile else "task", x))
        return x

    def pieces(self, make: Any, i: int) -> dict[str, Tensor]:
        """Chunk i of epoch 0 in dataset order: its Pile-half and task-half activations, unshuffled."""
        self.capture = []
        try:
            make(i)
            got = {k: torch.cat([x for d, x in self.capture if d == k]) for k in ("pile", "task")}
        finally:
            self.capture = None
        return got

    # ---- stages -----------------------------------------------------------------------------------------------------

    def normaliser(self, make: Any) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """One pass over epoch 0: norm_tokens / 2 tokens of D drawn uniformly from each dataset (dt training.py 165-190),
        their Welford mean and biased std. Datasets are epoch 0's chunks in stream order."""
        sz, idx, got, L = self.sae, None, {"pile": [], "task": []}, {}
        for i in range(self.n_chunks):
            p = self.pieces(make, i)
            if idx is None:
                L = {k: len(v) for k, v in p.items()}
                idx = dict(zip(L, DS.norm_indices([self.n_chunks * n for n in L.values()], sz.norm_tokens, sz.norm_seed), strict=True))
            for k, x in p.items():
                got[k].append(DS.take(idx[k], i * L[k], x[:, 1] - x[:, 0]))
        samples = [torch.cat(got[k]) for k in ("pile", "task")]
        mean, std = DS.welford(samples, sz.norm_batch)
        return mean, std, {"chunk_tokens": L, "lengths": {k: self.n_chunks * n for k, n in L.items()}, "sample": [len(x) for x in samples],
                           "scale": DS.scale_of(std, sz.target_rms), "mean_norm": float(mean.norm())}  # fmt: skip

    def scaler_rows(self, make: Any, lengths: dict[str, int]) -> Iterator[Tensor]:
        """The scaler's rows: per-dataset prefixes in proportion to the datasets' sizes (dt activations.py 212-245),
        no shuffling, in batches of ls_batch (the pinned caller's 16,384 // 8). Ours: the two prefixes are read in
        one pass, chunk by chunk, so batches of the two datasets interleave (the sums are order-free)."""
        want = dict(zip(lengths, DS.sample_counts(list(lengths.values()), self.sae.ls_tokens), strict=True))
        buf: dict[str, list[Tensor]] = {k: [] for k in want}
        for i in range(self.n_chunks):
            if not any(want.values()):
                break
            for k, x in self.pieces(make, i).items():
                x = x[: want[k]]
                want[k] -= len(x)
                buf[k].append(x)
                b = torch.cat(buf[k])
                n = len(b) // self.sae.ls_batch * self.sae.ls_batch
                yield from b[:n].split(self.sae.ls_batch)
                buf[k] = [b[n:]]
        assert not any(want.values()), f"the stream ran out before the scaler's prefixes: {want}"
        for v in buf.values():
            if len(b := torch.cat(v)):
                yield b

    def scale(self, sae: Any, make: Any, lengths: dict[str, int], out: Any) -> dict[str, dict[str, Any]]:
        """Both forms of latent scaling, every latent, in one pass: the pinned fits (base and fine-tune targets per form)
        and our float64 sums on the same batches; checks: encoder inputs, float32 parity, the numerator identity."""
        fns = [*DS.pinned_fits(self.P, sae, "lit"), *DS.pinned_fits(self.P, sae, "D"),
               partial(DS.sums64, sae, self.P, "lit"), partial(DS.sums64, sae, self.P, "D")]  # fmt: skip
        with RUN.exact():
            r = DS.fan_out(self.scaler_rows(make, lengths), fns)
        res, bar = {}, self.th["identity_rel"]
        for j, form in enumerate(("lit", "D")):
            (bM, cnt, BM, C, _), (bF, _, BF, _, _), s64 = r[2 * j], r[2 * j + 1], r[4 + j]
            on = s64["C"] > 0
            rel = lambda a, b, on=on: float(((a.double().cpu() - b).abs() / b.abs().clamp_min(1e-300))[on].max()) if bool(on.any()) else 0.0
            ident = ((BF.double().cpu() - BM.double().cpu()) - s64["B_sD"]).abs() / (s64["B_M"].abs() + s64["B_F"].abs()).clamp_min(1e-300)
            chk = {"parity_B_M": rel(BM, s64["B_M"]), "parity_B_F": rel(BF, s64["B_F"]), "parity_C": rel(C, s64["C"]),
                   "identity": float(ident[on].max()) if bool(on.any()) else 0.0, "count_equal": bool(torch.equal(cnt.double().cpu(), s64["count"]))}  # fmt: skip
            out.d.setdefault("scaler_checks", {})[form] = chk | s64["checks"]
            out.check(f"{form}: encoder inputs and targets", all(s64["checks"].values()), s64["checks"])
            if bar is not None:
                out.check(f"{form}: numerator identity", chk["identity"] <= bar, {"identity": chk["identity"], "bar": bar})
            res[form] = {"beta_M": bM.double().cpu(), "beta_F": bF.double().cpu(), "count": s64["count"]}
        return res

    def train(self, s: Any, mean: Tensor, std: Tensor, make: Any, sizes: list[int], seed: int, k: int, out: Any) -> Any:
        """dl's trainSAE with BatchTopKTrainer at the toolkit's arguments (dt training.py 885-905, 1032-1049); our hook
        after each step: non-finite parameters, the pooled FVE at each validation and its stop, the rehearsal's dead set."""
        sz, th, key = self.sae, self.th, s.item.key
        with RUN.exact():
            val = [x[:, 1] - x[:, 0] for x in (self.pair(r, s, False) for r in self.val_rows.split(64))]
        D = torch.cat(val)[: sz.val_tokens]
        vs = ValSet(list(D.split(sz.val_batch)))
        cfg = {"trainer": BatchTopKTrainer, "dict_class": BatchTopKSAE, "activation_dim": sz.d, "dict_size": sz.latents, "lr": sz.lr,
               "steps": sz.steps, "device": "cuda", "warmup_steps": sz.warmup, "layer": int(s.site[-1]), "lm_name": "commute-67m-ft-base",
               "wandb_name": key, "k": k, "activation_mean": mean, "activation_std": std, "encoder_init_norm": sz.init_norm,
               "target_rms": sz.target_rms, "auxk_alpha": sz.aux_alpha, "threshold_start_step": sz.threshold_start, "seed": seed}  # fmt: skip
        curve, keep, t0 = out.d.setdefault("curve", []), {}, time.time()

        def hook(tr: Any, step: int) -> None:
            keep["tr"] = tr
            if step % sz.log_steps == 0 and not all(bool(torch.isfinite(p).all()) for p in tr.ae.parameters()):
                out.check("finite loss (parameters finite)", False, {"step": step})
            if step > 0 and step % sz.validate_every == 0:
                f = DS.pooled_fve(tr.ae, vs)
                dead = float((tr.num_tokens_since_fired >= tr.dead_feature_threshold).float().mean())
                curve.append({"step": step, "fve_pooled": f, "dead": dead, "theta": float(tr.ae.threshold), "ms_per_step": 1000 * (time.time() - t0) / step})
                log(f"[{key}] step {step}: {json.dumps(curve[-1])}")
                b = (th["fve_stop"] or {}).get(key)
                if step == sz.validate_every and b is not None and f is not None:
                    out.check(f"pooled FVE at step {step}", f >= b, {"fve": f, "bar": b})
            if sz.dead_fixture_step and step == sz.dead_fixture_step:
                tr.num_tokens_since_fired[: sz.k] = tr.dead_feature_threshold  # rehearsal: a known dead set
            if sz.dead_fixture_step and step == sz.dead_fixture_step + 1:
                out.check("AuxK on for the known dead set", tr.pre_norm_auxk_loss != -1 and tr.dead_features > 0,
                          {"dead_features": tr.dead_features})  # fmt: skip

        data = (x[:, 1] - x[:, 0] for x in CC.batches(sizes, make, sz.batch, 0))
        save = OUT["large"] / key
        torch.backends.cuda.matmul.allow_tf32 = True  # their TF32 for the SAE; `exact` turns it off around forwards
        try:
            model: Any = trainSAE(data, cfg, steps=sz.steps, save_steps=sz.validate_every, save_dir=str(save), log_steps=sz.log_steps,
                             validate_every_n_steps=sz.validate_every, validation_data=vs, end_of_step_logging_fn=hook,
                             run_wandb_finish=False)  # fmt: skip
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        tr = keep["tr"]
        assert tr.dead_feature_threshold == sz.dead_tokens, tr.dead_feature_threshold
        fresh: Any = BatchTopKSAE(sz.d, sz.latents, k, activation_mean=mean, activation_std=std, target_rms=sz.target_rms).to("cuda", torch.float32)
        fresh.load_state_dict(torch.load(save / "model_final.pt", weights_only=True))
        same = bool(torch.equal(fresh.encode(vs.b[0]), model.encode(vs.b[0])))
        out.check("saved SAE reloads to identical encodings", same, {})
        dead = float((tr.num_tokens_since_fired >= tr.dead_feature_threshold).float().mean())
        out.d["train"] = {"steps": sz.steps, "dead_end": dead, "minutes": (time.time() - t0) / 60, "fve_end": DS.pooled_fve(model, vs)}
        for p in save.glob("checkpoint_*.pt"):
            p.unlink()
        out.check("dead share at the end", dead <= th["dead_max"], {"dead": dead, "bar": th["dead_max"]})
        return model

    @torch.no_grad()
    def kl(self, s: Any, sae: Any, masks: dict[str, Tensor]) -> dict[str, Any]:
        """The KL-gap patches (spec 3.7) on the read windows and on Pile validation rows: F's forward (its plant by a hook
        that does not stop the forward) against the same forward with the site replaced at positions 1.. by each arm."""
        out = {}
        for name, rows, task, n in (("read", s.read, True, 256), ("pile", self.kl_rows, False, self.sae.kl_batch)):
            tot: dict[str, float] = {}
            last: dict[str, float] = {}
            for b in rows.split(n):
                b = b.cuda()
                with RUN.exact():
                    h = self.pair(b, s, task)
                    fills = DS.patches(sae, h, masks)
                    plant = s.plant if task else None
                    with s.ft.at(s.dW):
                        ref = s.ft.run(b, edit={s.site: put_plant(plant)} if plant else None)[0]
                        for k, v in fills.items():
                            lg = s.ft.run(b, edit={s.site: put_fill(v.view(len(b), b.shape[1] - 1, -1))})[0]
                            a, z = DS.kl_sums(ref, lg)
                            tot[k], last[k] = tot.get(k, 0.0) + float(a.sum()), last.get(k, 0.0) + float(z.sum())
            m = rows.shape[0]
            kl_all = {k: v / (m * (rows.shape[1] - 1)) for k, v in tot.items()}
            kl_last = {k: v / m for k, v in last.items()}
            out[name] = {"kl_all": kl_all, "kl_last": kl_last}
        return out

    # ---- one item ---------------------------------------------------------------------------------------------------

    def run_item(self, it: Any) -> dict[str, Any]:
        sz, th, t_item = self.sae, self.th, time.time()
        k = it.k or sz.k
        out = RUN.Out(OUT["res"] / "items" / f"{it.key}.json", {"experiment": EXPERIMENT, "key": it.key, "card": torch.cuda.get_device_name()})
        s = self.side(it)
        if manifest_only():  # INPUT_MANIFEST_ONLY=1 dry run: the side's loads are listed; no forward
            log(f"[{it.key}] manifest only: side built, compute skipped")
            return {}
        self.before(s, out)
        sizes, make, tx = self.stream(s)
        out.d["stream"] = tx
        out.check("stream holds every step", tx["tokens"] // sz.batch >= sz.steps, {"tokens": tx["tokens"], "need": sz.steps * sz.batch})
        with RUN.exact():
            mean, std, out.d["normaliser"] = self.normaliser(make)
            h_read = self.pair(s.read, s, True)
        D_read = h_read[:, 1] - h_read[:, 0]
        K = DS.capacity(mean, D_read)
        out.d["K"], out.d["bar"] = K, DS.bar(K)
        out.write()
        log(f"[{it.key}] normaliser {json.dumps(out.d['normaliser'])}; K {K}")
        sae = self.train(s, mean, std, make, sizes, it.seed, k, out)
        out.write()
        r = DS.parts(sae, D_read)
        if r is None:
            out.d["read"] = "not read: no change on the read set"
            return self.finish(out, t_item)
        alive = r["fired"] >= sz.alive_min
        scal = self.scale(sae, make, out.d["normaliser"]["lengths"], out)
        reads = {form: DS.scaler_read(scal[form], alive, r["s"], sz) for form in scal}
        out.d["read"] = {k2: v for k2, v in r.items() if k2 not in ("s", "fired")} | {f: {k2: v for k2, v in x.items() if k2 != "nu"} for f, x in reads.items()}
        torch.save({"sae": sae.state_dict(), "s": r["s"].float(), "fired": r["fired"], "alive": alive,
                    "betas": {f: {k2: v.float() for k2, v in x.items()} for f, x in scal.items()}}, OUT["large"] / f"{it.key}.pt")  # fmt: skip
        out.write()
        log(f"[{it.key} read] {json.dumps(jsonable(out.d['read']))}")
        masks = {f"S_{f}_{c}": DS.specific(reads[f]["nu"], scal[f]["beta_M"], alive, c, False) for f in reads for c in (sz.nu_cut, sz.nu_wide)}
        out.d["kl"] = kl = self.kl(s, sae, masks)
        for name, v in kl.items():
            hF, gap, empty = v["kl_all"]["h_F"], v["kl_all"]["gap"], v["kl_all"]["empty"]
            if th["kl_hF_floor"] is not None:
                out.check(f"KL path: h_F patch at its floor ({name})", hF <= th["kl_hF_floor"], {"kl": hF, "bar": th["kl_hF_floor"]})
            out.check(f"KL path: empty subset is the gap exactly ({name})", empty == gap, {"empty": empty, "gap": gap})
            floor = th["kl_hF_floor"] * th["kl_gap_mult"] if None not in (th["kl_hF_floor"], th["kl_gap_mult"]) else None
            v["recovered_all"], v["recovered_last"] = DS.recovered(v["kl_all"], floor), DS.recovered(v["kl_last"], floor)
        log(f"[{it.key} kl] {json.dumps(jsonable(kl))}")
        return self.finish(out, t_item)

    def finish(self, out: Any, t_item: float) -> dict[str, Any]:
        out.d |= {"done": True, "minutes": (time.time() - t_item) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        out.write()
        log(f"ITEM {out.d['key']} DONE in {out.d['minutes']:.1f} min")
        return out.d


def put_plant(plant: tuple[Tensor, int]) -> Any:
    def f(x: Tensor) -> Tensor:
        x = x.clone()
        x[:, plant[1]] += plant[0].to(x.device, x.dtype)
        return x

    return f


def put_fill(v: Tensor) -> Any:
    def f(x: Tensor) -> Tensor:
        x = x.clone()
        x[:, 1:] = v.to(x.dtype)
        return x

    return f


def settings[C](cls: type[C], s: dict[str, Any]) -> C:
    return config_from_settings(cls, {k: tuple(v) if isinstance(v, list) else v for k, v in s.items()})


EXPERIMENT = json.loads(PRED.read_text())["experiment"]


def load(rehearse: bool, filed: bool = True) -> Job:
    """The job at one size; `filed` (every GPU job) refuses unless predictions.json is committed with its E-number."""
    P = json.loads(PRED.read_text())
    if filed:
        require_filed(PRED)
        assert P["experiment"][1:].isdigit(), "claim the E-number (fleet/ecode) and file it in results/predictions.json"
    size = "rehearsal" if rehearse else "full"
    if filed and not rehearse:
        unset = [k for k, v in P["thresholds"]["full"].items() if v is None]
        assert not unset, f"thresholds the rehearsal sets are not filed yet: {unset}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    return Job(json.loads(RUN.PRED.read_text()), json.loads(RUN2.PRED.read_text()), P, size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("item",))
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or ("rehearse" if a.rehearse else "run1")
    OUT.update(res=HERE / "results" / tag, large=HERE / "results" / "large" / tag)
    job = load(a.rehearse, filed=not manifest_only())
    log(f"[{EXPERIMENT}] {a.keys} tag {tag}; {job.sae}")
    try:
        for key in a.keys:
            p = OUT["res"] / "items" / f"{key}.json"
            if p.is_file() and json.loads(p.read_text()).get("done"):
                log(f"[item] {key} done before, skipped")
                continue
            job.run_item(RUN.Item.parse(key))
            torch.cuda.empty_cache()
    except RUN.Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"FINISHED {' '.join(a.keys)} in {(time.time() - job.t0) / 60:.1f} min; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
