"""E298's states and its three featurizations behind one interface, so run.py reads every item the same way.

States (`Worlds.make`), ids '<block>/<name>':
    built/<edit>        E247's seven built edits, F, M, D and read positions as E268 builds them (kstar_needed.world:
                        E257's World with E242's matched scale of the KL-matched random); built/do_nothing is F = M
    built/rand_<edit>   new: E290's random rule on that edit, a Gaussian per tensor the edit changed at that tensor's
                        Frobenius norm of F - M (needed_neurons.matched_random), seed built_random_seed + the edit's
                        index in built_random_of, read on the edit's own D and positions
    built/head_<e>      E280's head-(2, 4) edits (v2, o2, q2: one slice of one layer-2 matrix times 2), its code, its D
                        (E257's e62/copy_s0 windows, last position); built/rand_head_<e> their randoms as above
    trained/<t>_s<s>    E147's step-30 full fine-tune, unscaled, on E237's D at the last position (E290's World, item
                        full/<t>_s<s>); trained/<t>_s<s>_rand its E290 random (seed 161000 + E290's item index)

Readers (`reader(feat)`), each: state(F, M, toks, world) -> what the reads need; writes -> num, den, dd, changed, leftover
(one `stand_in.write_change` pass, every position of D in F's own run); cache -> F's run (.top, .margin); scan -> the
swap test (`needed.Scan`); identity -> x' = x on the first windows. VPD: `kstar_vpd.featurize` and `needed.scan`
(E268); neurons: `needed_neurons` (E277, E229's p97 bounds); SAE: `sae_feat` sites (E245's SAEs) and `needed_sae.scan`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import needed as IX
from commute import needed_neurons as NN
from commute import needed_sae as NS
from commute import old_tasks as OT
from commute import rho
from commute import rule_retention as R
from commute import sae_feat as SF
from commute import stand_in as SI
from commute.jobs import load_experiment

KN: Any = load_experiment("kstar_needed", "run")  # E268: world() (the built states), E226 (split_tol), log/dump/stop
NNR: Any = load_experiment("needed_neurons", "run")  # E277: matched_random, load_e229, E229 (feat_of, load_pct)
IA: Any = load_experiment("ia3_control", "run")  # E290: World (the trained states and their randoms), load_config
HE: Any = load_experiment("head_edit", "run")  # E280: edited, data, load_config (the head-(2, 4) built edits)


class Worlds:
    """The built world (E268's) and the trained world (E290's), each made when first asked for."""

    def __init__(self, cfg: Any, vpd: bool) -> None:
        self.cfg, self.vpd = cfg, vpd
        self._built: Any = None
        self._trained: Any = None

    @property
    def built(self) -> Any:
        if self._built is None:
            self._built = KN.world()  # its uv, delta, bases: the VPD featurization's inputs
        return self._built

    @property
    def trained(self) -> Any:
        if self._trained is None:
            w = IA.World(IA.load_config(), "fig2")
            if self.vpd:
                from commute import vpd

                w.uv, w.delta = vpd.component_weights(w.net.model), vpd.weight_deltas(w.net.model)
                w.bases = KV.bases(w.uv, KN.E226.load_config()[0].split_tol)
            self._trained = w
        return self._trained

    def of(self, item: str) -> Any:
        return self.built if item.startswith("built/") else self.trained

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        block, name = item.split("/")
        if block == "trained":
            return self.trained.make(f"full/{name}")
        assert block == "built", item
        w = self.built
        if not name.startswith("rand_"):
            return self.head(name) if name.startswith("head_") else w.make(name)
        base = name.removeprefix("rand_")
        F, M, toks, at, _ = self.head(base) if base.startswith("head_") else w.make(base)
        dw, rec = NNR.matched_random(F.net, M.net, self.cfg.built_random_seed + self.cfg.built_random_of.index(base))
        del F
        return R.Model(OT.with_delta(w.kl.net, dw).net), M, toks, at, rec | {"random_of": base}

    def head(self, name: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        """E280's head-(2, 4) edit head_<v2|o2|q2>, built by its code on the built world's M, on its D (E257's
        e62/copy_s0 windows, last position), as experiments/head_edit/run.py main reads it."""
        cfg, w = HE.load_config(), self.built
        toks, at = HE.data(w, cfg)
        return HE.edited(w, cfg, name.removeprefix("head_")), w.kl.M, toks, at, {"edit_of": "E280"}


def batches(toks: Tensor, b: int) -> list[Tensor]:
    return list(toks.split(b))


def _changed(n: int, idx: set[int]) -> Tensor:
    ch = torch.zeros(n, dtype=torch.bool)
    ch[sorted(idx)] = True
    return ch


class VPD:
    name = "vpd"

    def state(self, F: R.Model, M: R.Model, toks: Tensor, w: Any) -> KV.Feat:
        diffs = KV.undecomposed_differences(M, F, sorted(w.uv))
        assert not diffs, f"parameters outside the decomposition differ: {diffs[:4]}"
        return KV.featurize(M, F, w.uv, w.delta, w.bases)

    def writes(self, st: KV.Feat, F: R.Model, toks: Tensor, b: int) -> tuple[Tensor, ...]:
        num, den, dd = SI.write_change(F.net, SI.vpd_sites(st), st.n, batches(toks, b))
        return num, den, dd, _changed(st.n, st.changed), _changed(st.n, set(st.rem))

    def cache(self, st: KV.Feat, F: R.Model, toks: Tensor, at: slice, b: int) -> Any:
        return IX.cache(F, st, toks, at, b)

    def scan(self, st: KV.Feat, F: R.Model, run: Any, at: slice, ex: Tensor, pt: Tensor, cfg: Any, log: Any) -> IX.Scan:
        return IX.scan(F, st, run, at, ex, pt, IX.rows_per_forward(cfg.rows, cfg.logit_rows, run.top.shape[1]), log)

    def identity(self, st: KV.Feat, F: R.Model, run: Any, at: slice, ex: Tensor, cfg: Any, log: Any) -> IX.Scan:
        """E268's control: ident_vars subcomponents and the remainder of every matrix."""
        only = {j: torch.cat([torch.arange(min(cfg.ident_vars, s.rem - s.off)), torch.tensor([s.rem - s.off])]) for j, s in enumerate(st.sites)}
        return IX.scan(F, st, run, at, ex, ex, IX.rows_per_forward(cfg.rows, cfg.logit_rows, run.top.shape[1]), log, only)


@dataclass
class NeuronState:
    p: rho.Pair
    ch: Tensor
    sites: list[Any]


class Neurons:
    name = "neurons"

    def __init__(self, cfg: Any) -> None:
        c229 = NNR.load_e229()
        self.feat = NNR.E229.feat_of(c229, NNR.E229.load_pct(c229), cfg.band)

    def state(self, F: R.Model, M: R.Model, toks: Tensor, w: Any) -> NeuronState:
        p = rho.Pair(F, M)
        return NeuronState(p, NN.variables(p)[1], NN.sites(p, self.feat, toks.shape[1]))

    def writes(self, st: NeuronState, F: R.Model, toks: Tensor, b: int) -> tuple[Tensor, ...]:
        n = len(st.ch)
        num, den, dd = SI.write_change(F.net, st.sites, n, batches(toks, b))
        return num, den, dd, st.ch, torch.arange(n) % NN.PER_LAYER == NN.H + NN.NEUR

    def cache(self, st: NeuronState, F: R.Model, toks: Tensor, at: slice, b: int) -> Any:
        return NN.cache(F.net, toks, at, b)

    def scan(self, st: NeuronState, F: R.Model, run: Any, at: slice, ex: Tensor, pt: Tensor, cfg: Any, log: Any) -> IX.Scan:
        return NN.scan(F.net, self.feat, run, at, ex, pt, IX.rows_per_forward(cfg.rows, cfg.logit_rows, run.top.shape[1]), log)

    def identity(self, st: NeuronState, F: R.Model, run: Any, at: slice, ex: Tensor, cfg: Any, log: Any) -> IX.Scan:
        """E277's control: every variable."""
        return self.scan(st, F, run, at, ex, ex, cfg, log)


class SAE:
    name = "sae"

    def __init__(self, cfg: Any) -> None:
        e245 = load_experiment("sae_r_a", "run")
        self.saes = e245.load_saes(e245.load_config("scored")[0])  # E234's base-only SAEs as E245 and E263 load them
        self.cfg = cfg

    def state(self, F: R.Model, M: R.Model, toks: Tensor, w: Any) -> list[SF.SAESite]:
        for N in (F, M):
            N.net.requires_grad_(False)
        return SF.sites(F, M, self.saes, toks.shape[1])

    def writes(self, st: list[SF.SAESite], F: R.Model, toks: Tensor, b: int) -> tuple[Tensor, ...]:
        n = SF.n_vars(st)
        duck: list[Any] = list(st)  # stand_in.write_change is typed for VPDSite; SAESite has its members
        num, den, dd = SI.write_change(F.net, duck, n, batches(toks, b))
        return num, den, dd, _changed(n, SF.changed_vars(st)), _changed(n, {s.rem for s in st})

    def cache(self, st: list[SF.SAESite], F: R.Model, toks: Tensor, at: slice, b: int) -> Any:
        return NS.cache(F, st, toks, at, b)

    def _rows(self, run: Any) -> int:
        return IX.rows_per_forward(self.cfg.rows, self.cfg.logit_rows, run.top.shape[1])

    def scan(self, st: list[SF.SAESite], F: R.Model, run: Any, at: slice, ex: Tensor, pt: Tensor, cfg: Any, log: Any) -> IX.Scan:
        return NS.scan(F, st, run, at, ex, pt, self._rows(run), cfg.sae_err_rows, cfg.sae_chunk, log)

    def identity(self, st: list[SF.SAESite], F: R.Model, run: Any, at: slice, ex: Tensor, cfg: Any, log: Any) -> IX.Scan:
        """As E268's: per site ident_vars latents active on the windows tested, and the error term; none skipped."""
        only = {}
        for j, s in enumerate(st):
            sw = NS.Swap(s, run.outs[j])
            only[j] = torch.cat([sw.active(ex.to(run.top.device), cfg.ident_vars, cfg.sae_chunk).cpu(), torch.tensor([sw.W])])
        return NS.scan(F, st, run, at, ex, ex, self._rows(run), cfg.sae_err_rows, cfg.sae_chunk, log, only)


def reader(feat: str, cfg: Any) -> Any:
    return {"vpd": VPD, "neurons": lambda: Neurons(cfg), "sae": lambda: SAE(cfg)}[feat]()
