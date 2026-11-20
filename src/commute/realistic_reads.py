"""Every scorecard method on one saved update of one task, for P2.38 run 2 (E211; plan
reports/plans/2026-11-18_plan_p2-38-run2.md). Each reading is an existing shared step, called on a `Read` (a task's read
rows, its target per row, the read position and steering's fitting windows); nothing here is a new measurement.

Units: "lo", the log-odds of each row's target over the whole vocabulary (the share's unit); for a label task also
"contrast", the logit of the row's label minus the other label's (E140's unit, scored for the label pair by
decisions.md 2026-11-18 "Label tasks scored on the two-label contrast"). Every share is a numerator's mean over the
gain's mean in the same unit, nan when that gain is under MIN_GAIN (1).
  gain       T1 - T0, the fp32 target model (the neuron reader's T cells)
  prior      T0: W0's mean on the read rows, per unit (no update read)
  share      (L01 - L00) / (T1 - T0), released decomposition, do-nothing form (`masked_reads.share_cells_by`, its own T)
  untrained  the same on W0's SVD and its rotation (`untrained.do_nothing_cells`, E136's k_m), fp32
  neuron     (N01 - N00) / gain, each block's firing kept on its TopK neurons, k 64 and k matched (E143's)
  steering   pooled fair steering (`fair_steering`, E143's): vector from the fitting windows, (layer, strength) picked on
             the first n_select of them in the scored unit, its rise on the read rows over the gain; random draws beside
  k_half     E98's count of neurons carrying half the r_half numerator (`neuron_r`), per unit
  erank      `update_spectrum.state` of the update; size `realistic.update_size`
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import fair_steering as FS
from commute import masked_reads as MR
from commute import neuron_r as NR
from commute import steering as ST
from commute import untrained as U
from commute import update_spectrum as US
from commute.data import LAST, NEXT, PAIR
from commute.local_faith import MIN_GAIN
from commute.realistic import update_size

DOWN = tuple(f"h.{l}.mlp.down_proj" for l in FS.LAYERS)


@dataclass
class Read:
    """One task's rows. toks is what the model sees: an LM window holds its own target at pos + 1 (data.NEXT), a label
    window ends at pos and its label rides apart in tgt (data.LAST). fit / fit_tgt: steering's fitting windows, same layout."""

    name: str
    toks: Tensor
    tgt: Tensor
    pos: int
    fit: Tensor
    fit_tgt: Tensor
    ab: tuple[int, int] | None = None  # the two label tokens of a label task; None for an LM task

    def __post_init__(self) -> None:
        if self.ab is None:
            assert torch.equal(self.tgt, self.toks[:, self.pos + 1]) and torch.equal(self.fit_tgt, self.fit[:, self.pos + 1])
        else:
            assert self.pos == self.toks.shape[1] - 1 and set(self.tgt.unique().tolist()) <= set(self.ab)

    @property
    def scored(self) -> str:
        return "lo" if self.ab is None else "contrast"

    def fns(self) -> dict[str, ST.Readout]:
        return {"lo": ST.logodds_at} | ({} if self.ab is None else {"contrast": ST.contrast_at(self.ab)})

    def units_of(self, tgt: Tensor) -> MR.UnitsOf:
        """Logits at pos (B, V) of the rows `rows` -> unit -> (B,) float64 CPU."""
        return lambda lg, rows: {u: f(lg[:, None], 0, tgt[rows]) for u, f in self.fns().items()}

    def windows(self) -> tuple[Tensor, int | Tensor, int | None]:
        """(windows, target, ci_len) as `untrained.do_nothing_cells` reads them."""
        if self.ab is None:
            return self.toks, NEXT, None
        return torch.cat([self.toks, self.tgt[:, None]], 1), LAST, self.toks.shape[1]

    def single(self) -> dict[str, ST.Readout]:
        """The units read against one target token per row: E98's k_half reads them (it also reads log P of that token)."""
        return self.fns()


def pair_at(logits: Tensor, pos: int, tgt: Tensor) -> Tensor:
    """The per-row two-form contrast (E223): log p(a) - log p(b) at `pos`, tgt (B, 2) = (a, b) per row; the logit
    difference, as `steering.contrast_at` for one fixed pair."""
    lg, t = logits[:, pos].float(), tgt.to(logits.device)
    return (lg.gather(1, t[:, :1]) - lg.gather(1, t[:, 1:])).squeeze(1).double().cpu()


@dataclass
class PairRead(Read):
    """A Read scored on a two-form pair per row (E223): toks end at pos (the label layout), tgt and fit_tgt (N, 2) hold
    each row's (a, b). One unit, "contrast" (`pair_at`); no k_half (it reads one target per row)."""

    def __post_init__(self) -> None:
        assert self.ab is None and self.pos == self.toks.shape[1] - 1 == self.fit.shape[1] - 1
        assert self.tgt.shape == (len(self.toks), 2) and self.fit_tgt.shape == (len(self.fit), 2)
        assert bool((self.tgt[:, 0] != self.tgt[:, 1]).all()) and bool((self.fit_tgt[:, 0] != self.fit_tgt[:, 1]).all())

    @property
    def scored(self) -> str:
        return "contrast"

    def fns(self) -> dict[str, ST.Readout]:
        return {"contrast": pair_at}

    def windows(self) -> tuple[Tensor, int | Tensor, int | None]:
        return torch.cat([self.toks, self.tgt], 1), PAIR, self.toks.shape[1]

    def single(self) -> dict[str, ST.Readout]:
        return {}


def ratio(num: float, gain: float) -> float:
    return num / gain if gain >= MIN_GAIN else float("nan")


@dataclass
class Reader:
    """The readings of one task's states. ks: k per block for the neuron reader; k_m: E136's on-counts (untrained gates);
    S: E98's k_half settings (neuron_r predictions.json); c: n_select, strengths, n_random."""

    net: ST.Net
    r: Read
    decomps: dict[str, U.Decomposition]
    k_m: dict[str, int]
    k_fixed: int
    S: dict[str, Any]
    c: dict[str, Any]
    batch: int = 32  # the untrained read's batch (E136 read E53's windows at 32)
    w0: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ks = {"k64": dict.fromkeys(FS.LAYERS, self.k_fixed), "kmatched": {l: int(self.k_m[k]) for l, k in enumerate(DOWN)}}
        self.zero_idx = torch.zeros(len(self.r.fit), dtype=torch.long)
        self.read_idx = torch.zeros(len(self.r.toks), dtype=torch.long)
        sel = slice(0, self.c["n_select"])
        self.w0["sel"] = self.steered(self.r.fit[sel], self.r.fit_tgt[sel], None)[self.r.scored]
        self.w0["rows"] = self.steered(self.r.toks, self.r.tgt, None)
        self.w0["untrained"] = {u: self.untrained_cells(None, u) for u in self.r.fns()}

    # ----------------------------------------------------------------------------- one reading each

    def steered(self, toks: Tensor, tgt: Tensor, how: tuple[Tensor, int, float] | None) -> dict[str, Tensor]:
        """Per unit, W0's reading of `toks` steered by (vector, layer, strength), or W0 itself (None)."""
        idx = torch.zeros(len(toks), dtype=torch.long)
        lg = (torch.cat([self.net.run(toks[s : s + FS.BATCH])[0][:, self.r.pos] for s in range(0, len(toks), FS.BATCH)]) if how is None
              else FS.logits_steered(self.net, toks, idx, self.r.pos, how[0], how[1], how[2]))  # fmt: skip
        return self.r.units_of(tgt)(lg, slice(0, len(toks)))

    def untrained_cells(self, dW: ST.DW | None, u: str) -> dict[str, float]:
        w, token, ci_len = self.r.windows()
        target = None if dW is None else {k: x + dW[k].to(x.device) for k, x in self.net.W0.items()}
        unit = None if u == "lo" else self.r.fns()[u]
        return U.do_nothing_cells(self.net.model, w, self.r.pos, token, self.decomps, self.k_m, self.batch, ci_len, target, unit)

    def share(self, dW: ST.DW) -> dict[str, Any]:
        c = MR.share_cells_by(self.net.model, self.net.W0, dW, self.r.toks, self.r.pos, self.r.units_of(self.r.tgt))
        num = {u: float((c["L01"][u] - c["L00"][u]).mean()) for u in c["T0"]}
        gain = {u: float((c["T1"][u] - c["T0"][u]).mean()) for u in c["T0"]}
        return {"share": {u: ratio(num[u], gain[u]) for u in num}, "numerator": num, "gain": gain}

    def untrained(self, dW: ST.DW) -> dict[str, Any]:
        out: dict[str, Any] = {"share": {}, "numerator": {}}
        for u, c0 in self.w0["untrained"].items():
            c1 = self.untrained_cells(dW, u)
            out["numerator"][u] = {n: c1[n] - c0[n] for n in self.decomps}
            out["share"][u] = {n: ratio(c1[n] - c0[n], c1["T"] - c0["T"]) for n in self.decomps}
        return out

    def neurons(self, dW: ST.DW) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
        """The neuron reading per k, and the fp32 T cells per unit (the gain and the prior come from them)."""
        out: dict[str, Any] = {}
        T: dict[str, dict[str, Tensor]] = {}
        for kn, k in self.ks.items():
            c = MR.topk_cells_by(self.net, dW, self.r.toks, self.r.pos, k, self.r.units_of(self.r.tgt))
            num = {u: float((c["N01"][u] - c["N00"][u]).mean()) for u in c["T0"]}
            gain = {u: float((c["T1"][u] - c["T0"][u]).mean()) for u in c["T0"]}
            out[kn] = {"share": {u: ratio(num[u], gain[u]) for u in num}, "numerator": num}
            T = {"T0": c["T0"], "T1": c["T1"]}
        return out, T

    def steering(self, dW: ST.DW, gain: dict[str, float], gen: torch.Generator) -> dict[str, Any]:
        """E143's `Job.kind` for the pooled vector, the selection in the scored unit, every reading per unit."""
        r, n = self.r, self.c["n_select"]
        v = FS.vectors(self.net, r.fit, self.zero_idx, 1, dW, r.pos)["pooled"]
        grid = {(l, s): float((self.steered(r.fit[:n], r.fit_tgt[:n], (v[l], l, s))[r.scored] - self.w0["sel"]).mean())
                for l in FS.LAYERS for s in self.c["strengths"]}  # fmt: skip
        setting = FS.pick(grid)
        rec: dict[str, Any] = {"grid": {f"{l},{s}": x for (l, s), x in grid.items()}, "setting": setting}
        if setting is None:
            return {**rec, "share": dict.fromkeys(r.fns(), float("nan")), "rise": {}}
        l, s = setting
        got = self.steered(r.toks, r.tgt, (v[l], l, s))
        rec["rise"] = {u: float((got[u] - self.w0["rows"][u]).mean()) for u in got}
        rec["share"] = {u: ratio(rec["rise"][u], gain[u]) for u in got}
        rec["off_kl"] = FS.off_kl(self.net, r.toks, r.pos, self.read_idx, (v[l], l, s))
        rnd = [self.steered(r.toks, r.tgt, (FS.random_like(v[l], gen), l, s))[r.scored] for _ in range(self.c["n_random"])]
        rec["random"] = [ratio(float((x - self.w0["rows"][r.scored]).mean()), gain[r.scored]) for x in rnd]
        rec["random_mean"] = statistics.fmean(rec["random"])
        return rec

    def k_half(self, dW: ST.DW) -> dict[str, Any]:
        """E98's k_half per unit (k_half_fill.carrying's calls), with the r_half share and numerator it rests on."""
        reads, S, out = NR.Reads(self.r.name, self.r.toks, self.r.tgt, self.r.pos), self.S, {}
        for u, f in self.r.single().items():
            rec = NR.shares(NR.state_cells(self.net, reads, dW, S["batch"], f), S["gain_floor"])
            full = rec["num"]["r_half"]
            score = NR.ranking(self.net, reads, dW, S["batch"], None if u == "lo" else f)
            order = torch.sort(score, descending=True, stable=True).indices
            nums = NR.grid_numerators(self.net, reads, dW, order, S["k_grid"], S["batch"], f)
            out[u] = {"k_half": NR.k_half(nums, full, rec["r_half"], S["carry_floor"]), "r_half": rec["r_half"], "r_half_num": full,
                      "G": rec["G"], "end_diff": nums[NR.N_NEURONS] - full, "numerators": nums}  # fmt: skip
        return out

    # ----------------------------------------------------------------------------- one state

    def state(self, dW: ST.DW, seed: int) -> dict[str, Any]:
        """Every reading of one update, CPU fp32 dW over the 24 matrices."""
        spec = US.state(dW)
        neuron, T = self.neurons(dW)
        gain = {u: float((T["T1"][u] - T["T0"][u]).mean()) for u in T["T0"]}
        rec: dict[str, Any] = {"size": update_size(dW, self.net.W0), "erank": spec["erank"], "gain": gain,
                               "prior": {u: float(T["T0"][u].mean()) for u in T["T0"]}, "neuron": neuron}  # fmt: skip
        rec["share"] = self.share(dW)
        rec["untrained"] = self.untrained(dW)
        rec["steering"] = self.steering(dW, gain, torch.Generator().manual_seed(self.c["random_seed"] + seed))
        rec["k_half"] = self.k_half(dW)
        return rec


def numerators(rec: dict[str, Any]) -> dict[str, float]:
    """Every numerator and gain of a state record, flat: all exactly 0 at a zero update (the stage's first control)."""
    out = {f"gain {u}": g for u, g in rec["gain"].items()}
    out |= {f"share {u}": x for u, x in rec["share"]["numerator"].items()}
    out |= {f"untrained {u} {n}": x for u, d in rec["untrained"]["numerator"].items() for n, x in d.items()}
    out |= {f"neuron {k} {u}": x for k, d in rec["neuron"].items() for u, x in d["numerator"].items()}
    out |= {f"steering {u}": x for u, x in rec["steering"].get("rise", {}).items()}
    out |= {f"k_half {u} r_half_num": d["r_half_num"] for u, d in rec["k_half"].items()}
    return out
