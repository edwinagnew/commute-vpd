"""E90's readings on W0 and on states, as stage1.py and stage2.py both take them; each finished row is written to `out` at once."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import apparatus as AP
import rules
import torch
from apparatus import LARGE, Capability, Site, dump, log, stop
from torch import Tensor

from commute import readout as RO
from commute import removal as RM
from commute import steering as ST


@dataclass
class Job:
    """What every reading needs: the model, the settings, the saved records, the results and where they go."""

    net: ST.Net
    S: dict[str, Any]
    rec: dict[Path, str]
    res: dict[str, Any]
    out: Path

    def state(self, path: Path) -> ST.DW:
        try:
            return torch.load(AP.checked(path, self.rec, self.res["sha256"]), weights_only=True)
        except ValueError as e:
            stop(self.out, self.res, str(e))

    def site(self, path: Path) -> tuple[Site, Any]:
        try:
            return AP.kv_site(AP.checked(path, self.rec, self.res["sha256"]))
        except ValueError as e:
            stop(self.out, self.res, str(e))


@dataclass
class Basis:
    """A capability's removal at each block, and its random twins of the same rank."""

    bases: dict[int, Tensor]
    randoms: dict[int, list[Tensor]]


def value_form(j: Job, c: Capability, tag: int) -> Basis:
    return with_random(j, {b: u.float() for b, u in c.bases(j.S["blocks"]).items()}, tag)


def with_random(j: Job, bases: dict[int, Tensor], tag: int) -> Basis:
    return Basis(bases, AP.random_bases(bases, j.S["random_seed"], tag, j.S["n_random"]))


def control(j: Job, c: Capability, f: Basis) -> dict[str, bool]:
    """W0's log-odds of f_C at X_C on held rows under each basis, against the same removal of random bases; alone."""
    w0 = AP.read_all(j.net, c.read, None, f.bases, f.randoms)
    rec: dict[str, Any] = {"rank": c.rank, "n_windows": len(c.read.toks)}
    for b, u in f.bases.items():
        drops = [float((w0["clean"][1] - w0[b, i][1]).mean()) for i in range(len(f.randoms[b]))]
        drop, band = float((w0["clean"][1] - w0[b, -1][1]).mean()), RM.band_of(drops)
        rec[str(b)] = {"drop": drop, "band": band, "teeth": rules.teeth(drop, band), "random": drops, "nats_drop": float((w0["clean"][0] - w0[b, -1][0]).mean()),
                       "variance_share": AP.variance_share(c.fit_h[b], u), "variance_share_random": sum(AP.variance_share(c.fit_h[b], r) for r in f.randoms[b]) / len(f.randoms[b])}  # fmt: skip
        log(f"[control {c.name} block {b}] W0 drop {drop:.4f} band {band:.4f} teeth {rec[str(b)]['teeth']} nats {rec[str(b)]['nats_drop']:.4f} variance share {rec[str(b)]['variance_share']:.4f} random {rec[str(b)]['variance_share_random']:.4f}")
    j.res.setdefault("control", {})[c.name] = rec
    dump(j.out, j.res)
    return {str(b): rec[str(b)]["teeth"] for b in f.bases}


def do_nothing(j: Job, c: Capability, f: Basis, on: dict[str, bool], block: int) -> None:
    """R1 on C's own read site: W0 against itself, gain 0."""
    w0 = AP.read_all(j.net, AP.capped_held(c.read, 256, j.S["control_seed"]), None, {block: f.bases[block]}, {block: f.randoms[block]})
    z = AP.state_row(w0, w0, [block], j.S["n_random"], j.S["gain_floor"])[str(block)]["A_lo"]
    v = rules.verdict(on[str(block)], z, 0.0, 0.0)
    j.res.setdefault("R1", {})[c.name] = {"verdict": v, "A_lo": z}
    log(f"[R1 {c.name}] do-nothing A_lo {z} -> {v} (filed {rules.NO_GAIN})")


def read_state(j: Job, name: str, s: Site, dW: ST.DW | None, w0: AP.Reads, f: Basis) -> dict[str, Any]:
    r = AP.state_row(w0, AP.read_all(j.net, s, dW, f.bases, f.randoms), f.bases, j.S["n_random"], j.S["gain_floor"])
    AP.say_row(name, r, f.bases)
    j.res.setdefault("states", {})[name] = r
    dump(j.out, j.res)
    return r


def read_write(j: Job, name: str, s: Site, train: Tensor, table: tuple[list[int], list[int]], G: float, w0: AP.Reads, f: Basis, key_at: int | None = None) -> dict[str, Any] | None:
    """The hand write of a table at matched gain G, keyed at index `key_at` (default `s.half`), read like a state."""
    return read_matched(j, name, s, AP.table_inputs(j.net, train, s.half, table[0], table[1], key_at), G, w0, f)


def read_matched(j: Job, name: str, s: Site, inputs: tuple[Tensor, Tensor], G: float, w0: AP.Reads, f: Basis) -> dict[str, Any] | None:
    """The write X A = U at matched gain G, read like a state; None when the match failed (the row reads no call)."""
    st, rec, X = AP.hand_write(j.net, inputs[0], inputs[1], s, G, j.S["write_rel_tol"])
    torch.save(X.float().cpu(), LARGE / f"X_{name}.pt")
    j.res.setdefault("writes", {})[name] = rec
    log(f"[write {name}] {json.dumps(rec)}")
    return read_state(j, f"{name}_write", s, st, w0, f) if st else None


def table_of(w: dict[str, Any]) -> tuple[list[int], list[int]]:
    return list(w["task"].keys), list(w["task"].values)


def possession(j: Job, c: Capability, w0: Tensor, held: tuple[Tensor, Tensor, Tensor], path: Path, values: list[int]) -> dict[str, Any]:
    """The readout for C and its shuffled twin, read on the held windows (residuals, W0's logits, value index): the record, with
    b* (None for no possessed block). An early check raises `RO.ReadoutStop`: the caller says what a miss does."""
    R, (hh, hw, hy) = j.S["readout"], held
    probes, twin = AP.train_pair(c.fit_h, c.y, w0, R, log)
    torch.save({"probes": probes.state_dict(), "twin": twin.state_dict(), "values": values}, path)
    with torch.no_grad():
        loss = RO.xent64(probes(hh), hy)
    acc, shuffled = AP.accuracy(probes, hh, hy), AP.accuracy(twin, hh, hy)
    own = float((hw.argmax(-1) == hy).double().mean())
    flags = {b: rules.possessed(acc[b], shuffled[b], own, R["possess_num"], R["possess_den"]) for b in range(len(acc))}
    rec = {"acc": acc, "shuffled": shuffled, "own": own, "possessed": flags, "b_star": rules.b_star(flags, j.S["blocks"][0], RO.LAST), "heldout_loss": loss, "model_loss": RO.xent64(hw[None], hy)[0]}
    log(f"[readout {c.name}] {json.dumps(rec)} (block 0 printed only)")
    return rec
