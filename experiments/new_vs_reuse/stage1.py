"""E90 stage 1: the new measurements, control first and alone. Plan: reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md,
"Stages". Rules: results/predictions.json (refused unless committed unchanged), rules.py. Not run as of its commit.
  (i)   the readout for C = W0's guess at blocks 1-2, block 0 printed only; a held-out loss over the bar stops
  (ii)  the control, alone: W0's drop on f_C at X_C under each basis against 20 random bases; a C with no teeth is out; no
        C left stops. The do-nothing state read there (R1)
  (iii) R2, R3, R7: each state's share with its band as it is read, its hand write at matched gain. A miss is a gap, reported
  (iv)  input-dependent lowercase, emoticon and one ladder rung: not designed yet, see `input_dependent`
The definition is frozen (plan, "Scope as restated"). Only a sha256 mismatch, the readout's early checks or its fit bar stop the
job. Writes results/stage1.json after each step, results/stop.json on a stop, results/large/{readout,bases,X_*}.pt.
Usage: sbatch cluster/new_vs_reuse.sbatch stage1
"""

from __future__ import annotations

import json
import math
from typing import Any

import apparatus as AP
import reading as RD
import rules
import torch
from apparatus import LARGE, RESULTS, Capability, Site, dump, log, stop
from reading import Basis, Job
from torch import Tensor

from commute import readout as RO
from commute import removal as RM
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit

OUT = RESULTS / "stage1.json"
REG, PAIR = AP.EXP / "regating_table/results/large", AP.EXP / "open_pair/results/large"
LADDER, COVER = AP.EXP / "task_ladder/results/large", AP.EXP / "reading_a_coverage/results/large"
RUNG, TRIGGER, TARGET, HALF = "r1_t534_v369", 534, 369, 20


def guess_capability(j: Job, rows: Tensor, vals: Tensor) -> tuple[Capability, Tensor, Tensor, Tensor, Tensor]:
    """W0's guess at X_C on fit and held rows; the unfit values leave. Returns the capability (read: every held window), the
    fit logits, the held residuals, the held logits and the held value index, all over the fit values."""
    P, G, nf, net = j.S["pile"], j.S["guess_scan"], j.S["pile"]["fit_row_count"], j.net
    cand, raw = AP.scan_guess(net, rows[:nf], vals, HALF, P["cap"], G["n_fit_windows"], G["seed"])
    hcand, hraw = AP.scan_guess(net, rows[nf:], vals, HALF, P["held_cap"], G["n_held_windows"], G["seed"] + 1)
    _, h, w0, y = AP.keep_guess(net, cand, vals, HALF, P["cap"])
    ht, hh, hw, hy = AP.keep_guess(net, hcand, vals, HALF, P["held_cap"])
    fit = AP.fit_values(y, len(vals), P["unfit_below"])
    j.res["guess_positions"] = {"raw_fit": raw, "raw_held": hraw, "fit_windows": len(y), "held_windows": len(hy), "fit_per_value": torch.bincount(y, minlength=len(vals)).tolist(),
                                "fit_values": vals[fit.cpu()].tolist()}  # fmt: skip
    log(f"[guess positions] {json.dumps(j.res['guess_positions'])}")
    new = torch.full((len(vals),), -1, device=y.device)
    new[fit] = torch.arange(len(fit), device=y.device)
    f, hf = new[y] >= 0, new[hy] >= 0
    read = Site(ht[hf.cpu()], vals[hy[hf].cpu()], HALF)
    return Capability("W0's guess", h[:, f], new[y[f]], read, len(fit) - 1), w0[f][:, fit], hh[:, hf], hw[hf][:, fit], new[hy[hf]]


def possession(j: Job, c: Capability, w0: Tensor, hh: Tensor, hw: Tensor, hy: Tensor) -> int | None:
    """(i). Returns b*, or None for no possessed block. An early check or a fit miss stops."""
    R = j.S["readout"]
    try:
        probes, twin = AP.train_pair(c.fit_h, c.y, w0, R, log)
    except RO.ReadoutStop as e:
        stop(OUT, j.res, str(e))
    torch.save({"probes": probes.state_dict(), "twin": twin.state_dict(), "values": j.res["guess_positions"]["fit_values"]}, LARGE / "readout.pt")
    with torch.no_grad():
        loss = RO.xent64(probes(hh), hy)
    acc, shuffled = AP.accuracy(probes, hh, hy), AP.accuracy(twin, hh, hy)
    own = float((hw.argmax(-1) == hy).double().mean())
    flags = {b: rules.possessed(acc[b], shuffled[b], own, R["possess_num"], R["possess_den"]) for b in range(len(acc))}
    block = rules.b_star(flags, j.S["blocks"][0], RO.LAST)
    j.res["readout"] = {"acc": acc, "shuffled": shuffled, "own": own, "possessed": flags, "b_star": block, "heldout_loss": loss, "model_loss": RO.xent64(hw[None], hy)[0]}
    log(f"[readout] {json.dumps(j.res['readout'])} (block 0 printed only)")
    dump(OUT, j.res)
    bar = R["fit_bar_last_block_heldout_loss"]
    if rules.fit_stops(loss[RO.LAST], bar):
        stop(OUT, j.res, f"the readout's held-out last-block loss {loss[RO.LAST]:.4f} is over the filed bar {bar}")
    return block


def r2(j: Job, f: Basis, on: dict[str, bool]) -> dict[str, Any]:
    if not any(on.values()):
        return rules.r2(False, math.nan, math.nan, math.nan, math.nan)
    si, wi = j.site(REG / "windows_ident32.pt")
    sr, _ = j.site(REG / "windows_rand32.pt")
    w0i, w0r = AP.read_all(j.net, si, None, f.bases, f.randoms), AP.read_all(j.net, sr, None, f.bases, f.randoms)
    ident = RD.read_state(j, "ident32_s0", si, j.state(REG / "ident32_s0.pt"), w0i, f)
    rand = RD.read_state(j, "rand32_s0", sr, j.state(REG / "rand32_s0.pt"), w0r, f)
    write = RD.read_write(j, "ident32", si, wi["train"].tokens, RD.table_of(wi), ident["G_logodds"], w0i, f)
    saved = json.loads((AP.EXP / "regating_table/results/stage0b.json").read_text())["states"]["ident32/write"]["alpha"]
    log(f"[write ident32] alpha against E46's saved {saved:.4f}, matched there on its own full_read")
    return rules.r2(on["1"], ident["1"]["A_lo"], rand["1"]["A_lo"], write["1"]["A_lo"] if write else math.nan, AP.band_over([x for x in (ident, rand, write) if x], 1), write is not None)


def r3(j: Job, block: int | None, f: Basis, on: dict[str, bool]) -> dict[str, Any]:
    if block is None:
        return rules.r3(None, math.nan, math.nan, math.nan)
    if not any(on.values()):
        return rules.r3(False, math.nan, math.nan, math.nan)
    sk, wk = j.site(PAIR / "amend1/windows_rank1_trained.pt")
    sc, _ = j.site(PAIR / "windows_cipher_trained.pt")
    w0k, w0c = AP.read_all(j.net, sk, None, f.bases, f.randoms), AP.read_all(j.net, sc, None, f.bases, f.randoms)
    rank1 = RD.read_state(j, "rank1_s0", sk, j.state(PAIR / "amend1/rank1_s0.pt"), w0k, f)
    cipher = RD.read_state(j, "cipher_s0", sc, j.state(PAIR / "cipher_s0.pt"), w0c, f)
    write = RD.read_write(j, "rank1", sk, wk["train"].tokens, RD.table_of(wk), rank1["G_logodds"], w0k, f)
    b, c = str(block), write[str(block)]["A_lo"] if write else math.nan
    band = AP.band_over([x for x in (rank1, cipher, write) if x], block)
    return {**rules.r3(on[b], rank1[b]["A_lo"], c, band, write is not None), "block": block, "cipher_share": cipher[b]["A_lo"],
            "cipher_verdict": rules.verdict(on[b], cipher[b]["A_lo"], c, band, write is not None)}  # fmt: skip


def r7(j: Job, rows: Tensor) -> dict[str, Any]:
    """Control first and alone, then the state and its constant push at block 1 (block 2 printed beside)."""
    c = AP.leaning_capability(j.net, "W0's leaning", rows, TRIGGER, TARGET, HALF, j.S, j.S["n_control"])
    f = RD.with_random(j, {b: RM.pattern_basis(c.fit_h[b], c.y) for b in j.S["blocks"]}, 3)
    on = RD.control(j, c, f)
    if not any(on.values()):
        return rules.r7(False, math.nan, math.nan, math.nan)
    p = LADDER / f"windows_{RUNG}.pt"
    w = torch.load(AP.checked(p, j.rec, j.res["sha256"]), weights_only=False)
    assert (w["task"]["trigger"], w["task"]["target"]) == (TRIGGER, TARGET), w["task"]
    held = w["heldout"]
    s = Site(held.tokens, torch.full((len(held.tokens),), TARGET), held.half)
    w0 = AP.read_all(j.net, s, None, f.bases, f.randoms)
    state = RD.read_state(j, f"{RUNG}_s0", s, j.state(COVER / RUNG / "dW_ft_step10_s0.pt"), w0, f)
    push = RD.read_write(j, RUNG, s, w["train"].tokens, ([TRIGGER], [TARGET]), state["G_logodds"], w0, f)
    return rules.r7(on["1"], state["1"]["A_lo"], push["1"]["A_lo"] if push else math.nan, AP.band_over([x for x in (state, push) if x], 1), push is not None)


def controls(j: Job, rows: Tensor) -> tuple[int | None, Basis, dict[str, bool], Basis, dict[str, bool]]:
    """(i) and (ii): the readout, then each C's control alone; nothing reads a state before this returns."""
    S = j.S
    values = json.loads((AP.EXP / "open_pair/results/amend1.json").read_text())["c0"]["rank1"]["values"]
    guess, w0, hh, hw, hy = guess_capability(j, rows, torch.tensor(sorted(set(values))))
    block = possession(j, guess, w0, hh, hw, hy)
    guess.read = AP.capped_held(guess.read, S["n_control"], S["control_seed"])
    gf = RD.value_form(j, guess, 1)
    gon = RD.control(j, guess, gf)
    keys = torch.tensor(list(torch.load(REG / "windows_ident32.pt", weights_only=False)["task"].keys))
    ident = AP.identity_capability(j.net, "token identity", rows, keys, HALF, S, S["n_control"])
    idf = RD.value_form(j, ident, 2)
    ion = RD.control(j, ident, idf)
    torch.save({"guess": gf.bases, "identity": idf.bases}, LARGE / "bases.pt")
    if block is not None:
        RD.do_nothing(j, guess, gf, gon, block)
    RD.do_nothing(j, ident, idf, ion, 1)
    dump(OUT, j.res)
    return block, gf, gon, idf, ion


def input_dependent(j: Job) -> None:
    """The place for the first pass's input-dependent lowercase, emoticon and one ladder rung (plan, "Scope as restated"). The
    lead's next session designs them; each then reads like `r7`: its control alone, the state, its spared control."""
    j.res["input_dependent"] = "not designed: the lead's next session"
    log(f"[input-dependent tasks] {j.res['input_dependent']}")


def main() -> None:
    S = AP.filed()["settings"]
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    j = Job(ST.Net("cuda"), S, AP.records(), {"sha256": {}}, OUT)
    rows = pile_rows(*S["pile"]["rows"], log=log)
    block, gf, gon, idf, ion = controls(j, rows)
    readings: dict[str, Any] = j.res.setdefault("readings", {})
    for name, read in (("R2", lambda: r2(j, idf, ion)), ("R3", lambda: r3(j, block, gf, gon)), ("R7", lambda: r7(j, rows))):
        readings[name] = read()
        log(f"[{name}] {json.dumps(readings[name])}")
        dump(OUT, j.res)
    input_dependent(j)
    j.res |= {"gaps": rules.gaps(readings), "complete": True}
    dump(OUT, j.res)
    log(f"[gaps, reported; the definition is frozen] {j.res['gaps']}")
    hard_exit(0)


if __name__ == "__main__":
    main()
