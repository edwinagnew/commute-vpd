"""E90 stage 2: the full table on every saved seed, whatever R3 read (plan, "Scope as restated"). Rules: results/predictions.json,
rules.py. Not run as of its commit.
  row 1     E62's copy and lookup under head (2,4): W0's teeth on repeated random sequences against 20 random single heads, the
            band on every state, W0's log-odds for B under the head at the marker (reported), the verdict per seed.
            A "reuses C" here is a different rule's reading beside E62's filed "copy_not_elicitation", which stands.
  rows 2-4  rank-1, cipher, elicit under W0's guess; ident32, rand32, copy-24 under token identity; each state against the hand
            write of its own table at its own gain.
  I1 seeds  seeds 1-2 of the ladder table, only if Amendment 1's I1 read "reuses C" or "not through C" (I2 and I3 read in stage 1,
            a gap and "not specific", and stand, so they cannot ask for seeds).
  table     the deliverable, `final_table.py`.
Token identity's teeth come from Amendment 1 (results/amend1.json, its filing results/predictions_amend1.json), and copy-24's control
is read the same way. Stage 2 refuses without a complete stage 1 and a complete Amendment 1.
Writes results/stage2.json per state, resumes from it; exit 3 at the deadline. Usage: sbatch cluster/new_vs_reuse.sbatch stage2
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import apparatus as AP
import final_table as FT
import input_dep as ID
import reading as RD
import rules
import torch
from apparatus import EXP, LARGE, RESULTS, Site, dump, log
from reading import Basis, Job
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute import removal as RM
from commute import steering as ST
from commute.data import pile_rows
from commute.identity import log_odds
from commute.jobs import hard_exit

OUT, FIRST, AMENDED = RESULTS / "stage2.json", RESULTS / "stage1.json", RESULTS / "amend1.json"
CUE, REG = EXP / "induction_cue/results", EXP / "regating_table/results/large"
PAIR = EXP / "open_pair/results/large"
HEAD, HEAD_BLOCK, EXIT_RESUME, HALF = (2, 4), 2, 3, 20


def resume_or_exit(j: Job, deadline: float, name: str) -> None:
    dump(OUT, j.res)
    if time.monotonic() > deadline:
        log(f"deadline with {name} next; resubmit to resume (exit {EXIT_RESUME})")
        hard_exit(EXIT_RESUME)


# ---- row 1, the mechanism form ----


def under_heads(net: ST.Net, s: Site, dW: ST.DW | None, H: list[CP.Head], draws: list[list[CP.Head]], means: Tensor) -> AP.Reads:
    """`read_all`'s shape, with the head ablated in place of a basis removed."""
    out: AP.Reads = {"clean": AP.reads(net, s, dW)}
    for i, heads in [(-1, H), *enumerate(draws)]:
        with CP.ablated(net, heads, means):
            out[HEAD_BLOCK, i] = AP.reads(net, s, dW)
    return out


def induction_lo(net: ST.Net, toks: Tensor, length: int) -> float:
    """W0's mean log-odds of the token that followed the earlier occurrence, over the later copies."""
    logits = net.run(toks)[0]
    return float(log_odds(logits[:, length + 1 : -1], toks[:, length + 2 :].to(logits.device)).double().mean())


def mechanism_teeth(net: ST.Net, H: list[CP.Head], draws: list[list[CP.Head]], means: Tensor, res: dict[str, Any]) -> bool:
    """W0's drop on f_C at X_C (E62's prefix-score rows) under the head, against the random heads."""
    sc = json.loads((CUE / "predictions.json").read_text())["settings"]["score"]
    ranked = KD.vocab(cast(KD.Ctx, SimpleNamespace(net=net)))[3]  # vocab reads only ctx.net
    pool = ranked[sc["rank_from"] : int(len(ranked) * (1 - sc["rank_drop_last"]))]
    toks = CP.repeated_random(pool, sc["length"], sc["repeats"], sc["sequences"], sc["sos"], sc["seed"])
    clean = induction_lo(net, toks, sc["length"])

    def fall(heads: list[CP.Head]) -> float:
        with CP.ablated(net, heads, means):
            return clean - induction_lo(net, toks, sc["length"])

    random, drop = [fall(d) for d in draws], fall(H)
    band = RM.band_of(random)
    res["control"] = {"clean": clean, "drop": drop, "band": band, "teeth": rules.teeth(drop, band), "random": random}
    log(f"[control in-context copying] W0 log-odds {clean:.4f} drop under head {drop:.4f} band {band:.4f} teeth {res['control']['teeth']}")
    return res["control"]["teeth"]


def row1(j: Job, deadline: float) -> None:
    S, net = j.S, j.net
    w = torch.load(AP.checked(CUE / "large/windows.pt", j.rec, j.res["sha256"]), weights_only=False)
    heads = torch.load(CUE / "large/heads.pt", weights_only=False)
    assert [tuple(h) for h in heads["H"]] == [HEAD], heads["H"]
    H, means = [HEAD], heads["means"].to(net.device)
    draws = CP.random_others(H, 4, 6, S["n_random"], S["random_seed"])
    on = mechanism_teeth(net, H, draws, means, j.res)
    saved = json.loads((CUE / "run.json").read_text())["states"]
    sites = {m: Site(w["read"].tokens, w["target"][m]["read"], w["read"].half) for m in ("copy", "lookup")}
    w0 = {m: under_heads(net, s, None, H, draws, means) for m, s in sites.items()}
    j.res["W0_at_marker"] = {m: {"clean": float(r["clean"][1].mean()), "under_head": float(r[HEAD_BLOCK, -1][1].mean())} for m, r in w0.items()}
    log(f"[W0 at the marker] {json.dumps(j.res['W0_at_marker'])}")
    done = j.res.setdefault("states", {})
    for m, site in sites.items():
        for s in range(3):
            name = f"{m}_s{s}"
            resume_or_exit(j, deadline, name)
            if name not in done:
                reads = under_heads(net, site, j.state(CUE / "large" / f"{name}.pt"), H, draws, means)
                done[name] = AP.state_row(w0[m], reads, [HEAD_BLOCK], S["n_random"], S["gain_floor"])
                AP.say_row(name, done[name], [HEAD_BLOCK])
                log(f"[{name}] share against E62's saved {saved[name]['share']:.4f}")
    band = AP.band_over([done[f"{m}_s{s}"] for m in sites for s in range(3)], HEAD_BLOCK)
    share = lambda m, s: done[f"{m}_s{s}"][str(HEAD_BLOCK)]["A_lo"]
    verdicts = {a: rules.over_seeds([rules.verdict(on, share(a, s), share(b, s), band) for s in range(3)]) for a, b in (("copy", "lookup"), ("lookup", "copy"))}
    j.res["row1"] = {"band": band, "verdicts": verdicts, "shares": {m: [share(m, s) for s in range(3)] for m in sites},
                     "beside": "E62's filed copy_not_elicitation, reading mixed, stands; this is a different rule's reading"}  # fmt: skip
    log(f"[row 1] {json.dumps(j.res['row1'])}")


# ---- rows 2-4, the value form ----


def rows_of() -> dict[str, list[tuple[str, Path, Path]]]:
    """C -> (state name, its windows, its state file)."""
    a1 = PAIR / "amend1"
    per = lambda n, d, wf, seeds: [(f"{n}_s{s}", wf, d / f"{n}_s{s}.pt") for s in seeds]
    return {"guess": [*per("rank1", a1, a1 / "windows_rank1_trained.pt", range(3)), *per("cipher", PAIR, PAIR / "windows_cipher_trained.pt", range(3)), *per("elicit", PAIR, PAIR / "windows_elicit_trained.pt", range(3))],
            "identity": [*per("ident32", REG, REG / "windows_ident32.pt", range(2)), *per("rand32", REG, REG / "windows_rand32.pt", range(2))],
            "copy24": [(f"copy24_s{s}", a1 / "windows_copy_trained.pt", a1 / f"copy_s{s}.pt") for s in range(3)]}  # fmt: skip


def read_rows(j: Job, form: str, f: Basis, block: int, on: bool, deadline: float) -> None:
    """Each state, then its own table's hand write at its own gain; per map, the verdict over its seeds."""
    done, w0s = j.res.setdefault("states", {}), {}
    for name, wf, sf in rows_of()[form]:
        resume_or_exit(j, deadline, name)
        if f"{name}_row" in done:
            continue
        s, w = j.site(wf)
        if wf not in w0s:
            w0s[wf] = AP.read_all(j.net, s, None, f.bases, f.randoms)
        w0 = w0s[wf]
        st = RD.read_state(j, name, s, j.state(sf), w0, f)
        write = RD.read_write(j, name, s, w["train"].tokens, RD.table_of(w), st["G_logodds"], w0, f)
        done[f"{name}_row"] = {"state": st[str(block)]["A_lo"], "control": write[str(block)]["A_lo"] if write else math.nan, "band": st[str(block)]["band"], "write_band": write[str(block)]["band"] if write else math.nan}
    rows = {name: done[f"{name}_row"] for name, _, _ in rows_of()[form]}
    band = RM.max_band([x for r in rows.values() for x in (r["band"], r["write_band"])])
    per: dict[str, list[str]] = {}
    for name, r in rows.items():
        per.setdefault(name.rsplit("_", 1)[0], []).append(rules.verdict(on, r["state"], r["control"], band, not math.isnan(r["control"])))
    j.res.setdefault("rows", {})[form] = {"block": block, "band": band, "verdicts": {m: rules.over_seeds(v) for m, v in per.items()}, "shares": {n: r["state"] for n, r in rows.items()}}
    log(f"[{form}] {json.dumps(j.res['rows'][form])}")


def rows2to4(j: Job, first: dict[str, Any], amend: dict[str, Any], deadline: float) -> None:
    bases = torch.load(LARGE / "bases.pt", weights_only=True, map_location="cpu")
    block = first["readout"]["b_star"]
    if block is None:
        j.res.setdefault("rows", {})["guess"] = {"verdicts": {m: {"verdict": "unreadable: no readout"} for m in ("rank1", "cipher", "elicit")}}
    else:
        read_rows(j, "guess", RD.with_random(j, bases["guess"], 1), block, first["control"]["W0's guess"][str(block)]["teeth"], deadline)
    read_rows(j, "identity", RD.with_random(j, bases["identity"], 2), 1, amend["control"]["token identity"]["1"]["teeth"], deadline)
    rows = pile_rows(*j.S["pile"]["rows"], log=log)
    keys = torch.tensor(RD.table_of(j.site(PAIR / "amend1/windows_copy_trained.pt")[1])[0])
    c = AP.identity_capability(j.net, "token identity, copy-24's keys", rows, keys, HALF, j.S, j.S["n_control"])
    c.read = AP.own_top1(j.net, c.read)
    f = RD.value_form(j, c, 4)
    on = RD.control(j, c, f)
    read_rows(j, "copy24", f, 1, on["1"], deadline)


def ladder_seeds(j: Job, amend: dict[str, Any], deadline: float) -> None:
    """The plan's "a labelled verdict gets seeds 1-2", on the windows, bases and b* Amendment 1 saved. Nothing unless I1 read a label."""
    i1 = amend["readings"]["I1"]
    if i1["verdict"] not in (rules.REUSES, rules.NOT_THROUGH):
        return
    w = torch.load(LARGE / "windows_ladder.pt", weights_only=True)
    tb = ID.Table(w["keys"], w["answers"], w["train"], w["train_answer"], w["held"], w["held_answer"], [])
    block = i1["block"]
    f = RD.with_random(j, torch.load(LARGE / "bases_ladder.pt", weights_only=True, map_location="cpu"), 5)
    on = {str(block): amend["control"]["W0's guess, ladder"][str(block)]["teeth"]}
    per = j.res.setdefault("I1_per_seed", {"0": i1})
    for seed in (1, 2):
        resume_or_exit(j, deadline, f"ladder seed {seed}")
        if str(seed) not in per:
            per[str(seed)] = ID.read_table(j, "ladder", tb, f, on, block, seed)
    j.res["I1_seeds"] = rules.over_seeds([r["verdict"] for r in per.values()])
    log(f"[I1 seeds] {json.dumps(j.res['I1_seeds'])}")


def not_ready() -> str | None:
    """The first file stage 2 reads that is missing or not marked complete; None when both are."""
    for path in (FIRST, AMENDED):
        if not path.exists() or not json.loads(path.read_text()).get("complete"):
            return path.name
    return None


def main() -> None:
    P = AP.filed()
    S = P["settings"]
    missing = not_ready()
    if missing:
        log(f"{missing} has not completed: nothing runs (exit 1)")
        hard_exit(1)
    first, amend = json.loads(FIRST.read_text()), json.loads(AMENDED.read_text())
    deadline = time.monotonic() + 60 * S["deadline_min"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    res: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {"sha256": {}}
    j = Job(ST.Net("cuda"), S, AP.records(), res, OUT)
    row1(j, deadline)
    rows2to4(j, first, amend, deadline)
    ladder_seeds(j, amend, deadline)
    res |= {"table": FT.final_table(P, json.loads((RESULTS / "stage0.json").read_text()), first, amend, res), "complete": True}
    dump(OUT, res)
    for r in res["table"]:
        log(f"[table] {json.dumps(r)}")
    log("every claim: a circuit from old parts reads as reuse. E90 stage 2 complete")
    hard_exit(0)


if __name__ == "__main__":
    main()
