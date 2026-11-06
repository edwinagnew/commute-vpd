"""E125 Amendment 1, 2026-11-06: E125's removal read at the repeated word's own position (HALF - 1) and at the trigger's
block 2, on the four saved fine-tunes. DESCRIPTIVE: E125's verdicts stand as filed under `decisions.md` rows 81 and 87;
nothing here maps to a verdict. Plan: reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md, "Amendment 1". Settings,
controls and expected values: results/predictions_amend1.json (refuses unless it and predictions.json are committed
unchanged). One cluster debug job; results/amend1.json written after every unit, every number printed:

  controls  C1: the ridge probe refit at HALF - 1 (run.py's beside_acc_at_word path) reproduces stage 0's accuracies;
            C2: real_s0's trigger-site share at l* reproduces stage 1's P3. A miss: exit 1.
  sites     per site (word blocks 0-2, trigger block 2): the random direction on all four states first (exit 1 if one is
            over E125's do-nothing bar), then the probe direction's removal share per state.

    sbatch cluster/fresh_reuse_pair_amend1.sbatch
"""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from run import (
    HALF,
    LARGE,
    RESULTS,
    Config,
    load_config,
    load_dW,
    log,
    part,
    say,
    states,
    task,
)
from torch import Tensor

from commute import case_label as CL
from commute import definition_reads as DFR
from commute import removal as RM
from commute import steering as ST
from commute.jobs import hard_exit, require_filed
from commute.open_pair import jsonable

PRED_A, OUT = RESULTS / "predictions_amend1.json", RESULTS / "amend1.json"
WORD = HALF - 1  # the repeated word's own position; the label is read at the trigger, HALF


class Stop(Exception):
    pass


def dump(res: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(res), indent=1))
    tmp.replace(OUT)


def word_probe(net: ST.Net, C: Config, t: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
    """run.py probes' fit at HALF - 1: the repeat probe (weights per block) and its Hewitt-Liang control, both read on the
    held-out-word task windows at the same position."""
    pt, yp = t["probe"]["toks"], t["probe"]["y"]
    held, yh = part(t, "real", "held")
    h, hh = DFR.resid_at(net, pt, WORD), DFR.resid_at(net, held, WORD)
    W, _, acc = CL.value_probes(h, yp, hh, yh, 2, C.ridge_lambda)
    ctrl = CL.control_table(t["vocab"], 2, C.control_seed)
    _, _, acc_c = CL.value_probes(h, ctrl[pt[:, WORD]], hh, ctrl[held[:, WORD]], 2, C.ridge_lambda)
    return W[:, 1].cpu(), {"acc_repeat": acc, "acc_control": acc_c, "selectivity": [a - c for a, c in zip(acc, acc_c, strict=True)]}


def share(net: ST.Net, t: dict[str, Any], R: dict[str, Any], name: str, dW: ST.DW, block: int, u: Tensor, site: int) -> dict[str, float]:
    fit, yf = part(t, name.split("_s")[0], "fit")
    return DFR.removal_share(net, fit, t["ab"], yf, dW, block, u, HALF, R["gain_floor"], site=site)


def controls(net: ST.Net, C: Config, A: dict[str, Any], R: dict[str, Any], t: dict[str, Any], dWs: dict[str, ST.DW],
             w_trig_all: Tensor, rand: Tensor, res: dict[str, Any]) -> Tensor:  # fmt: skip
    s0 = json.loads((RESULTS / "stage0.json").read_text())
    s1 = json.loads((RESULTS / "stage1.json").read_text())
    w_word, res["probe_at_word"] = word_probe(net, C, t)
    say("probe at the word", res["probe_at_word"])
    want = s0["P0"]["beside_acc_at_word"]
    gaps = [abs(a - b) for a, b in zip(res["probe_at_word"]["acc_repeat"], want, strict=True)]
    res["C1"] = {"want": want, "max_gap": max(gaps), "pass": max(gaps) <= A["probe_acc_tol"]}
    say("C1", res["C1"])
    dump(res)
    if not res["C1"]["pass"]:
        raise Stop(f"C1: the refit probe at the word misses stage 0's accuracies by {max(gaps):.5f} > {A['probe_acc_tol']}")
    l_star = s0["l_star"]
    w_trig = w_trig_all[l_star]
    got = share(net, t, R, "real_s0", dWs["real_s0"], l_star, RM.unit_basis(w_trig), HALF)
    rnd = share(net, t, R, "real_s0", dWs["real_s0"], l_star, rand, HALF)
    want_s = s1["reads"]["real_s0"]["P3"]
    res["C2"] = {"want": want_s, "got": got, "random_got": rnd, "random_want": s1["do_nothing"]["real_s0"]["share"],
                 "pass": abs(got["share"] - want_s) <= A["share_tol"]}  # fmt: skip
    say("C2", res["C2"])
    dump(res)
    if not res["C2"]["pass"]:
        raise Stop(f"C2: real_s0's trigger share {got['share']:.5f} misses stage 1's {want_s:.5f} by more than {A['share_tol']}")
    return w_word


def site_unit(net: ST.Net, C: Config, R: dict[str, Any], t: dict[str, Any], dWs: dict[str, ST.DW], u: Tensor, rand: Tensor,
              block: int, site: int) -> dict[str, Any]:  # fmt: skip
    """The random direction on every state first; over E125's do-nothing bar on any state stops before the probe's read."""
    out: dict[str, Any] = {"block": block, "position": site, "random": {}, "probe": {}}
    for n in states(C):
        out["random"][n] = share(net, t, R, n, dWs[n], block, rand, site)
    say(f"random b{block} pos{site}", out["random"])
    if not CL.do_nothing_ok([v["share"] for v in out["random"].values()], R["do_nothing_max"]):
        out["stopped"] = f"the random direction's share is over {R['do_nothing_max']} on a state"
        return out
    for n in states(C):
        out["probe"][n] = share(net, t, R, n, dWs[n], block, u, site)
    say(f"probe b{block} pos{site}", out["probe"])
    return out


def run(net: ST.Net, C: Config, R: dict[str, Any], A: dict[str, Any], t0: float) -> int:
    res: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"sites": {}}
    assert "stopped" not in res, f"amend1 stopped before ({res.get('stopped')}): move amend1.json aside to start again"
    t, s1 = task(), json.loads((RESULTS / "stage1.json").read_text())
    dWs = {n: load_dW(n, s1["states"][n]) for n in states(C)}
    w_trig = torch.load(LARGE / "probes.pt")["w"]  # stage 0's trigger probe, (blocks, d)
    rand = RM.random_basis(w_trig.shape[1], 1, [C.random_dir_seed])  # E125's one seeded random direction
    try:
        w_word = controls(net, C, A, R, t, dWs, w_trig, rand, res)  # cheap: redone on every rerun
        log("STAGE 0 DONE (controls)")
        units = [(f"word_b{b}", w_word[b], b, WORD) for b in A["word_blocks"]] + [(f"trigger_b{b}", w_trig[b], b, HALF) for b in A["trigger_blocks"]]
        for i, (key, w, block, site) in enumerate(units, 1):
            if key in res["sites"]:
                continue
            if time.monotonic() - t0 > (A["stage_minutes"] - A["unit_minutes"]) * 60:
                log(f"[time] too little left for {key}: exit 3, resubmit")
                dump(res)
                return 3
            res["sites"][key] = site_unit(net, C, R, t, dWs, RM.unit_basis(w), rand, block, site)
            dump(res)
            if "stopped" in res["sites"][key]:
                raise Stop(f"{key}: {res['sites'][key]['stopped']}")
            log(f"STAGE {i} DONE ({key})")
    except Stop as e:
        res["stopped"] = str(e)
        log(f"[stop] {res['stopped']}")
        dump(res)
        return 1
    res["e125_verdicts_unchanged"] = {n: s1["reads"][n]["verdict"] for n in states(C)}
    dump(res)
    log("AMEND1 SHARES " + json.dumps({k: {n: v["share"] for n, v in u["probe"].items()} for k, u in res["sites"].items()}))
    return 0


def main() -> None:
    t0 = time.monotonic()
    C, P = load_config()
    require_filed(PRED_A)
    A = json.loads(PRED_A.read_text())["settings"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    code = run(ST.Net("cuda"), C, P["rules"], A, t0)
    log({0: "AMEND1 DONE", 1: "AMEND1 STOPPED (exit 1)"}.get(code, f"amend1: exit {code}, more to do"))
    hard_exit(code)


if __name__ == "__main__":
    main()
