"""P2.28 (E187), the 67M half: do the share and the cheap readings still separate the nine built pairs when the two sides
are matched in update size? Plan reports/plans/2026-11-16_plan_p2-28-size-matched.md; rules and expected values
results/predictions.json (the run refuses unless it is committed unchanged). Descriptive: sides are named as built, no
reuse or new verdict. The Qwen half is qwen.py; the figure is plot.py (CPU, after both).

Size is ||δW|| / ||W0|| over the 24 matrices. Per pair, the side with the larger mean size is the larger side, and every
one of its saved updates is multiplied by one scalar c = mean size of the smaller side / mean size of the larger side, so
the side means match and each side keeps its own seed spread. Only the larger side is read again; the smaller side keeps
its saved readings (plot.py). Each reading is the function that wrote the scorecard's cell, on the same rows:
  share       E136's trained route split (share_untrained Job.trained)
  steering    E143's fair steering, pooled and per key (holes_fill Job.state); e62's from E135 (steering_copy_lookup)
  neuron top  E143's TopK neurons, k 64 and k matched (holes_fill Job.state)
  k_half      E98's carrying set on E98's rows (k_half_fill carrying, E98's and E147's state loaders)
  erank       update_spectrum.state, of c dW and of dW (equal by construction: a check that c is one scalar)

  stage 0, alone, a miss writes, logs STOP and exits 1:
    (a) sizes and c per pair, from E136's files and again from E98's (k_half's own files): the two c agree;
    (b) c = 1 through every reader on one state each, against the saved reading (tolerances: predictions.json);
    (c) the random floor: a Gaussian change at the larger side's first state's per-matrix norms times c, its gain on
        that side's rows (reported: a share of a gain under the gain floor is undefined, so on the 67M the floor
        of a gap is the seed ranges, as on the scorecard).
  stage 1: every larger-side state, each written as it lands; the projected time checked after four states.
Writes results/m67.json; prints "STAGE n DONE". A rerun resumes from what is written.

    sbatch --comment=E187 cluster/size_matched.sbatch
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from commute import steering as ST
from commute import update_spectrum as US
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions.json", RESULTS / "m67.json"
E136 = load_experiment("share_untrained", "run")  # Job.trained: the share
HF = load_experiment("holes_fill", "run")  # Job.state: TopK neurons and fair steering; lo_at
KH = load_experiment("k_half_fill", "run")  # States, carrying: k_half
SC = load_experiment("steering_copy_lookup", "run")  # e62's steering
R96 = load_experiment("refit_96key", "run")  # inventories, state_name: the 96-key states
BUILT = ("e62_copy_lookup", "e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3",
         "open_pair_c3_copy", "e58_rank1_96", "e58_copy_prior_cipher")  # fmt: skip
E62_LARGE = REPO / "experiments/induction_cue/results/large"


def log(msg: str) -> None:
    print(msg, flush=True)


def load(rel: str) -> Any:
    return json.loads((REPO / rel).read_text())


def write(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(OUT)


class Stop(Exception):
    """A filed stop: m67.json records why; the job exits 1."""


def check(out: dict[str, Any], name: str, ok: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"pass": ok, **detail}
    log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
    if not ok:
        out["stopped"] = name
        write(out)
        raise Stop(name)


def times(dW: ST.DW, c: float) -> ST.DW:
    return {k: v * c for k, v in dW.items()}


def pairs() -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, tuple[list[str], list[str]]], dict[str, Any]]:
    """Per pair, (reuse, new) in E136's state names and in E98's (k_half's); and one inventory with every task and state."""
    e136, e147 = load("experiments/share_untrained/results/inventory.json"), load("experiments/refit_96key/results/inventory.json")
    _, full = R96.inventories(e147)  # E136's inventory with the 96-key tasks and states added
    sd = e147["seeds"]
    names = {p["id"]: (p["reuse"], p["new"]) for p in e136["pairs"] if p["id"] in BUILT}
    names |= {p["id"]: ([R96.state_name(p["reuse"], s) for s in sd], [R96.state_name(p["new"], s) for s in sd]) for p in e147["pairs"] if p["id"] in BUILT}
    e98 = {k: (r["reuse"], r["new"]) for k, r in load("experiments/neuron_r/results/predictions.json")["rows"].items() if k in BUILT}
    e98 |= {k: v for k, v in names.items() if k.startswith("e58_")}  # E147's names, read by k_half_fill's States.e147
    assert set(names) == set(e98) == set(BUILT), (sorted(names), sorted(e98))
    used = {s for r, n in names.values() for s in r + n}
    tasks = {full["states"][s]["task"] for s in used}
    inv = {"tasks": {t: full["tasks"][t] for t in tasks}, "states": {s: full["states"][s] for s in used}, "pairs": []}
    return names, e98, inv


class Readers:
    """One model, every reader; `c` multiplies each update the holes_fill reader loads."""

    def __init__(self) -> None:
        self.names, self.e98_names, inv = pairs()
        self.e136 = E136.Job(inv)  # loads the model
        self.net: ST.Net = self.e136.net
        hf = load("experiments/holes_fill/results/predictions.json")
        self.hf = HF.Job(self.net, inv, hf["settings"], hf["thresholds"])
        k_m = load("experiments/share_untrained/results/stage0.json")["k_m"]  # as holes_fill's stage 0 sets them
        self.hf.ks = {"k64": {l: int(hf["settings"]["k_fixed"]) for l in HF.FS.LAYERS}, "kmatched": {l: int(k_m[key]) for l, key in enumerate(HF.DOWN)}}
        self.c, load_dW = 1.0, self.hf.dW
        self.hf.dW = lambda s: times(load_dW(s), self.c)  # every holes_fill read takes the update times c
        self.S = load("experiments/neuron_r/results/predictions.json")["settings"]
        self.identity_tol = load("experiments/neuron_r/results/predictions.json")["stage0"]["identity_tol"]
        self.kh = KH.States(self.net, self.S)
        s = load("experiments/steering_copy_lookup/results/predictions.json")["settings"]
        self.sc_cfg = SC.Cfg(s["half"], s["n"], tuple(s["strengths"]), s["n_random"], s["random_seed"])
        run62 = load("experiments/induction_cue/results/run.json")
        self.sc_pair, _ = SC.load_pair(self.sc_cfg, E62_LARGE, run62)
        self.sc_states = SC.load_states(E62_LARGE, run62)
        self.sc_w0 = SC.baselines(self.net, self.sc_pair)
        self.W0_norm = math.sqrt(sum(float(w.double().pow(2).sum()) for w in self.net.W0.values()))

    def size(self, dW: ST.DW) -> float:
        return math.sqrt(sum(float(v.double().pow(2).sum()) for v in dW.values())) / self.W0_norm

    def e98(self, name: str) -> tuple[ST.DW, Any, float | None, float]:
        """E98's (or E147's) update, rows, p_learn and saved r_half, through k_half_fill's loaders (sha256 checked)."""
        if name.startswith("e58/"):
            m, s = name.removeprefix("e58/").rsplit("_s", 1)
            return self.kh.e147(m, int(s))
        return self.kh.e98(name)

    def share(self, state: str, c: float) -> dict[str, float]:
        rs = self.e136.trained(state, times(self.e136.delta(state), c))
        return {"r": rs["shares"]["r"], "gain": rs["gain"]}

    def holes(self, state: str, c: float, i: int) -> dict[str, Any]:
        self.c = c
        rec, _ = self.hf.state(state, i)
        self.c = 1.0
        return {"gain": rec["gain"], "neuron_top": {k: v["share"] for k, v in rec["neuron"].items()},
                "steering": {k: v["share"] for k, v in rec.get("steering", {}).items()}}  # fmt: skip

    def e62_steering(self, state: str, c: float) -> float:
        """E135's pooled steering share (share_lo) of one E62 state times c; the cross-applied partner unscaled."""
        name = state.split("/")[1]
        side, seed = name.split("_")
        other = f"{'lookup' if side == 'copy' else 'copy'}_{seed}"
        dW = times(self.sc_states[name], c)
        g = SC.gains(self.net, self.sc_pair, {name: dW, other: self.sc_states[other]}, self.sc_w0)
        vs = SC.vectors(self.net, self.sc_pair, dW)
        gen = torch.Generator().manual_seed(self.sc_cfg.random_seed + SC.NAMES.index(name))  # E135's draw for this state
        rec = SC.kind_record(self.net, self.sc_pair, self.sc_cfg, side, other.split("_")[0], vs["pooled"], self.sc_w0, g[name], g[other], gen)
        return float(rec["share_lo"])

    def k_half(self, name: str, c: float) -> dict[str, Any]:
        dW, r, p_learn, _ = self.e98(name)
        rec = KH.carrying(self.net, r, times(dW, c), self.S, p_learn)
        keep = ("k_half", "why_none", "r_half", "r_half_num", "G", "identity", "telescope", "end_diff")
        return {k: rec[k] for k in keep}


# ----------------------------------------------------------------------------- stage 0


def sizes(rd: Readers, out: dict[str, Any]) -> None:
    """(a) every state's size, c and the larger side per pair; the same from E98's files; the two c agree."""
    tol = out["T"]["c_agree_rel"]
    for pair in BUILT:
        e136 = [[rd.size(rd.e136.delta(s)) for s in side] for side in rd.names[pair]]
        e98 = [[rd.size(rd.e98(s)[0]) for s in side] for side in rd.e98_names[pair]]
        larger = "reuse" if fmean(e136[0]) > fmean(e136[1]) else "new"
        big, small = (0, 1) if larger == "reuse" else (1, 0)
        c, c98 = fmean(e136[small]) / fmean(e136[big]), fmean(e98[small]) / fmean(e98[big])
        out["pairs"][pair] = {"sizes": {"reuse": e136[0], "new": e136[1]}, "sizes_e98": {"reuse": e98[0], "new": e98[1]},
                              "larger": larger, "c": c, "c_e98": c98}  # fmt: skip
        check(out, f"a c agrees {pair}", abs(c98 / c - 1) <= tol, {"c": c, "c_e98": c98, "tol": tol})
    write(out)


def reproduce(rd: Readers, out: dict[str, Any]) -> None:
    """(b) c = 1 through every reader, against the saved reading of the same state."""
    T = out["T"]
    st136 = load("experiments/share_untrained/results/stage1.json")["states"]
    hf1, r96 = load("experiments/holes_fill/results/stage1.json")["states"], load("experiments/refit_96key/results/stage1.json")["states"]
    for state, share_ref, hf_ref in (("e54/cipher_s0", st136["e54/cipher_s0"]["trained"], hf1["e54/cipher_s0"]),
                                     ("e58/cipher_s0", r96["e58/cipher_s0"]["e136"]["trained"], r96["e58/cipher_s0"]["holes_fill"])):  # fmt: skip
        got = rd.share(state, 1.0)["r"]
        check(out, f"b share {state}", abs(got - share_ref) <= T["share_tol"], {"got": got, "saved": share_ref, "tol": T["share_tol"]})
        h = rd.holes(state, 1.0, 0)
        for k, v in h["neuron_top"].items():
            ref = hf_ref["neuron"][k]["share"]
            check(out, f"b neuron {k} {state}", abs(v - ref) <= T["neuron_tol"], {"got": v, "saved": ref, "tol": T["neuron_tol"]})
        for k, v in h["steering"].items():
            ref = hf_ref["steering"][k]["share"]
            check(out, f"b steering {k} {state}", abs(v - ref) <= T["share_tol"], {"got": v, "saved": ref, "tol": T["share_tol"]})
    got = rd.e62_steering("e62/lookup_s0", 1.0)
    ref = load("experiments/steering_copy_lookup/results/stage1.json")["states"]["lookup_s0"]["steering"]["pooled"]["share_lo"]
    check(out, "b e62 steering lookup_s0", abs(got - ref) <= T["share_tol"], {"got": got, "saved": ref, "tol": T["share_tol"]})
    kh = load("experiments/k_half_fill/results/cells.json")["states"]
    for name in ("cipher_s0", "e58/cipher_s0"):
        k = rd.k_half(name, 1.0)
        ok = k["k_half"] == kh[name]["k_half"] and abs(k["r_half"] - kh[name]["r_half"]) <= T["neuron_tol"]
        check(out, f"b k_half {name}", ok, {"got": [k["k_half"], k["r_half"]], "saved": [kh[name]["k_half"], kh[name]["r_half"]]})


def random_floor(rd: Readers, out: dict[str, Any]) -> None:
    """(c) a Gaussian change at the larger side's first state's per-matrix norms times c: its gain on that side's rows."""
    gen = torch.Generator().manual_seed(out["T"]["random_seed"])
    for pair in BUILT:
        p = out["pairs"][pair]
        state = rd.names[pair][0 if p["larger"] == "reuse" else 1][0]
        R = {}
        for k, v in times(rd.e136.delta(state), p["c"]).items():
            g = torch.randn(v.shape, generator=gen).to(v.device)
            R[k] = g * (v.norm() / g.norm())
        t = rd.hf.task(rd.e136.inv["states"][state]["task"])
        gain = float((HF.lo_at(rd.net, t, t.read.toks, R) - t.lo0).mean())
        p["random"] = {"state": state, "size": rd.size(R), "gain": gain, "under_gain_floor": abs(gain) < MIN_GAIN}
        log(f"[0c random {pair}] {json.dumps(p['random'])}")
    write(out)


# ----------------------------------------------------------------------------- stage 1


def read_state(rd: Readers, state: str, c: float, i: int) -> dict[str, Any]:
    """One larger-side state times c, through every reader but k_half (read on E98's name by the caller)."""
    dW = rd.e136.delta(state)
    rec: dict[str, Any] = {"c": c, "size": rd.size(times(dW, c)), "share": rd.share(state, c), **rd.holes(state, c, i)}
    if state.startswith("e62/"):  # holes_fill reads no steering on E62's rows; E135's is the scorecard's
        rec["steering"] = {"pooled": rd.e62_steering(state, c)}
    er = [US.state(times(dW, x))["erank"] for x in (1.0, c)]
    rec["erank"] = {"unscaled": er[0], "scaled": er[1]}
    return rec


def stage1(rd: Readers, out: dict[str, Any], t_job: float) -> None:
    T, t0, n_new = out["T"], time.time(), 0
    todo = [(pair, s, n98) for pair in BUILT for s, n98 in larger(rd, out, pair)]
    for i, (pair, state, n98) in enumerate(todo):
        p = out["pairs"][pair]
        if state in p.get("states", {}):
            continue
        rec = read_state(rd, state, p["c"], i)
        rec["k_half"] = kh = rd.k_half(n98, p["c"])
        ok = kh["identity"] <= rd.identity_tol and abs(kh["telescope"]) <= rd.S["telescope_tol"] and abs(kh["end_diff"]) <= rd.S["grid_end_tol"]
        check(out, f"1 E98 identity, telescope, grid end {n98}", ok, {k: kh[k] for k in ("identity", "telescope", "end_diff")})
        e = rec["erank"]
        check(out, f"1 erank unchanged {state}", abs(e["scaled"] / e["unscaled"] - 1) <= T["erank_rel"], e)
        p.setdefault("states", {})[state] = rec | {"e98_name": n98}
        log(f"[1 {pair} {state}] {json.dumps(jsonable({k: v for k, v in rec.items() if k != 'k_half'} | {'k_half': kh['k_half']}))}")
        write(out)
        n_new += 1
        minutes = ((time.time() - t0) / n_new * (len(todo) - i - 1) + (time.time() - t_job)) / 60
        out["projected_minutes"] = minutes
        if n_new >= 4 and minutes > T["projected_limit_min"]:
            check(out, "1 projected time", False, {"minutes": minutes, "limit": T["projected_limit_min"]})
    write(out)


def larger(rd: Readers, out: dict[str, Any], pair: str) -> list[tuple[str, str]]:
    """The larger side's states: (E136 name, E98 name), in the inventory's order (both lists are)."""
    i = 0 if out["pairs"][pair]["larger"] == "reuse" else 1
    a, b = rd.names[pair][i], rd.e98_names[pair][i]
    assert len(a) == len(b), (pair, a, b)
    return list(zip(a, b, strict=True))


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if P["experiment"] == "E???":
        raise SystemExit("refusing to start: predictions.json has no experiment number")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every saved reading
    rd = Readers()
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"pairs": {}}
    out.pop("stopped", None)
    out |= {"card": torch.cuda.get_device_name(), "T": P["m67"]["thresholds"], "experiment": P["experiment"]}
    try:
        if not out.get("stage0_pass"):
            sizes(rd, out)
            reproduce(rd, out)
            random_floor(rd, out)
            out["stage0_pass"] = True
            write(out)
        print("STAGE 0 DONE", flush=True)
        stage1(rd, out, t_job)
        print("STAGE 1 DONE", flush=True)
    except Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
