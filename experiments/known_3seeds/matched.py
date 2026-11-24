"""E303 (cell P2.77), stage 3 of known_3seeds: E187's size-matched share and pooled steering share of E299's seed-2
states, so that the matched series of Fig 3a/4 reads the same states as the unscaled one (seeds 0-2 a side, one variant a
side: known3.json). Card ~/commute-data/cards/E303.md (reports/predictions/2026-11-24_card_p277-matched-seed2.md).
Descriptive: no verdict on reuse.

E187's rule (size_matched/run.py docstring): per pair only the larger side is read again, every one of its updates times
the pair's c; the smaller side keeps its unscaled readings. c and the larger side are E187's, as saved in
size_matched/results/m67.json .pairs.<pair>.{c,larger}. Each new state is read once per pair on which it is a larger-side
state: e46/rand32_s2 (e46_copy_random, c 0.66), e53/cipher_s2 (e53_lowercase_cipher, c 0.90), e54/cipher_own_s2
(e54_own, c 0.52). e46/ident32_s2 and e53/lowercase_s2 are on smaller sides, so their matched reading is their unscaled
one (stage2.json, via known3.json); they are listed in "smaller" and not read. The readers are E187's own, called as
size_matched calls them (Readers.share: E136's trained share; Readers.holes: E143's fair steering, pooled and per key);
holes_fill's neuron reads are skipped (no k set), as Fig 3a/4 draw no neuron reading.

  control, alone: e46/rand32_s0 and e53/cipher_s0 (larger sides: an fp32 key-value update and a bf16 E53 update) read
           again at their pair's c against m67.json's saved share r and pooled steering share. Tolerance: E299's
           reader_share_tol (results/predictions.json, 0.01, its stage 0 bar on the E136 share and E143 steering,
           measured 0 there); a miss writes, prints STOP and exits 1.
  then the three states, each written to results/stage3.json as it lands (a rerun resumes); every number is logged.

    sbatch --comment=E303 cluster/known_3seeds_matched.sbatch
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import torch

from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage3.json"
K3 = load_experiment("known_3seeds", "run")  # inventories, NEW, Config, PRED, load, write
A = load_experiment("known_3seeds", "assemble")  # PAIRS: the task a side per pair
SM = K3.SM  # size_matched run.py: Readers.share, Readers.holes, times, E136, HF
M67 = "experiments/size_matched/results/m67.json"
CONTROL = {"e46/rand32_s0": "e46_copy_random", "e53/cipher_s0": "e53_lowercase_cipher"}


def log(msg: str) -> None:
    print(msg, flush=True)


def targets() -> tuple[list[tuple[str, str]], list[str]]:
    """(pair, state) for every new state on a pair's larger side; the new states on a smaller side, as 'pair state'."""
    m67, read, smaller = K3.load(M67)["pairs"], [], []
    for state in sorted(K3.NEW):
        for pair, tasks in A.PAIRS.items():
            for side, task in zip(("reuse", "new"), tasks, strict=True):
                if state.rsplit("_s", 1)[0] != task:
                    continue
                if m67[pair]["larger"] == side:
                    read.append((pair, state))
                else:
                    smaller.append(f"{pair} {state}")
    return read, smaller


class Scaled:
    """E187's two readers on E299's states. `c` multiplies each update holes_fill loads, as size_matched Readers sets up."""

    share = SM.Readers.share  # E136 Job.trained of the update times c: {"r", "gain"}
    holes = SM.Readers.holes  # holes_fill Job.state of the update times c: {"gain", "neuron_top", "steering"}

    def __init__(self, states: list[str]) -> None:
        _, full = K3.inventories()
        st = {s: full["states"][s] for s in states}
        inv = {"tasks": {s["task"]: full["tasks"][s["task"]] for s in st.values()}, "states": st, "pairs": []}
        self.e136 = SM.E136.Job(inv)  # loads the model
        hf = K3.load(f"{K3.E143_R}/predictions.json")
        self.hf = SM.HF.Job(self.e136.net, inv, hf["settings"], hf["thresholds"])  # ks left empty: no neuron reads
        self.c, load_dW = 1.0, self.hf.dW
        self.hf.dW = lambda s: SM.times(load_dW(s), self.c)

    def read(self, state: str, c: float) -> dict[str, Any]:
        """One state times c, in m67.json's per-state form."""
        return {"c": c, "share": self.share(state, c), **self.holes(state, c, 0)}


def control(rd: Scaled, out: dict[str, Any], tol: float) -> None:
    m67 = K3.load(M67)["pairs"]
    for state, pair in CONTROL.items():
        saved = m67[pair]["states"][state]
        got = rd.read(state, m67[pair]["c"])
        diffs = {"share r": got["share"]["r"] - saved["share"]["r"], "steering pooled": got["steering"]["pooled"] - saved["steering"]["pooled"]}
        ok = all(abs(d) <= tol for d in diffs.values())  # nan misses
        out["control"][state] = {"pair": pair, "got": got, "saved_share_r": saved["share"]["r"], "saved_steering_pooled": saved["steering"]["pooled"],
                                 "diff": diffs, "tol": tol, "pass": ok}  # fmt: skip
        log(f"[control {state} c {m67[pair]['c']:.6g}] {json.dumps(jsonable(out['control'][state]))}")
        if not ok:
            out["stopped"] = f"control {state}: {diffs} over {tol}"
            K3.write(OUT.name, out)
            raise K3.Stop(out["stopped"])


def main() -> int:
    t0 = time.time()
    require_filed(K3.PRED)
    tol = config_from_settings(K3.Config, json.loads(K3.PRED.read_text())["settings"]).reader_share_tol
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every saved reading
    todo, smaller = targets()
    todo.sort(key=lambda t: not t[1].startswith("e53/"))  # after the e53 control: E136 keeps one task's gates at a time
    m67 = K3.load(M67)["pairs"]
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"control": {}, "pairs": {}}
    if "stopped" in out:
        print(f"STOP: stage3.json records a stop ({out['stopped']}): move it aside to start again", flush=True)
        return 1
    rd = Scaled(sorted(CONTROL) + [s for _, s in todo])
    out |= {"experiment": "E303", "host": socket.gethostname(), "card": torch.cuda.get_device_name(), "smaller": smaller, "m67": M67}
    log(f"[host] {out['host']} {out['card']}; to read {todo}; smaller sides, unscaled reading kept: {smaller}")
    try:
        if not all(out["control"].get(s, {}).get("pass") for s in CONTROL):
            control(rd, out, tol)
            K3.write(OUT.name, out)
    except K3.Stop as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print("CONTROL DONE", flush=True)
    for pair, state in todo:
        p = out["pairs"].setdefault(pair, {"c": m67[pair]["c"], "larger": m67[pair]["larger"], "states": {}})
        if state in p["states"]:
            continue
        p["states"][state] = rec = rd.read(state, p["c"])
        log(f"[3 {pair} {state} c {p['c']:.6g}] {json.dumps(jsonable(rec))} ({(time.time() - t0) / 60:.1f} min)")
        K3.write(OUT.name, out)
    out["pass"] = True
    K3.write(OUT.name, out)
    print("STAGE 3 DONE", flush=True)
    log(f"[done] {(time.time() - t0) / 60:.1f} min; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
