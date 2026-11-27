"""E318 (cell P2.88): Fig 3's random reference for E314's error share. E314's and E317's files stay as they ran; this
file reuses run.py's Reader and complement.py's read (error share and reconstruction share on the same forwards).
Settings and expected values: results/predictions_random.json (refused unless committed unchanged).

The random updates are E302's (experiments/random_logodds_share, the random line of the paper's known-answer figure),
rebuilt exactly: per fine-tune state and E228 seed, finetune.random_direction(dW, seed) from the fp32 update on the CPU,
times the draw's saved signed_c (results/states/<state>.json .draws.<seed>). E302 scaled each draw until it raised the
trained answer by 1.00-1.10 log-odds (bf16 reader); 27 of 144 draws reached it ("ok"), the rest hit the 100x cap.
Here each draw is read in fp32 on E136's rows through complement.read: gain, error share and reconstruction share.
A share is reported for the ok draws (Fig 3's set) over the fp32 gain read here, flagged `under_floor` if that gain is
under 1 log-odds; a capped draw is read too, its shares null with the reason.

  stage 0  alone: (a) the first state's fine-tune read again gives E314's saved error share (tolerance 0: E317 read it
           bit for bit); (b) the first ok draw's rebuilt size |c d| / |dW| equals E302's size_rel within 1e-9
           relative (same code on the same file; a wrong file or seed is far off). A miss prints STOP and exits 1.
  stage 1  the ok draws, then the capped ones, each written to results/results_random.json as it lands; a rerun resumes.
           Per draw E302's bf16 gain is kept beside the fp32 gain read here (recorded, no stop).

    sbatch --comment=E318 cluster/sae_proxies_random.sbatch
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any

import torch

from commute.finetune import random_direction
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

R = load_experiment("sae_proxies", "run")  # Reader, states, check, Stop, load, REPO
C = load_experiment("sae_proxies", "complement")  # read: both shares on the same forwards
PRED, OUT = R.RESULTS / "predictions_random.json", R.RESULTS / "results_random.json"
E302 = "experiments/random_logodds_share/results/states"
E314_OUT = "experiments/sae_proxies/results/results.json"


@dataclass(frozen=True)
class Config:
    random_gain_min: float
    size_rel_tol: float

    def __post_init__(self) -> None:
        assert self.random_gain_min == 1.0, "the error share's random floor (E314)"


def log(msg: str) -> None:
    print(msg, flush=True)


def write(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(OUT)


def norm(d: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))


def draws(sts: list[Any]) -> list[tuple[Any, str, dict[str, Any]]]:
    """(state, seed, E302's draw record), ok draws first, each group in state order."""
    out = []
    for st in sts:
        for seed, d in R.load(f"{E302}/{st.name.replace('/', '_')}.json")["draws"].items():
            out.append((st, seed, d))
    assert len(out) == 3 * len(sts), len(out)
    return sorted(out, key=lambda t: t[2]["status"] != "ok")


def rebuild(rd: Any, st: Any, seed: str, d: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float]:
    """E302's change, c times the draw on the CPU fp32 update, and its size relative to the update."""
    dW = rd.delta(st)
    change = {k: d["signed_c"] * v for k, v in random_direction(dW, int(seed)).items()}
    return change, norm(change) / norm(dW)


def stage0(rd: Any, sts: list[Any], ds: list[Any], c: Config, out: dict[str, Any]) -> None:
    if out.get("stage0", {}).get("pass"):
        log("[stage 0] already passed")
        return
    res: dict[str, Any] = {"host": socket.gethostname(), "card": torch.cuda.get_device_name(), "checks": {}}
    out["stage0"], ch, st = res, res["checks"], sts[0]
    try:
        again = C.read(rd, rd.reads[st.rows], rd.delta(st), 1e-6, random=False)["error_share"]
        saved = float(R.load(E314_OUT)["states"][st.name]["error_share"])
        res["reread"] = {"state": st.name, "error_share": again, "e314": saved}
        R.check(ch, f"error share of {st.name} read again minus E314's", again - saved, 0.0)
        st1, seed, d = ds[0]
        _, size = rebuild(rd, st1, seed, d)
        res["rebuild"] = {"state": st1.name, "seed": seed, "size_rel": size, "e302": d["size_rel"]}
        R.check(ch, f"rebuilt size_rel of {st1.name} {seed} relative to E302's", size / d["size_rel"] - 1, c.size_rel_tol)
    except R.Stop as e:
        res["stopped"] = str(e)
        write(out)
        raise
    res["pass"] = True
    write(out)


HEAD = ("gain", "gain_e302", "error_share", "recon_share", "sum_share")


def stage1(rd: Any, ds: list[Any], c: Config, out: dict[str, Any]) -> None:
    out.setdefault("draws", {})
    t0, n_new = time.time(), 0
    for st, seed, d in ds:
        key = f"{st.name} {seed}"
        if key in out["draws"]:
            continue
        change, size = rebuild(rd, st, seed, d)
        # an ok draw is Fig 3's reference: its shares over its own fp32 gain, flagged if that fell under the floor
        rec = C.read(rd, rd.reads[st.rows], change, 1e-6, random=False)
        rec["under_floor"] = abs(rec["gain"]) < c.random_gain_min
        if d["status"] != "ok":  # capped in E302: not in Fig 3's reference, shares null whatever the gain
            rec |= {"error_share": None, "recon_share": None, "sum_share": None, "share_why_none": f"E302 status {d['status']}"}
        meta = {"state": st.name, "task": st.task, "seed": st.seed, "side": st.side, "draw_seed": int(seed), "status": d["status"],
                "signed_c": d["signed_c"], "size_rel": size, "gain_e302": d["gain"], "regating_share_e302": d["r"]}  # fmt: skip
        out["draws"][key] = meta | rec
        write(out)
        n_new += 1
        log(f"[1 {key}] {json.dumps(jsonable({k: out['draws'][key].get(k) for k in ('status', *HEAD)}))}")
        log(f"[1 {key} all] {json.dumps(jsonable(out['draws'][key]))}")
        left = sum(f"{s.name} {k}" not in out["draws"] for s, k, _ in ds)
        log(f"[1 time] {len(out['draws'])} of {len(ds)}; {left * (time.time() - t0) / n_new / 60:.1f} min left")


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"] == "E318"
    c = config_from_settings(Config, P["settings"])
    e314 = config_from_settings(R.Config, json.loads(R.PRED.read_text())["settings"])
    sts = R.states()
    ds = draws(sts)
    log(f"[draws] {len(ds)}, ok {sum(d['status'] == 'ok' for _, _, d in ds)}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": "E318"}
    with torch.no_grad():
        rd = R.Reader(e314, sts)
        log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()}")
        try:
            stage0(rd, sts, ds, c, out)
        except R.Stop as e:
            print(f"STOP: {e}: tell the lead", flush=True)
            return 1
        print("STAGE 0 DONE", flush=True)
        stage1(rd, ds, c, out)
    print("STAGE 1 DONE", flush=True)
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, {(time.time() - t_job) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    hard_exit(main())
