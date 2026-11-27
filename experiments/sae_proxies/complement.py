"""E317 (cell P2.85 run 2): the positive control for E314's error share, its complement. E314's run.py, results.json and
predictions.json stay as they ran; this file reuses run.py's Reader (rows, runs, the error patch) and adds one patch.
Settings and expected values: results/predictions_complement.json (refused unless committed unchanged; E314's
predictions.json is required unchanged too, since the Reader is built from its settings).

Reconstruction share. At the 8 block-output sites the base model runs with each site's output set to
hat(o_FT) + e(o_cur), computed as o_cur + (hat(o_FT) - hat(o_cur)) (the same sum, and exactly o_cur when o_FT is the
base's own output); all 8 at once, every position. recon_share = (L_recon - T0) / (T1 - T0), the floors of the error
share (a trained state under |gain| 1e-6 nan; a random update under |gain| 1 log-odds null with the reason).
sum_share = error_share + recon_share: 1 if the two parts add. The error share is read again in the same job and must
equal E314's saved value per state within the run-to-run floor stage 0 measures.

  stage 0  the zero update (both numerators exactly 0, both shares nan); the fine-tuned model run twice (floor); the
           full patch (o_FT at all 8 sites) reproduces T1 within the floor.
  stage 1  the 48 states and their random updates (E314's seeds), written to results/results_complement.json as each
           lands; a rerun resumes.
Prints every scored number, "STAGE n DONE"; a miss prints "STOP: ..." and exits 1.

    sbatch --comment=E317 cluster/sae_proxies_complement.sbatch
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute.finetune import random_direction
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

R = load_experiment("sae_proxies", "run")  # Reader, states, check, Stop, stable_seed
PRED, OUT = R.RESULTS / "predictions_complement.json", R.RESULTS / "results_complement.json"
E314_OUT = "experiments/sae_proxies/results/results.json"


@dataclass(frozen=True)
class Config:
    gain_floor: float
    random_gain_min: float

    def __post_init__(self) -> None:
        assert (self.gain_floor, self.random_gain_min) == (1e-6, 1.0), "the error share's floors (E314)"


def write(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(OUT)


def hat(rd: Any, site: str, o: Tensor) -> Tensor:
    ae = rd.saes[site]
    return ae.decode(ae.encode(o))


def recon_patch(rd: Any, hat_ft: dict[str, list[Tensor]], i: int) -> dict[str, Any]:
    return {s: (lambda o, s=s: o + (hat_ft[s][i] - hat(rd, s, o))) for s in R.OUTER}


def share(num: float, gain: float, floor: float, random: bool) -> tuple[float | None, str | None]:
    """num / gain; under the floor nan (a trained state) or None with the reason (a random update)."""
    if abs(gain) >= floor:
        return num / gain, None
    return (None, f"|gain| {abs(gain):.3g} under {floor:g} log-odds") if random else (math.nan, None)


def read(rd: Any, r: Any, dW: dict[str, Tensor], floor: float, random: bool) -> dict[str, Any]:
    """Both patches of one update on one task's rows, on the same forwards."""
    base, ft = rd.run(r, None), rd.run(r, dW, keep=R.OUTER)
    e_ft = {s: [rd.err(s, o) for o in ft.acts[s]] for s in R.OUTER}
    h_ft = {s: [hat(rd, s, o) for o in ft.acts[s]] for s in R.OUTER}
    l_err, l_rec = rd.run(r, None, edit=lambda i: rd.patch(e_ft, i)).lo, rd.run(r, None, edit=lambda i: recon_patch(rd, h_ft, i)).lo
    gain = ft.lo - base.lo
    es, why = share(l_err - base.lo, gain, floor, random)
    rs, _ = share(l_rec - base.lo, gain, floor, random)
    rec = {"T0": base.lo, "T1": ft.lo, "gain": gain, "L_patch": l_err, "L_recon": l_rec, "error_share": es, "recon_share": rs,
           "sum_share": None if es is None or rs is None else es + rs}  # fmt: skip
    return rec | ({"share_why_none": why} if why else {})


def stage0(rd: Any, sts: list[Any], c: Config, out: dict[str, Any]) -> float:
    if out.get("stage0", {}).get("pass"):
        log("[stage 0] already passed")
        return out["stage0"]["floor"]["max_abs_logit_ft_twice"]
    res: dict[str, Any] = {"host": socket.gethostname(), "card": torch.cuda.get_device_name(), "checks": {}}
    out["stage0"], ch, st = res, res["checks"], sts[0]
    r = rd.reads[st.rows]
    try:
        z = res["zero_update"] = read(rd, r, {k: torch.zeros_like(w) for k, w in rd.net.W0.items()}, c.gain_floor, random=False)
        R.check(ch, "zero update gain", z["gain"], 0.0)
        R.check(ch, "zero update recon numerator L_recon - T0", z["L_recon"] - z["T0"], 0.0)
        R.check(ch, "zero update error numerator L_patch - T0", z["L_patch"] - z["T0"], 0.0)
        R.check(ch, "zero update both shares nan (0 = nan)", 0.0 if math.isnan(z["recon_share"]) and math.isnan(z["error_share"]) else 1.0, 0.0)
        dW = rd.delta(st)
        base, a, b = rd.run(r, None), rd.run(r, dW, keep=R.OUTER), rd.run(r, dW)
        floor = float((a.at_half - b.at_half).abs().max())
        full = rd.run(r, None, edit=lambda i: {s: (lambda _o, v=a.acts[s][i]: v) for s in R.OUTER})
        res["floor"] = {"state": st.name, "max_abs_logit_ft_twice": floor}
        res["full_patch"] = {"state": st.name, "share": (full.lo - base.lo) / (a.lo - base.lo), "max_abs_logit_vs_ft": float((full.at_half - a.at_half).abs().max())}
        log(f"[0 floor] {json.dumps(res['floor'])}\n[0 full patch] {json.dumps(res['full_patch'])}")
        R.check(ch, f"full patch max |logit - FT logit| {st.name} (tol = run-to-run floor)", res["full_patch"]["max_abs_logit_vs_ft"], floor)
    except R.Stop as e:
        res["stopped"] = str(e)
        write(out)
        raise
    res["pass"] = True
    write(out)
    return floor


def saved_share(v: Any) -> float | None:
    return None if v is None else float(v)  # a saved nan is the string "nan"


def same_as_e314(new: float | None, saved: float | None) -> float:
    """|new - saved|; 0 when both are missing alike (None, or nan), inf when only one is."""
    if new is None or saved is None:
        return 0.0 if new is None and saved is None else math.inf
    if math.isnan(new) or math.isnan(saved):
        return 0.0 if math.isnan(new) and math.isnan(saved) else math.inf
    return abs(new - saved)


HEAD = ("gain", "error_share", "recon_share", "sum_share")


def stage1(rd: Any, sts: list[Any], c: Config, floor: float, out: dict[str, Any]) -> None:
    e314 = R.load(E314_OUT)["states"]
    out.setdefault("states", {})
    for k in ("stopped", "stopped_state"):
        out.pop(k, None)
    t0, n_new = time.time(), 0
    for st in sts:
        if st.name in out["states"]:
            continue
        r, dW, seed = rd.reads[st.rows], rd.delta(st), R.stable_seed(st.name)
        assert e314[st.name]["random"]["seed"] == seed, st.name
        rec = read(rd, r, dW, c.gain_floor, random=False)
        rnd = read(rd, r, random_direction(dW, seed), c.random_gain_min, random=True) | {"seed": seed}
        diffs = {"trained": same_as_e314(rec["error_share"], saved_share(e314[st.name]["error_share"])),
                 "random": same_as_e314(rnd["error_share"], saved_share(e314[st.name]["random"]["error_share"]))}  # fmt: skip
        meta = {"task": st.task, "seed": st.seed, "side": st.side, "regating_share": st.regating_share, "rows": st.rows}
        full = meta | rec | {"random": rnd, "error_share_vs_e314": diffs, "checks": {}}
        log(f"[1 {st.name}] {json.dumps(jsonable(meta | {k: rec[k] for k in HEAD} | {'random': {k: rnd[k] for k in HEAD}, 'error_share_vs_e314': diffs}))}")
        log(f"[1 {st.name} all] {json.dumps(jsonable(full))}")
        try:
            for k, d in diffs.items():
                R.check(full["checks"], f"error share {k} {st.name} vs E314 (tol = run-to-run floor)", d, floor)
        except R.Stop as e:
            out["stopped"], out["stopped_state"] = str(e), full  # kept out of "states", so a rerun reads it again
            raise
        out["states"][st.name] = full
        write(out)
        n_new += 1
        left = sum(s.name not in out["states"] for s in sts)
        log(f"[1 time] {len(out['states'])} of {len(sts)}; {(time.time() - t0) / n_new * left / 60:.1f} min left")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    require_filed(R.PRED)
    require_filed(PRED)
    P, P314 = json.loads(PRED.read_text()), json.loads(R.PRED.read_text())
    assert P["experiment"] == "E317" and P314["experiment"] == "E314"
    c, c314 = config_from_settings(Config, P["settings"]), config_from_settings(R.Config, P314["settings"])
    sts = R.states()
    if missing := [f"{s.name}: {s.file}" for s in sts if not (R.REPO / s.file).is_file()]:
        print(f"STOP: update files missing: {missing}", flush=True)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": "E317"}
    t_job = time.time()
    with torch.no_grad():
        rd = R.Reader(c314, sts)
        log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()} states {len(sts)} row sets {len(rd.reads)}")
        try:
            floor = stage0(rd, sts, c, out)
            print("STAGE 0 DONE", flush=True)
            stage1(rd, sts, c, floor, out)
        except R.Stop as e:
            write(out)
            print(f"STOP: {e}: tell the lead", flush=True)
            return 1
    print("STAGE 1 DONE", flush=True)
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, {(time.time() - t_job) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    hard_exit(main())
