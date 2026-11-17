"""P3.4 run 2, the reads of the saved states on the MATS cluster, every state on the same card type (L40), $0. Plan:
reports/plans/2026-11-12_plan_qwen-rl-readable-run2.md; rules in results/predictions_run2.json "controls" and "worked".

  --stage gpu --controls-only, then --stage gpu [--states R_s0@150,...]: the controls first and alone (W0, and E69's
        saved R update at step 100, seed 0), then each state of results/run2/chosen.json (or the named ones) not read
        yet: the output change and its multiple of the masked error (run 1's `output_change`), the greedy GSM8K
        pass rate and fits share on the 1319 test problems (run 1's `pass_rate`, fp32, TF32 off), and |δW|.
        Each state's file is checked against the sha256 its pod recorded first.
        Exit 3 when the time left is under 1.2 times the longest state so far: resubmit, done states skip.
  --stage share [--sites N]   CPU: the outside share of every read state against a Gaussian change of the same norms
        per matrix, and |δW| beside the decomposition's leftover norm, by E152's step-0 code unchanged
        (experiments/qwen_instrument_check/step0.py `site_record`, `table1`, `table2`); one line per matrix, resumable.
  --stage verdict   no model: R against N on problems that fit at both (paired), and the filed "worked" reading.
Results in results/run2_reads/ (tiny: results/run2_tiny/): controls.json, states/<pod>@<step>.json (one per state, so jobs
with different states run side by side), reads.log, share_sites.jsonl, share.json, verdict.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import match
import torch

from commute import qwen_d3
from commute import qwen_rl as q
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import load, plus, set_weights, sha256
from commute.qwen_vpd import SITES, decomp_id, parse_site

RUN1 = load_experiment("qwen_rl_readable", "run")  # output_change, pass_rate
RUN2 = load_experiment("qwen_rl_readable", "run2")
S0 = load_experiment("qwen_instrument_check", "step0")  # E152 step 0
JOB = RUN2.JOB
PRED = RUN2.PRED
E69_R100 = EXPERIMENTS / "qwen_rl/results/large/lr1e-6w10/dW/R/step_100.safetensors"
D = dict[str, Any]


@dataclass(frozen=True)
class Read:
    kl_batch: int = 4  # run 1's `output_change`


class Reader(JOB.Job):
    """E69's job (tokenizer, GSM8K, the fp32 model) writing this stage's files; GSM8K answers under <large>/reads/acc."""

    def __init__(self, size: str, large: Path, stage: str) -> None:
        self.pod, self.dir = Read(), out_dir(size)
        self.dir.mkdir(parents=True, exist_ok=True)
        super().__init__(JOB.config(size, RUN2.TAG), stage)
        self.large, self.shared_large = large / "reads", JOB.LARGE
        self.large.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (self.dir / "reads.log").open("a") as f:
            f.write(line + "\n")

    def where(self, name: str) -> Path:
        return self.dir / f"{name}.json"


def out_dir(size: str) -> Path:
    return RUN2.RESULTS / ("run2_reads" if size == "full" else "run2_tiny")


def states(size: str, large: Path, names: list[str] | None) -> dict[str, D]:
    """name -> {file, sha256}: the named states, or every kept step of results/run2/chosen.json, R and N first."""
    d = RUN2.results_dir(size)
    if names is None:
        c = JOB.read_json(d / "chosen.json")
        names = [f"{a}_s{s}@{t}" for a in RUN2.ARMS for s in c for t in c[s]["keep"]]
    out = {}
    for n in names:
        pod, t = n.split("@")
        rec = JOB.read_json(d / f"{pod}.json").get("saves", {}).get(t)
        if rec is None:
            sys.exit(f"{n}: not in {pod}.json saves")
        out[n] = {"file": large / rec["file"], "sha256": rec["sha256"]}
    return out


# ----------------------------------------------------------------------------- GPU reads


def read_one(job: Reader, name: str, path: Path | None, rows: torch.Tensor, w0: q.Records, refs: D) -> D:
    """One state, W0 when `path` is None: |δW|, the output change and the GSM8K pass rate, run 1's reads."""
    t0 = time.time()
    dW = load(path) if path is not None else {}
    norm = math.sqrt(sum(float(x.double().pow(2).sum()) for x in dW.values()))
    W = plus(job.W0, dW) if dW else None
    del dW
    with set_weights(job.model, W):
        kl = RUN1.output_change(job, rows)
        acc = RUN1.pass_rate(job, name, w0)
    del W
    torch.cuda.empty_cache()
    return {"dw_norm": norm, "kl": kl, "multiple": kl / refs["masked_error"], **acc, "seconds": time.time() - t0,
            "file": None if path is None else str(path)}


def controls(job: Reader, out: D, rows: torch.Tensor, w0: q.Records, P: D, wall_s: float) -> None:
    """W0 and E69's R at step 100 inside their filed bands (controls.json), before any state is read; a miss stops the
    stage: every later job refuses while controls.json has "stopped"."""
    for name, path in (("W0", None), ("e69_R100", E69_R100)):
        done = [r["seconds"] for r in out["controls"].values()]
        if name not in out["controls"] and done and time.time() - job.t0 + 1.2 * max(done) > wall_s:
            job.log("[control] time left under 1.2 x the W0 read: resubmit --controls-only (W0 is saved)")
            hard_exit(3)
        if name not in out["controls"]:
            out["controls"][name] = r = read_one(job, name, path, rows, w0, P["refs"])
            job.put("controls", out)
            job.log(f"[control] {name} {json.dumps(r, default=str)}")
        for key, band in P["controls"][name].items():
            got = out["controls"][name].get(key)
            job.check(out, f"control {name}: {key} in its filed band", q.in_band(got, band), {key: got, "band": band},
                      band=True)  # a stop at full size; logged only at the tiny size (8 GSM8K problems)
    out["done"] = True
    job.put("controls", out)


def stage_gpu(job: Reader, out: D, todo: dict[str, D], P: D, wall_s: float, named: bool) -> None:
    """The controls (a --controls-only job, first), else the states: one file each under states/, so jobs with different
    states run side by side. A state named with --states whose file is absent is an error (exit 1); from chosen.json it
    is skipped. A file whose sha256 differs from its pod's stops that state: its own file gets "stopped" (exit 2)."""
    job.fp32()
    rows, w0 = qwen_d3.maths_rows().tokens, JOB.load_rollouts(job.shared_large / "acc" / "W0_gsm8k.jsonl")
    if not todo:
        controls(job, out, rows, w0, P, wall_s)
    longest = max((r["seconds"] for r in out["controls"].values()), default=0.0)
    for name, s in todo.items():
        dest = job.dir / "states" / f"{name}.json"
        if dest.is_file():
            continue
        if not s["file"].is_file():
            if named:
                sys.exit(f"{name}: {s['file']} is missing (downloads land in results/run2_dl/<pod>/dW/)")
            job.log(f"[reads] {name}: {s['file']} not downloaded yet, skipped")
            continue
        if time.time() - job.t0 + 1.2 * longest > wall_s:
            job.log(f"[reads] time left under 1.2 x the longest state ({longest:.0f} s): resubmit")
            hard_exit(3)
        got = sha256(s["file"])
        if got != s["sha256"]:
            bad = {"stopped": "file sha256 differs from its pod's", "file": str(s["file"]), "sha256": got, "pod_sha256": s["sha256"]}
            JOB.write_text(dest, json.dumps(bad, indent=1))
            job.log(f"STOP: {name} {json.dumps(bad)}")
            hard_exit(2)
        r = read_one(job, name, s["file"], rows, w0, P["refs"])
        longest = max(longest, r["seconds"])
        JOB.write_text(dest, json.dumps(r, indent=1, default=float))
        job.log(f"[reads] {name} {json.dumps(r, default=str)}")
    job.log("STAGE gpu DONE")


def read_states(d: Path) -> dict[str, D]:
    """The read states, without any stopped on its sha256 check."""
    st = {p.stem: json.loads(p.read_text()) for p in sorted((d / "states").glob("*.json"))}
    return {n: r for n, r in st.items() if "stopped" not in r}


# ----------------------------------------------------------------------------- CPU: outside share


def site_lines(path: Path) -> dict[str, D]:
    """Per-matrix records written so far; a line cut off by a killed job is dropped."""
    done = {}
    for line in path.read_text().splitlines() if path.is_file() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[r["site"]] = r
    return done


def stage_share(d: Path, files: dict[str, Path], n_sites: int, wall_s: float, band: list[float]) -> None:
    """E152 step 0's Table 1 (outside share against its Gaussian null, pooled by each update's |δW_s|²) and Table 2
    (|δW| beside the leftover norm) for the read states; per-matrix lines in share_sites.jsonl."""
    t0, log = time.time(), d / "share_sites.jsonl"
    assert decomp_id(S0.DECOMP) == S0.NEW, S0.DECOMP
    done = site_lines(log)
    assert all(set(r["updates"]) == set(files) for r in done.values()), "share_sites.jsonl read another state set"
    comps, w0 = S0.Tensors(sorted(S0.DECOMP.glob("*.safetensors"))), S0.Tensors(sorted(S0.W0_DIR.glob("model-*.safetensors")))
    dws, rel, alive = {n: S0.Tensors([p]) for n, p in files.items()}, S0.recon_errors(), S0.alive_sets()
    sites, checked, longest = SITES[:n_sites], set(), 0.0
    for site in sites:
        if site in done:
            continue
        if time.time() - t0 + 1.5 * longest > wall_s:
            print(f"time left under 1.5 x the longest matrix ({longest:.0f} s): resubmit", flush=True)
            hard_exit(3)
        kind = parse_site(site)[1]
        rec = S0.site_record(site, comps, w0, dws, alive[site], rel[site], kind not in checked)
        checked |= {kind} if rec["checked"] else set()
        longest = max(longest, rec["seconds"])
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[site] = rec
        print(f"{site}: {rec['seconds']:.0f} s", flush=True)
    recs, names = [done[s] for s in sites], list(files)
    out = {"table1": S0.table1(recs, names), "table2": S0.table2(recs, names), "sites": len(recs)}
    if len(recs) == len(SITES):  # the control: E152's ratio for the same file, by the same code
        got = out["table1"]["updates"]["e69_R100"]["ratio"]
        out["control"] = {"ratio": got, "band": band, "holds": q.in_band(got, band)}
        print(f"[control] e69_R100 outside share / null {got}: {'ok' if out['control']['holds'] else 'MISSED'} {band}")
    JOB.write_text(d / "share.json", json.dumps(out, indent=1))
    print(json.dumps({n: {"share_ratio": out["table1"]["updates"][n]["ratio"], "norm": out["table2"]["updates"][n]["norm"]}
                      for n in names} | {"leftover_norm": out["table2"]["delta_total"]}, indent=1))
    print("STAGE share DONE", flush=True)


# ----------------------------------------------------------------------------- verdict


def stage_verdict(size: str, large: Path, P: D) -> None:
    """R against N of the same seed and step on problems that fit at both (`qwen_rl.gain`, paired), then `match.worked`."""
    d, acc = out_dir(size), large / "reads" / "acc"
    st, c = read_states(d), JOB.read_json(RUN2.results_dir(size) / "chosen.json")
    for n, r in st.items():
        other = n.replace("R_", "N_", 1)
        if n.startswith("R_") and other in st:
            r["vs_N"] = q.gain(*(JOB.load_rollouts(acc / f"{x}_gsm8k.jsonl") for x in (other, n)))
    steps = {s: {k: v["step"] for k, v in c[s]["targets"].items() if v["status"] == "confirmed"} for s in c}
    out = {"steps": steps, "worked": match.worked(steps, st, P["match"]["targets"]),
           "states": {n: {k: r.get(k) for k in ("multiple", "pass", "fits", "dw_norm", "vs_N")} for n, r in st.items()}}
    JOB.write_text(d / "verdict.json", json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("gpu", "share", "verdict"))
    ap.add_argument("--size", default="full", choices=("full", "tiny"))
    ap.add_argument("--large", type=Path, default=RUN2.RESULTS / "run2_dl",
                    help="the states' root: the runner's downloads, <root>/<pod>/dW/step_<t>.safetensors (tiny: run2.py --large)")
    ap.add_argument("--states", default=None, help="comma-separated <pod>@<step>; default: every kept step in chosen.json")
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--sites", type=int, default=len(SITES), help="share: the first N matrices only (the rehearsal)")
    ap.add_argument("--wall-min", type=float, default=28.0)
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if a.stage == "verdict":
        stage_verdict(a.size, a.large, P)
        return
    if a.stage == "share":
        st = read_states(out_dir(a.size))
        files = {"e69_R100": E69_R100} | {n: Path(r["file"]) for n, r in st.items()}  # the control: E152 read it as R_s0
        stage_share(out_dir(a.size), files, a.sites, a.wall_min * 60, P["controls"]["e69_R100_share_ratio"])
        hard_exit(0)
    job = Reader(a.size, a.large, "controls")
    out = job.get("controls") or {"controls": {}}
    if out.get("stopped"):
        sys.exit(f"the reads stopped at {out['stopped']!r}: the lead decides")
    if not a.controls_only and not out.get("done"):
        sys.exit("the controls are not read yet: submit --controls-only first")
    todo = {} if a.controls_only else states(a.size, a.large, a.states.split(",") if a.states else None)
    stage_gpu(job, out, todo, P, a.wall_min * 60, bool(a.states))
    hard_exit(0)


if __name__ == "__main__":
    main()
