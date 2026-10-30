"""keydependent_patch (E56): does an input-dependent cheap baseline, E52's neuron patching (B2), tell E54's elicit map
from its matched cipher where steering could not; and does r's lead over steering hold at five seeds? Plan:
reports/plans/2026-10-30_plan_baseline-next-draft.md (approved 2026-10-30). Every setting, rule and threshold:
results/predictions.json (filed before the run, read here, never edited).

  run.py --job 1   E54's six saved states ({elicit,cipher,cipher_own}_s{0,1}).
                   stage 0, alone (exit 1 on a miss): sha256 against E54's stage2.json; E52's identity patch and full
                   residual patch; r and S reproduce stage2.json and S_lo reproduces amend1.json.
                   then B2 (N16, k50) per state and P1 on two seeds (interim)               -> results/job1.json
  run.py --job 2   refuses unless job1.json passed stage 0, P1 held there, and job1.json and predictions.json are
                   committed unchanged.
                   stage 0, alone (exit 1 on a miss): the windows files and W0's elicit rate reproduce E54's stage1.json;
                   seed-0 elicit r reproduces stage2.json.
                   then per seed 2, 3, 4: E54's elicit and cipher fine-tunes (the cipher read at the step matching the
                   elicit update norm, and at its own step), elicit and matched cipher saved to <large>/; r, S, S_lo,
                   N16 and k50 on all three states; then P1 and P2 on five seeds, P3 on seeds 2-4 -> results/job2.json
Results are written after every state; every scored number is also printed. The target model and its decomposition,
fp32, TF32 off.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from commute import elicit as EL
from commute import keydep as KD
from commute import keydep_score as KS
from commute import neurons as NE
from commute import steering as ST

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E54 = HERE.parent / "keydependent_elicit" / "results"
KINDS = ("elicit", "cipher", "cipher_own")
OLD, NEW = ["s0", "s1"], ["s2", "s3", "s4"]
log = KD.log


def dump(name: str, obj: dict[str, Any]) -> None:
    fin = lambda o: (str(o) if isinstance(o, float) and not math.isfinite(o) else
                     {str(k): fin(v) for k, v in o.items()} if isinstance(o, dict) else
                     [fin(v) for v in o] if isinstance(o, list | tuple) else o)
    (RESULTS / name).write_text(json.dumps(fin(obj), indent=1, sort_keys=True))


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


def check_committed(*paths: Path) -> None:
    for p in paths:
        rel = str(p.relative_to(REPO))
        if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
            sys.exit(f"{rel} is not committed unchanged: commit it first")


def task_of(tasks: dict[str, ST.Task], n: str) -> ST.Task:
    return tasks["elicit" if n.startswith("elicit") else "cipher"]


# ----------------------------------------------------------------------------- the readings


def readout(ctx: KD.Ctx, t: ST.Task, dW: ST.DW, g: dict[str, float], base: dict[str, Any]) -> dict[str, Any]:
    """E54's r and S (keydep.readings) and Amendment 1's S_lo at B3's block."""
    rd = KD.readings(ctx, t, dW, g, base)
    lo = ST.steer_at(ctx.net, t, dW, rd["layer"], ST.logodds_at)
    return {"r": rd["r"], "S": rd["S"], "layer": rd["layer"], "S_block0": rd["S_block0"], "G_logodds": rd["G_logodds"],
            "S_lo": EL.share_or_nan(lo["share"], lo["G"]), "G_lo_score": lo["G"]}


def b2(net: ST.Net, t: ST.Task, dW: ST.DW, G_score: float, grid: list[int]) -> dict[str, Any]:
    """E52's B2 on one state: N16 = the top-16 share (nan when G_score is within 1e-6 of 0), k50 and its grid index."""
    p = NE.neuron_patch(net, t, dW, G_score, grid)
    return {"N16": EL.share_or_nan(p["shares"][16], G_score), "k50": p["k50"], "k50_index": p["k50_index"],
            "b2_shares": p["shares"], "top16": p["order"][:16].tolist()}


def say(n: str, st: dict[str, Any]) -> None:
    keys = ("G", "G_score", "G_logodds", "r", "S", "S_lo", "layer", "N16", "k50", "k50_index", "readable")
    log(f"[{n}] " + " ".join(f"{k} {st[k]:.12g}" if isinstance(st[k], float) else f"{k} {st[k]}" for k in keys if k in st))


def read_new(ctx: KD.Ctx, t: ST.Task, dW: ST.DW, base: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    g = ST.gains(ctx.net, t, dW)
    st: dict[str, Any] = {**g, "readable": abs(g["G"]) >= P["settings"]["floor_min"]}
    if st["readable"]:
        st.update(readout(ctx, t, dW, g, base))
        st.update(b2(ctx.net, t, dW, g["G_score"], P["settings"]["b2_grid"]))
    return st


# ----------------------------------------------------------------------------- job 1


def stage0_job1(ctx: KD.Ctx, tasks: dict[str, ST.Task], dws: dict[str, ST.DW], bases: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    th = P["stage0_job1"]["thresholds"]
    s2, a1 = json.loads((E54 / "stage2.json").read_text()), json.loads((E54 / "amend1.json").read_text())
    out: dict[str, Any] = {"states": {}}
    for n, dW in dws.items():
        t, path = task_of(tasks, n), E54 / "large" / f"{n}.pt"
        g = ST.gains(ctx.net, t, dW)
        st = {**g, "sha256": KD.sha256(path), **NE.patch_controls(ctx.net, t, dW, g["G_score"]), **readout(ctx, t, dW, g, bases[t.name]),
              "readable": abs(g["G"]) >= P["settings"]["floor_min"]}
        ref, ref_lo = s2["states"][n], a1["states"][n]
        st["d"] = {"r": st["r"] - ref["r"], "S": st["S"] - ref["S"], "S_lo": st["S_lo"] - ref_lo["S_lo"]}
        st["ok"] = bool(st["sha256"] == s2["finetunes"][n]["sha256"] and st["layer"] == ref["layer"]
                        and abs(st["identity_share"]) <= th["identity_share"] and st["identity_max_abs"] <= th["identity_max_abs"]
                        and abs(st["full_patch_share"] - 1) <= th["full_patch_tol"]
                        and all(abs(v) <= th["reproduce_tol"] for v in st["d"].values()))
        out["states"][n] = st
        say(f"stage0 {n}", st)
        log(f"[stage0 {n}] identity_share {st['identity_share']:.2e} identity_max_abs {st['identity_max_abs']:.2e} "
            f"full_patch_share {st['full_patch_share']:.10f} d {json.dumps(st['d'])} sha {st['sha256'][:12]} ok {st['ok']}")
    out["pass"] = all(st["ok"] for st in out["states"].values())
    return out


def job1(ctx: KD.Ctx, tasks: dict[str, ST.Task], P: dict[str, Any]) -> None:
    bases = {n: KD.base_of(ctx, t) for n, t in tasks.items()}
    dws = {n: torch.load(E54 / "large" / f"{n}.pt", weights_only=True) for n in (f"{k}_{s}" for k in KINDS for s in OLD)}
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "stage0": stage0_job1(ctx, tasks, dws, bases, P)}
    dump("job1.json", out)
    if not out["stage0"]["pass"]:
        sys.exit("[stage0 job 1] FAIL: stop and tell the user")
    out["states"] = {n: dict(st) for n, st in out["stage0"]["states"].items()}
    for n, st in out["states"].items():
        if st["readable"]:
            st.update(b2(ctx.net, task_of(tasks, n), dws[n], st["G_score"], P["settings"]["b2_grid"]))
        say(f"b2 {n}", st)
        dump("job1.json", out)
    out["P1_two"] = KS.p1(out["states"], OLD, P["thresholds"]["min_gap"])
    out["branch"] = KS.branch_job1(out["P1_two"])
    dump("job1.json", out)
    log(f"[score job 1] {json.dumps(out['P1_two'], default=str)}")
    log(f"[branch job 1] {out['branch']}")


# ----------------------------------------------------------------------------- job 2


def stage0_job2(ctx: KD.Ctx, tasks: dict[str, ST.Task], bases: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    th = P["stage0_job2"]["thresholds"]
    s1, s2 = json.loads((E54 / "stage1.json").read_text()), json.loads((E54 / "stage2.json").read_text())
    out: dict[str, Any] = {}
    for n, t in tasks.items():
        rate = KD.geo_rate(t.lp0)
        out[n] = {"sha256": KD.sha256(E54 / "large" / f"windows_{n}.pt"), "rate": rate, "rate_ref": s1[n]["rate"],
                  "rate_rel": rate / s1[n]["rate"] - 1, "keys": list(t.keys) == s1[n]["keys"],
                  "values": [int(t.table[x]) for x in t.keys] == s1[n]["values"]}
        out[n]["ok"] = bool(out[n]["sha256"] == s1["sha256"][n] and out[n]["keys"] and out[n]["values"]
                            and abs(out[n]["rate_rel"]) <= th["rate_rel_tol"])
        log(f"[stage0 job 2 {n}] W0 rate {rate:.10e} ref {s1[n]['rate']:.10e} rel {out[n]['rate_rel']:.1e} ok {out[n]['ok']}")
    path = E54 / "large" / "elicit_s0.pt"
    dW = torch.load(path, weights_only=True)
    rr = KD.route_r(ctx, tasks["elicit"], dW, bases["elicit"])
    r = EL.share_or_nan(rr["r"], rr["G_logodds"])  # as keydep.readings forms it
    ref = s2["states"]["elicit_s0"]["r"]
    out["elicit_s0"] = {"sha256": KD.sha256(path), "r": r, "r_ref": ref, "dr": r - ref, "G_logodds": rr["G_logodds"]}
    out["elicit_s0"]["ok"] = bool(out["elicit_s0"]["sha256"] == s2["finetunes"]["elicit_s0"]["sha256"] and abs(r - ref) <= th["reproduce_tol"])
    log(f"[stage0 job 2 elicit_s0] r {r:.10f} ref {ref:.10f} dr {r - ref:.1e} ok {out['elicit_s0']['ok']}")
    out["pass"] = all(v["ok"] for v in out.values())
    return out


def seed_states(ctx: KD.Ctx, ws: dict[str, Any], s: str, out: dict[str, Any]) -> dict[str, ST.DW]:
    """One new seed's three fine-tunes; the elicit and matched cipher weight changes saved with their sha256."""
    seed, meta = int(s[1:]), out["finetunes"]
    dws: dict[str, ST.DW] = {}
    dws[f"elicit_{s}"], meta[f"elicit_{s}"] = KD.elicit_ft(ctx, ws["elicit"], seed)
    c, co = f"cipher_{s}", f"cipher_own_{s}"
    dws[c], dws[co], meta[c], meta[co] = KD.cipher_ft(ctx, ws["cipher"], seed, meta[f"elicit_{s}"]["norm"])
    for n in (f"elicit_{s}", c):
        torch.save(dws[n], ctx.large / f"{n}.pt")
        meta[n]["sha256"] = KD.sha256(ctx.large / f"{n}.pt")
    for n in dws:
        log(f"[ft {n}] step {meta[n]['step']} ||dW|| {meta[n]['norm']:.4f}"
            + (f" at_bound {meta[n]['at_bound']} same_state {meta[n]['same_state']}" if "at_bound" in meta[n] else "")
            + (f" learned {meta[n]['learned']}" if "learned" in meta[n] else ""))
    return dws


def job2(ctx: KD.Ctx, tasks: dict[str, ST.Task], P: dict[str, Any]) -> None:
    j1 = json.loads((RESULTS / "job1.json").read_text())
    if not j1.get("stage0", {}).get("pass") or j1.get("P1_two", {}).get("status") != "pass":
        sys.exit("job 1 did not pass stage 0, or P1 did not hold on two seeds: the decision tree stops here; tell the user")
    bases = {n: KD.base_of(ctx, t) for n, t in tasks.items()}
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "stage0": stage0_job2(ctx, tasks, bases, P),
                           "finetunes": {}, "states": {}}
    dump("job2.json", out)
    if not out["stage0"]["pass"]:
        sys.exit("[stage0 job 2] FAIL: stop and tell the user")
    ws = {n: torch.load(E54 / "large" / f"windows_{n}.pt", weights_only=False) for n in tasks}
    for s in NEW:
        dws = seed_states(ctx, ws, s, out)
        dump("job2.json", out)
        for n, dW in dws.items():
            out["states"][n] = read_new(ctx, task_of(tasks, n), dW, bases[task_of(tasks, n).name], P)
            say(f"read {n}", out["states"][n])
            dump("job2.json", out)
    score(out, j1, P)


def score(out: dict[str, Any], j1: dict[str, Any], P: dict[str, Any]) -> None:
    th, every = P["thresholds"], {**j1["states"], **out["states"]}
    same = [s for s in NEW if out["finetunes"][f"cipher_{s}"]["same_state"]]
    out["P1"] = KS.p1(every, OLD + NEW, th["min_gap"])
    out["P2"] = KS.p2(every, OLD + NEW, th["min_gap"], th["ratio_min"])
    out["P3"] = KS.p3(every, NEW, th["p3_tol"], same)
    out["branch"] = KS.branch(out["P1"], out["P2"], out["P3"])
    notes = [f"cipher_{s} at_bound" for s in NEW if out["finetunes"][f"cipher_{s}"]["at_bound"]]
    notes += [f"elicit_{s} did not learn (scored at step 300)" for s in NEW if not out["finetunes"][f"elicit_{s}"]["learned"]]
    out["notes"] = notes
    dump("job2.json", out)
    for k in ("P1", "P2", "P3"):
        log(f"[score {k}] {json.dumps(out[k], default=str)}")
    log(f"[branch] {out['branch']}" + "".join(f" [{n}]" for n in notes))


# ----------------------------------------------------------------------------- main


def main() -> None:
    exp = HERE.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, choices=["1", "2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=exp / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="where job 2's six new weight changes go")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    check_committed(PRED, *([RESULTS / "job1.json"] if a.job == "2" else []))
    P = json.loads(PRED.read_text())
    assert P["settings"]["b2_grid"][-1] == ST.N_LAYERS * ST.D_MLP and 16 in P["settings"]["b2_grid"]
    assert P["settings"]["route_batch"] == KD.BATCH, "predictions.json settings disagree with the code"
    a.large.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, a.large.resolve())
    tasks = {n: ST.load_task(ctx.net, E54 / "large" / f"windows_{n}.pt", n, n) for n in ("elicit", "cipher")}
    (job1 if a.job == "1" else job2)(ctx, tasks, P)


if __name__ == "__main__":
    main()
