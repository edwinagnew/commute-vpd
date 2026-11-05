"""E97 Amendment 1: edges with integrated gradients, and the remaining 67M pairs. Plan:
reports/plans/2026-11-05_plan_e97-amend1.md; scoping reports/reviews/2026-11-05_review_e97-eapig-scoping.md (option 2).
Rules and expected values: results/predictions_amend1.json (committed first; the run refuses otherwise; never edited),
`commute.edge_rules`. Reads: `commute.edges`. Pairs: pairs.py. Table: amend1_table.py.

  stage 0, alone, exact: the fp32 floor measured first (the module and our forward, each against itself in fp64, at W0);
           no edge patched against `net.run`; every edge patched against the corrupted run; E97's two plants on the
           24-key copy table, every edge patched alone (block 3: mlp3 -> logits is the whole gain, every other edge 0;
           block 1: measured). Exit 1 on any miss.
  stage 1: E62's six states. Every gain-path edge patched exactly; each part's edges summed against E97's carry (MLPs)
           or its own joint patch (heads: E97's head is o_proj's input, ours its write; predictions "heads_departure")
           (stage 0's last check: a miss exits 1); IG beside. W0's circuits on the copy and lookup pairs (the copy pair's
           is W0's C); each state's own circuit and its cross-faithfulness under W0's C on both pairs. The band and the
           tree's outcome are written to amend1_stage1.json before anything reads them.
  stage 2: only on the tree's "copy high, lookup low": lowercase and cipher on whole rows, IG with the top 50 exact.
  stage 3: the same gate; E97's own stage 3 (run.stage3, its file stage3.json, its rule and a), then every state's
           edges, its circuit on its own pair, and E97's located rule again with the edges' nominees added.
  table:   results/table_amend1.json, E97's rows plus the edge columns, from the stage files alone.

Each stage file is written as each state finishes (a stage resumes from it); "STAGE <n> DONE" is printed only after
its file is complete. `--stages 0,1,2` and `--stages 3` are the two jobs (cluster/e97_amend1*.sbatch).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import amend1_table
import job as JB
import run as E97
import sites
import table
import torch
from job import RESULTS, Job, dump, load, say, stop
from pairs import Pairs, c_pair, own_pair
from sites import STAGES, TASKS

from commute import edge_rules as ER
from commute import edges as EG
from commute import locator as L
from commute import locator_rules as LR
from commute import steering as ST
from commute.jobs import hard_exit
from commute.keydep import log, sha256

REPO = Path(__file__).resolve().parents[2]
PRED = RESULTS / "predictions_amend1.json"
OUT = {k: RESULTS / f"amend1_stage{k}.json" for k in range(4)}


def done(k: int | str) -> None:
    log(f"STAGE {k} DONE")


def dev(J: Job, t: str, s: int) -> ST.DW:
    return {k: v.to(J.net.device) for k, v in J.dW(TASKS[t].states[s]).items()}


# ----------------------------------------------------------------------------- stage 0


def floor(net: ST.Net, p: EG.Pair) -> dict[str, Any]:
    """At W0: the module's read and ours, each in fp32 against itself in fp64; the bar is the sum of the two floors."""
    col, one = (0 if p.unit == "lo" else 1), torch.ones(ER.VALID.shape)
    m32, f32 = L.read(net, p.site)[:, col], EG.per_read(net, p, None, one)
    with EG.fp64(net):
        m64, f64 = L.read(net, p.site)[:, col], EG.per_read(net, p, None, one)
    fm, ff = float((m32 - m64).abs().max()), float((f32 - f64).abs().max())
    return {"floor_module": fm, "floor_ours": ff, "bar": fm + ff, "no_patch": float((f32 - m32).abs().max())}


def agree(x: torch.Tensor, ref: torch.Tensor, bar: float) -> dict[str, Any]:
    d = float((x - ref).abs().max())
    return {"max_diff": d, "bar": bar, "ok": d <= bar}


def plants(J: Job, A: Pairs, th: dict[str, Any]) -> dict[str, Any]:
    """E97's two plants rebuilt as E97's stage 0 built them, every gain-path edge patched alone on their site."""
    _, w = sites.kv(J.F, sites.COPY24)
    gp, Gt, out = A.gain("T13 copy-24"), float(J.state("T13 copy-24", 0).G[0]), {}
    for block in J.P["settings"]["plant_blocks"]:
        dW, fit = L.plant(J.net, gp.site, w["train"].tokens, w["train"].half, w["table"], block, Gt, J.th["plant_rel_tol"])
        g, sc = EG.gain_exact(J.net, gp, dW)
        sh = {e: LR.frac(x, g["G"]) for e, x in sc.items()}
        key, rec = f"mlp{block}->logits", {"fit": fit, "gain": g, "G_module": float(L.state(J.net, J.base("kv_copy24"), dW).G[0])}
        rec["direct_share"], rec["mlp_to_later_parts_share"] = sh[key], sum(x for e, x in sh.items() if e.startswith(f"mlp{block}->") and e != key)
        if block == 3:
            rec["misses"] = [f"{e} share {x}" for e, x in sh.items() if not abs(x - (1.0 if e == key else 0.0)) <= th["reread_tol"]]
        out[f"block{block}"] = (rec, dW)
    return out


def stage0(J: Job, A: Pairs, th: dict[str, Any]) -> None:
    path, rec = OUT[0], dict[str, Any](files=J.F.seen)
    rec["floor"] = {"plant site, gain unit lo": floor(J.net, A.gain("T13 copy-24")), "copy pair, unit gap": floor(J.net, A["copy"])}
    bar = max(f["bar"] for f in rec["floor"].values())
    say("amend1 stage0 floor", rec["floor"])
    dump(path, rec)
    P = plants(J, A, th)
    rec["plants"] = {k: r for k, (r, _) in P.items()}
    gp, cp, dW3, one = A.gain("T13 copy-24"), A["copy"], P["block3"][1], torch.ones(ER.VALID.shape)
    rec["checks"] = {**{f"no edge patched, W0, {k}": {"max_diff": f["no_patch"], "bar": f["bar"], "ok": f["no_patch"] <= f["bar"]} for k, f in rec["floor"].items()},
                     "no edge patched, block-3 plant": agree(EG.per_read(J.net, gp, dW3, one), L.read(J.net, gp.site, dW3)[:, 0], bar),
                     "every edge patched, block-3 plant = W0": agree(EG.per_read(J.net, gp, dW3, 0 * one), L.read(J.net, gp.site)[:, 0], bar),
                     "every edge patched, copy pair at W0 = W0 on B'": agree(EG.per_read(J.net, cp, None, 0 * one), L.read(J.net, replace(cp.site, toks=cp.bad))[:, 1], bar)}  # fmt: skip
    rec["missed"] = [k for k, c in rec["checks"].items() if not c["ok"]] + [f"block-3 plant: {m}" for m in rec["plants"]["block3"]["misses"]]
    rec["status"] = "pass" if not rec["missed"] else "miss"
    dump(path, rec)
    say("amend1 stage0", {k: rec[k] for k in ("checks", "missed", "status")} | {"plants": {k: {x: v[x] for x in ("fit", "direct_share", "mlp_to_later_parts_share")} for k, v in rec["plants"].items()}})
    if rec["missed"]:
        stop(path, rec, f"amendment stage 0: {rec['missed'][:5]}")
    done(0)


# ----------------------------------------------------------------------------- the reads per state


def w0_circuits(J: Job, A: Pairs, names: list[str], out: dict[str, Any], path: Path, S: dict[str, Any]) -> None:
    for n in names:
        if n not in out.setdefault("w0", {}):
            t0 = time.monotonic()
            out["w0"][n] = EG.circuit(J.net, A[n], None, S["grid"], S["faith_bar"], S["steps"], S["min_gap"])[0] | {"seconds": time.monotonic() - t0}
            dump(path, out)
            say(f"W0 circuit {n}", {k: v for k, v in out["w0"][n].items() if k != "edges"})


def state_reads(J: Job, A: Pairs, t: str, s: int, out: dict[str, Any], S: dict[str, Any], gain: str, beside: list[str]) -> dict[str, Any]:
    """One state: its gain path (`gain`: "exact" or "ig"), its own circuit, and its cross-faithfulness under W0's C
    circuit on its own pair, and on each pair of `beside` under that pair's W0 circuit."""
    t0, dW = time.monotonic(), dev(J, t, s)
    g, sc = EG.gain_exact(J.net, A.gain(t), dW) if gain == "exact" else EG.gain_ig(J.net, A.gain(t), dW, S["steps"], S["n_exact"])
    rec: dict[str, Any] = {"gain": g, "edge_nominees": ER.edge_nominees(sc, S["nominate_top"])}
    own, cp = own_pair(t), c_pair(t)
    if own and cp:
        circ, mask = EG.circuit(J.net, A[own], dW, S["grid"], S["faith_bar"], S["steps"], S["min_gap"])
        w0 = EG.mask_of(out["w0"][cp].get("edges", []))
        rec |= {"own_pair": own, "c_pair": cp, "circuit": circ, "iou_with_w0_c": ER.iou(mask, w0),
                "cross": {"own": EG.cross(J.net, A[own], dW, w0, S["min_gap"])} | {b: EG.cross(J.net, A[b], dW, EG.mask_of(out["w0"][b].get("edges", [])), S["min_gap"]) for b in beside}}  # fmt: skip
    if t in STAGES[1]:
        rec["scores"] = sc  # E62's every edge, for stage 0's last check (6 x 927 numbers)
    rec["seconds"] = time.monotonic() - t0
    return rec


def each_state(J: Job, A: Pairs, tasks: list[str], out: dict[str, Any], path: Path, S: dict[str, Any], gain: str, beside: dict[str, list[str]]) -> None:
    for t in tasks:
        for s in sorted(TASKS[t].states):
            k = JB.sid(t, s)
            out.setdefault("index", {}).setdefault(t, {})[str(s)] = k
            if k in out.setdefault("states", {}):
                continue
            out["states"][k] = state_reads(J, A, t, s, out, S, gain, beside.get(t, []))
            dump(path, out)
            r = out["states"][k]
            say(f"edges {t} s{s}", {"top_edge": r["gain"]["top_edge"], "G": r["gain"]["G"], "cross": {b: c["faith"] for b, c in r.get("cross", {}).items()}, "circuit_n": r.get("circuit", {}).get("n"), "seconds": r["seconds"]})


# ----------------------------------------------------------------------------- stage 1: E62


def stage1(J: Job, A: Pairs, S: dict[str, Any]) -> None:
    path, e97 = OUT[1], load(E97.STAGE[1])
    out = load(path)
    w0_circuits(J, A, ["copy", "lookup"], out, path, S)
    each_state(J, A, STAGES[1], out, path, S, "exact", {"T5 copy": ["lookup"], "T5 lookup": ["copy"]})
    miss = []
    for t in STAGES[1]:
        for s in sorted(TASKS[t].states):
            r, k = table_rec(out, t, s), f"{t} s{s}"
            sc, G = r["scores"], r["gain"]["G"]
            carry = {n: v["lo"] for n, v in table_rec(e97, t, s)["carry"].items()}
            sums, joint = {n: LR.frac(x, G) for n, x in ER.out_sums(sc).items() if n != "emb"}, r["gain"]["joint_share"]
            mlps = {n: v for n, v in carry.items() if n.startswith("mlp")}  # the same read in E97 and here
            heads = {n: v for n, v in joint.items() if not n.startswith("mlp")}  # E97's head keeps its own o_proj change
            r["check"] = {"sum_vs_carry_mlps": ER.sum_misses(sums, mlps, S["edge_sum_tol"]), "sum_vs_joint_heads": ER.sum_misses(sums, heads, S["edge_sum_tol"]),
                          "joint_vs_carry_mlps": ER.sum_misses(joint, mlps, S["reread_tol"]), "sum_share": sums,
                          "head_joint_minus_e97_carry": {n: joint[n] - carry[n] for n in heads}}  # fmt: skip
            miss += [f"{k} {c}: {m}" for c in ("sum_vs_carry_mlps", "sum_vs_joint_heads", "joint_vs_carry_mlps") for m in r["check"][c]]
            if k not in out.setdefault("ig_beside", {}):
                out["ig_beside"][k] = ig_beside(J, A, t, s, sc, G, S)
                dump(path, out)
    out["missed"] = miss
    dump(path, out)
    if miss:
        stop(path, out, f"amendment stage 0's last check, on E62: {miss[:5]}")
    verdict(out, S)
    dump(path, out)
    say("amend1 stage1", {k: out[k] for k in ("band", "copy_top_edges", "outcome", "go")})
    done(1)


def table_rec(js: dict[str, Any], t: str, s: int) -> dict[str, Any]:
    return js["states"][js["index"][t][str(s)]]


def ig_beside(J: Job, A: Pairs, t: str, s: int, exact: dict[str, float], G: float, S: dict[str, Any]) -> dict[str, Any]:
    """IG against every exact edge on an E62 state: how far the whole-row reads (IG) can be trusted."""
    ig = ER.as_dict(EG.ig_gain(J.net, A.gain(t), dev(J, t, s), S["steps"]))
    d = {e: abs(ig[e] - exact[e]) for e in exact}
    return {"max_abs_diff_share": LR.frac(max(d.values()), G), "ig_sum_share": LR.frac(sum(ig.values()), G), "ig_top_edge": ER.top_edge(ig), "exact_top_edge": ER.top_edge(exact)}


def verdict(out: dict[str, Any], S: dict[str, Any]) -> None:
    faith = lambda t: [table_rec(out, t, s)["cross"]["own"]["faith"] for s in sorted(TASKS[t].states)]
    out["copy_top_edges"] = [table_rec(out, "T5 copy", s)["gain"]["top_edge"] for s in sorted(TASKS["T5 copy"].states)]
    out["band"] = ER.band(faith("T5 copy"), faith("T5 lookup"))
    out["go"], out["outcome"] = ER.outcome(out["copy_top_edges"], S["expected_top_edge_copy"], out["band"])


# ----------------------------------------------------------------------------- stages 2 and 3


def gate() -> None:
    s1 = load(OUT[1])
    if not s1.get("go"):
        log(f"stages 2 and 3 do not run: {s1.get('outcome', 'stage 1 has not finished')} (the plan's tree)")
        amend1_table.write()
        hard_exit(0)


def stage2(J: Job, A: Pairs, S: dict[str, Any]) -> None:
    path = OUT[2]
    out = load(path)
    out["band"] = load(OUT[1])["band"]
    w0_circuits(J, A, ["lowercase", "cipher"], out, path, S)
    each_state(J, A, STAGES[2], out, path, S, "ig", {"T7 lowercase": [], "T8 cipher": []})
    for t, other in (("T7 lowercase", "cipher"), ("T8 cipher", "lowercase")):
        for s in sorted(TASKS[t].states):
            r = table_rec(out, t, s)
            if f"under_w0_{other}" not in r["cross"]:
                r["cross"][f"under_w0_{other}"] = EG.cross(J.net, A[own_pair(t) or ""], dev(J, t, s), EG.mask_of(out["w0"][other].get("edges", [])), S["min_gap"])
                dump(path, out)
    done(2)


def stage3(J: Job, A: Pairs, S: dict[str, Any]) -> None:
    E97.stage3(J, load(E97.STAGE[1])["a"])
    J.clear()
    s3, path = load(E97.STAGE[3]), OUT[3]
    names = [t for t in STAGES[3] if t not in s3.get("skipped", {})]
    out = load(path)
    out["band"] = load(OUT[1])["band"]
    w0_circuits(J, A, sorted({p for t in names for p in [c_pair(t)] if p}), out, path, S)
    each_state(J, A, JB.with_controls(names), out, path, S, "exact", {})
    for k, r in s3["states"].items():
        r["edge_nominees"] = out["states"].get(k, {}).get("edge_nominees", [])
    out["located"] = {t: table.locate_task(s3, t, s3["a"], J.th) for t in names}
    dump(path, out)
    say("amend1 stage3 located", {t: r["part"] for t, r in out["located"].items()})
    done(3)


# ----------------------------------------------------------------------------- main


def filed(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not path.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", default="0,1,2")
    args = ap.parse_args()
    want = {int(x) for x in args.stages.split(",")}
    P, P97 = filed(PRED), E97.filed()
    S = P["settings"] | {"nominate_top": P97["thresholds"]["nominate_top"], "reread_tol": P97["thresholds"]["reread_tol"]}
    s0 = load(OUT[0])
    if s0.get("status", "pass") != "pass":
        sys.exit(f"{OUT[0]} holds a missed stage 0: move it aside; this run deletes nothing")
    if 0 not in want and s0.get("status") != "pass":
        sys.exit("the amendment's stage 0 has not passed: run it first")
    if sha256(sites.E96 / "means.pt") != P97["settings"]["natural_means_sha256"]:
        sys.exit("STOP: E96's means.pt is not the filed one")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E96 and E97
    net = ST.Net(args.device)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {git} predictions {sha256(PRED)} stages {sorted(want)}")
    J = Job(net, sites.Files(), P97, L.natural_means(torch.load(sites.E96 / "means.pt", weights_only=True, map_location="cpu"), net.device))
    A = Pairs(J, S["pair_seed"])
    for k, f in ((0, lambda: stage0(J, A, S)), (1, lambda: stage1(J, A, S))):
        if k in want and not (k == 0 and s0.get("status") == "pass"):
            t0 = time.monotonic()
            f()
            log(f"[amend1 stage {k}] {time.monotonic() - t0:.0f} s")
    if 2 in want or 3 in want:
        gate()
    for k, f in ((2, stage2), (3, stage3)):
        if k in want:
            t0 = time.monotonic()
            f(J, A, S)
            log(f"[amend1 stage {k}] {time.monotonic() - t0:.0f} s")
    amend1_table.write()
    done("table")
    hard_exit(0)


if __name__ == "__main__":
    main()
