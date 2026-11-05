"""E97: name the part of the 67M each fine-tune's gain runs through, or say "not found" at head and MLP grain, with a
control on the same task's own seed, across the saved 67M tasks. No training. Plan: reports/plans/2026-11-05_plan_e97-locator.md.
Rules and expected values: results/predictions.json (committed first; the run refuses otherwise; never edited),
`commute.locator_rules`. Reads: `commute.locator`. Sites and states: sites.py.

  stage 0, alone: two planted one-matrix writes on the 24-key copy table (blocks 3 and 1), whose reads are exact by
           construction; E62's and E53's gains re-read; W0's whole-row gap re-reads A0's. Exit 1 on any miss.
  stage 1, the control: copy against lookup, E62's six states. a is read from the copy seeds and written to stage1.json
           before anything uses it. Stage 3 runs only if the carry map names the filed head on every copy seed.
  stage 2: lowercase against cipher, E53's step-300 states, on whole rows (the E96 check's F2); E53's own 127-token
           windows beside, removal only.
  stage 3: the other saved pairs; the constant-answer tasks beside only, against their same-windows null.
  table:   results/table.json, one row per (task, seed), from the stage files alone.

Writes results/stage{0,1,2,3}.json as each state finishes (a stage resumes from its file), table.json last; every number
is also printed. `--stages 0,1,2` and `--stages 3` are the two jobs (cluster/e97_locator*.sbatch).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import job as JB
import sites
import table
import torch
from job import RESULTS, Job, dump, load, say, stop
from sites import STAGES, TASKS

from commute import locator as L
from commute import locator_rules as LR
from commute import steering as ST
from commute.copying import finite
from commute.jobs import hard_exit
from commute.keydep import log, sha256
from commute.locate import ranked

REPO = Path(__file__).resolve().parents[2]
PRED, TABLE = RESULTS / "predictions.json", RESULTS / "table.json"
STAGE = {k: RESULTS / f"stage{k}.json" for k in range(4)}


# ----------------------------------------------------------------------------- stage 0


def regain(J: Job) -> dict[str, Any]:
    """E62's gains against run.json; E53's on its windows against E96's fp32 read; W0's whole-row gap against A0 (bf16)."""
    tol, out = J.th["g_tol"], {}
    put = lambda k, x, ref, src: out.__setitem__(k, {"read": x, "ref": ref, "ok": finite(x) and abs(x - ref) <= tol, "ref_source": src})
    run62 = json.loads((sites.CUE / "run.json").read_text())["states"]
    for t in STAGES[1]:
        for s in sorted(TASKS[t].states):
            put(f"{t} s{s} G", float(J.state(t, s).G[0]), run62[f"{t.split()[1]}_s{s}"]["G"], "induction_cue/results/run.json")
    e96 = json.loads((sites.E96 / "stage2_stage0.json").read_text())["states"]
    a0 = json.loads(sites.E53.A0.read_text())
    for t in STAGES[2]:
        m = t.split()[1]
        for s in sorted(TASKS[t].states):
            put(f"{t} s{s} G, 127-token windows", float(J.state(t, s, f"e53w_{m}").G[0]), e96[f"{m}_s{s}"]["G_fp32"], "locate_mechanism/results/stage2_stage0.json")
        put(f"W0 {m} gap, whole rows", float(J.base(f"e53_{m}").lo0[:, 1].mean()), a0[m]["mean"], "performed_anchor/results/a0.json (bf16)")
    return out


def stage0(J: Job) -> None:
    path, th = STAGE[0], J.th
    rec: dict[str, Any] = {"plants": {}, "files": J.F.seen}
    b, (_, w) = J.base("kv_copy24"), sites.kv(J.F, sites.COPY24)
    Gt = float(J.state("T13 copy-24", 0).G[0])
    for block in J.P["settings"]["plant_blocks"]:
        dW, fit = L.plant(J.net, b.site, w["train"].tokens, w["train"].half, w["table"], block, Gt, th["plant_rel_tol"])
        r = L.reads(J.net, L.state(J.net, b, dW))
        rec["plants"][f"block{block}"] = {"fit": fit, "misses": LR.plant_misses(r, f"h.{block}.mlp.down_proj", block, th), "inheritance": LR.inheritance(r["carry"], block), "reads": r}
        say(f"stage0 plant block {block}", rec["plants"][f"block{block}"])
        dump(path, rec)
    rec["regain"] = regain(J)
    say("stage0 regain", rec["regain"])
    misses = [f"plant {k}: {m}" for k, p in rec["plants"].items() for m in p["misses"]] + [k for k, r in rec["regain"].items() if not r["ok"]]
    rec["status"] = "pass" if not misses else "miss"
    rec["missed"] = misses
    dump(path, rec)
    if misses:
        stop(path, rec, f"stage 0: {misses[:5]}")


# ----------------------------------------------------------------------------- stage 1


def singles(r: dict[str, Any]) -> dict[str, float]:
    return {n: v["share"]["lo"] for n, v in r["removal"].items() if "+" not in n and not n.startswith("mlp")}


def capability(J: Job, out: dict[str, Any]) -> dict[str, Any]:
    """W0's second-copy top-1 with each part off at the copy windows' means (E62's settings); c_share = 1 - off / intact."""
    C = json.loads((sites.CUE / "predictions.json").read_text())["settings"]["induction"]
    means = J.base("e62_copy").means
    base = L.top1_off(J.net, means, None, C)
    names = [L.name(p) for p in L.PARTS] + sorted({n for t in STAGES[1] for n in out["located"][t]["nominated"] if "+" in n})
    parts = {n: {"top1_off": (x := L.top1_off(J.net, means, n, C)), "c_share": 1 - x / base} for n in names}
    return {"top1_base": base, "means": "W0's on E62's copy read windows", "parts": parts}


def stage1(J: Job) -> None:
    path, E = STAGE[1], J.P["expected"]["stage1"]
    out = JB.stage_reads(J, STAGES[1], path)
    copy = [table.rec_of(out, "T5 copy", s) for s in sorted(TASKS["T5 copy"].states)]
    tops = [LR.top_part(singles(r)) for r in copy]
    out["a"] = a = LR.a_from([singles(r)[h] if h else math.nan for r, h in zip(copy, tops, strict=True)])
    out["a_read"] = {"top_head_by_removal": tops, "shares": [singles(r)[h] if h else math.nan for r, h in zip(copy, tops, strict=True)]}
    dump(path, out)
    say("stage1 a", {"a": a, **out["a_read"]})
    if not (finite(a) and a > 0):
        stop(path, out, f"stage 1: a reads {a}; the located rule would be degenerate")
    JB.locate(J, STAGES[1], out, path, a)
    head, look = E["head"], [table.rec_of(out, "T5 lookup", s) for s in sorted(TASKS["T5 lookup"].states)]
    carry_tops = [LR.top_part({n: v["lo"] for n, v in r["carry"].items()}) for r in copy]
    out["control"] = {"carry_tops": carry_tops, "expected": head, "passes": LR.control_passes(carry_tops, head)}
    plant = load(STAGE[0])["plants"]["block3"]["reads"]["removal"][head]["share"]["lo"]
    look_shares = [r["removal"][head]["share"]["lo"] for r in look]
    out["hand_write"] = {"plant_share_at_head": plant, "lookup_shares_at_head": look_shares, "within_lookup": LR.write_within(plant, look_shares), "under_a": finite(plant) and plant < a}
    out["capability"] = capability(J, out)
    out["expected_table"] = {"copy_removal_at_head": {"read": [r["removal"][head]["share"]["lo"] for r in copy], "band": E["copy_removal_band"]},
                             "lookup_removal_at_head": {"read": look_shares, "at_most": E["lookup_removal_at_most"]},
                             "lookup_carry_top": {"read": [LR.top_part({n: v["lo"] for n, v in r["carry"].items()}) for r in look], "lean": E["lookup_carry_lean"]}}  # fmt: skip
    out["outcome"] = "carry names the filed head on every copy seed" if out["control"]["passes"] else E["miss_branch"]
    dump(path, out)
    say("stage1 control", {k: out[k] for k in ("control", "hand_write", "expected_table", "outcome")})
    say("stage1 capability", out["capability"])


# ----------------------------------------------------------------------------- stage 2


def windows64(J: Job) -> dict[str, Any]:
    """Beside: E53's own 127-token windows (read with 64 tokens of context), removal only, the windows' own means."""
    t0, res = time.monotonic(), {}
    for t in STAGES[2]:
        for s in sorted(TASKS[t].states):
            st = J.state(t, s, f"e53w_{t.split()[1]}")
            res[f"{t} s{s}"] = {"G": L.units(st.G), "removal": {L.name(p): L.removal(J.net, st, L.name(p)) for p in L.PARTS}}
    res["seconds"] = time.monotonic() - t0
    return res


def stage2(J: Job, a: float) -> None:
    path = STAGE[2]
    out = JB.stage_reads(J, STAGES[2], path)
    out["a"] = a
    JB.locate(J, STAGES[2], out, path, a)
    floor = load(STAGE[0])["plants"]["block1"]["inheritance"]
    lows = [table.rec_of(out, "T7 lowercase", s) for s in sorted(TASKS["T7 lowercase"].states)]
    carry = [{n: v["lo"] for n, v in r["carry"].items()} for r in lows]
    nominated = [c[n] for c in carry for n in ranked(c)[: J.th["nominate_top"]]]
    can = LR.can_nominate(floor, nominated)
    out["inheritance"] = {"floor_block1_plant": floor, "nominated_carry_shares": nominated, "can_nominate": can}
    if not can:
        out["cannot_nominate"] = {"T7 lowercase": f"the block-1 plant's inheritance downstream ({floor:.3f}) exceeds the smallest nominated carry share: this grain cannot nominate here"}
    dump(path, out)
    say("stage2 inheritance", out["inheritance"])
    out["windows64"] = windows64(J)
    dump(path, out)
    say("stage2 windows64", out["windows64"])


# ----------------------------------------------------------------------------- stage 3


def stage3(J: Job, a: float) -> None:
    path = STAGE[3]
    out = load(path)
    out.update({"a": a, "files": J.F.seen, "skipped": {}})
    names = []
    for t in STAGES[3]:
        missing = [str(f) for u in (t, TASKS[t].control) for f in TASKS[u].states.values() if not f.is_file()]
        if missing:
            out["skipped"][t] = f"states missing, not rebuilt: {missing}"
            continue
        JB.read_tasks(J, [t], out, path)
        names.append(t)
    JB.cross_pairs(J, names, out, path)
    JB.locate(J, names, out, path, a)
    vpd = J.P["expected"]["stage3"]["emoticon_vpd_matrix"]
    tops = [LR.top_part({n: v["lo"] for n, v in table.rec_of(out, "T1 emoticon", s)["write"]["matrix"].items()}) for s in sorted(TASKS["T1 emoticon"].states)]
    out["vpd_agreement"] = {"write_top_matrix": tops, "vpd": vpd, "agrees": [t == vpd for t in tops]}
    dump(path, out)
    say("stage3 skipped", out["skipped"])
    say("stage3 vpd agreement", out["vpd_agreement"])


# ----------------------------------------------------------------------------- the table, load and main


def write_table() -> None:
    stages = {k: js for k in (1, 2, 3) if "located" in (js := load(STAGE[k]))}
    rows = table.rows(stages)
    dump(TABLE, {"table": rows, "limit": "a new circuit built from old parts still reads as reuse", "verdict": "left for baseline-lead, except 'does not apply' on constant-answer rows"})
    for r in rows:
        say("table", r)


def filed() -> dict[str, Any]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(PRED.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", default="0,1,2,3")
    args = ap.parse_args()
    want = {int(x) for x in args.stages.split(",")}
    P = filed()
    s0 = load(STAGE[0])
    if 0 in want and s0.get("status", "pass") != "pass":
        sys.exit(f"{STAGE[0]} holds a missed stage 0: move it aside; this run deletes nothing")
    if 0 not in want and s0.get("status") != "pass":
        sys.exit("stage 0 has not passed: run it first")
    nat_file = sites.E96 / "means.pt"
    if sha256(nat_file) != P["settings"]["natural_means_sha256"]:
        sys.exit("STOP: E96's means.pt is not the filed one")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E96
    net = ST.Net(args.device)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {git} predictions {sha256(PRED)} stages {sorted(want)}")
    J = Job(net, sites.Files(), P, L.natural_means(torch.load(nat_file, weights_only=True, map_location="cpu"), net.device))
    for k, f in ((0, stage0), (1, stage1)):
        if k in want and not (k == 0 and s0.get("status") == "pass"):
            t0 = time.monotonic()
            f(J)
            log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
    a = load(STAGE[1]).get("a", math.nan)
    if (2 in want or 3 in want) and not (finite(a) and a > 0):
        stop(STAGE[1], load(STAGE[1]), f"no bar a in stage1.json ({a}): stage 1 first")
    if 2 in want:
        t0 = time.monotonic()
        stage2(J, a)
        log(f"[stage 2 done] {time.monotonic() - t0:.0f} s")
    if 3 in want:
        if not load(STAGE[1]).get("control", {}).get("passes"):
            log("stage 3 does not run: stage 1's control did not pass (the plan's tree)")
        else:
            J.clear()
            t0 = time.monotonic()
            stage3(J, a)
            log(f"[stage 3 done] {time.monotonic() - t0:.0f} s")
    write_table()
    hard_exit(0)


if __name__ == "__main__":
    main()
