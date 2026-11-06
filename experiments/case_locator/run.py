"""E114 (case_locator), 2026-11-06: E97's locator and E100's carry / own / graft reads on E112's four saved case-label
states, so the one pair read under the adopted definition gets a weight-side "where the gain runs" column. No training.
Plan: reports/plans/2026-11-06_plan_e114-locator-case-label.md. Rules: results/predictions.json (the run refuses unless it
is committed unchanged). Reads: `commute.locator`; the located rule: `commute.locator_rules`.

The site: E112's task windows (task.pt in settings.e112_large), read at the marker (position 20, the word at 19), the
target the state's task's label (A or B) for the word, "gap" against the other label. Scored on the fit windows (train
words), where both arms gained and E112 read its gain and P3; E112's held-out windows (unseen words) carry the W0 guard,
and the real states' removal beside: the twin's labels are random per word, so it has no gain on unseen words.

  0  alone: W0's held-out accuracy and each state's gain re-read E112's; all 24 matrices reverted reads G 0 on each state;
     E97's two one-matrix plants (blocks 3 and 1), fitted on the train windows, read their whole gain at the edited part.
  1  per state: E97's reads (`locator.reads`), E100's split of the 28 parts and its sum of single reverts; the real
     states' greedy pairs read on every state; E97's located rule on real against the same seed's twin.

Writes results/stage0.json and stage1.json as each step finishes, prints every number and "STAGE <n> DONE".
    sbatch cluster/case_locator.sbatch
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import torch

from commute import case_label as CL
from commute import locator as L
from commute import locator_rules as LR
from commute import steering as ST
from commute.anchors import HALF
from commute.jobs import hard_exit, require_filed
from commute.keydep import log, sha256
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
EXP, RESULTS = HERE.parent, HERE / "results"
PRED = RESULTS / "predictions.json"
E112 = EXP / "case_label" / "results"
NAT = EXP / "locate_mechanism" / "results" / "means.pt"
STATES = ("real_s0", "real_s1", "twin_s0", "twin_s1")
SEEDS = (0, 1)
NAMES = [L.name(p) for p in L.PARTS]


def dump(k: int, rec: dict[str, Any]) -> None:
    (RESULTS / f"stage{k}.json").write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True))


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def stop(k: int, rec: dict[str, Any], why: str) -> NoReturn:
    rec["stopped"] = why
    dump(k, rec)
    log(f"STOP: {why}: tell the lead (exit 1)")
    hard_exit(1)


# ----------------------------------------------------------------------------- inputs (no model: the dry run reads these)


def checked(path: Path, sha: str) -> Path:
    if sha256(path) != sha:
        raise ValueError(f"{path}: sha256 differs from the filed {sha}")
    return path


def load_task(S: dict[str, Any]) -> dict[str, Any]:
    large = Path(S["e112_large"]).expanduser()
    return torch.load(checked(large / "task.pt", S["task_sha256"]), weights_only=False)


def load_dW(S: dict[str, Any], name: str) -> ST.DW:
    """fp32 on CPU, its sha256 against E112's stage1.json (saved in bf16)."""
    rec = json.loads((E112 / "stage1.json").read_text())["states"][name]
    path = checked(Path(S["e112_large"]).expanduser() / Path(rec["path"]).name, rec["sha256"])
    return {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}


def site(t: dict[str, Any], task: str, part: str) -> L.Site:
    """One read per window at the marker; target the task's label for the word before it, alt the other label. Each
    window is its own row for the standard error (task.pt keeps no source row)."""
    toks = t[part]
    assert toks.shape[1] == 2 * HALF + 1 and bool((toks[:, HALF] == t["marker"]).all()), "not E112's windows"
    a, b = t["ab"]
    y = torch.tensor([t["y"][task][int(w)] for w in toks[:, HALF - 1]])
    tgt, alt = torch.where(y == 1, a, b), torch.where(y == 1, b, a)
    return L.windows(toks, tgt, HALF, torch.arange(len(toks)), alt)


# ----------------------------------------------------------------------------- stage 0


def guards(net: ST.Net, t: dict[str, Any], bases: dict[str, L.Base], dWs: dict[str, ST.DW], th: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """W0's held-out accuracy (label gap > 0) against E112's stage 0; each state's gap gain on the fit windows against
    E112's stage 2 gain; all 24 matrices reverted reads G 0 (unit lo) on each state."""
    acc0 = json.loads((E112 / "stage0.json").read_text())["w0_heldout_accuracy"]
    gain = json.loads((E112 / "stage2.json").read_text())["states"]
    rec: dict[str, Any] = {"w0_heldout_accuracy": {}, "gain": {}, "G_all": {}}
    for task in ("real", "twin"):
        x = float((L.read(net, site(t, task, "held"))[:, 1] > 0).double().mean())
        rec["w0_heldout_accuracy"][task] = {"read": x, "ref": acc0[task], "ok": abs(x - acc0[task]) <= th["acc_tol"]}
    for n in STATES:
        st = L.state(net, bases[n.split("_")[0]], dWs[n])
        g = float(st.G[1])
        rec["gain"][n] = {"read_gap": g, "ref": gain[n]["gain"], "ok": abs(g - gain[n]["gain"]) <= th["g_tol"], "G": L.units(st.G)}
        z = L.units((L.read(net, st.b.site, L.reverted(st.dW, list(st.dW))) - st.b.lo0).mean(0))
        rec["G_all"][n] = {**z, "ok": LR.zero_ok([z["lo"]], th["zero_tol"])}
    misses = [f"{k} {n}" for k, v in rec.items() for n, r in v.items() if not r["ok"]]
    return rec, misses


def stage0(net: ST.Net, P: dict[str, Any], t: dict[str, Any], bases: dict[str, L.Base], dWs: dict[str, ST.DW]) -> None:
    th = P["thresholds"]
    rec: dict[str, Any] = {"plants": {}}
    rec["guards"], misses = guards(net, t, bases, dWs, th)
    say("stage0 guards", rec["guards"])
    dump(0, rec)
    b = bases["real"]
    Gt = float(L.state(net, b, dWs["real_s0"]).G[0])
    a, bb = t["ab"]
    table = CL.table(t["y"]["real"], a, bb, t["vocab"])
    for block in P["settings"]["plant_blocks"]:
        dW, fit = L.plant(net, b.site, t["train"], HALF, table, block, Gt, th["plant_rel_tol"], key_at=HALF - 1)
        r = L.reads(net, L.state(net, b, dW))
        p = rec["plants"][f"block{block}"] = {"fit": fit, "misses": LR.plant_misses(r, f"h.{block}.mlp.down_proj", block, th), "reads": r}
        misses += [f"plant block{block}: {m}" for m in p["misses"]]
        say(f"stage0 plant block {block}", {k: p[k] for k in ("fit", "misses")})
        dump(0, rec)
    rec |= {"status": "miss" if misses else "pass", "missed": misses}
    dump(0, rec)
    if misses:
        stop(0, rec, f"stage 0: {misses[:5]}")


# ----------------------------------------------------------------------------- stage 1


def read_state(net: ST.Net, st: L.State) -> dict[str, Any]:
    """E97's reads plus E100's split of every part and the sum of the 28 single reverts (own shares added)."""
    t0 = time.monotonic()
    rec = L.reads(net, st)
    rec["split"] = {n: L.split(net, st, n) for n in NAMES}
    rec["sum_own"] = {u: sum(v["own"][u] for v in rec["split"].values()) for u in ("lo", "gap")}
    rec["seconds"] = time.monotonic() - t0
    return rec


def locate(out: dict[str, Any], a: float, th: dict[str, Any], unit: str) -> dict[str, Any]:
    """E97's located rule (`locator_table/table.py` `locate_task`): real against the same seed's twin, every seed."""
    recs, ctls = [out["states"][f"real_s{s}"] for s in SEEDS], [out["states"][f"twin_s{s}"] for s in SEEDS]
    noms = list(dict.fromkeys(n for r in recs for n in LR.nominees(r, th["nominate_top"])))
    parts = {}
    for n in noms:
        rd = [LR.seed_reading({"G": r["G"][unit], "fall": r["removal"][n]["fall"][unit]}, {"G": c["G"][unit], "fall": c["removal"][n]["fall"][unit]}, a, th)
              for r, c in zip(recs, ctls, strict=True)]  # fmt: skip
        parts[n] = {**LR.task_reading(rd), "shares": [r["removal"][n]["share"][unit] for r in recs], "control_shares": [c["removal"][n]["share"][unit] for c in ctls]}
    return {"unit": unit, "nominated": noms, "parts": parts, "part": LR.best_located(parts)}


def leans(out: dict[str, Any]) -> dict[str, Any]:
    """Per real seed, unit lo: the top removal part of the 28, its share, the same seed's twin share there, and its carry,
    own and graft. Reported against the plan's leans; no rule."""
    res = {}
    for s in SEEDS:
        r, c = out["states"][f"real_s{s}"], out["states"][f"twin_s{s}"]
        top = LR.top_part({n: r["removal"][n]["share"]["lo"] for n in NAMES})
        res[f"real_s{s}"] = {"top": top} if top is None else {
            "top": top, "share": r["removal"][top]["share"]["lo"], "twin_share": c["removal"][top]["share"]["lo"],
            **{k: r["split"][top][k]["lo"] for k in ("carry", "own", "graft")}, "sum_own": r["sum_own"]["lo"]}  # fmt: skip
    return res


def stage1(net: ST.Net, P: dict[str, Any], t: dict[str, Any], bases: dict[str, L.Base], dWs: dict[str, ST.DW]) -> None:
    out: dict[str, Any] = {"states": {}, "heldout_beside": {}}
    sts = {n: L.state(net, bases[n.split("_")[0]], dWs[n]) for n in STATES}
    for n in STATES:
        out["states"][n] = rec = read_state(net, sts[n])
        dump(1, out)
        say(f"state {n}", rec)
        if not LR.zero_ok([rec["write"]["G_all"]["lo"]], P["thresholds"]["zero_tol"]):
            stop(1, out, f"{n}: all 24 reverted reads G {rec['write']['G_all']['lo']}")
    for p in sorted({out["states"][f"real_s{s}"].get("pair") for s in SEEDS} - {None}):
        for n in STATES:
            if p not in out["states"][n]["removal"]:
                out["states"][n]["removal"][p] = L.removal(net, sts[n], p)
                say(f"pair {p} on {n}", out["states"][n]["removal"][p])
    a = P["thresholds"]["a"]
    out["located"] = {u: locate(out, a, P["thresholds"], u) for u in ("lo", "gap")}
    out["leans"] = leans(out)
    dump(1, out)
    say("located", {u: {"part": v["part"], "parts": {n: {k: x[k] for k in ("reading", "per_seed", "shares", "control_shares")} for n, x in v["parts"].items()}} for u, v in out["located"].items()})
    say("leans", out["leans"])
    held = bases["real_held"]
    for n in ("real_s0", "real_s1"):
        st = L.state(net, held, dWs[n])
        out["heldout_beside"][n] = {"G": L.units(st.G), "removal": {m: L.removal(net, st, m) for m in NAMES}}
        say(f"held-out beside {n}", out["heldout_beside"][n])
    dump(1, out)


# ----------------------------------------------------------------------------- main


def main() -> None:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    S = P["settings"]
    if (RESULTS / "stage0.json").is_file() and json.loads((RESULTS / "stage0.json").read_text()).get("status") != "pass":
        sys.exit("results/stage0.json holds an unfinished or missed stage 0: move it aside; this run deletes nothing")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E96, E97, E112
    net = ST.Net("cuda")
    log(f"[host] {torch.cuda.get_device_name()} predictions {sha256(PRED)}")
    t = load_task(S)
    dWs = {n: load_dW(S, n) for n in STATES}
    nat = L.natural_means(torch.load(checked(NAT, S["natural_means_sha256"]), weights_only=True, map_location="cpu"), net.device)
    bases = {task: L.base(net, site(t, task, "fit"), nat) for task in ("real", "twin")}
    bases["real_held"] = L.base(net, site(t, "real", "held"), nat)
    for k, f in ((0, stage0), (1, stage1)):
        t0 = time.monotonic()
        f(net, P, t, bases, dWs)
        log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
        print(f"STAGE {k} DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
