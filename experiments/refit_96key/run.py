"""E147 (refit_96key), cell P2.12: E58's four 96-key pairs fine-tuned again with their own recipe and seeds, the updates kept this
time, and read with the readers the other pairs of E133's table were read with. Plan reports/plans/2026-11-10_plan_e147-refit-96key.md;
card 2026-11-10_plan_p2-12-card.md. Maps, windows and E58's saved reads: results/inventory.json; settings, tolerances and the filed
predictions: results/predictions.json (both committed before the run; the run refuses otherwise). Sides are named as built; no
verdict on reuse.

  stage 0  controls, alone, exit 1 on a miss, written to results/stage0.json:
           (a) the readers on states they have read before: holes_fill's own stage 0, unchanged (zero update and strength 0 read
               0; its three reproductions), then c3/rank1_s0 through E136's, E98's and E143's readers against their saved readings;
           (b) per map and seed: the windows' sha256 against E58's record; the fine-tune (`keydep.finetune`, read at step 30), saved
               in bf16 to results/large/dW_<map>_s<seed>.pt; the trained share read from the saved file against E58's step-30 read,
               within the map's seed range. A miss stops the job: that is the result. The 7 updates kept by earlier runs are
               compared with their refits beside (reported, not a stop).
  stage 1  every refit state: E136's untrained shares (SVD, rotated), E98's r_n and r_half, E143's TopK neurons (k 64, k matched)
           and fair steering (pooled, per-key), each state written to results/stage1.json as it lands; then E143's rule per
           reading and pair, the verdicts against the share, and the filed predictions beside.
Each stage prints `STAGE n DONE` after its file is written. A rerun resumes: stage 0 keeps the states it has checked, stage 1 the
states it has read; the readers' controls (a) rerun first every time (about 2 min).

    sbatch --comment=E147 cluster/refit_96key.sbatch
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import keydep as KD
from commute import neuron_r as NR
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE = RESULTS / "large"
PRED, INV = RESULTS / "predictions.json", RESULTS / "inventory.json"
E136_R, E98_R, E143_R = "experiments/share_untrained/results", "experiments/neuron_r/results", "experiments/holes_fill/results"
E136 = load_experiment("share_untrained", "run")  # Job: the trained share (route_split) and the SVD and rotated shares
HF = load_experiment("holes_fill", "run")  # Job: TopK neurons and fair steering; stage0: its controls
RULES = load_experiment("holes_fill", "rules")  # E143's filed rule: call, wider, compare
HF.RESULTS = LARGE  # holes_fill's stage 0 writes its record on a stop here (stage0.json), never into holes_fill's results
STEP = 30
log = KD.log
_json: dict[str, Any] = {}


class Stop(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


def load(rel: str) -> Any:
    if rel not in _json:
        _json[rel] = json.loads((REPO / rel).read_text())
    return _json[rel]


def dig(x: Any, keys: list[str]) -> Any:
    for k in keys:
        x = x[k]
    return x


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def write(name: str, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"{name}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(RESULTS / name)


def check(checks: dict[str, Any], name: str, value: float, tol: float) -> None:
    """A filed control: |value| <= tol (nan misses), recorded and logged; a miss raises Stop."""
    ok = abs(value) <= tol
    checks[name] = {"value": value, "tol": tol, "pass": ok}
    log(f"[check {name}] {value:.4g} (tol {tol:g}): {ok}")
    if not ok:
        raise Stop(f"{name}: {value:.6g} over {tol:g}")


def state_name(m: str, s: int) -> str:
    return f"e58/{m}_s{s}"


def state_file(m: str, s: int) -> Path:
    return LARGE / f"dW_{m}_s{s}.pt"


def saved_read(inv: dict[str, Any], m: str, s: int) -> dict[str, Any]:
    file, key = inv["maps"][m]["saved"][str(s)]
    return load(file)["reads"][key]


def seed_range(inv: dict[str, Any], m: str) -> float:
    r = [saved_read(inv, m, s)["r"] for s in inv["seeds"]]
    return max(r) - min(r)


def inventories(inv: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """E136's inventory format for this run's states plus the reader control state (E136's Job loads every task it is given),
    and E136's whole inventory with them added (holes_fill's Job loads tasks lazily; its stage 0 reads E136's states)."""
    e136 = load(f"{E136_R}/inventory.json")
    c3 = inv["reader_control_state"]
    c3_task = e136["states"][c3]["task"]
    tasks = {f"e58/{m}": {"format": "kv", "path": spec["windows"], "batch": 64} for m, spec in inv["maps"].items()}
    states = {state_name(m, s): {"task": f"e58/{m}", "file": str(state_file(m, s).relative_to(REPO))} for m in inv["maps"] for s in inv["seeds"]}
    mine = {"tasks": {**tasks, c3_task: e136["tasks"][c3_task]}, "states": {**states, c3: e136["states"][c3]}}
    return mine, {"tasks": {**e136["tasks"], **mine["tasks"]}, "states": {**e136["states"], **mine["states"]}, "pairs": e136["pairs"]}


class Run:
    """One model shared by the three experiments' readers."""

    def __init__(self, inv: dict[str, Any], P: dict[str, Any]) -> None:
        self.inv, self.T = inv, P["thresholds"]
        mine, full = inventories(inv)
        self.e136 = E136.Job(mine)  # loads the model and the SVD decompositions
        self.e136.k = {k: int(v) for k, v in load(f"{E136_R}/stage0.json")["k_m"].items()}  # fixed by E136 before any state
        self.net: ST.Net = self.e136.net
        hf = load(f"{E143_R}/predictions.json")
        self.hf = HF.Job(self.net, full, hf["settings"], hf["thresholds"])
        nr = load(f"{E98_R}/predictions.json")["settings"]
        self.nr_batch, self.nr_floor = nr["batch"], nr["gain_floor"]
        self.nr_reads: dict[str, NR.Reads] = {}

    def neuron_r(self, state: str) -> dict[str, Any]:
        """E98's cells and shares on the task's held-out windows (`steering.load_task`, as neuron_r's load_reads)."""
        task = self.e136.inv["states"][state]["task"]
        if task not in self.nr_reads:
            path = REPO / self.e136.inv["tasks"][task]["path"]
            self.nr_reads[task] = NR.task_reads(ST.load_task(self.net, path, task, task))
        dW = {k: v.float() for k, v in torch.load(REPO / self.e136.inv["states"][state]["file"], weights_only=True).items()}
        return NR.shares(NR.state_cells(self.net, self.nr_reads[task], dW, self.nr_batch), self.nr_floor)


# ----------------------------------------------------------------------------- stage 0


def readers(run: Run, res: dict[str, Any]) -> None:
    """(a) holes_fill's stage 0 as filed, then each reader on the control state against its own saved reading."""
    try:
        res["holes_fill_stage0"] = HF.stage0(run.hf)  # sets run.hf.ks (k 64 and k matched per block)
    except HF.Stopped as e:
        raise Stop(f"holes_fill stage 0: {e} (its record: results/large/stage0.json)") from e
    T, ch, c3 = run.T, res["checks"], run.inv["reader_control_state"]
    e136, e98 = load(f"{E136_R}/stage1.json")["states"][c3], load(f"{E98_R}/cells.json")["rows"][c3.replace("/", "_")]
    e143 = load(f"{E143_R}/stage1.json")["states"][c3]
    got = run.e136.read_state(c3)
    for k in ("trained", "svd", "rotated"):
        check(ch, f"a E136 {k} {c3}", got[k] - e136[k], T["reader_share_tol"])
    nr = run.neuron_r(c3)
    for k in ("r_n", "r_half"):
        check(ch, f"a E98 {k} {c3}", nr[k] - e98[k], T["reader_neuron_tol"])
    hf, _ = run.hf.state(c3, 0)
    for k in ("k64", "kmatched"):
        check(ch, f"a E143 neuron {k} {c3}", hf["neuron"][k]["share"] - e143["neuron"][k]["share"], T["reader_neuron_tol"])
    for k in ("pooled", "perkey"):
        check(ch, f"a E143 steering {k} {c3}", hf["steering"][k]["share"] - e143["steering"][k]["share"], T["reader_share_tol"])


def refit(run: Run, m: str, s: int, res: dict[str, Any]) -> None:
    """(b) one map and seed: fine-tune, save, read the share from the saved file, check it against E58's."""
    spec, name = run.inv["maps"][m], state_name(m, s)
    want, got = dig(load(spec["windows_sha256"][0]), spec["windows_sha256"][1]), KD.sha256(REPO / spec["windows"])
    if got != want:
        raise Stop(f"{spec['windows']}: sha256 {got}, E58 recorded {want}")
    w = torch.load(REPO / spec["windows"], weights_only=False)
    ft = KD.finetune(KD.Ctx(run.net, {}, {}, {}, LARGE), w, s, (STEP,))  # keydep.finetune reads only ctx.net
    d = ft.deltas[STEP]
    rec: dict[str, Any] = {"p_heldout": ft.evals[STEP]["p_o"], "elapsed_s": ft.evals[STEP]["elapsed_s"]}
    if kept_path := spec["saved_weights"].get(str(s)):  # compared in the kept file's dtype; reported, not a stop
        kept = torch.load(REPO / kept_path, weights_only=True)
        rec["kept"] = {"file": kept_path, "max_abs_diff": max(float((d[k].to(kept[k].dtype).float() - kept[k].float()).abs().max()) for k in d)}
    path = state_file(m, s)
    torch.save({k: v.bfloat16() for k, v in d.items()}, path)
    rec |= {"file": str(path.relative_to(REPO)), "sha256": KD.sha256(path)}
    rs = run.e136.trained(name, run.e136.delta(name))  # the share of the saved bf16 update
    saved = saved_read(run.inv, m, s)
    rec |= {"r": rs["shares"]["r"], "gain": rs["gain"], "saved_r": saved["r"], "saved_gain": saved["G_logodds_route"]}
    res["states"][name] = rec
    say(f"0 refit {name}", rec)
    try:
        check(res["checks"], f"b reproduce {name}", rec["r"] - rec["saved_r"], run.T["reproduce_tol"][m])
    finally:
        write("stage0.json", res)


def stage0(run: Run) -> None:
    path = RESULTS / "stage0.json"
    res: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}, "checks": {}}
    if "stopped" in res:
        raise Stop(f"stage 0 stopped before ({res['stopped']}): move stage0.json aside to start again")
    res |= {"host": socket.gethostname(), "card": torch.cuda.get_device_name()}
    try:
        readers(run, res)
        write("stage0.json", res)
        for m in run.inv["maps"]:
            for s in run.inv["seeds"]:
                ok = res["checks"].get(f"b reproduce {state_name(m, s)}", {}).get("pass")
                if not (ok and KD.sha256(state_file(m, s)) == res["states"][state_name(m, s)]["sha256"]):
                    refit(run, m, s, res)
    except Stop as e:
        res["stopped"] = str(e)
        write("stage0.json", res)
        raise
    res["pass"] = True
    write("stage0.json", res)


# ----------------------------------------------------------------------------- stage 1

READ = {
    "share": lambda r: r["e136"]["trained"],
    "untrained": lambda r: r["e136"]["svd"],
    "untrained_rotated": lambda r: r["e136"]["rotated"],
    "neuron": lambda r: r["neuron_r"]["r_n"],
    "neuron_half": lambda r: r["neuron_r"]["r_half"],
    "neuron_k64": lambda r: r["holes_fill"]["neuron"]["k64"]["share"],
    "neuron_kmatched": lambda r: r["holes_fill"]["neuron"]["kmatched"]["share"],
    "steer_pooled": lambda r: r["holes_fill"]["steering"]["pooled"]["share"],
    "steer_perkey": lambda r: r["holes_fill"]["steering"]["perkey"]["share"],
}


def read_state(run: Run, name: str, i: int) -> dict[str, Any]:
    e136 = run.e136.read_state(name)
    nr = run.neuron_r(name)
    if not (nr["identity"] <= run.T["identity_tol"] and abs(nr["telescope"]) <= run.T["telescope_tol"]):
        raise Stop(f"{name}: E98's identity {nr['identity']:.3g} or telescope {nr['telescope']:.3g} over its tolerance")
    hf, _ = run.hf.state(name, i)
    return {"e136": {k: e136[k] for k in ("trained", "svd", "rotated", "gain", "gain_fp32")},
            "neuron_r": {k: nr[k] for k in ("G", "num", "r_n", "r_half", "d_n", "i_n", "e_n", "readable", "p_target", "identity", "telescope")},
            "holes_fill": hf}  # fmt: skip


def score(inv: dict[str, Any], states: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    """E143's rule per reading and pair; each family against the share; the filed predictions beside."""
    T, ids = P["thresholds"], [p["id"] for p in inv["pairs"]]
    side = lambda m, f: [float(f(states[state_name(m, s)])) for s in inv["seeds"]]  # float(): a saved nan is the string "nan"
    calls = {k: {p["id"]: RULES.call(side(p["reuse"], f), side(p["new"], f), T) for p in inv["pairs"]} for k, f in READ.items()}
    fam = {"untrained": calls["untrained"], "neuron": calls["neuron"], "neuron_half": calls["neuron_half"],
           "neuron_top": {p: RULES.wider(calls["neuron_k64"][p], calls["neuron_kmatched"][p]) for p in ids},
           "steering": {p: RULES.wider(calls["steer_pooled"][p], calls["steer_perkey"][p]) for p in ids}}  # fmt: skip
    verdicts = {f: RULES.compare(c, calls["share"]) for f, c in fam.items()}
    for v in verdicts.values():
        if v["share_count"] == 0 and v["verdict"] in ("rival A", "rival B"):  # the filed edge: no bar to meet
            v["verdict"] = "no call: the share separates none"
    pr = P["predictions"]
    filed = {**{f"share {p}": [calls["share"][p]["call"], pr["share_calls"][p]] for p in ids},
             **{f"verdict {f}": [verdicts[f]["verdict"], pr["verdicts"][f]] for f in fam}}
    return {"calls": calls, "families": fam, "verdicts": verdicts,
            "filed": {k: {"read": a, "filed": b, "as_filed": a == b} for k, (a, b) in filed.items()}}  # fmt: skip


def stage1(run: Run, P: dict[str, Any], t_job: float) -> None:
    path = RESULTS / "stage1.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    order = [state_name(m, s) for m in run.inv["maps"] for s in run.inv["seeds"]]
    t0, n_new = time.time(), 0
    for i, name in enumerate(order):
        if name in out["states"]:
            continue
        out["states"][name] = st = read_state(run, name, i)
        n_new += 1
        say(f"1 {name}", {**st["e136"], **{k: f(st) for k, f in READ.items()}})
        write("stage1.json", out)
        left = sum(n not in out["states"] for n in order)
        out["projected_minutes"] = minutes = ((time.time() - t0) / n_new * left + (time.time() - t_job)) / 60
        log(f"[1 time] {len(out['states'])} of {len(order)} states; projected {minutes:.1f} min for the job")
        if n_new >= 4 and minutes > run.T["projected_limit_min"]:
            out["stopped"] = f"projected {minutes:.1f} min over {run.T['projected_limit_min']}"
            write("stage1.json", out)
            raise Stop(out["stopped"])
    out["score"] = score(run.inv, out["states"], P)
    write("stage1.json", out)
    for k in ("calls", "families", "verdicts", "filed"):
        say(f"1 score {k}", out["score"][k])


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    require_filed(INV)
    P, inv = json.loads(PRED.read_text()), json.loads(INV.read_text())
    T = P["thresholds"]
    assert P["experiment"].startswith("E147") and T["gain_floor"] == MIN_GAIN
    off = {m: (seed_range(inv, m), T["reproduce_tol"][m]) for m in inv["maps"] if abs(seed_range(inv, m) - T["reproduce_tol"][m]) > 1e-9}
    if off:
        sys.exit(f"filed reproduce_tol differs from the saved seed ranges: {off}")
    files = [spec["windows"] for spec in inv["maps"].values()] + [f for spec in inv["maps"].values() for f in spec["saved_weights"].values()]
    if missing := [f for f in files if not (REPO / f).is_file()]:
        sys.exit(f"files missing: {missing}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E58 trained and E136/E98/E143 read
    run = Run(inv, P)
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()}")
    try:
        stage0(run)
        print("STAGE 0 DONE", flush=True)
        stage1(run, P, t_job)
        print("STAGE 1 DONE", flush=True)
    except Stop as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
