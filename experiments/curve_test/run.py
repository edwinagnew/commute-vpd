"""E192 (cell P1.23): a filed test of E191's curve reading of E184 on fresh fine-tunes, under the fourth reuse wording
(tested, not adopted; no change to decisions.md). Card: card.md; filed rules: results/predictions.json (the run refuses
unless it is committed unchanged). The search is E184's (commute.rank_read) on list all, its two greedy passes, run to 16
names with no stop at a bar and no stall rule; no free-direction fits. After every step the chosen names' closure is read
twice: on the full next-token distribution, and on the read windows whose last position is the task's answer position.

Stages, cheapest first, in one job; each item writes results/<item>.json after every step and resumes from it:
    controls  l1_neuron, l2_triggered, l4_bit as E184 built them (its calibrated s, its gap repeated within reload_tol),
              searched as every item is; E184's filed run-1 readings for them on list all -> results/controls.json; exit 1
              on a miss, before any training
    train     seeds 2 and 3 of the ten sides (train.py) -> results/large/<side>_s<seed>.pt, records in results/train.json
    search    every trained item, `STAGE search <item> DONE` after each
    score     rule.score -> results/score.json
Inputs (pod: sync, and stage every results/large by hard link): E184's results/pcs.pt and control JSONs; what E173's
setup reads (E137, E154, E157 files); open_pair large/windows_{elicit,cipher}_trained.pt and large/amend1/windows_copy_*;
regating_table large/windows_{ident32,rand32}.pt; case_label and fresh_reuse_pair large/task.pt; the key-table windows
files old_tasks reads. This folder's results/large must be a real directory on the pod. Exit 0 done, 1 a stop or refusal.

    python experiments/curve_test/run.py          (on a pod: PRED_SHA=<sha of results/predictions.json> python ...)
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.rank_read as RK
import commute.variable_read as VR
from commute import old_tasks as OT
from commute.jobs import hard_exit, load_experiment, require_filed

SIM: Any = load_experiment("simplest_intervention", "run")  # E184: setup, control_maker, misses, log, dump, stop
TR: Any = load_experiment("curve_test", "train")
RU: Any = load_experiment("curve_test", "rule")
EXPERIMENT = "E192"
WORDING = "under the fourth wording, curve reading, provisional (testing only, not adopted)"
HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
LARGE = RESULTS / "large"
CONTROLS = ("l1_neuron", "l2_triggered", "l4_bit")
ITEMS = tuple(f"{side}_s{s}" for side in TR.SIDES for s in TR.SEEDS)
log, dump, stop = SIM.log, SIM.dump, SIM.stop
Made = tuple[VR.Target, Tensor, Tensor, Callable[[Tensor], Tensor] | None, dict[str, Any]]  # F, fit, read, answer mask, info


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


# ---- the search: E184's greedy passes to 16 names, every step read twice --------------------------------------------


def pass_to_cap(c: Any, m: RK.Model, d: RK.Data, ans: tuple[Tensor, Tensor] | None, out: dict[str, Any], r: str, path: Path) -> None:
    """One greedy pass (r "1": writes only, "2": writes and reads) on list all to n = cap. RK.greedy runs with no bar and no
    stall rule (c.cfg.patience is out of reach), so it returns only at the cap or when no name lowers the check KL; then a
    step with no name is recorded and the search goes on with the next scoring seed, up to cap such steps."""
    st = out.setdefault("lists", {}).setdefault("all", {}).setdefault(r, {"curve": [], "state": [], "done": False})
    if st["done"]:
        return
    I, cap = RK.Interv.from_json(st["state"], m.net.device), c.cfg.cap

    def tick() -> None:
        e = st["curve"][-1]
        e["closure_answer"] = 1 - RK.mean_kl(m, I, *ans, c.cfg.eval_batch) / out["kl_answer"] if ans else None
        st["state"] = I.to_json()
        log(f"{path.stem} [all] pass {r} n {e['n']}: closure {e['closure']:.4f}, answer {e['closure_answer']}, check KL {e['check_kl']:.5g}"
            + (" (no name lowers the check KL)" if e.get("no_name") else f" (+ {e['names'][-1]})"))
        dump(path, out)

    while I.n < cap and sum(e.get("no_name", False) for e in st["curve"]) < cap:
        RK.greedy(m, d, I, st["curve"], m.sp.masks("all", m.net.device), r == "2", c.cfg, math.inf, tick, c.cfg.search_seed)
        if I.n < cap:
            st["curve"].append({"n": I.n, "closure": RK.closure(m, I, d, c.cfg), "check_kl": RK.check_kl(m, I, d, c.cfg),
                                "names": I.names(m.sp), "no_name": True})  # fmt: skip
            tick()
    st["done"] = True
    dump(path, out)


def search(c: Any, name: str, make: Callable[[], Made], floor: float) -> dict[str, Any]:
    path = RESULTS / f"{name}.json"
    out = load(path)
    if out.get("done"):
        return out
    F, fit, read, answer, info = make()
    out |= {"experiment": EXPERIMENT, "item": name, "wording": WORDING, **info}
    m, d = RK.prepare(c.net, c.sp, F, fit, read, c.cfg, c.cfg.subset_seed)
    ans = None
    if answer is not None:
        mask = answer(d.read)
        ans = (d.read[mask], d.lp_read[mask.to(d.lp_read.device)])
        out |= {"n_answer": int(mask.sum()), "kl_answer": RK.mean_kl(m, RK.Interv(), *ans, c.cfg.eval_batch) if mask.any() else 0.0}
        ans = ans if out["kl_answer"] >= floor else None  # under the floor: no answer closure is divided
    out["kl"] = d.kl0
    log(f"{name}: KL(F || M) {d.kl0:.6g} nats on {d.read.shape[0]} read windows; answer windows {out.get('n_answer')}, KL {out.get('kl_answer')}")
    if d.kl0 >= floor:
        for r in ("1", "2"):
            pass_to_cap(c, m, d, ans, out, r, path)
    out["done"] = True
    dump(path, out)
    del m, d, F
    torch.cuda.empty_cache()
    return out


# ---- the items ------------------------------------------------------------------------------------------------------


def control(c: Any, name: str, th: dict[str, Any]) -> Made:
    """E184's control at E184's calibrated s; its gap (mean TV to M on E173's read windows) must repeat E184's."""
    e184 = load(SIM.RESULTS / f"{name}.json")
    make, info = SIM.control_maker(c, name)
    F = make(e184["s"])
    gap = VR.mean_tv(F, VR.net_target(c.net), c.e.read, c.e.cfg.read_batch)["mean"]
    out = {**info, "s": e184["s"], "gap": gap, "gap_e184": e184["gap"]}
    log(f"{name}: s {e184['s']:.6f}, gap {gap:.6f} (E184 {e184['gap']:.6f})")
    if abs(gap - e184["gap"]) > th["reload_tol"]:
        stop(RESULTS / f"{name}.json", out, f"{name}: the gap does not repeat E184's within reload_tol")
    return F, c.e.fit, c.e.read, None, out


def answers(task: OT.OldTask) -> Callable[[Tensor], Tensor]:
    """The windows whose last position is the task's answer position, as old_tasks builds its "shows" windows: a key table's
    marker then key; a case or repeat word then the marker."""
    if task.kind == "kv":
        files = [torch.load(p, weights_only=False)["task"] for p in task.sources]
        marker, keys = int(files[0].marker), torch.tensor([k for f in files for k in f.keys])
        return lambda t: (t[:, -2] == marker) & torch.isin(t[:, -1], keys)
    t0 = torch.load(task.sources[0], weights_only=False)
    pool = [w for pr in [*t0["train_pairs"], *t0["held_pairs"]] for w in pr] if task.kind == "case" else [*t0["train_words"], *t0["held_words"]]
    marker, words = int(t0["marker"]), torch.tensor([int(w) for w in pool])
    return lambda t: (t[:, -1] == marker) & torch.isin(t[:, -2], words)


def trained(c: Any, name: str, rec: dict[str, Any]) -> Made:
    """F = M plus the saved fine-tune, on the windows E184 read that side on (E173's)."""
    side, path = name.rsplit("_s", 1)[0], LARGE / f"{name}.pt"
    info = {"state": str(path.relative_to(HERE)), "learned": rec["learned"]}
    if side in ("twin", "fact"):
        F = VR.net_target(VR.load_state(c.net, path, OT.clone_net))
        return F, c.e.fit, c.e.read, lambda t: VR.xor2_keyed(c.e.x, t), info
    task = OT.TASKS[side]
    if task.group not in c.cache:
        c.cache.clear()
        c.cache[task.group] = OT.task_windows(task, c.e.sets, c.e.cfg)[:2]
    return VR.net_target(OT.old_f(c.net, path)), *c.cache[task.group], answers(task), info


# ---- the stages -----------------------------------------------------------------------------------------------------


def controls(c: Any, pred: dict[str, Any]) -> bool:
    th, out = pred["thresholds"], {"experiment": EXPERIMENT, "readings": {}, "misses": []}
    for name in CONTROLS:
        rec = search(c, name, lambda n=name: control(c, n, th), th["kl_floor"])
        rd = {str(b): RK.reading(rec["kl"], th["kl_floor"], rec.get("lists", {}).get("all", {}), b) for b in th["control_bars"]}
        out["readings"][name] = rd
        out["misses"] += [x for s in pred["control_stop"][name] for x in SIM.misses(name, {"readings": {"all": rd}, **rec}, s)]
        log(f"{name} [all] readings: {rd}")
    out["passed"] = not out["misses"]
    dump(RESULTS / "controls.json", out)
    log(f"CONTROLS {'passed' if out['passed'] else 'STOP: ' + str(out['misses'])}")
    return out["passed"]


def train_all(c: Any) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E58, E112 and E125 set it
    LARGE.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "train.json"
    recs = load(path)
    for name in ITEMS:
        if name not in recs:
            recs[name] = TR.train(c.net, name, LARGE / f"{name}.pt", log)
            dump(path, recs)
            log(f"STAGE train {name} DONE: learned {recs[name]['learned']}")
    return recs


def main() -> int:
    pred = json.loads(PRED.read_text())
    th = pred["thresholds"]
    c = SIM.setup(SIM.Config(), load(SIM.PRED)["thresholds"], "1")  # E184's net, E173 context, principal directions
    c.cfg = dataclasses.replace(c.cfg, patience=10**9)  # no stall rule; E184's sizes and seeds otherwise
    if not controls(c, pred):
        return 1
    recs = train_all(c)
    for name in ITEMS:
        search(c, name, lambda n=name: trained(c, n, recs[n]), th["kl_floor"])
        print(f"STAGE search {name} DONE", flush=True)
    score = RU.score({n: load(RESULTS / f"{n}.json") for n in ITEMS}, TR.SEEDS, th["bar"], th["kl_floor"])
    dump(RESULTS / "score.json", {"experiment": EXPERIMENT, "wording": WORDING, "bar": th["bar"], **score})
    for reading, s in score.items():
        log(f"{reading}: {s['verdict']} ({s['n_holds']} of {s['n_read']} read hold); " + "; ".join(
            f"{k} {v['gap']:.4f}" if v["read"] else f"{k} not read {v['why']}" for k, v in s["pairs"].items()))
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
