"""recruitment_test (E38): does the fine-tuned model itself act through the subcomponents Gamma recruits at the trigger?
Plan: reports/plans/2026-10-27_plan_regating-recruitment.md (confirmed 2026-10-27). Definitions: the docstrings of
commute.recruitment (stages 0-1), commute.routing (stage 2), commute.census (stage 3).

States: E18's recipe regenerated in-job (arm-A fine-tune of all 24 matrices, Adam 3e-5, batch 32, 10 steps; seeds 0 and
1 to the task's target; seed 0 to ' http' at the same triggers, the null), do-nothing placement, Gamma frozen. The
recruited set S of a state: alive subcomponents whose mean trigger importance over the training windows rises by more
than 0.3 (`recruit.switched(regating.shift_of(...), 0.3)`). Every number reads the full model (target weights W0 + dW,
no Gamma, no masks) on the held-out windows, except where named.

  run.py --stages 0      the apparatus, alone: fine-tunes the nine states, caches each weight change in
                         results/large/states/, checks them against E18 -> results/stage0.json; exit 1 on any fail
  run.py --stages 1,2,3  refuses unless stage0.json passed; stage 1 (recruitment test) -> stage1.json, and stops there
                         when the plan's decision tree reads 'fail'; stage 2 (routing trace) -> stage2.json; stage 3
                         (piece census) -> stage3.json and results/large/stage3_pieces.json
Every scored number is also printed to the log (and results/large/full.log). Refuses to start without
results/predictions.json. Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt,
large/windows.pt}, experiments/task_ladder/results/large/windows_<task>.pt,
experiments/regating_leak/results/{regating_leak.json, large/<task>_s{0,1}_shift.pt}; Pile val rows 0..2000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import census as CE
from commute import recruit as R
from commute import recruitment as RT
from commute import regating as G
from commute import routing as RO
from commute.faithfulness import model_state
from commute.signatures import top_by

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"  # may be a symlink into ~/commute-data/results-large/; created through it, never replaced
STATES = LARGE / "states"
PRED = RESULTS / "predictions.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E18 = EXP / "regating_leak" / "results"


@dataclass(frozen=True)
class Cfg:
    tasks: tuple[str, ...] = ("emoticon", "r1_t534_v369", "r2_t359_v816")
    seeds: tuple[int, ...] = (0, 1)
    null_token: int = 3944  # ' http', E18's null target
    lr: float = 3e-5
    steps: int = 10
    thr: float = G.THR  # 0.3: the recruited set
    n_random: int = G.N_RANDOM  # 20 random same-size sets (recruit.random_like seeds 0..19)
    top_k: tuple[int, ...] = (25, 50, 100, 200, 400)
    t_tol: float = 0.02  # stage 0: T0 and T1 against E18
    p_tol: float = 0.01  # stage 0: held-out P(target) against E18
    jaccard_min: float = 0.95  # stage 0: in-job S against the saved one
    size_tol: float = 0.05
    empty_tol: float = 1e-5  # stage 0: removing nothing
    busy_kl_min: float = 1.0  # stage 0: removing the busy set from W0 moves the trigger by >= 1 nat of KL
    busy_min_n: int = 100  # ... checked where the busy set has >= 100 pieces (emoticon); reported on the small sets
    min_gain: float = 1.0  # a state's removed shares are scorable only with G >= 1 nat (E18's rule)
    pile: tuple[str, int, int] = ("val", 0, 2000)  # stages 2 and 3
    min_nat: int = 200  # stage 2: Pile positions before the target, else that target is unscored
    n_census_random: int = 5
    trigger_share: float = 0.1  # stage 3: a task's trigger tokens are those at >= 10% of its training triggers
    census_ratio: float = 3.0  # stage 3's one check: S's share of target pieces >= 3 x the random sets'

    def __post_init__(self) -> None:
        assert self.top_k == tuple(sorted(self.top_k)) and self.n_census_random <= self.n_random
        assert self.pile[2] >= 2000 and self.seeds == (0, 1), "the plan: 2,000 census rows; P1 reads two seeds"


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(k: int, obj: dict[str, Any]) -> None:
    (RESULTS / f"stage{k}.json").write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def names(task: str) -> list[str]:
    return [f"{task}/s{s}" for s in CFG.seeds] + [f"{task}/null"]


def references() -> dict[str, Any]:
    """E18's numbers (regating_leak.json): T0 = G1.<task>.cells.T0; T1 of seed s = T0 + tasks.<task>.seeds.<s>.gain;
    held-out P(target) = tasks.<task>.seeds.<s>.p_heldout; the null's P(' http') = tasks.<task>.null.p_heldout."""
    e18 = json.loads((E18 / "regating_leak.json").read_text())
    out = {}
    for n in CFG.tasks:
        T0, s = e18["G1"][n]["cells"]["T0"], e18["tasks"][n]
        out[n] = {"T0": T0, "T1": {str(k): T0 + s["seeds"][str(k)]["gain"] for k in CFG.seeds},
                  "p": {str(k): s["seeds"][str(k)]["p_heldout"] for k in CFG.seeds}, "p_null": s["null"]["p_heldout"]}
    return out


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.device, self.model = device, vpd.load_component_model(device)
        self.W0, self.comps, self.tok = vpd.target_weights(self.model), vpd.component_weights(self.model), vpd.tokenizer()
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}
        self.tasks: dict[str, G.Task] = {}
        self.dw: dict[str, dict[str, Tensor]] = {}
        self.shift: dict[str, dict[str, Tensor]] = {}

    def prepare(self, task: str) -> G.Task:
        self.tasks[task] = G.prepare(self.model, task, *G.load_windows(task, TRANSPORT, LADDER))
        return self.tasks[task]

    def weights(self, name: str) -> dict[str, Tensor]:
        return {k: w + self.dw[name][k].to(w.device, w.dtype) for k, w in self.W0.items()}

    def add_state(self, name: str, dW: dict[str, Tensor]) -> None:
        self.dw[name] = dW
        self.shift[name] = G.shift_of(self.model, self.W0, dW, self.tasks[name.split("/")[0]], self.alive)[0]

    def S(self, name: str) -> R.Sel:
        return R.switched(self.shift[name], CFG.thr)

    def load_states(self, s0: dict[str, Any]) -> None:
        """Stages >= 1 in a new job: W0's task readings again, and the weight changes stage 0 cached and checked."""
        for n in CFG.tasks:
            self.prepare(n)
            for name in names(n):
                path = STATES / f"{name.replace('/', '_')}.pt"
                if not path.is_file() or sha256(path) != s0["states"][name]["sha256"]:
                    sys.exit(f"{path} is missing or not the file stage 0 checked: rerun --stages 0")
                self.add_state(name, torch.load(path, weights_only=True))


# ----------------------------------------------------------------------------- stage 0: the apparatus


def make_state(run: Run, task: str, name: str, ref: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune one state, cache its weight change, and check it against E18 (T1, P(target), the recruited set)."""
    from commute.local_faith import route_cells

    t, null = run.tasks[task], name.endswith("null")
    seed = 0 if null else int(name[-1])
    dW, ev = G.finetune(run.model, t, CFG.null_token if null else t.target, "adam", CFG.lr, CFG.steps, seed, run.device, log)
    path = STATES / f"{name.replace('/', '_')}.pt"
    torch.save(dW, path)
    run.add_state(name, dW)
    saved = torch.load(E18 / "large" / f"{task}_s{seed}_shift.pt", weights_only=False)["shift_null" if null else "shift"]
    S, S_saved = run.S(name), R.switched({k: v.float().cpu() for k, v in saved.items()}, CFG.thr)
    rec: dict[str, Any] = {"sha256": sha256(path), "heldout_eval": ev, "n_S": R.size(S), "n_S_saved": R.size(S_saved),
                           "jaccard_saved": R.jaccard(S, S_saved)}
    rec["set_ok"] = rec["jaccard_saved"] >= CFG.jaccard_min and abs(R.size(S) / R.size(S_saved) - 1) <= CFG.size_tol
    if null:
        rec.update(p=ev["p_o"], p_ref=ref["p_null"], T1_ok=True)
    else:
        with G.state_of(run.model, run.W0, dW):
            cur = route_cells(run.model, t.held.tokens, t.half, t.target, G.BATCH)
        rec.update(T1=cur["T"], T1_ref=ref["T1"][str(seed)], p=cur["p_T"], p_ref=ref["p"][str(seed)])
        rec["T1_ok"] = abs(rec["T1"] - rec["T1_ref"]) <= CFG.t_tol
    rec["p_ok"] = abs(rec["p"] - rec["p_ref"]) <= CFG.p_tol
    rec["ok"] = rec["T1_ok"] and rec["p_ok"] and rec["set_ok"]
    log(f"[stage0 {name}] T1 {rec.get('T1', float('nan')):.4f} (E18 {rec.get('T1_ref', float('nan')):.4f}) | P {rec['p']:.4f} "
        f"(E18 {rec['p_ref']:.4f}) | kl_other {ev['kl_other']:.4f} | |S| {rec['n_S']} saved {rec['n_S_saved']} J "
        f"{rec['jaccard_saved']:.4f} | {'ok' if rec['ok'] else 'FAIL'}")
    return rec


def apparatus(run: Run, task: str) -> dict[str, Any]:
    """Removing nothing returns each state's T1 (and W0's T0 after the states); removing the busy set (top-|S| alive by
    W0's mean trigger importance over the training windows, |S| of seed 0) from W0 moves the trigger by >= 1 nat."""
    t, model = run.tasks[task], run.model
    held, h, tok = t.held.tokens, t.half, int(t.target)
    empty = {k: torch.zeros(0, dtype=torch.long) for k in run.alive}
    ref0 = RT.window_logits(model, held)
    T0 = RT.readout(model, held, h, tok, ref0)["lo"]
    out: dict[str, Any] = {"T0_path": T0, "empty": {}}
    for name in names(task)[:-1]:
        W = run.weights(name)
        with model_state(model, target=W):
            ref1 = RT.window_logits(model, held)
            T1 = RT.readout(model, held, h, tok, ref1)["lo"]
        e = RT.removed(model, W, run.comps, empty, held, h, tok, ref1)
        out["empty"][name] = {"T1_path": T1, "T1_empty": e["lo"], "diff": e["lo"] - T1, "kl_other": e["kl_other"]}
        del ref1
    busy = top_by(t.m0, run.alive, R.size(run.S(f"{task}/s0")))
    b = RT.removed(model, run.W0, run.comps, busy, held, h, tok, ref0)
    out["restore_diff"] = RT.readout(model, held, h, tok, ref0)["lo"] - T0
    out["busy"] = {"n": R.size(busy), **b, "ok": R.size(busy) < CFG.busy_min_n or b["kl_trigger"] >= CFG.busy_kl_min}
    out["empty_ok"] = all(abs(v["diff"]) <= CFG.empty_tol for v in out["empty"].values()) and abs(out["restore_diff"]) <= CFG.empty_tol
    log(f"[stage0 {task}] empty removal diffs {[round(v['diff'], 8) for v in out['empty'].values()]} restore "
        f"{out['restore_diff']:.2e} | busy |{out['busy']['n']}| from W0: KL trigger {b['kl_trigger']:.4f} kl_other "
        f"{b['kl_other']:.4f} lo {b['lo']:.4f} (T0 {T0:.4f}) | {'ok' if out['empty_ok'] and out['busy']['ok'] else 'FAIL'}")
    return out


def stage0(run: Run) -> dict[str, Any]:
    refs = references()
    out: dict[str, Any] = {"refs": refs, "tasks": {}, "states": {}, "cfg": CFG.__dict__}
    for n in CFG.tasks:
        t = run.prepare(n)
        rec: dict[str, Any] = {"T0": t.base["T"], "T0_ref": refs[n]["T0"], "T0_ok": abs(t.base["T"] - refs[n]["T0"]) <= CFG.t_tol}
        log(f"[stage0 {n}] T0 {rec['T0']:.4f} (E18 {rec['T0_ref']:.4f}) | {'ok' if rec['T0_ok'] else 'FAIL'}")
        for name in names(n):
            out["states"][name] = make_state(run, n, name, refs[n])
        rec["apparatus"] = a = apparatus(run, n)
        rec["ok"] = rec["T0_ok"] and all(out["states"][x]["ok"] for x in names(n)) and a["empty_ok"] and a["busy"]["ok"]
        out["tasks"][n] = rec
        dump(0, out)
        if not rec["ok"]:
            break  # early stop: the states cached so far stay, a rerun of stage 0 overwrites them
    out["pass"] = len(out["tasks"]) == len(CFG.tasks) and all(v["ok"] for v in out["tasks"].values())
    dump(0, out)
    log(f"[stage0] {'PASS' if out['pass'] else 'FAIL: stop and fix'}")
    return out


# ----------------------------------------------------------------------------- stage 1: recruitment test


def stage1_state(run: Run, task: str, name: str, ref0: list[Tensor], T0: float) -> dict[str, Any]:
    t, model, alive = run.tasks[task], run.model, run.alive
    held, h, tok = t.held.tokens, t.half, int(t.target)
    W, S = run.weights(name), run.S(name)
    with model_state(model, target=W):
        own = RT.readout(model, held, h, tok, ref0)  # T1, and the fine-tune's own KL to W0
        ref1 = RT.window_logits(model, held)
    T1, gain = own["lo"], own["lo"] - T0
    sets = {"S": S, "busy": top_by(t.m0, alive, R.size(S)), "null": run.S(f"{task}/null"),
            **{f"top{k}": top_by(run.shift[name], alive, k) for k in CFG.top_k}}
    reads = {k: RT.removed(model, W, run.comps, v, held, h, tok, ref1) for k, v in sets.items()}
    rnd = [RT.removed(model, W, run.comps, R.random_like(S, alive, i), held, h, tok, ref1) for i in range(CFG.n_random)]
    w0 = RT.removed(model, run.W0, run.comps, S, held, h, tok, ref0)
    del ref1
    g = gain if gain >= CFG.min_gain else 0.0  # unscorable state: every share nan, so it neither passes nor fails a bar
    rho = {k: RT.share(T1, r["lo"], g) for k, r in reads.items()}
    rr = [RT.share(T1, r["lo"], g) for r in rnd]
    rec = {"T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain, "kl_ft": own["kl_other"],
           "kl_ft_trigger": own["kl_trigger"], "n": {k: R.size(v) for k, v in sets.items()}, "by_layer_S": R.by_layer(S),
           "J_busy_S": R.jaccard(sets["busy"], S), "J_null_S": R.jaccard(sets["null"], S), "rho": rho,
           "kl_other": {k: r["kl_other"] for k, r in reads.items()}, "kl_trigger": {k: r["kl_trigger"] for k, r in reads.items()},
           "random": {"rho": rr, "kl_other": [r["kl_other"] for r in rnd]}, "rho_S": rho["S"], "random_mean": sum(rr) / len(rr),
           "random_max": max(rr), "rho_null": rho["null"], "rho_S_W0": RT.share(T0, w0["lo"], g), "kl_other_S_W0": w0["kl_other"]}
    log(f"[stage1 {name}] T0 {T0:.4f} T1 {T1:.4f} G {gain:.4f} | kl_ft {rec['kl_ft']:.4f} | |S| {rec['n']['S']} by layer "
        f"{rec['by_layer_S']} | rho_S {rho['S']:.4f} random mean {rec['random_mean']:.4f} max {rec['random_max']:.4f} | "
        f"rho_busy {rho['busy']:.4f} rho_null {rho['null']:.4f} (|null| {rec['n']['null']}) | rho_S at W0 {rec['rho_S_W0']:.4f}")
    log(f"[stage1 {name}] kl_other S {rec['kl_other']['S']:.4f} busy {rec['kl_other']['busy']:.4f} null "
        f"{rec['kl_other']['null']:.4f} random mean {sum(rec['random']['kl_other']) / CFG.n_random:.4f} | J(busy, S) "
        f"{rec['J_busy_S']:.4f} J(null, S) {rec['J_null_S']:.4f} | top-k " + " ".join(
            f"k{k} rho {rho[f'top{k}']:.4f} kl {rec['kl_other'][f'top{k}']:.4f}" for k in CFG.top_k))
    log(f"[stage1 {name}] random draws {json.dumps([round(x, 4) for x in rr])}")
    return rec


def stage1(run: Run) -> dict[str, Any]:
    out: dict[str, Any] = {"states": {}}
    by_task: dict[str, dict[str, Any]] = {}
    for n in CFG.tasks:
        t = run.tasks[n]
        ref0 = RT.window_logits(run.model, t.held.tokens)
        T0 = RT.readout(run.model, t.held.tokens, t.half, int(t.target), ref0)["lo"]
        for name in names(n)[:-1]:
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = stage1_state(run, n, name, ref0, T0)
            dump(1, out)
        del ref0
    out.update(RT.readings(by_task))
    dump(1, out)
    log(f"[stage1] tasks {json.dumps(out['tasks'])} | fails {json.dumps(out['task_fails'])}")
    log(f"[stage1] reading '{out['reading']}' | P1 {json.dumps(out['P1'])} | P2 {json.dumps(out['P2'])} | P3 "
        f"{json.dumps(out['P3'])} | P2 fails while P1 passes: {out['P2_fails_while_P1_passes']}")
    return out


# ----------------------------------------------------------------------------- stage 2: routing trace


def stage2(run: Run, s1: dict[str, Any]) -> dict[str, Any]:
    from commute.data import pile_rows

    model, rows = run.model, pile_rows(*CFG.pile, log=log)
    pm = RO.pile_means(model, rows, {n: R.precede_mask(rows, int(run.tasks[n].target)) for n in CFG.tasks})
    out: dict[str, Any] = {"n_positions": {n: pm["by"][n]["n"] for n in CFG.tasks}, "tasks": {}}
    for n in CFG.tasks:
        t = run.tasks[n]
        mu_t, mu_all = pm["by"][n]["mean"], pm["all"]

        def a1(name: str, t: G.Task = t) -> dict[str, Tensor]:
            with model_state(model, target=run.weights(name)):
                return RO.trigger_inputs(model, t.held.tokens, t.half)

        a0 = RO.trigger_inputs(model, t.held.tokens, t.half)
        tr_null = RO.trace(a0, a1(f"{n}/null"), mu_t, mu_all)
        rec: dict[str, Any] = {"n_positions": pm["by"][n]["n"], "stage1_task_fails": s1["task_fails"][n], "null": tr_null, "seeds": {}}
        rec["scorable"] = rec["n_positions"] >= CFG.min_nat and not rec["stage1_task_fails"]
        for name in names(n)[:-1]:
            bl = R.by_layer(run.S(name))
            layer = int(max(bl, key=lambda x: bl[x]))  # the layer holding most of S; ties go to the lowest layer
            tr = RO.trace(a0, a1(name), mu_t, mu_all)
            rec["seeds"][name[-1]] = {"trace": tr, "P4": RO.p4_state(tr, tr_null, layer)}
            log(f"[stage2 {name}] layer {layer} | " + " | ".join(
                f"{k} cos {v['cos']:.3f} (null {tr_null[k]['cos']:.3f}, random {v['random_cos']:.3f}) closer {v['closer']:.3f} "
                f"closer_all {v['closer_all']:.3f}" for k, v in tr.items()))
            log(f"[stage2 {name}] P4 {json.dumps(rec['seeds'][name[-1]]['P4'])}")
        rec["P4_pass"] = all(s["P4"]["pass"] for s in rec["seeds"].values()) if rec["scorable"] else "unscored"
        log(f"[stage2 {n}] {rec['n_positions']} Pile positions | null closer " + " ".join(
            f"{k} {v['closer']:.3f}" for k, v in tr_null.items()) + f" | P4 {rec['P4_pass']}")
        out["tasks"][n] = rec
        dump(2, out)
    scored = {n: v["P4_pass"] for n, v in out["tasks"].items() if v["scorable"]}
    out["P4"] = {"per_task": scored, "pass": all(scored.values()) if scored else "unscored"}
    dump(2, out)
    log(f"[stage2] P4 {json.dumps(out['P4'])}")
    return out


# ----------------------------------------------------------------------------- stage 3: piece census


def piece_row(run: Run, row: dict[str, Any], target: int, triggers: set[int]) -> dict[str, Any]:
    def named(lst: list[tuple[int, float, int]]) -> list[list[Any]]:
        return [[i, run.tok.decode([i]), round(r, 2), c] for i, r, c in lst]

    return {**row, "top_next": named(row["top_next"]), "top_cur": named(row["top_cur"]),
            "class": CE.classify(row, target, triggers)}


def stage3(run: Run) -> dict[str, Any]:
    from commute.data import pile_rows

    S0 = {n: run.S(f"{n}/s0") for n in CFG.tasks}
    rand = {n: [R.random_like(S0[n], run.alive, i) for i in range(CFG.n_census_random)] for n in CFG.tasks}
    sels = [S0[n] for n in CFG.tasks] + [x for n in CFG.tasks for x in rand[n]]
    union = {k: torch.cat([s[k] for s in sels]).unique() for k in run.alive}
    c = CE.run_census(run.model, pile_rows(*CFG.pile, log=log), union, log=log)
    table = CE.pieces_table(c)
    out: dict[str, Any] = {"n_positions": c.n_all, "n_pieces": len(table), "tasks": {}}
    big: dict[str, Any] = {}
    for n in CFG.tasks:
        t = run.tasks[n]
        trig, cnt = t.train.tokens[:, t.half].unique(return_counts=True)
        triggers = {int(x) for x, k in zip(trig, cnt, strict=True) if k >= CFG.trigger_share * len(t.train)}
        s, r = CE.class_shares(table, S0[n], int(t.target), triggers), [CE.class_shares(table, x, int(t.target), triggers) for x in rand[n]]
        r_mean = sum(x["target_share"] for x in r) / len(r)
        shift = run.shift[f"{n}/s0"]
        rows_S: list[dict[str, Any]] = sorted((piece_row(run, table[p], int(t.target), triggers) | {"rise": float(shift[p[0]][p[1]])}
                         for p in R.members(S0[n])), key=lambda x: -x["rise"])
        out["tasks"][n] = {"target": [int(t.target), run.tok.decode([int(t.target)])],
                           "triggers": [[i, run.tok.decode([i])] for i in sorted(triggers)], "S": s, "random": r,
                           "random_target_share_mean": r_mean, "check": s["target_share"] >= CFG.census_ratio * r_mean and s["target_share"] > 0,
                           "pieces_S": rows_S}
        big[n] = [[piece_row(run, table[p], int(t.target), triggers) for p in R.members(x)] for x in rand[n]]
        log(f"[stage3 {n}] target {out['tasks'][n]['target']} triggers {out['tasks'][n]['triggers']} | S {json.dumps(s['counts'])} "
            f"target share {s['target_share']:.3f} | random target share mean {r_mean:.3f} "
            f"{[round(x['target_share'], 3) for x in r]} | check (>= {CFG.census_ratio}x) {out['tasks'][n]['check']}")
        for x in rows_S:
            log(f"[stage3 {n}] {x['matrix']}:{x['index']} rise {x['rise']:.3f} on {x['on_rate']:.4f} ({x['n_on']}) {x['class']} | "
                f"next {[(y[1], y[2]) for y in x['top_next'][:5]]} | cur {[(y[1], y[2]) for y in x['top_cur'][:5]]}")
        dump(3, out)
    (LARGE / "stage3_pieces.json").write_text(json.dumps(_finite(big)))
    return out


# ----------------------------------------------------------------------------- main


def stages_arg(s: str) -> list[int]:
    ks = sorted({int(x) for x in s.split(",")})
    if not set(ks) <= {0, 1, 2, 3}:
        raise argparse.ArgumentTypeError("stages are 0, 1, 2, 3")
    return ks


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=stages_arg, required=True, help="comma list from 0,1,2,3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    stages = args.stages
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    s0, s1 = read_json(RESULTS / "stage0.json"), read_json(RESULTS / "stage1.json")
    if 0 not in stages and not (s0 and s0.get("pass") is True):
        sys.exit("stage 0 has not passed (results/stage0.json): run --stages 0 alone first")
    if {2, 3} & set(stages) and 1 not in stages and not (s1 and s1.get("reading") not in (None, "fail")):
        sys.exit("stages 2 and 3 run only after stage 1 did not read 'fail' (results/stage1.json)")
    STATES.resolve().mkdir(parents=True, exist_ok=True)  # through the results/large symlink, if there is one
    t0 = time.time()
    run = Run(args.device)
    log(f"[start] stages {stages} on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}")
    if 0 in stages:
        s0 = stage0(run)
        if not s0["pass"]:
            sys.exit(1)
    else:
        assert s0 is not None
        run.load_states(s0)
    if 1 in stages:
        s1 = stage1(run)
        if s1["reading"] == "fail":
            log(f"[stage1] reading 'fail': stages 2 and 3 do not run (plan's decision tree) ({time.time() - t0:.0f} s)")
            return
    if 2 in stages:
        assert s1 is not None
        stage2(run, s1)
    if 3 in stages:
        stage3(run)
    log(f"[done] stages {stages} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
