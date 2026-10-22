"""regating_down (E29): what do the subcomponents the fine-tune turns down at the trigger do, read at W0?
Plan: reports/2026-10-22_plan_regating-down.md. Definitions: commute.turned_down's docstring; E18's in
reports/2026-10-21_plan_regating-leak.md (Amendment 3: the pool, the strong turned-down set, the within-pool shuffle).

Per task-seed, from a saved fixed change s (E18's three tasks x seeds 0, 1; E28's four ladder tasks where saved), at W0:
  overlap   J(S_down, N_comp) against same-size draws from the pool; N_comp = top-|S_down| pool members by W0's Pile
            contrast for "the next token is one of the task's five competitors"
  function  competitor mass at the trigger of the CI-masked model under g0, clip(g0 + s-), and s- shuffled within the pool
Control first and alone: J_nat of S_on reproduces E18's on the three seed-0 states (and pattern (ii) E18 amendment 3's
'down' recovery there); a J_nat failure stops the job.

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/regating_down.json (after every task-seed), results/large/full.log
Inputs (gitignored): experiments/regating_leak/results/large/<task>_s{0,1}_shift.pt,
experiments/regating_tasks/results/large/<task>_shift.pt (optional), experiments/transport_test/results/{windows.json,
large/alive.pt, large/windows.pt}, experiments/task_ladder/results/large/windows_<task>.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import recruit as R
from commute import regating as G
from commute import turned_down as D

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "regating_down.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E18 = EXP / "regating_leak" / "results"
E28 = EXP / "regating_tasks" / "results" / "large"

E18_TASKS, E18_SEEDS = ("emoticon", "r1_t534_v369", "r2_t359_v816"), (0, 1)
E28_TASKS = ("r0_t352_v310", "r1_t588_v452", "r2_t275_v2667", "r0_t588_v320")
NAT_ROWS, MIN_NAT = ("val", 0, 600), 200
J_TOL, RHO_TOL, SHIFT_TOL = 0.02, 0.02, 1e-4
KAPPA_COMP = 0.15  # E18's switched-on overlap with the target's machinery read kappa 0.15 to 0.28
SPECIFIC, GENERIC = 0.2, 0.05

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-22_plan_regating-down.md. Filed before any run; never edited after.",
    "inputs": "E18's fixed changes s (emoticon, r1_t534_v369, r2_t359_v816; seeds 0 and 1) and E28's (r0_t352_v310, "
              "r1_t588_v452, r2_t275_v2667, r0_t588_v320; one seed; skipped if not saved). Everything read at W0 with the "
              "released subcomponents, Gamma frozen, Delta out of every masked read; no fine-tune",
    "definitions": {
        "pool P": f"alive with m0 >= {D.POOL_MIN} (m0: W0's mean trigger importance over the training windows)",
        "S_down": f"s <= -{D.DOWN_THR} within P", "S_on": f"s > {G.THR}",
        "competitors": f"top {D.K} tokens by W0's mean next-token probability at the trigger of the held-out windows "
                       "(target model, no masks), the target excluded",
        "contrast(c)": f"mean g at Pile {NAT_ROWS} positions whose next token is in c minus mean g over all positions",
        "N_target": "top-|S_on| alive by contrast(target) (E18's J_nat set)",
        "N_comp": "top-|S_down| within P by contrast(the five competitors' union)",
        "pool null": f"{G.N_RANDOM} draws of S_down's size per matrix, uniform over P (S_down's members allowed)",
        "kappa": "(J - null mean) / (1 - null mean)",
        "patterns": "at the trigger only, g0 elsewhere: (i) g0; (ii) clip(g0 + s-); (iii) s-'s in-pool values permuted "
                    f"within each matrix's pool, out-of-pool entries kept, clipped, {G.N_RANDOM} draws",
        "drop_comp": "(competitor mass under (i) - under the pattern) / competitor mass under (i), mean over windows "
                     "of the CI-masked model's trigger probabilities",
    },
    "control": f"J_nat = J(S_on, N_target) within {J_TOL} of E18 (regating_leak.json natural.<task>.s0.J: 0.2844, "
               f"0.1546, 0.2432) on the three seed-0 states; runs first, alone; fail: stop. With it, pattern (ii)'s "
               f"recovery (lo(ii) - lo(i)) / gap within {RHO_TOL} of E18 amendment 3 (regating_leak_amend3.json "
               "tasks.<task>.seeds.0.rho.half_down: 0.238, 0.162, 0.387): fail marks that state's function read unscored",
    "scorable": f"overlap: >= {MIN_NAT} Pile positions for the competitors' union and |S_down| >= 1; function: pattern "
                "(ii) raises the target's mean log-prob over (i) (amendment 3: the turned-down half recovers 0.15 to 0.39); "
                f"both: min over alive of s + m0 >= -{SHIFT_TOL} (the saved change matches the in-job m0)",
    "expected": {
        "W0 / pattern (i)": "drop_comp 0 by construction; J undefined",
        "pool null": "J about |S_down| / (2|P| - |S_down|): ~0.46 emoticon (243 of 385), ~0.39 r1_t534 (134 of 239), "
                     "~0.18 r2_t359 (73 of 238); kappa ~0, sd ~0.03",
        "pattern (iii)": "the target falls (amendment 3: shuffled down recovers -0.06 to -0.41 of its unshuffled), so "
                         "mass probably moves toward the competitors: drop_comp(iii) <= 0",
        "J(S_down, N_target), J(S_on, N_comp)": "low (the halves aim at different tokens), not scored",
    },
    "predictions": [
        {"id": "P1_overlap", "rule": f"per scorable task-seed: J_down = J(S_down, N_comp) > null max and kappa >= "
                                     f"{KAPPA_COMP} reads 'competitors' machinery'; J_down <= null max reads 'no better "
                                     "than the pool'; 'between' otherwise. Absolute, in Jaccard; kappa relative to the null",
         "predicted": "'competitors' machinery' on most task-seeds, low-medium confidence (the half is specific and a "
                      "real switch-off, but it may be W0's generic next-is-a-function-word machinery, not the competitors')",
         "blind_to": "the trigger position only; held-out windows; the next-token definition of machinery (E6 flagged it "
                     "as weak); five competitors by W0's mean probability (a competitor the fine-tune never fought counts); "
                     "the top-k cut; one model",
         "positive_control": "the J_nat reproduction (the same contrast and top-k on the switched-on half)",
         "noise_floor": f"{G.N_RANDOM} pool draws, sd reported; E18 seeds 0 and 1 agree within 0.03 on J_nat"},
        {"id": "P2_function", "rule": f"per scorable task-seed: d = drop_comp(ii) - mean drop_comp(iii); d >= {SPECIFIC} "
                                      f"with drop_comp(ii) > 0 reads 'switches off competitors specifically'; |d| <= "
                                      f"{GENERIC} reads 'generic lowering'; 'between' otherwise. Absolute, in shares of the "
                                      "competitors' mass under g0",
         "predicted": "'switches off competitors specifically', low-medium confidence",
         "blind_to": "the trigger only; held-out windows; five competitors; the clip; mass toward 'rest' is not split "
                     "further; one model",
         "positive_control": "pattern (ii) raises the target's log-prob (checked in-job; amendment 3's down recovery "
                             "reproduced on the seed-0 E18 states)",
         "noise_floor": f"{G.N_RANDOM} shuffles, sd reported; E18 seeds 0 and 1 agree within 0.02 on recoveries"},
    ],
    "reported_not_scored": "competitor strings and mean probabilities; |P|, |S_down|, |S_on| and S_down by layer; "
                           "J(S_down, N_target); J(S_on, N_comp); J_nat on every task-seed; under (ii) where the removed "
                           "competitor mass goes (target vs rest); the ratio J_down / null mean",
    "decision_tree": "both 'competitors' on most task-seeds: H' gains a mechanism sentence (switch on the target's "
                     "machinery, switch off its competitors'), next the same read on the larger model; both "
                     "'generic/pool': the turned-down half is a generic lowering and the switched-on half carries the "
                     "specific part; mixed: report, no action. Only a control failure stops the line",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def shift_files() -> list[tuple[str, str, Path]]:
    """(task, label, path) of every saved change found; a missing E28 file is logged and skipped."""
    out = [(n, f"s{s}", E18 / "large" / f"{n}_s{s}_shift.pt") for n in E18_TASKS for s in E18_SEEDS]
    for n in E28_TASKS:
        found = sorted(E28.glob(f"{n}*_shift.pt"))  # <task>_shift.pt as agreed; a seed or label suffix is kept as the label
        out += [(n, "e28" + p.name[len(n) : -len("_shift.pt")], p) for p in found]
        if not found:
            log(f"[inputs] no {E28}/{n}*_shift.pt: {n} skipped")
    return out


def references() -> dict[str, Any]:
    e18, a3 = json.loads((E18 / "regating_leak.json").read_text()), json.loads((E18 / "regating_leak_amend3.json").read_text())
    return {n: {"J_nat": e18["natural"][n]["s0"]["J"], "rho_down": a3["tasks"][n]["seeds"]["0"]["rho"]["half_down"],
                "gap": a3["tasks"][n]["seeds"]["0"]["gap"]} for n in E18_TASKS}


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.model = vpd.load_component_model(device)
        self.tok = vpd.tokenizer()
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}
        self.tasks: dict[str, G.Task] = {}
        self.comp: dict[str, Tensor] = {}
        self.target: dict[str, int] = {}
        self.con: dict[str, Any] = {}

    def prepare(self, name: str) -> dict[str, Any]:
        """W0's readings on the task's windows and its five competitors."""
        train, held, target = G.load_windows(name, TRANSPORT, LADDER)
        assert isinstance(target, int), name
        self.target[name] = target
        self.tasks[name] = t = G.prepare(self.model, name, train, held, target)
        probs = D.mean_next_probs(self.model, held.tokens, held.half, G.BATCH)
        ids, p = D.top_competitors(probs, target)
        self.comp[name] = ids
        rec = {"target": target, "target_str": self.tok.decode([target]), "p_target": float(probs[target]),
               "competitors": [{"id": int(i), "str": self.tok.decode([int(i)]), "p": float(q)} for i, q in zip(ids, p, strict=True)],
               "n_held": len(held), "L00": t.base["L"]}
        log(f"[{name}] target {rec['target_str']!r} p {rec['p_target']:.4f} | competitors "
            + ", ".join(f"{c['str']!r} {c['p']:.4f}" for c in rec["competitors"]))
        return rec

    def pile(self, names: list[str]) -> dict[str, Any]:
        from commute.data import pile_rows

        rows = pile_rows(*NAT_ROWS, log=log)
        masks = {**{f"{n}/target": R.precede_mask(rows, self.target[n]) for n in names},
                 **{f"{n}/comp": D.precede_any(rows, self.comp[n]) for n in names}}
        self.con = D.contrasts(self.model, rows, masks, self.alive)
        npos = {k: v["n_positions"] for k, v in self.con.items()}
        log(f"[pile] positions {json.dumps(npos)}")
        return npos


def control(run: Run, refs: dict[str, Any]) -> dict[str, Any]:
    """J_nat of each E18 seed-0 S_on against E18's value."""
    out: dict[str, Any] = {}
    for n in E18_TASKS:
        on = R.switched(D.load_shift(E18 / "large" / f"{n}_s0_shift.pt"), G.THR)
        j = G.j_nat(on, run.con[f"{n}/target"]["contrast"], run.alive)["J"]
        out[n] = {"J_nat": j, "ref": refs[n]["J_nat"], "diff": j - refs[n]["J_nat"], "ok": abs(j - refs[n]["J_nat"]) <= J_TOL}
        log(f"[control {n}] J_nat {j:.4f} E18 {refs[n]['J_nat']:.4f} -> {'ok' if out[n]['ok'] else 'FAIL'}")
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def sets_of(run: Run, name: str, shift: dict[str, Tensor]) -> dict[str, Any]:
    """The overlap numbers of one task-seed."""
    from commute.signatures import top_by

    t = run.tasks[name]
    pool = D.pool_of(t.m0, run.alive)
    down, on = D.down_set(shift, pool), R.switched(shift, G.THR)
    ct, cc = run.con[f"{name}/target"], run.con[f"{name}/comp"]
    n_target, n_comp = top_by(ct["contrast"], run.alive, R.size(on)), top_by(cc["contrast"], pool, R.size(down))
    ov = D.overlap(down, n_comp, pool, G.N_RANDOM)
    return {"n_pool": R.size(D.as_sel(pool)), "n_down": R.size(down), "n_on": R.size(on),
            "by_layer_down": R.by_layer(down), "n_positions_comp": cc["n_positions"], "overlap": ov,
            "ratio_to_null": G.div(ov["J"], ov["random_mean"]), "J_down_target": R.jaccard(down, n_target),
            "J_on_comp": R.jaccard(on, n_comp), "J_nat": G.j_nat(on, ct["contrast"], run.alive),
            "scorable": cc["n_positions"] >= MIN_NAT and R.size(down) >= 1, "pool": pool}


def function_of(run: Run, name: str, shift: dict[str, Tensor], pool: dict[str, Tensor]) -> dict[str, Any]:
    """Competitor, target and rest mass at the trigger under patterns (i), (ii), (iii)."""
    t = run.tasks[name]
    g0, h, comp = t.base["gates"], t.half, run.comp[name]

    def read(gates: R.Gates) -> dict[str, float]:
        return D.masses(run.model, t.held.tokens, h, run.target[name], comp, gates, G.BATCH)

    ref, dn = read(g0), read(D.down_pattern(g0, h, shift))
    sh = [read(D.shuffled_pattern(g0, h, shift, pool, i)) for i in range(G.N_RANDOM)]
    drops = [D.drop(ref, m) for m in sh]
    mean = sum(drops) / len(drops)
    sd = (sum((x - mean) ** 2 for x in drops) / (len(drops) - 1)) ** 0.5
    moved = {k: dn[k] - ref[k] for k in ("p_target", "p_comp", "p_rest")}
    return {"i": ref, "ii": dn, "iii": sh, "drop_ii": D.drop(ref, dn), "drop_iii": drops, "drop_iii_mean": mean,
            "drop_iii_sd": sd, "d": D.drop(ref, dn) - mean, "moved_ii": moved, "lo_ii_minus_i": dn["lo_target"] - ref["lo_target"],
            "raises_target": dn["logp_target"] > ref["logp_target"], "L00_check": ref["lo_target"] - t.base["L"]}


def one(run: Run, name: str, label: str, path: Path, refs: dict[str, Any]) -> dict[str, Any]:
    shift = D.load_shift(path)
    t = run.tasks[name]
    slack = min(float((shift[k] + t.m0[k])[run.alive[k]].min()) for k in shift)
    sets = sets_of(run, name, shift)
    pool = sets.pop("pool")
    fn = function_of(run, name, shift, pool)
    rec: dict[str, Any] = {"task": name, "label": label, "file": str(path.relative_to(EXP.parent)), "min_s_plus_m0": slack,
                           "consistent": slack >= -SHIFT_TOL, "sets": sets, "function": fn}
    if label == "s0" and name in refs:
        rho = G.div(fn["lo_ii_minus_i"], refs[name]["gap"])
        rec["rho_down"] = {"rho": rho, "ref": refs[name]["rho_down"], "ok": abs(rho - refs[name]["rho_down"]) <= RHO_TOL}
    rec["scorable_overlap"] = sets["scorable"] and rec["consistent"]
    rec["scorable_function"] = fn["raises_target"] and rec["consistent"] and rec.get("rho_down", {}).get("ok", True)
    ov = sets["overlap"]
    log(f"[{name} {label}] s+m0 min {slack:.2e} | P {sets['n_pool']} S_down {sets['n_down']} S_on {sets['n_on']} by layer "
        f"{sets['by_layer_down']} | comp positions {sets['n_positions_comp']} | J_down {ov['J']:.4f} null mean "
        f"{ov['random_mean']:.4f} sd {ov['random_sd']:.4f} max {ov['random_max']:.4f} kappa {ov['kappa']:.4f} | J(S_down, "
        f"N_target) {sets['J_down_target']:.4f} J(S_on, N_comp) {sets['J_on_comp']:.4f} J_nat {sets['J_nat']['J']:.4f}")
    log(f"[{name} {label}] (i) {json.dumps(fn['i'])} | (ii) {json.dumps(fn['ii'])} | drop_ii {fn['drop_ii']:.4f} drop_iii "
        f"mean {fn['drop_iii_mean']:.4f} sd {fn['drop_iii_sd']:.4f} d {fn['d']:.4f} | moved {json.dumps(fn['moved_ii'])} | "
        f"raises target {fn['raises_target']} | lo(i) - L00 {fn['L00_check']:.2e} | rho_down {json.dumps(rec.get('rho_down'))}")
    log(f"[{name} {label}] drop_iii draws {json.dumps([round(x, 4) for x in fn['drop_iii']])}")
    rec["reading"] = {"P1_overlap": reading_overlap(rec), "P2_function": reading_function(rec)}
    log(f"[{name} {label}] readings {json.dumps(rec['reading'])}")
    return rec


def reading_overlap(rec: dict[str, Any]) -> str:
    if not rec["scorable_overlap"]:
        return "unscorable"
    ov = rec["sets"]["overlap"]
    if ov["J"] <= ov["random_max"]:
        return "no better than the pool"
    return "competitors' machinery" if ov["kappa"] >= KAPPA_COMP else "between"


def reading_function(rec: dict[str, Any]) -> str:
    if not rec["scorable_function"]:
        return "unscorable"
    fn = rec["function"]
    if fn["d"] >= SPECIFIC and fn["drop_ii"] > 0:
        return "switches off competitors specifically"
    return "generic lowering" if abs(fn["d"]) <= GENERIC else "between"


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    t0 = time.time()
    refs = references()
    files = shift_files()
    names = list(dict.fromkeys(n for n, _, _ in files))
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "refs": refs, "tasks": {}, "states": {}}
    log(f"[start] {out['device']}; refs {json.dumps(refs)}; task-seeds {[(n, lab) for n, lab, _ in files]}")
    for n in names:
        out["tasks"][n] = r.prepare(n)
    dump(out)
    out["pile_positions"] = r.pile(names)
    out["control"] = c = control(r, refs)
    dump(out)
    if not c["ok"]:
        out["status"] = "stopped at the control"
        dump(out)
        log(f"CONTROL FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for n, lab, p in files:
        out["states"][f"{n}/{lab}"] = one(r, n, lab, p, refs)
        dump(out)
    out["seconds"], out["status"] = time.time() - t0, "done"
    out["readings"] = {k: v["reading"] for k, v in out["states"].items()}
    dump(out)
    log("readings " + json.dumps(out["readings"]))
    log(f"wrote {OUT} ({out['seconds']:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        RESULTS.mkdir(parents=True, exist_ok=True)
        PRED.write_text(json.dumps({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        print(f"filed {PRED}")
        return
    run(args.device)


if __name__ == "__main__":
    main()
