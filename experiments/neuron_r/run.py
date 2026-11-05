"""neuron_r (E98): a neuron analogue of VPD's r on the 67M's MLP neurons, no decomposition, read on the method table's
rows where VPD's r has readings and scored against the construction labels (reuse vs new, by assumption). Plan:
reports/plans/2026-11-05_plan_e98-neuron-r.md. Cells and rules: `commute.neuron_r` (its docstring defines r_half, r_n,
d_n, i_n, e_n). Every setting, row and threshold: results/predictions.json (filed before the run, never edited after);
stage 0's control rules: results/predictions_amend1.json (Amendment 1, 2026-11-05).
$0, saved states only, plus two built controls.

  sha256 of every state and windows file, before the model loads                       -> results/stage0.json
  stage 0, alone (exit 1 on a miss): W0 on every task (all cells 0); E56's full-width share reproduced from r_n on
          elicit_s0 and cipher_s0; code_step10 as do-nothing (|G| under the gain floor); three built controls (a fine-tune
          of c_fc and attention only: d_n = i_n = 0; of the last down_proj only: d_n = 1, the rest 0; of every down_proj:
          e_n = 0, r_half = r_n, d_n > 0, r_n recorded) -> stage0.json
  stage 1: the cells of every state of the rows, written as each finishes                -> results/cells.json
  stage 2: per scored state, the attribution order and the grid of top-k patches, k_half  -> results/carrying.json
  score: P1 (r_half), P1r (r_n), P2 (e_n), P3 (k_half), and every row's verdicts unfiltered -> results/score.json
Every number is printed. Target model only, fp32, TF32 off.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from commute import anchors as AN
from commute import finetune as FT
from commute import keydep as KD
from commute import neuron_r as NR
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment

HERE = Path(__file__).resolve().parent
RESULTS, REPO, EXP = HERE / "results", HERE.parents[1], HERE.parent
PRED, AMEND1 = RESULTS / "predictions.json", RESULTS / "predictions_amend1.json"  # AMEND1: stage 0's control rules
E56, E62 = EXP / "keydependent_patch" / "results", EXP / "induction_cue" / "results"
E63 = load_experiment("neuron_sparsity", "run")  # E63's files, state table and task loaders (E46, E54, E56, E53, code)
OPI = load_experiment("open_pair_instruments", "common")  # E55 C3's paths and E58's records
C3_MAPS, E62_MAPS = ("rank1", "cipher", "copy"), ("copy", "lookup")
E53_TASKS = ("lowercase", "broad_cipher", "e53_cipher")  # next-token targets: no learned level, so no P(target) filter
DEVICE, FULL = "cuda", NR.N_NEURONS
log = KD.log
Files = dict[str, tuple[Path, str]]


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


class Out:
    """One results file: each record written and logged as it lands; a miss (ok False) writes, logs FAIL and exits 1."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").stdout.strip(),
                                    "git_dirty": git("status", "--porcelain", "--untracked-files=no").stdout, "rows": {}}

    def dump(self) -> None:
        (RESULTS / self.name).write_text(json.dumps(E63.clean(self.out), indent=1, sort_keys=True))

    def put(self, key: str, rec: dict[str, Any], ok: bool | None = None) -> dict[str, Any]:
        self.out["rows"][key] = {**rec, **({} if ok is None else {"ok": bool(ok)})}
        self.dump()
        log(f"[{self.name} {key}] ok {ok} {json.dumps(E63.clean(rec), sort_keys=True)}")
        if ok is False:
            self.out["stopped"] = key
            self.dump()
            log(f"FAIL: {key}: {rec.get('failed', '')} stop and tell the lead")
            sys.exit(1)
        return rec


def load(path: Path) -> Any:
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------- files, states, windows


def files() -> Files:
    """E63's files (E46, E54, E56 seeds 2-4, E53 step 300, code_step10, their windows) plus C3's 24 states and three
    windows files and E62's six states and windows, each with the sha256 its source run recorded."""
    F = E63.files()
    rec, wsha, seeds = OPI.e58_records(), OPI.e58_windows_sha(), load(OPI.RESULTS / "seeds.json")["states"]
    for m in C3_MAPS:
        F[f"c3/windows_{m}"] = (OPI.windows_path(m, "trained"), wsha[m]["trained"])
        for s in range(8):
            F[f"c3/{m}_s{s}"] = (OPI.state_path(m, s), (rec if s in OPI.OLD_SEEDS else seeds)[f"{m}_s{s}"]["ft"]["sha256"])
    run = load(E62 / "run.json")
    F["e62/windows"] = (E62 / "large" / "windows.pt", run["windows_sha256"])
    F |= {f"e62/{m}_s{s}": (E62 / "large" / f"{m}_s{s}.pt", run["states"][f"{m}_s{s}"]["sha256"]) for m in E62_MAPS for s in range(3)}
    return F


def state_table() -> dict[str, tuple[str, str]]:
    """Each state's (key in `files`, task). E63's names for E46, E54, E56 and E53; c3_<map>_s<seed>, e62_<map>_s<seed>."""
    out = E63.state_table()
    out |= {f"c3_{m}_s{s}": (f"c3/{m}_s{s}", f"c3_{m}") for m in C3_MAPS for s in range(8)}
    return out | {f"e62_{m}_s{s}": (f"e62/{m}_s{s}", f"e62_{m}") for m in E62_MAPS for s in range(3)}


def state(F: Files, key: str) -> ST.DW:
    return {k: v.float() for k, v in torch.load(F[key][0], weights_only=True).items()}


def load_reads(net: ST.Net, F: Files) -> dict[str, NR.Reads]:
    """Every task's held-out read windows, the ones VPD's r read: E63's KV and E53 tasks, C3's trained-key held-out
    windows (`windows_<map>_trained.pt`, as C3's instruments), E62's read windows with each map's target."""
    tasks, _ = E63.load_tasks(net, E63.TT.performed_anchor(), F)
    out = {n: NR.task_reads(t) for n, t in tasks.items()}
    for m in C3_MAPS:
        out[f"c3_{m}"] = NR.task_reads(ST.load_task(net, F[f"c3/windows_{m}"][0], f"c3_{m}", m))
    w = torch.load(F["e62/windows"][0], weights_only=False)
    assert w["read"].tokens.shape[1] == 2 * AN.HALF + 1 and w["read"].half == AN.HALF
    return out | {f"e62_{m}": NR.Reads(f"e62_{m}", w["read"].tokens, w["target"][m]["read"], AN.HALF) for m in E62_MAPS}


def p_learn_of(task: str, S: dict[str, Any]) -> float | None:
    return None if task in E53_TASKS else S["p_learn"]


# ----------------------------------------------------------------------------- stage 0


def e56_share(c: dict[str, torch.Tensor]) -> float:
    """r_n's patch in E56's unit: log P, scoring half (odd windows), share of that half's gain."""
    s = slice(1, None, 2)
    return float((c["lp_L01"] - c["lp_T0"])[s].mean()) / float((c["lp_T1"] - c["lp_T0"])[s].mean())


def control(net: ST.Net, w: dict[str, Any], name: str, seed: int, A: dict[str, Any]) -> tuple[ST.DW, dict[str, Any]]:
    """E54's elicit fine-tune (`keydep.finetune`: arm A, Adam at anchors.FT_LR, batch 32, 300 steps, read at E12's scored
    step) with one change, the matrices trained: every down_proj ('down_only'), the last block's alone ('down_last',
    amend1 names it) or the other 16 matrices ('cfc_attn')."""
    down = tuple(k for k in net.W0 if k.endswith("mlp.down_proj"))
    last = tuple(A["controls"]["down_last"]["matrices"])
    assert set(last) <= set(down)
    mats = {"down_only": down, "down_last": last}.get(name) or tuple(k for k in net.W0 if k not in down)
    cfg = FT.ArmAConfig(o_token=w["table"], lr=AN.FT_LR, steps=KD.E12_CHECKS[-1], checkpoints=KD.E12_CHECKS, seed=seed,
                        ci_len=w.get("ci_len"), matrices=mats)
    ft = FT.finetune_arm_a(net.model, w["train"], w["heldout"], cfg, net.device, log)
    step, learned = KD.scored_step(ft.evals)
    d = ft.deltas[step]
    assert set(d) == set(mats)
    dW = {k: d[k] if k in d else torch.zeros_like(v) for k, v in net.W0.items()}
    return dW, {"n_matrices": len(mats), "step": step, "learned": learned, "p_heldout": {k: e["p_o"] for k, e in ft.evals.items()}}


def control_failed(name: str, rec: dict[str, Any], band: float) -> list[str]:
    """The conditions of predictions_amend1.json that this control's record misses (empty: it passes)."""
    n = rec["num"]
    if not rec["readable"]:
        return [f"unreadable: |G| {abs(rec['G']):.3g} < gain_floor"]
    conds: dict[str, bool]
    if name == "cfc_attn":
        conds = {"|d_n| <= band": abs(n["d_n"]) <= band, "|i_n| <= band": abs(n["i_n"]) <= band}
    elif name == "down_only":
        conds = {"|e_n| <= band": abs(n["e_n"]) <= band, "|r_half - r_n| <= band": abs(n["r_half"] - n["r_n"]) <= band,
                 "d_n > 0": rec["d_n"] > 0}  # r_n is recorded, not stopped on
    else:
        conds = {f"|{k}| <= band": abs(n[k]) <= band for k in ("r_n", "r_half", "i_n", "e_n")}
        conds["|d_n - 1| <= band / G"] = abs(rec["d_n"] - 1) <= band / abs(rec["G"])
    return [f"{c} (band {band:.3g}, num {json.dumps(n)}, d_n {rec['d_n']:.4g})" for c, ok in conds.items() if not ok]


def stage0(net: ST.Net, reads: dict[str, NR.Reads], F: Files, P: dict[str, Any], A: dict[str, Any], o: Out) -> None:
    c, S = P["stage0"], P["settings"]
    B, floor = S["batch"], S["gain_floor"]
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    worst = 0.0
    for n, r in reads.items():
        dev = NR.max_deviation(NR.state_cells(net, r, zero, B))
        worst = max(worst, dev)
        o.put(f"w0/{n}", {"max_abs": dev, "n_windows": len(r.toks)}, dev <= c["identity_tol"])
    band = max(c["identity_tol"], c["band_multiple"] * worst)
    o.put("band", {"w0_max_abs": worst, "band": band})
    job1 = load(E56 / "job1.json")["states"]
    for n in ("elicit_s0", "cipher_s0"):
        sh = e56_share(NR.state_cells(net, reads[n.split("_")[0]], state(F, f"e54/{n}"), B))
        ref = job1[n]["b2_shares"][str(FULL)]
        o.put(f"e56/{n}", {"r_n_logp_score_half": sh, "e56_full_share": ref, "diff": sh - ref}, abs(sh - ref) <= c["e56_tol"])
    code = state(F, "e36/code_step10")
    for n in c["do_nothing_tasks"]:
        rec = NR.shares(NR.state_cells(net, reads[n], code, B), floor)
        o.put(f"do_nothing/{n}", rec, not rec["readable"])
    w = torch.load(F["e54/windows_elicit"][0], weights_only=False)
    for name in ("cfc_attn", "down_last", "down_only"):
        dW, meta = control(net, w, name, c["control_seed"], A)
        rec = {**meta, **NR.shares(NR.state_cells(net, reads["elicit"], dW, B), floor), "band": band}
        rec["failed"] = control_failed(name, rec, band)
        o.put(f"control/{name}", rec, not rec["failed"])
    o.out["pass"] = True
    o.dump()
    log("[stage0] pass")


# ----------------------------------------------------------------------------- stages 1, 2 and the score


def stage1(net: ST.Net, reads: dict[str, NR.Reads], F: Files, P: dict[str, Any]) -> dict[str, dict[str, Any]]:
    S, o = P["settings"], Out("cells.json")
    for n, (key, task) in state_table().items():
        rec = NR.shares(NR.state_cells(net, reads[task], state(F, key), S["batch"]), S["gain_floor"])
        rec |= {"task": task, "scored": NR.scored(rec, p_learn_of(task, S))}
        o.put(n, rec, rec["identity"] <= P["stage0"]["identity_tol"] and abs(rec["telescope"]) <= S["telescope_tol"])
    return o.out["rows"]


def stage2(net: ST.Net, reads: dict[str, NR.Reads], F: Files, P: dict[str, Any], cells: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    S, o, ks = P["settings"], Out("carrying.json"), {}
    for n, (key, task) in state_table().items():
        if not cells[n]["scored"]:
            continue
        r, dW, full = reads[task], state(F, key), cells[n]["num"]["r_half"]
        score = NR.ranking(net, r, dW, S["batch"])
        order = torch.sort(score, descending=True, stable=True).indices
        nums = NR.grid_numerators(net, r, dW, order, S["k_grid"], S["batch"])
        ks[n] = NR.k_half(nums, full, cells[n]["r_half"], S["carry_floor"])
        o.put(n, {"numerators": nums, "r_half_num": full, "k_half": ks[n], "end_diff": nums[FULL] - full,
                  "top": order[: S["n_top_saved"]].tolist(), "top_score": score[order[: S["n_top_saved"]]].tolist()},
              abs(nums[FULL] - full) <= S["grid_end_tol"])
    return ks


def score(cells: dict[str, dict[str, Any]], ks: dict[str, int | None], P: dict[str, Any]) -> dict[str, Any]:
    S, R, pr = P["settings"], P["rows"], P["predictions"]
    verd = lambda key, names, filt=True: {n: NR.row_verdict(cells, R[n], key, S, filt) for n in names}
    out: dict[str, Any] = {
        "P1": NR.majority_fails(verd("r_half", pr["P1"]["rows"]), S["min_rows"]),
        "P1r": NR.majority_fails(verd("r_n", pr["P1r"]["rows"]), S["min_rows"]),
        "P2": NR.outside_share(cells, {n: R[n] for n in pr["P2"]["rows"]}, S),
        "P3": NR.carrying_rule(ks, pr["P3"]["low_states"], pr["P3"]["low_max"], pr["P3"]["other_min_exclusive"]),
        "reported": {f: {k: verd(k, list(R), f == "filtered") for k in NR.SHARES} for f in ("filtered", "unfiltered")},
    }
    for p in ("P1", "P1r", "P2", "P3"):
        log(f"[score {p}] {out[p]['status']} {json.dumps(E63.clean(out[p]), sort_keys=True)}")
    for f, by_key in out["reported"].items():
        for k, rows in by_key.items():
            log(f"[reported {f} {k}] " + " ".join(f"{n} {v['verdict']}" for n, v in rows.items()))
    return out


def main() -> None:
    for f in (PRED, AMEND1):
        if not f.is_file():
            sys.exit(f"file the predictions first: {f}")
        rel = str(f.relative_to(REPO))
        if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
            sys.exit(f"{rel} is not committed unchanged: commit it first")
    P, A = load(PRED), load(AMEND1)
    S = P["settings"]
    assert S["k_grid"] == [2**i for i in range(14)] + [FULL]
    assert all(set(r["reuse"] + r["new"]) <= set(state_table()) for r in P["rows"].values())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    F, o = files(), Out("stage0.json")
    got = {k: KD.sha256(p) if p.is_file() else "missing" for k, (p, _) in F.items()}
    o.put("sha256", {"n_files": len(F), "mismatch": sorted(k for k, (_, h) in F.items() if got[k] != h)},
          all(got[k] == h for k, (_, h) in F.items()))
    net = ST.Net(DEVICE)
    reads = load_reads(net, F)
    stage0(net, reads, F, P, A, o)
    cells = stage1(net, reads, F, P)
    ks = stage2(net, reads, F, P, cells)
    res = Out("score.json")
    res.out["rows"] = score(cells, ks, P)
    res.out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res.dump()
    log(f"[done] peak GPU {res.out['peak_gpu_gb']:.1f} GB")
    hard_exit(0)


if __name__ == "__main__":
    main()
