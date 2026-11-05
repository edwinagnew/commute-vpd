"""E102 (qwen_carrying), stage A, 2026-11-05: E98's carrying sets on Qwen3-1.7B's SwiGLU neurons, and two agent-free
readings of them, the engaged share and the prompt overlap. No agents, no decomposition. Plan:
reports/plans/2026-11-05_plan_e102-qwen-carrying.md. Rules: results/predictions.json (filed before the run). Cells and
rules: `commute.qwen_neurons`.

  run.py --states gauss,a1_s0,a1_s1      job 1: the W0 stage first (results/w0.json missing), then E94's Gaussian
                                         change, control (i) (results/control.json), then the states named
  run.py --states a2_s42_matched,a2_s142_matched,a2_s42_full,a2_s142_full   job 2: refuses until control.json passed
  run.py --score                         CPU: results/score.json from what is written (each job also runs it last)

W0 stage, cheapest first: E0 the do-nothing change through the cell path; E1 W0's log P on both row sets against E92's;
the prompted contrast and C1a; then the FineWeb-Edu pass (each neuron's top 20 windows, large/tops.safetensors), the
corpus check C0 and the engaged values (large/engaged.safetensors). A state's numbers go to results/states/ and the log
as it ends, its scores to large/states/; a state already written is skipped, so a lost job resumes.
Exit 0 done; 1 a stop (results/stopped.json; every run refuses while it exists) or a refusal; 3 the time budget: resubmit.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from commute import qwen_comparators as qc
from commute import qwen_neurons as QN
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment

SC: Any = load_experiment("qwen_carrying", "score")  # the files, the state names, the scorer
RESULTS, LARGE, STATES, P, put = SC.RESULTS, SC.LARGE, SC.STATES, SC.P, SC.put
PRED, STOPPED, W0F, CONTROL, GAUSS = SC.PRED, SC.STOPPED, SC.W0F, SC.CONTROL, SC.GAUSS
NAMED, GAUSS_READS = SC.NAMED, SC.GAUSS_READS
E92 = EXPERIMENTS / "qwen_comparators" / "results"
FORMS = EXPERIMENTS / "qwen_tofu_forms" / "results" / "forms.jsonl"


@dataclass(frozen=True)
class Config:
    """Every size of the run, from predictions.json settings.sizes."""

    K: int
    ks: tuple[int, ...]
    batch: int
    corpus_batch: int
    contrast_batch: int
    window_rows: int
    quantile_rows: int
    check_rows: int
    n_top: int
    n_draws: int
    seed: int
    budget_s: float

    def __post_init__(self) -> None:
        assert self.ks[-1] == QN.N and self.K in self.ks and list(self.ks) == sorted(self.ks), self.ks
        assert self.window_rows >= self.n_top and min(self.quantile_rows, self.check_rows, self.n_draws) > 1


_sizes: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"]["sizes"].items()}
CFG = Config(**_sizes)


def lab_class() -> Any:
    E91 = load_experiment("qwen_size_grid", "run")

    class Lab(E91.JOB.Job):  # E45's tokenizer, problems and model (as E92); the log is this experiment's
        def log(self, msg: str) -> None:
            line = f"[{time.time() - self.t0:7.1f}s] {msg}"
            print(line, flush=True)
            with (RESULTS / "run.log").open("a") as f:
                f.write(line + "\n")

    return Lab, E91


def stop(lab: Any, why: str, detail: Any) -> None:
    STOPPED.write_text(json.dumps({"stopped": why, "detail": detail}, indent=1, default=float) + "\n")
    lab.log(f"STOP: {why}: {json.dumps(detail, default=float)}")
    hard_exit(1)


def check(lab: Any, out: dict[str, Any], name: str, holds: bool, detail: dict[str, Any]) -> None:
    """Logged when it passes too; a miss writes stopped.json and exits 1."""
    out.setdefault("checks", {})[name] = {"holds": bool(holds), **detail}
    lab.log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        stop(lab, name, detail)


def rel(x: float, ref: float) -> float:
    return abs(x - ref) / max(abs(ref), 1e-12)


# ----------------------------------------------------------------------------- rows


def row_sets(lab: Any) -> dict[str, qc.Rows]:
    items = json.loads((lab.LARGE_P / "a1_rows.json").read_text())["items"]
    kept = {rec["row"] for rec in map(json.loads, FORMS.read_text().splitlines()) if rec["kept"]}
    return {"a1": qc.a1_rows(lab.tok, lab.sets["test"], items), "tofu": qc.tofu_rows(lab.tok, lab.end_id, kept)}


def contrast_seqs(lab: Any) -> tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]:
    """Anchor 1's 256 items (E92's a1 rows before packing): the same answer ids after the unprompted chat and after the
    prompted one (E45's chosen system turn, `format`, no shots: the behaviour W0 follows on 94.1%)."""
    from commute.qwen_anchors import candidate
    from commute.qwen_eval import prompt

    items = json.loads((lab.LARGE_P / "a1_rows.json").read_text())["items"]
    chosen = json.loads((EXPERIMENTS / "qwen_updates" / "results" / "P.json").read_text())["chosen"]
    out: tuple[list[Any], list[Any]] = ([], [])
    for r in items:
        q = lab.sets["test"][r["idx"]]["question"]
        for side, text in enumerate((prompt(lab.tok, q), candidate(lab.tok, chosen, q, ()))):
            p = lab.ids(text)
            out[side].append((p + r["ids"], len(p)))
    return out


def fineweb(lab: Any) -> torch.Tensor:
    path, n = LARGE / "fineweb_rows.safetensors", CFG.window_rows + CFG.quantile_rows + CFG.check_rows
    if not path.is_file():
        from commute.qwen_data import fineweb_rows

        t0 = time.time()
        save_file({"rows": torch.tensor(fineweb_rows(lab.tok, n), dtype=torch.int32)}, str(path))
        lab.log(f"[fineweb] {n} rows of 512 fetched in {time.time() - t0:.0f} s")
    rows = load_file(str(path))["rows"].long()
    assert rows.shape == (n, 512), rows.shape
    return rows


# ----------------------------------------------------------------------------- the W0 stage


def stage_w0(lab: Any, model: Any, rows: dict[str, qc.Rows]) -> None:
    C, out = P["controls"], {}
    sub = qc.Rows("a1_16", *(x[:16] for x in (rows["a1"].tokens, rows["a1"].answer, rows["a1"].first)))
    r = QN.rank_state(model, sub, lab.W0, CFG.batch)
    nums = QN.grid(model, sub, lab.W0, torch.arange(QN.N), [QN.N], CFG.batch)
    dev = float((r["T1"] - r["T0"]).abs().max())
    check(lab, out, "E0_do_nothing", dev <= C["E0_max"] and abs(nums[QN.N]) <= C["E0_max"],
          {"max_T1_T0": dev, "num_N": nums[QN.N], "max_score": float(r["score"].abs().max()),
           "set_size": len(QN.top_set(r["score"].sum(0), CFG.K))})
    for name, R in rows.items():
        got = float(qc.read(model, R).logp.double().mean())
        ref = json.loads((E92 / f"W0_{name}.json").read_text())["all"]["mean_logp"]
        check(lab, out, f"E1_W0_{name}", rel(got, ref) <= C["E1_rel"], {"here": got, "e92": ref})
    t0 = time.time()
    unp, pro = contrast_seqs(lab)
    pad = lab.end_id
    con = QN.answer_means(model, pro, CFG.contrast_batch, pad) - QN.answer_means(model, unp, CFG.contrast_batch, pad)
    sets = {"even": QN.top_set(con[0], CFG.K), "odd": QN.top_set(con[1], CFG.K), "all": QN.top_set(con.sum(0), CFG.K)}
    draws = QN.random_overlaps(sets["even"], sets["odd"], CFG.n_draws, torch.Generator().manual_seed(CFG.seed))
    halves = QN.overlap(sets["even"], sets["odd"])
    out["contrast"] = {"halves": halves, "draws": draws, "sets": sets, "seconds": time.time() - t0}
    save_file({"contrast": con.contiguous()}, str(LARGE / "contrast.safetensors"))
    ok = QN.finite(halves) and len(draws) >= 2 and halves - max(draws) >= C["min_gap"]
    check(lab, out, "C1a_contrast_halves", ok, {"halves": halves, "draw_max": max(draws, default=None), "sizes": {k: len(v) for k, v in sets.items()}})
    corpus(lab, model, rows, out)
    out["passed"] = True
    put(W0F, out)


def corpus(lab: Any, model: Any, rows: dict[str, qc.Rows], out: dict[str, Any]) -> None:
    fw, t0 = fineweb(lab), time.time()
    w, q = CFG.window_rows, CFG.window_rows + CFG.quantile_rows
    tops = QN.corpus_tops(model, fw[:w], CFG.corpus_batch, CFG.n_top)
    save_file({**tops, "fineweb_rows": fw[:w].int().contiguous()}, str(LARGE / "tops.safetensors"))
    lab.log(f"[corpus] top {CFG.n_top} of {w} rows per neuron in {time.time() - t0:.0f} s, {lab.peaks()}")
    sample = QN.sorted_sample(model, fw[w:q])
    ones = torch.ones_like(fw[q:], dtype=torch.bool)
    pile = QN.engaged(model, qc.Rows("check", fw[q:], ones, ones), sample, CFG.batch)
    band = P["controls"]["C0_band"]
    check(lab, out, "C0_corpus", band[0] <= float(pile.median()) <= band[1], {"median": float(pile.median())})
    eng = {name: QN.engaged(model, R, sample, CFG.batch) for name, R in rows.items()}
    save_file({k: v.contiguous() for k, v in eng.items()}, str(LARGE / "engaged.safetensors"))
    out["corpus"] = {"seconds": time.time() - t0, "median_engaged": {k: float(v.median()) for k, v in eng.items()}}
    del sample


# ----------------------------------------------------------------------------- the Gaussian change and control (i)


def gaussian(lab: Any, model: Any, E91: Any) -> dict[str, torch.Tensor]:
    """E94's gauss_small direction (`qwen_d3_rows/run.py` gaussian): `split.random_like` of anchor 1 seed 0's δW per
    site, Σ = W0's input moments on E57's 128 maths rows of 512 (as `qwen_d3.Decomp.sigma`), seed `null_seed(0, site)`.
    Scaled by b to anchor 1's KL on E91's 16 maths rows (a departure: E94 matches its masked K00 - K0). bf16, saved."""
    from commute import qwen_d3 as d3
    from commute import qwen_maps, split
    from commute import qwen_reading as qr
    from commute.qwen_sft import mean_kl
    from commute.qwen_vpd import SITES

    path = LARGE / "gauss_unit.safetensors"
    if not path.is_file():
        t0, ref = time.time(), d3.load_state(d3.grid_state("anchor1_s0", "full"))
        mrows = torch.tensor([r[: d3.SIGMA_LEN] for r in json.loads(d3.MATHS.read_text())["rows"][: d3.MATHS_ROWS]])
        sig = qwen_maps.input_moments(model, list(mrows.to(E91.DEV).split(d3.SIGMA_BATCH)), list(SITES))
        unit = {s: split.random_like(ref[s].to(E91.DEV, d3.F64), sig[s].to(E91.DEV), d3.null_seed(0, s)).to(torch.bfloat16).cpu()
                for s in SITES}
        del sig, ref
        save_file(unit, str(path))
        lab.log(f"[gauss] direction built in {time.time() - t0:.0f} s, {lab.peaks()}")
    unit = load_file(str(path))
    if not GAUSS.is_file():
        rows16 = lab.maths()[:16]
        target = json.loads((EXPERIMENTS / "qwen_size_grid" / "results" / "size_grid.json").read_text())["arms"]["anchor1_s0"]["full"]["kl"]
        G = P["settings"]["gauss"]
        srch = qr.scale_search(lambda b: mean_kl(model, rows16, lab.W0, scaled(lab.W0, unit, b)), target, G["match"], G["evals"])
        norm = math.sqrt(sum(float(u.double().pow(2).sum()) for u in unit.values())) * srch["b"]
        put(GAUSS, {"search": srch, "b": srch["b"], "norm": norm, "size": "matched" if srch["matched"] else "size unmatched"})
        lab.log(f"[gauss] {json.dumps(json.loads(GAUSS.read_text()), default=float)}")
    return scaled(lab.W0, unit, json.loads(GAUSS.read_text())["b"])


def scaled(W0: dict[str, torch.Tensor], unit: dict[str, torch.Tensor], b: float) -> dict[str, torch.Tensor]:
    return {s: W0[s] + b * unit[s].to(W0[s].device, torch.float32) for s in W0}


def control(lab: Any) -> None:
    w0, g = json.loads(W0F.read_text()), json.loads((STATES / "gauss_a1.json").read_text())
    c = w0["contrast"]
    rec = QN.control_i(c["halves"], c["draws"], QN.overlap(g["carry"], c["sets"]["all"]), P["controls"]["min_gap"])
    put(CONTROL, rec)
    lab.log(f"[control i] {json.dumps(rec, default=float)}")
    if not rec["passed"]:
        stop(lab, "control (i) missed", rec)


# ----------------------------------------------------------------------------- one state


def read_state(lab: Any, model: Any, R: qc.Rows, name: str, W: dict[str, torch.Tensor], e92: Path | None) -> None:
    t0 = time.time()
    (LARGE / "states").mkdir(parents=True, exist_ok=True)
    r = QN.rank_state(model, R, W, CFG.batch)
    d = r["T1"] - r["T0"]
    G = qc.share(d, d, r["row_of"])
    lab.log(f"[{name} rank] gain {G['gain']:.6g} se {G['gain_se']}, {time.time() - t0:.0f} s, {lab.peaks()}")
    out: dict[str, Any] = {"name": name, "rows": R.name, "gain": G}
    if e92 is not None:
        ref = json.loads(e92.read_text())["gain"]["all"]["gain"]
        check(lab, out, f"E1_gain_{name}", rel(G["gain"], ref) <= P["controls"]["E1_rel"], {"here": G["gain"], "e92": ref})
    total = r["score"].sum(0)
    nums = QN.grid(model, R, W, torch.sort(total, descending=True, stable=True).indices, CFG.ks, CFG.batch)
    full = nums[QN.N]
    halves = [QN.top_set(r["score"][h], CFG.K) for h in (0, 1)]
    out |= {"nums": {str(k): v for k, v in nums.items()}, "k_half": QN.k_half(nums, G, P["settings"]["carry_floor"]),
            "neuron_share": full / G["gain"] if G["share"] is not None else None,
            "top_K_of_numerator": nums[CFG.K] / full if full != 0 else None,
            "first_order_over_numerator": float(total.sum()) / len(d) / full if full != 0 else None,
            "carry": QN.top_set(total, CFG.K), "halves_overlap": QN.overlap(*halves), "seconds": time.time() - t0}
    save_file({k: r[k].contiguous() for k in ("T0", "T1", "row_of", "score")}, str(LARGE / "states" / f"{name}.safetensors"))
    put(STATES / f"{name}.json", out)
    lab.log(f"[{name}] {json.dumps({k: v for k, v in out.items() if k != 'carry'}, default=float)}, {lab.peaks()}")


def budget(lab: Any, t_start: float, est: float) -> None:
    if time.time() - t_start + est > CFG.budget_s:
        lab.log(f"BUDGET: {time.time() - t_start:.0f} s used, next unit about {est:.0f} s; resubmit the same command")
        hard_exit(3)


def states(lab: Any, model: Any, E91: Any, names: list[str], rows: dict[str, qc.Rows], t_start: float) -> None:
    from commute import qwen_d3 as d3
    from commute.qwen_sft import load, plus

    per_row = P["settings"]["first_guess_s_per_row"]
    for name in names:
        if name == "gauss":
            if not all((STATES / f"{g}.json").is_file() for g in GAUSS_READS):
                budget(lab, t_start, 300 + per_row * sum(len(R.tokens) for R in rows.values()))
                W = gaussian(lab, model, E91)
                for g in GAUSS_READS:
                    if not (STATES / f"{g}.json").is_file():
                        read_state(lab, model, rows[g.split("_")[1]], g, W, None)
                del W
            control(lab)
            continue
        if (STATES / f"{name}.json").is_file():
            lab.log(f"[{name}] already written")
            continue
        arm, size, which = NAMED[name]
        rec = d3.grid_state(arm, size)
        budget(lab, t_start, 30 + per_row * len(rows[which].tokens))
        t0 = time.time()
        W = plus(lab.W0, load(rec["path"], E91.DEV))
        read_state(lab, model, rows[which], name, W, E92 / "states" / f"{arm}_step{rec['step']}.json")
        per_row = max(per_row, (time.time() - t0 - 30) / len(rows[which].tokens))
        del W


# ----------------------------------------------------------------------------- main


def filed() -> None:
    """Refuse unless predictions.json is committed and unchanged (filed before the run)."""
    rel_path, repo = str(PRED.relative_to(EXPERIMENTS.parent)), str(EXPERIMENTS.parent)
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False)
    if git("diff", "--quiet", "HEAD", "--", rel_path).returncode or not git("ls-files", rel_path).stdout.strip():
        sys.exit(f"{rel_path} is not committed unchanged: commit it first")


def main() -> None:
    ap = argparse.ArgumentParser(description="E102 stage A")
    ap.add_argument("--states", help="comma-separated: gauss, then names of " + ", ".join(NAMED))
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score:
        print(json.dumps({k: v for k, v in SC.score().items() if k != "per_state"}, indent=1, default=float))
        return
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    names = a.states.split(",") if a.states else []
    if not names or any(n != "gauss" and n not in NAMED for n in names) or ("gauss" in names and names[0] != "gauss"):
        sys.exit("give --states from: gauss (first, if given), " + ", ".join(NAMED))
    if "gauss" not in names and not CONTROL.is_file():
        sys.exit("control (i) has not passed: run the first job (--states gauss,...) first")
    Lab, E91 = lab_class()
    lab = Lab("A", "E102", PRED)
    lab.LARGE_P = E91.JOB.LARGE / "P"
    t_start, model = time.time(), lab.fp32()
    model.eval()
    rows = row_sets(lab)
    lab.log(f"[rows] {json.dumps({k: [list(R.tokens.shape), int(R.answer.sum())] for k, R in rows.items()})}; {CFG}")
    if not W0F.is_file():
        stage_w0(lab, model, rows)
    states(lab, model, E91, names, rows, t_start)
    sc = SC.score()
    lab.log(f"[score] {json.dumps({k: v for k, v in sc.items() if k not in ('per_state', 'stage_b_w0_picks_tofu')}, default=float)}")
    lab.log(f"[done] {names}, {lab.peaks()}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
