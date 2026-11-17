"""P3.6 run 2 (qwen_cheap_readings): the top-set overlap of the 19 saved Qwen3-1.7B updates, each beside a random change
at its own per-matrix norms, E92's columns for the 8 updates E92 has no row for, and the three contrasts with seed spread.
No decomposition is read, nothing is trained, no update is labelled reuse or new; the overlap is a description. Run 1:
run1.py (E162). Card card_run2.md; rules results/predictions_run2.json (the run refuses unless it is committed unchanged),
applied by rules_run2.py; the overlap is `commute.qwen_overlap`, E92's columns `commute.qwen_comparators.read_state`.

    sbatch --comment='E### run2' cluster/qwen_cheap_readings_run2.sbatch

Rows: E92's for each update's kind (anchor 1, TOFU, GSM8K gold); E120's pools read the gold rows, one row set for both.
An update's floor: CFG.n_floor_draws random changes at its per-matrix norms, seeds from CFG.first_seed (run 1's first two).
  stage 0  alone: the do-nothing update reads overlap exactly 1.0 and on_both nan, and through read_state norm 0, log P
           unchanged, no layer; W0's log P and anchor 1 seed 0 reproduce E92's stored reads (E118's C1). Miss: exit 1.
  stage 1  the floor draws at the two reference updates, before any real update; band = their larger relative spread
           (rules_run2.band). Unreadable or over CFG.band_max: exit 1.
  stage 2  per update: norm2 against its record (else exit 1), the real overlap, its floor (stage 1's for a reference),
           the rules; E92's readings where E92 has no row. Written updates are skipped; projected over the limit: exit 3.
  stage 3  CPU: run 1's erank, E92's columns and the overlap joined per update; the call; the contrasts.
Each stage writes results/run2_stage<n>.json and prints "STAGE n DONE"; every number is printed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from commute import qwen_comparators as qc
from commute import qwen_overlap as QO
from commute import qwen_sets
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import linears, load, plus
from commute.qwen_vpd import HF_NAME, HF_REVISION, fp32_matmuls, load_target

R1 = load_experiment("qwen_cheap_readings", "run1")  # updates(), recorded_norm2(), E92 (run 1's join)
RULES = load_experiment("qwen_cheap_readings", "rules")  # svd_call: the majority rule, reused as the overlap's call
R2 = load_experiment("qwen_cheap_readings", "rules_run2")
E92 = load_experiment("qwen_comparators", "run")  # A1_ITEMS, FORMS, STEER_SEED, RESULTS: its row inputs and stored reads
RESULTS = Path(__file__).resolve().parent / "results"
PRED = RESULTS / "predictions_run2.json"
GOLD = ("R_s0", "R_s70", "N_s0", "N_s70", "F_s0", "u1", "u2", "e40", "gold_s0", "known_s0", "known_s1", "new_s0", "new_s1")
ROWS = {"a1_s0": "a1", "a1_s1": "a1", **dict.fromkeys(("a2_s42", "a2_s142", "a2long_s0", "a2long_s1"), "tofu"), **dict.fromkeys(GOLD, "gold")}
READINGS = ("erank", "overlap", "on_both", "size", "prior_rank", "steer", "random")


@dataclass(frozen=True)
class Config:
    n_floor_draws: int = 4  # decision card 2026-11-11 (qwen-overlap-floor): option (a) 4, option (b) 2
    floor_refs: tuple[str, ...] = ("F_s0", "a2_s42")  # the smallest and the largest norm of the 19 (run 1)
    first_seed: int = 162000  # draws 162000 and 162001 are run 1's
    band_max: float = 0.33  # stop: 10x run 1's largest two-draw spread (0.0332, F_s0)
    zero_logp_max: float = 1e-5  # E92's and E118's do-nothing bar
    norm_rel: float = 1e-9  # E118 C1: the same file summed in fp64
    rel: float = 1e-3  # E118 C1 (E102's E1_rel)
    norm2_rel_tol: float = 1e-6  # run 1's C2
    projected_limit_min: float = 100.0

    def __post_init__(self) -> None:
        assert self.n_floor_draws >= 2 and len(self.floor_refs) == 2, "a band needs two draws at each of two references"

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.first_seed, self.first_seed + self.n_floor_draws))


CFG = Config()


def log(msg: str) -> None:
    print(msg, flush=True)


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"run2_stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(RESULTS / f"run2_stage{n}.json")


def stored(n: int) -> dict[str, Any] | None:
    p = RESULTS / f"run2_stage{n}.json"
    out = json.loads(p.read_text()) if p.is_file() else None
    return None if out is None or "stopped" in out else out


class Stopped(Exception):
    """A filed stop: the stage's file records why."""


def check(n: int, out: dict[str, Any], name: str, holds: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"holds": bool(holds), **detail}
    log(f"[{n} check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail)}")
    if not holds:
        out["stopped"] = name
        write(n, out)
        raise Stopped(name)


def rel(x: float | None, ref: float) -> float:
    return abs(x - ref) / abs(ref) if x is not None else math.inf


class Lab:
    """The fp32 model at W0, each row set and W0's E92 read of it, built once per job."""

    def __init__(self) -> None:
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(HF_NAME, revision=HF_REVISION)
        self.end_id, self.test = int(self.tok.convert_tokens_to_ids("<|im_end|>")), qwen_sets.load()["test"]
        self.model = load_target("cuda", torch.float32)
        self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        self._rows: dict[str, qc.Rows] = {}
        self._w0: dict[str, qc.Read] = {}

    def rows(self, name: str) -> qc.Rows:
        if name not in self._rows:
            if name == "a1":
                R = qc.a1_rows(self.tok, self.test, json.loads(E92.A1_ITEMS.read_text())["items"])
            elif name == "tofu":
                R = qc.tofu_rows(self.tok, self.end_id, {r["row"] for r in map(json.loads, E92.FORMS.read_text().splitlines()) if r["kept"]})
            else:
                R = qc.gold_rows(self.tok, self.test, self.end_id)
            self._rows[name] = R
            log(f"[rows {name}] {list(R.tokens.shape)}, {int(R.answer.sum())} answer positions")
        return self._rows[name]

    def w0(self, name: str) -> qc.Read:
        if name not in self._w0:
            self._w0[name] = qc.read(self.model, self.rows(name))
        return self._w0[name]

    def draws(self, label: str, rows: str, norms: dict[str, float]) -> list[dict[str, Any]]:
        out = []
        for seed in CFG.seeds:
            t0 = time.time()
            ov = QO.overlap(self.model, self.rows(rows), QO.random_weights(self.W0, norms, seed))
            out.append({"seed": seed, **ov})
            log(f"[floor {label} seed {seed}] overlap {ov['overlap']:.6f} on_both {ov['on_both']:.6f}; {time.time() - t0:.0f} s")
            torch.cuda.empty_cache()
        return out


def loaded(name: str, path: Path, rec2: dict[str, float], n: int, out: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """The update on the card and its per-matrix norms; stops if its summed squares are off its record (the wrong file)."""
    dW = load(path, "cuda")
    norms = {s: float(x.double().norm()) for s, x in dW.items()}
    off = sum(v * v for v in norms.values()) / rec2[name] - 1
    check(n, out, f"norm2 {name}", abs(off) <= CFG.norm2_rel_tol, {"norm2_vs_record": off})
    return dW, norms


# ----------------------------------------------------------------------------- stage 0, the controls alone


def stage0(lab: Lab, files: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    zero = {s: torch.zeros_like(w) for s, w in lab.W0.items()}
    ov = QO.overlap(lab.model, lab.rows("gold"), plus(lab.W0, zero))
    exact = ov["overlap"] == 1.0 and math.isnan(ov["on_both"]) and all(b["overlap_min"] == 1.0 for b in ov["per_block"].values())
    check(0, out, "C0_zero_overlap", exact, {"overlap": ov["overlap"], "on_both": ov["on_both"], "positions": ov["positions"]})
    rec, st = qc.read_state(lab.model, lab.W0, zero, lab.rows("gold"), lab.w0("gold"), E92.STEER_SEED)
    worst = float((st.logp - lab.w0("gold").logp).abs().max())
    check(0, out, "C0_zero_e92", rec["size"]["total"]["norm"] == 0 and worst <= CFG.zero_logp_max and rec["steering"]["layer"] is None,
          {"norm": rec["size"]["total"]["norm"], "max_logp_diff": worst, "layer": rec["steering"]["layer"]})  # fmt: skip
    del zero
    for name in ("a1", "tofu", "gold"):
        ref = json.loads((E92.RESULTS / f"W0_{name}.json").read_text())["all"]["mean_logp"]
        got = qc.summary(lab.w0(name))["all"]["mean_logp"]
        check(0, out, f"C1_W0_{name}", rel(got, ref) <= CFG.rel, {"here": got, "e92": ref})
    T = json.loads((EXPERIMENTS / R1.E92["a1_s0"]).read_text())
    rec, _ = qc.read_state(lab.model, lab.W0, load(files["a1_s0"], "cuda"), lab.rows("a1"), lab.w0("a1"), E92.STEER_SEED)
    sv, tv = rec["steering"], T["steering"]
    got = {k: sv[k]["all"]["share"] if sv[k] else None for k in ("steer", "random")}
    holds = (rel(rec["size"]["total"]["norm"], T["size"]["total"]["norm"]) <= CFG.norm_rel and sv["layer"] == tv["layer"]
             and rel(rec["gain"]["all"]["gain"], T["gain"]["all"]["gain"]) <= CFG.rel
             and all(got[k] is not None and abs(got[k] - tv[k]["all"]["share"]) <= tv[k]["all"]["share_se"] for k in got))  # fmt: skip
    check(0, out, "C1_e92_a1_s0", holds, {"norm": rec["size"]["total"]["norm"], "gain": rec["gain"]["all"]["gain"], "layer": sv["layer"],
                                          "shares": got, "stored": {k: tv[k]["all"] for k in got}})  # fmt: skip
    return out


# ----------------------------------------------------------------------------- stage 1, the floor band


def stage1(lab: Lab, files: dict[str, Path], rec2: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"refs": {}}
    for ref in CFG.floor_refs:
        dW, norms = loaded(ref, files[ref], rec2, 1, out)
        del dW
        out["refs"][ref] = {"rows": ROWS[ref], "draws": lab.draws(ref, ROWS[ref], norms)}
        write(1, out)
    O = {r: [d["overlap"] for d in v["draws"]] for r, v in out["refs"].items()}
    B = {r: [d["on_both"] for d in v["draws"]] for r, v in out["refs"].items()}
    out["band"] = {"overlap": R2.band(O, "overlap"), "on_both": R2.band(B, "share"),
                   "spreads": {r: {"overlap": R2.spread(O[r], "overlap"), "on_both": R2.spread(B[r], "share")} for r in O}}  # fmt: skip
    run1 = json.loads((RESULTS / "run1_stage1.json").read_text())["refs"]["F_s0"]["draws"]  # recorded, not a stop
    out["vs_run1"] = {str(d["seed"]): d["overlap"] - run1[str(d["seed"])]["overlap"] for d in out["refs"]["F_s0"]["draws"] if str(d["seed"]) in run1}
    log(f"[1 band] {json.dumps(out['band'])}; F_s0 draws minus run 1's: {json.dumps(out['vs_run1'])}")
    b = out["band"]
    check(1, out, "C2_band", b["overlap"] is not None and b["on_both"] is not None and b["overlap"] <= CFG.band_max, b)
    return out


# ----------------------------------------------------------------------------- stage 2, every update


def against(x: float, draws: list[dict[str, Any]], key: str, band: float | None) -> dict[str, Any]:
    kind = "overlap" if key == "overlap" else "share"
    v = [d[key] for d in draws]
    return {"value": x, "floor": sum(v) / len(v), "draws": v, "vs_floor": R2.vs_floor(x, v, band, kind),
            "reproducible": R2.reproducible(v, band, kind), "rel_gap": R2.rel_gap(x, v, kind)}  # fmt: skip


def update(lab: Lab, name: str, path: Path, rec2: dict[str, float], s1: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    t0, rows = time.time(), ROWS[name]
    dW, norms = loaded(name, path, rec2, 2, out)
    real = QO.overlap(lab.model, lab.rows(rows), plus(lab.W0, dW))
    log(f"[2 {name} real] overlap {real['overlap']:.6f} on_both {real['on_both']:.6f} on {rows}")
    draws = s1["refs"][name]["draws"] if name in s1["refs"] else lab.draws(name, rows, norms)
    rec: dict[str, Any] = {"file": str(path.relative_to(EXPERIMENTS)), "rows": rows, "real": real,
                           "overlap": against(real["overlap"], draws, "overlap", s1["band"]["overlap"]),
                           "on_both": against(real["on_both"], draws, "on_both", s1["band"]["on_both"]), "floor_draws": draws}  # fmt: skip
    if name not in R1.E92:
        rec["e92"], _ = qc.read_state(lab.model, lab.W0, dW, lab.rows(rows), lab.w0(rows), E92.STEER_SEED)
        log(f"[2 {name} e92] {json.dumps(rec['e92'])}")
    del dW
    torch.cuda.empty_cache()
    rec["seconds"] = time.time() - t0
    log(f"[2 {name}] {json.dumps({k: rec[k] for k in ('overlap', 'on_both', 'seconds')})}")
    return rec


def stage2(lab: Lab, files: dict[str, Path], rec2: dict[str, float], s1: dict[str, Any], t_job: float) -> dict[str, Any]:
    p = RESULTS / "run2_stage2.json"
    out: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"states": {}}
    out.pop("stopped", None)
    todo, t1 = [n for n in files if n not in out["states"]], time.time()
    for i, name in enumerate(todo):
        out["states"][name] = update(lab, name, files[name], rec2, s1, out)
        write(2, out)
        minutes = (time.time() - t_job + (time.time() - t1) / (i + 1) * (len(todo) - i - 1)) / 60
        log(f"[2 time] {i + 1} of {len(todo)} this job; projected {minutes:.1f} min for the job")
        if i < len(todo) - 1 and minutes > CFG.projected_limit_min:
            raise SystemExit(3)
    return out


# ----------------------------------------------------------------------------- stage 3, the join and the contrasts


def e92_columns(name: str, s: dict[str, Any]) -> dict[str, float | None]:
    rec = s["e92"] if "e92" in s else json.loads((EXPERIMENTS / R1.E92[name]).read_text())
    st = rec["steering"]
    return {"size": rec["size"]["total"]["ratio_norm"], "prior_rank": rec["rank_W0"]["first"]["median_rank"],
            "steer": st["steer"]["all"]["share"] if st["steer"] else None, "random": st["random"]["all"]["share"] if st["random"] else None}  # fmt: skip


def stage3(s2: dict[str, Any]) -> dict[str, Any]:
    run1 = json.loads((RESULTS / "run1_stage2.json").read_text())["states"]
    st = s2["states"]
    per = {n: {"erank": run1[n]["erank"], "erank_floor": run1[n]["floor_erank"], "overlap": s["overlap"]["value"],
               "overlap_floor": s["overlap"]["floor"], "overlap_rel_gap": s["overlap"]["rel_gap"], "on_both": s["on_both"]["value"],
               "on_both_floor": s["on_both"]["floor"], "on_both_rel_gap": s["on_both"]["rel_gap"], "rows": s["rows"],
               "e92_source": "run 2" if "e92" in s else R1.E92[n], **e92_columns(n, s)} for n, s in st.items()}  # fmt: skip
    calls = {n: s["overlap"]["vs_floor"] for n, s in st.items()}

    def tally(key: str, field: str) -> dict[str, int]:
        got = [s[key][field] for s in st.values()]
        return {v: got.count(v) for v in sorted(set(got))}

    out = {"per_update": per, "overlap_vs_floor": calls, "call": RULES.svd_call(list(calls.values())), "counts": tally("overlap", "vs_floor"),
           "on_both_counts": tally("on_both", "vs_floor"), "reproducible": {k: tally(k, "reproducible") for k in ("overlap", "on_both")},
           "contrasts": R2.contrasts(per, READINGS)}  # fmt: skip
    log(f"[3 score] {json.dumps({k: v for k, v in out.items() if k != 'per_update'})}")
    return out


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    filed = {k: P["settings"].get(k) for k in asdict(CFG)}
    assert filed == json.loads(json.dumps(asdict(CFG))) and P["settings"]["k"] == QO.K, (filed, asdict(CFG))
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    files, rec2 = R1.updates(), R1.recorded_norm2()
    missing = [str(p) for p in files.values() if not p.is_file()]
    assert set(files) == set(ROWS) and not missing and len([n for n in files if n not in R1.E92]) == 8, missing
    lab = Lab()
    try:
        if stored(0) is None:
            write(0, stage0(lab, files))
        log("STAGE 0 DONE")
        s1 = stored(1)
        if s1 is None or "band" not in s1:
            s1 = stage1(lab, files, rec2)
            write(1, s1)
        log(f"STAGE 1 DONE; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        s2 = stage2(lab, files, rec2, s1, t_job)
        log(f"STAGE 2 DONE; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        write(3, stage3(s2))
        log("STAGE 3 DONE")
    except Stopped as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    except SystemExit as e:
        log(f"EXIT {e.code}: more to do, resubmit the same command")
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    hard_exit(main())
