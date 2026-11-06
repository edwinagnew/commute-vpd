"""E103's second job: E102 stage A's neuron readings on the known-answer pair. Needs run.py's train.json done.
Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md. Rules: results/predictions.json (`neurons`).

E102's code (`qwen_carrying/run.py --states`) reads only E94's six named states and writes into E102's own results, so
this file calls the same functions in `commute.qwen_neurons` and reads E102's saved W0 files (the prompted contrast set,
the TOFU rows' engaged values, the FineWeb rows, the Gaussian direction and its scale) without rebuilding them.

  C2 first and alone: the fictional matched state, seed 42, re-read on the TOFU rows, reproduces E102's gain and
     carrying set, or the job stops (1 min on an L40).
  Then each state's carrying set on the real rows (both new states, the four fictional states, the Gaussian change)
  and on the TOFU rows (the two new states; E102's own sets for the rest), and the engaged values of the real rows.
  Each read is written to results/neurons/ as it ends and skipped on a resume; the score goes to results/neurons.json.
Exit 0 done; 1 a stop (results/stopped.json).
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from commute import carrying_jobs as CJ
from commute import qwen_comparators as qc
from commute import qwen_d3 as d3
from commute import qwen_neurons as QN
from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load, plus

RUN: Any = load_experiment("known_answer_pair", "run")  # the paths, Lab, stop, filed
E102: Any = load_experiment("qwen_carrying", "run")  # its Config, fineweb, scaled
CFG, E102R = E102.CFG, E102.RESULTS
OUT_DIR, SCORE = RUN.RESULTS / "neurons", RUN.RESULTS / "neurons.json"
ENGAGED_REAL = RUN.LARGE / "engaged_real.safetensors"
FICT = {"fict_s42_matched": ("anchor2_s0", "at_norm"), "fict_s142_matched": ("anchor2_s142", "at_norm"),
        "fict_s42_full": ("anchor2_s0", "full"), "fict_s142_full": ("anchor2_s142", "full")}
E102_NAME = {"fict_s42_matched": "a2_s42_matched", "fict_s142_matched": "a2_s142_matched", "fict_s42_full": "a2_s42_full",
             "fict_s142_full": "a2_s142_full", "gauss": "gauss_tofu"}  # E102's reads on the TOFU rows
PAIRS = {"matched": ("real_matched", ("fict_s42_matched", "fict_s142_matched")),
         "full": ("real_full", ("fict_s42_full", "fict_s142_full"))}


def real_rows(lab: Any) -> qc.Rows:
    """Pair 0 of the 217 real rows (TOFU's own wording, as trained), packed in rows of 256 as E92's TOFU rows; rows the
    wordings dropped stay as context and mark no answer."""
    kept = set(json.loads((RUN.RESULTS / "forms.json").read_text())["kept_rows"])
    seqs = []
    for i, r in enumerate(tofu.real_rows()):
        ids, p = tofu.seq(lab.tok, lab.end_id, r["question"] + tofu.ASK, r["answer"])
        seqs.append((ids, p if i in kept else len(ids)))
    return qc.rows_of("real", seqs, qc.ROW_LEN, qc.ROW_LEN)


def weights(lab: Any, name: str) -> dict[str, torch.Tensor]:
    if name == "gauss":
        unit = load_file(str(E102.LARGE / "gauss_unit.safetensors"))
        return E102.scaled(lab.W0, unit, json.loads(E102.GAUSS.read_text())["b"])
    path = {"real_matched": RUN.MATCHED, "real_full": RUN.END}.get(name) or d3.grid_state(*FICT[name])["path"]
    return plus(lab.W0, load(path, RUN.DEV))


def read(lab: Any, R: qc.Rows, name: str, W: dict[str, torch.Tensor]) -> dict[str, Any]:
    """One state on one row set: its gain and top-K carrying set (E98's ranking, as E102's `read_state`)."""
    t0 = time.time()
    r = QN.rank_state(lab.model, R, W, CFG.batch)
    d = r["T1"] - r["T0"]
    rec = {"name": name, "rows": R.name, "gain": qc.share(d, d, r["row_of"]), "carry": QN.top_set(r["score"].sum(0), CFG.K),
           "seconds": time.time() - t0}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}__{R.name}.json").write_text(json.dumps(rec, default=float) + "\n")
    lab.log(f"[{name} on {R.name}] gain {rec['gain']['gain']:.5g}, set {len(rec['carry'])}, {rec['seconds']:.0f} s, {lab.peaks()}")
    return rec


def control(lab: Any, R: qc.Rows) -> None:
    C, ref = lab.pred["neurons"]["C2"], json.loads((E102R / "states" / "a2_s42_matched.json").read_text())
    W = weights(lab, "fict_s42_matched")
    rec = read(lab, R, "fict_s42_matched", W)
    rel, same = abs(rec["gain"]["gain"] / ref["gain"]["gain"] - 1), QN.overlap(rec["carry"], ref["carry"])
    detail = {"gain": rec["gain"]["gain"], "e102_gain": ref["gain"]["gain"], "rel": rel, "set_overlap": same}
    holds = rel <= C["gain_rel"] and same >= C["min_set_overlap"]
    lab.log(f"[C2] {'ok' if holds else 'MISSED'} {json.dumps(detail)}")
    if not holds:
        RUN.stop(lab, "C2: E102's read not reproduced", detail)


def pair_side(known: float | None, new: Sequence[float | None], floor: float | None, draws: Sequence[float]) -> dict[str, Any]:
    """The filed rule: the known-answer state against the fictional state's two seeds, on the same rows and size.
    band = the larger of the two seeds' spread and twice the sd of the known state's random draws. "known higher" iff
    known - mean(new) > band and known > floor; "new higher" iff mean(new) - known > band and mean(new) > floor; "level"
    otherwise. floor = the larger of the Gaussian change's value and every draw. "unread" with any value missing or not
    finite, or under two finite draws."""
    ok = [x for x in draws if QN.finite(x)]
    if not QN.finite(known) or len(new) != 2 or not all(QN.finite(x) for x in new) or not QN.finite(floor) or len(ok) < 2:
        return {"verdict": "unread", "known": known, "new": list(new), "floor": floor, "n_draws": len(ok)}
    k, n0, n1, g = float(known), float(new[0]), float(new[1]), float(floor)  # type: ignore[arg-type]
    m, band, floor_all = (n0 + n1) / 2, max(abs(n0 - n1), 2 * statistics.stdev(ok)), max(g, max(ok))
    verdict = ("known higher" if k - m > band and k > floor_all else "new higher" if m - k > band and m > floor_all else "level")
    return {"verdict": verdict, "known": k, "new": [n0, n1], "gap": k - m, "band": band, "floor": floor_all}


def measures(carry: list[int], eng: list[float], contrast: list[int], gen: torch.Generator) -> dict[str, Any]:
    n = CFG.n_draws
    return {"n": len(carry), "prompt_overlap": QN.overlap(carry, contrast), "prompt_overlap_draws": QN.random_overlaps(carry, contrast, n, gen),
            "engaged": CJ.set_share(eng, carry) if carry else None, "engaged_draws": QN.random_engaged(carry, eng, n, gen)}


def score(lab: Any, sets: dict[str, dict[str, list[int]]], eng: dict[str, list[float]]) -> dict[str, Any]:
    contrast = json.loads(E102.W0F.read_text())["contrast"]["sets"]["all"]
    per = {rows: {n: measures(c, eng[rows], contrast, torch.Generator().manual_seed(CFG.seed + 1 + i))
                  for i, (n, c) in enumerate(sorted(by.items()))} for rows, by in sets.items()}
    rules: dict[str, Any] = {}
    for size, (known, new) in PAIRS.items():
        for rows in sets:
            for k in ("engaged", "prompt_overlap"):
                v = per[rows]
                rules[f"{size}_{rows}_{k}"] = pair_side(v[known][k], [v[x][k] for x in new], v["gauss"][k], v[known][f"{k}_draws"])
    out = {"per_state": per, "rules": rules,
           "set_overlap": {rows: {s: QN.overlap(sets[rows][k], sets[rows][n[0]]) for s, (k, n) in PAIRS.items()} for rows in sets}}
    SCORE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    lab.log(f"[score] {json.dumps({k: v['verdict'] for k, v in rules.items()})}")
    return out


def engaged_real(lab: Any, R: qc.Rows) -> torch.Tensor:
    if not ENGAGED_REAL.is_file():
        fw, w = E102.fineweb(lab), CFG.window_rows
        sample = QN.sorted_sample(lab.model, fw[w : w + CFG.quantile_rows])
        save_file({"real": QN.engaged(lab.model, R, sample, CFG.batch).contiguous()}, str(ENGAGED_REAL))
        del sample
        torch.cuda.empty_cache()
    return load_file(str(ENGAGED_REAL))["real"]


def main() -> None:
    if RUN.STOPPED.is_file():
        raise SystemExit(f"{RUN.STOPPED} exists: read it, then remove it")
    RUN.filed()
    if not RUN.is_done("T"):
        raise SystemExit("run.py's stage T must finish first")
    lab = RUN.Lab("N", "E103 neurons", RUN.PRED)
    lab.fp32().eval()
    kept = {r["row"] for r in map(json.loads, E102.FORMS.read_text().splitlines()) if r["kept"]}  # E84's, as E102's rows
    rows = {"real": real_rows(lab), "tofu": qc.tofu_rows(lab.tok, lab.end_id, kept)}
    lab.log(f"[rows] {json.dumps({k: [list(R.tokens.shape), int(R.answer.sum())] for k, R in rows.items()})}")
    if not (OUT_DIR / "fict_s42_matched__tofu.json").is_file():
        control(lab, rows["tofu"])
    eng = {"real": engaged_real(lab, rows["real"]).tolist(), "tofu": load_file(str(E102.LARGE / "engaged.safetensors"))["tofu"].tolist()}
    todo = {"real_matched": ("real", "tofu"), "real_full": ("real", "tofu"), **dict.fromkeys((*FICT, "gauss"), ("real",))}
    for name, on in todo.items():
        missing = [x for x in on if not (OUT_DIR / f"{name}__{x}.json").is_file()]
        if missing:
            W = weights(lab, name)
            for x in missing:
                read(lab, rows[x], name, W)
            del W
            torch.cuda.empty_cache()
    sets = {x: {f.name.split("__")[0]: json.loads(f.read_text())["carry"] for f in sorted(OUT_DIR.glob(f"*__{x}.json"))} for x in rows}
    for name, e102 in E102_NAME.items():  # E102's own sets on the TOFU rows; the C2 re-read stands for seed 42 matched
        sets["tofu"].setdefault(name, json.loads((E102R / "states" / f"{e102}.json").read_text())["carry"])
    score(lab, sets, eng)
    lab.log(f"[neurons] done, {lab.peaks()}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()
