"""E169 (qwen_library_readings), cell P3.7: the cheap readings of every state of the Qwen3-1.7B pair library (E164, E165),
side A minus side B, at the late save and the matched (5x) save. No decomposition is read, nothing is trained, no update
is labelled reuse or new, and no score joins the readings. Card card.md; plan plan.md; rules results/predictions.json
(the run refuses unless it is committed unchanged), applied by rules.py; the scorecard is plot.py (CPU, after the pods).

    E120_PRED_SHA=<sha> .venv/bin/python experiments/qwen_library_readings/run.py --pairs copy_cipher

Readings per state, each the function E92, E162 and E166 used: update size (`qwen_comparators.update_size`, total
ratio_norm), prior (W0's median rank of each answer's first token on the side's rows), steering (E92's `steer`, share of
the gain on the scoring half), effective rank of the update (`update_spectrum`, fp64, on the card) and top-neuron overlap
(`qwen_overlap.overlap`, K = 480 of 6144). Rows: each side's own trained sequences packed as the library packs them
(`qwen_pairs.own_rows` for built sides; a twin's training sequences, the shuffled side's by that seed's derangement).
Per pair, in results/<pair>/:
  stage 0  alone, first: the do-nothing update (zeros) on side A's rows: overlap exactly 1.0 at every block, on_both nan,
           size 0, max |log P - W0's| <= zero_logp_max, no steering layer, erank None. A miss stops the job (exit 1).
  stage 1  the random floor, before any real state: random changes at the per-matrix norms of the library's smallest- and
           largest-norm states (fetched once per job), read on both sides' rows (twins: seed 0's); the floor band per
           reading (rules.floor_band). An overlap or erank band unreadable, or an overlap band over overlap_band_max: exit 1.
  stage 2  every state, streamed: fetch from W&B (the next one while this one is read), sha256 against the library's
           record and |δW| against its recorded norm (else exit 1), the five readings, delete the file. Resumable: a
           state already in stage2.json is skipped.
Each stage writes results/<pair>/stage<n>.json and prints "STAGE n DONE <pair>"; every number is printed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from commute import qwen_comparators as qc
from commute import qwen_d3 as d3
from commute import qwen_library as L
from commute import qwen_overlap as QO
from commute import qwen_pairs as P
from commute import update_spectrum as US
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_files import fetch
from commute.qwen_sft import linears, load, plus, sha256
from commute.qwen_vpd import HF_NAME, HF_REVISION, fp32_matmuls, load_target

R = load_experiment("qwen_library_readings", "rules")
LIB = load_experiment("qwen_pair_library", "library")  # twin(): a twin's sequences as E164 trained them
E92 = load_experiment("qwen_comparators", "run")  # STEER_SEED
RESULTS = Path(__file__).resolve().parent / "results"
PRED = RESULTS / "predictions.json"
LIBRARY = EXPERIMENTS / "qwen_pair_library" / "results"


@dataclass(frozen=True)
class Config:
    overlap_draws: int = 4  # E166's four draws per reference
    steer_draws: int = 2  # steering costs ~35 forwards a draw: two, the fewest that give a range
    erank_draws: int = 2  # E162's two Gaussian draws per matrix
    first_seed: int = 169000
    zero_logp_max: float = 1e-5  # E92's and E118's do-nothing bar
    norm_rel_tol: float = 1e-6  # |δW| against the library's fp64 record of the same file
    overlap_band_max: float = 0.2  # stop: 2.5x E166's whole floor range (0.998 at norm 0.27 to 0.92 at norm 18.1)

    def __post_init__(self) -> None:
        assert min(self.overlap_draws, self.steer_draws, self.erank_draws) >= 2, "a range needs two draws"
        assert self.steer_draws <= self.overlap_draws, "steering draws reuse the first overlap draws' changes"


CFG = Config()


class Stopped(Exception):
    """A filed stop: the stage's file records why."""


def log(msg: str) -> None:
    print(msg, flush=True)


def write(path: Path, out: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)


def stored(path: Path) -> dict[str, Any] | None:
    out = json.loads(path.read_text()) if path.is_file() else None
    return None if out is None or "stopped" in out else out


def check(path: Path, out: dict[str, Any], name: str, holds: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"holds": bool(holds), **detail}
    log(f"[check {path.parent.name}/{path.stem}] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail)}")
    if not holds:
        out["stopped"] = name
        write(path, out)
        raise Stopped(name)


# ----------------------------------------------------------------------------- the library's states


def states(pair: str) -> list[dict[str, Any]]:
    """Every state this run reads of one pair, from the library's per-arm JSONs: artifact, sha256, step, |δW|."""
    p, out = R.PAIRS[pair], []
    for side in (p.a, p.b):
        for seed in R.SEEDS:
            arm = json.loads((LIBRARY / p.run / f"{side}_s{seed}.json").read_text())
            for save in p.saves:
                rec = next(s for s in arm["saves"] if s["labels"] == [save])
                out.append({"key": f"{side}_s{seed}_{save}", "side": side, "seed": seed, "save": save, "step": rec["step"],
                            "artifact": rec["artifact"], "sha256": rec["sha256"], "prefix": p.prefix,
                            "dw_norm": arm["states"][save]["dw_norm"]})
    return out


def references() -> tuple[dict[str, Any], dict[str, Any]]:
    """The smallest- and the largest-norm state of the whole set (every pair, both saves)."""
    every = sorted((s for p in R.PAIRS for s in states(p)), key=lambda s: s["dw_norm"])
    return every[0], every[-1]


def fetch_state(st: dict[str, Any], root: Path) -> Path:
    """The state's file from W&B, checked against the library's sha256 (a mismatch raises, so the job stops)."""
    dest, name = root / st["key"], f"{st['save']}.safetensors"
    t0 = time.time()
    fetch(dest, log, artifact=st["artifact"], kind=st["prefix"], ok=lambda d: (d / name).is_file())
    got = sha256(dest / name)
    log(f"[fetch {st['key']}] {time.time() - t0:.0f} s; sha256 {'matches' if got == st['sha256'] else 'DIFFERS'}")
    if got != st["sha256"]:
        raise Stopped(f"sha256 {st['key']}: {got} against the library's {st['sha256']}")
    return dest / name


# ----------------------------------------------------------------------------- the model, the rows, the readings


class Lab:
    """The fp32 model at W0, each side's rows and W0's E92 read of them, built once per job."""

    def __init__(self) -> None:
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(HF_NAME, revision=HF_REVISION)
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.model = load_target("cuda", torch.float32)
        self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        self._rows: dict[str, qc.Rows] = {}
        self._w0: dict[str, qc.Read] = {}
        self._seqs: dict[str, list[Any]] = {}

    def train_seqs(self, family: str) -> list[Any]:
        if family not in self._seqs:
            self._seqs[family] = d3.gold_train(self.tok) if family == "gold" else L.template_pool(log)[0]
        return self._seqs[family]

    def rows(self, side: str, seed: int) -> qc.Rows:
        """A built side's own rows (seed-free); a twin side's training sequences packed, the shuffled side's per seed."""
        key = rows_key(side, seed)
        if key not in self._rows:
            if side in P.SIDES:
                rows = P.own_rows(self.tok, self.end_id, side, P.tasks(LIBRARY / "words.json")[side]["train"]).rows
            else:
                seqs = self.train_seqs(side.removesuffix("_shuf"))
                rows = P.packed(key, LIB.twin(seqs, seed) if side.endswith("_shuf") else seqs)
            self._rows[key], self._w0[key] = rows, qc.read(self.model, rows)
            log(f"[rows {key}] {list(rows.tokens.shape)}, {int(rows.answer.sum())} answer positions, "
                f"{int(rows.first.sum())} answers; W0 {json.dumps(qc.summary(self._w0[key]))}")
        return self._rows[key]

    def w0(self, side: str, seed: int) -> qc.Read:
        self.rows(side, seed)
        return self._w0[rows_key(side, seed)]

    def e92(self, dW: dict[str, torch.Tensor], side: str, seed: int) -> dict[str, Any]:
        """E92's read_state (size, prior, gain, steering) of one change on one side's rows, the model left at W0."""
        w0 = self.w0(side, seed)
        rec, st = qc.read_state(self.model, self.W0, dW, self.rows(side, seed), w0, E92.STEER_SEED)
        sv = rec["steering"]
        return {"size": rec["size"]["total"]["ratio_norm"], "prior_rank": rec["rank_W0"]["first"]["median_rank"],
                "steer": sv["steer"]["all"]["share"] if sv["steer"] else None, "gain": rec["gain"]["all"]["gain"],
                "layer": sv["layer"], "random_dir": sv["random"]["all"]["share"] if sv["random"] else None,
                "max_logp_diff": float((st.logp - w0.logp).abs().max()), "e92": rec}

    def readings(self, dW: dict[str, torch.Tensor], side: str, seed: int) -> dict[str, Any]:
        """The five readings of one change on one side's rows."""
        out = self.e92(dW, side, seed)
        ov = QO.overlap(self.model, self.rows(side, seed), plus(self.W0, dW))
        er = US.aggregate({s: US.matrix(x) for s, x in sorted(dW.items())})
        torch.cuda.empty_cache()
        return out | {"erank": er["erank"], "overlap": ov["overlap"], "on_both": ov["on_both"], "overlap_blocks": ov["per_block"]}


def rows_key(side: str, seed: int) -> str:
    return f"{side}_s{seed}" if side.endswith("_shuf") else side


# ----------------------------------------------------------------------------- stage 0, the do-nothing update alone


def stage0(lab: Lab, pair: str, path: Path) -> dict[str, Any]:
    side, out = R.PAIRS[pair].a, {}
    zero = {s: torch.zeros_like(w) for s, w in lab.W0.items()}
    r = lab.readings(zero, side, 0)
    exact = r["overlap"] == 1.0 and math.isnan(r["on_both"]) and all(b["overlap_min"] == 1.0 for b in r["overlap_blocks"].values())
    detail = {k: r[k] for k in ("overlap", "on_both", "size", "max_logp_diff", "layer", "erank")} | {"rows": side}
    check(path, out, "C0_do_nothing", exact and r["size"] == 0 and r["max_logp_diff"] <= CFG.zero_logp_max
          and r["layer"] is None and r["erank"] is None, detail)
    out["w0"] = {k: qc.summary(lab.w0(s, 0)) for k, s in (("a", R.PAIRS[pair].a), ("b", R.PAIRS[pair].b))}
    return out


# ----------------------------------------------------------------------------- stage 1, the random floor


def ref_norms(ref: dict[str, Any], root: Path) -> dict[str, float]:
    path = fetch_state(ref, root)
    norms = QO.site_norms(path)
    total = math.sqrt(sum(v * v for v in norms.values()))
    log(f"[ref {ref['key']}] |δW| {total:.6f}, library {ref['dw_norm']:.6f}")
    path.unlink()
    return norms


def gaussian(shapes: dict[str, tuple[int, ...]]) -> list[dict[str, dict[str, float]]]:
    """Per draw, each matrix's erank of one iid Gaussian at its shape (scale-free), fp64 on the card (E162's floor)."""
    out = []
    for i in range(CFG.erank_draws):
        gen = torch.Generator("cuda").manual_seed(CFG.first_seed + i)
        out.append({s: US.matrix(torch.randn(*sh, generator=gen, device="cuda", dtype=torch.float64)) for s, sh in sorted(shapes.items())})
    return out


def side_floor(lab: Lab, side: str, norms: dict[str, float]) -> dict[str, list[float | None]]:
    got: dict[str, list[float | None]] = {"overlap": [], "steer": []}
    for i in range(CFG.overlap_draws):
        t0, W = time.time(), QO.random_weights(lab.W0, norms, CFG.first_seed + i)
        got["overlap"].append(QO.overlap(lab.model, lab.rows(side, 0), W)["overlap"])
        if i < CFG.steer_draws:
            got["steer"].append(lab.e92({s: W[s] - lab.W0[s] for s in W}, side, 0)["steer"])
        del W
        torch.cuda.empty_cache()
        steer = got["steer"][i] if i < CFG.steer_draws else "-"
        log(f"[floor {side} draw {i}] overlap {got['overlap'][-1]:.6f} steer {steer}; {time.time() - t0:.0f} s")
    return got


def stage1(lab: Lab, pair: str, refs: dict[str, dict[str, float]], G: list[Any], path: Path) -> dict[str, Any]:
    p, out = R.PAIRS[pair], {"refs": {}}
    for name, norms in refs.items():
        er = [US.aggregate({s: g[s] | {"norm2": norms[s] ** 2} for s in g})["erank"] for g in G]
        out["refs"][name] = {"erank": er, **{side: side_floor(lab, side, norms) for side in (p.a, p.b)}}
        write(path, out)
    F = out["refs"]
    out["band"] = {"erank": R.floor_band({n: F[n]["erank"] for n in F}, {n: F[n]["erank"] for n in F}),
                   **{k: R.floor_band({n: F[n][p.a][k] for n in F}, {n: F[n][p.b][k] for n in F}) for k in ("overlap", "steer")}}
    log(f"[floor {pair}] band {json.dumps(out['band'])}")
    b = out["band"]
    check(path, out, "C1_band", b["erank"] is not None and b["overlap"] is not None and b["overlap"] <= CFG.overlap_band_max, b)
    return out


# ----------------------------------------------------------------------------- stage 2, every state, streamed


def read_one(lab: Lab, st: dict[str, Any], file: Path, path: Path, out: dict[str, Any]) -> dict[str, Any]:
    t0, dW = time.time(), load(file, "cuda")
    norm = math.sqrt(sum(float(x.double().pow(2).sum()) for x in dW.values()))
    check(path, out, f"norm {st['key']}", abs(norm / st["dw_norm"] - 1) <= CFG.norm_rel_tol, {"norm": norm, "library": st["dw_norm"]})
    r = lab.readings(dW, st["side"], st["seed"])
    del dW
    file.unlink()
    rec = {**{k: st[k] for k in ("side", "seed", "save", "step", "artifact")}, **r, "seconds": time.time() - t0}
    log(f"[state {st['key']}] {json.dumps({k: v for k, v in rec.items() if k not in ('e92', 'overlap_blocks')})}")
    return rec


def stage2(lab: Lab, pair: str, root: Path, path: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    out.pop("stopped", None)
    todo = [s for s in states(pair) if s["key"] not in out["states"]]
    with ThreadPoolExecutor(1) as pool:
        nxt: Future[Path] | None = pool.submit(fetch_state, todo[0], root) if todo else None
        for i, st in enumerate(todo):
            assert nxt is not None
            file = nxt.result()
            nxt = pool.submit(fetch_state, todo[i + 1], root) if i + 1 < len(todo) else None
            out["states"][st["key"]] = read_one(lab, st, file, path, out)
            write(path, out)
            log(f"[time {pair}] {i + 1} of {len(todo)} states this job")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E169: the cheap readings of the pair library's states")
    ap.add_argument("--pairs", required=True, help=f"comma-separated, of {','.join(R.PAIRS)}")
    ap.add_argument("--states", type=Path, default=Path("/root/states"), help="scratch folder for one or two fetched states")
    a = ap.parse_args()
    pairs = a.pairs.split(",")
    assert pairs and set(pairs) <= set(R.PAIRS), pairs
    require_filed(PRED)
    filed = json.loads(PRED.read_text())["settings"]
    assert {k: filed.get(k) for k in asdict(CFG)} == asdict(CFG), (filed, asdict(CFG))
    lab = Lab()
    try:
        for pair in pairs:
            if stored(RESULTS / pair / "stage0.json") is None:
                write(RESULTS / pair / "stage0.json", stage0(lab, pair, RESULTS / pair / "stage0.json"))
            log(f"STAGE 0 DONE {pair}")
        need = [p for p in pairs if "band" not in (stored(RESULTS / p / "stage1.json") or {})]
        if need:
            refs = {r["key"]: ref_norms(r, a.states) for r in references()}
            G = gaussian({s: tuple(w.shape) for s, w in lab.W0.items()})
            for pair in need:
                write(RESULTS / pair / "stage1.json", stage1(lab, pair, refs, G, RESULTS / pair / "stage1.json"))
        for pair in pairs:
            log(f"STAGE 1 DONE {pair}; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        for pair in pairs:
            stage2(lab, pair, a.states, RESULTS / pair / "stage2.json")
            log(f"STAGE 2 DONE {pair}; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    except Stopped as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
