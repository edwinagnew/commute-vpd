"""Cell P2.90: Qwen3-1.7B random references for the one re-gating share, built as the 67M paper's are (P2.82, E310,
experiments/random_logodds_share/run.py): a random change scaled until it moves the trained answers by one log-odds,
then the share read on it. E328's KL-matched run-1 randoms moved the answers by -5 to -20 log-odds (+1.2 on gold), so
their shares are not comparable with the fine-tunes'; this fixes that.

Per decomposition (VPD p-9707d6ff first, then the neuron control p-015e0710, attention δW zeroed as P3.9), per side
(copy, cipher, known, invented, gold, gold_shuf), draw d = 0, 1, 2:
  direction  `share_reading.gaussian` at the per-matrix norms of the side's seed-d last fine-tune (fetched from W&B,
             sha256-checked, deleted once its norms are taken), draw seed E310's seeds[d] (E228's)
  sign       s = sign(gain(+probe_x d) - gain(-probe_x d)), the odd part of the gain (a tie takes +1)
  scale      `perturb.match` finds x with gain(s x d) in the band target_gain x (1 +- gain_rel_tol); the other sign is
             tried when the first does not reach it (E310). A draw that never reaches the band is written with its
             largest gain and share nan.
  share      `share.share` of the answer cells on all 128 rows (`share_reading.answer_cells`, the Lab's four forwards),
             floor gain >= 1 (`local_faith.MIN_GAIN`, one-sided, as E310); KL(W0 + change || W0) per position over the
             same rows, from the same forwards, reported (not matched).
gain = mean over whole answers of T1 - T0 in the whole-answer log-odds (`share.answer_logodds`). During the search each
gain takes the T cells only (`share_reading.answer_target`: one plain forward per row, T0 cached once per row set), on
the first `search_rows` rows; the written gain is the final full read's.

  control  first, per decomposition and row set: a zero change reads gain and numerator exactly 0 (E310's control a),
           on the search's T path and on the full read; a miss stops the job (results/stopped.json, exit 1).
  draws    this shard's (side, draw) pairs: every N-th of the decomposition's 18 from the I-th (--shard I N); one
           JSON line per draw in results/qwen_random_<tag>.jsonl as it lands (sign, scale, evals, gain, numerator,
           share, KL, size, seconds); a rerun skips the draws written.
Prints "QWEN_RANDOM DONE" at the end. Settings: results/predictions_random.json "settings.qwen_random".

  Four pods, one decomposition and half its draws each; every pod runs its decomposition's zero control first:
    PRED_SHA=<sha> python experiments/share_unified/qwen_random.py --decomp "$COMMUTE_QWEN_NEW_DIR" --shard 0 2 --tag vpd0
    PRED_SHA=<sha> python experiments/share_unified/qwen_random.py --decomp "$COMMUTE_QWEN_NEW_DIR" --shard 1 2 --tag vpd1
    PRED_SHA=<sha> python experiments/share_unified/qwen_random.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" --shard 0 2 --tag neuron0
    PRED_SHA=<sha> python experiments/share_unified/qwen_random.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" --shard 1 2 --tag neuron1
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/share_unified/qwen_random.py --dry   (CPU: rows, spans, draws)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import library_states as LS
from commute import perturb as PT
from commute import share as SH
from commute import share_reading as SR
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.local_faith import MIN_GAIN
from commute.qwen_sft import linears, load
from commute.qwen_vpd import SITES, decomp_id, decomp_kinds, parse_site

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions_random.json"
OUT = RESULTS / "qwen_random_all.jsonl"  # main sets results/qwen_random_<tag>.jsonl
E310_PRED = EXPERIMENTS / "random_logodds_share" / "results" / "predictions.json"
RR = load_experiment("share_unified", "qwen_reread")  # rows_and_spans: the row sets as E328 read them
QS, L = RR.QS, RR.L  # P3.8/P3.9's run.py; E57's Lab module
DW = dict[str, Tensor]


@dataclass(frozen=True)
class Config:
    arms: tuple[str, ...]
    save: str  # the fine-tunes whose norms the draws take
    seeds: tuple[int, ...]  # draw d's Gaussian seed (E310's, E228's)
    probe_x: float
    scale_cap: float
    target_gain: float
    gain_rel_tol: float
    search_rows: int  # rows of the search's T-only gains; the share always reads all rows
    n16: int  # rows of the zero control

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        assert set(self.arms) <= set(QS.ARMS) and len(set(self.seeds)) == len(self.seeds) >= 1
        assert self.target_gain * (1 - self.gain_rel_tol) >= MIN_GAIN, "a matched gain must pass the form's floor"
        assert 0 < self.probe_x < self.scale_cap and 0 < self.n16 <= self.search_rows <= L.CFG.n_rows


def done() -> set[str]:
    return {json.loads(ln)["item"] for ln in OUT.read_text().splitlines() if ln} if OUT.is_file() else set()


def put(lab: Any, rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    lab.log(f"[item {rec['item']}] {json.dumps(rec, default=float)}")


def size(d: DW) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))


class Search:
    """Gains of c·R on one row set's first search rows, T cells only, each c read once (E310's `Scaled`)."""

    def __init__(self, lab: Any, R: DW, rows: Tensor, spans: SH.Spans, t0: Tensor) -> None:
        self.lab, self.R, self.rows, self.spans, self.t0 = lab, R, rows, spans, t0
        self.seen: dict[float, float] = {}

    def gain(self, c: float) -> float:
        if c not in self.seen:
            self.lab.set_weights("FT", self.R, c)
            t1 = SR.answer_target(self.lab, "FT", self.rows, self.spans)
            self.lab.drop_set("FT")
            self.seen[c] = float((t1 - self.t0).mean())
            self.lab.log(f"  [search c {c:+.6g}] gain {self.seen[c]:.6g}")
        return self.seen[c]


def full_read(lab: Any, R: DW, c: float, rows: Tensor, spans: SH.Spans) -> dict[str, Any]:
    """The share of W0 + c·R on all rows, floor MIN_GAIN one-sided, and KL(W0 + c·R || W0) per position."""
    kl = [0.0]

    def kl_row(f: dict[str, Tensor]) -> None:
        kl[0] += L.kl(f["T1"], f["T0"])

    lab.set_weights("FT", R, c)
    cells = SR.answer_cells(lab, rows, spans, kl_row)
    lab.drop_set("FT")
    gain = SH.cells_gain(cells, "ans")
    return {"gain": gain, "numerator": SH.numerator(cells, "ans"), "kl": kl[0] / rows.numel(),
            "share_any": SH.share(cells, "ans", gain, SH.Floor(MIN_GAIN, two_sided=False))}


def one_draw(lab: Any, R: DW, rows: Tensor, spans: SH.Spans, t0: Tensor, c: Config) -> dict[str, Any]:
    start = time.time()
    k = c.search_rows
    sc = Search(lab, R, rows[:k], spans.take(0, k), t0)
    plus, minus = sc.gain(c.probe_x), sc.gain(-c.probe_x)
    first = -1.0 if plus - minus < 0 else 1.0
    tries, matched = {}, None
    for sign in (first, -first):
        m = PT.match(lambda x, s=sign: sc.gain(s * x), c.target_gain, c.scale_cap, c.gain_rel_tol)
        tries["+" if sign > 0 else "-"] = {"status": m["status"], "x": m["x"], "gain": m["f"], "evals": len(m["grid"]) + len(m["trace"])}
        if m["status"] == "ok":
            matched = sign * m["x"]
            break
    cs = matched if matched is not None else max(sc.seen, key=lambda x: sc.seen[x])
    status = "ok" if matched is not None else tries["+" if first > 0 else "-"]["status"]
    search_s = time.time() - start
    rd = full_read(lab, R, cs, rows, spans)
    return {"status": status, "signed_c": cs, "probe": {"x": c.probe_x, "plus": plus, "minus": minus, "tie": plus == minus},
            "probe_sign": first, "tries": tries, "search_gain": sc.seen[cs], "evals": len(sc.seen), "size": abs(cs) * size(R),
            "gain": rd["gain"], "numerator": rd["numerator"], "kl": rd["kl"],
            "share": rd["share_any"] if status == "ok" else math.nan, "max_search_gain": max(sc.seen.values()),
            "search_seconds": search_s, "seconds": time.time() - start}


def control(lab: Any, dec: str, data: dict[str, tuple[Tensor, SH.Spans]], t0: dict[str, Tensor], c: Config,
            shapes: dict[str, tuple[int, ...]]) -> None:
    zero = SR.with_zeros({}, shapes)  # one shared zero tensor per shape
    for rs, (rows, spans) in data.items():
        name = f"{dec}/zero/{rs}"
        if name in done():
            continue
        k = RR.w0_rows(spans, c.n16, len(rows))  # the first rows holding whole answers on n16 rows (as E328's W0 control)
        g = Search(lab, zero, rows[: c.search_rows], spans.take(0, c.search_rows), t0[rs]).gain(1.0)
        rd = full_read(lab, zero, 1.0, rows[:k], spans.take(0, k))
        put(lab, {"item": name, "decomp": dec, "kind": "zero", "rowset": rs, "search_gain": g, **rd})
        lab.check(f"zero_{dec}_{rs}", g == 0 and rd["gain"] == 0 and rd["numerator"] == 0,
                  {"search_gain": g, "gain": rd["gain"], "numerator": rd["numerator"]})


def draws(lab: Any, dec: str, data: dict[str, tuple[Tensor, SH.Spans]], t0: dict[str, Tensor], c: Config,
          shapes: dict[str, tuple[int, ...]], fetch_to: Path, shard: tuple[int, int]) -> None:
    name = lambda st: f"{dec}/{st.arm}_d{st.seed}"  # draw d takes seed d's fine-tune
    mine = pairs(c)[shard[0] :: shard[1]]
    todo = [st for a, d in mine if name(st := LS.state(QS.RUN_OF[a], a, d, c.save)) not in done()]
    for st, path in QS.streamed(todo, fetch_to, lab.log):
        dW = QS.own_sites(load(path))
        norms = {s: float(m.double().norm()) for s, m in dW.items()}
        del dW
        R = SR.gaussian(norms, shapes, c.seeds[st.seed])
        rs = QS.ROWSET[st.arm]
        rows, spans = data[rs]
        rec = one_draw(lab, R, rows, spans, t0[rs], c)
        put(lab, {"item": name(st), "decomp": dec, "kind": "random", "arm": st.arm, "rowset": rs, "draw": st.seed,
                  "draw_seed": c.seeds[st.seed], "norms_from": st.key, "norms_sha256": st.sha256, "fine_tune_size": size_of(norms),
                  "answers": len(spans.row), **rec})
        del R


def pairs(c: Config) -> list[tuple[str, int]]:
    """Every (side, draw) of one decomposition, sides in settings order; a shard takes every N-th from the I-th."""
    return [(a, d) for a in c.arms for d in range(len(c.seeds))]


def size_of(norms: dict[str, float]) -> float:
    return math.sqrt(sum(n * n for n in norms.values()))


def one_decomp(directory: str, c: Config, data: dict[str, tuple[Tensor, SH.Spans]], fetch_to: Path,
               shard: tuple[int, int]) -> None:
    dec = decomp_id(Path(directory))
    QS.point(dec)
    L.RESULTS, L.PRED = RESULTS, PRED
    QS.OWN = frozenset(s for s in SITES if parse_site(s)[1] in decomp_kinds(Path(directory)))
    lab = L.Lab(f"P2.90 random {dec}", "unused", directory)
    with torch.no_grad():
        shapes = {s: tuple(lin.weight.shape) for s, lin in linears(lab.model).items()}
        k = c.search_rows
        assert all(len(sp.take(0, k).row) for _, sp in data.values()), "a row set has no whole answer in the search rows"
        t0 = {rs: SR.answer_target(lab, "W0", rows[:k], spans.take(0, k)) for rs, (rows, spans) in data.items()}
        control(lab, dec, data, t0, c, shapes)  # every shard, first
        draws(lab, dec, data, t0, c, shapes, fetch_to, shard)
    lab.log(f"[{dec} done] {lab.peaks()}")
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", action="append", default=[], help="a decomposition directory; give both, VPD first")
    ap.add_argument("--fetch-to", default=str(RESULTS / "large" / "states"), help="where states land, two at a time")
    ap.add_argument("--dry", action="store_true", help="CPU: rows, answer spans and the draws, then stop")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"),
                    help="every N-th (side, draw) of each decomposition from the I-th (0-based)")
    ap.add_argument("--tag", default="all", help="output results/qwen_random_<tag>.jsonl (one per pod)")
    a = ap.parse_args()
    global OUT
    OUT = RESULTS / f"qwen_random_{a.tag}.jsonl"
    shard = (a.shard[0], a.shard[1])
    assert 0 <= shard[0] < shard[1], shard
    c = config_from_settings(Config, json.loads(PRED.read_text())["settings"]["qwen_random"])
    assert list(c.seeds) == json.loads(E310_PRED.read_text())["settings"]["seeds"], "draw seeds must be E310's"
    data = RR.rows_and_spans(c.arms)
    mine = pairs(c)[shard[0] :: shard[1]]
    print(f"shard {shard}, tag {a.tag}: {len(mine)} of {len(pairs(c))} draws per decomposition {mine}; "
          f"{len(done())} items in {OUT.name}; rows {[(rs, len(sp.row)) for rs, (_, sp) in data.items()]}", flush=True)
    if a.dry:
        hard_exit(0)
    require_filed(PRED)
    assert a.decomp, "--decomp, once per decomposition"
    for d in a.decomp:
        one_decomp(d, c, data, Path(a.fetch_to), shard)
    print("QWEN_RANDOM DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
