"""Cell P2.90: App H's Qwen3-1.7B re-gating share (figure qwen_share) re-read with the one share for both models,
`commute.share` in the whole-answer log-odds: r = mean over answers of (L01 - L00) / mean over answers of (T1 - T0), an
answer being one whole trained response (its text and end token, the tokens the library's training loss was taken
over) inside one of the side's own 128 rows of 256 tokens (`share_reading.answer_spans`; a response cut by a row's
edge is left out).

The same states, rows and random references as the figure (`P/figures/src/make_qwen_share.py`): per decomposition
(VPD p-9707d6ff, all 196 matrices; the neuron control p-015e0710, its 84 MLP matrices, attention δW set to zero as
P3.9 did), the six sides' seeds 0-2 at the last save (P3.8/P3.9 run 2, fetched from W&B, sha256-checked, deleted once
read), and each side's run-1 random change (draw 0 at its seed-0 last state's per-matrix norms, times the KL-matched
scale run 1 recorded: `<side>_s0_last_kl`). The forwards are the Lab's (`Lab.passes`, experiments/qwen_reading/lab.py),
as the KL reading made them; the KL cells K0, K00, K01 are taken from the same logits and written beside the saved ones,
which checks that the states and rows are the old ones.

  W0 control  first, per decomposition and row set, on 16 rows: W0 as the changed model; |numerator| must be at most
              zero_rel_tol x mean |L00|, or the job stops (results/stopped.json, exit 1).
  items       the random references, then the fine-tunes; one JSON line per item in results/qwen_reread.jsonl as it
              lands (new share, numerator, gain, cell means, answers, the old KL share and K means, seconds); a rerun
              skips the items written.
Prints "QWEN_REREAD DONE" at the end. One H100: VPD read 128 rows of these four forwards in 69 s (P3.8 run 2 log).

    python experiments/share_unified/qwen_reread.py --decomp "$COMMUTE_QWEN_NEW_DIR" --decomp "$COMMUTE_QWEN_FROZEN_DIR"
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/share_unified/qwen_reread.py --dry   (CPU: rows, spans, items)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import library_states as LS
from commute import share as SH
from commute import share_reading as SR
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.qwen_sft import linears, load
from commute.qwen_vpd import SITES, decomp_id, decomp_kinds, parse_site

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "qwen_reread.jsonl"
QS = load_experiment("qwen_library_share", "run")  # P3.8/P3.9: point, rows_of, own_sites, streamed, RUN_OF, ROWSET
L = QS.L  # E57's Lab module (experiments/qwen_reading/lab.py)


@dataclass(frozen=True)
class Config:
    arms: tuple[str, ...]  # the figure's six sides
    seeds: tuple[int, ...]
    save: str  # the figure's save
    draw: int  # run 1's Gaussian draw
    n16: int  # rows of the W0 control
    zero_rel_tol: float  # |numerator| / mean |L00| at W0 (E57's id_tol)
    gain_floor: float  # two-sided, as the 67M's E136 reads

    def __post_init__(self) -> None:
        assert set(self.arms) <= set(QS.ARMS) and self.n16 <= L.CFG.n_rows and self.gain_floor > 0
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seeds", tuple(self.seeds))


def rows_and_spans(arms: tuple[str, ...]) -> dict[str, tuple[Tensor, SH.Spans]]:
    """Each row set's tokens (asserted equal to the rows P3.8 read) and its whole answers."""
    end, out = SR.end_id(), {}
    for arm in arms:
        rs = QS.ROWSET[arm]
        if rs not in out:
            r = SR.marked_rows(rs)
            assert torch.equal(r.tokens, QS.rows_of(arm)), f"{rs}: rows differ from P3.8's"
            out[rs] = (r.tokens, SR.answer_spans(r, end))
    return out


def done() -> set[str]:
    return {json.loads(ln)["item"] for ln in OUT.read_text().splitlines() if ln} if OUT.is_file() else set()


def put(lab: Any, rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    lab.log(f"[item {rec['item']}] {json.dumps(rec, default=float)}")


def read(lab: Any, dW: dict[str, Tensor] | None, rows: Tensor, spans: SH.Spans, c: Config) -> dict[str, Any]:
    """The share of W0 + dW (None: W0 as the changed model) on rows, and the KL cells' means from the same logits."""
    t0, K = time.time(), {"K0": 0.0, "K00": 0.0, "K01": 0.0}

    def kl_cells(f: dict[str, Tensor]) -> None:
        K["K0"] += L.kl(f["T0"], f["L00"])
        K["K00"] += L.kl(f["T1"], f["L00"])
        K["K01"] += L.kl(f["T1"], f["L01"])

    lab.set_weights("FT", dW)
    cells = SR.answer_cells(lab, rows, spans, kl_cells)
    lab.drop_set("FT")
    gain = SH.cells_gain(cells, "ans")
    s = time.time() - t0
    return {"share": SH.share(cells, "ans", gain, SH.Floor(c.gain_floor, two_sided=True)), "numerator": SH.numerator(cells, "ans"),
            "gain": gain, "means": {k: float(v["ans"].mean()) for k, v in cells.items()}, "answers": len(spans.row),
            "rows": len(rows), "K_means": {k: v / rows.numel() for k, v in K.items()}, "seconds": s, "s_per_row": s / len(rows)}


def old(path: Path) -> dict[str, Any]:
    rec = json.loads(path.read_text())
    return {"old_kl_share": rec["regating"], "old_status": rec["status"], "old_K_means": {k: rec["means"][k] for k in ("K0", "K00", "K01")}}


def w0_rows(spans: SH.Spans, n16: int, n_rows: int) -> int:
    """The first rows that hold whole answers on n16 rows (all rows if fewer do): the W0 control's rows."""
    with_answers = sorted(set(spans.row.tolist()))
    return with_answers[n16 - 1] + 1 if len(with_answers) >= n16 else n_rows


def w0_controls(lab: Any, dec: str, data: dict[str, tuple[Tensor, SH.Spans]], c: Config) -> None:
    for rs, (rows, spans) in data.items():
        name = f"{dec}/W0/{rs}"
        if name in done():
            continue
        k = w0_rows(spans, c.n16, len(rows))
        rec = read(lab, None, rows[:k], spans.take(0, k), c)
        bar = c.zero_rel_tol * abs(rec["means"]["L00"])
        put(lab, {"item": name, "decomp": dec, "kind": "W0", "rowset": rs, "bar": bar} | rec)
        lab.check(f"W0_{dec}_{rs}", abs(rec["numerator"]) <= bar, {"numerator": rec["numerator"], "bar": bar})


def randoms(lab: Any, dec: str, data: dict[str, tuple[Tensor, SH.Spans]], c: Config) -> None:
    shapes = {s: tuple(lin.weight.shape) for s, lin in linears(lab.model).items()}
    for arm in c.arms:
        key = f"{arm}_s{c.seeds[0]}_{c.save}_kl"
        name, src = f"{dec}/run1/{key}", QS.RESULTS_OF[dec] / "run1" / f"{key}.json"
        if name in done():
            continue
        r1 = json.loads(src.read_text())["random"]
        assert r1["draw_seed"] == c.draw and set(r1["norms"]) == set(SITES), src
        R = SR.gaussian(r1["norms"], shapes, c.draw)
        for m in R.values():
            m.mul_(r1["scale"])
        rows, spans = data[QS.ROWSET[arm]]
        put(lab, {"item": name, "decomp": dec, "kind": "random", "arm": arm, "rowset": QS.ROWSET[arm], "scale": r1["scale"],
                  "source": str(src.relative_to(HERE.parents[1]))} | read(lab, R, rows, spans, c) | old(src))
        del R


def finetunes(lab: Any, dec: str, data: dict[str, tuple[Tensor, SH.Spans]], c: Config, fetch_to: Path) -> None:
    todo = [st for a in c.arms for s in c.seeds if f"{dec}/run2/{(st := LS.state(QS.RUN_OF[a], a, s, c.save)).key}" not in done()]
    for st, path in QS.streamed(todo, fetch_to, lab.log):
        src = QS.RESULTS_OF[dec] / "run2" / f"{st.key}.json"
        dW = QS.own_sites(load(path))
        rows, spans = data[QS.ROWSET[st.arm]]
        put(lab, {"item": f"{dec}/run2/{st.key}", "decomp": dec, "kind": "fine-tune", "arm": st.arm, "seed": st.seed,
                  "save": st.save, "step": st.step, "rowset": QS.ROWSET[st.arm], "sha256": st.sha256,
                  "source": str(src.relative_to(HERE.parents[1]))} | read(lab, dW, rows, spans, c) | old(src))
        del dW


def one_decomp(directory: str, c: Config, data: dict[str, tuple[Tensor, SH.Spans]], fetch_to: Path) -> None:
    dec = decomp_id(Path(directory))
    QS.point(dec)  # P3.8/P3.9's folders for the old records; the Lab then writes to this experiment's results/
    L.RESULTS, L.PRED = RESULTS, PRED
    QS.OWN = frozenset(s for s in SITES if parse_site(s)[1] in decomp_kinds(Path(directory)))
    lab = L.Lab(f"P2.90 {dec}", "unused", directory)
    with torch.no_grad():
        w0_controls(lab, dec, data, c)
        randoms(lab, dec, data, c)
        finetunes(lab, dec, data, c, fetch_to)
    lab.log(f"[{dec} done] {lab.peaks()}")
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", action="append", default=[], help="a decomposition directory; give both, VPD first")
    ap.add_argument("--fetch-to", default=str(HERE / "results" / "large" / "states"), help="where states land, two at a time")
    ap.add_argument("--dry", action="store_true", help="CPU: rows, answer spans and the items, then stop")
    a = ap.parse_args()
    c = config_from_settings(Config, json.loads(PRED.read_text())["settings"]["qwen_reread"])
    data = rows_and_spans(c.arms)
    for rs, (rows, sp) in data.items():
        per_row = torch.bincount(sp.row, minlength=len(rows))
        print(f"[spans] {rs}: {len(sp.row)} whole answers on {len(rows)} rows (per row {int(per_row.min())}-{int(per_row.max())}, "
              f"rows with none {int((per_row == 0).sum())}), tokens per answer {int(sp.n.min())}-{int(sp.n.max())} "
              f"(mean {float(sp.n.double().mean()):.1f}); W0 control rows {w0_rows(sp, c.n16, len(rows))}", flush=True)
    if a.dry:
        hard_exit(0)
    require_filed(PRED)
    assert a.decomp, "--decomp, once per decomposition"
    for d in a.decomp:
        one_decomp(d, c, data, Path(a.fetch_to))
    print("QWEN_REREAD DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
