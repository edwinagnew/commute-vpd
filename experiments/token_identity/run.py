"""E95: what does "reuses token identity" say beyond "the state needs the input token after block 1"? One debug job, nothing trained.
Filing: results/predictions.json; pure rules: read_rules.py; plan: reports/plans/2026-11-05_plan_e95-token-identity.md.
Stage 1, control and null, alone: W0's teeth and do-nothing on both key sets, the write's placement, then every row under the 20 matched
draws only, each block-2 write's gain against its floor, and the thresholds from those draws, written before the basis is read.
Stage 2, the basis: read 1, the hand write at block 2 on ident32 and rand32; read 2, ident32b and rand32b by E90 Amendment 2's rule.
Writes results/run.json as each stage ends, results/large/{reads_*,X_*}.pt. Usage: sbatch cluster/token_identity.sbatch
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import read_rules as RR
import torch
from torch import Tensor

from commute import perturb as PT
from commute import readout as RO
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment

A2 = load_experiment("new_vs_reuse", "amend2")
AP, RD, RA, S1, log = A2.AP, A2.RD, A2.RA, A2.S1, A2.log

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
OUT, FILING = RESULTS / "run.json", RESULTS / "predictions.json"
AMEND2 = AP.RESULTS / "amend2.json"
BLOCK, WRITE_2, SEEDS, ROTATIONS = 1, "h.2.mlp.down_proj", (0, 1), range(1, 21)
TABLE_A, TABLE_B = ("ident32", "rand32"), ("ident32b", "rand32b")
TAG_A, TAG_B = 2, 6  # E90's identity tag, so the same draws; 6 is unused by E90's 1-5
REPRODUCE_TOL = A2.REPRODUCE_TOL
Reads = dict[Any, tuple[Tensor, Tensor]]  # as apparatus.Reads
Rows = dict[str, tuple[Any, Any]]  # name -> (state, reads); a failed write is (None, None)


def halt(res: dict[str, Any], why: str) -> NoReturn:
    res["stopped"] = why
    AP.dump(OUT, res)
    log(f"STOP: {why} (exit 1)")
    hard_exit(1)


def records() -> dict[Path, str]:
    """E90's saved hashes, and the replicate tables', which E90 never read."""
    groups, out = json.loads((S1.REG.parent / "stage0a.json").read_text())["groups"], dict(AP.records())
    for n in TABLE_B:
        out[S1.REG / f"windows_{n}.pt"] = groups["e12/build"]["sha256"][n]
        out |= {S1.REG / f"{n}_s{s}.pt": groups[f"e12/{n}"]["states"][f"s{s}"]["sha256"] for s in SEEDS}
    return out


def checked(j: Any, path: Path) -> Path:
    try:
        return AP.checked(path, j.rec, j.res["sha256"])
    except ValueError as e:
        halt(j.res, str(e))


# ---- stage 1: control and null ----


def control(j: Any, form: str, name: str, rows: Tensor, table: str, tag: int, bases: dict[int, Tensor] | None) -> tuple[Any, Any, bool]:
    """X_C on the table's keys, the matched draws, W0's teeth and the do-nothing state, as Amendment 2. `bases` None: fit, as copy-24."""
    keys = torch.tensor(list(torch.load(S1.REG / f"windows_{table}.pt", weights_only=False)["task"].keys))
    c = AP.identity_capability(j.net, name, rows, keys, S1.HALF, j.S, j.S["n_control"])
    c.read = AP.own_top1(j.net, c.read)
    f = A2.matched_basis(j, form, c, bases or {BLOCK: c.bases([BLOCK])[BLOCK].float()}, tag)
    if not f.bases:
        return c, f, False
    on = RD.control(j, c, f)
    RD.do_nothing(j, c, f, on, BLOCK)
    return c, f, on[str(BLOCK)]


def reproduce(j: Any, saved: dict[str, Any]) -> None:
    """Table A's control is Amendment 2's: same basis, same draws."""
    got, was = j.res["control"]["token identity"][str(BLOCK)], saved["control"]["token identity"][str(BLOCK)]
    for field in ("drop", "band", "variance_share", "variance_share_random"):
        if not abs(got[field] - was[field]) < REPRODUCE_TOL:
            halt(j.res, f"control {field} {got[field]} against Amendment 2's {was[field]}")


def placement(j: Any, s: Any, u: Tensor) -> None:
    """The removal must reach the input of block 2's write and not that of block 1's, each against a second clean pass (the numeric
    floor, expected 0); the key must sit at `half` only, so it cannot reach the write another way."""
    clean, again = j.net.run(s.toks, capture=True)[1], j.net.run(s.toks, capture=True)[1]
    cut = j.net.run(s.toks, AP.removal_edit(BLOCK, s.half, u), capture=True)[1]
    change = lambda a, site: float((a[site][:, s.half] - clean[site][:, s.half]).norm() / clean[site][:, s.half].norm())
    earlier = int((s.toks[:, : s.half] == s.toks[:, s.half : s.half + 1]).any(1).sum())
    rec = {"mlp1_change": change(cut, "mlp1"), "mlp1_floor": change(again, "mlp1"), "mlp2_change": change(cut, "mlp2"), "mlp2_floor": change(again, "mlp2"), "windows_with_key_earlier": earlier, "n_windows": len(s.toks)}
    j.res["placement"] = rec
    log(f"[placement] {json.dumps(rec)}")
    if rec["mlp1_change"] > rec["mlp1_floor"] or not rec["mlp2_change"] > rec["mlp2_floor"] or earlier:
        halt(j.res, f"the removal does not sit between the two writes: {rec}")


def under_draws(j: Any, s: Any, dW: Any, f: Any) -> Reads:
    return {"clean": AP.reads(j.net, s, dW), **{(BLOCK, i): AP.reads(j.net, s, dW, AP.removal_edit(BLOCK, s.half, r)) for i, r in enumerate(f.randoms[BLOCK])}}


def draw_shares(j: Any, w0: Reads, st: Reads, f: Any) -> list[float]:
    return [RO.shares(w0["clean"], w0[BLOCK, i], st["clean"], st[BLOCK, i], j.S["gain_floor"])["A_lo"] for i in range(len(f.randoms[BLOCK]))]


def fit_write(j: Any, name: str, key: str, s: Any, w: dict[str, Any], G: float) -> tuple[Any, tuple[Tensor, Tensor], dict[str, Any]]:
    """The existing hand write at matched gain, `AP.hand_write` unchanged, at `key`: perturb reads WRITE_KEY when called."""
    with patch.object(PT, "WRITE_KEY", key):
        inputs = AP.table_inputs(j.net, w["train"].tokens, s.half, *RD.table_of(w))
        dW, rec, X = AP.hand_write(j.net, *inputs, s, G, j.S["write_rel_tol"])
    j.res.setdefault("writes", {})[name] = {**rec, "key": key}
    log(f"[write {name} at {key}] {json.dumps(rec)}")
    path = LARGE / f"X_{name.split('_')[0]}_{key.split('.')[1]}.pt"
    if not path.exists():
        torch.save(X.float().cpu(), path)
    return dW, inputs, rec


def write_floor(j: Any, name: str, s: Any, inputs: tuple[Tensor, Tensor], rec: dict[str, Any], lo0: Tensor) -> None:
    """The same write, the same alpha, the answers rotated to other keys: its gain on the true targets, log-odds."""
    A, U = inputs
    rotated = [] if rec["status"] != "ok" else [float((AP.reads(j.net, s, AP.zero_except(j.net, WRITE_2, rec["alpha"] * PT.key_write(A, U.roll(r, 1))))[1] - lo0).mean()) for r in ROTATIONS]
    out = {**RR.write_works(rec["status"], rec.get("G_write", float("nan")), rotated), "rotated": rotated}
    j.res.setdefault("write_check", {})[name] = out
    log(f"[write check {name}] gain {out['gain']} floor {out['floor']} works {out['works']}")


def null_stage(j: Any, f: Any, table: str, key: str) -> tuple[Any, Reads, Rows]:
    """A table's W0, states and writes under the draws. A block-2 write also gets its floor and the state-minus-write thresholds."""
    s, w = AP.kv_site(checked(j, S1.REG / f"windows_{table}.pt"))
    w0, rows = under_draws(j, s, None, f), {}
    A2.task_shares(j, f"windows_{table}", s, f)
    for seed in SEEDS:
        name = f"{table}_s{seed}"
        dW = torch.load(checked(j, S1.REG / f"{name}.pt"), weights_only=True)
        st = under_draws(j, s, dW, f)
        rows[name] = (dW, st)
        dw, inputs, rec = fit_write(j, name, key, s, w, float((st["clean"][1] - w0["clean"][1]).mean()))
        wr = under_draws(j, s, dw, f) if dw else None
        rows[f"{name}_write"] = (dw, wr)
        if key == WRITE_2:
            write_floor(j, name, s, inputs, rec, w0["clean"][1])
            shares = draw_shares(j, w0, wr, f) if wr else [float("nan")] * len(f.randoms[BLOCK])
            j.res.setdefault("thresholds", {})[name] = RR.gaps_under_draws(draw_shares(j, w0, st, f), shares)
            log(f"[thresholds {name}] gap under the draws from {j.res['thresholds'][name]['lo']} to {j.res['thresholds'][name]['hi']}")
    AP.dump(OUT, j.res)
    return s, w0, rows


# ---- stage 2: the basis ----


def basis_stage(j: Any, f: Any, s: Any, w0: Reads, rows: Rows, table: str) -> None:
    """The basis's reads on W0 and each row; every row's reads saved per window for a later resampling."""
    u = f.bases[BLOCK]
    w0[BLOCK, -1] = AP.reads(j.net, s, None, AP.removal_edit(BLOCK, s.half, u))
    for name, (dW, reads) in rows.items():
        if reads is None:
            j.res["states"][name] = {}
            continue
        reads[BLOCK, -1] = AP.reads(j.net, s, dW, AP.removal_edit(BLOCK, s.half, u))
        j.res["states"][name] = AP.state_row(w0, reads, [BLOCK], j.S["n_random"], j.S["gain_floor"])
        AP.say_row(name, j.res["states"][name], [BLOCK])
    saved = {n: {str(k): (v[0].float(), v[1].float()) for k, v in r.items()} for n, (_, r) in rows.items() if r}
    torch.save({"W0": {str(k): (v[0].float(), v[1].float()) for k, v in w0.items()}, **saved}, LARGE / f"reads_{table}.pt")
    AP.dump(OUT, j.res)


def reproduce_states(j: Any, saved: dict[str, Any]) -> None:
    for name in (f"{t}_s{s}" for t in TABLE_A for s in SEEDS):
        got, was = j.res["states"][name][str(BLOCK)], saved["states"][name][str(BLOCK)]
        if not abs(got["A_lo"] - was["A_lo"]) < REPRODUCE_TOL or not abs(got["band"] - was["band"]) < REPRODUCE_TOL:
            halt(j.res, f"{name}: share {got['A_lo']}, band {got['band']} against Amendment 2's {was['A_lo']}, {was['band']}")


def read_one(j: Any, teeth: bool) -> None:
    def seed_row(name: str) -> dict[str, Any]:
        st, wr = j.res["states"][name][str(BLOCK)], j.res["states"][f"{name}_write"].get(str(BLOCK), {})
        nan = float("nan")
        return RR.seed_row(st["A_lo"], wr.get("A_lo", nan), wr.get("random", [nan]), j.res["thresholds"][name], j.res["write_check"][name]["works"])

    rows = {t: {f"s{s}": seed_row(f"{t}_s{s}") for s in SEEDS} for t in TABLE_A}
    gaps = {t: {f"s{s}": {"A_lo": rows[t][f"s{s}"]["gap"], "random": j.res["thresholds"][f"{t}_s{s}"]["draw_gaps"]} for s in SEEDS} for t in TABLE_A}
    difference = RR.paired(gaps["ident32"], gaps["rand32"])
    j.res["read1"] = {"rows": rows, "difference": difference, **RR.read1(teeth, list(rows["ident32"].values()), list(rows["rand32"].values()), difference["call"])}
    log(f"[read 1] {json.dumps(j.res['read1'])}")


def read_two(j: Any, teeth: bool, reason: str | None) -> None:
    names = [f"{t}_s{s}" for t in TABLE_B for s in SEEDS]
    by = RA.read_block(j.res["states"], names, BLOCK, teeth, reason)
    row = lambda t: {f"s{s}": j.res["states"][f"{t}_s{s}"][str(BLOCK)] for s in SEEDS}
    own = {t: RR.own_bands([r["A_lo"] for r in row(t).values()], [r["band"] for r in row(t).values()], teeth and not reason) for t in TABLE_B}
    pair = RR.paired(row("ident32b"), row("rand32b"))
    j.res["read2"] = {"filed": by, "own_bands": own, "paired": pair, "outcome": RR.read2(by["status"]["ident32b"], by["status"]["rand32b"], pair["call"])}
    log(f"[read 2] {json.dumps(j.res['read2'])}")


def refused() -> str | None:
    git = lambda *a: subprocess.run(["git", "-C", str(AP.REPO), *a], capture_output=True, text=True, check=False).stdout
    rel = str(FILING.relative_to(AP.REPO))
    if not git("ls-files", rel).strip() or git("diff", "HEAD", "--", rel):
        return f"{rel} is not committed unchanged"
    return None if json.loads(AMEND2.read_text()).get("complete") else "E90 Amendment 2 has not completed"


def main() -> None:
    why = refused()
    if why:
        log(f"{why}: nothing runs (exit 1)")
        hard_exit(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    LARGE.mkdir(parents=True, exist_ok=True)
    saved = json.loads(AMEND2.read_text())
    S = json.loads(AP.PRED.read_text())["settings"]
    j = RD.Job(ST.Net("cuda"), S, records(), {"sha256": {}, "states": {}}, OUT)
    rows = pile_rows(*S["pile"]["rows"], log=log)
    basis_a = torch.load(AP.LARGE / "bases.pt", weights_only=True, map_location="cpu")["identity"][BLOCK]
    _, f_a, teeth_a = control(j, "identity", "token identity", rows, "ident32", TAG_A, {BLOCK: basis_a})
    reproduce(j, saved)
    _, f_b, teeth_b = control(j, "table_b", "token identity, table B's keys", rows, "ident32b", TAG_B, None)
    AP.dump(OUT, j.res)
    placement(j, AP.kv_site(checked(j, S1.REG / "windows_ident32.pt"))[0], f_a.bases[BLOCK])
    a = {t: null_stage(j, f_a, t, WRITE_2) for t in TABLE_A}
    b = {t: null_stage(j, f_b, t, PT.WRITE_KEY) for t in TABLE_B} if f_b.bases else {}

    for t, (s, w0, r) in a.items():
        basis_stage(j, f_a, s, w0, r, t)
    reproduce_states(j, saved)
    read_one(j, teeth_a)
    for t, (s, w0, r) in b.items():
        basis_stage(j, f_b, s, w0, r, t)
    if b:
        read_two(j, teeth_b, j.res["matched"]["table_b"][str(BLOCK)]["reason"])
    else:
        j.res["read2"] = {"outcome": RR.NO_CALL_REPLICATE, "reason": j.res["matched"]["table_b"][str(BLOCK)]["reason"]}
    j.res["complete"] = True
    AP.dump(OUT, j.res)
    log("within means not shown specific, never shown not specific; a circuit from old parts reads as reuse. E95 complete")
    hard_exit(0)


if __name__ == "__main__":
    main()
