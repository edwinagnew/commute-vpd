"""E151 Amendment 1 (results/predictions_amend1.json): run 1 (job 58739) passed every control and stopped on an empty
common-word pool. Stage 0 runs run 1's controls unchanged and first, logs where W0's probabilities sit (before any table
check), then draws the common-word cipher by the filed ladder: the first rung whose table is complete and whose estimated
prior is >= common_prior_min, each key taking its highest-P unused admissible word (fewest-candidate keys first). The
realised prior must clear common_prior_min; pair B is counted only at or above b_prior_min. Stages 1 and 2 are run.py's.
Writes results/amend1_stage{0,1,2}.json (run 1's stage0.json is kept).

    sbatch --comment='E151 rerun: common-word ladder after empty pool' cluster/disagreeing_pairs_amend1.sbatch
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import rules
import run as R
import torch
from job import LARGE, NEW, PRED, RESULTS, Run, check, load, log, say

from commute import anchors as AN
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute.data import Windows
from commute.heldout_keys import keyed
from commute.jobs import hard_exit

PRED_A1 = RESULTS / "predictions_amend1.json"
L = load(PRED_A1)["ladder"]
RANKS = sorted({r["min_rank"] for r in L["rungs"]})
RATIOS = sorted({r["median_ratio"] for r in L["rungs"]})
B = rules.COUNTED[1]


def q5(x: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(x.double(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return dict(zip(("min", "25%", "median", "75%", "max"), q.tolist(), strict=True))


@torch.no_grad()
def window_logp(fwd: Any, w: Windows, device: str, batch: int = 64) -> torch.Tensor:
    """(windows, vocab) float64: W0's log P at `half` on each window (as kv.key_distributions computes it), a row each."""
    return torch.cat([torch.log_softmax(fwd(w.tokens[s : s + batch].to(device))[:, w.half].float(), -1).double().cpu()
                      for s in range(0, len(w), batch)])  # fmt: skip


def w0_tables(run: Run, src: dict[str, Any], keys: list[int], trained: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """G (keys, vocab): W0's geometric-mean P(v | M k) over every window, for the filters (as run 1). The windows the filed
    prior reads, the trained keys' held-out windows (save_map's windows_<m>_trained.pt "heldout"): LP (windows, vocab),
    W0's log P on each, and each window's key. A trained key without a held-out window has no row, as in the prior."""
    tr, ho = src["train"], src["heldout"]
    both = Windows(torch.cat([tr.tokens, ho.tokens]), torch.cat([tr.g, ho.g]), torch.cat([tr.row, ho.row]), torch.cat([tr.pos, ho.pos]), tr.half)
    fwd = lambda x: run.net.run(x)[0]
    geo = kv.key_distributions(fwd, both, run.net.device, geometric=True)
    hw = keyed(ho, trained)
    return torch.stack([geo[k] for k in keys]), window_logp(fwd, hw, run.net.device), hw.tokens[:, hw.half].tolist()


def diagnostics(G: torch.Tensor, f: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Where the probabilities sit: per-key quantiles of the r-th largest G; words and keys passing each filter and pair."""
    d: dict[str, Any] = {f"rank {r} G per key": q5(G.topk(r, dim=1).values[:, -1]) for r in RANKS}
    d["filters"] = {a if a == b else f"{a} & {b}": {"words": int((f[a] & f[b]).any(0).sum()), "keys": int((f[a] & f[b]).any(1).sum())}
                    for a, b in itertools.combinations_with_replacement(f, 2)}  # fmt: skip
    return d


def rung(G: torch.Tensor, LP: torch.Tensor, wkeys: list[int], f: dict[str, torch.Tensor], rg: dict[str, Any], keys: list[int],
         pool: list[int]) -> dict[str, Any]:  # fmt: skip
    """One rung: its table by rules.assign, the per-key candidate counts and best P, and the estimated filed prior (nan
    for an incomplete table)."""
    ok = rules.rung_ok(f, rg["min_rank"], rg["median_ratio"], L["value_max"])
    P = G[:, pool]
    cols = rules.assign(P, ok)
    vals = {k: (pool[c] if c is not None else None) for k, c in zip(keys, cols, strict=True)}
    count = ok.sum(1).double()
    return {**rg, "complete": all(c is not None for c in cols), "n_keys_assigned": sum(c is not None for c in cols),
            "candidates_per_key": {"min": int(count.min()), "median": float(count.median())},
            "best_P_per_key": q5((P * ok).max(1).values), "prior_est": rules.window_prior(LP, wkeys, vals),
            "values": [vals[k] for k in keys]}  # fmt: skip


def one_word(G: torch.Tensor, keys: list[int], pool: list[int], marker: int, out: dict[str, Any]) -> kv.KVTask | None:
    """Run 1's one-word draw, unchanged, with its candidate count and the chosen word's max P over keys logged."""
    m = R.draw_one_word(G, keys, pool, marker)
    mx = G[:, pool].max(0).values
    out["one_word_draw"] = {"n_candidates": int((mx < R.MAPS.one_word_max).sum()), "word": m.values[0] if m else None,
                            "max_P_over_keys": float(G[:, m.values[0]].max()) if m else math.nan}  # fmt: skip
    say("0 one-word draw", out["one_word_draw"])
    return m


def ladder(run: Run, src: dict[str, Any], keys: list[int], trained: list[int], marker: int, out: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Everything logged before a check: the diagnostics, every rung, the rung taken, the one-word draw."""
    G, LP, wkeys = w0_tables(run, src, keys, trained)
    _, _, _, ranked, tok = KD.vocab(run.ctx)
    banned = set(keys) | set(run.F["a3"]["maps"]["elicit"]["values"])
    pool = [v for v in ranked[: R.MAPS.common_ranks] if v not in banned]
    f = rules.filters(G, pool, RANKS, RATIOS, L["value_max"])
    f[f"P >= {L['band_floor_reported']:g} (run 1 band floor, reported only)"] = G[:, pool] >= L["band_floor_reported"]
    out["diagnostics"] = {"n_pool": len(pool), **diagnostics(G, f)}
    say("0 diagnostics", out["diagnostics"])
    out["rungs"] = rungs = [rung(G, LP, wkeys, f, rg, keys, pool) for rg in L["rungs"]]
    for r in rungs:
        say(f"0 rung {r['rung']}", {k: v for k, v in r.items() if k != "values"})
    taken = rules.take_rung(rungs, L["common_prior_min"])
    out["rung_taken"] = None if taken is None else rungs[taken]["rung"]
    log(f"[0 rung taken] {out['rung_taken']} (first complete rung with estimated prior >= {L['common_prior_min']:g})")
    pool1 = [v for v in ranked[AN.VALUE_RANK0 : AN.VALUE_RANK0 + AN.POOL] if v not in banned]
    common = None if taken is None else kv.KVTask("common_cipher", marker, tuple(keys), tuple(rungs[taken]["values"]))
    return {"common_cipher": common, "one_word": one_word(G, keys, pool1, marker, out)}, tok


def build_tables(run: Run, ch: dict[str, Any], out: dict[str, Any]) -> None:
    """The ladder's record first; then run 1's file and split checks, the rung, the two tables saved and their priors."""
    a3 = run.F["a3"]
    keys = a3["keys"]["keys"]
    trained, held = OP.split_keys(keys, len(keys) // 4, 0)
    src = torch.load(R.A3_DIR / "windows_elicit.pt", weights_only=False)
    made, tok = ladder(run, src, keys, trained, src["task"].marker, out)
    check(ch, "0 sha256 windows_elicit.pt", float(KD.sha256(R.A3_DIR / "windows_elicit.pt") != a3["maps"]["elicit"]["sha256"]["full"]), 0)
    check(ch, "0 key split is Amendment 3's (0 = yes)", float(held != a3["keys"]["held"]), 0)
    check(ch, "0 a ladder rung qualifies (0 = yes)", float(made["common_cipher"] is None), 0)
    check(ch, "0 one-word candidate found (0 = yes)", float(made["one_word"] is None), 0)
    tables = {n: m for n, m in made.items() if m is not None}  # both, past the two checks above
    rec: dict[str, Any] = {"maps": {}, "tables": {"letter_of": {str(k): tok.decode([k]).strip()[:1].lower() for k in keys}}}
    for name, m in tables.items():
        R.A6.save_map(run.ctx, src, m, "ranks 0-2999 ladder" if name == "common_cipher" else "cipher_pool", trained, held, LARGE, rec)
        run.add_new(name)
        log(f"[0 table {name}] first values {[tok.decode([v]) for v in m.values[:12]]} prior {run.map_prior(name):.6e}")
    out["tables"] = {n: {**rec["maps"][n], "prior": run.prior[n], "keys": list(m.keys)} for n, m in tables.items()}
    say("0 priors and gaps (decades)", {**run.prior, **{f"gap {p}": R.prior_gap(run, *ms) for p, ms in R.PAIRS.items()}})
    for n in NEW:
        check(ch, f"0 table {n}: zero update reads 0", rec["maps"][n]["zero_max_abs"], 0)
    p = run.prior["common_cipher"]
    out["prior_est_taken"] = est = next(r["prior_est"] for r in out["rungs"] if r["rung"] == out["rung_taken"])
    log(f"[0 common_cipher prior] realised {p:.6e}, the rung's estimate {est:.6e} (same windows: equal to rounding)")
    check(ch, f"0 table common_cipher realised prior >= {L['common_prior_min']:g} (0 = yes)", float(not p >= L["common_prior_min"]), 0)
    check(ch, f"0 table one_word prior under {R.MAPS.one_word_max:g} (0 = yes)", float(not run.prior["one_word"] < R.MAPS.one_word_max), 0)
    run.premise[B] = out["b_premise"] = rules.b_premise(p, L["b_prior_min"])
    log(f"[0 pair B premise] common_cipher prior {p:.4e} {'>=' if run.premise[B] else '<'} {L['b_prior_min']:g}: "
        f"{'counted under the filed rule' if run.premise[B] else 'condition not met (not a stop)'}")  # fmt: skip


def stage0(run: Run, out: dict[str, Any], _t_job: float) -> None:
    build_tables(run, R.controls(run, out), out)


if __name__ == "__main__":
    hard_exit(R.main(stage0, "amend1_stage", (PRED, PRED_A1)))
