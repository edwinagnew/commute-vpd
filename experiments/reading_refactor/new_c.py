"""E309 (cell P2.79) stage "scans": the NEW code's side of the readings no earlier reproduction exercised, to
results/new_c.json, keyed as old_c.py keys them (compare.py holds each against old_c_A / old_c_B):

  needed.scan (VPD)       x2, now `scan_sites` over `needed.Swap` sites (cases / hooks / set): the identity control
                          (`identity_scan`, E268's sizes) and every variable against one derangement of 2 windows
  needed_sae.scan         e58/copy_s0_scaled, now `scan_sites` over `needed_sae.Cases`: the same two scans
  SAE write pass          `eb.write_sums` on `sae_feat.sites` (the leftover mask from SAESite.leftover), then
                          `eb.eb(..., leftovers=True)` and `eb.eb(..., leftovers=False)` over E245's K*
  neurons, no remainder   copy24_s0: `eb.write_sums` on `needed_neurons.sites`, `eb.eb(..., leftovers=False)` over E277's K
  per-key steering        E135's copy_s0: `fair_steering.steering_share` under Rule("perkey", key_pos = half - 1, ...)
The x2 and copy24_s0 items are built as new_b.py builds them (World, `rescaled`, `e229_f`). The SAE item and its SAEs
are built by E251's World and E245's `load_saes` (load_experiment, inputs only: no SAE builder is in src/ yet); every
reading above is a src/commute call.

  --dry  CPU: build every input that needs no model (rows, updates, pool, D, K) and stop before the model
    ~/commute-agents/fleet/cpu-run --mem 16G python3 experiments/reading_refactor/new_c.py --dry
    sbatch --comment='E309 stage: scans' experiments/reading_refactor/new_c.sbatch
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from new_b import (
    E229_AT,
    LO,
    copy24_held,
    e229_f,
    e229_read,
    neuron_bounds,
    npz,
    rescaled,
    settings,
    take,
    val,
)
from old import Out as OldOut
from old_c import N_PAIRS, SAE_ITEM, SEED, digest

from commute import eb as EB
from commute import fair_steering as FS
from commute import items as IT
from commute import needed as IX
from commute import needed_sae as NS
from commute import sae_feat as SF
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.windows import read_rows

RESULTS = EXPERIMENTS / "reading_refactor" / "results"


class Out(OldOut):
    def __init__(self) -> None:
        self.path = RESULTS / "new_c.json"
        self.d: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.errors: list[str] = []


def put_scan(o: OldOut, item: str, sc: IX.Scan, **s: Any) -> None:
    val(o, f"needed_set_size/{item}", int((sc.first_pair >= 0).sum()), **s)
    val(o, f"scan_flips/{item}", int(sc.flips.sum()), **s)
    val(o, f"scan_first_digest/{item}", digest(sc.first_pair), **s)
    val(o, f"scan_max_dev/{item}", sc.max_dev, cases=sc.rows, **s)


# ----------------------------------------------------------------------------- inputs (CPU, no model)


def inputs() -> dict[str, Any]:
    s135, lv, c229, nr = settings("steering_copy_lookup"), settings("kstar_levels"), settings("topk_upper"), settings("needed_neurons")
    e62 = read_rows("e62/copy", "held")
    fit = take(e62, slice(0, None, 2))
    pool = torch.tensor(torch.load(EXPERIMENTS / "induction_cue" / "results" / "large" / "windows.pt", weights_only=False)["pools"]["keys"])
    return {
        "e135": {"fit": fit, "read": take(e62, slice(1, None, 2)), "dW": IT.load_dw(IT.resolve("e62/copy_s0")), "pool": pool,
                 "rule": FS.Rule("perkey", s135["half"] - 1, tuple(s135["strengths"]), None, len(fit.toks), torch.float32)},
        "x2_D": copy24_held(), "layer": lv["e195_layer"], "C": lv["e195_C"], "split_tol": settings("kstar_vpd")["split_tol"], "e268": settings("kstar_needed"),
        "copy_D": e229_read("copy24", c229["read_seed"], c229["n_read"]), "nr": nr,
        "hi": neuron_bounds(c229["bands"][nr["band"]], c229["percentile_qs"], c229["percentiles_sha256"]),
        "copy_K": npz("experiments/needed_neurons/results/large/items_copy24_s0.npz")["in_K"],
        "sae_K": npz("experiments/sae_r_a/results/large/items_e58-copy_s0_scaled.npz")["main_in_K_polished"], "f2": settings("fig2_standard"),
    }  # fmt: skip


# ----------------------------------------------------------------------------- gpu part: readings


def steering_perkey(o: OldOut, w: Any, x: dict[str, Any]) -> None:
    a = x["e135"]
    gain = FS.target_gain(w.net, a["read"], a["dW"], LO, a["rule"].batch)
    rec = FS.steering_share(w.net, a["fit"], a["fit"], a["read"], a["dW"], a["rule"], LO, gain, a["pool"])
    val(o, "steering_perkey/e62/copy_s0", rec["share"], rule=repr(a["rule"]), gain=gain, setting=rec["setting"])


def neurons_no_rem(o: OldOut, w: Any, x: dict[str, Any]) -> None:
    from commute import kstar as K
    from commute import needed_neurons as NN
    from commute import rho
    from commute import rule_retention as R

    F, M, toks = e229_f(w.net, "copy24_s0"), R.Model(w.net.net), x["copy_D"].cuda()
    p = rho.Pair(F, M)
    s = EB.write_sums(F.net, NN.sites(p, K.Feat(None, list(x["hi"].to("cuda"))), toks.shape[1]), toks, x["nr"]["cache_batch"], NN.variables(p)[1])
    val(o, "E_b_neurons/copy24_s0@fresh_no_leftover", EB.eb(s, x["copy_K"], leftovers=False), batch=x["nr"]["cache_batch"], n_leftover=int(s.leftover.sum()) if s.leftover is not None else None)


def vpd_scan(o: OldOut, w: Any, x: dict[str, Any]) -> None:
    from commute import kstar_vpd as KV
    from commute import rule_retention as R

    c = x["e268"]
    M = R.Model(w.net.net)
    F, toks, at = rescaled(M, x["layer"], x["C"], 2.0), x["x2_D"].cuda(), slice(None)  # E214's x2: every position
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases(x["split_tol"]))
    r = IX.identity_scan(F, feat, toks, at, c["examples"], c["ident_vars"], c["rows"], c["logit_rows"], c["cache_batch"], print)
    val(o, "scan_flips/x2@vpd_identity", r["flips"], cases=r["cases"])
    val(o, "scan_max_dev/x2@vpd_identity", r["max_dev"], cases=r["cases"])
    run = IX.cache(F, feat, toks, at, c["cache_batch"])
    ex, pt = IX.derangements(len(toks), 1, N_PAIRS, SEED)
    sc = IX.scan(F, feat, run, at, ex, pt, IX.rows_per_forward(c["rows"], c["logit_rows"], run.top.shape[1]), print)
    put_scan(o, "x2@vpd_P1m2", sc, n_vars=feat.n, windows=tuple(toks.shape))


def sae(o: OldOut, x: dict[str, Any]) -> None:
    E251: Any = load_experiment("sae_r_a_random", "run")  # inputs only (see the module docstring)
    cfg, _ = E251.E245.load_config("scored")
    w, saes = E251.World(E251.E242.load_config()[0]), E251.E245.load_saes(cfg)
    F, M, toks, at, _ = w.make(SAE_ITEM)
    for N in (F, M):  # as sae_eb `read`
        N.net.requires_grad_(False)
        for e in N.extra.values():
            e.requires_grad_(False)
    ss = SF.sites(F, M, saes, toks.shape[1])
    n = SF.n_vars(ss)
    changed = torch.zeros(n, dtype=torch.bool)
    changed[sorted(SF.changed_vars(ss))] = True
    s = EB.write_sums(F.net, ss, toks, cfg.gain_batch, changed)
    val(o, f"E_b_sae/{SAE_ITEM}@fresh", EB.eb(s, x["sae_K"], leftovers=True), batch=cfg.gain_batch)
    val(o, f"E_b_sae/{SAE_ITEM}@fresh_no_leftover", EB.eb(s, x["sae_K"], leftovers=False), batch=cfg.gain_batch, n_leftover=int(s.leftover.sum()) if s.leftover is not None else None)
    f2 = x["f2"]
    run = NS.cache(F, ss, toks, at, f2["cache_batch"])
    rows = IX.rows_per_forward(f2["rows"], f2["logit_rows"], run.top.shape[1])
    ex = torch.arange(min(f2["examples"], len(toks))).to(run.top.device)
    only = {j: torch.cat([NS.Swap(q, run.outs[j]).active(ex, f2["ident_vars"], f2["sae_chunk"]).cpu(), torch.tensor([q.rem - q.off])]) for j, q in enumerate(ss)}
    sc = NS.scan(F, ss, run, at, ex, ex, rows, f2["sae_err_rows"], f2["sae_chunk"], print, only)
    val(o, f"scan_flips/{SAE_ITEM}@sae_identity", int(sc.flips.sum()), cases=sc.rows)
    val(o, f"scan_max_dev/{SAE_ITEM}@sae_identity", sc.max_dev, cases=sc.rows)
    ex, pt = IX.derangements(len(toks), 1, N_PAIRS, SEED)
    sc = NS.scan(F, ss, run, at, ex, pt, rows, f2["sae_err_rows"], f2["sae_chunk"], print)
    put_scan(o, f"{SAE_ITEM}@sae_P1m2", sc, n_vars=n, windows=tuple(toks.shape))


def main(is_dry: bool) -> int:
    x = inputs()
    a = x["e135"]
    print({k: tuple(a[k].toks.shape) for k in ("fit", "read")}, "pool", tuple(a["pool"].shape), a["rule"], {k: tuple(x[k].shape) for k in ("x2_D", "copy_D", "hi", "copy_K", "sae_K")})
    print("keys of every fit window in the pool:", bool(torch.isin(a["fit"].toks[:, a["rule"].key_pos], a["pool"]).all()), "| E229 read position", E229_AT)
    if is_dry:
        print("NEW_C inputs built (dry: no model)", flush=True)
        return 0
    from commute.world import World

    o = Out()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every old run
    o.d["_meta"] = {"card": torch.cuda.get_device_name(), "torch": torch.__version__}
    w = World.load("cuda")
    for n, f in (("steering_perkey", steering_perkey), ("neurons_no_rem", neurons_no_rem), ("vpd_scan", vpd_scan)):
        o.run(n, lambda o, f=f, w=w: f(o, w, x))
        torch.cuda.empty_cache()
    del w
    torch.cuda.empty_cache()
    o.run("sae", lambda o: sae(o, x))
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"NEW_C DONE; errors: {o.errors or 'none'}", flush=True)
    return 1 if o.errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="build the inputs on CPU, stop before the model")
    hard_exit(main(ap.parse_args().dry))
