"""E272 stage 3 (cell P2.66, gated: pods, gate 6 by whip-lead): E_b over the interchange-needed K of each selected
panel state, P1.54's path (E268, experiments/kstar_needed) called unchanged: E257's World (`rb_states`) turns a saved
dW into F, `kstar_vpd.featurize` splits it over the released decomposition's variables, `needed.cache`/`derangements`/
`scan` find K (the variables whose write, interchanged with the write on another window of D, changes F's argmax at
the read position on some tested pair), and `stand_in.write_change`/`explained_share` give E_b over K and changed.
Never mean ablation (that is K*, `kstar_vpd.search`, not called here). K is found on `derangements` x `examples`
tested pairs, a subset of the definition's pairs: every E_b here is labelled "sampled estimate (P pairs)".

D per task: the first n_eb read windows of results/read's Read (`tasks.build`), cut to end at the read position and
read at the last position (rb_states' `rl_windows` form). Label and prompt tasks read the argmax over the vocabulary,
as every P1.54 item does; the two-label contrast is a reading of stage 2, not of K.

  0    alone: the do-nothing (a zero dW) has no changed variable, dd exactly 0 and E_b exactly 1; the identity
       interchange flips no argmax (E268's R0a); E247's x2 control's saved E_b over K* repeats to 1e-11 (R0c); its
       E_b over K by a fresh scan, against E268's saved value              -> results/eb/stage0.json
  TASK i/n M  every n-th state of TASK picked at matched M (kl: first; size: only if kl reads), from the i-th, and
               its Gaussians  -> results/eb/TASK.json

    PRED_SHA=<sha> python experiments/convergence_panel/eb.py 0
    PRED_SHA=<sha> python experiments/convergence_panel/eb.py TASK 0/1 kl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import kstar_vpd as KV
from commute import needed as IX
from commute import stand_in as SI
from commute.jobs import hard_exit, load_experiment, write_stage
from commute.open_pair import jsonable

RUN: Any = load_experiment("convergence_panel", "run")
KN: Any = load_experiment("kstar_needed", "run")  # E268: world, Config, saved, identity, batch_rows
KP = KN.KP
OUT = RUN.RESULTS / "eb"


def e_b_file(w: Any, kc: Any, dW_file: Path, toks: torch.Tensor) -> dict[str, Any]:
    """E_b over K for one saved dW on windows toks, read at the last position."""
    F, M, toks, at, _ = w.from_dw(dW_file, toks.cuda(), KP.LAST)
    return e_b(w, kc, F, M, toks, at)


def e_b(w: Any, kc: Any, F: Any, M: Any, toks: torch.Tensor, at: slice) -> dict[str, Any]:
    """E_b over K and changed of F against M on D = toks at the read positions `at`, with K's facts."""
    t0 = time.time()
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    run = IX.cache(F, feat, toks, at, kc.cache_batch)
    m = min(kc.examples, len(toks))
    ex, pt = IX.derangements(len(toks), kc.derangements, m, kc.seed)
    sc = IX.scan(F, feat, run, at, ex, pt, KN.batch_rows(kc, run.top.shape[1]), RUN.log)
    K = sc.first_pair >= 0
    rec = {"E_b_K": SI.explained_share(num, den, dd, K & ch), "label": f"sampled estimate ({len(ex)} pairs: {kc.derangements} derangements x {m})",
           "size_K": int(K.sum()), "n_changed": len(feat.changed), "n_changed_K": int((K & ch).sum()), "dd_max": float(dd.abs().max()),
           "n_vars": feat.n, "n_windows": len(toks), "min_margin": float(run.margin.min()), "seconds": time.time() - t0,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}  # fmt: skip
    del F, run
    torch.cuda.empty_cache()
    return rec


def windows(r: Any, n: int) -> torch.Tensor:
    """The first n read windows, cut to end at the read position (an LM window carries its target after it)."""
    return r.toks[:n, : r.pos + 1]


def stage0(w: Any, kc: Any, th: dict[str, Any]) -> None:
    path, out = OUT / "stage0.json", dict[str, Any](experiment="E272")
    C, _ = RUN.load()
    net = w.kl.net
    zero = RUN.LARGE / "eb_zero.pt"
    zero.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.zeros_like(v).bfloat16().cpu() for k, v in net.W0.items()}, zero)
    r = RUN.TK.build(RUN.ST.Net("cuda"), C, "lower").read
    out["do_nothing"] = e_b_file(w, kc, zero, windows(r, th["eb"]["n_windows"]))
    a = KN.saved(kc.control, "built")
    eb = SI.explained_share(a["num"], a["den"], a["dd"], a["Kstar"] & a["changed"])
    out["x2_Kstar"] = {"got": eb, "filed": kc.control_E_b_Kstar, "rel_dev": KN.rel(eb, kc.control_E_b_Kstar)}
    out["identity_last"] = KN.identity(w, kc, KP.LAST)
    write_stage(path, jsonable(out))
    F, M, toks, at, _ = w.make(kc.control)  # E268's x2 item: its windows, every read position
    x2, ref = e_b(w, kc, F, M, toks, at), th["eb"]["x2_E_b_K"]
    out["x2_K"] = x2 | {"saved": ref["value"], "tol": ref["tol"]}
    miss = [f"do-nothing {k} {out['do_nothing'][k]}" for k, want in (("n_changed", 0), ("dd_max", 0.0), ("E_b_K", 1.0)) if out["do_nothing"][k] != want]
    miss += [f"identity: {out['identity_last']['flips']} argmax changes"] if out["identity_last"]["flips"] else []
    miss += [f"x2 E_b over K* {eb!r}"] if out["x2_Kstar"]["rel_dev"] > kc.transcription_rel_tol else []
    miss += [f"x2 E_b over K {x2['E_b_K']!r} off E268's {ref['value']!r}"] if not abs(x2["E_b_K"] - ref["value"]) <= ref["tol"] else []
    write_stage(path, jsonable(out | {"misses": miss, "passed": not miss}))
    RUN.log(f"[eb stage0] {json.dumps(jsonable(out))}")
    if miss:
        raise SystemExit(f"STOP: eb stage 0: {miss}: tell the lead")


def task(w: Any, kc: Any, th: dict[str, Any], name: str, shard: tuple[int, int], match: str) -> None:
    """Every n-th of TASK's states picked by `match` ("kl" or "size": both levels) and their Gaussians, from the i-th."""
    gate = json.loads((OUT / "stage0.json").read_text()) if (OUT / "stage0.json").is_file() else {}
    assert gate.get("passed"), "eb stage 0 has not passed"
    C, _ = RUN.load()
    path = OUT / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": name, "states": {}}
    r = RUN.TK.build(RUN.ST.Net("cuda"), C, name).read
    toks = windows(r, th["eb"]["n_windows"])
    picked = {f: s for f, s in RUN.picked(name).items() if any(p.startswith(match) for p in s["picked"])}
    files = sorted(picked) + [f"random/{f}" for f in sorted(picked) if picked[f]["seed"] == 0]  # stage 2's Gaussians
    for f in files[shard[0] :: shard[1]]:
        if f not in out["states"]:
            out["states"][f] = e_b_file(w, kc, random_file(f, C) if f.startswith("random/") else RUN.REPO / f, toks)
            write_stage(path, jsonable(out))
            RUN.log(f"[eb {name}] {f}: {json.dumps(jsonable(out['states'][f]))}")


def random_file(item: str, C: Any) -> Path:
    """Stage 2's Gaussian of a seed-0 state (`finetune.random_direction`, C.random_seed), written once in bf16."""
    f = item.removeprefix("random/")
    p = RUN.LARGE / "eb_random" / Path(f).name
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.bfloat16() for k, v in RUN.FT.random_direction(RUN.load_dW(f), C.random_seed).items()}, p)
    return p


def main() -> int:
    a = sys.argv[1:]
    if not a or (a[0] != "0" and (a[0] not in RUN.TASKS or len(a) != 3 or a[2] not in ("kl", "size"))):
        sys.exit("usage: eb.py 0 | eb.py TASK i/n kl|size")
    _, th = RUN.load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    w, kc = KN.world(), KN.load_config()
    if a[0] == "0":
        stage0(w, kc, th)
    else:
        i, n = (int(x) for x in a[1].split("/"))
        task(w, kc, th, a[0], (i, n), a[2])
    print(f"EB STAGE {' '.join(a)} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
