"""E272 stage 3 (cell P2.66, gated: pods, gate 6 by whip-lead): E_b over the interchange-needed K of each selected
panel state, P1.54's path (E268, experiments/kstar_needed) called unchanged: E257's World (`rb_states`) turns a saved
dW into F, `kstar_vpd.featurize` splits it over the released decomposition's variables, `needed.cache`/`derangements`/
`scan` find K (the variables whose write, interchanged with the write on another window of D, changes F's argmax at
the read position on some tested pair), and `stand_in.write_change`/`explained_share` give E_b over K and changed.
Never mean ablation (that is K*, `kstar_vpd.search`, not called here). K is found on `derangements` x `examples`
tested pairs, a subset of the definition's pairs: every E_b here is labelled "sampled estimate (P pairs)".

D per task: the first n_eb read windows of results/read's Read (`tasks.build`), cut to end at the read position and
read at the last position (rb_states' `rl_windows` form). Label and prompt tasks read the argmax over the vocabulary,
as every P1.54 item does; the two-label contrast is a reading of stage 2, not of K. LM tasks' read windows are their
hardest positions by W0 log-odds, prompt and label tasks' a seeded random draw (a known difference, plan 2026-11-23).

  windows SET M  (cluster, $0, before any pod) each task's D, built once, to results/large/eb_windows/TASK.pt, with
       its shape and sha256 in results/wide/eb/windows_manifest.json; every later mode loads D from there (and stops
       on a sha256 mismatch), so a pod needs no dataset and D is exactly the cluster's
  0    alone: the do-nothing (a zero dW) has no changed variable, dd exactly 0 and E_b exactly 1; the identity
       interchange flips no argmax (E268's R0a); E247's x2 control's saved E_b over K* repeats to 1e-11 (R0c); its
       E_b over K by a fresh scan, against E268's saved value              -> results/eb/stage0.json (with the card)
  TASK i/n M [SET]  every n-th state of TASK that SET (nine, the default, or wide: Amendment 5's 18) picked at a match
               starting M (kl, kl_hi, ..., size_lo), from the i-th, its seed-0 Gaussian first -> results/eb/TASK.json
               (nine), results/wide/eb/TASK.json (wide), where score.py reads each set's E_b. A written state is
               skipped; a state that misses `check` stops the job with nothing written for it.

    python experiments/convergence_panel/eb.py windows wide kl_hi
    PRED_SHA=<sha> python experiments/convergence_panel/eb.py 0
    PRED_SHA=<sha> python experiments/convergence_panel/eb.py TASK 0/1 kl_hi wide
"""

from __future__ import annotations

import hashlib
import json
import math
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
OUT = RUN.RESULTS / "eb"  # stage 0's gate, shared by both sets
WINDOWS, MANIFEST = RUN.LARGE / "eb_windows", RUN.RESULTS / "wide" / "eb" / "windows_manifest.json"
MATCHES = ("kl", "size", "kl_hi", "kl_lo", "size_hi", "size_lo")
SEC_PER_64 = 738.0  # s per state, 200 windows of 64 tokens, H100 (E268, definition-lead e268_dl/run_rl_p0.log:8)
# Slowdown against the H100: scan rate 8,800 cases/s on E268's H100 (kstar_needed/results/run_rl_p0.log, site lines)
# against 3,517/s on a cluster L40 (job 64133, site 0). Another card stops the job before its first state.
CARD = {"H100": 1.0, "L40": 2.5}


def gb() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def card() -> str:
    return torch.cuda.get_device_name()


def e_b_file(w: Any, kc: Any, dW_file: Path, toks: torch.Tensor) -> dict[str, Any]:
    """E_b over K for one saved dW on windows toks, read at the last position; seconds and peak memory of the whole state."""
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    F, M, toks, at, _ = w.from_dw(dW_file, toks.cuda(), KP.LAST)
    return e_b(w, kc, F, M, toks, at) | {"seconds_state": time.time() - t0}


def e_b(w: Any, kc: Any, F: Any, M: Any, toks: torch.Tensor, at: slice) -> dict[str, Any]:
    """E_b over K and changed of F against M on D = toks at the read positions `at`, with K's facts."""
    t0 = time.time()
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    run = IX.cache(F, feat, toks, at, kc.cache_batch)
    RUN.log(f"    [eb] cache of {tuple(toks.shape)} written: max GPU {gb():.2f} GB")
    m = min(kc.examples, len(toks))
    ex, pt = IX.derangements(len(toks), kc.derangements, m, kc.seed)
    sc = IX.scan(F, feat, run, at, ex, pt, KN.batch_rows(kc, run.top.shape[1]), lambda s: RUN.log(f"{s}; max GPU {gb():.2f} GB"))
    K = sc.first_pair >= 0
    rec = {"E_b_K": SI.explained_share(num, den, dd, K & ch), "label": f"sampled estimate ({len(ex)} pairs: {kc.derangements} derangements x {m})",
           "size_K": int(K.sum()), "n_changed": len(feat.changed), "n_changed_K": int((K & ch).sum()), "dd_max": float(dd.abs().max()),
           "n_vars": feat.n, "n_windows": len(toks), "tokens": toks.shape[1], "min_margin": float(run.margin.min()),
           "seconds": time.time() - t0, "peak_gpu_gb": gb(), "card": card()}  # fmt: skip
    del F, run
    torch.cuda.empty_cache()
    return rec


def windows(r: Any, n: int) -> torch.Tensor:
    """The first n read windows, cut to end at the read position (an LM window carries its target after it)."""
    return r.toks[:n, : r.pos + 1].contiguous().clone()  # a clone: torch.save of a view saves the whole read


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()


def save_windows(C: Any, th: dict[str, Any], which: str, match: str) -> None:
    """Each task with a state in the set's `match` picks: its D to WINDOWS, its shape, dtype and sha256 to MANIFEST."""
    man: dict[str, Any] = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    WINDOWS.mkdir(parents=True, exist_ok=True)
    for name in (t for t in RUN.TASKS if states(t, which, match)):
        toks, f = windows(RUN.TK.build(RUN.ST.Net("cuda"), C, name).read, th["eb"]["n_windows"]), WINDOWS / f"{name}.pt"
        torch.save(toks, f)
        man[name] = {"file": str(f.relative_to(RUN.REPO)), "shape": list(toks.shape), "dtype": str(toks.dtype), "sha256": sha(toks)}
        write_stage(MANIFEST, man)
        RUN.log(f"[eb windows] {name}: {man[name]}")


def load_windows(C: Any, th: dict[str, Any], name: str) -> torch.Tensor:
    """TASK's D from the manifest's file (stops unless its sha256 and shape match); built from the task without one."""
    e = (json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}).get(name)
    if e is None:
        return windows(RUN.TK.build(RUN.ST.Net("cuda"), C, name).read, th["eb"]["n_windows"])
    toks = torch.load(RUN.REPO / e["file"], weights_only=True)
    if sha(toks) != e["sha256"] or list(toks.shape) != e["shape"] or e["shape"][0] != th["eb"]["n_windows"]:
        raise SystemExit(f"STOP: {e['file']}: sha256 {sha(toks)}, shape {list(toks.shape)} against the manifest's {e}")
    return toks


def stage0(w: Any, kc: Any, th: dict[str, Any]) -> None:
    path, out = OUT / "stage0.json", dict[str, Any](experiment="E272", card=card())
    C, _ = RUN.load()
    net = w.kl.net
    zero = RUN.LARGE / "eb_zero.pt"
    zero.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.zeros_like(v).bfloat16().cpu() for k, v in net.W0.items()}, zero)
    out["do_nothing"] = e_b_file(w, kc, zero, load_windows(C, th, "lower"))
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


def states(name: str, which: str, match: str) -> list[str]:
    """The seed-0 Gaussians (stage 2's) first, then TASK's states that set `which` picked at a match starting `match`."""
    picked = {f: s for f, s in RUN.picked(name, which).items() if any(p.startswith(match) for p in s["picked"])}
    return [f"random/{f}" for f in sorted(picked) if picked[f]["seed"] == 0] + sorted(picked)


def check(rec: dict[str, Any], toks: torch.Tensor, n_windows: int) -> list[str]:
    """Stop rules after each state (plan, 2026-11-23): a dW that loaded empty reads E_b exactly 1 with nothing changed."""
    want = SEC_PER_64 * toks.shape[1] / 64 * CARD[next(k for k in CARD if k in rec["card"])]
    miss = [f"E_b_K {rec['E_b_K']!r} not in [0, 1]"] if not (math.isfinite(rec["E_b_K"]) and 0 <= rec["E_b_K"] <= 1) else []
    miss += [f"windows {(rec['n_windows'], rec['tokens'])} not {(n_windows, toks.shape[1])}"] if (rec["n_windows"], rec["tokens"]) != (n_windows, toks.shape[1]) else []
    miss += [f"changed {rec['n_changed']} of {rec['n_vars']}"] if not rec["n_changed"] > 0.99 * rec["n_vars"] else []
    miss += [f"dd_max {rec['dd_max']!r}"] if not rec["dd_max"] > 0 else []
    miss += [f"{rec['seconds_state']:.0f} s, not within 2x of {want:.0f} s"] if not want / 2 <= rec["seconds_state"] <= 2 * want else []
    return miss


def task(w: Any, kc: Any, th: dict[str, Any], name: str, shard: tuple[int, int], match: str, which: str) -> None:
    """Every n-th of `states` from the i-th, each checked, then written to the set's eb file as it finishes."""
    gate = json.loads((OUT / "stage0.json").read_text()) if (OUT / "stage0.json").is_file() else {}
    assert gate.get("passed") and gate.get("card") == card(), f"eb stage 0 has not passed on this card ({card()}): {gate.get('card')}"
    assert any(k in card() for k in CARD), f"no time rule for {card()}: add its slowdown to CARD from a measured scan rate"
    C, P = RUN.load()[0], json.loads(RUN.PRED.read_text())
    path = RUN.set_dir(which) / "eb" / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": name, "set": which, "states": {}}
    toks, n = load_windows(C, th, name), th["eb"]["n_windows"]
    files = states(name, which, match)
    RUN.log(f"[eb {which} {name}] {len(files)} states, shard {shard}: {len(files[shard[0] :: shard[1]])}; windows {tuple(toks.shape)}")
    for f in files[shard[0] :: shard[1]]:
        if f in out["states"]:
            continue
        rec = e_b_file(w, kc, random_file(f, C) if f.startswith("random/") else RUN.REPO / f, toks)
        RUN.log(f"[eb {which} {name}] {f}: {json.dumps(jsonable(rec))}")
        if miss := check(rec, toks, n):
            raise SystemExit(f"STOP: eb {name} {f}: {miss}; not written: tell the lead")
        if f.startswith("random/"):  # the filed expected value has no tolerance: logged for the lead, never a stop
            RUN.log(f"[eb {which} {name}] Gaussian E_b_K {rec['E_b_K']:.4g}; filed expected (no tolerance): {P['expected']['eb']}")
        out["states"][f] = rec
        write_stage(path, jsonable(out))


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
    sets = ([], ["nine"], ["wide"])
    ok_task = len(a) in (3, 4) and a[0] in RUN.TASKS and a[2] in MATCHES and a[3:] in sets
    if not a or not (a == ["0"] or ok_task or (a[0] == "windows" and len(a) == 3 and [a[1]] in sets and a[2] in MATCHES)):
        sys.exit(f"usage: eb.py windows nine|wide M | eb.py 0 | eb.py TASK i/n M [nine|wide]; M in {'|'.join(MATCHES)}")
    C, th = RUN.load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if a[0] == "windows":
        save_windows(C, th, a[1], a[2])
    elif a[0] == "0":
        stage0(KN.world(), KN.load_config(), th)
    else:
        i, n = (int(x) for x in a[1].split("/"))
        task(KN.world(), KN.load_config(), th, a[0], (i, n), a[2], (a[3:] or ["nine"])[0])
    print(f"EB STAGE {' '.join(a)} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
