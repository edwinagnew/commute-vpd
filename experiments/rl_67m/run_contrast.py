"""E223 (rl_67m), cell P2.41 run 5: E222's readings (the re-gating share and the scorecard baselines) of E220's saved
states, scored on a per-row two-form contrast in place of the text's next-token log-odds. Card
reports/plans/2026-11-20_plan_rl-67m-contrast-card.md; plan reports/plans/2026-11-19_plan_rl-67m.md (section "Stage 3b
(E223)"); settings, expected values and rules results/predictions_E223.json (the run refuses unless it is committed
unchanged). Readers: E222's (`commute.realistic_reads.Reader` through experiments/realistic_tasks/run2.py's `reader`),
on a `realistic_reads.PairRead` (unit "contrast" = log p(a) - log p(b) per row; no k_half). Pair calls:
score_contrast.py on the CPU. The anchor (french_en) is not re-read: no two-form pair is natural for it, so the score
uses E222's records (results/stage3/french_en_demo_s*.json, log-odds) unchanged.

Rows (`pair_rows`), on stage 1's Pile val prompt rule scanned to n_scan prompts (E222's candidates) for the read rows,
and on the behaviour's training prompts (`rl_arms.train_prompts`) for steering's fitting windows; 64 tokens, read at 63:
  lower   prompts whose own text continuation earns the reward (E222's rows) and whose next token a is a lowercase
          word with a space (letters only) whose capitalised form b (first letter upper) is one vocabulary token
  repeat  prompts whose text does not copy; at the first of the n_copy positions where the copy and the text part,
          a = the token a copy writes, b = the text's token; the window is the 64 tokens before that position

  ctl     first and alone, exit 1 on a miss: E211's L1 seed-0 state at size 0.005 re-read with the per-row pair score
          (each row's pair = its label, the other label) to E211's saved fixed-pair contrast readings; a zero update
          reads every numerator exactly 0 on L1's rows and on each behaviour's pair rows (timed); E53's lowercase
          seed-0 state read on its case-pair contrast, recorded beside (no saved value exists)
  1 2     one pick (repeat, lower): refuses until contrast_ctl.json is done; a Gaussian update at the first state's
          per-matrix norms first (gain at or over MIN_GAIN in the contrast exits 1), then its 12 states, each written
          to results/contrast/<behaviour>_<arm>_s<seed>.json as it finishes

    sbatch --comment='E223 rehearse: contrast ctl' cluster/rl_67m_contrast.sbatch ctl
    sbatch --comment=E223 --dependency=afterok:<ctl id> cluster/rl_67m_contrast.sbatch 1     (and 2)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import realistic_reads as RR
from commute import rl_arms as RA
from commute import rl_tasks as RT
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_E223.json", RESULTS / "contrast"
S2 = load_experiment("rl_67m", "run_stage2")  # E220's context: the model, behaviours, held-out prompts, pools
S3 = load_experiment("rl_67m", "run_stage3")  # E222's Config, state loader and log
R2 = load_experiment("realistic_tasks", "run2")  # E211's reader, rows, zero check and Out
ARMS = ("rl", "shuffled", "own", "demo")
JOBS = ("ctl", "1", "2")
L1_STATE = "s0_0.005"  # E211's L1 state the ctl reproduces (results/run2_L1.json)


def case_partner(texts: list[str]) -> Tensor:
    """(V,) per token: its capitalised form's id when the token is a space and lowercase letters and that form is one
    vocabulary token, else -1."""
    ids = {t: i for i, t in reversed(list(enumerate(texts)))}  # the first id of each string
    out = torch.full((len(texts),), -1, dtype=torch.long)
    for i, t in enumerate(texts):
        if len(t) > 1 and t[0] == " " and t[1:].isalpha() and t[1:].islower():
            out[i] = ids.get(" " + t[1].upper() + t[2:], -1)
    return out


def pair_rows(name: str, p: RT.Prompts, earn: Tensor, partner: Tensor, n_copy: int) -> tuple[Tensor, Tensor]:
    """(windows (N, L), pairs (N, 2)) of the prompts that keep a pair (module docstring), in prompt order."""
    L = p.toks.shape[1]
    if name == "lower":
        a = p.true[:, 0]
        b = partner[a]
        keep = earn & (b >= 0)
        return p.toks[keep], torch.stack([a, b], 1)[keep]
    assert name == "repeat"
    copy = torch.tensor([list(e) for e in p.extra], dtype=torch.long)
    differ = copy != p.true[:, :n_copy]
    assert torch.equal(differ.any(1), ~earn), "the reward and the copy check disagree"
    keep = differ.any(1)
    i = differ.int().argmax(1)  # the first position where they part
    seq = torch.cat([p.toks, p.true], 1)
    win = torch.stack([seq[j, int(i[j]) : int(i[j]) + L] for j in range(len(seq))])
    pairs = torch.stack([copy.gather(1, i[:, None])[:, 0], p.true.gather(1, i[:, None])[:, 0]], 1)
    return win[keep], pairs[keep]


def prompts(x: Any, name: str, n_scan: int) -> tuple[RA.Reader, RT.Prompts, RT.Prompts]:
    """The behaviour's reward reader, its scanned held-out prompts (E222's candidates) and its training prompts."""
    r = x.reader(name)
    rows = cached_rows("val", x.st1["pile_val_start"], x.st1["pile_val_n"])
    held, _ = RT.pile_prompts(r.b, rows, x.texts, x.tok.encode, dataclasses.replace(r.sz, n=n_scan))
    assert torch.equal(held.toks[:256], x.samples[name]["prompts"].long()), "the scan does not start with stage 1's held-out prompts"
    return r, held, RA.train_prompts(r, x.pile_train, x.articles, x.windows, x.C.n_demo_prompts)


def read_of(x: Any, C: Any, name: str) -> tuple[RR.PairRead, dict[str, Any]]:
    r, held, train = prompts(x, name, C.n_scan)
    partner = case_partner(x.texts)
    (toks, tgt), (fit, fit_tgt) = (pair_rows(name, p, r.rewards(p.true, p.extra, 1).bool(), partner, r.sz.n_copy) for p in (held, train))
    info = {"candidates": len(held.toks), "kept": len(toks), "train_candidates": len(train.toks), "fit_kept": len(fit)}
    return RR.PairRead(name, toks[: C.n_read], tgt[: C.n_read], toks.shape[1] - 1, fit[: C.n_fit], fit_tgt[: C.n_fit]), info


def reader(x: Any, C: Any, C2: Any, name: str, o: Any) -> RR.Reader:
    r, info = read_of(x, C, name)
    log(f"[rows {name}] {json.dumps(info)}")
    if len(r.toks) < C.n_read or len(r.fit) < C2.n_select:
        raise o.stop(f"{name}: {info} gives fewer than n_read {C.n_read} pair rows or n_select {C2.n_select} fitting windows")
    rd = R2.reader(x.net, C2, r)
    o.put(f"read {name}", info | {"n_rows": len(r.toks), "n_fit": len(r.fit), "pos": r.pos, "prior_w0": float(rd.w0["rows"]["contrast"].mean())})
    return rd


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ----------------------------------------------------------------------------- jobs


def l1_reproduce(x: Any, C2: Any, th2: dict[str, Any], o: Any) -> None:
    """E211's L1 state on the per-row pair score: (label, other label) per row gives contrast_at's numbers exactly."""
    r, _, _ = R2.task_data(x.net, C2, "L1")
    assert r.ab is not None
    ab = r.ab
    pair = lambda t: torch.stack([t, ab[0] + ab[1] - t], 1)
    rd = R2.reader(x.net, C2, RR.PairRead("L1", r.toks, pair(r.tgt), r.pos, r.fit, pair(r.fit_tgt)))
    R2.zero_check(o, rd)
    o.d["zero L1"] = o.d.pop("zero")
    s = json.loads((R2.RESULTS / "run2_L1.json").read_text())["states"][L1_STATE]
    path = REPO / s["file"]
    if (sha := hashlib.sha256(path.read_bytes()).hexdigest()) != s["sha256"]:
        raise o.stop(f"{path} has sha256 {sha}, E211 recorded {s['sha256']}")
    rec = rd.state({k: v.float() for k, v in torch.load(path, weights_only=False).items()}, 0)
    pick = lambda q: {"share": q["share"]["share"]["contrast"], "svd": q["untrained"]["share"]["contrast"]["svd"],
                      "rotated": q["untrained"]["share"]["contrast"]["rotated"], "gain": q["gain"]["contrast"],
                      "neuron_k64": q["neuron"]["k64"]["share"]["contrast"], "neuron_kmatched": q["neuron"]["kmatched"]["share"]["contrast"],
                      "steer_pooled": q["steering"]["share"]["contrast"], "erank": q["erank"]}  # fmt: skip
    got, saved = pick(rec), pick(s)
    am = json.loads((RESULTS / "predictions_E223_amend1.json").read_text())  # 2026-11-20: relative bar, job 62141
    scale = {k: 1.0 if k == "share" else abs(saved["erank"] if k == "erank" else saved["gain"]) for k in got}
    rows = {k: {"got": got[k], "saved": saved[k], "tol": am["rel_tol"] * scale[k]} for k in got}
    o.put(f"reproduce L1_{L1_STATE}", rows)
    if bad := [k for k, v in rows.items() if not abs(v["got"] - v["saved"]) <= v["tol"]]:
        raise o.stop(f"L1_{L1_STATE} does not reproduce on the pair score: {bad}")


def e53_beside(x: Any, C2: Any, o: Any) -> None:
    """E53's lowercase seed-0 state on its case-pair contrast (E53's windows, lowercase form against capitalised)."""
    r, partner = R2.e53_read("lowercase"), case_partner(x.texts)
    keep, fkeep = partner[r.tgt] >= 0, partner[r.fit_tgt] >= 0
    pair = lambda t: torch.stack([t, partner[t]], 1)
    pr = RR.PairRead("e53_lowercase", r.toks[keep, : r.pos + 1], pair(r.tgt[keep]), r.pos, r.fit[fkeep, : r.pos + 1], pair(r.fit_tgt[fkeep]))
    rd = R2.reader(x.net, C2, pr)
    inv = json.loads((R2.E136 / "inventory.json").read_text())["states"]
    rec = rd.state({k: v.float() for k, v in torch.load(REPO / inv["e53/lowercase_s0"]["file"], weights_only=False).items()}, 0)
    o.put("e53 lowercase_s0 contrast (beside)", {"kept": int(keep.sum()), "of": len(keep), "fit_kept": int(fkeep.sum())} | R2.brief(rec))


def ctl(x: Any, C: Any, C2: Any, th2: dict[str, Any], o: Any) -> None:
    l1_reproduce(x, C2, th2, o)
    for p in ("1", "2"):
        rd, t0 = reader(x, C, C2, x.names[p], o), time.time()
        R2.zero_check(o, rd)
        o.d[f"zero {x.names[p]}"] = o.d.pop("zero")
        o.put(f"zero seconds {x.names[p]}", time.time() - t0)
    e53_beside(x, C2, o)


def write_state(name: str, rec: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.tmp"
    tmp.write_text(json.dumps(jsonable(rec), indent=1))
    tmp.replace(OUT / f"{name}.json")
    log(f"[state {name}] {json.dumps(jsonable(R2.brief(rec) | {k: rec[k] for k in ('arm', 'seed', 'seconds', 'e220')}))}")


def behaviour(x: Any, C: Any, C2: Any, p: str, o: Any) -> None:
    done = RESULTS / "contrast_ctl.json"
    if not (done.is_file() and json.loads(done.read_text()).get("done")):
        raise o.stop("contrast_ctl.json is not done: run ctl first and alone")
    name = x.names[p]
    rd = reader(x, C, C2, name, o)
    for i, (arm, seed) in enumerate((a, s) for a in ARMS for s in C.seeds):
        dW, meta = S3.load_state(p, arm, seed, o)
        if i == 0:  # the do-nothing control: a Gaussian update at this state's per-matrix norms
            g = rd.state(FT.random_direction(dW, C.gauss_draw), C.gauss_draw)
            o.put("gauss", g, shown=R2.brief(g))
            if not g["gain"]["contrast"] < MIN_GAIN:
                raise o.stop(f"a Gaussian update gains {g['gain']['contrast']:.3f} in the contrast on {name}, at or over {MIN_GAIN}")
        t0 = time.time()
        rec = {"behaviour": name, "arm": arm, "seed": seed, **meta, **rd.state(dW, seed)}
        write_state(f"{name}_{arm}_s{seed}", rec | {"seconds": time.time() - t0})
        o.d.setdefault("states", {})[f"{name}_{arm}_s{seed}"] = meta["file"]
        o.write()


def main() -> int:
    job = sys.argv[1] if len(sys.argv) == 2 else ""
    if job not in JOBS:
        sys.exit(f"usage: run_contrast.py {{{' | '.join(JOBS)}}}")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E223")
    P2 = json.loads(R2.PRED.read_text())  # E211's filed reader settings and reproduce tolerances
    C, C2 = config_from_settings(S3.Config, P["settings"]), config_from_settings(R2.Config, P2["settings"])
    x = S2.Ctx()  # E220's: the model (tf32 off), tokenizer, stage 1's records and held-out samples, Pile rows
    o = R2.Out(job)
    o.path, o.d["experiment"] = RESULTS / f"contrast_{job}.json", P["experiment"]
    try:
        ctl(x, C, C2, P2["thresholds"], o) if job == "ctl" else behaviour(x, C, C2, job, o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    print(f"STAGE 3b {job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
