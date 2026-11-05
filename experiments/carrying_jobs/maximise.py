"""carrying_jobs (E101), stage 1's scoring job: the essay's test "Explanations That Could Lose". Plan:
reports/plans/2026-11-05_plan_e101-carrying-jobs.md, "Stage 1". Run as `sbatch cluster/carrying_jobs.sbatch maximise`.

  C2 self test, before any agent text: per pool neuron, its own Pile windows ranked 2-6 (decoded, re-encoded) as examples,
         the windows at the same ranks of 5 other pool neurons as near misses; win share >= self_test_min_win (exit 1)
  agent texts: results/stage1/maximiser/b<NN>.json (brief maximiser.md): each text's peak W0 firing of its neuron
  win rule: `commute.carrying_jobs.outcome` (strong = strong_frac of the neuron's rank-1 Pile window; more strong
         examples than strong near misses wins)
Reads large/key.json and large/pile_top.pt (stage 0). Writes results/stage1/max_scores.json, the file score.py reads:
{"self_test", "neurons": {id: {"examples": [peak], "near_misses": [peak], "outcome"}}, "missing_batches", "dropped"}.
Every score is printed. A missing batch file or a dropped neuron has no entry: the scorer reads it as unscorable.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

from commute import carrying_jobs as CJ
from commute import keydep as KD
from commute import steering as ST
from commute import vpd
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, STAGE1 = RESULTS / "large", RESULTS / "stage1"
RUN = load_experiment("carrying_jobs", "run")  # its `firing`: W0's GELU outputs, flat index l x 3072 + i
log = KD.log
MAX_TOKENS, BATCH, SELF_OTHERS = 256, 16, 5


@torch.no_grad()
def peaks(net: ST.Net, seqs: list[list[int]], flats: list[int]) -> list[float]:
    """Per sequence, the max over its tokens of W0's firing of neuron flats[k]. Right-padded batches; causal, so padding
    cannot change a real token's firing, and padded positions are masked out of the max."""
    out: list[float] = []
    for s in range(0, len(seqs), BATCH):
        chunk, fl = seqs[s : s + BATCH], torch.tensor(flats[s : s + BATCH])
        n = max(len(q) for q in chunk)
        toks = torch.tensor([q + [0] * (n - len(q)) for q in chunk], device=RUN.DEVICE)
        f = RUN.firing(net, toks).cpu()  # (B, n, 12288)
        v = f[torch.arange(len(chunk))[:, None], torch.arange(n)[None, :], fl[:, None]]
        v[torch.arange(n)[None, :] >= torch.tensor([len(q) for q in chunk])[:, None]] = -torch.inf
        out += v.max(dim=1).values.tolist()
    return out


def own_windows(key: dict[str, Any], top: dict[str, Any], rows: torch.Tensor, tok: Any, ctx: int) -> dict[str, list[list[int]]]:
    """id -> its Pile windows ranked 2-6 as token ids: the ±ctx slice around the peak, decoded as text and encoded again."""
    out = {}
    for i, k in key["neurons"].items():
        wins = []
        for r in range(1, 1 + 5):
            row, pos = rows[int(top["rows"][r, k["flat"]])], int(top["pos"][r, k["flat"]])
            text = tok.decode(row[max(0, pos - ctx) : pos + ctx + 1].tolist())
            wins.append(tok(text).input_ids[:MAX_TOKENS])
        out[i] = wins
    return out


def c2(net: ST.Net, key: dict[str, Any], wins: dict[str, list[list[int]]], S: dict[str, Any]) -> dict[str, Any]:
    rng, ids, seqs, flats = random.Random(S["seed"] + 4), sorted(key["neurons"]), [], []
    for i in ids:
        flat = key["neurons"][i]["flat"]
        others = rng.sample([j for j in ids if j != i], SELF_OTHERS)
        for r, w in enumerate(wins[i]):  # example r: own rank r+2; near miss r: another neuron's window of the same rank
            seqs += [w, wins[others[r]][r]]
            flats += [flat, flat]
    p = peaks(net, seqs, flats)
    outs = []
    for n, i in enumerate(ids):
        ex, nm = p[10 * n : 10 * n + 10 : 2], p[10 * n + 1 : 10 * n + 10 : 2]
        o = CJ.outcome(ex, nm, key["neurons"][i]["top_act"], S["strong_frac"])
        outs.append(o)
        log(f"[C2 {i}] top {key['neurons'][i]['top_act']:.4f} own {json.dumps([round(x, 4) for x in ex])} "
            f"other {json.dumps([round(x, 4) for x in nm])} -> {o}")
    st = CJ.self_test(outs, S["self_test_min_win"])
    log(f"[C2] {json.dumps(st)} (expect >= {S['self_test_min_win']}; filed 0.9 or more)")
    return st


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())
    S = P["settings"]
    key = json.loads((LARGE / "key.json").read_text())
    top = torch.load(LARGE / "pile_top.pt")
    rows = pile_rows(*S["pile_cache"], log=log)[slice(*top["window_rows"])]
    tok, net = vpd.tokenizer(), ST.Net(RUN.DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    STAGE1.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"self_test": c2(net, key, own_windows(key, top, rows, tok, S["context"]), S), "neurons": {}}
    if not out["self_test"]["ok"]:
        (STAGE1 / "max_scores.json").write_text(json.dumps(out, indent=1))
        log("FAIL: C2 below self_test_min_win: the text round trip or the 30% rule is broken; stop and tell the lead")
        sys.exit(1)
    texts, missing, bad = CJ.load_texts(STAGE1 / "maximiser", key["batches"], S["n_examples"], S["n_near_misses"])
    log(f"[texts] {len(texts)} neurons read, missing batch files {missing}, dropped ids {bad}")
    ids = sorted(texts)
    seqs, flats = [], []
    for i in ids:
        for t in texts[i]["examples"] + texts[i]["near_misses"]:
            seqs.append(tok(t).input_ids[:MAX_TOKENS])
            flats.append(key["neurons"][i]["flat"])
    p, ne = peaks(net, seqs, flats), S["n_examples"]
    for n, i in enumerate(ids):
        ex, nm = p[(ne + S["n_near_misses"]) * n :][:ne], p[(ne + S["n_near_misses"]) * n + ne :][: S["n_near_misses"]]
        o = CJ.outcome(ex, nm, key["neurons"][i]["top_act"], S["strong_frac"])
        out["neurons"][i] = {"examples": ex, "near_misses": nm, "outcome": o}
        log(f"[score {i}] top {key['neurons'][i]['top_act']:.4f} examples {json.dumps([round(x, 4) for x in ex])} "
            f"near {json.dumps([round(x, 4) for x in nm])} -> {o}")
    out |= {"missing_batches": missing, "dropped": bad}
    (STAGE1 / "max_scores.json").write_text(json.dumps(out, indent=1))
    log(f"[done] {len(ids)} neurons scored; outcomes {json.dumps(CJ.self_test([v['outcome'] for v in out['neurons'].values()], 0)['counts'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
