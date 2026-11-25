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
The test itself is `commute.carrying_jobs` (peaks, own_windows, self_test_run, text_scores), moved there unchanged for
E102 stage B (2026-11-05); this file as it ran is in git before that commit.
"""

from __future__ import annotations

import json
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
MAX_TOKENS, BATCH = 256, 16


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())
    S = P["settings"]
    key = json.loads((LARGE / "key.json").read_text())
    top = torch.load(LARGE / "pile_top.pt")
    rows = pile_rows(*S["pile_cache"], log=log)[slice(*top["window_rows"])]
    tok, net = vpd.tokenizer(), ST.Net(RUN.DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    fire = lambda toks: RUN.firing(net, toks)
    encode = lambda text: tok(text).input_ids
    STAGE1.mkdir(parents=True, exist_ok=True)
    flats = {i: k["flat"] for i, k in key["neurons"].items()}
    wins = CJ.own_windows(flats, top, rows, tok.decode, encode, S["context"], MAX_TOKENS)
    st = CJ.self_test_run(fire, key["neurons"], wins, S["seed"] + 4, S["strong_frac"], S["self_test_min_win"], BATCH, RUN.DEVICE, log)
    out: dict[str, Any] = {"self_test": st, "neurons": {}}
    if not st["ok"]:
        (STAGE1 / "max_scores.json").write_text(json.dumps(out, indent=1))
        log("FAIL: C2 below self_test_min_win: the text round trip or the 30% rule is broken; stop and tell the lead")
        sys.exit(1)
    texts, missing, bad = CJ.load_texts(STAGE1 / "maximiser", key["batches"], S["n_examples"], S["n_near_misses"])
    log(f"[texts] {len(texts)} neurons read, missing batch files {missing}, dropped ids {bad}")
    out["neurons"] = CJ.text_scores(fire, key["neurons"], texts, encode, S["strong_frac"], MAX_TOKENS, BATCH, RUN.DEVICE, log)
    out |= {"missing_batches": missing, "dropped": bad}
    (STAGE1 / "max_scores.json").write_text(json.dumps(out, indent=1))
    log(f"[done] {len(out['neurons'])} neurons scored; outcomes {json.dumps(CJ.self_test([v['outcome'] for v in out['neurons'].values()], 0)['counts'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
