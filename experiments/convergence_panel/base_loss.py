"""E272 post-hoc check, a data-only baseline (plan reports/plans/2026-11-22_plan_convergence-panel.md, "Post-hoc check:
data-only baseline", filed 2026-11-23 before computing; descriptive, no call). Base model M's mean loss on each wide KL
"hi" task's held rows at every scored token, nats per token and nats per UTF-8 byte, under two token rules: "trained"
(the tokens the fine-tune trains on) and "all" (every non-padding token, plus the trained ones). Control: each task's
`rows.w0_logp` of results/train/TASK.json re-read on the train job's own held_mask (difference 0 to float noise).
-> results/wide/base_loss.json; score.py reads it into summary.json's `wide.base_loss`. A rerun skips written tasks.

    sbatch --comment="E272 stage: base loss" cluster/convergence_panel.sbatch base_loss
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor
from transformers.convert_slow_tokenizer import bytes_to_unicode

from commute import panel as PN
from commute import steering as ST
from commute import vpd
from commute.jobs import hard_exit, load_experiment, write_stage

RUN: Any = load_experiment("convergence_panel", "run")
OUT = RUN.RESULTS / "wide" / "base_loss.json"
LM = ("arxiv", "french", "german", "spanish", "swahili", "yoruba")  # Amendment 7: held_mask is the read positions only


def byte_lengths(tok: Any) -> Tensor:
    """UTF-8 bytes of each token id (byte-level BPE pieces through GPT-2's byte map; an added token's own text; 0 for
    special tokens such as end-of-text)."""
    dec, special = {c: b for b, c in bytes_to_unicode().items()}, set(tok.all_special_ids)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    n = [0 if i in special else len(bytes(dec[c] for c in s)) if all(c in dec for c in s) else len(s.encode()) for i, s in enumerate(pieces)]
    return torch.tensor(n, dtype=torch.long)


def masks(C: Any, name: str, t: PN.Task, eos: int) -> dict[str, Tensor]:
    """The two token rules on the held rows. Continued text trains every token after position 0 (french after its
    English prefix); lower's held_mask is already every token; prompt and label tasks' held_mask is what they train."""
    trained = PN.full_mask(t.held, C.en_len if name == "french" else 1) if name in LM else t.held_mask
    return {"trained": trained, "all": (PN.full_mask(t.held) & (t.held != eos)) | trained}


def loss(net: ST.Net, held: Tensor, mask: Tensor, nbytes: Tensor) -> dict[str, float]:
    n, nats = int(mask[:, 1:].sum()), -PN.target_logp(net, held, mask, None)
    b = int(nbytes[held[:, 1:][mask[:, 1:]]].sum())
    return {"tokens": n, "bytes": b, "nats_per_token": nats, "nats_per_byte": nats * n / b}


def main() -> int:
    C, _ = RUN.load()
    names = json.loads((RUN.RESULTS / "wide" / "dropped.json").read_text())["kept"]["kl_hi"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, tok = ST.Net("cuda"), vpd.tokenizer()
    nbytes = byte_lengths(tok)
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": "E272", "card": torch.cuda.get_device_name(), "tasks": {}}
    for name in (n for n in names if n not in out["tasks"]):
        t = RUN.TK.build(net, C, name)
        rec: dict[str, Any] = {k: loss(net, t.held, m, nbytes) for k, m in masks(C, name, t, tok.eos_token_id).items()}
        saved = json.loads((RUN.RESULTS / "train" / f"{name}.json").read_text())["rows"]["w0_logp"]
        rec["control"] = {"saved_w0_logp": saved, "diff": PN.target_logp(net, t.held, t.held_mask, None) - saved}
        out["tasks"][name] = rec
        write_stage(OUT, out)
        RUN.log(f"[base_loss {name}] {json.dumps(rec)}")
    print(f"STAGE base_loss DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
