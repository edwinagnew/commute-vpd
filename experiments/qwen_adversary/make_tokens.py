"""Write the 128 generic text rows the adversary run reads (plan reports/plans/2026-10-27_plan_qwen-adversary-128.md).

Rows: `commute.qwen_data.fineweb_rows` at its pinned revision; the first four must equal the text rows of
experiments/qwen_port/results/tokens.json (the 4-row readings the control reproduces), which is asserted.

Output: results/tokens.json = {"seq_len": 512, "rows": [[int] x 128], "source": "..."}.

Usage (repo root; CPU job with network):
  HF_HUB_CACHE=$HOME/.cache/huggingface/hub ~/commute-agents/fleet/cpu-run -c 2 --mem 8G -t 20 \
    .venv/bin/python experiments/qwen_adversary/make_tokens.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from commute.qwen_data import (
    FINEWEB,
    FINEWEB_CONFIG,
    FINEWEB_REV,
    SEQ_LEN,
    fineweb_rows,
)
from commute.qwen_vpd import HF_NAME, HF_REVISION

HERE = Path(__file__).parent
OUT, PORT_TOKENS = HERE / "results" / "tokens.json", HERE.parent / "qwen_port" / "results" / "tokens.json"
N_ROWS = 128  # the authors' eval batch (launch_config.yaml eval.batch_size)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(HF_NAME, revision=HF_REVISION)
    rows = fineweb_rows(tok, N_ROWS)
    port = json.loads(PORT_TOKENS.read_text())
    port_text = [r for r, k in zip(port["rows"], port["kinds"], strict=True) if k == "text"]
    assert rows[: len(port_text)] == port_text, "first rows differ from qwen_port's text rows"
    source = (f"{FINEWEB}/{FINEWEB_CONFIG}@{FINEWEB_REV}: first {N_ROWS} train documents with >= {SEQ_LEN} tokens, "
              f"first {SEQ_LEN} tokens, no special tokens; tokenizer {HF_NAME}@{HF_REVISION}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seq_len": SEQ_LEN, "rows": rows, "source": source}))
    print("FINISHED", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
