"""Write the fixed token rows that the JAX reference and the torch port both read (plan
reports/plans/2026-10-27_plan_qwen-port-and-baseline.md, item 3).

Eight rows, each one document starting at position 0: four generic text rows of exactly 512 tokens (the first four
FineWeb-Edu sample-10BT documents at least 512 tokens long, cut to 512) and four GSM8K test prompts (the first four
test questions, with the GSM8K baseline's instruction, Qwen3 chat template, enable_thinking=False, generation prompt
added, no response). GSM8K rows are stored unpadded; a consumer that needs fixed shapes pads them itself, and only
positions below each row's length are read.

Output: results/tokens.json = {"seq_len": 512, "rows": [[int, ...] x 8], "kinds": ["text" x 4, "gsm8k" x 4],
"sources": {...}}.

Usage (repo root; commute venv, which has jinja2 for the chat template; CPU job with network):
  HF_HUB_CACHE=$HOME/.cache/huggingface/hub ~/commute-agents/fleet/cpu-run -c 2 --mem 8G -t 20 \
    .venv/bin/python experiments/qwen_port/make_tokens.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

OUT = Path(__file__).parent / "results" / "tokens.json"
SEQ_LEN, N_TEXT, N_GSM8K = 512, 4, 4
MODEL, MODEL_REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEXT, TEXT_CONFIG = "HuggingFaceFW/fineweb-edu", "sample-10BT"
GSM8K, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
# Same instruction as experiments/qwen_gsm8k_baseline/run.py (copied: that script runs in the commute venv).
INSTRUCTION = ("Solve the problem step by step. Then end your response with one final line of exactly this form: "
               "The answer is N.\nwhere N is the final numerical answer, written as a plain number.")


def text_rows(tok, revision: str) -> list[list[int]]:
    rows = []
    for doc in load_dataset(TEXT, TEXT_CONFIG, split="train", streaming=True, revision=revision):
        ids = tok(doc["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= SEQ_LEN:
            rows.append(ids[:SEQ_LEN])
        if len(rows) == N_TEXT:
            return rows
    raise AssertionError("ran out of documents")


def gsm8k_rows(tok) -> list[list[int]]:
    test = load_dataset(GSM8K, "main", split="test", revision=GSM8K_REV)
    rows = []
    for i in range(N_GSM8K):
        msgs = [{"role": "user", "content": f"{test[i]['question']}\n\n{INSTRUCTION}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append(tok(text, add_special_tokens=False)["input_ids"])
        assert len(rows[-1]) <= SEQ_LEN, len(rows[-1])
    return rows


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    text_rev = HfApi().dataset_info(TEXT).sha
    rows = text_rows(tok, text_rev) + gsm8k_rows(tok)
    out = {
        "seq_len": SEQ_LEN,
        "rows": rows,
        "kinds": ["text"] * N_TEXT + ["gsm8k"] * N_GSM8K,
        "sources": {
            "tokenizer": f"{MODEL}@{MODEL_REV}",
            "text": f"{TEXT}/{TEXT_CONFIG}@{text_rev}: first {N_TEXT} train documents with >= {SEQ_LEN} tokens, "
                    f"first {SEQ_LEN} tokens, no special tokens added",
            "gsm8k": f"{GSM8K}/main@{GSM8K_REV}: test questions 0-{N_GSM8K - 1} + the baseline's instruction, "
                     "chat template, enable_thinking=False, add_generation_prompt=True, unpadded",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print("lengths", [len(r) for r in rows])
    print("FINISHED", OUT)


if __name__ == "__main__":
    main()
