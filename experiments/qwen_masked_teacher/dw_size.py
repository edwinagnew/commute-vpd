"""δW's squared size per block, E76's (iii) denominator, which the run did not save: the arms' `blocks` need it to read as a size, not a share."""
from __future__ import annotations

import json

import torch
from common import DW_FILE, RESULTS
from safetensors import safe_open

from commute.qwen_vpd import SITES, parse_site

out = [0.0] * 28
with safe_open(str(DW_FILE), "pt") as f:
    for s in SITES:
        out[parse_site(s)[0]] += float(f.get_tensor(s).float().pow(2).sum(dtype=torch.float64))
(RESULTS / "dw_blocks.json").write_text(json.dumps({"blocks": out, "sum": sum(out), "late": sum(out[21:])}) + "\n")
print(sum(out), sum(out[21:]))
