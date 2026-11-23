"""Head-4 weight share of every layer-2 q/k/v/o subcomponent (rows 512:640 of U for q/k/v, rows 512:640 of V for o)."""
import json, sys
import numpy as np
from commute import vpd
W0, uv = vpd.checkpoint_weights()
out = {}
for m in "qkvo":
    k = f"h.2.attn.{m}_proj"; U, V = uv[k][0].double(), uv[k][1].double()
    x = V.T if m == "o" else U
    s = ((x[:, 512:640]**2).sum(1) / (x**2).sum(1)).numpy()
    out[m] = {"n": len(s), "median": float(np.median(s)), "max": float(s.max()), "over_0.3": int((s > 0.3).sum()), "over_0.5": int((s > 0.5).sum())}
out["max_all"] = max(v["max"] for v in out.values())
out["median_all"] = float(np.median(np.concatenate([((uv[f"h.2.attn.{m}_proj"][1].double().T if m == "o" else uv[f"h.2.attn.{m}_proj"][0].double())[:, 512:640]**2).sum(1).numpy() / ((uv[f"h.2.attn.{m}_proj"][1].double().T if m == "o" else uv[f"h.2.attn.{m}_proj"][0].double())**2).sum(1).numpy() for m in "qkvo"])))
open(sys.argv[1], "w").write(json.dumps(out, indent=1)); print(out)
