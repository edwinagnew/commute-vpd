import json, statistics as S
from pathlib import Path
R = Path("/mnt/nw/home/e.agnew/commute-agents/redteam-lead")
L = lambda p: json.loads((R/p).read_text())
def g(a,b):
    if None in a+b: return None
    return round(S.fmean(a)-S.fmean(b),4)
def call(a,b):
    if None in a+b: return "-"
    return "apart" if min(a)>max(b) or max(a)<min(b) else "nocall"
def show(pair, sides):
    print(pair, {k:(g(*v), call(*v), [round(x,3) if x is not None else None for x in v[0]], [round(x,3) if x is not None else None for x in v[1]]) for k,v in sides.items()})
st = L("experiments/steering_copy_lookup/results/stage1.json")["states"]
show("e62_copy_lookup", {k:([st[f"copy_s{i}"]["steering"][k]["share_lo"] for i in range(3)],[st[f"lookup_s{i}"]["steering"][k]["share_lo"] for i in range(3)]) for k in ("pooled","perkey")})
h = L("experiments/holes_fill/results/stage1.json"); inv={p["id"]:p for p in L("experiments/share_untrained/results/inventory.json")["pairs"]}
for pair in h["score"]["families"]["steering"]:
    get=lambda n,k: h["states"][n]["steering"].get(k,{}).get("share")
    show(pair, {k:([get(n,k) for n in inv[pair]["reuse"]],[get(n,k) for n in inv[pair]["new"]]) for k in ("pooled","perkey")})
r=L("experiments/refit_96key/results/stage1.json"); i2=L("experiments/refit_96key/results/inventory.json")
for p in i2["pairs"]:
    a=[f"e58/{p['reuse']}_s{s}" for s in i2["seeds"]]; b=[f"e58/{p['new']}_s{s}" for s in i2["seeds"]]
    show(p["id"], {k:([r["states"][n]["holes_fill"]["steering"][k]["share"] for n in a],[r["states"][n]["holes_fill"]["steering"][k]["share"] for n in b]) for k in ("pooled","perkey")})
t3=L("experiments/real_vs_twin/results/stage3.json")
for pair,w in {"e123_case_label":"case","e125_repeat_label":"repeat"}.items():
    arm=lambda x:[f"{w}_{x}_s{i}" for i in range(3)]
    show(pair, {k:([t3["states"][n]["steering"][k]["share"]["contrast"] for n in arm("real")],[t3["states"][n]["steering"][k]["share"]["contrast"] for n in arm("twin")]) for k in ("pooled","perkey")})
