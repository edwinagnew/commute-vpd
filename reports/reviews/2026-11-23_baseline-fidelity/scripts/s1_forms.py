import json, statistics as S
from pathlib import Path
R = Path("/mnt/nw/home/e.agnew/commute-agents/redteam-lead/experiments")
st = json.loads((R/"prior_steering/results/stage1.json").read_text())
sc = st["score"]["pairs"]; states = st["states"]
inv = {p["id"]:p for p in json.loads((R/"share_untrained/results/inventory.json").read_text())["pairs"]}
i2 = json.loads((R/"refit_96key/results/inventory.json").read_text())
mem = {}
for p,v in inv.items(): mem[p]=(v["reuse"],v["new"])
for p in i2["pairs"]: mem[p["id"]]=([f"e58/{p['reuse']}_s{s}" for s in i2["seeds"]],[f"e58/{p['new']}_s{s}" for s in i2["seeds"]])
print(sorted(states)[:50])
for p in sc:
    out={}
    for k in ("pooled","perkey"):
        try:
            a=[states[n][k]["share"] for n in mem[p][0]]; b=[states[n][k]["share"] for n in mem[p][1]]
        except KeyError as e:
            # e62 names may differ
            out[k]=f"missing {e}"; continue
        if None in a+b: out[k]=None; continue
        gap=S.fmean(a)-S.fmean(b); ap = min(a)>max(b) or max(a)<min(b)
        out[k]=(round(gap,3),"apart" if ap else "nocall")
    print(p, sc[p]["steering"]["filed"]["kind"], out)
