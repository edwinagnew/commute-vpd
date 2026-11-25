import json, math
from commute.jobs import load_experiment
RUN = load_experiment("convergence_panel", "run"); SC = load_experiment("convergence_panel", "score")
P = json.loads(RUN.PRED.read_text()); R = P["rules"]
tier = SC.raters(P, "raters_wide")["consensus"]
m12 = json.loads((RUN.RESULTS/"summary.json").read_text())["wide"]["kl_hi"]["means"]
n10 = json.loads((RUN.RESULTS/"wide/run4/score_n10.json").read_text())["per_task"]
def go(name, er, eb, pr):
    h = SC.hope1(tier, er, pr, R)
    c = h if "tau" in h else h.get("consensus", h)
    print(name, "erank tau %.3f partial %.3f p_partial %.3f" % (c["tau"], c.get("tau_partial_prior", math.nan), c.get("p_partial", math.nan)))
    import scipy.stats as st
    ts = sorted(eb); print("   E_b tau", round(st.kendalltau([tier[t] for t in ts],[eb[t] for t in ts]).statistic,3), "n", len(ts))
base_er, base_eb, base_pr = m12["erank"], m12["E_b"], m12["prior"]
def sub(d, drop, upd=None): x = {k:v for k,v in d.items() if k not in drop}; x.update(upd or {}); return x
print("tiers", {t: tier.get(t) for t in base_er})
go("12 run3", base_er, base_eb, base_pr)
go("12 - arxiv", sub(base_er,["arxiv"]), sub(base_eb,["arxiv"]), sub(base_pr,["arxiv"]))
go("12 - german", sub(base_er,["german"]), sub(base_eb,["german"]), sub(base_pr,["german"]))
go("12 - both (old sentiment/spanish)", sub(base_er,["arxiv","german"]), sub(base_eb,["arxiv","german"]), sub(base_pr,["arxiv","german"]))
new = {t: n10[t]["erank"]["mean"] for t in n10}; neb = {t: n10[t]["E_b"]["mean"] for t in n10}; npr = {t: n10[t]["prior"]["mean"] for t in n10}
go("10 n10", new, neb, npr)
go("10 - sentiment", sub(new,["sentiment"]), sub(neb,["sentiment"]), sub(npr,["sentiment"]))
