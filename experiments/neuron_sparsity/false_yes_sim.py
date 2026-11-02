"""The filed separation rule's false-yes rate and power, as amended (Amendment 12), without its reference rule and guard.
Each test is one task against one task. Task means ~ N(0, tau^2), tau = 10 seed-noise SDs, the scenario of
predictions.json reading_notes.one_task_tests; a kind effect moves elicit towards sparser. Pure computation, no results.

Output, seed 630, 400,000 draws (per number; "any of five" assumes independent numbers, which they are not):
  kind 0 task SD: T2 follows T1 0.087 (any of five 0.367); T2 a third task 0.056 (any of five 0.251)
  kind 1 task SD: T2 follows T1 0.363; T2 a third task 0.297
  kind 2 task SD: T2 follows T1 0.724; T2 a third task 0.672
  family B pooled as first filed, T2 follows T1, kind 0: 0.017 (any of five 0.084), the 2% and 9% filed before Amendment 12

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 3G python experiments/neuron_sparsity/false_yes_sim.py
"""

import numpy as np

rng=np.random.default_rng(630)
N=400000; tau=10.0
def side(ne,nn,kind,task_e=None,task_n=None):
    te=rng.normal(0,tau,N) if task_e is None else task_e
    tn=rng.normal(0,tau,N) if task_n is None else task_n
    e=te-kind+rng.normal(0,1,(ne,N)); n=tn+rng.normal(0,1,(nn,N))     # lower = sparser
    gap=n.mean(0)-e.mean(0); sp=np.maximum(np.ptp(e,axis=0),np.ptp(n,axis=0))
    return (gap>0)&(gap>sp), te
for kind in (0.0,tau,2*tau):
    t1,te=side(5,5,kind)
    t2_follow=t1
    t2_third,_=side(2,2,kind,task_e=te)       # same elicit task, third (new-side) task
    t3b,_=side(2,2,kind); t3c,_=side(2,2,kind)
    a=(t1&t2_follow&t3b&t3c).mean(); b=(t1&t2_third&t3b&t3c).mean()
    print(f"kind={kind/tau:.0f} task SD: T2 follows T1 {a:.3f} (any of five {1-(1-a)**5:.3f}); T2 third task {b:.3f} (any of five {1-(1-b)**5:.3f}); single test {t1.mean():.3f}")
# first-filed: family B pooled, 4 per side from two tables per kind, range includes between-task
te1,te2,tn1,tn2=[rng.normal(0,tau,N) for _ in range(4)]
e=np.stack([te1+rng.normal(0,1,N),te1+rng.normal(0,1,N),te2+rng.normal(0,1,N),te2+rng.normal(0,1,N)])
n=np.stack([tn1+rng.normal(0,1,N),tn1+rng.normal(0,1,N),tn2+rng.normal(0,1,N),tn2+rng.normal(0,1,N)])
gap=n.mean(0)-e.mean(0);sp=np.maximum(np.ptp(e,axis=0),np.ptp(n,axis=0));tb=(gap>0)&(gap>sp)
t1,_=side(5,5,0.0);tc,_=side(2,2,0.0);x=(t1&tb&tc).mean();print(f"first-filed pooled B, T2 follows: {x:.3f}, any of five {1-(1-x)**5:.3f}")
