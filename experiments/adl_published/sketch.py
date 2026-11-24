import sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pos=list(range(9)); fig,ax=plt.subplots(1,2,figsize=(8,3),sharey=True)
for a,(t,y) in zip(ax,[("rival a: reproduces",[.9,.8,.7,.6,.6,.5,.5,.5,.4]),("rival b: misses",[.05,-.02,.03,0,.01,-.01,.02,0,.01])]):
    a.plot(pos,y,"o-",label="3-organism mean (made up)"); a.axhline(0.05,ls="--",c="gray",label="random mean"); a.fill_between(pos,0.05-0.1,0.05+0.1,color="gray",alpha=.2)
    a.set_title(t); a.set_xlabel("position")
ax[0].set_ylabel("CE rise on training data (nats)"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(sys.argv[1],dpi=80)
