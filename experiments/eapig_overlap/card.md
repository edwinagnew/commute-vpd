cell: P2.69
rival_a: The reuse reading (an update that strengthens parts the original model already has leaves its circuit in place) predicts the trained copy-like tasks (E62 copy, key-grid copy) keep the base circuit: their edge IoU with M's 5% circuit near the x2 control's, inside M's resampling range
rival_b: The new reading (an update that adds a lookup adds wiring) predicts the trained tasks' edge IoU with M's circuit below M's resampling range on their windows, as the built new edits (new_few, new_thin) are filed to read
rung: strategy
smallest_gap: 0.03
spread: 0.02
n: 44
design: unpaired
unit: updates (7 built and matched-random items, 7 E62 states and random, 15 key-grid states and 15 randoms), each one F against M on its own windows
spread_source: reports/plans/2026-11-23_plan_p2-69-eap-ig-overlap.md
spread_of: proxy: Wang et al.'s Fig. 2 right, edge Jaccard of a circuit against itself on perturbed data, 0.98 to 0.94 for Pythia (about 0.02 per step); the same reading (edge IoU of 5% circuits) on another model; this run measures ours per task set (M against M, 5 full-size resamples)
floor: none (descriptive); every IoU is reported beside M's resampling range on its task set, and only items whose two circuits are each at least 70% faithful (Wang's formula) enter the headline
stop: control 1 (x2's edge set equal to M's, exactly) and control 2 (new_few and new_thin under M's resampling range) both miss: the method cannot tell strengthening from new wiring on built edits, the values are reported as those of a method that failed its controls, and no further EAP-IG overlap work is proposed
sketch: experiments/eapig_overlap/figures/sketch.png
rehearsal: cluster only: the whole run is cluster debug at $0
descriptive: user 2026-11-23 via whip-lead ("yes also to 1 and 2"), a published baseline reported as a description

What runs (spec `reports/plans/2026-11-23_plan_p2-69-eap-ig-overlap.md`, revision 1, and its appendix): Wang et al.
2025's circuit overlap before against after an update, with their pinned code (FinetuneCircuits `fdaeeaac63d5`'s `eap/`
tree, copied to node-local scratch with every file's sha256 checked, and their notebook's KL and faithfulness functions
copied verbatim into `wang_notebook.py`), on our 67M converted to a TransformerLens model from each item's full native
state. Stage 0 checks the adapter on M and x2 and is the only stop; the controls are reported, not stops.

Smallest gap: one moved edge changes an IoU of 46-edge circuits by about 0.02 to 0.04 (spec, "Why x2 may miss"), so
counts go beside every IoU.

Departure not in the spec's list (writer, 2026-11-23): pygraphviz 1.13 (FC's pin) ships as a source archive only and needs
the graphviz headers, which the cluster lacks. It is not installed; `pygraphviz_stub.py` is staged as `pygraphviz` beside
`eap/`, because `eap/graph.py` imports it at its top for `Graph.to_graphviz` only, which nothing here calls. Every other
pin of FC's `requirements.txt` is installed as filed (`requirements-fc.txt`).

Run (one debug job after the venv job):

    sbatch --comment='E___ venv' cluster/eapig_overlap_venv.sbatch
    sbatch --comment='E___ eapig_overlap' cluster/eapig_overlap.sbatch
