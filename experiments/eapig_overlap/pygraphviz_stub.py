"""Stand-in for pygraphviz, staged as `pygraphviz.py` beside FC's eap/ (run.py `stage_tree`). FC's eap/graph.py imports
pygraphviz at its top for `Graph.to_graphviz` only, which P2.69 never calls. pygraphviz 1.13 (FC's pin) ships as a source
archive only and needs the graphviz headers to build, which the cluster lacks; so nothing is installed and any use of
this module raises (spec departure noted in experiments/eapig_overlap/card.md)."""


def __getattr__(name: str) -> object:
    raise ImportError(f"pygraphviz.{name}: pygraphviz is not installed in the P2.69 venv (graph drawing is not used)")
