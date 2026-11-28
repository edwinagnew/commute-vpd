# definition-lead digest, 2026-11-27

Fig 2 now reads all sixteen known-answer tasks at all three featurizations, and it is in the paper. Fine-tunes keep a few hundredths of their write in the needed set, above random updates of the same size, but tasks known to reuse old parts and tasks known to be new overlap, so this share does not sort them. A bug in the top-neuron swap was found and fixed on the way; it moved no reading visibly.

**Cells.** P1.66 (Fig 2 on all 16 known-answer tasks, due 11-28): filled, 3 of 3 runs. P1.65 (paper numbers re-read over the needed set, rescale-only control at three seeds): filled earlier today, see below. No other open cell of mine; whip-lead's schedule has none for me, so I finish.

**Earlier today, P1.65 (filled; pods $9.2 + $8.3).** The graded-retained score still reads 0.999 or more on all 20 fine-tunes; line 438's SAE error share becomes 0.78 (was 0.83), its margin 0.104 still under the 0.2 bar. The rescale-only control reads close to the full fine-tunes at VPD in no consistent order, and two to six times above them at SAE. Premise check doubtful: whether the reading sees reuse depends on the featurization (~/commute-data/premise/P1.65.md). Page: reports/analyses/2026-11-26_analysis_p1-65-readthrough-fills.md.

**What ran.**
- E326 rerun 1 ($0.49): confirmed the cause of the top-neuron identity miss (0.027 and 0.069 against ~8e-5): the old swap hook split activations at their bound with the rerun's copy. Fixed in d2d81436; the identity now reads at most 7.8e-5.
- E326 rerun 2 ($5.02): all 72 top-neuron items reread under the fix; shares moved by at most 3.8e-6. The e58 reproduction of E298 moved by one needed neuron of 4171; per your answer to ask 60 it is reported as the fix's effect.
- Score and three-panel figure on a short pod (under $0.30). redteam-lead passed the rows and the page (one MUST fixed: "no higher" became "overlap").
- Page: reports/analyses/2026-11-27_analysis_p1-66-fig2-all-tasks.md. Rows in the paper via writeup-lead (9fe7ae3).

**For you.** The premise check read "doubtful": since the retained share does not separate reuse from new on the 16 tasks, what role does it play in the definition? whip-lead is putting this to you (~/commute-data/premise/P1.66.md). No asks of mine are open.

**Money.** $5.07 this session on RunPod for P1.66, $17.5 earlier today for P1.65 (status() shows no machines). Nothing in blocked.txt.

**Friction noted.** cpu-run refused Python as "node completing" while sinfo showed the node up (1215); I edited a page while a resumed writer agent was editing it, duplicating a sentence (1216); the runner brief did not say how interleaved VPD lists split (1211); plot.py had to ride a pod (1212).

## Not run
- The 24 e58 top-neuron rows under the fixed hook (they stay on the old hook, about 3.4e-4 relative; whip-lead: skip).
- E277's and E298's top-neuron needed sets under the fixed hook.
- Marking the old-hook e58 rows in the figure (redteam SHOULD 5; whip-lead: skip).
