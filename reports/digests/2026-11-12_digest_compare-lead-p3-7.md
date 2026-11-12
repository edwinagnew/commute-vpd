# compare-lead digest 2026-11-12 (afternoon): P3.7 filled

P3.7, due 11-14: filled, 1 of 2 runs used (E169). We read five cheap readings on qwen-lead's Qwen3-1.7B pair library, side A minus side B, at the late save and at the matched-size save, each beside a random-change floor. Top-neuron overlap and effective rank separate the two sides of the built pairs beyond their random bands, but every passing overlap gap coincides with side A being the smaller weight change, so none is told apart from update size. No reuse or new label is given; this is a description.

## Cells
All compare-lead cells are filled or closed. P3.7 filled (1/2 runs). L2.2 waits on L1.3 and L2.4 on L1.5 (definition-lead's part 1).

## What ran
E169 on three pods in parallel (RunPod H100 SXM, Vast H100 PCIe, Vast A100), 70-80 min each, about $20 in all, every pod terminated with its results home and sha256-checked. The do-nothing check read exactly on all four pairs; all 42 saved updates matched their sha256 and weight-change norm.
Page: `reports/analyses/2026-11-12_analysis_p3-7-library-cheap-readings.md`; figure: `experiments/qwen_library_readings/figures/scorecard.png`. Research log v331 published.

## For the 11-18 gate
- The premise check on P3.7 was doubtful (`~/commute-data/premise/P3.7.md`): the library's pair sides differ systematically in weight change, so a side A minus side B reading mostly measures update size. P3.6's premise check pointed the same way. A comparison matched in weight change would need a new cell.
- The steering reading has a random band on the gold twin only; on the built pairs and the template twin it is unreadable.

## Asks waiting on the user
None.

## Friction noted today
748 (auto-mode classifier blocked `fleet/cell add`), 753 (pod labels cannot hold spaces or colons), 756 (`pod_setup.sh` `git init` without a token breaks `require_filed`), 757 (`download()` of a directory flattens into its target).
