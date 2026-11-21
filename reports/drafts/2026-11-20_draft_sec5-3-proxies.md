# Draft: §5.3 "Proxies against the baselines and the definition", two versions (writeup-lead, 2026-11-20)

Two alternative LaTeX drafts of one Results subsection of "A Mechanistic Foundation for the Elicitation Hypothesis".
Not in `main.tex`; the user picks one when the reference and the scoring land (~11-22 to 11-24). Both use the paper's
`\open` macro and a boxed placeholder for the planned figure. Revised after the user's 20:55 message (the comparison
with the baselines, crosscoder included, is part of the main story; the comparison with $R$ under wording A is one part
of the evidence; size, prior and random changes are checks, not the frame) and whip-lead's 21:10 steer (Version A
framed as "the share and the base prior agree with the definition on trained built tasks; here is where each fails",
one baselines row per method; the crosscoder row an `\open` gap naming its update-size-ladder rerun).

Wording A (decisions.md row "Definition wording A adopted", 2026-11-20): a fine-tune reuses the base model, for a
behaviour, to the extent that the base model's mechanisms, each up to an output gain, can stand in for those of the
variables the behaviour runs through and still preserve the behaviour (soft intervention).

---

## Version A: the hoped-for version

**Right when:** $R$ under wording A passes its known-answer control (cell P1.46: every built reuse edit above both
built new edits by more than its band, random changes low), and on the trained built tasks (nine pairs at matched size,
five key tasks at one size) the re-gating share and the base prior each rank the fine-tunes as $R$ does, beyond the
matched random band (compare-lead scoring).

```latex
\subsection{Proxies against the baselines and the definition}\label{sec:proxies}

\paragraph{Two cheap readings agree with the definition on trained built tasks.} We read $R$ under the adopted
wording at the VPD featurization: a variable counts as kept when the base model's mechanism, up to one output gain,
can stand in for the fine-tune's and the behaviour survives. On the nine built pairs at matched size and the five key
tasks at one size, the re-gating share and the base model's prior on the trained answer both rank the fine-tunes as
$R$ does \open{rank correlation of each with $R$ under wording A, and the band from matched random changes; cell P1.46
for $R$, compare-lead scoring}. The share sets all nine pairs apart in the built direction, and keeps all nine after
size matching. The prior points the built way on all nine, but with one value a side on eight it gives a direction,
not a call. The two are different readings: the share's gaps rank with the prior's at only 0.02.

\paragraph{Where each fails.} The prior fails where answers are equally likely. At one update size the share puts the
cipher whose answers the base model finds as likely as copy's 0.40 below copy, although the two priors differ only
1.6-fold \open{which side $R$ under wording A takes on this task; cell P1.46}. The share fails three checks. On the
twins, whose second side can only be memorised, it reads the real task higher on only one of four. On the hand-made
edits, neither of its forms put a reuse edit above a new edit in any of six comparisons. And its log-odds form has no
random reading, because no random change moves the trained answer by the one log-odds it needs.

\paragraph{The baselines.} Table~\ref{tab:proxies} gives one row per method. Update size never agrees with how the pairs
were built, and gain-only training reaches half the gain on both sides of four pairs. The neuron-level readings and
steering each reverse three to five pairs.

\paragraph{Caveats.} The evidence is one 67M model and tasks built for the purpose. At VPD, the definition's question,
whether base subcomponents up to a gain can stand in, is close to what the share computes. Agreement there is partly
by construction, and counts as independent only where $R$ read at a featurization not built from VPD agrees
\open{$R$ under wording A at SAE latents, cell P1.40, or at heads}. Over the five key tasks the share still ranks with
the prior at 0.9, so on most trained tasks the two cannot be told apart.

\begin{table}[t]
\centering\small
\caption{Each method on the nine built pairs, against $R$ under the adopted wording, and where it fails. Calls: pairs
apart with the side built to reuse higher, apart and lower, and no call (Table~\ref{tab:scorecard}).}
\label{tab:proxies}
\begin{tabular}{lccp{5.2cm}}
\toprule
method & calls & with $R$ & where it fails \\
\midrule
Re-gating share, log-odds & 9 / 0 / 0 & \open{P1.46} & twins; hand-made edits; no random reading \\
Base model's prior & 1 / 0 / 8 & \open{P1.46} & answers of equal prior; one value a side \\
Re-gating share, KL form & 9 / 0 / 0 & \open{P1.46} & random changes reach it on 67 of 88 \\
Top-$k$ gate & 6 / 1 / 2 & \open{P1.46} & loses the trained gate's choice \\
Neuron share, top-neuron share & 3--4 / 4--5 / 0--2 & \open{P1.46} & reverse four or five pairs \\
Gain-only reach & 3 of 9 & \open{P1.46} & new tasks reach 0.72 to 0.91 of their gain \\
Crosscoder & \open{P2.49 run 2} & \open{P2.49 run 2} & \open{rerun up an update-size ladder} \\
Update size & 0 / 6 / 3 & \open{P1.46} & the side built new moved further \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\fbox{\parbox{0.9\linewidth}{\centering\small\textbf{Planned figure: the share and the prior against the definition}}}
\caption{Do the share and the prior rank the trained built fine-tunes as the definition does? Each dot is one
fine-tune: a side of the nine built pairs at matched size, or a seed of the five key tasks at one size. Horizontal:
$R$ under the adopted wording at VPD. Vertical: the share (left) or the prior in decades (right). Grey crosses are
matched random changes, three per fine-tune; the starred dot is the cipher matched in prior to copy.}
\label{fig:proxies}
\end{figure}
\open{Fig.~\ref{fig:proxies} and the ``with $R$'' column of Table~\ref{tab:proxies} need cell P1.46 and compare-lead's
scoring; the crosscoder row waits for cell P2.49's second run.}
```

---

## Version B: the negative version

**Right when:** either no proxy ranks the fine-tunes as $R$ under wording A does once size and prior are partialled
out, beyond the matched random band (compare-lead scoring); or $R$ under wording A fails its known-answer control
(cell P1.46), so there is no reference. The third paragraph holds the second branch; keep it or delete it.

```latex
\subsection{Proxies against the baselines and the definition}\label{sec:proxies}

\paragraph{The share beats every baseline on the built pairs.} The re-gating share is the only reading that sets all
nine built pairs apart in the direction they were built, and it keeps all nine after the two sides' update sizes are
matched. On the three pairs whose sides already have equal sizes, its gaps run from 0.52 to 0.62. Update size never
agrees: wherever it calls a pair, the side built new has the larger update. Steering, the whole-width neuron share and
the top-neuron share each reverse three to five pairs. \open{Crosscoder calls on the nine pairs from its rerun up an
update-size ladder, cell P2.49 run 2.} The base model's prior with steering gives the built direction on all nine,
though the share's gaps rank with theirs at only 0.02 and $-0.30$. This is agreement with how the pairs were built, and
on that test the share is the most useful method we have (Table~\ref{tab:scorecard}).

\paragraph{No proxy tracks the definition.} Scored against $R$ under the adopted wording at the VPD featurization, no
proxy ranks the six hand-made edits, the nine pairs at matched size and the five key tasks as $R$ does once update
size and prior are partialled out
\open{per-proxy partial rank correlations and the matched random band; cell P1.46 for $R$, compare-lead scoring}.
Earlier readings pointed this way. Over the five key tasks at one size, the share ranks with the base model's prior at
a Spearman correlation of 0.9. In the KL form, every key task's share sits inside its own random band. On the
hand-made edits, neither form put a reuse edit above a new edit in any of the six comparisons. One reading cuts the
other way: the share puts the prior-matched cipher 0.40 below copy, so it is not the prior alone.

\paragraph{Or there is no reference.} \open{Keep this paragraph only if $R$ under wording A fails its known-answer
control (cell P1.46); otherwise delete it.} $R$ under the adopted wording did not order the hand-made edits
\open{lowest reuse edit against highest new edit, and its band; cell P1.46}. With no reading of the definition that
passes its own known-answer test, no proxy can be scored against the definition at this featurization. This repeats
the earlier reading at VPD, which put the ten-fold reuse edit at 0.9755, below a thin new edit at 0.9963. Only a
post-hoc reading with update size removed ordered the six, by 0.57 against a band of 0.25.

\paragraph{What a passing proxy would need.} Four things, none yet in hand. First, a reading of the definition that
passes its known-answer control, with random changes reading low. Second, agreement with that reading after size and
prior are removed, beyond the band of matched random changes; in its log-odds form the share has no random reading at
all. Third, agreement at a featurization not built from VPD
\open{$R$ under wording A at SAE latents, cell P1.40}, since at VPD the share and the definition's test are nearly the
same construction. Fourth, the same on fine-tunes nobody built (Sections~\ref{sec:real_text} and~\ref{sec:scale}).

\begin{figure}[t]
\centering
\fbox{\parbox{0.9\linewidth}{\centering\small\textbf{Planned figure: proxies against the baselines and the definition}}}
\caption{Does any proxy order the fine-tunes as the definition does, beyond what update size, the prior and a random
change give? One row per method: the share in both forms, the gating causal test, gain-only reach, effective rank,
steering, the prior, update size and the crosscoder. Left: each method's calls on the nine built pairs. Right, on the
same rows: rank correlation with $R$ under the adopted wording over the six hand-made edits, the nine pairs at matched
size and the five key tasks, with update size and prior partialled out. The grey band is the same correlation with
each fine-tune's reading replaced by its three matched random changes; the dashed line is zero. If $R$ fails its
known-answer control, the right panel is replaced by $R$ on the six edits with its band, and no proxy is scored.}
\label{fig:proxies}
\end{figure}
\open{Fig.~\ref{fig:proxies}: right panel needs cell P1.46 and compare-lead's scoring; left panel can be drawn now
from Table~\ref{tab:scorecard}, except the crosscoder row (cell P2.49 run 2).}
```

---

## Every number used, with its source

All paths under `/mnt/nw/home/e.agnew/commute-agents/`. Each was read in the source on 2026-11-20 ~21:00.

| number | used as | source |
|---|---|---|
| all 9 built pairs apart (9/0/0), all 9 kept after size matching | share vs build, A ¶1, B ¶1, table | `writeup-lead/paper/paper/sec5_baselines_scorecard_caveats.tex` l.22, 70-71; sources in `.numbers.md` |
| prior 1 / 0 / 8, one value a side on 8, built way on all 9 | A ¶1, table | sec5 tex l.36, 49-50, 83-84; `.numbers.md` l.20 (P2.26, P2.11) |
| 0.52 to 0.62 | share gaps on the three equal-size pairs (B) | `compare-lead/reports/analyses/2026-11-16_analysis_part2-gate-page-update.md` l.47 |
| update size 0 / 6 / 3, never agrees | baseline | sec5 tex l.37, 75-76 |
| three to five pairs reversed (neuron share 3/4/2, top-neuron 4/5/0, steering 4/3/2) | baselines, table | sec5 tex l.31, 33, 35; part2-gate-page-update l.48 |
| top-k gate 6 / 1 / 2 | table | `writeup-lead/reports/analyses/2026-11-20_analysis_e230-topk-gate.md` l.15-16, 82 |
| gain-only: rule held on 3 of 9, both sides reach half on 4, new tasks 0.72 to 0.91 | A ¶3, table | `main.tex` l.903-905; `writeup-lead/reports/analyses/2026-11-22_analysis_e231-gain-only.md` l.81 |
| prior + steering give the built direction on all 9 | B ¶1 | part2-gate-page-update l.51; sec5 tex l.84-85 |
| 0.02 and $-0.30$ | share gaps' rank correlation with prior and steering gaps | `.numbers.md` l.50 (`compare-lead` `experiments/method_table/results/gate_1116.json`, recomputed 0.0167) |
| twins: real task higher on only 1 of 4 | A ¶2 | sec5 tex l.22, 110 and the note at l.130-134: both twin tallies (scorecard's and the KL-form run's own rule) give real higher on 1; they disagree on the other three, so only that count is used |
| 0.40 below copy; priors 1.6-fold apart | share at one update size, prior-matched cipher vs copy | `compare-lead/experiments/key_grid/results/summary.json` `.rules.P1.dist_copy` = 0.403; `.log10_prior` copy -4.228, prior_cipher -4.429 (10^0.20 = 1.58) |
| Spearman 0.9 | share (log-odds) vs base prior over the five key tasks at one size | `key_grid/results/summary.json` `.spearman_vs_prior.r` = 0.9 |
| every key task's KL share inside its random band | B ¶2 | `key_grid/results/summary.json` `.rules.P4` false on cipher, prior_cipher, copy; `.table.kl_share.<task>.band` contains each mean on all five |
| 67 of 88 | random change reaches the fine-tune's KL share | `writeup-lead/reports/analyses/2026-11-21_analysis_e228-random-kl.md` l.7, 88 |
| log-odds form has no random reading (one log-odds needed) | random check | same report l.16-17 |
| none of six comparisons | hand-made edits, both share forms | same report l.137, 200-201 ("0 of 6") |
| 0.9755, 0.9963 | squared-cosine $R$ at VPD, ten-fold reuse edit vs thin new edit (B) | `writeup-lead/reports/analyses/2026-11-20_analysis_e226-kstar-vpd.md` l.17, 79 |
| 0.57 against 0.25 | size-removed $R$ margin on the six edits vs half-inputs band (B) | `writeup-lead/reports/analyses/2026-11-21_analysis_p1-42-r-reanalysis.md` l.64, 73 |
| six hand-made edits, nine pairs, five key tasks; three random draws | item sets | `fleet/cell show P1.46`; `compare-lead/reports/plans/2026-11-20_plan_p2-50-key-grid.md` |

Open gaps, by cell: P1.46 ($R$ under wording A per item, its known-answer verdict, its random band); compare-lead
scoring (rank agreement per proxy, size and prior partialled, random band); P1.45 (gating causal test, in B's figure);
P1.40 ($R$ at SAE latents, the non-VPD check); P2.49 run 2 (crosscoder up an update-size ladder). P2.50's results file
exists (`compare-lead/experiments/key_grid/results/summary.json`) but has no write-up yet; numbers from it are
provisional.
