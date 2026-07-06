# analysis/

Statistics and plotting scripts. **Every figure and table in the thesis must be
regenerable by a script in this directory** (CLAUDE.md; groundwork §8) — no
hand-typed numbers anywhere in the LaTeX.

Arrives with E1+: aggregation of `results/*.csv` (mean ± std over the 25
(seed, fold) evaluations), paired Wilcoxon signed-rank tests, and the
critical-difference diagram for the headline comparison. Scripts must never
mix simulation regimes (exact / shots / noisy / hardware) in one table.
