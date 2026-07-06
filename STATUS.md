# Project Status

Current phase: **Phase 1 — Foundations (weeks 1–4)**
Timeline: 20-week plan, see docs/groundwork.md §10.

## Experiment checklist
- [x] E0 — Infrastructure (env, loaders, splits, runner, tests) — done 2026-07-06
- [ ] E1 — Classical baseline sweep (all datasets × all baselines, Optuna-tuned)
- [ ] E2 — Quantum kernels (QSVM vs RBF-SVM, alignment, concentration)
- [ ] E3 — VQC grid (feature map × ansatz × depth × optimizer)
- [ ] E4 — Encoding ablation + Fourier-expressivity demo
- [ ] E5 — Hybrid models (PCA→VQC, CNN→quantum head, transfer learning)
- [ ] E6 — Barren-plateau / trainability study
- [ ] E7 — Sample-efficiency learning curves
- [ ] E8 — Noise ladder (local) + real-QPU validation runs
- [ ] E9 — Resource accounting (wall-clock, params, depth, shots)
- [ ] E10 — Stretch: provable-separation dataset (optional)

## Notes / decisions log
- (Claude Code: append dated one-line entries here when decisions are made,
  e.g. "2026-07-10: pinned pennylane==0.4x, qiskit==2.x" or
  "2026-07-21: dropped Wine dataset, too easy — all models at 100%.")
- 2026-07-06: pinned stack — qiskit 2.3.0 / aer 0.17.2 / qml 0.9.0 / runtime 0.45.1, pennylane 0.45.1 + pennylane-qiskit 0.45.0 (import-tested against qiskit 2.x, no conflict), sklearn 1.9.0, torch 2.12.1, numpy 2.4.6; full freeze in requirements-lock.txt.
- 2026-07-06: E0 core datasets with committed splits (5-fold stratified × seeds 0–4, ids s{seed}_f{fold}, checksummed JSON): moons, circles, iris_binary (versicolor vs virginica), wine_binary, breast_cancer, digits_3v5, banknote (OpenML 1462). Deferred additively: hidden-manifold + Qiskit ad-hoc (E2), MNIST/F-MNIST pair (image tier).
- 2026-07-06: wine_binary = classes (1,2) — hardest pair by deterministic probe (`python -m data.tabular`): CV acc 0.9917 vs (0,1) 0.9923 and (0,2) 1.0000; near-tie with (0,1).
- 2026-07-06: synthetic generation params fixed — n=500, moons noise 0.2, circles noise 0.1 factor 0.5, generation seed 1234 (distinct from run seeds); loaders take no parameters, variants require a new dataset name + new splits.
- 2026-07-06: shared preprocessing = per-feature min–max → [0, π], fit on training fold only, test folds clipped; identical features for classical and quantum models; classical rows log regime=classical.
- 2026-07-06: tuning protocol (per §8, user-approved): one Optuna study (50 trials) per (model, dataset), trials scored by 3-fold CV on the training portion of s0_f0; locked config evaluated on all 25 folds; non-nested — residual overlap stated as a Ch. 6 limitation.

## Blockers
- (none)
