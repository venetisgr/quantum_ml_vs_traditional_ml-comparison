# Project Status

Current phase: **Phase 1 — Foundations (weeks 1–4)**
Timeline: 20-week plan, see docs/groundwork.md §10.

## Experiment checklist
- [ ] E0 — Infrastructure (env, loaders, splits, runner, tests)
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

## Blockers
- (none)
