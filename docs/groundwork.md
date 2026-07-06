# Thesis Groundwork — Classical ML vs Quantum ML vs Hybrid Models

*Working document, v0.3 — July 2026. Scope confirmed: **Master's level · standard ML benchmarks · local simulation · small free real-QPU runs in E8 · 3–6-month window (planned at ~5 months / 20 weeks)**. Remaining open questions in Section 12.*

---

## 1. Framing and research questions

**Working title options**
- "Classical, Quantum, and Hybrid Machine Learning: A Systematic Empirical Comparison under Matched Conditions"
- "Benchmarking Quantum and Hybrid Quantum–Classical Models against Classical Baselines on Near-Term Hardware and Simulators"

**Core research questions (RQs)** — these drive both the theory PDF and the code:

- **RQ1 (Performance):** Under matched data, preprocessing, and hyperparameter-search budgets, how do quantum models (kernels, variational classifiers) compare to strong classical baselines on accuracy/F1/AUC?
- **RQ2 (Design factors):** How do data-encoding strategy, ansatz choice, circuit depth, and entanglement affect quantum-model performance and trainability?
- **RQ3 (Hybrid value):** Do hybrid architectures (classical feature extraction + quantum head, quantum kernels + classical SVM, quantum transfer learning) outperform either pure approach — at matched parameter counts?
- **RQ4 (Robustness & cost):** How do shot noise, simulated device noise, and (optionally) real hardware affect results, and what are the real resource costs (wall-clock, circuit depth, shots)?
- **RQ5 (Data regime):** Does relative performance change with training-set size (sample efficiency) and feature dimensionality?

**Honest framing (important for credibility):** at the qubit counts a thesis can reach (≲ 20–25 qubits), every quantum model is classically simulable, so the thesis *cannot* demonstrate quantum advantage. It instead measures **empirical competitiveness and behavior at NISQ scale** — exactly the framing used by the strongest recent benchmark literature (Bowles/Ahmed/Schuld 2024; Schnabel & Roth 2025). State this in the introduction and conclusion; it protects you from the most common examiner criticism.

---

## 2. Deliverable A — the theoretical PDF (chapter structure)

**Ch. 1 — Introduction** (~6–10 pp)
Motivation and the hype-vs-evidence tension; RQ1–RQ5; contributions; thesis outline.

**Ch. 2 — Classical machine learning foundations** (~12–18 pp)
Supervised learning setup; bias–variance and generalization; kernel methods and SVMs (needed later for quantum kernels); neural networks and gradient descent; ensembles (RF, gradient boosting); evaluation methodology (CV, metrics, statistical testing).

**Ch. 3 — Quantum computing foundations** (~10–15 pp)
Qubits, superposition, entanglement; gates and circuits; measurement and expectation values; the NISQ era, noise channels, and error mitigation basics; classical simulability limits.

**Ch. 4 — Quantum machine learning** (~18–25 pp; the heart of the theory part)
- Taxonomy: CC/CQ/QC/QQ (data type × processing type).
- Data encoding: basis, angle, amplitude, data re-uploading; why encoding determines expressivity (Fourier-series view of PQC models).
- Variational quantum algorithms and parameterized quantum circuits; parameter-shift rule.
- Quantum kernel methods: fidelity kernels, feature maps (ZZ/IQP), connection "QML models are kernel methods."
- Architectures: VQC, data re-uploading classifier, QCNN.
- Trainability: barren plateaus (mechanisms, mitigations); exponential kernel concentration.
- Hybrid architectures: dressed quantum circuits, quantum transfer learning, quantum layers in PyTorch.

**Ch. 5 — Related work: comparative studies and the advantage debate** (~8–12 pp)
Systematic benchmarks and their conclusions; "power of data" argument; classical surrogates and dequantization; provable-separation constructions (discrete-log dataset); where quantum data changes the picture.

**Ch. 6 — Methodology** (~10–14 pp)
Datasets and preprocessing; model zoo and hyperparameter spaces; fairness protocol (Section 8 below); simulators, noise models, hardware; reproducibility setup; metrics and statistical tests.

**Ch. 7 — Experiments and results** (~20–30 pp)
One section per experiment family E1–E9 (Section 7 below), each with hypothesis → setup → results → discussion.

**Ch. 8 — Discussion** (~6–10 pp)
Synthesis across RQs; limitations and threats to validity (small scale, simulability, HPO budget sensitivity); implications for when quantum/hybrid approaches are worth considering.

**Ch. 9 — Conclusion and future work** (~3–5 pp)

**Appendices:** circuit diagrams; full hyperparameter tables; extended result tables; guide to the code repository.

*(Page counts assume a Master's thesis; they scale down ~40% for a Bachelor's and up for a PhD-adjacent report — see open questions.)*

---

## 3. Deliverable B — the coding component

Interpreting "a coding one" as: a reproducible code repository **plus** a practical/experiments report (some programs require the latter as a separate PDF — please confirm, Section 12).

**Repository skeleton**

```
qml-thesis/
├── environment.yml            # pinned versions
├── README.md                  # how to reproduce every figure/table
├── data/                      # loaders + generators (no raw data in git)
│   ├── synthetic.py           # moons, circles, parity, hidden-manifold, ad-hoc
│   ├── tabular.py             # UCI/sklearn loaders + preprocessing
│   └── images.py              # digits/MNIST subsets, PCA/AE reduction
├── models/
│   ├── classical.py           # sklearn/XGBoost wrappers, param-matched MLP
│   ├── qkernels.py            # fidelity kernels, feature maps
│   ├── vqc.py                 # feature-map × ansatz grid, re-uploading
│   ├── qcnn.py
│   └── hybrid.py              # CNN→quantum head, transfer learning, AE→VQC
├── experiments/
│   ├── configs/               # one YAML per run family (E1…E9)
│   └── run.py                 # config-driven runner, seeds, logging
├── analysis/                  # stats tests, plots, tables → thesis figures
├── results/                   # CSV/JSON logs (git-tracked, small)
└── notebooks/                 # exploration only, not source of truth
```

**Software stack (recommended)**
- Python 3.11+, scikit-learn, PyTorch, XGBoost, Optuna (HPO), pandas/matplotlib.
- **PennyLane** (+ `lightning.qubit` simulator) as the primary QML framework — best autodiff/PyTorch integration for variational and hybrid models, and the Bowles et al. `qml-benchmarks` package builds on it (reusable model implementations and dataset generators).
- **Qiskit 2.x + qiskit-machine-learning + Aer** for device noise models and IBM hardware runs; the `pennylane-qiskit` plugin lets PennyLane models run on IBM backends, so you don't have to reimplement anything for hardware.
- Experiment tracking: MLflow or plain CSV + config hashes (simpler to defend).

**Simulation & hardware plan (updated after scoping — validated July 2026)**
- **Important:** IBM retired its cloud-hosted simulators and IBM Quantum Lab on 15 May 2024, so there is no "IBM simulator in the cloud" anymore. All simulation runs **locally** on your machine — which is actually better for a thesis: free, unlimited, queue-free, and fully reproducible.
- Local stack: PennyLane `lightning.qubit` for training variational/hybrid models; Qiskit **Aer** for shot-based and noisy simulation. Roughly 4 GB of RAM handles ~27 qubits, so this thesis's ≤ 20-qubit regime runs comfortably on a laptop; consider Colab/GPU only if the E3 grids get slow.
- Device realism *without* hardware: Qiskit Runtime **local testing mode** — fake backends built from real QPU snapshots (coupling map, basis gates, calibrated noise applied automatically), or `NoiseModel.from_backend()` on a real device's calibration data. This makes experiment E8 fully offline.
- *Confirmed in scope:* a small set of real-QPU validation runs via the free IBM **Open Plan** (~10 min per 28-day window; since March 2026, logging 20 min within 12 months unlocks a one-time 180-min allocation — verify the promo is still offered when you sign up — incl. the Heron r2 `ibm_kingston`). Strategy: accumulate the 20 minutes deliberately across the first two 28-day windows, opt into the 180-min allocation, and spend it all on E8 near the end.

---

## 4. Literature map (starter bibliography, grouped; ★ = must-read)

**Foundations & reviews**
- ★ Biamonte et al., "Quantum machine learning," *Nature* 549, 195 (2017).
- ★ Schuld & Petruccione, *Machine Learning with Quantum Computers*, Springer (2021) — main textbook backbone for Ch. 4.
- ★ Cerezo et al., "Variational quantum algorithms," *Nat. Rev. Phys.* 3, 625 (2021).
- Preskill, "Quantum computing in the NISQ era and beyond," *Quantum* 2, 79 (2018).
- Rodríguez-Díaz et al., "A Survey of Quantum Machine Learning: Foundations, Algorithms, Frameworks, Data and Applications," *ACM Computing Surveys* (2025).
- "Supervised Quantum Machine Learning: A Future Outlook from Qubits to Enterprise Applications," arXiv:2505.24765 (2025) — good for the outlook/discussion chapter.
- "A review of quantum machine learning algorithms, applications, and emerging advantages," *Discover Computing* (2026) — recent comparative review, directly on your topic.

**Encoding & expressivity (feeds RQ2, E4)**
- ★ Schuld, Sweke, Meyer, "Effect of data encoding on the expressive power of variational quantum machine learning models," *Phys. Rev. A* 103, 032430 (2021) — Fourier-series view; basis for a beautiful demo experiment.
- Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier," *Quantum* 4, 226 (2020).
- LaRose & Coyle, "Robust data encodings for quantum classifiers," *Phys. Rev. A* 102, 032420 (2020).

**Quantum kernels (feeds E2)**
- ★ Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature* 567, 209 (2019).
- Schuld & Killoran, "Quantum machine learning in feature Hilbert spaces," *PRL* 122, 040504 (2019).
- ★ Schuld, "Supervised quantum machine learning models are kernel methods," arXiv:2101.11020 (2021).
- Liu, Arunachalam, Temme, "A rigorous and robust quantum speed-up in supervised machine learning," *Nat. Phys.* 17, 1013 (2021) — the discrete-log provable-separation dataset (stretch experiment E10).
- Kübler, Buchholz, Schölkopf, "The inductive bias of quantum kernels," NeurIPS (2021).
- Thanasilp et al., "Exponential concentration in quantum kernel methods," *Nat. Commun.* 15 (2024).
- Schnabel & Roth, "Quantum kernel methods under scrutiny: a benchmarking study," *Quantum Mach. Intell.* 7, 58 (2025).

**Variational classifiers & QNN architectures (feeds E3, E5)**
- Farhi & Neven, "Classification with quantum neural networks on near term processors," arXiv:1802.06002 (2018).
- Mitarai et al., "Quantum circuit learning," *Phys. Rev. A* 98, 032309 (2018).
- Schuld et al., "Circuit-centric quantum classifiers," *Phys. Rev. A* 101, 032308 (2020).
- ★ Abbas et al., "The power of quantum neural networks," *Nat. Comput. Sci.* 1, 403 (2021) — effective dimension; optional analysis experiment.
- Cong, Choi, Lukin, "Quantum convolutional neural networks," *Nat. Phys.* 15, 1273 (2019).
- Hur, Kim, Park, "Quantum convolutional neural network for classical data classification," *Quantum Mach. Intell.* 4, 3 (2022).

**Trainability & generalization (feeds E6, E7)**
- ★ McClean et al., "Barren plateaus in quantum neural network training landscapes," *Nat. Commun.* 9, 4812 (2018).
- Cerezo et al., "Cost-function-dependent barren plateaus in shallow parametrized quantum circuits," *Nat. Commun.* 12, 1791 (2021).
- Larocca et al., "Barren plateaus in variational quantum computing," *Nat. Rev. Phys.* (2025 review; arXiv:2405.00781).
- Caro et al., "Generalization in quantum machine learning from few training data," *Nat. Commun.* 13, 4919 (2022).
- Gil-Fuster, Eisert, Bravo-Prieto, "Understanding quantum machine learning also requires rethinking generalization," *Nat. Commun.* 15 (2024).

**The classical-vs-quantum debate (feeds Ch. 5 and your discussion)**
- ★ Huang et al., "Power of data in quantum machine learning," *Nat. Commun.* 12, 2631 (2021) — why data access often erases quantum advantage; also introduces the projected quantum kernel.
- ★ Bowles, Ahmed, Schuld, "Better than classical? The subtle art of benchmarking quantum machine learning models," arXiv:2403.07059 (2024) — the methodological template for your whole empirical part; open-source `qml-benchmarks` package (PennyLane-based).
- Schreiber, Eisert, Meyer, "Classical surrogates for quantum learning models," *PRL* 131, 100803 (2023).
- Cerezo et al., "Does provable absence of barren plateaus imply classical simulability?" arXiv:2312.09121.
- Huang et al., "Quantum advantage in learning from experiments," *Science* 376, 1182 (2022) — the quantum-data escape hatch; motivates optional Tier-3 datasets.

**Hybrid models (feeds E5)**
- ★ Mari et al., "Transfer learning in hybrid classical-quantum neural networks," *Quantum* 4, 340 (2020) — the "dressed quantum circuit"; notably one of the stronger QNN-family models in the Bowles et al. benchmark.

**Domain application reviews (pick per your domain choice)**
- Health: "A systematic review of quantum machine learning for digital health," *npj Digit. Med.* 8, 237 (2025).
- Chemistry/pharma: Smaldone et al., "Quantum machine learning in drug discovery," *Chem. Rev.* 125, 5436 (2025).
- HEP: Wu et al., "Application of quantum machine learning using the quantum kernel algorithm on high-energy physics analysis at the LHC," *Phys. Rev. Research* 3, 033221 (2021).
- Finance: "Quantum computing for financial transformation" review, arXiv:2604.08180 (2026).

*Search tips for expanding this: follow citations of Bowles et al. 2024 on Semantic Scholar; arXiv categories quant-ph + cs.LG; keywords "quantum kernel benchmark", "variational quantum classifier comparison".*

---

## 5. Datasets (tiered plan)

**Practical constraints that drive selection:** feature count must map to ≤ ~20 qubits after reduction (angle encoding uses ~1 qubit/feature); ~100–5,000 samples keeps simulation tractable; start with binary classification (extend to multi-class only if time allows); standardize features to [0, π] or [−π, π] for angle encoding.

| Tier | Dataset | Size / dims | Why it's in |
|---|---|---|---|
| 0 — synthetic (controlled) | moons, circles (sklearn) | any / 2 | sanity checks, decision-boundary visuals |
| 0 | parity / XOR-like | any / 4–10 | classically hard-for-linear, easy visuals |
| 0 | `qml-benchmarks` generators (linearly separable, hidden-manifold, downscaled-MNIST families, etc.) | parameterized | direct comparability with Bowles et al. results |
| 0 | Qiskit "ad hoc" dataset | any / 2–3 | constructed to favor ZZ-feature-map kernels — tests best-case quantum kernel |
| 1 — small tabular | Iris (2-class subsets), Wine | 150 / 4, 178 / 13 | classic QML testbeds, cheap |
| 1 | Breast Cancer Wisconsin | 569 / 30→PCA | most-used real dataset in QML papers |
| 1 | Pima Diabetes, Heart (Cleveland), Banknote | ~300–1,400 / 4–13 | variety of difficulty; health flavor |
| 1 | German Credit or credit-card fraud subset | 1,000+ / reduce | imbalanced + finance flavor (if finance domain chosen) |
| 2 — images (reduced) | sklearn digits (8×8), 2-class pairs | 1,797 / 64→PCA-8/16 | QCNN + hybrid CNN experiments |
| 2 | MNIST / Fashion-MNIST binary pairs (e.g., 0-vs-1, 3-vs-5, shirt-vs-pullover) | subsample 1–5k / PCA or AE to 4–16 | harder image tier; standard in the literature |
| 3 — quantum-native (optional, stretch) | PennyLane quantum datasets (molecular/spin systems); QDataSet (Perrier et al., *Sci. Data* 2022) | varies | where quantum models are *expected* to shine; strengthens the discussion |
| 3 (stretch) | Liu–Arunachalam–Temme discrete-log dataset | generated | the one provable quantum–classical separation |

**Finalized core set (standard-benchmark focus, per scoping):** moons + one `qml-benchmarks` generator family (e.g., hidden-manifold) + Qiskit ad-hoc (synthetic) · Iris 2-class, Wine 2-class, Breast Cancer Wisconsin, Banknote (tabular) · sklearn digits pair + one MNIST/Fashion-MNIST binary pair at PCA-8/16 (image) = **8–9 datasets**. The finance/health rows above become optional; Tier 3 stays a stretch goal. Enough for statistical claims, small enough to finish.

---

## 6. Model zoo

**Classical baselines (all via scikit-learn/XGBoost, default + tuned variants)**
Logistic regression · SVM-linear · SVM-RBF · Random Forest · Gradient boosting (XGBoost) · k-NN · small MLP (parameter-matched to the VQCs, e.g., ≤ ~200 params).

**Quantum models**
- Fidelity **quantum kernel + SVM** — angle-encoding kernel and ZZ/IQP feature map (depth 1–2); optionally the projected quantum kernel (Huang et al.).
- **VQC**: {angle, ZZ} feature maps × {hardware-efficient/EfficientSU2, StronglyEntanglingLayers} ansätze × depth {1, 3, 5}; parameter-shift gradients + Adam; also SPSA/COBYLA for the optimizer comparison.
- **Data re-uploading classifier** (1–4 qubits) — strong small model, cheap to run.
- **QCNN** (8 qubits) for the image tier.

**Hybrid models**
- PCA (or small autoencoder) → VQC (classical compression front-end).
- CNN feature extractor → quantum head (PennyLane `TorchLayer`) vs the *same* CNN → parameter-matched dense head — the cleanest hybrid-vs-classical test.
- Quantum transfer learning ("dressed quantum circuit," Mari et al.) on the image tier.
- Note for Ch. 4: quantum kernel + classical SVM is itself a hybrid (CQ) pipeline — use the taxonomy to organize all of this.

**Ablations built in:** entanglement on/off (Bowles et al. found removing entanglement often doesn't hurt — replicating this on your datasets is an easy, publishable-feeling result), depth sweeps, encoding swaps.

---

## 7. Experiment suite (the coding work, in build order)

| ID | Experiment | Key question | Main output |
|---|---|---|---|
| E0 | Infrastructure: loaders, config runner, seeding, logging, unit tests on toy data | — | reproducible pipeline |
| E1 | Classical baseline sweep: all baselines × all core datasets, tuned via Optuna (fixed trial budget) | reference performance | master results table |
| E2 | Quantum kernels: kernel matrices per dataset; QSVM vs RBF-SVM; kernel–target alignment; kernel-value concentration vs qubit count | RQ1, RQ2 | acc table + alignment/concentration plots |
| E3 | VQC grid: feature map × ansatz × depth on 3 datasets; Adam(param-shift) vs SPSA vs COBYLA; training curves | RQ1, RQ2 | heatmaps + convergence plots |
| E4 | Encoding study: same ansatz, different encodings; plus Fourier-expressivity demo (fit a 1-D function, show accessible frequency spectrum grows with re-uploads) | RQ2 | figure pair that basically writes Ch. 4.2 |
| E5 | Hybrids: PCA→VQC vs raw VQC; CNN→quantum head vs CNN→matched dense head; transfer learning variant | RQ3 | hybrid comparison table |
| E6 | Trainability: gradient-variance vs qubits (2–16) and depth → barren-plateau demonstration; local vs global cost | RQ2 | log-scale variance plot |
| E7 | Sample efficiency: learning curves (accuracy vs n_train ∈ {20…1000}) for best classical / quantum / hybrid per dataset | RQ5 | learning-curve figures |
| E8 | Noise & hardware: exact sim → shot-based sim (1k–10k shots) → fake-backend/Aer device-noise model → real IBM QPU (free Open Plan) for 2–3 best small models with readout mitigation | RQ4 | degradation waterfall chart |
| E9 | Resource accounting: wall-clock, #params, circuit depth, total shots per model | RQ4 | cost-vs-accuracy scatter |
| E10 | *Stretch:* discrete-log separation dataset or a quantum-data task (Tier 3) | advantage frontier | one focused figure |

Each experiment gets: hypothesis stated up front, config file, one figure/table, and a mapped thesis subsection — this keeps writing and coding synchronized.

---

## 8. Fairness & evaluation protocol (what makes the comparison defensible)

- **Identical splits everywhere:** 5-fold stratified CV × 5 seeds, splits generated once, stored, and shared across all models. Report mean ± std.
- **Matched tuning budget:** same number of Optuna trials (e.g., 30–50) and same search-space "size philosophy" for classical and quantum models — the single most common flaw in QML papers is tuned-quantum vs default-classical. Operationally (to stay tractable for quantum models): per (model, dataset), run **one** Optuna study (50 trials), each trial scored by 3-fold CV on the training portion of seed-0/fold-0; lock the best config; evaluate the locked config on all 25 (seed, fold) test folds. Not nested per outer fold — state the residual data overlap as a limitation in Ch. 6.
- **Parameter matching** for NN-style comparisons (MLP vs VQC; dense head vs quantum head).
- **Metrics:** accuracy, balanced accuracy, F1, ROC-AUC; convergence behavior; plus cost metrics from E9.
- **Statistics:** paired Wilcoxon signed-rank across datasets/seeds; critical-difference diagram for the headline comparison; state significance level up front.
- **Simulation regimes clearly separated:** exact statevector vs finite shots vs noisy vs hardware — never mix in one table.
- **Reproducibility:** pinned `environment.yml`, global seed policy, every figure regenerable from `analysis/` scripts, config hash logged per run.

---

## 9. Expected findings (hypotheses to state, not conclusions)

Based on the benchmark literature: tuned classical models will likely match or beat quantum models on most Tier 0–2 tasks; quantum kernels may win on quantum-friendly constructions (ad hoc data, Tier 3); hybrids will likely track their classical backbone; entanglement removal will often be harmless at these scales; noise will visibly degrade hardware results. Framing these as pre-registered hypotheses makes the thesis rigorous regardless of which way results fall.

---

## 10. Timeline — 20-week plan (3–6-month window, planned at ~5 months)

**Phase 1 — Foundations & infrastructure (weeks 1–4)**

| Week | Focus |
|---|---|
| 1 | Supervisor sign-off on RQs/scope; read the ★ papers; repo + `environment.yml`; create IBM Quantum account (Open Plan) and run a smoke-test job on `ibm_kingston` |
| 2 | E0 infrastructure (loaders, config runner, seeding, logging, unit tests); Ch. 1 + Ch. 3 skeletons |
| 3 | E1 classical baseline sweep with Optuna; draft Ch. 2 |
| 4 | **M1:** baseline results frozen; Ch. 2–3 drafted; ~10 QPU min logged (window 1) |

**Phase 2 — Core quantum & hybrid experiments (weeks 5–12)**

| Week | Focus |
|---|---|
| 5–6 | E2 quantum kernels; log ~10 more QPU min in window 2 → reach 20 min → **opt into the one-time 180-min allocation** |
| 7–8 | E3 VQC grid (feature map × ansatz × depth × optimizer); draft Ch. 4 alongside |
| 9 | E4 encoding ablation + Fourier-expressivity demo |
| 10–11 | E5 hybrids (PCA→VQC; CNN→quantum head vs matched dense head; transfer learning) |
| 12 | **M2:** full comparison grid done on all 8–9 datasets; Ch. 4–5 drafted |

**Phase 3 — Analysis, hardware, writing (weeks 13–20)**

| Week | Focus |
|---|---|
| 13 | E6 barren-plateau study + E7 learning curves |
| 14–15 | E8: local noise ladder, then real-QPU runs (2–3 best models, ≤ 8 qubits, readout mitigation) spending the 180-min allocation |
| 16 | E9 resource accounting; **results freeze** |
| 17–18 | Statistics, critical-difference diagrams, final figures; write Ch. 6–7 |
| 19 | Ch. 8–9, abstract, intro polish; repo README + reproducibility pass |
| 20 | Buffer: supervisor review loop, formatting, submission prep |

**QPU budget tactics (180 min is small — spend it wisely):**
- **Never train on hardware** — parameter-shift training needs thousands of circuit evaluations. Train in simulation; use the QPU only for (a) inference of trained VQC/re-uploading models on test subsets and (b) estimating small quantum-kernel Gram matrices (e.g., 30×30).
- Individual circuits use only milliseconds of QPU time, so accumulating the 20-min threshold takes deliberate effort: batch many circuits per job and use larger shot counts (~4k) in windows 1–2.
- Log backend name, calibration date, and job IDs for every hardware run — examiners like this, and it makes E8 reproducible-in-spirit.

**Compression/extension rule:** if the deadline lands nearer 3 months (13 weeks): drop E10, shrink the E3 grid to one dataset, reduce E4 to a single-figure mini-demo, and merge weeks 17–19. If nearer 6 months: add E10 and a multi-class extension of the best three models.

---

## 11. Assumptions I've made (flagging, not assuming silently)

**Confirmed by you:** Master's level · standard ML benchmark datasets · simulation-based (running locally, since IBM's cloud simulators were retired in May 2024).

Still assumed:
1. Core task = **supervised binary classification** (the standard for classical-vs-quantum comparisons); multi-class/regression only as extensions.
2. Language/stack = **Python**, thesis written in **LaTeX**.
3. "Coding PDF" = reproducible repo + a written experiments report.

If any of these is wrong, say so and I'll restructure.

## 12. Open questions for you

*(Answered so far: Master's · local simulation · standard benchmarks · 3–6-month window · real-QPU runs in scope.)* Remaining — answer whenever, none of these block starting:

1. Any **university template or supervisor constraints** (e.g., "must use Qiskit only", page limits, language)?
2. Does your program require the coding part as a **separate PDF report**, or is the repo + appendix enough?
3. Should scope stay at **binary classification**, or must regression/multi-class/generative appear?
