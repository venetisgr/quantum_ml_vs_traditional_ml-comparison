# QML Thesis — Project Memory

## What this project is
Master's thesis: systematic comparison of classical ML vs quantum ML vs hybrid models
under matched conditions. Deliverables: a LaTeX thesis + this reproducible repo.
The full plan (research questions, chapter map, datasets, model zoo, experiments
E0–E10, evaluation protocol) lives in **docs/groundwork.md**. Read it before
planning or modifying any experiment. Read **STATUS.md** at session start.

## Stack
Python 3.11+ · PennyLane (`lightning.qubit`) for variational/hybrid models ·
Qiskit 2.x + qiskit-machine-learning + qiskit-aer for noise models and IBM hardware ·
scikit-learn, PyTorch, XGBoost, Optuna, pandas, matplotlib. Pin all versions in
`environment.yml`.

## Layout
- `data/` — loaders, preprocessing, `make_splits.py`, stored splits in `data/splits/`
- `models/` — `classical.py`, `qkernels.py`, `vqc.py`, `qcnn.py`, `hybrid.py`
- `experiments/` — `run.py` (config-driven) + `configs/*.yaml` (one per run family)
- `analysis/` — stats + plotting scripts; every thesis figure is generated here
- `results/` — CSV/JSON logs (small, git-tracked)
- `notebooks/` — exploration only, never source of truth

## Commands
- Setup: `conda env create -f environment.yml`
- Run an experiment: `python experiments/run.py --config experiments/configs/<name>.yaml`
- Tests: `pytest tests/`

## Non-negotiable rules (fairness protocol — the thesis's credibility depends on these)
- **IMPORTANT: all models use the same stored splits.** 5-fold stratified CV × 5 seeds,
  generated once by `data/make_splits.py`, saved to `data/splits/`. Never re-split ad hoc.
- Matched hyperparameter budget: the same number of Optuna trials (default 50) for
  classical AND quantum models. Never compare tuned-quantum against default-classical.
- MLP baselines are parameter-matched to the VQCs; always log parameter counts.
- Never mix simulation regimes (exact statevector / finite shots / noisy / hardware)
  in one results table — keep them as separate, labeled columns or tables.
- Every result row logs: dataset, model, config hash, seed, split id, regime, metrics.
  Append to `results/*.csv`; never overwrite or delete past runs.
- No hand-typed numbers in the thesis: every figure and table must be regenerable by a
  script in `analysis/`.

## Hardware guardrail
- **IMPORTANT: never submit jobs to real IBM QPUs without my explicit confirmation in
  the current session.** Free Open Plan minutes are scarce (~10 min per 28 days).
  Default every run to local simulation (`lightning.qubit`, `AerSimulator`, or
  qiskit-ibm-runtime fake backends).
- Never train on hardware. Hardware is for inference of pre-trained models and small
  kernel Gram matrices only (see docs/groundwork.md §10 "QPU budget tactics").
- IBM credentials come from the saved QiskitRuntimeService account / environment
  variables. Never hardcode or commit tokens.

## Workflow
- One experiment family (E0–E10) per session. Propose a plan and wait for my approval
  before writing code for a new experiment.
- Write tests for loaders and the runner; run `pytest` after changes.
- Small, frequent commits with messages like `E2: add ZZ feature map kernel`.
- When an experiment milestone completes, update STATUS.md in the same commit.
