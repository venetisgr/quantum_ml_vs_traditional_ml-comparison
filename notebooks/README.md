# notebooks/

Exploration only — never source of truth (groundwork §3). Anything that feeds
the thesis must graduate into `data/`, `models/`, `experiments/`, or
`analysis/` with tests.

## Companion notebooks (committed with executed outputs)

These import the tested pipeline code and demonstrate it interactively — they
compute nothing the pipeline can't:

| Notebook | Shows |
|---|---|
| `01_dataset_tour.ipynb` | the 7 core datasets (table + plots), digits 3v5 samples, and the shared `[0, π]` AngleScaler fit on a real stored training fold |
| `02_splits_and_runner.ipynb` | anatomy of a stored splits file, running `e0_smoke` through the runner API (into a temp dir!), reading/aggregating the results CSV, and the determinism check against the committed results |

Rule of thumb for new notebooks: point every demo run at a temporary
`results_dir` — `results/*.csv` are append-only, git-tracked thesis logs.
