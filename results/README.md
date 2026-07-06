# results/

Append-only CSV logs, one file per experiment family (`e0_smoke.csv`,
`e1_baselines.csv`, ...). Files are small and git-tracked. **Never overwrite,
rewrite, or delete rows** (CLAUDE.md); `experiments/run.py` enforces this by
only ever appending, and by refusing to touch a file whose header differs from
its schema. Analysis scripts must filter by `config_hash`/`regime` rather than
editing these files.

## Row schema (written by `experiments/run.py`)

| Column | Meaning |
|---|---|
| `run_id` | random 12-hex identifier of this row |
| `timestamp_utc` | ISO-8601 UTC time the row was written |
| `experiment` | experiment family name from the config (also the file name) |
| `dataset` | dataset registry name (see `data/__init__.py`) |
| `model` | model registry name (see `models/__init__.py`) |
| `config_hash` | first 10 hex chars of sha256 over the canonicalized YAML config |
| `seed` | run seed (0–4); seeds RNGs and selects the split family |
| `split_id` | stored split identifier `s{seed}_f{fold}` from `data/splits/` |
| `regime` | one of `exact`, `shots`, `noisy`, `hardware`, `classical` — never mix regimes in one analysis table (§8) |
| `n_train` / `n_test` | fold sizes |
| `n_features` | feature count as loaded (before any model-side reduction) |
| `n_params` | learned parameter count; empty where the notion does not apply (e.g. kernel SVMs) |
| `accuracy`, `balanced_accuracy`, `f1`, `roc_auc` | test-fold metrics (§8); `f1` uses positive class 1; `roc_auc` empty if the model exposes no scores |
| `train_time_s` / `eval_time_s` | wall-clock seconds (resource accounting, E9) |
| `git_commit` | short hash of HEAD when the row was produced. A row committed to git in the *following* commit necessarily references that commit's parent — expected, not an error. |

## Conventions

- One (seed, fold) evaluation per row; aggregation (mean ± std over the 25
  rows) happens in `analysis/`, never here.
- A locked config evaluated on all 25 folds shares one `config_hash` across
  its 25 rows — that is how analysis scripts group runs.
- Simulation regimes stay in separate tables/columns in every downstream
  analysis (fairness protocol §8).
