---
name: verify
description: Runtime verification recipe for this repo — how to build the env and drive the experiment pipeline end-to-end without touching committed results or splits.
---

# Verifying this repo at its surfaces

The runtime surfaces are three CLIs (no server, no GUI):

1. `python experiments/run.py --config experiments/configs/<name>.yaml`
2. `python data/make_splits.py --all [--verify]`
3. `python -m data.tabular` (wine-pair probe; ignore the harmless runpy RuntimeWarning)

## Environment

No conda in CI/sandbox containers — build a venv from the pinned pip set instead:

```bash
python3.11 -m venv /tmp/qml-venv
/tmp/qml-venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r <(python -c "import yaml; d=yaml.safe_load(open('environment.yml'))['dependencies']; \
print('\n'.join(next(x['pip'] for x in d if isinstance(x, dict))))")
```

The cpu extra index keeps torch small; versions are unchanged. Banknote's first
load fetches from OpenML (~10 s through a proxy; cached in ~/scikit_learn_data).

## Drive it (never against committed artifacts)

- **Always pass `--results-dir <scratch>`** when exercising the runner:
  `results/*.csv` are append-only, git-tracked thesis logs. A verification row
  in a committed CSV is pollution you cannot delete (CLAUDE.md).
- Happy path: run `e0_smoke.yaml` twice into one scratch dir → 25 then 50 rows;
  accuracies of both halves must be identical AND equal to the committed
  `results/e0_smoke.csv` (full pipeline determinism).
- Splits integrity: `python data/make_splits.py --all --verify` must print
  `OK: ... match regeneration exactly`; plain `--all` must skip all existing
  files and leave `git status data/splits/` empty.
- Guards worth re-probing after runner changes: unknown dataset/model in a
  config (KeyError, exit 1), missing splits file via `--splits-dir <empty>`
  (FileNotFoundError, exit 1), invalid regime and unknown config keys
  (ValueError, exit 1), foreign CSV header in the results dir (RuntimeError
  "refusing to append", exit 1, file left untouched), bad flags (exit 2).

## Gotchas

- Error paths surface as Python tracebacks (by design for a research tool);
  the message text is the contract, not the formatting.
- Only direct deps are pinned; transitive packages float (e.g.
  charset_normalizer). `requirements-lock.txt` records one known-good freeze.
- Never run `make_splits.py --force` during verification.
