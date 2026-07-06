"""Generate the shared cross-validation splits (fairness protocol, groundwork §8).

5-fold stratified CV × 5 seeds (0–4) per dataset → 25 (seed, fold) index sets,
stored as data/splits/<dataset>.json. These files are generated ONCE, committed,
and then shared by every model, forever (CLAUDE.md). Accordingly this script
- only creates MISSING files by default (existing ones are skipped),
- refuses to overwrite unless --force is passed, and
- offers --verify to prove stored files still match a fresh regeneration.

Usage:
  python data/make_splits.py --all                # create any missing files
  python data/make_splits.py --dataset moons      # repeatable
  python data/make_splits.py --all --verify       # regenerate in memory + compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # allow `python data/make_splits.py` from anywhere
    sys.path.insert(0, _ROOT)

import sklearn
from sklearn.model_selection import StratifiedKFold

from data import DATASETS, load_dataset

SPLITS_DIR = Path(__file__).resolve().parent / "splits"
SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5


def split_id(seed: int, fold: int) -> str:
    return f"s{seed}_f{fold}"


def checksum_of(splits: dict) -> str:
    """sha256 over the canonical JSON of the index sets (tamper evidence)."""
    canon = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def make_splits_payload(name: str) -> dict:
    """Deterministically derive all 25 (seed, fold) index sets for a dataset."""
    X, y, meta = load_dataset(name)
    splits: dict[str, dict] = {}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            splits[split_id(seed, fold)] = {
                "train": [int(i) for i in train_idx],
                "test": [int(i) for i in test_idx],
            }
    return {
        "dataset": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": list(SEEDS),
        "n_folds": N_FOLDS,
        "sklearn_version": sklearn.__version__,
        "loader_meta": meta,
        "checksum": checksum_of(splits),
        "splits": splits,
    }


def write_splits(name: str, force: bool = False) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}.json"
    if path.exists() and not force:
        print(f"{name}: {path} already exists — skipping (splits are generated once; "
              "use --verify to check integrity, --force only if you know what you are doing)")
        return path
    if path.exists() and force:
        print(f"{name}: OVERWRITING {path} (--force). This invalidates every past "
              "result for this dataset — make sure that is intended and documented.")
    payload = make_splits_payload(name)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{name}: wrote {len(payload['splits'])} splits -> {path}")
    return path


def verify(name: str) -> list[str]:
    """Compare the stored splits file against a fresh regeneration."""
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        return [f"{name}: missing splits file {path}"]
    stored = json.loads(path.read_text())
    problems: list[str] = []
    if stored["checksum"] != checksum_of(stored["splits"]):
        problems.append(f"{name}: stored checksum does not match stored indices")
    fresh = make_splits_payload(name)
    if stored["splits"] != fresh["splits"]:
        problems.append(f"{name}: stored indices differ from regeneration")
    if stored["loader_meta"] != fresh["loader_meta"]:
        problems.append(f"{name}: loader metadata drifted (stored "
                        f"{stored['loader_meta']} vs fresh {fresh['loader_meta']})")
    if stored["sklearn_version"] != sklearn.__version__ and not problems:
        print(f"{name}: note — file written with sklearn {stored['sklearn_version']}, "
              f"verifying under {sklearn.__version__}; indices still match")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS),
                        help="dataset to process (repeatable)")
    parser.add_argument("--all", action="store_true", help="process every registered dataset")
    parser.add_argument("--verify", action="store_true",
                        help="verify stored files instead of writing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files (DANGER: breaks the shared-splits guarantee)")
    args = parser.parse_args(argv)

    names = sorted(DATASETS) if args.all else (args.dataset or [])
    if not names:
        parser.error("pass --dataset NAME (repeatable) or --all")

    if args.verify:
        problems = [problem for name in names for problem in verify(name)]
        if problems:
            print("\n".join(problems))
            raise SystemExit(1)
        print(f"OK: {len(names)} splits file(s) match regeneration exactly")
    else:
        for name in names:
            write_splits(name, force=args.force)


if __name__ == "__main__":
    main()
