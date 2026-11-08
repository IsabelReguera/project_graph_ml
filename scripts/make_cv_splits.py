"""
Build K-fold cross-validation splits on top of an existing split.

    pool             = train + val of BASE_SPLIT (its test set is never part of the CV)
    fold_i/val.txt   = i-th fold of the pool
    fold_i/train.txt = the other K-1 folds

Each fold is a regular split folder, readable with load_split().
Run once, then commit the .txt files. Never overwrites unless --force.

Usage:
    python scripts/make_cv_splits.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import KFold

from gmnbind.data.splits import load_split, write_codes

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "index" / "splits"

# --- CV parameters --------------------------------------------------------------
BASE_SPLIT = "v2020_casf"          # split whose train+val is divided into folds
CV_NAME    = "v2020_casf_cv5"      # output: data/index/splits/CV_NAME/fold_i/
K          = 5
SEED       = 47
# ---------------------------------------------------------------------------------


def make_cv_folds(pool, k, seed):
    """Return a list of (train, val) code lists, one per fold."""
    pool = sorted(pool)                      # result depends only on the seed
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(pool):
        folds.append((sorted(pool[i] for i in train_idx), sorted(pool[i] for i in val_idx)))
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing folds")
    args = parser.parse_args()

    out_dir = SPLITS_DIR / CV_NAME
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists. Splits must stay fixed: change CV_NAME or use --force.")

    base = load_split(SPLITS_DIR / BASE_SPLIT)
    pool = base["train"] + base["val"]          # the test set of BASE_SPLIT is left out of the CV

    for i, (train, val) in enumerate(make_cv_folds(pool, K, SEED)):
        fold_dir = out_dir / f"fold_{i}"
        write_codes(fold_dir / "train.txt", train)
        write_codes(fold_dir / "val.txt", val)
        print(f"fold_{i}: train {len(train)} | val {len(val)}")

    info = {"created": datetime.now().isoformat(timespec="seconds"),
            "base_split": BASE_SPLIT, "k": K, "seed": SEED, "pool_size": len(pool)}
    (out_dir / "cv_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()