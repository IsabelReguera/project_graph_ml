"""
Build the CASF-2016 split and save it as text files.

    test  = CASF-2016 core set (all 285 complexes, independent of the index filters)
    val   = VAL_FRACTION of (index - core), sampled at random from the refined set
    train = all the other complexes of the index (core and val excluded)

Run once, then commit the .txt files. An existing split is never overwritten unless --force.

Usage:
    python scripts/make_splits.py
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from gmnbind.data.splits import read_codes, write_codes

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "data" / "index"

# --- Split parameters ---------------------------------------------------------
INDEX_CSV    = INDEX_DIR / "index.csv"           # index used for train/val
CORE_TXT     = INDEX_DIR / "codes_core.txt"      # CASF-2016 core set (test)
REFINED_TXT  = INDEX_DIR / "codes_refined.txt"   # refined set (val candidates)
SPLIT_NAME   = "v2020_casf"                      # output: data/index/splits/SPLIT_NAME/
VAL_FRACTION = 0.15                               # fraction of (index - core) used for validation
SEED         = 47
# -------------------------------------------------------------------------------


def make_casf_split(index_codes, core_codes, refined_codes, f_val, seed):
    index_set, core_set = set(index_codes), set(core_codes)
    pool = index_set - core_set                 # everything that can go in train/val
    n_val = round(len(pool) * f_val)

    # val candidates: refined complexes in the pool.
    # Sorted before sampling -> the result depends only on the seed, not on file order.
    candidates = sorted(set(refined_codes) & pool)
    if len(candidates) < n_val:
        raise ValueError(f"VAL_FRACTION={f_val} needs {n_val} validation complexes, "
                         f"but only {len(candidates)} refined complexes are available")
    val = sorted(random.Random(seed).sample(candidates, n_val))

    val_set = set(val)
    train = sorted(pool - val_set)
    test = sorted(core_set)
    return {"train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()

    out_dir = INDEX_DIR / "splits" / SPLIT_NAME
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists. Splits must stay fixed: change SPLIT_NAME or use --force.")

    index_codes = pd.read_csv(INDEX_CSV)["pdb_code"].str.lower().tolist()
    core_codes = read_codes(CORE_TXT)
    refined_codes = read_codes(REFINED_TXT)

    missing_core = sorted(set(core_codes) - set(index_codes))
    if missing_core:
        print(f"WARNING: {len(missing_core)}/{len(core_codes)} core complexes are not in {INDEX_CSV.name}. "
              f"They stay in test.txt but must be added to the index so they get preprocessed:\n  {missing_core}")

    split = make_casf_split(index_codes, core_codes, refined_codes, VAL_FRACTION, SEED)
    for name, codes in split.items():
        write_codes(out_dir / f"{name}.txt", codes)

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "index": INDEX_CSV.name, "core": CORE_TXT.name, "refined": REFINED_TXT.name,
        "val_fraction": VAL_FRACTION, "n_val": len(split["val"]), "seed": SEED,
        "sizes": {k: len(v) for k, v in split.items()},
        "core_missing_from_index": missing_core,
    }
    (out_dir / "split_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])} -> {out_dir}")


if __name__ == "__main__":
    main()