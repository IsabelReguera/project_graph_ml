"""
Extract the PDB codes from a PDBbind index file and save them to a text file (one code per line).
Optionally keep only the codes that are also present in another index CSV (column `pdb_code`).

Usage:
    python scripts/extract_codes.py data/raw/pdbbind_v2020/INDEX_refined_set.2020 data/index/refined_codes.txt \
        --keep-in data/index/parsed_index.csv
"""
import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("index_file")
parser.add_argument("output_txt")
parser.add_argument("--keep-in", default=None, help="CSV with a pdb_code column: keep only codes present there")
args = parser.parse_args()

codes = []
with open(args.index_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            codes.append(line.split()[0].lower())
print(f"{len(codes)} codes in {args.index_file}")

if args.keep_in:
    allowed = set(pd.read_csv(args.keep_in)["pdb_code"].str.lower())
    dropped = [c for c in codes if c not in allowed]
    codes = [c for c in codes if c in allowed]
    print(f"{len(codes)} kept, {len(dropped)} not in {args.keep_in}: {dropped}")

out = Path(args.output_txt)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(codes) + "\n", encoding="utf-8")
print(f"Saved to {out}")