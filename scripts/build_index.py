"""
Build the cleaned PDBbind index from the official INDEX file. [INDEX_general_PL.2020R1.lst]

Usage:
    python scripts/build_index.py --config configs/data/index_v2020.yaml
"""

# Import libraries - Settings
#===========================
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/build_index.py → radice della repo
DATA_DIR = REPO_ROOT / "data"

#regex [PDB_code, resolution, release_year, binding_data[type,compare,unit], reference, ligand_name, comments]
INDEX_REGEX = re.compile(
    r"(\w+)\s+"                            # pdb code
    r"([\d.]+|NMR)\s+"                     # resolution
    r"(\d{4})\s+"                          # release year
    r"(\w+)([=<~]{0,2})([\d\.]+)(\w+)\s+"  # binding data
    r"//\s[\w\.]+\s+"                      # reference (ignored)
    r"\(([^)]+?)\)"                        # ligand name
    r"\s*(.*)"                             # comments
)

UNIT_TO_MOLAR = {"mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}


#Extract and parse PDBbind index file
#============================
def parse_index(path: Path) -> pd.DataFrame:
    """Parse the PDBBind index file and return a DataFrame."""
    rows, skipped = [], 0
    skipped = 0
    with open(path, encoding="utf-8") as f:
        lines = [
            ln.strip()
            for ln in f
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        for ln in lines:
            match=INDEX_REGEX.match(ln)
            if match:
                (pdb, res, year, btype, comp, val, unit, ligand, comment) = match.groups()
                rows.append({
                    "pdb_code": pdb,
                    "resolution": res,
                    "binding_type": btype,
                    "compare": comp,
                    "value": val,
                    "unit": unit,
                    "ligand_name": ligand,
                    "comments": comment.strip()
                })
            else:
                skipped += 1
    print(f"Parsed: {len(rows)} | Skipped: {skipped}")
    df=pd.DataFrame(rows)
    return df


#Clean the parsed index according to the filters specified in the config file
#============================
def clean_index(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Take the parsed index DataFrame and apply the filters specified in the config file."""
    
    if filters["exclude_nmr"]:
        df = df[df["resolution"] != "NMR"] #exclude NMR structures
    df = df[df["compare"].isin(filters["allowed_compare"])].copy() #filter by allowed compare values
    df["resolution"] = df["resolution"].astype(np.float32)
    df["value"] = df["value"].astype(np.float32)
    
    df = df[df["resolution"] <= filters["max_resolution"]] # Filter by maximum resolution
    pattern = "|".join(filters["exclude_comments"]) # Filter out comments containing specified patterns
    df = df[~df["comments"].str.contains(pattern, case=False, na=False)] # Filter out comments containing specified patterns
    df = df[df["binding_type"].isin(filters["binding_types"])] # Filter by binding types
    
    # Convert binding values to molar units
    df["value_M"] = df["value"] * df["unit"].map(UNIT_TO_MOLAR).fillna(1)
    df = df[df["value_M"] > 0]
    df["pAffinity"] = (-np.log10(df["value_M"])).astype(np.float32).round(4)
    return df


#Add ligand SMILES to the DataFrame by reading the SDF files in the structures directory
#============================
def add_ligand_smiles(df: pd.DataFrame, structures_dir: Path) -> pd.DataFrame:
    smiles, missing = [], 0
    
    def get_clean_smiles(smile_path:Path) -> str | None:
        #load molecola from SDF files
        suppl = Chem.SDMolSupplier(smile_path, removeHs=True, sanitize=True)
        if len(suppl) == 0 or suppl[0] is None:
            return None
        mol = suppl[0]
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception:
            return None

    print("\nExtracting SMILES from SDF files...")
    for code in df["pdb_code"]:
        sdf = structures_dir/ code / f"{code}_ligand.sdf"
        if not sdf.exists():
            missing += 1
            smiles.append(None)
            print(f"Impossible to find SDF file for [{code}] -> Path does not exist: {sdf}")
        else:
            smiles.append(get_clean_smiles(sdf))
            
    df["SMILE"] = smiles
    n_before = len(df)
    df = df[df["SMILE"].notna()]
    print(f"Missing SDF: {missing} | Invalid/missing SMILES removed: {n_before - len(df)}")
    return df
    
#MAIN
#============================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/index_v2020.yaml")
    args = parser.parse_args()
    print(f"Building PDBbind index from config [{args.config}]...")
    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    df = parse_index(REPO_ROOT / cfg["index_file"])
    n0 = len(df)
    df = clean_index(df, cfg["filters"])
    df = add_ligand_smiles(df, REPO_ROOT / cfg["structures_dir"])

    out = REPO_ROOT / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["pdb_code", "resolution", "pAffinity", "SMILE"]].to_csv(out, index=False)
    print(f"{n0} → {len(df)} complexes.\nSaved to {out}")
    
if __name__ == "__main__":
    main()
    
    #Test the new index against the old one
    """
    import pandas as pd
    old = pd.read_csv("data/index/PL_cleaned_index.csv")
    new = pd.read_csv("data/index/v2020_kd_cleaned_index.csv")
    print(len(old), len(new))
    print(set(old.pdb_code) ^ set(new.pdb_code))
    """