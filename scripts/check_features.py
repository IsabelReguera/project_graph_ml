"""
Check that the current featurization code + configs/data/complex_features.yaml
rebuild the graphs stored in data/processed/v1 (on a sample of complexes).

For every graph:
    x, edge_index                 must be exactly equal
    edge_attr categorical columns must be exactly equal (bond type, conjugation)
    edge_attr distance column     may differ by float rounding (< ATOL, the precision of the
                                  input coordinates): torch.cdist is matmul-based for > 25 atoms

Usage:
    python scripts/check_features_v1.py
"""
from pathlib import Path

import torch
import yaml

from gmnbind.data.protein import Protein
from gmnbind.data.ligand import Ligand

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "pdbbind_v2020" / "generalset"
V1 = REPO_ROOT / "data" / "processed" / "test"
CONFIG = REPO_ROOT / "configs" / "data" / "complex_features.yaml"
N_SAMPLES = 20
# Å, tolerance on distances: the precision of the coordinates in the PDB/SDF files (3-4 decimals).
# torch.cdist (matmul-based in float32 for > 25 atoms) gives differences of ~1e-5 - 1e-4 Å across versions.
ATOL = 1e-3

# edge_attr columns: name -> exact comparison (True) or tolerance (False)
EDGE_COLUMNS = {
    "protein": {0: ("bond_type", True), 1: ("distance", False)},
    "ligand":  {0: ("bond_type", True), 1: ("conjugated", True), 2: ("distance", False)},
}


def compare(kind, new, old):
    """Return (status, details). status: 'identical' | 'rounding' | 'DIFFERENT'."""
    x, edge_index, edge_attr, _ = new
    if x.shape != old.x.shape or edge_index.shape != old.edge_index.shape:
        return "DIFFERENT", (f"nodes {old.x.shape[0]} -> {x.shape[0]}, "
                             f"edges {old.edge_index.shape[1]} -> {edge_index.shape[1]}")
    if not torch.equal(x, old.x):
        cols = (x != old.x).any(dim=0).nonzero().flatten().tolist()
        return "DIFFERENT", f"node feature columns {cols}"
    if not torch.equal(edge_index, old.edge_index):
        return "DIFFERENT", "edge_index"
    if torch.equal(edge_attr, old.edge_attr):
        return "identical", ""

    max_diff = 0.0
    for col, (name, exact) in EDGE_COLUMNS[kind].items():
        d = (edge_attr[:, col] - old.edge_attr[:, col]).abs().max().item()
        if (exact and d > 0) or (not exact and d > ATOL):
            return "DIFFERENT", f"edge column {col} ({name}), max diff {d:.2e}"
        if not exact:
            max_diff = max(max_diff, d)
    return "rounding", f"max distance diff {max_diff:.1e} Å"


cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
codes = sorted(p.stem[len("protein_"):] for p in V1.glob("protein_*.pt"))[:N_SAMPLES]

counts = {"identical": 0, "rounding": 0, "DIFFERENT": 0}
for code in codes:
    new = {
        "protein": Protein.from_pdb_path(RAW / code / f"{code}_protein.pdb", cfg=cfg["protein"]).get_pyg_graph(),
        "ligand":  Ligand.from_sdf(str(RAW / code / f"{code}_ligand.sdf"), cfg=cfg["ligand"]).get_pyg_graph(),
    }
    line = [code]
    for kind in ("protein", "ligand"):
        old = torch.load(V1 / f"{kind}_{code}.pt", weights_only=False)
        status, details = compare(kind, new[kind], old)
        counts[status] += 1
        line.append(f"{kind} {status}" + (f" ({details})" if details else ""))
    print("  ".join(line))

n = 2 * len(codes)
print(f"\n{counts['identical']}/{n} graphs identical, {counts['rounding']}/{n} equal up to float rounding, "
      f"{counts['DIFFERENT']}/{n} DIFFERENT")
print("v1 is reproducible" if counts["DIFFERENT"] == 0 else "v1 is NOT reproducible: generate a new version")
