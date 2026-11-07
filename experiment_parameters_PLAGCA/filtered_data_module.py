"""
DataModule subclass that applies `graph_filters` on-the-fly.

Reuses the standard ComplexDataset/ComplexDataModule for everything
(disk layout, train/val split, collate_fn) and only changes the Dataset
objects so that `__getitem__` returns graphs with edges already filtered.

Nothing on disk is modified: the preprocessed .pt files (protein at 8 A,
ligand at bonds + 5 A) stay the same.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sklearn.model_selection import train_test_split
from complex_dataset import ComplexDataset, build_complex_dataset
from utils import ComplexDataModule

from graph_filters import (
    filter_protein_edges_by_distance,
    keep_only_chemical_bonds_ligand,
)


class FilteredComplexDataset(ComplexDataset):
    """Wraps ComplexDataset and applies two edge filters to each sample."""

    def __init__(self, processed_dir, pdb_codes,
                 protein_max_distance=None, ligand_bonds_only=False):
        super().__init__(processed_dir, pdb_codes)
        self.protein_max_distance = protein_max_distance
        self.ligand_bonds_only = ligand_bonds_only

    def __getitem__(self, idx):
        protein, ligand = super().__getitem__(idx)
        if self.protein_max_distance is not None:
            protein = filter_protein_edges_by_distance(protein, self.protein_max_distance)
        if self.ligand_bonds_only:
            ligand = keep_only_chemical_bonds_ligand(ligand)
        return protein, ligand


class FilteredComplexDataModule(ComplexDataModule):
    """
    Drop-in replacement for ComplexDataModule that uses FilteredComplexDataset.
    Extra kwargs: protein_max_distance (float|None), ligand_bonds_only (bool).
    """

    def __init__(self, processed_dir, csv_path, batch_size=16, limit=None,
                 train_val_split=0.2, seed=47,
                 protein_max_distance=None, ligand_bonds_only=False):
        super().__init__(processed_dir, csv_path, batch_size=batch_size,
                         limit=limit, train_val_split=train_val_split, seed=seed)
        self.protein_max_distance = protein_max_distance
        self.ligand_bonds_only = ligand_bonds_only

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = build_complex_dataset(
                self.processed_dir, self.csv_path, limit=self.limit
            )
            all_codes = full_dataset.pdb_codes
            train_codes, val_codes = train_test_split(
                all_codes, test_size=self.train_val_split, random_state=self.seed
            )
            self.train_ds = FilteredComplexDataset(
                self.processed_dir, train_codes,
                protein_max_distance=self.protein_max_distance,
                ligand_bonds_only=self.ligand_bonds_only,
            )
            self.val_ds = FilteredComplexDataset(
                self.processed_dir, val_codes,
                protein_max_distance=self.protein_max_distance,
                ligand_bonds_only=self.ligand_bonds_only,
            )
            print(
                f"Train: {len(self.train_ds)} complexes | Val: {len(self.val_ds)} complexes "
                f"| protein_max_distance={self.protein_max_distance} "
                f"| ligand_bonds_only={self.ligand_bonds_only}"
            )
