"""
On-the-fly edge filters for PyG Data objects of protein/ligand graphs.

Applied inside the Dataset `__getitem__`, so the preprocessed .pt files on
disk (protein = distance threshold 8 A, ligand = bonds + 5 A spatial) stay
untouched. The filters take the loaded graph and return a copy with only
the edges we want to keep.

Schema (from src/utils.py):
- protein edge_attr columns: [bond_type (0=spatial contact, 1=peptide), distance_A]
- ligand  edge_attr columns: [bond_type (0=fake/spatial, 1..4=chemical), is_conjugated, distance_A]
"""

import torch


def _apply_edge_mask(data, mask):
    """
    Return a clone of `data` where `edge_index` and `edge_attr` are kept only
    on positions where `mask` is True. Mask is 1D and length == num_edges.
    """
    out = data.clone()
    out.edge_index = data.edge_index[:, mask]
    if data.edge_attr is not None:
        out.edge_attr = data.edge_attr[mask]
    return out


def filter_protein_edges_by_distance(protein_data, max_distance_angstrom):
    """
    Keep only protein edges with distance <= max_distance_angstrom.
    Expects edge_attr[:, 1] to carry Ca-Ca distance in Angstrom.
    If max_distance_angstrom is None or no edge_attr, returns data unchanged.
    """
    if max_distance_angstrom is None:
        return protein_data
    if protein_data.edge_attr is None or protein_data.edge_index.numel() == 0:
        return protein_data
    distances = protein_data.edge_attr[:, 1]
    mask = distances <= float(max_distance_angstrom)
    return _apply_edge_mask(protein_data, mask)


def keep_only_chemical_bonds_ligand(ligand_data):
    """
    Keep only ligand edges that correspond to real chemical bonds
    (bond_type != 0). Edges added purely by the spatial cutoff have
    bond_type == 0 and are dropped.
    Expects edge_attr[:, 0] to carry bond_type.
    """
    if ligand_data.edge_attr is None or ligand_data.edge_index.numel() == 0:
        return ligand_data
    bond_type = ligand_data.edge_attr[:, 0]
    mask = bond_type != 0
    return _apply_edge_mask(ligand_data, mask)
