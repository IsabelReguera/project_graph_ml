import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pandas as pd



class ComplexDataset(Dataset):
    """
    Dataset che carica coppie (protein_*.pt, ligand_*.pt) già processate da PDBbindDataset.
    Ogni elemento è una tupla (protein_data, ligand_data) con lo stesso pdb_code e pAffinity.
    """

    def __init__(self, processed_dir, pdb_codes):
        self.processed_dir = processed_dir
        self.pdb_codes = list(pdb_codes)

    def __len__(self):
        return len(self.pdb_codes)

    def __getitem__(self, idx):
        code = self.pdb_codes[idx]
        protein = torch.load(
            os.path.join(self.processed_dir, f'protein_{code}.pt'), weights_only=False
        )
        ligand = torch.load(
            os.path.join(self.processed_dir, f'ligand_{code}.pt'), weights_only=False
        )
        return protein, ligand


def complex_collate_fn(batch):
    """Collate per DataLoader: separa le tuple e crea due Batch PyG."""
    proteins, ligands = zip(*batch) # unzip per ottenere due liste separate
    return Batch.from_data_list(list(proteins)), Batch.from_data_list(list(ligands))


def build_complex_dataset(processed_dir, csv_path, limit=None):
    """
    Costruisce un ComplexDataset a partire dai file .pt già su disco.
    Considera solo i pdb_code per cui esistono sia protein_*.pt che ligand_*.pt.
    """
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.iloc[:limit]

    print(f"Building ComplexDataset...")
    #potenziale aggiunta creazione .pt mancanti, attualmente utilizza solo quelli già presenti (creati da pdbbind_dataset.py)

    valid_codes = []
    for code in df['pdb_code']:
        p_path = os.path.join(processed_dir, f'protein_{code}.pt')
        l_path = os.path.join(processed_dir, f'ligand_{code}.pt')
        if os.path.exists(p_path) and os.path.exists(l_path):
            valid_codes.append(code)
    
    print(f"Found {len(valid_codes)} valid complexes {('out of ' + str(limit)) if limit is not None else ''}")
    print(f"ComplexDataset[{len(valid_codes)}] built")
    return ComplexDataset(processed_dir, valid_codes)
