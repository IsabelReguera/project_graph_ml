import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pandas as pd

import pytorch_lightning as pl
from complex_dataset import ComplexDataset, complex_collate_fn, build_complex_dataset
from sklearn.model_selection import train_test_split


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

# ---------------------------------------------------------------------------
# Lightning Data Module
# ---------------------------------------------------------------------------
class ComplexDataModule(pl.LightningDataModule):
    """LightningDataModule for ComplexDataset, it manages train/val split and DataLoader."""
    
    def __init__(self, processed_dir, csv_path, batch_size=16, limit=None, train_val_split=0.2, seed=47):
        super().__init__()
        self.processed_dir = processed_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.limit = limit
        self.train_val_split = train_val_split
        self.seed = seed
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = build_complex_dataset(self.processed_dir, self.csv_path, limit=self.limit)
            all_codes = full_dataset.pdb_codes

            # Split train/val with PDB code
            train_codes, val_codes = train_test_split(all_codes, test_size=self.train_val_split, random_state=self.seed)

            self.train_ds = ComplexDataset(self.processed_dir, train_codes)
            self.val_ds = ComplexDataset(self.processed_dir, val_codes)
    
            print(f"Train: {len(self.train_ds)} complexes | Val: {len(self.val_ds)} complexes")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=complex_collate_fn, 
            num_workers=4,
            persistent_workers=True 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=complex_collate_fn, 
            num_workers=4, 
            persistent_workers=True
        )


# --- SMOKE TEST ---
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = build_complex_dataset(
        processed_dir='data/processed',
        csv_path='data/raw/PL_cleaned_index.csv',
        limit=10
    )
    print(f"Complessi validi: {len(dataset)}")

    if len(dataset) == 0:
        print("Nessun complesso trovato. Eseguire prima pdbbind_dataset.py per generare i .pt.")
    else:
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=complex_collate_fn)
        prot_batch, lig_batch = next(iter(loader))
        print(f"Protein batch — x: {prot_batch.x.shape}, edge_index: {prot_batch.edge_index.shape}, "
              f"edge_attr: {prot_batch.edge_attr.shape}, y: {prot_batch.y.shape}")
        print(f"Ligand  batch — x: {lig_batch.x.shape},  edge_index: {lig_batch.edge_index.shape}, "
              f"edge_attr: {lig_batch.edge_attr.shape},  y: {lig_batch.y.shape}")
